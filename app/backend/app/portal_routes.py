"""Patient and admin portal HTTP API."""

from __future__ import annotations

import uuid
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .config import get_settings
from .database import SessionLocal, init_db
from .inference import InferenceEngine
from .portal_auth import issue_admin_token, require_admin
from .portal_db import Patient, UploadedImage, Visit
from .portal_service import (
    add_training_images,
    admin_kpis,
    build_dataset_version,
    classify_visit_crops,
    create_patient,
    create_visit,
    dataset_summary,
    evaluation_bundle,
    analysis_from_stored,
    generate_report_for_visit,
    list_models,
    patient_to_dict,
    queue_training_job,
    real_world_test,
    run_detection_for_image,
    save_label,
    seed_model_registry,
    set_active_model,
    store_visit_image,
    training_status,
    skip_label,
    unlabeled_queue,
    visit_detail,
    visit_history,
)
from .portal_runtime import ICDAS_NOT_DEPLOYED, get_runtime
from .training_reset import execute_training_reset, reset_plan

router = APIRouter(prefix="/api/v1", tags=["portal"])
logger = logging.getLogger("icdas.portal_routes")


def get_db():
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine(request: Request) -> InferenceEngine | None:
    return getattr(request.app.state, "engine", None)


class PatientIn(BaseModel):
    public_id: Optional[str] = None
    name: str
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = None
    phone: Optional[str] = None
    visit_date: Optional[str] = None
    notes: Optional[str] = None


class LabelIn(BaseModel):
    label_id: int
    grade: int = Field(..., ge=0, le=4)
    confirm_overwrite: bool = False
    track: str = "initial"


class AdminLogin(BaseModel):
    password: str


class TrainIn(BaseModel):
    dataset_name: Optional[str] = None


class SplitPreviewIn(BaseModel):
    plan: dict = Field(default_factory=dict)
    expected_fingerprint: Optional[str] = None


class SplitBulkAssignIn(BaseModel):
    selected_image_ids: list[int] = Field(default_factory=list)
    split_name: str


class PatientGroupAssignmentIn(BaseModel):
    selected_image_ids: list[int] = Field(default_factory=list)
    photo_ids: Optional[str] = None
    action: str = Field(..., pattern="^(create|existing|unassigned)$")
    target_patient_id: Optional[str] = None
    confirm_reassignment: bool = False
    expected_patient_id: Optional[str] = None


class PatientGroupAssignmentPreviewIn(BaseModel):
    photo_ids: str = ""
    action: str = Field(..., pattern="^(create|existing|unassigned)$")
    target_patient_id: Optional[str] = None


class PatientPhotoGroupReviewIn(BaseModel):
    suggestion_key: str = Field(..., min_length=1, max_length=160)
    photo_ids: list[str] = Field(..., min_length=1, max_length=1000)
    decision: str = Field(..., pattern="^(confirm|reject)$")


class BoxesIn(BaseModel):
    boxes: list[dict]
    deleted_ids: list[int] = Field(default_factory=list)


class TrainingIdentityIn(BaseModel):
    patient_id: str = Field(..., min_length=2, max_length=64)
    image_id: Optional[str] = Field(default=None, max_length=64)


class CropStatusIn(BaseModel):
    label_id: int
    status: str


class TrainingResetIn(BaseModel):
    scope: str = Field(..., description="dataset or full")
    confirm: bool = False
    confirmation_text: str = ""


class TranslateIn(BaseModel):
    language: str = "en"
    labels: dict


@router.get("/portal/health")
def portal_health():
    return get_runtime().status()


@router.post("/patients")
def api_create_patient(body: PatientIn, db: Session = Depends(get_db)):
    try:
        patient = create_patient(db, body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    visit = create_visit(db, patient, body.visit_date or "", body.notes)
    return {
        "patient": patient_to_dict(patient),
        "visit_id": visit.id,
        "visit_date": visit.visit_date,
    }


@router.get("/patients/{public_id}")
def api_get_patient(public_id: str, db: Session = Depends(get_db)):
    try:
        return visit_history(db, public_id)
    except KeyError:
        raise HTTPException(404, "Patient not found.")


@router.get("/patients/{public_id}/history")
def api_history(public_id: str, db: Session = Depends(get_db)):
    try:
        return visit_history(db, public_id)
    except KeyError:
        raise HTTPException(404, "Patient not found.")


@router.get("/visits/{visit_id}")
def api_visit(visit_id: int, db: Session = Depends(get_db)):
    try:
        return visit_detail(db, visit_id)
    except KeyError:
        raise HTTPException(404, "Visit not found.")


@router.delete("/visits/{visit_id}")
def api_soft_delete_visit(
    visit_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from datetime import datetime

    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if visit is None:
        raise HTTPException(404, "Visit not found.")
    visit.deleted_at = datetime.utcnow()
    db.commit()
    return {"visit_id": visit_id, "deleted": True}


@router.post("/visits/{visit_id}/images")
async def api_upload_images(
    visit_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    visit = db.query(Visit).filter(Visit.id == visit_id, Visit.deleted_at.is_(None)).first()
    if visit is None:
        raise HTTPException(404, "Visit not found.")
    results = []
    for f in files:
        data = await f.read()
        filename = f.filename or "image.jpg"
        if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(400, "Choose a JPG or PNG photo.")
        if len(data) > int(get_settings().max_upload_mb) * 1024 * 1024:
            raise HTTPException(413, "Photo exceeds the upload size limit.")
        try:
            rec, payload = store_visit_image(db, visit, data, filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        results.append(payload)
        _ = rec
    return {"visit_id": visit_id, "images": results}


@router.post("/images/{image_id}/quality")
def api_quality(image_id: int, db: Session = Depends(get_db)):
    image = db.query(UploadedImage).filter(UploadedImage.id == image_id).first()
    if image is None:
        raise HTTPException(404, "Image not found.")
    import json

    return {"image_id": image_id, "quality": json.loads(image.quality_json or "{}")}


@router.post("/images/{image_id}/detect")
def api_detect(image_id: int, db: Session = Depends(get_db)):
    image = db.query(UploadedImage).filter(UploadedImage.id == image_id).first()
    if image is None:
        raise HTTPException(404, "Image not found.")
    try:
        return run_detection_for_image(db, image)
    except RuntimeError as exc:
        logger.exception("Patient visit detection failed")
        raise HTTPException(503, "Tooth detection is temporarily unavailable. Please try again later.") from exc


@router.post("/visits/{visit_id}/analyze")
def api_analyze(
    visit_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    visit = db.query(Visit).filter(Visit.id == visit_id, Visit.deleted_at.is_(None)).first()
    if visit is None:
        raise HTTPException(404, "Visit not found.")
    detections = []
    try:
        for image in list(visit.images):
            detections.append(run_detection_for_image(db, image))
    except RuntimeError as exc:
        logger.exception("Patient visit detection failed")
        raise HTTPException(503, "Tooth detection is temporarily unavailable. Please try again later.") from exc
    runtime = get_runtime()
    if not runtime.icdas_ok():
        return {
            "detections": detections,
            "icdas_status": "NOT_TRAINED / NOT_DEPLOYED",
            "prediction_mode": "NOT_DEPLOYED",
            "icdas_loaded": False,
            "teeth": [],
            "teeth_detected": sum(d.get("teeth_detected") or 0 for d in detections),
            "teeth_analyzed": 0,
            "message": ICDAS_NOT_DEPLOYED,
            "disclaimer": detections[0].get("disclaimer") if detections else ICDAS_NOT_DEPLOYED,
        }
    try:
        db.expire_all()
        visit = db.query(Visit).filter(Visit.id == visit_id).first()
        analysis = classify_visit_crops(db, visit, include_heatmaps=True)
    except RuntimeError as exc:
        logger.exception("Patient visit classification failed")
        raise HTTPException(503, "ICDAS classification is temporarily unavailable. Please try again later.") from exc
    return {
        "detections": detections,
        "icdas_status": "DEMO MODE — not production active" if runtime.icdas_demo else "DEPLOYED",
        "prediction_mode": runtime.prediction_mode,
        **analysis,
    }


@router.post("/visits/{visit_id}/classify")
def api_classify(
    visit_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    visit = db.query(Visit).filter(Visit.id == visit_id, Visit.deleted_at.is_(None)).first()
    if visit is None:
        raise HTTPException(404, "Visit not found.")
    try:
        return classify_visit_crops(db, visit, include_heatmaps=True)
    except RuntimeError as exc:
        logger.exception("Patient visit classification failed")
        raise HTTPException(503, "ICDAS classification is temporarily unavailable. Please try again later.") from exc


@router.post("/visits/{visit_id}/report")
def api_report(
    visit_id: int,
    request: Request,
    language: str = "en",
    db: Session = Depends(get_db),
):
    if language not in {"en", "hi", "kn"}:
        raise HTTPException(400, "language must be en, hi, or kn.")
    visit = db.query(Visit).filter(Visit.id == visit_id, Visit.deleted_at.is_(None)).first()
    if visit is None:
        raise HTTPException(404, "Visit not found.")
    analysis = analysis_from_stored(visit, language=language)
    if analysis is None:
        if not get_runtime().icdas_ok():
            raise HTTPException(503, ICDAS_NOT_DEPLOYED)
        try:
            analysis = classify_visit_crops(db, visit, include_heatmaps=True, language=language)
        except RuntimeError as exc:
            logger.exception("Patient visit report classification failed")
            raise HTTPException(503, "ICDAS classification is temporarily unavailable. Please try again later.") from exc
    if not (analysis.get("teeth") or []):
        raise HTTPException(503, ICDAS_NOT_DEPLOYED)
    try:
        return generate_report_for_visit(db, visit, analysis, language)
    except RuntimeError as exc:
        logger.exception("Patient report generation failed")
        raise HTTPException(503, "The report could not be generated right now. Please try again later.") from exc


@router.post("/translate")
def api_translate(body: TranslateIn):
    from .groq_service import translate_ui_bundle

    if body.language not in {"en", "hi", "kn"}:
        raise HTTPException(400, "language must be en, hi, or kn.")
    return {"language": body.language, "labels": translate_ui_bundle(body.labels, body.language)}


@router.post("/admin/login")
def api_admin_login(body: AdminLogin):
    settings = get_settings()
    if body.password != settings.admin_password:
        raise HTTPException(401, "Invalid admin password.")
    return {"token": issue_admin_token(), "role": "admin"}


@router.get("/admin/kpis")
def api_kpis(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    seed_model_registry(db)
    return admin_kpis(db)


@router.get("/admin/patients")
def api_admin_patients(
    q: Optional[str] = None,
    limit: int = 1000,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    query = db.query(Patient).filter(Patient.is_anonymous.is_(False))
    if q:
        like = f"%{q}%"
        query = query.filter((Patient.name.ilike(like)) | (Patient.public_id.ilike(like)))
    rows = query.order_by(Patient.id.desc()).limit(max(1, min(limit, 1000))).all()
    return {"patients": [patient_to_dict(p) for p in rows]}


@router.post("/admin/patients")
def api_admin_create_patient(
    body: PatientIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    return api_create_patient(body, db)


@router.post("/admin/training/images", status_code=202)
async def api_training_upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    patient_manifest: Optional[UploadFile] = File(default=None),
    patient_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .portal_service import MAX_BYTES

    max_batch_bytes = 512 * 1024 * 1024

    try:
        if not files:
            raise ValueError("Choose at least one photo to upload.")
        if len(files) > 250:
            raise ValueError("Upload at most 250 photographs per batch.")
        filenames = [Path(f.filename or "").name for f in files]
        if any(not name for name in filenames):
            raise ValueError("Every upload must have a filename.")
        if len({name.casefold() for name in filenames}) != len(filenames):
            raise ValueError("Upload filenames must be unique within a batch.")
        declared_total_bytes = 0
        for upload, name in zip(files, filenames):
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"{name}: only JPG, JPEG, and PNG photos are accepted.")
            if upload.size is not None and upload.size > MAX_BYTES:
                raise ValueError(f"{name}: file exceeds size limit ({MAX_BYTES // (1024 * 1024)} MB).")
            declared_total_bytes += int(upload.size or 0)
        if declared_total_bytes > max_batch_bytes:
            raise ValueError("Upload batch exceeds the 512 MB limit. Submit it as multiple batches.")

        if patient_id and patient_manifest is not None:
            raise ValueError("Choose one patient for the batch or upload a per-photo CSV map, not both.")
        identity_map: dict[str, dict[str, str | None]]
        if patient_manifest is not None:
            from .training_workflow import parse_import_manifest, validate_training_upload_identities

            text = (await patient_manifest.read()).decode("utf-8-sig")
            identity_map = parse_import_manifest(text, filenames)
        else:
            from .training_workflow import validate_training_upload_identities

            selected_patient_id = (patient_id or "").strip() or None
            identity_map = {
                name.casefold(): {"patient_id": selected_patient_id, "image_id": ""}
                for name in filenames
            }
        identity_map = validate_training_upload_identities(db, filenames, identity_map)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    batch_id = uuid.uuid4().hex
    items = []
    rejected = []
    received_total_bytes = 0
    for upload, name in zip(files, filenames):
        try:
            data = await upload.read()
            if len(data) > MAX_BYTES:
                raise ValueError(f"{name}: file exceeds size limit ({MAX_BYTES // (1024 * 1024)} MB).")
            if received_total_bytes + len(data) > max_batch_bytes:
                raise ValueError("Upload batch exceeds the 512 MB limit. Submit remaining photos as another batch.")
            received_total_bytes += len(data)
            items.extend(
                add_training_images(
                    db,
                    [(name, data)],
                    {name.casefold(): identity_map[name.casefold()]},
                    batch_id,
                )
            )
        except (RuntimeError, ValueError) as exc:
            rejected.append({"filename": name, "error": str(exc)})

    queued = [item for item in items if item.get("processing_status") == "QUEUED"]
    if queued:
        from .training_workflow import process_training_upload_batch

        background_tasks.add_task(process_training_upload_batch, batch_id)
    return {
        "upload_batch_id": batch_id,
        "count": len(items),
        "images": items,
        "rejected": rejected,
        "queued_count": len(queued),
        "unique_count": sum(1 for i in items if i.get("duplicate_status") == "UNIQUE"),
        "duplicate_count": sum(1 for i in items if i.get("processing_status") == "DUPLICATE"),
        "note": "Photos without an explicit registered patient mapping remain Unknown / Unassigned. Detector V2 processes unique photos in the background; assign anonymous patient groups after detection.",
    }


@router.get("/admin/training/patient-groups")
def api_patient_group_inventory(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .patient_grouping import anonymous_patient_inventory

    return anonymous_patient_inventory(db)


@router.get("/admin/training/patient-groups/review")
def api_patient_photo_group_review_inventory(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .patient_grouping import patient_photo_group_review_inventory

    return patient_photo_group_review_inventory(db)


@router.post("/admin/training/patient-groups/review")
def api_review_patient_photo_group(
    body: PatientPhotoGroupReviewIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .patient_grouping import record_patient_photo_group_review

    try:
        return record_patient_photo_group_review(
            db,
            suggestion_key=body.suggestion_key,
            photo_ids=body.photo_ids,
            decision=body.decision,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/admin/training/patient-groups/preview")
def api_preview_patient_group_assignment(
    body: PatientGroupAssignmentPreviewIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .patient_grouping import preview_training_photo_assignment

    try:
        return preview_training_photo_assignment(
            db,
            body.photo_ids,
            action=body.action,
            target_patient_id=body.target_patient_id,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/training/patient-groups/assign")
def api_assign_patient_group(
    body: PatientGroupAssignmentIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .patient_grouping import assign_training_photos, resolve_photo_ids

    try:
        if body.photo_ids is not None:
            if body.selected_image_ids:
                raise ValueError("Provide Photo IDs or database photo IDs, not both.")
            selected_image_ids = resolve_photo_ids(db, body.photo_ids)["training_image_ids"]
        else:
            selected_image_ids = body.selected_image_ids
        return assign_training_photos(
            db,
            selected_image_ids,
            action=body.action,
            target_patient_id=body.target_patient_id,
            confirm_reassignment=body.confirm_reassignment,
            expected_patient_id=body.expected_patient_id,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/admin/training/patient-groups/audit")
def api_patient_group_audit(
    limit: int = 200,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .portal_db import TrainingAuditEvent

    rows = (
        db.query(TrainingAuditEvent)
        .filter(TrainingAuditEvent.event_type == "patient_group_assignment")
        .order_by(TrainingAuditEvent.id.desc())
        .limit(max(1, min(limit, 1000)))
        .all()
    )
    return {
        "events": [
            {
                "id": row.id,
                "actor": row.actor,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "details": json.loads(row.details_json or "{}"),
            }
            for row in rows
        ]
    }


@router.get("/admin/training/uploads")
def api_training_upload_history(
    limit: int = 1000,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import training_upload_history

    return training_upload_history(db, limit)


@router.post("/admin/training/uploads/{upload_batch_id}/resume")
def api_resume_training_uploads(
    upload_batch_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .portal_db import TrainingImage
    from .training_workflow import UPLOAD_FAILED, UPLOAD_QUEUED, process_training_upload_batch

    count = (
        db.query(TrainingImage)
        .filter(
            TrainingImage.upload_batch_id == upload_batch_id,
            TrainingImage.processing_status.in_([UPLOAD_FAILED, UPLOAD_QUEUED]),
            TrainingImage.duplicate_status == "UNIQUE",
        )
        .count()
    )
    if count == 0:
        raise HTTPException(409, "This batch has no failed or queued unique photos to resume.")
    background_tasks.add_task(process_training_upload_batch, upload_batch_id, retry_failed=True)
    return {"upload_batch_id": upload_batch_id, "resumed_count": count, "status": "QUEUED"}


@router.put("/admin/training/images/{image_id}/identity")
def api_training_identity(
    image_id: int,
    body: TrainingIdentityIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import update_training_identity

    try:
        return update_training_identity(db, image_id, body.patient_id, body.image_id)
    except KeyError:
        raise HTTPException(404, "Training image not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/admin/training/images")
def api_list_training_images(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    from .training_workflow import list_training_images

    return list_training_images(db)


@router.get("/admin/training/images/{image_id}")
def api_training_image(
    image_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import training_image_detail

    try:
        return training_image_detail(db, image_id)
    except KeyError:
        raise HTTPException(404, "Training image not found.")


@router.put("/admin/training/images/{image_id}/boxes")
def api_save_boxes(
    image_id: int,
    body: BoxesIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import save_training_boxes

    try:
        return save_training_boxes(db, image_id, body.boxes, body.deleted_ids)
    except KeyError:
        raise HTTPException(404, "Training image not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/training/images/{image_id}/verify-boxes")
def api_verify_boxes(
    image_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import verify_training_boxes

    try:
        return verify_training_boxes(db, image_id)
    except KeyError:
        raise HTTPException(404, "Training image not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/training/photos/{photo_id}/deactivate")
@router.post("/admin/training/images/{image_id}/deactivate")
def api_deactivate_training_image(
    image_id: Optional[int] = None,
    photo_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import deactivate_training_image

    target_id = photo_id if photo_id is not None else image_id
    if target_id is None:
        raise HTTPException(400, "Photo ID required.")
    try:
        return deactivate_training_image(db, target_id)
    except KeyError:
        raise HTTPException(404, "Training image not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/admin/training/queue")
def api_label_queue(
    image_id: Optional[int] = None,
    label_id: Optional[int] = None,
    resume: bool = False,
    patient_id: Optional[str] = None,
    split_name: Optional[str] = None,
    grade: Optional[int] = None,
    annotation_status: Optional[str] = None,
    mode: Optional[str] = None,
    track: Optional[str] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    return unlabeled_queue(
        db,
        image_id=image_id,
        label_id=label_id,
        resume=resume,
        patient_id=patient_id,
        split_name=split_name,
        grade=grade,
        annotation_status=annotation_status,
        mode=mode,
        track=track,
    )


@router.get("/admin/training/labeling/photos")
def api_labeling_photo_inventory(
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import training_labeling_photo_inventory

    return training_labeling_photo_inventory(db, search=search)


@router.get("/admin/training/crops")
def api_training_crops(
    training_image_id: Optional[int] = None,
    patient_id: Optional[str] = None,
    split_name: Optional[str] = None,
    grade: Optional[int] = None,
    annotation_status: Optional[str] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import training_crop_listing

    if grade is not None and not 0 <= grade <= 4:
        raise HTTPException(400, "grade must be ICDAS 0–4.")
    return training_crop_listing(
        db,
        training_image_id=training_image_id,
        patient_id=patient_id,
        split_name=split_name,
        grade=grade,
        annotation_status=annotation_status,
    )


@router.get("/admin/training/audit")
def api_training_audit(
    limit: int = 200,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import training_audit_listing

    return training_audit_listing(db, limit=limit)


@router.post("/admin/training/labels")
def api_save_label(
    body: LabelIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return save_label(
            db,
            body.label_id,
            body.grade,
            confirm_overwrite=body.confirm_overwrite,
            track=body.track,
        )
    except KeyError:
        raise HTTPException(404, "Label not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/admin/training/labels/{label_id}/history")
def api_label_history(
    label_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import label_history_listing

    try:
        return label_history_listing(db, label_id)
    except KeyError:
        raise HTTPException(404, "Label not found.")


@router.post("/admin/training/labels/{label_id}/undo")
def api_undo_label(
    label_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import undo_icdas_label

    try:
        return undo_icdas_label(db, label_id)
    except KeyError:
        raise HTTPException(404, "Label not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class SkipIn(BaseModel):
    label_id: int


@router.post("/admin/training/labels/status")
def api_crop_status(
    body: CropStatusIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import set_crop_annotation_status

    try:
        return set_crop_annotation_status(db, body.label_id, body.status)
    except KeyError:
        raise HTTPException(404, "Label not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/training/labels/skip")
def api_skip_label(
    body: SkipIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return skip_label(db, body.label_id)
    except KeyError:
        raise HTTPException(404, "Label not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/admin/training/reset/plan")
def api_training_reset_plan(
    scope: str = "dataset",
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return reset_plan(scope, db)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/training/reset")
def api_training_reset(
    body: TrainingResetIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return execute_training_reset(
            db,
            body.scope,
            body.confirm,
            body.confirmation_text,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/admin/dataset")
def api_dataset(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    return dataset_summary(db)


@router.get("/admin/dataset/patients")
def api_dataset_patients(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    from .training_workflow import patient_assignment_inventory

    return patient_assignment_inventory(db)


@router.post("/admin/dataset/preview-split")
def api_preview_split(
    body: SplitPreviewIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import preview_dataset_split

    try:
        return preview_dataset_split(db, body.plan)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/training/splits/assign")
def api_bulk_assign_split(
    body: SplitBulkAssignIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    from .training_workflow import bulk_assign_training_images

    try:
        return bulk_assign_training_images(db, body.selected_image_ids, body.split_name)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/dataset/build")
def api_build(
    body: Optional[SplitPreviewIn] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    body = body or SplitPreviewIn()
    if not body.expected_fingerprint:
        raise HTTPException(400, "Preview the split and confirm the matching assignment fingerprint before building.")
    try:
        return build_dataset_version(db, body.plan, body.expected_fingerprint)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/train")
def api_train(
    body: TrainIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return queue_training_job(db, body.dataset_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/admin/train/status")
def api_train_status(
    job_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    return training_status(db, job_id)


@router.get("/admin/models")
def api_models(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    return {"models": list_models(db)}


@router.post("/admin/models/{model_id}/set-active")
@router.post("/admin/models/{model_id}/deploy")
def api_set_active(
    model_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return set_active_model(db, model_id, True)
    except KeyError:
        raise HTTPException(404, "Model not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/admin/models/{model_id}/rollback")
def api_rollback(
    model_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return set_active_model(db, model_id, False)
    except KeyError:
        raise HTTPException(404, "Model not found.")


@router.get("/admin/evaluation")
def api_eval(_: None = Depends(require_admin)):
    return evaluation_bundle()


@router.post("/admin/real-world-test")
async def api_real_world(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    data = await file.read()
    try:
        return real_world_test(data, file.filename or "image.jpg", db=db)
    except (RuntimeError, ValueError) as exc:
        logger.exception("Admin sample inference failed")
        message = "Photo could not be processed. Choose a clear JPG or PNG." if isinstance(exc, ValueError) else "AI analysis is temporarily unavailable. Please try again later."
        raise HTTPException(400 if isinstance(exc, ValueError) else 503, message) from exc
