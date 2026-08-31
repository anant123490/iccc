"""Patient and admin portal HTTP API."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
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


class AdminLogin(BaseModel):
    password: str


class TrainIn(BaseModel):
    dataset_name: Optional[str] = None


class BoxesIn(BaseModel):
    boxes: list[dict]
    deleted_ids: list[int] = Field(default_factory=list)


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
    patient = create_patient(db, body.model_dump())
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
        try:
            rec, payload = store_visit_image(db, visit, data, f.filename or "image.jpg")
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
        raise HTTPException(503, str(exc)) from exc


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
        raise HTTPException(503, str(exc)) from exc
    runtime = get_runtime()
    if not runtime.icdas_ok():
        return {
            "detections": detections,
            "icdas_status": "NOT_TRAINED / NOT_DEPLOYED",
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
        raise HTTPException(503, str(exc)) from exc
    return {"detections": detections, "icdas_status": "DEPLOYED", **analysis}


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
        raise HTTPException(503, str(exc)) from exc


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
            raise HTTPException(503, str(exc)) from exc
    if not (analysis.get("teeth") or []):
        raise HTTPException(503, ICDAS_NOT_DEPLOYED)
    try:
        return generate_report_for_visit(db, visit, analysis, language)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


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
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    query = db.query(Patient)
    if q:
        like = f"%{q}%"
        query = query.filter((Patient.name.ilike(like)) | (Patient.public_id.ilike(like)))
    rows = query.order_by(Patient.id.desc()).limit(200).all()
    return {"patients": [patient_to_dict(p) for p in rows]}


@router.post("/admin/patients")
def api_admin_create_patient(
    body: PatientIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    return api_create_patient(body, db)


@router.post("/admin/training/images")
async def api_training_upload(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    blobs = []
    for f in files:
        blobs.append((f.filename or "image.jpg", await f.read()))
    try:
        items = add_training_images(db, blobs)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400 if isinstance(exc, ValueError) else 503, str(exc)) from exc
    dups = [
        i
        for i in items
        if i.get("duplicate_status") in {"EXACT_DUPLICATE", "LIKELY_DUPLICATE", "INVALID"}
    ]
    return {
        "count": len(items),
        "images": items,
        "duplicates": dups,
        "unique_count": sum(1 for i in items if i.get("duplicate_status") == "UNIQUE"),
        "note": "Duplicate files are kept on disk but excluded from the ICDAS dataset.",
    }


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
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    return unlabeled_queue(db, image_id=image_id, label_id=label_id, resume=resume)


@router.post("/admin/training/labels")
def api_save_label(
    body: LabelIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        return save_label(db, body.label_id, body.grade)
    except KeyError:
        raise HTTPException(404, "Label not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class SkipIn(BaseModel):
    label_id: int


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
    _: None = Depends(require_admin),
):
    try:
        return reset_plan(scope)
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


@router.post("/admin/dataset/build")
def api_build(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    try:
        return build_dataset_version(db)
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
        raise HTTPException(400 if isinstance(exc, ValueError) else 503, str(exc)) from exc
