"""Admin ICDAS training-dataset workflow. Does not train YOLO or mix patient photos."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import shutil
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .admin_progress import (
    MIN_DATASET_CROPS,
    crop_is_resolved,
    dataset_build_ready,
    dataset_classes_ready,
    dataset_min_crops_message,
    imbalance_warning_text,
    missing_classes_message,
    missing_icdas_classes,
    validate_icdas_grade,
)
from .config import PROJECT_ROOT, get_settings
from .crop_identity import (
    CONFLICT_MESSAGE,
    CROP_CONFLICT,
    CROP_DUP_EXACT,
    CROP_POSSIBLE,
    CROP_UNIQUE,
    apply_crop_identity,
    crop_status_caption,
    refresh_crop_identities,
    sync_exact_hash_group,
)
from .image_phash import average_hash_hex, is_likely_duplicate
from .portal_db import (
    DatasetVersion,
    LabelingProgress,
    ModelVersion,
    Patient,
    TrainingAuditEvent,
    TrainingImage,
    TrainingJob,
    TrainingLabel,
)
from .storage_paths import (
    ICDAS_DATASETS,
    MODELS_ICDAS,
    TRAIN_DETECTED,
    TRAIN_LABELED,
    TRAIN_UPLOADS,
    TRAIN_VERSIONS,
    ensure_dirs,
)
from .tooth_detector import (
    crop_xyxy_rgb,
    detect_rgb,
    detector_available,
    detector_error,
    draw_boxes_rgb,
    validate_xyxy_box,
)

logger = logging.getLogger("icdas.training_workflow")

SPLIT_SEED = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)
TARGET_PATIENTS = 100
STATUS_UNIQUE = "UNIQUE"
STATUS_EXACT = "EXACT_DUPLICATE"
STATUS_LIKELY = "LIKELY_DUPLICATE"
STATUS_INVALID = "INVALID"
UPLOAD_QUEUED = "QUEUED"
UPLOAD_PROCESSING = "PROCESSING"
UPLOAD_COMPLETED = "COMPLETED"
UPLOAD_FAILED = "FAILED"
UPLOAD_DUPLICATE = "DUPLICATE"
UPLOAD_INVALID = "INVALID"
INCLUDED = "INCLUDED"
EXCLUDED_DUPLICATE = "EXCLUDED_DUPLICATE"
ANNOTATION_STATUSES = {"UNLABELED", "LABELED", "UNCERTAIN", "UNGRADABLE"}
PATIENT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,63}$")
IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")

ACTIVE_LABEL = or_(TrainingLabel.active.is_(True), TrainingLabel.active.is_(None))
UNIQUE_IMAGE = or_(
    TrainingImage.duplicate_status == STATUS_UNIQUE,
    TrainingImage.duplicate_status.is_(None),
)
IS_ACTIVE_IMAGE = or_(
    TrainingImage.is_active.is_(True),
    TrainingImage.is_active.is_(None),
)
NOT_EXCLUDED = or_(
    TrainingImage.exclude_from_dataset.is_(False),
    TrainingImage.exclude_from_dataset.is_(None),
)
NOT_SKIPPED = or_(TrainingLabel.skipped.is_(False), TrainingLabel.skipped.is_(None))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_patient_id(value: str | None) -> str:
    """Accept a pseudonymous token; never accept names, emails, or phone formatting."""
    patient_id = (value or "").strip()
    if (
        not PATIENT_ID_RE.fullmatch(patient_id)
        or not any(ch.isalpha() for ch in patient_id)
        or not any(ch.isdigit() for ch in patient_id)
        or sum(ch.isdigit() for ch in patient_id) >= 10
    ):
        raise ValueError("Use a pseudonymous ID code such as PAT001, not a patient's name or contact details.")
    return patient_id


def validate_image_id(value: str | None) -> str:
    image_id = (value or "").strip()
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError("Image ID must be a unique 2–64 character identifier using letters, numbers, '.', '_' or '-'.")
    return image_id


def parse_import_manifest(manifest_text: str, filenames: list[str]) -> dict[str, dict[str, str]]:
    """Parse ``filename,patient_id[,image_id]``; require exact filename coverage."""
    reader = csv.DictReader(io.StringIO(manifest_text.lstrip("\ufeff")))
    headers = {str(name or "").strip().lower() for name in (reader.fieldnames or [])}
    if not {"filename", "patient_id"}.issubset(headers):
        raise ValueError("Import CSV must contain filename and patient_id columns; image_id is optional.")
    mapping: dict[str, dict[str, str]] = {}
    for row in reader:
        normalized = {str(k or "").strip().lower(): str(v or "").strip() for k, v in row.items()}
        filename = Path(normalized.get("filename", "")).name
        if not filename:
            raise ValueError("Import CSV contains an empty filename.")
        key = filename.casefold()
        if key in mapping:
            raise ValueError(f"Import CSV contains duplicate filename {filename}.")
        mapping[key] = {
            "patient_id": normalized.get("patient_id", "").strip(),
            "image_id": validate_image_id(normalized["image_id"]) if normalized.get("image_id") else "",
        }
    upload_names = [Path(name).name for name in filenames]
    if len({name.casefold() for name in upload_names}) != len(upload_names):
        raise ValueError("Uploaded filenames must be unique within a batch when using a CSV manifest.")
    expected = {name.casefold() for name in upload_names}
    missing = sorted(expected - set(mapping))
    extra = sorted(set(mapping) - expected)
    if missing or extra:
        raise ValueError(
            "Import CSV filenames must exactly match the upload batch. "
            f"Missing rows: {missing[:5]}; unexpected rows: {extra[:5]}"
        )
    image_ids = [item["image_id"] for item in mapping.values() if item["image_id"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Import CSV contains duplicate image_id values.")
    return mapping


def _audit(db: Session, event_type: str, entity_type: str, entity_id: object, details: dict) -> None:
    db.add(
        TrainingAuditEvent(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=str(entity_id),
            actor="admin",
            details_json=json.dumps(details, sort_keys=True, default=str),
        )
    )


def _sync_image_annotation_status(image: TrainingImage) -> None:
    active = [lab for lab in image.labels if lab.active is not False]
    image.crop_count = len(active)
    image.n_crops = len(active)
    if not image.boxes_verified:
        image.annotation_status = "UNLABELED"
    elif not active:
        image.annotation_status = "UNLABELED"
    elif any(_effective_annotation_status(lab) == "UNLABELED" for lab in active):
        image.annotation_status = "IN_PROGRESS"
    else:
        image.annotation_status = "COMPLETE"


def _effective_annotation_status(label: TrainingLabel) -> str:
    if label.grade is not None:
        return "LABELED"
    status = (label.annotation_status or "").upper()
    if status in {"UNCERTAIN", "UNGRADABLE"}:
        return status
    if label.skipped:
        return "UNCERTAIN"
    return "UNLABELED"


def next_dataset_version_name(db: Session) -> tuple[str, int]:
    nums: list[int] = []
    for row in db.query(DatasetVersion).all():
        name = (row.name or "").strip().lower()
        if name.startswith("v") and name[1:].isdigit():
            nums.append(int(name[1:]))
        match = re.fullmatch(r"dataset_v(\d+)", name)
        if match:
            nums.append(int(match.group(1)))
        if row.version_number:
            nums.append(int(row.version_number))
    if ICDAS_DATASETS.exists():
        for p in ICDAS_DATASETS.iterdir():
            match = re.fullmatch(r"dataset_v(\d+)", p.name.lower())
            if p.is_dir() and match:
                nums.append(int(match.group(1)))
    n = max(nums, default=0) + 1
    return f"dataset_v{n:03d}", n


def next_icdas_model_dir() -> tuple[str, Path]:
    nums: list[int] = []
    if MODELS_ICDAS.exists():
        for p in MODELS_ICDAS.iterdir():
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
                nums.append(int(p.name[1:]))
    n = max(nums, default=0) + 1
    dest = MODELS_ICDAS / f"v{n}"
    return f"v{n}", dest


def classify_duplicate(db: Session, content_hash: str, phash: str | None) -> tuple[str, int | None]:
    exact = (
        db.query(TrainingImage)
        .filter(
            TrainingImage.content_hash == content_hash,
            TrainingImage.duplicate_status == STATUS_UNIQUE,
            IS_ACTIVE_IMAGE,
        )
        .first()
    )
    if exact is None:
        exact = (
            db.query(TrainingImage)
            .filter(TrainingImage.content_hash == content_hash, IS_ACTIVE_IMAGE)
            .order_by(TrainingImage.id.asc())
            .first()
        )
        if exact is not None:
            canon = exact.canonical_id or exact.id
            return STATUS_EXACT, canon
    else:
        return STATUS_EXACT, exact.id
    if phash:
        uniques = (
            db.query(TrainingImage)
            .filter(TrainingImage.duplicate_status == STATUS_UNIQUE, IS_ACTIVE_IMAGE)
            .all()
        )
        for other in uniques:
            if is_likely_duplicate(phash, other.phash):
                return STATUS_LIKELY, other.id
    return STATUS_UNIQUE, None


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buf.tofile(str(path))


def _detect_and_store_crops(db: Session, rec: TrainingImage, rgb: np.ndarray) -> list[dict]:
    det = detect_rgb(rgb, source_name=rec.filename)
    overlay = det["overlay_rgb"] if det["overlay_rgb"] is not None else rgb
    ov_path = TRAIN_DETECTED / str(rec.id) / "overlay.jpg"
    _write_rgb(ov_path, overlay)
    rec.overlay_path = str(ov_path)
    rec.n_crops = det["n_kept"]
    rec.crop_count = det["n_kept"]
    crops = []
    for idx, item in enumerate(det["items"]):
        crop_rgb = det["crops"][idx][1]
        lab = TrainingLabel(
            image_id=rec.id,
            crop_path="",
            grade=None,
            x1=item.x1,
            y1=item.y1,
            x2=item.x2,
            y2=item.y2,
            confidence=float(item.confidence),
            index_in_image=idx,
            crop_hash=None,
            box_verified=False,
            active=True,
        )
        db.add(lab)
        db.flush()
        crop_path = TRAIN_DETECTED / str(rec.id) / f"crop_{lab.id}.jpg"
        _write_rgb(crop_path, crop_rgb)
        lab.crop_path = str(crop_path)
        apply_crop_identity(db, lab, crop_rgb)
        crops.append(
            {
                "label_id": lab.id,
                "crop_id": lab.crop_id,
                "confidence": item.confidence,
                "x1": item.x1,
                "y1": item.y1,
                "x2": item.x2,
                "y2": item.y2,
                "grade": None,
            }
        )
    return crops, overlay


def validate_registered_patient(db: Session, patient_id: str | None) -> str:
    patient_id = (patient_id or "").strip()
    if not 2 <= len(patient_id) <= 64 or any(ord(char) < 32 for char in patient_id):
        raise ValueError("Select a valid registered Patient ID.")
    if not db.query(Patient.id).filter(
        Patient.public_id == patient_id,
        Patient.is_anonymous.is_(False),
    ).first():
        raise ValueError(
            f"Patient ID {patient_id} is not registered. Select an existing patient or register one first."
        )
    return patient_id


def validate_training_upload_identities(
    db: Session,
    filenames: list[str],
    identity_map: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, str | None]]:
    """Validate explicit patient links while allowing photos to remain Unknown."""
    names = [Path(name).name for name in filenames]
    if not names or len({name.casefold() for name in names}) != len(names):
        raise ValueError("Upload filenames must be nonempty and unique in this batch.")
    normalized = {str(key).casefold(): dict(value) for key, value in identity_map.items()}
    expected = {name.casefold() for name in names}
    if set(normalized) != expected:
        raise ValueError("The upload map must cover every filename exactly once.")
    patient_ids: set[str] = set()
    image_ids: list[str] = []
    for name in names:
        metadata = normalized[name.casefold()]
        patient_id = str(metadata.get("patient_id") or "").strip()
        metadata["patient_id"] = validate_registered_patient(db, patient_id) if patient_id else None
        if metadata.get("image_id"):
            metadata["image_id"] = validate_image_id(metadata["image_id"])
            image_ids.append(metadata["image_id"])
        patient_ids.add(metadata["patient_id"])
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Upload map contains duplicate image_id values.")
    existing_image_ids = {
        row[0]
        for row in db.query(TrainingImage.image_id).filter(TrainingImage.image_id.in_(image_ids)).all()
    } if image_ids else set()
    if existing_image_ids:
        raise ValueError("These image IDs already exist: " + ", ".join(sorted(existing_image_ids)[:5]))
    return normalized


def _training_upload_payload(rec: TrainingImage) -> dict:
    return {
        "training_image_id": rec.id,
        "upload_batch_id": rec.upload_batch_id,
        "image_id": rec.image_id,
        "patient_id": rec.patient_id,
        "filename": rec.filename,
        "duplicate_status": rec.duplicate_status,
        "canonical_id": rec.canonical_id,
        "excluded_from_dataset": bool(rec.exclude_from_dataset),
        "processing_status": rec.processing_status,
        "processing_error": rec.processing_error,
        "processing_attempts": int(rec.processing_attempts or 0),
        "processing_updated_at": rec.processing_updated_at.isoformat() if rec.processing_updated_at else None,
        "n_crops": rec.n_crops,
        "crop_count": rec.crop_count,
        "review_status": rec.review_status,
        "detector_ran": rec.processing_status == UPLOAD_COMPLETED and rec.duplicate_status == STATUS_UNIQUE,
        "split_name": rec.split_name or "UNASSIGNED",
        "dataset_version": rec.dataset_version,
    }


def stage_training_image(
    db: Session,
    filename: str,
    data: bytes,
    decode_fn,
    *,
    patient_id: str | None,
    image_id: str | None = None,
    upload_batch_id: str,
) -> dict:
    """Safely persist one original image and queue detector work."""
    ensure_dirs()
    filename = Path(filename).name
    patient_id = validate_registered_patient(db, patient_id) if patient_id else None
    patient_record = (
        db.query(Patient)
        .filter(Patient.public_id == patient_id, Patient.is_anonymous.is_(False))
        .first()
        if patient_id
        else None
    )
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"{filename}: only JPG, JPEG, and PNG photos are accepted.")
    if not data:
        raise ValueError(f"{filename}: file is empty.")
    image_id = validate_image_id(image_id) if image_id else uuid.uuid4().hex
    if db.query(TrainingImage).filter(TrainingImage.image_id == image_id).first():
        raise ValueError(f"Image ID {image_id} already exists.")

    digest = sha256_bytes(data)
    rgb = None
    phash = None
    status = STATUS_UNIQUE
    canonical_id = None
    invalid_reason = None
    try:
        rgb = decode_fn(data, filename)
        phash = average_hash_hex(rgb)
        status, canonical_id = classify_duplicate(db, digest, phash)
        if status == STATUS_EXACT and canonical_id is None:
            status = STATUS_UNIQUE
    except ValueError as exc:
        status = STATUS_INVALID
        invalid_reason = str(exc)

    stored_filename = f"{image_id}{suffix}"
    dest = TRAIN_UPLOADS / stored_filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = TrainingImage(
        filename=stored_filename,
        original_filename=filename,
        path=str(dest),
        image_id=image_id,
        patient_id=patient_id,
        patient_record_id=patient_record.id if patient_record else None,
        upload_timestamp=datetime.utcnow(),
        upload_batch_id=upload_batch_id,
        processing_status=(
            UPLOAD_INVALID if status == STATUS_INVALID
            else UPLOAD_DUPLICATE if status != STATUS_UNIQUE
            else UPLOAD_QUEUED
        ),
        processing_error=invalid_reason,
        processing_attempts=0,
        processing_updated_at=datetime.utcnow(),
        annotation_status="UNLABELED",
        split_name="UNASSIGNED",
        split_locked=False,
        dataset_version=None,
        crop_count=0,
        review_status=("FLAGGED" if status != STATUS_UNIQUE else "PENDING"),
        detector_version="Tooth Detector V2",
        overlay_path=None,
        n_crops=0,
        content_hash=digest,
        phash=phash,
        duplicate_status=status,
        canonical_id=canonical_id,
        boxes_verified=False,
        exclude_from_dataset=status != STATUS_UNIQUE,
    )
    created_file = False
    try:
        with dest.open("xb") as handle:
            handle.write(data)
        created_file = True
        db.add(rec)
        db.commit()
        db.refresh(rec)
    except Exception:
        db.rollback()
        if created_file:
            dest.unlink(missing_ok=True)
        raise
    return _training_upload_payload(rec)


def process_training_image(db: Session, training_image_id: int, *, retry_failed: bool = False) -> dict | None:
    """Claim and run Detector V2 once; persist failures so an admin can resume."""
    eligible = [UPLOAD_QUEUED]
    if retry_failed:
        eligible.append(UPLOAD_FAILED)
    claimed = (
        db.query(TrainingImage)
        .filter(
            TrainingImage.id == training_image_id,
            TrainingImage.processing_status.in_(eligible),
            TrainingImage.duplicate_status == STATUS_UNIQUE,
        )
        .update(
            {
                TrainingImage.processing_status: UPLOAD_PROCESSING,
                TrainingImage.processing_error: None,
                TrainingImage.processing_attempts: TrainingImage.processing_attempts + 1,
                TrainingImage.processing_updated_at: datetime.utcnow(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    rec = db.query(TrainingImage).filter(TrainingImage.id == training_image_id).first()
    if not claimed or rec is None:
        return _training_upload_payload(rec) if rec else None
    try:
        if not detector_available():
            raise RuntimeError(detector_error() or "Tooth Detector V2 is unavailable.")
        from .portal_service import _decode_upload

        data = Path(rec.path).read_bytes()
        rgb = _decode_upload(data, rec.filename)
        _detect_and_store_crops(db, rec, rgb)
        rec.processing_status = UPLOAD_COMPLETED
        rec.processing_error = None
        rec.processing_updated_at = datetime.utcnow()
        rec.review_status = "PENDING"
        rec.split_name = "UNASSIGNED"
        rec.split_locked = False
        rec.dataset_version = None
        _sync_image_annotation_status(rec)
        db.commit()
        db.refresh(rec)
        return _training_upload_payload(rec)
    except Exception as exc:
        db.rollback()
        logger.exception("Tooth Detector V2 failed for upload image %s", training_image_id)
        rec = db.query(TrainingImage).filter(TrainingImage.id == training_image_id).first()
        if rec is None:
            return None
        rec.processing_status = UPLOAD_FAILED
        rec.processing_error = f"{type(exc).__name__}: {exc}"[:2000]
        rec.processing_updated_at = datetime.utcnow()
        rec.review_status = "DETECTION_FAILED"
        db.commit()
        db.refresh(rec)
        return _training_upload_payload(rec)


def process_training_upload_batch(upload_batch_id: str, *, retry_failed: bool = False) -> None:
    """Background worker; queued and failed records survive process restarts."""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        allowed = [UPLOAD_QUEUED]
        if retry_failed:
            allowed.append(UPLOAD_FAILED)
        image_ids = [
            row[0]
            for row in db.query(TrainingImage.id)
            .filter(
                TrainingImage.upload_batch_id == upload_batch_id,
                TrainingImage.processing_status.in_(allowed),
                TrainingImage.duplicate_status == STATUS_UNIQUE,
            )
            .order_by(TrainingImage.id.asc())
            .all()
        ]
        for image_id in image_ids:
            process_training_image(db, image_id, retry_failed=retry_failed)
    finally:
        db.close()


def training_upload_history(db: Session, limit: int = 1000) -> dict:
    rows = (
        db.query(TrainingImage)
        .filter(TrainingImage.upload_batch_id.is_not(None))
        .order_by(TrainingImage.upload_timestamp.desc(), TrainingImage.id.desc())
        .limit(max(1, min(int(limit), 2000)))
        .all()
    )
    batches: dict[str, dict] = {}
    images = []
    for rec in rows:
        item = _training_upload_payload(rec)
        item["uploaded_at"] = rec.upload_timestamp.isoformat() if rec.upload_timestamp else None
        images.append(item)
        batch_id = rec.upload_batch_id
        summary = batches.setdefault(
            batch_id,
            {
                "upload_batch_id": batch_id,
                "uploaded_at": item["uploaded_at"],
                "total": 0,
                "completed": 0,
                "processing": 0,
                "queued": 0,
                "failed": 0,
                "duplicates": 0,
                "invalid": 0,
                "tooth_crops": 0,
            },
        )
        summary["total"] += 1
        summary["tooth_crops"] += int(rec.n_crops or 0)
        status_key = {
            UPLOAD_COMPLETED: "completed",
            UPLOAD_PROCESSING: "processing",
            UPLOAD_QUEUED: "queued",
            UPLOAD_FAILED: "failed",
            UPLOAD_DUPLICATE: "duplicates",
            UPLOAD_INVALID: "invalid",
        }.get(rec.processing_status, "failed")
        summary[status_key] += 1
    for summary in batches.values():
        if summary["queued"] or summary["processing"]:
            summary["status"] = "PROCESSING"
        elif summary["failed"]:
            summary["status"] = "FAILED"
        elif summary["duplicates"] or summary["invalid"]:
            summary["status"] = "COMPLETED_WITH_WARNINGS"
        else:
            summary["status"] = "COMPLETED"
    return {"batches": list(batches.values()), "images": images, "count": len(images)}


def ingest_training_image(
    db: Session,
    filename: str,
    data: bytes,
    decode_fn,
    *,
    patient_id: str,
    image_id: str | None = None,
    upload_batch_id: str | None = None,
) -> dict:
    """Synchronous compatibility wrapper for one registered patient's photo."""
    staged = stage_training_image(
        db,
        filename,
        data,
        decode_fn,
        patient_id=patient_id,
        image_id=image_id,
        upload_batch_id=upload_batch_id or uuid.uuid4().hex,
    )
    if staged["processing_status"] == UPLOAD_QUEUED:
        result = process_training_image(db, staged["training_image_id"])
        return result or staged
    return staged


def update_training_identity(
    db: Session,
    training_image_id: int,
    patient_id: str,
    image_id: str | None = None,
) -> dict:
    rec = db.query(TrainingImage).filter(TrainingImage.id == training_image_id).first()
    if rec is None:
        raise KeyError(training_image_id)
    patient_id = validate_registered_patient(db, patient_id)
    patient_record = db.query(Patient).filter(
        Patient.public_id == patient_id,
        Patient.is_anonymous.is_(False),
    ).one()
    new_image_id = validate_image_id(image_id) if image_id else rec.image_id
    duplicate = (
        db.query(TrainingImage)
        .filter(TrainingImage.image_id == new_image_id, TrainingImage.id != rec.id)
        .first()
    )
    if duplicate:
        raise ValueError(f"Image ID {new_image_id} already exists.")
    old = {
        "patient_id": rec.patient_id,
        "patient_record_id": rec.patient_record_id,
        "image_id": rec.image_id,
    }
    rec.patient_id = patient_id
    rec.patient_record = patient_record
    rec.image_id = new_image_id
    # Identity edits invalidate the current, mutable assignment. Existing
    # DatasetVersion manifests remain immutable historical records.
    rec.dataset_version = None
    rec.split_name = "UNASSIGNED"
    rec.split_locked = False
    _audit(
        db,
        "identity_updated",
        "training_image",
        rec.image_id,
        {"old": old, "new": {"patient_id": patient_id, "image_id": new_image_id}},
    )
    db.commit()
    return {
        "training_image_id": rec.id,
        "image_id": rec.image_id,
        "patient_id": rec.patient_id,
        "split_name": rec.split_name,
    }


def list_training_images(db: Session) -> dict:
    rows = db.query(TrainingImage).order_by(TrainingImage.id.asc()).all()
    items = []
    active_rows = [r for r in rows if r.is_active is not False]
    counts = Counter(r.duplicate_status or STATUS_UNIQUE for r in active_rows)
    processing_counts = Counter(r.processing_status or UPLOAD_COMPLETED for r in active_rows)
    for r in rows:
        items.append(
            {
                "training_image_id": r.id,
                "image_id": r.image_id,
                "patient_id": r.patient_id,
                "filename": r.filename,
                "duplicate_status": r.duplicate_status,
                "canonical_id": r.canonical_id,
                "n_crops": r.n_crops,
                "crop_count": r.crop_count,
                "split_name": r.split_name,
                "split_locked": bool(r.split_locked),
                "dataset_version": r.dataset_version,
                "annotation_status": r.annotation_status,
                "review_status": r.review_status,
                "upload_timestamp": r.upload_timestamp.isoformat() if r.upload_timestamp else None,
                "upload_batch_id": r.upload_batch_id,
                "processing_status": r.processing_status or UPLOAD_COMPLETED,
                "processing_error": r.processing_error,
                "processing_attempts": int(r.processing_attempts or 0),
                "processing_updated_at": r.processing_updated_at.isoformat() if r.processing_updated_at else None,
                "boxes_verified": bool(r.boxes_verified),
                "exclude_from_dataset": bool(r.exclude_from_dataset),
                "is_active": bool(r.is_active if r.is_active is not None else True),
                "content_hash": r.content_hash,
            }
        )
    return {
        "count": len(items),
        "active_count": len(active_rows),
        "status_counts": dict(counts),
        "processing_status_counts": dict(processing_counts),
        "images": items,
    }


def training_labeling_photo_inventory(db: Session, search: str | None = None) -> dict:
    """Read-only photo picker and per-photo ICDAS progress for the admin UI."""
    images = db.query(TrainingImage).order_by(TrainingImage.id.asc()).all()
    labels = (
        db.query(TrainingLabel)
        .join(TrainingImage)
        .order_by(TrainingImage.id.asc(), TrainingLabel.index_in_image.asc(), TrainingLabel.id.asc())
        .all()
    )
    by_image: dict[int, list[TrainingLabel]] = defaultdict(list)
    label_by_id = {int(label.id): label for label in labels}
    image_by_id = {int(image.id): image for image in images}
    for label in labels:
        by_image[int(label.image_id)].append(label)

    photos = []
    for image in images:
        image_labels = [label for label in by_image.get(int(image.id), []) if label.active is not False]
        eligible = [
            label
            for label in image_labels
            if label.box_verified is True
            and image.boxes_verified is True
            and (image.duplicate_status in {STATUS_UNIQUE, None})
            and image.is_active is not False
        ]
        grade_counts = {str(grade): 0 for grade in range(5)}
        labeled = uncertain = ungradable = unlabeled = 0
        first_unlabeled = None
        last_labeled = None
        for label in eligible:
            status = _effective_annotation_status(label)
            if status == "LABELED" and label.grade is not None:
                labeled += 1
                grade_counts[str(int(label.grade))] += 1
                last_labeled = label
            elif status == "UNCERTAIN":
                uncertain += 1
            elif status == "UNGRADABLE":
                ungradable += 1
            else:
                unlabeled += 1
                if first_unlabeled is None:
                    first_unlabeled = label

        if image.is_active is False:
            status = "INACTIVE"
        elif image.duplicate_status not in {STATUS_UNIQUE, None}:
            status = "EXCLUDED DUPLICATE"
        elif (image.processing_status or UPLOAD_COMPLETED) not in {UPLOAD_COMPLETED, UPLOAD_DUPLICATE}:
            status = "PROCESSING"
        elif not image.boxes_verified:
            status = "NEEDS BOX VERIFICATION"
        elif not eligible:
            status = "NO ELIGIBLE CROPS"
        elif labeled == len(eligible):
            status = "FULLY LABELED"
        elif labeled:
            status = "PARTIALLY LABELED"
        elif uncertain or ungradable:
            status = "PARTIALLY LABELED"
        else:
            status = "NOT LABELED"

        original_filename = image.original_filename
        display_filename = original_filename or image.filename
        photo_id = f"PHOTO-{int(image.id):03d}"
        item = {
            "training_image_id": int(image.id),
            "photo_id": photo_id,
            "photo_number": int(image.id),
            "reference_number": image.image_id,
            "image_id": image.image_id,
            "original_filename": original_filename,
            "filename": display_filename,
            "original_filename_available": bool(original_filename),
            "patient_id": image.patient_id or (image.patient_record.public_id if image.patient_record else None),
            "split_name": image.split_name or "UNASSIGNED",
            "processing_status": image.processing_status or UPLOAD_COMPLETED,
            "duplicate_status": image.duplicate_status or STATUS_UNIQUE,
            "is_active": bool(image.is_active if image.is_active is not None else True),
            "boxes_verified": bool(image.boxes_verified),
            "detected_crops": len(image_labels),
            "eligible_crops": len(eligible),
            "labeled_crops": labeled,
            "remaining_unlabeled_crops": unlabeled,
            "uncertain_crops": uncertain,
            "ungradable_crops": ungradable,
            "grade_counts": grade_counts,
            "labeling_status": status,
            "first_unlabeled_crop_id": first_unlabeled.crop_id if first_unlabeled else None,
            "first_unlabeled_label_id": int(first_unlabeled.id) if first_unlabeled else None,
            "last_labeled_crop_id": last_labeled.crop_id if last_labeled else None,
            "last_labeled_label_id": int(last_labeled.id) if last_labeled else None,
        }
        needle = str(search or "").strip().casefold()
        searchable = (
            photo_id,
            str(image.id),
            str(image.image_id or ""),
            str(display_filename or ""),
            str(image.patient_id or ""),
        )
        if not needle or any(needle in value.casefold() for value in searchable):
            photos.append(item)

    progress = db.query(LabelingProgress).filter(LabelingProgress.track == "initial").first()
    resume_position = None
    if progress and progress.label_id:
        resume_label = label_by_id.get(int(progress.label_id))
        if resume_label:
            resume_image = image_by_id.get(int(resume_label.image_id))
            if resume_image:
                resume_position = {
                    "photo_id": f"PHOTO-{int(resume_image.id):03d}",
                    "training_image_id": int(resume_image.id),
                    "crop_id": resume_label.crop_id,
                    "label_id": int(resume_label.id),
                    "updated_at": progress.updated_at.isoformat() if progress.updated_at else None,
                }

    latest_saved = (
        db.query(TrainingAuditEvent)
        .filter(TrainingAuditEvent.event_type == "label_saved")
        .order_by(TrainingAuditEvent.id.desc())
        .first()
    )
    last_saved_position = None
    if latest_saved:
        saved_label = next(
            (label for label in labels if str(label.crop_id) == str(latest_saved.entity_id)),
            None,
        )
        if saved_label:
            saved_image = image_by_id.get(int(saved_label.image_id))
            if saved_image:
                try:
                    saved_details = json.loads(latest_saved.details_json or "{}")
                except (TypeError, ValueError):
                    saved_details = {}
                saved_grade = (saved_details.get("new") or {}).get("grade", saved_label.grade)
                last_saved_position = {
                    "photo_id": f"PHOTO-{int(saved_image.id):03d}",
                    "training_image_id": int(saved_image.id),
                    "crop_id": saved_label.crop_id,
                    "label_id": int(saved_label.id),
                    "grade": saved_grade,
                    "saved_at": latest_saved.created_at.isoformat() if latest_saved.created_at else None,
                }

    return {
        "count": len(photos),
        "photos": photos,
        "saved_resume_position": resume_position,
        "last_saved_label": last_saved_position,
        "note": "Read-only photo and label progress inventory.",
    }


def training_image_detail(db: Session, image_id: int) -> dict:
    from .portal_service import _b64_rgb, _read_rgb

    rec = db.query(TrainingImage).filter(TrainingImage.id == image_id).first()
    if rec is None:
        raise KeyError(image_id)
    rgb = _read_rgb(rec.path)
    ov = _read_rgb(rec.overlay_path) if rec.overlay_path else None
    boxes = []
    for lab in sorted(rec.labels, key=lambda x: x.index_in_image):
        if lab.active is False:
            continue
        crop = _read_rgb(lab.crop_path) if Path(lab.crop_path).exists() else None
        boxes.append(
            {
                "label_id": lab.id,
                "crop_id": lab.crop_id,
                "x1": lab.x1,
                "y1": lab.y1,
                "x2": lab.x2,
                "y2": lab.y2,
                "confidence": lab.confidence,
                "grade": lab.grade,
                "annotation_status": _effective_annotation_status(lab),
                "box_verified": bool(lab.box_verified),
                "crop_duplicate_status": lab.crop_duplicate_status or CROP_UNIQUE,
                "duplicate_of_label_id": lab.duplicate_of_label_id,
                "crop_status_caption": crop_status_caption(lab),
                "crop_base64": _b64_rgb(crop),
            }
        )
    return {
        "training_image_id": rec.id,
        "image_id": rec.image_id,
        "patient_id": rec.patient_id,
        "filename": rec.filename,
        "duplicate_status": rec.duplicate_status,
        "canonical_id": rec.canonical_id,
        "boxes_verified": bool(rec.boxes_verified),
        "is_active": bool(rec.is_active if rec.is_active is not None else True),
        "n_crops": rec.n_crops,
        "crop_count": rec.crop_count,
        "split_name": rec.split_name,
        "dataset_version": rec.dataset_version,
        "annotation_status": rec.annotation_status,
        "review_status": rec.review_status,
        "upload_batch_id": rec.upload_batch_id,
        "processing_status": rec.processing_status or UPLOAD_COMPLETED,
        "processing_error": rec.processing_error,
        "processing_attempts": int(rec.processing_attempts or 0),
        "original_base64": _b64_rgb(rgb),
        "overlay_base64": _b64_rgb(ov),
        "boxes": boxes,
    }


def deactivate_training_image(db: Session, image_id: int) -> dict:
    """Soft-deactivates a training photo and its tooth crops/labels from future dataset builds."""
    rec = db.query(TrainingImage).filter(TrainingImage.id == image_id).first()
    if rec is None:
        raise KeyError(image_id)

    rec.is_active = False
    rec.exclude_from_dataset = True
    rec.review_status = "REJECTED"
    _audit(
        db,
        "training_image_deactivated",
        "training_image",
        rec.image_id,
        {"training_image_id": rec.id, "patient_id": rec.patient_id},
    )

    for lab in rec.labels:
        lab.active = False
        if lab.crop_hash:
            sync_exact_hash_group(db, lab.crop_hash)

    db.commit()
    return {
        "training_image_id": rec.id,
        "filename": rec.filename,
        "is_active": False,
        "deactivated": True,
        "message": f"Photo {rec.id} ({rec.filename}) deactivated successfully.",
    }


def save_training_boxes(db: Session, image_id: int, boxes: list[dict], deleted_ids: list[int]) -> dict:
    from .portal_service import _read_rgb

    rec = db.query(TrainingImage).filter(TrainingImage.id == image_id).first()
    if rec is None:
        raise KeyError(image_id)
    if rec.duplicate_status != STATUS_UNIQUE:
        raise ValueError("Box review applies only to UNIQUE canonical training images.")
    rgb = _read_rgb(rec.path)
    if rgb is None:
        raise ValueError("Original training image could not be read.")
    inactivated_hashes: list[str] = []
    for lid in deleted_ids or []:
        lab = db.query(TrainingLabel).filter(TrainingLabel.id == lid, TrainingLabel.image_id == rec.id).first()
        if lab:
            _audit(
                db,
                "crop_rejected",
                "training_crop",
                lab.crop_id,
                {"label_id": lab.id, "patient_id": rec.patient_id, "image_id": rec.image_id},
            )
            if lab.crop_hash:
                inactivated_hashes.append(lab.crop_hash)
            lab.active = False
            lab.box_verified = False
    kept = []
    h, w = rgb.shape[:2]
    for idx, box in enumerate(boxes or []):
        try:
            x1, y1, x2, y2 = validate_xyxy_box(box["x1"], box["y1"], box["x2"], box["y2"], w, h)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bounding box: {exc}") from exc
        crop_rgb = crop_xyxy_rgb(rgb, x1, y1, x2, y2)
        lid = box.get("label_id") or box.get("id")
        lab = None
        if lid:
            lab = (
                db.query(TrainingLabel)
                .filter(TrainingLabel.id == int(lid), TrainingLabel.image_id == rec.id)
                .first()
            )
        if lab is None:
            lab = TrainingLabel(
                image_id=rec.id,
                grade=None,
                annotation_status="UNLABELED",
                confidence=0.0,
                crop_path="",
            )
            db.add(lab)
            db.flush()
        old_box = (lab.x1, lab.y1, lab.x2, lab.y2)
        new_box = (x1, y1, x2, y2)
        if not lab.crop_path:
            _audit(
                db,
                "tooth_box_added",
                "training_crop",
                lab.crop_id,
                {
                    "label_id": lab.id,
                    "patient_id": rec.patient_id,
                    "image_id": rec.image_id,
                    "new_box": new_box,
                },
            )
        elif old_box != new_box:
            _audit(
                db,
                "tooth_box_updated",
                "training_crop",
                lab.crop_id,
                {
                    "label_id": lab.id,
                    "patient_id": rec.patient_id,
                    "image_id": rec.image_id,
                    "old_box": old_box,
                    "new_box": new_box,
                },
            )
        old_annotation_status = _effective_annotation_status(lab)
        if lab.crop_path and old_box != new_box and old_annotation_status != "UNLABELED":
            _audit(
                db,
                "annotation_invalidated_by_recrop",
                "training_crop",
                lab.crop_id,
                {
                    "old_grade": lab.grade,
                    "old_annotation_status": old_annotation_status,
                    "old_box": old_box,
                    "new_box": new_box,
                },
            )
            lab.grade = None
            lab.annotation_status = "UNLABELED"
            lab.skipped = False
        crop_path = TRAIN_DETECTED / str(rec.id) / f"crop_{lab.id}.jpg"
        _write_rgb(crop_path, crop_rgb)
        lab.crop_path = str(crop_path)
        lab.x1, lab.y1, lab.x2, lab.y2 = x1, y1, x2, y2
        lab.index_in_image = idx
        lab.active = True
        lab.box_verified = False
        lab.confidence = float(box.get("confidence") or lab.confidence or 0.0)
        apply_crop_identity(db, lab, crop_rgb)
        kept.append(lab)
    for digest in inactivated_hashes:
        sync_exact_hash_group(db, digest)
    rec.n_crops = len(kept)
    rec.crop_count = len(kept)
    rec.boxes_verified = False
    rec.review_status = "PENDING"
    rec.annotation_status = "UNLABELED"
    overlay = draw_boxes_rgb(rgb, [{"x1": l.x1, "y1": l.y1, "x2": l.x2, "y2": l.y2} for l in kept])
    ov_path = TRAIN_DETECTED / str(rec.id) / "overlay.jpg"
    _write_rgb(ov_path, overlay)
    rec.overlay_path = str(ov_path)
    db.commit()
    return training_image_detail(db, rec.id)


def verify_training_boxes(db: Session, image_id: int) -> dict:
    rec = db.query(TrainingImage).filter(TrainingImage.id == image_id).first()
    if rec is None:
        raise KeyError(image_id)
    active = [l for l in rec.labels if l.active]
    if not active:
        raise ValueError("Add at least one tooth box before verifying.")
    rec.boxes_verified = True
    rec.review_status = "APPROVED"
    for lab in active:
        lab.box_verified = True
        if lab.grade is not None:
            lab.annotation_status = "LABELED"
        elif _effective_annotation_status(lab) == "UNGRADABLE":
            lab.annotation_status = "UNGRADABLE"
            lab.skipped = False
        elif lab.skipped:
            lab.annotation_status = "UNCERTAIN"
        else:
            lab.annotation_status = "UNLABELED"
    _sync_image_annotation_status(rec)
    from .admin_portal_ops import remember_progress

    remember_progress(db, "verification", active[0].id, rec.id)
    _audit(
        db,
        "detections_verified",
        "training_image",
        rec.image_id,
        {"training_image_id": rec.id, "patient_id": rec.patient_id, "active_crop_count": len(active)},
    )
    db.commit()
    return {"training_image_id": rec.id, "boxes_verified": True, "n_crops": len(active)}


def _first_unlabeled(ordered: list[TrainingLabel]) -> TrainingLabel | None:
    for lab in ordered:
        if lab.grade is None and not lab.skipped and _effective_annotation_status(lab) == "UNLABELED":
            return lab
    return None


def labeling_queue(
    db: Session,
    image_id: int | None = None,
    label_id: int | None = None,
    resume: bool = False,
    patient_id: str | None = None,
    split_name: str | None = None,
    grade: int | None = None,
    annotation_status: str | None = None,
    mode: str | None = None,
    track: str | None = None,
) -> dict:
    from .admin_portal_ops import progress_for, remember_progress
    from .portal_service import _b64_rgb, _read_rgb

    inventory = dataset_inventory(db)
    chosen = (mode or ("resume" if resume else "")).strip().lower()
    ordered = _ordered_verified_labels(
        db,
        patient_id=patient_id,
        split_name=split_name,
        grade=grade,
        annotation_status=annotation_status,
    )
    if image_id is not None and chosen in {
        "resume", "first_unlabeled", "beginning", "photo", "crop", "relabel", "review", "skipped"
    }:
        ordered = [lab for lab in ordered if int(lab.image_id) == int(image_id)]
    active_track = track or (
        "verification"
        if chosen in {"photo", "verification"}
        else "relabel"
        if chosen == "relabel"
        else "review"
        if chosen == "review"
        else "initial"
    )
    if chosen == "review":
        ordered = [lab for lab in ordered if lab.grade is not None]
    elif chosen == "relabel":
        ordered = [lab for lab in ordered if lab.grade is not None]
    elif chosen == "skipped":
        ordered = [
            lab
            for lab in ordered
            if lab.skipped or _effective_annotation_status(lab) in {"UNCERTAIN", "UNGRADABLE"}
        ]
    if chosen == "photo" and image_id:
        ordered = [lab for lab in ordered if int(lab.image_id) == int(image_id)]
    ids = [lab.id for lab in ordered]
    unlabeled = _first_unlabeled(ordered)
    resume_label_id = unlabeled.id if unlabeled else None
    stored = progress_for(db, active_track)
    current_lab = None
    if chosen == "beginning" and ordered:
        current_lab = ordered[0]
    elif chosen == "first_unlabeled":
        current_lab = unlabeled
    elif chosen in {"crop", "relabel"} and label_id:
        current_lab = next((lab for lab in ordered if lab.id == int(label_id)), None)
    elif chosen in {"crop", "relabel"}:
        current_lab = None
    elif chosen == "photo" and image_id:
        current_lab = next((lab for lab in ordered if lab.id == int(label_id)), None) if label_id else (ordered[0] if ordered else None)
    elif label_id and chosen not in {"resume"}:
        current_lab = next((lab for lab in ordered if lab.id == int(label_id)), None)
    elif chosen in {"resume", ""} or resume:
        if (
            stored
            and stored.label_id
            and any(
                lab.id == stored.label_id and lab.grade is None and not lab.skipped
                for lab in ordered
            )
        ):
            current_lab = next(lab for lab in ordered if lab.id == stored.label_id)
        else:
            current_lab = unlabeled
    elif image_id:
        current_lab = next((lab for lab in ordered if lab.image_id == int(image_id)), None)
    elif resume_label_id:
        current_lab = unlabeled
    elif ordered:
        current_lab = ordered[0]
    if current_lab is not None and chosen in {
        "resume", "first_unlabeled", "beginning", "photo", "crop", "relabel", "review", "verification", "skipped", ""
    }:
        remember_progress(db, active_track, current_lab.id, current_lab.image_id)
        db.commit()

    current = None
    prev_id = next_id = None
    tooth_on_photo = photo_index = 0
    if current_lab is not None:
        idx = ids.index(current_lab.id)
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx + 1 < len(ids) else None
        same = [l for l in ordered if l.image_id == current_lab.image_id]
        tooth_on_photo = next((i + 1 for i, l in enumerate(same) if l.id == current_lab.id), 1)
        unique_ids = []
        for l in ordered:
            if l.image_id not in unique_ids:
                unique_ids.append(l.image_id)
        photo_index = unique_ids.index(current_lab.image_id) + 1 if current_lab.image_id in unique_ids else 0
        rgb = _read_rgb(current_lab.crop_path) if Path(current_lab.crop_path).exists() else None
        current = {
            "label_id": current_lab.id,
            "crop_id": current_lab.crop_id,
            "training_image_id": current_lab.image_id,
            "photo_id": f"PHOTO-{int(current_lab.image_id):03d}",
            "image_id": current_lab.image.image_id if current_lab.image else None,
            "patient_id": current_lab.image.patient_id if current_lab.image else None,
            "split_name": current_lab.image.split_name if current_lab.image else "UNASSIGNED",
            "filename": current_lab.image.filename if current_lab.image else "",
            "grade": current_lab.grade,
            "annotation_status": _effective_annotation_status(current_lab),
            "skipped": bool(current_lab.skipped),
            "crop_base64": _b64_rgb(rgb),
            "tooth_on_photo": tooth_on_photo,
            "teeth_on_photo": len(same),
            "global_index": idx + 1,
            "global_total": len(ids),
            "photo_index": photo_index,
            "photo_total": len(unique_ids),
            "crop_duplicate_status": current_lab.crop_duplicate_status or CROP_UNIQUE,
            "duplicate_of_label_id": current_lab.duplicate_of_label_id,
            "crop_status_caption": crop_status_caption(current_lab),
        }
    return {
        **inventory,
        "n_crops": inventory["crops"]["verified"],
        "n_labeled": inventory["crops"]["labeled"],
        "n_unlabeled": inventory["crops"]["unlabeled"],
        "n_skipped": inventory["crops"]["skipped"],
        "resume_label_id": resume_label_id,
        "prev_label_id": prev_id,
        "next_label_id": next_id,
        "current": current,
        "next_enabled": next_id is not None,
        "mode": chosen or "resume",
        "track": active_track,
        "n_verified": inventory["crops"]["verified"],
        "n_remaining": inventory["crops"]["unlabeled"],
        "progress_percent": round(
            (100.0 * inventory["crops"]["labeled"] / inventory["crops"]["verified"]), 1
        )
        if inventory["crops"]["verified"]
        else 0.0,
        "note": "One verified UNIQUE crop at a time. Groq is not used. Missing 2/3 on a photo is valid.",
    }


def undo_icdas_label(db: Session, label_id: int) -> dict:
    """Restore the previous saved grade from the audit trail. Does not invent a grade."""
    lab = db.query(TrainingLabel).filter(TrainingLabel.id == label_id).first()
    if lab is None:
        raise KeyError(label_id)
    events = (
        db.query(TrainingAuditEvent)
        .filter(TrainingAuditEvent.entity_id == lab.crop_id, TrainingAuditEvent.event_type == "label_saved")
        .order_by(TrainingAuditEvent.id.desc())
        .all()
    )
    previous_grade = None
    for event in events:
        details = json.loads(event.details_json or "{}")
        old = (details.get("old") or {}).get("grade")
        if old is not None:
            previous_grade = int(old)
            break
    if previous_grade is None:
        raise ValueError("This crop has no earlier ICDAS grade to restore.")
    return save_icdas_label(db, label_id, previous_grade, confirm_overwrite=True, track="relabel")


def save_icdas_label(
    db: Session,
    label_id: int,
    grade: int,
    confirm_overwrite: bool = False,
    track: str = "initial",
) -> dict:
    from .admin_portal_ops import remember_progress

    grade = validate_icdas_grade(grade)
    lab = db.query(TrainingLabel).filter(TrainingLabel.id == label_id).first()
    if lab is None:
        raise KeyError(label_id)
    img = lab.image
    if img is None or (img.duplicate_status not in {STATUS_UNIQUE, None}):
        raise ValueError("ICDAS labels can only be saved on UNIQUE training photos.")
    if lab.active is False or not lab.box_verified or not img.boxes_verified:
        raise ValueError("Crop must be an active verified tooth box before labeling.")
    if not Path(lab.crop_path).exists():
        raise ValueError("Crop file is missing or unreadable, so the label was not saved.")
    if lab.grade is not None and int(lab.grade) != int(grade) and not confirm_overwrite:
        return {
            "label_id": lab.id,
            "saved": False,
            "needs_confirmation": True,
            "previous_grade": int(lab.grade),
            "grade": int(grade),
        }
    previous = {
        "grade": lab.grade,
        "annotation_status": lab.annotation_status,
    }
    lab.grade = grade
    lab.annotation_status = "LABELED"
    lab.skipped = False
    labeled_dir = TRAIN_LABELED / str(grade)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    dest = labeled_dir / f"{lab.id}_{Path(lab.crop_path).name}"
    if Path(lab.crop_path).exists():
        shutil.copy2(lab.crop_path, dest)
    apply_crop_identity(db, lab)
    _sync_image_annotation_status(img)
    _audit(
        db,
        "label_saved",
        "training_crop",
        lab.crop_id,
        {
            "label_id": lab.id,
            "patient_id": img.patient_id,
            "image_id": img.image_id,
            "old": previous,
            "new": {"grade": grade, "annotation_status": "LABELED"},
        },
    )
    ordered = _ordered_verified_labels(db)
    nxt = None
    seen = False
    for item in ordered:
        if seen and item.grade is None and not item.skipped and track == "initial":
            nxt = item
            break
        if item.id == lab.id:
            seen = True
    if nxt is None and track == "initial":
        nxt = _first_unlabeled(ordered)
    remember_progress(db, track or "initial", (nxt or lab).id, (nxt or lab).image_id)
    db.commit()
    conflict = lab.crop_duplicate_status == CROP_CONFLICT
    return {
        "label_id": lab.id,
        "grade": grade,
        "saved": True,
        "autosaved": True,
        "crop_duplicate_status": lab.crop_duplicate_status or CROP_UNIQUE,
        "duplicate_of_label_id": lab.duplicate_of_label_id,
        "conflict": conflict,
        "conflict_message": CONFLICT_MESSAGE if conflict else None,
    }


def label_history_listing(db: Session, label_id: int) -> dict:
    """Return append-only history for one crop, including earlier overwrite values."""
    lab = db.query(TrainingLabel).filter(TrainingLabel.id == label_id).first()
    if lab is None:
        raise KeyError(label_id)
    rows = (
        db.query(TrainingAuditEvent)
        .filter(
            TrainingAuditEvent.entity_type == "training_crop",
            TrainingAuditEvent.entity_id == str(lab.crop_id),
        )
        .order_by(TrainingAuditEvent.id.desc())
        .limit(200)
        .all()
    )
    events = []
    for row in rows:
        try:
            details = json.loads(row.details_json or "{}")
        except (TypeError, json.JSONDecodeError):
            details = {}
        events.append(
            {
                "id": row.id,
                "event_type": row.event_type,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "actor": row.actor,
                "details": details,
            }
        )
    return {"label_id": lab.id, "crop_id": lab.crop_id, "count": len(events), "events": events}


def skip_icdas_label(db: Session, label_id: int) -> dict:
    """Leave unlabeled on purpose. Does not invent an ICDAS grade."""
    lab = db.query(TrainingLabel).filter(TrainingLabel.id == label_id).first()
    if lab is None:
        raise KeyError(label_id)
    img = lab.image
    if img is None or (img.duplicate_status not in {STATUS_UNIQUE, None}):
        raise ValueError("Skip applies only to UNIQUE training photos.")
    if lab.active is False or not lab.box_verified or not img.boxes_verified:
        raise ValueError("Crop must be an active verified tooth box.")
    previous = {"grade": lab.grade, "annotation_status": lab.annotation_status}
    lab.skipped = True
    lab.grade = None
    lab.annotation_status = "UNCERTAIN"
    if lab.crop_hash:
        sync_exact_hash_group(db, lab.crop_hash)
    _sync_image_annotation_status(img)
    _audit(
        db,
        "annotation_uncertain",
        "training_crop",
        lab.crop_id,
        {
            "label_id": lab.id,
            "patient_id": img.patient_id,
            "image_id": img.image_id,
            "old": previous,
            "new": {"grade": None, "annotation_status": "UNCERTAIN"},
        },
    )
    from .admin_portal_ops import remember_progress

    ordered = _ordered_verified_labels(db)
    next_label = _first_unlabeled(ordered)
    remember_progress(db, "initial", (next_label or lab).id, (next_label or lab).image_id)
    db.commit()
    return {"label_id": lab.id, "crop_id": lab.crop_id, "skipped": True, "grade": None, "annotation_status": "UNCERTAIN", "saved": True}


def set_crop_annotation_status(db: Session, label_id: int, status: str) -> dict:
    status = (status or "").strip().upper()
    if status not in {"UNCERTAIN", "UNGRADABLE"}:
        raise ValueError("Annotation status must be UNCERTAIN or UNGRADABLE.")
    lab = db.query(TrainingLabel).filter(TrainingLabel.id == label_id).first()
    if lab is None:
        raise KeyError(label_id)
    img = lab.image
    if img is None or (img.duplicate_status not in {STATUS_UNIQUE, None}):
        raise ValueError("Annotation applies only to UNIQUE training photos.")
    if lab.active is False or not lab.box_verified or not img.boxes_verified:
        raise ValueError("Crop must be an active verified tooth box before annotation.")
    old = {"grade": lab.grade, "annotation_status": lab.annotation_status}
    lab.grade = None
    lab.skipped = status == "UNCERTAIN"
    lab.annotation_status = status
    if lab.crop_hash:
        sync_exact_hash_group(db, lab.crop_hash)
    _sync_image_annotation_status(img)
    _audit(
        db,
        f"annotation_{status.lower()}",
        "training_crop",
        lab.crop_id,
        {
            "label_id": lab.id,
            "patient_id": img.patient_id,
            "image_id": img.image_id,
            "old": old,
            "new": {"grade": None, "annotation_status": status},
        },
    )
    from .admin_portal_ops import remember_progress

    ordered = _ordered_verified_labels(db)
    next_label = _first_unlabeled(ordered)
    remember_progress(db, "initial", (next_label or lab).id, (next_label or lab).image_id)
    db.commit()
    return {"label_id": lab.id, "crop_id": lab.crop_id, "grade": None, "annotation_status": status, "saved": True}


def _ordered_verified_labels(
    db: Session,
    *,
    patient_id: str | None = None,
    split_name: str | None = None,
    grade: int | None = None,
    annotation_status: str | None = None,
) -> list[TrainingLabel]:
    query = (
        db.query(TrainingLabel)
        .join(TrainingImage)
        .filter(
            ACTIVE_LABEL,
            TrainingLabel.box_verified.is_(True),
            UNIQUE_IMAGE,
            IS_ACTIVE_IMAGE,
            TrainingImage.boxes_verified.is_(True),
        )
        .order_by(TrainingImage.id.asc(), TrainingLabel.index_in_image.asc(), TrainingLabel.id.asc())
    )
    if patient_id:
        query = query.filter(TrainingImage.patient_id == patient_id)
    if split_name:
        query = query.filter(TrainingImage.split_name == split_name)
    if grade is not None:
        query = query.filter(TrainingLabel.grade == int(grade))
    if annotation_status:
        query = query.filter(TrainingLabel.annotation_status == annotation_status.upper())
    return query.all()


def training_crop_listing(
    db: Session,
    *,
    training_image_id: int | None = None,
    patient_id: str | None = None,
    split_name: str | None = None,
    grade: int | None = None,
    annotation_status: str | None = None,
) -> dict:
    query = (
        db.query(TrainingLabel)
        .join(TrainingImage)
        .filter(UNIQUE_IMAGE, IS_ACTIVE_IMAGE)
        .order_by(TrainingImage.id.asc(), TrainingLabel.index_in_image.asc())
    )
    if training_image_id is not None:
        query = query.filter(TrainingImage.id == int(training_image_id))
    if patient_id:
        query = query.filter(TrainingImage.patient_id == patient_id)
    if split_name:
        query = query.filter(TrainingImage.split_name == split_name)
    if grade is not None:
        query = query.filter(TrainingLabel.grade == int(grade))
    rows = []
    status_filter = annotation_status.upper() if annotation_status else None
    for label in query.all():
        status = _effective_annotation_status(label)
        if status_filter and status != status_filter:
            continue
        image = label.image
        rows.append(
            {
                "label_id": label.id,
                "crop_id": label.crop_id,
                "index_in_image": int(label.index_in_image or 0),
                "training_image_id": image.id,
                "image_id": image.image_id,
                "patient_id": image.patient_id,
                "filename": image.filename,
                "split_name": image.split_name or "UNASSIGNED",
                "dataset_version": image.dataset_version,
                "grade": label.grade,
                "annotation_status": status,
                "review_status": image.review_status,
                "box_verified": bool(label.box_verified and image.boxes_verified),
                "active": bool(label.active is not False),
                "crop_duplicate_status": label.crop_duplicate_status or CROP_UNIQUE,
                "confidence": float(label.confidence or 0.0),
                "box": {"x1": label.x1, "y1": label.y1, "x2": label.x2, "y2": label.y2},
            }
        )
    return {"count": len(rows), "crops": rows}


def training_audit_listing(db: Session, limit: int = 200) -> dict:
    safe_limit = min(max(int(limit or 200), 1), 1000)
    rows = (
        db.query(TrainingAuditEvent)
        .order_by(TrainingAuditEvent.id.desc())
        .limit(safe_limit)
        .all()
    )
    events = []
    for row in rows:
        try:
            details = json.loads(row.details_json or "{}")
        except (TypeError, json.JSONDecodeError):
            details = {"raw": row.details_json}
        events.append(
            {
                "id": row.id,
                "event_type": row.event_type,
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "actor": row.actor,
                "details": details,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return {"count": len(events), "events": events}


CROP_ELIGIBLE = or_(
    TrainingLabel.crop_duplicate_status.is_(None),
    TrainingLabel.crop_duplicate_status == CROP_UNIQUE,
)


def _eligible_label_query(db: Session):
    return (
        db.query(TrainingLabel)
        .join(TrainingImage)
        .filter(
            ACTIVE_LABEL,
            TrainingLabel.box_verified.is_(True),
            TrainingLabel.grade.isnot(None),
            NOT_SKIPPED,
            UNIQUE_IMAGE,
            IS_ACTIVE_IMAGE,
            TrainingImage.boxes_verified.is_(True),
            NOT_EXCLUDED,
            CROP_ELIGIBLE,
        )
    )


def _eligible_fingerprint(labs: list[TrainingLabel]) -> str:
    parts = [
        f"{lab.id}:{lab.image_id}:{lab.grade}:{lab.crop_hash or ''}"
        for lab in sorted(labs, key=lambda x: int(x.id))
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def split_source_images(image_ids: list[int], seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    """70/15/15 split of original intraoral photos (not individual crops)."""
    ids = sorted(set(image_ids))
    rng = np.random.default_rng(seed)
    order = np.array(ids, dtype=int)
    rng.shuffle(order)
    n = len(order)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": [int(order[0])], "val": [], "test": []}
    if n == 2:
        return {"train": [int(order[0])], "val": [int(order[1])], "test": []}
    n_train = int(round(n * SPLIT_FRACS[0]))
    n_val = int(round(n * SPLIT_FRACS[1]))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n - n_train - n_test)
    train = [int(x) for x in order[:n_train]]
    val = [int(x) for x in order[n_train : n_train + n_val]]
    test = [int(x) for x in order[n_train + n_val :]]
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return {"train": train, "val": val, "test": test}


def _unique_patient_ids(patient_ids: list[str]) -> list[str]:
    return sorted({str(value).strip() for value in patient_ids if str(value).strip()})


def patient_split_sizes(
    n_patients: int,
    fractions: tuple[float, float, float] | None = None,
) -> dict[str, int]:
    """Whole-patient counts for a 70/15/15 target. Splits stay non-empty once n >= 3."""
    if n_patients <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if fractions is not None:
        counts = _largest_remainder(n_patients, fractions)
        # Largest remainder can produce 2/1/0 at three patients for 70/15/15.
        # Keep all splits populated when every target is positive and there
        # are at least three indivisible patient/hash groups.
        if n_patients >= 3 and all(value > 0 for value in fractions):
            for empty in (index for index, count in enumerate(counts) if count == 0):
                donors = [index for index, count in enumerate(counts) if count > 1]
                if not donors:
                    break
                donor = max(
                    donors,
                    key=lambda index: (counts[index] - n_patients * fractions[index], counts[index], -index),
                )
                counts[donor] -= 1
                counts[empty] += 1
        return {"train": counts[0], "val": counts[1], "test": counts[2]}
    if n_patients == 1:
        return {"train": 1, "val": 0, "test": 0}
    if n_patients == 2:
        return {"train": 1, "val": 1, "test": 0}
    fracs = SPLIT_FRACS
    n_train = min(max(1, int(round(n_patients * fracs[0]))), n_patients - 2)
    n_val = min(max(1, int(round(n_patients * fracs[1]))), n_patients - n_train - 1)
    return {"train": n_train, "val": n_val, "test": n_patients - n_train - n_val}


def split_fraction_limitations(n_patients: int) -> list[str]:
    """Explain when whole patients cannot land on exactly 70/15/15."""
    sizes = patient_split_sizes(n_patients)
    if n_patients < 3:
        return [
            f"Only {n_patients} distinct patient(s). Train, validation, and test cannot all be non-empty, "
            "so a 70/15/15 patient split is not reliable."
        ]
    notes = []
    targets = {"train": SPLIT_FRACS[0], "val": SPLIT_FRACS[1], "test": SPLIT_FRACS[2]}
    for name, target in targets.items():
        actual = sizes[name] / n_patients
        if abs(actual - target) > 0.005:
            notes.append(
                f"{name} receives {sizes[name]} of {n_patients} patients ({actual:.1%}), "
                f"not exactly {target:.0%}. Patient counts are whole numbers, so this is the closest "
                "allocation that still leaves every split non-empty."
            )
    return notes


def class_support_limitations(patient_class_counts: dict[str, dict[str, int]]) -> list[str]:
    """A class can occupy every split only when at least three patients carry it."""
    carriers = {grade: 0 for grade in range(5)}
    for counts in patient_class_counts.values():
        for grade in range(5):
            if int(counts.get(str(grade), 0) or 0) > 0:
                carriers[grade] += 1
    notes = []
    for grade, n_patients in carriers.items():
        if n_patients < 3:
            notes.append(
                f"ICDAS {grade} occurs in {n_patients} patient(s). "
                "Those patients cannot be copied into more than one split, so this class cannot be "
                "guaranteed in train, validation, and test."
            )
    return notes


def _grade_vector(counts: dict[str, int] | None) -> list[int]:
    vector = [0, 0, 0, 0, 0]
    for key, value in (counts or {}).items():
        grade = int(key)
        if 0 <= grade <= 4:
            vector[grade] += int(value)
    return vector


def _largest_remainder(total: int, fractions: tuple[float, float, float]) -> list[int]:
    raw = [total * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    leftover = total - sum(counts)
    ranked = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in ranked[:leftover]:
        counts[index] += 1
    return counts


def _stratified_patient_split(
    patient_ids: list[str],
    patient_class_counts: dict[str, dict[str, int]],
    seed: int,
    fractions: tuple[float, float, float] | None = None,
) -> dict[str, list[str]]:
    """Assign each patient once, filling 70/15/15 quotas while tracking class counts.

    Each ICDAS grade gets its own 70/15/15 crop target. A patient is sent to the
    split that still needs the grades on that patient's teeth and still has room.
    """
    fracs = fractions or SPLIT_FRACS
    sizes = patient_split_sizes(len(patient_ids), fractions)
    vectors = {pid: _grade_vector(patient_class_counts.get(pid)) for pid in patient_ids}
    totals = [0, 0, 0, 0, 0]
    for vector in vectors.values():
        for grade, count in enumerate(vector):
            totals[grade] += count
    needs = {
        name: [0, 0, 0, 0, 0] for name in ("train", "val", "test")
    }
    for grade, total in enumerate(totals):
        shares = _largest_remainder(total, fracs)
        for name, share in zip(("train", "val", "test"), shares):
            needs[name][grade] = share
    rng = np.random.default_rng(seed)
    order = list(patient_ids)
    rng.shuffle(order)

    def rarity(pid: str) -> tuple[int, int]:
        present = [grade for grade, count in enumerate(vectors[pid]) if count > 0]
        if not present:
            return (10**9, 0)
        return (min(totals[grade] for grade in present), -max(present))

    order.sort(key=rarity)
    quotas = dict(sizes)
    assigned: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    rank = {"train": 2, "val": 1, "test": 0}
    for pid in order:
        hist = vectors[pid]
        best_name = None
        best_key = None
        for name in ("train", "val", "test"):
            if quotas[name] <= 0:
                continue
            score = sum(max(0, needs[name][grade]) * hist[grade] for grade in range(5))
            key = (score, quotas[name], rank[name])
            if best_key is None or key > best_key:
                best_key = key
                best_name = name
        if best_name is None:
            raise RuntimeError("Patient split ran out of quota.")
        assigned[best_name].append(pid)
        quotas[best_name] -= 1
        for grade in range(5):
            needs[best_name][grade] -= hist[grade]
    return assigned


def split_patients(
    patient_ids: list[str],
    seed: int = SPLIT_SEED,
    patient_class_counts: dict[str, dict[str, int]] | None = None,
    fractions: tuple[float, float, float] | None = None,
) -> dict[str, list[str]]:
    """Deterministic 70/15/15 split over patient groups, never image or crop rows.

    Without class counts this is a seeded shuffle. With counts, patients who carry
    rarer ICDAS grades are placed first so each split stays as close as possible
    to the overall 0–4 distribution. A patient is still written to only one split.
    """
    ids = _unique_patient_ids(patient_ids)
    if patient_class_counts is not None:
        return _stratified_patient_split(ids, patient_class_counts, seed, fractions)
    rng = np.random.default_rng(seed)
    order = np.asarray(ids, dtype=object)
    rng.shuffle(order)
    sizes = patient_split_sizes(len(order), fractions)
    train = [str(v) for v in order[: sizes["train"]]]
    val = [str(v) for v in order[sizes["train"] : sizes["train"] + sizes["val"]]]
    test = [str(v) for v in order[sizes["train"] + sizes["val"] :]]
    return {"train": train, "val": val, "test": test}


def _split_report_markdown(stats: dict) -> str:
    lines = [
        f"# Patient-level split {stats.get('name', '')}",
        "",
        stats.get("split_method", ""),
        "",
        f"Target fractions: {stats.get('split_fractions_target')}",
        f"Distinct patients: {stats.get('total_patients', 0)}",
        f"Original images: {stats.get('total_source_images_included', 0)}",
        f"Included tooth crops: {stats.get('total_tooth_crops_included', 0)}",
        "",
        "| Split | Patients | Actual / target | Deviation (pp) | Original images | Tooth crops | Labeled | Verified | ICDAS 0 | 1 | 2 | 3 | 4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        summary = (stats.get("split_summary") or {}).get(split) or {}
        classes = summary.get("class_counts") or {}
        lines.append(
            "| {split} | {patients} | {actual:.2f}% / {target:.2f}% | {deviation:+.2f} | {images} | {crops} | {labeled} | {verified} | {c0} | {c1} | {c2} | {c3} | {c4} |".format(
                split=split,
                patients=summary.get("patient_count", 0),
                actual=summary.get("actual_patient_percent", 0),
                target=summary.get("target_patient_percent", 0),
                deviation=summary.get("deviation_percentage_points", 0),
                images=summary.get("image_count", 0),
                crops=summary.get("crop_count", 0),
                labeled=summary.get("labeled_crop_count", 0),
                verified=summary.get("verified_crop_count", 0),
                c0=classes.get("0", 0),
                c1=classes.get("1", 0),
                c2=classes.get("2", 0),
                c3=classes.get("3", 0),
                c4=classes.get("4", 0),
            )
        )
    limitations = stats.get("split_limitations") or []
    lines.extend(["", "## Limitations", ""])
    if limitations:
        lines.extend(f"- {note}" for note in limitations)
    else:
        lines.append("- Patient counts match 70/15/15, and every ICDAS class 0–4 is carried by at least three patients.")
    excluded = stats.get("excluded_crop_counts") or {}
    if excluded:
        lines.extend(["", "## Excluded records", ""])
        lines.extend(f"- {status}: {count}" for status, count in sorted(excluded.items()))
    warnings = stats.get("readiness_warnings") or []
    if warnings:
        lines.extend(["", "## Readiness warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def _class_counts(labs: list[TrainingLabel]) -> dict[str, int]:
    c = Counter(int(l.grade) for l in labs)
    return {str(i): int(c.get(i, 0)) for i in range(5)}


def _workflow_status(
    db: Session,
    *,
    labeling_completed: bool,
    dataset_available: bool,
    dataset_ready: bool,
    small_dataset: bool,
) -> dict:
    from .portal_runtime import is_valid_icdas_checkpoint, resolve_icdas_checkpoint

    deployed = resolve_icdas_checkpoint()
    trained = False
    for model in db.query(ModelVersion).filter(ModelVersion.kind == "icdas").all():
        if str(model.dataset_version or "").lower() == "historical":
            continue
        try:
            metrics = json.loads(model.metrics_json or "{}")
        except json.JSONDecodeError:
            metrics = {}
        checkpoint = _find_keras(Path(model.path))
        if (
            checkpoint
            and all(key in metrics for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix"))
            and is_valid_icdas_checkpoint(checkpoint)
        ):
            trained = True
            break
    trained = trained or bool(deployed)
    classifier_status = (
        "DEPLOYED"
        if deployed
        else "TRAINED_NOT_DEPLOYED"
        if trained
        else "NOT_TRAINED / NOT_DEPLOYED"
    )
    return {
        "labeling_enabled": True,
        "dataset_available": dataset_available,
        "labeling_completed": labeling_completed,
        "dataset_ready_for_training": dataset_ready,
        "model_trained": trained,
        "model_deployed": bool(deployed),
        "small_dataset_warning": (
            "This dataset has fewer than 20 patients. Training can run, but validation and test scores will be unstable."
            if small_dataset
            else None
        ),
        "icdas_classifier": classifier_status,
    }


def dataset_inventory(db: Session) -> dict:
    refresh_crop_identities(db)
    all_imgs = db.query(TrainingImage).all()
    active_imgs = [i for i in all_imgs if i.is_active is not False]
    dup_counts = Counter(i.duplicate_status or STATUS_UNIQUE for i in active_imgs)
    unique_imgs = [
        i
        for i in active_imgs
        if (i.duplicate_status or STATUS_UNIQUE) == STATUS_UNIQUE
        and (i.processing_status or UPLOAD_COMPLETED) == UPLOAD_COMPLETED
    ]
    patient_ids = sorted({str(i.patient_id) for i in unique_imgs if i.patient_id})
    unassigned_patient_ids = sum(1 for i in unique_imgs if not (i.patient_id or "").strip())
    unreviewed_images = sum(1 for i in unique_imgs if not i.boxes_verified)
    unverified_crops = sum(
        1
        for image in unique_imgs
        for label in image.labels
        if label.active is not False and not label.box_verified
    )
    detector_failures = sum(1 for i in unique_imgs if int(i.n_crops or 0) == 0)
    eligible = list(_eligible_label_query(db).all())
    class_counts = _class_counts(eligible)
    labeled_n = sum(class_counts.values())

    detected = 0
    verified = 0
    skipped_n = 0
    uncertain_n = 0
    ungradable_n = 0
    unlabeled_n = 0
    photos_completed = 0
    photos_box_verified = 0
    photos_need_review = 0
    labeled_raw = 0
    crop_status_counts: Counter[str] = Counter()
    for im in unique_imgs:
        active = [l for l in im.labels if l.active is not False]
        detected += len(active)
        if im.boxes_verified:
            photos_box_verified += 1
            vlabs = [l for l in active if l.box_verified]
            verified += len(vlabs)
            resolved = 0
            for lab in vlabs:
                crop_status_counts[lab.crop_duplicate_status or CROP_UNIQUE] += 1
                annotation_status = _effective_annotation_status(lab)
                if annotation_status == "UNCERTAIN" or lab.skipped:
                    skipped_n += 1
                    uncertain_n += 1
                elif annotation_status == "UNGRADABLE":
                    ungradable_n += 1
                elif lab.grade is None:
                    unlabeled_n += 1
                else:
                    labeled_raw += 1
                if annotation_status in {"LABELED", "UNCERTAIN", "UNGRADABLE"}:
                    resolved += 1
            if vlabs and resolved == len(vlabs):
                photos_completed += 1
        else:
            photos_need_review += 1
            for lab in active:
                crop_status_counts[lab.crop_duplicate_status or CROP_UNIQUE] += 1

    conflict_n = int(crop_status_counts.get(CROP_CONFLICT, 0))
    exact_crop_n = int(crop_status_counts.get(CROP_DUP_EXACT, 0))
    possible_crop_n = int(crop_status_counts.get(CROP_POSSIBLE, 0))
    unique_crop_n = int(crop_status_counts.get(CROP_UNIQUE, 0))
    conflict_msg = CONFLICT_MESSAGE if conflict_n else None

    missing = missing_icdas_classes(class_counts)
    imb_text = imbalance_warning_text(class_counts)
    versions = db.query(DatasetVersion).order_by(DatasetVersion.id.desc()).all()
    latest = versions[0] if versions else None
    from .admin_portal_ops import progress_for

    verification_cursor = progress_for(db, "verification")
    labeling_cursor = progress_for(db, "initial")
    base_ready = dataset_build_ready(labeled_n, class_counts)
    identity_ready = unassigned_patient_ids == 0
    annotation_ready = unlabeled_n == 0
    review_ready = unreviewed_images == 0
    min_split_patients_ready = len(patient_ids) >= 3
    ready = (
        base_ready
        and identity_ready
        and annotation_ready
        and review_ready
        and unverified_crops == 0
        and min_split_patients_ready
    )
    min_msg = dataset_min_crops_message(labeled_n)
    if not ready:
        status = "NOT READY"
    elif not versions:
        status = "READY TO BUILD"
    else:
        status = "READY"

    unique_n = len(unique_imgs)
    photos_remaining_complete = max(0, photos_box_verified - photos_completed)
    wf = {
        "upload": {
            "done": unique_n > 0,
            "status": "done" if unique_n else "todo",
            "detail": f"{len(all_imgs)} uploaded · {unique_n} unique",
        },
        "review": {
            "done": unique_n > 0 and photos_need_review == 0,
            "status": "done" if unique_n and photos_need_review == 0 else "todo",
            "detail": f"{photos_need_review} unique photos still need box review",
        },
        "label": {
            "done": verified > 0 and unlabeled_n == 0,
            "status": "done" if verified and unlabeled_n == 0 else "todo",
            "detail": f"{labeled_n} labeled · {unlabeled_n} unlabeled · {skipped_n} skipped",
        },
        "dataset": {
            "done": bool(latest) and ready,
            "status": "done" if latest and ready else "todo",
            "detail": status,
        },
        "train": {
            "done": False,
            "status": "disabled" if not get_settings().allow_icdas_train else "todo",
            "detail": "Training is a separate step and is off by default.",
        },
    }
    latest_stats = json.loads(latest.statistics_json) if latest and latest.statistics_json else None
    return {
        "photos": {
            "uploaded": len(all_imgs),
            "unique": unique_n,
            "exact_duplicates": int(dup_counts.get(STATUS_EXACT, 0)),
            "likely_duplicates": int(dup_counts.get(STATUS_LIKELY, 0)),
            "invalid": int(dup_counts.get(STATUS_INVALID, 0)),
            "completed": photos_completed,
            "remaining": photos_remaining_complete,
            "box_verified": photos_box_verified,
            "need_box_review": photos_need_review,
            "detector_failures": detector_failures,
            "unassigned_patient_ids": unassigned_patient_ids,
            "patients": len(patient_ids),
            "target_patients": TARGET_PATIENTS,
            "min_patients_for_training": get_settings().min_training_patients,
            "patient_count_warning": len(patient_ids) < TARGET_PATIENTS,
            "unreviewed_images": unreviewed_images,
            "unverified_crops": unverified_crops,
        },
        "verification_progress": {
            "reviewed_photos": photos_box_verified,
            "total_photos": unique_n,
            "remaining_photos": unreviewed_images,
            "percent": round(100.0 * photos_box_verified / unique_n, 1) if unique_n else 0.0,
            "last_training_image_id": verification_cursor.image_id if verification_cursor else None,
        },
        "labeling_progress": {
            "labeled_crops": labeled_n,
            "unlabeled_crops": unlabeled_n,
            "last_label_id": labeling_cursor.label_id if labeling_cursor else None,
        },
        "crops": {
            "detected": detected,
            "verified": verified,
            "labeled": labeled_raw,
            "unlabeled": unlabeled_n,
            "skipped": skipped_n,
            "uncertain": uncertain_n,
            "ungradable": ungradable_n,
            "unique": unique_crop_n,
            "exact_duplicates": exact_crop_n,
            "possible_duplicates": possible_crop_n,
            "conflicts": conflict_n,
            "eligible": labeled_n,
        },
        "icdas_labels": class_counts,
        "images_uploaded": len(all_imgs),
        "unique_images": unique_n,
        "exact_duplicates": int(dup_counts.get(STATUS_EXACT, 0)),
        "likely_duplicates": int(dup_counts.get(STATUS_LIKELY, 0)),
        "invalid_images": int(dup_counts.get(STATUS_INVALID, 0)),
        "images_box_verified": photos_box_verified,
        "images_completed": photos_completed,
        "labeled_verified_crops": labeled_n,
        "eligible_training_crops": labeled_n,
        "crop_conflict_message": conflict_msg,
        "crop_duplicate_status_counts": dict(crop_status_counts),
        "class_counts": class_counts,
        "missing_classes": missing,
        "missing_classes_message": missing_classes_message(missing),
        "imbalance_warning": bool(imb_text),
        "imbalance_message": imb_text,
        "target_crops": 3000,
        "min_patients_for_training": get_settings().min_training_patients,
        "min_dataset_crops": MIN_DATASET_CROPS,
        "min_crops_message": min_msg,
        "classes_ready": dataset_classes_ready(class_counts),
        "crops_ready": labeled_n >= MIN_DATASET_CROPS,
        "dataset_ready": ready,
        "readiness_checks": {
            "patient_ids_assigned": unassigned_patient_ids == 0,
            "at_least_three_patients_for_nonempty_splits": min_split_patients_ready,
            "detections_reviewed": review_ready and unverified_crops == 0,
            "all_crops_resolved": annotation_ready,
            "patient_count_target": len(patient_ids) >= TARGET_PATIENTS,
        },
        "patient_count_warning": (
            f"{len(patient_ids)} unique patients are assigned; the target is about {TARGET_PATIENTS}."
            if len(patient_ids) < TARGET_PATIENTS
            else None
        ),
        "identity_warning": (
            f"{unassigned_patient_ids} UNIQUE photo(s) need a pseudonymous patient ID."
            if unassigned_patient_ids
            else None
        ),
        "review_warning": (
            (
                f"{unreviewed_images} UNIQUE photo(s) still need detector-box review."
                if unreviewed_images
                else f"{unverified_crops} active crop(s) have not been box-verified."
            )
            if unreviewed_images or unverified_crops
            else None
        ),
        "patient_split_warning": (
            f"At least 3 distinct patients are needed to build train/validation/test splits; {len(patient_ids)} assigned."
            if not min_split_patients_ready
            else None
        ),
        "annotation_warning": (
            f"{unlabeled_n} verified crop(s) still need a label, uncertainty, or ungradable status."
            if unlabeled_n
            else None
        ),
        "dataset_v1_ready": ready and any(v.name == "v1" for v in versions),
        "icdas_train_enabled": bool(get_settings().allow_icdas_train),
        "workflow_status": _workflow_status(
            db,
            labeling_completed=verified > 0 and unlabeled_n == 0 and unverified_crops == 0,
            dataset_available=unique_n > 0,
            dataset_ready=bool(latest) and ready,
            small_dataset=0 < len(patient_ids) < 20,
        ),
        "status": status,
        "workflow": wf,
        "latest_dataset": (
            {
                "name": latest.name,
                "n_train": latest.n_train,
                "n_valid": latest.n_valid,
                "n_test": latest.n_test,
                "statistics": latest_stats,
            }
            if latest
            else None
        ),
        "duplicate_status_counts": dict(dup_counts),
        "versions": [
            {
                "id": v.id,
                "name": v.name,
                "path": v.path,
                "status": v.status,
                "n_train": v.n_train,
                "n_valid": v.n_valid,
                "n_test": v.n_test,
                "statistics": json.loads(v.statistics_json) if v.statistics_json else None,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in versions
        ],
    }


def validate_built_dataset(root: Path) -> dict:
    issues: list[str] = []
    counts = {split: {str(g): 0 for g in range(5)} for split in ("train", "val", "test")}
    patient_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    crop_splits: dict[str, set[str]] = defaultdict(set)
    image_hash_splits: dict[str, set[str]] = defaultdict(set)
    crop_count = Counter()
    image_count: dict[str, set[str]] = {split: set() for split in counts}
    patient_count: dict[str, set[str]] = {split: set() for split in counts}
    man = root / "manifest.csv"
    if not man.exists():
        raise ValueError(f"Missing manifest.csv in {root}")
    with man.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status") or INCLUDED
            split = row.get("split") or ""
            if split not in counts:
                continue
            patient_id = (row.get("patient_id") or "").strip()
            image_id = (row.get("image_id") or "").strip()
            if not patient_id:
                if status == INCLUDED:
                    issues.append(f"Included split row has no patient_id (image {image_id or row.get('source_image_id')}).")
            else:
                patient_splits[patient_id].add(split)
                patient_count[split].add(patient_id)
            if not image_id:
                issues.append(f"Split row has no image_id (source image {row.get('source_image_id')}).")
            else:
                image_splits[image_id].add(split)
                image_count[split].add(image_id)
            sid = row.get("source_image_id") or ""
            if sid:
                source_splits[sid].add(split)
            image_hash = (row.get("content_hash") or "").strip()
            if image_hash:
                image_hash_splits[image_hash].add(split)
            if status == INCLUDED:
                destination = (row.get("dest") or "").strip()
                if not destination or not (root / destination).is_file():
                    issues.append(f"Included crop file is missing for {row.get('crop_id')} in {split}.")
                grade = str(row.get("grade") or "")
                if grade not in counts[split]:
                    issues.append(f"Included row has an invalid ICDAS grade for {split}.")
                    continue
                if (row.get("annotation_status") or "LABELED").upper() != "LABELED":
                    issues.append(f"Included crop {row.get('crop_id')} lacks a LABELED annotation status.")
                counts[split][grade] += 1
                crop_count[split] += 1
                crop_id = (row.get("crop_id") or "").strip()
                if not crop_id:
                    issues.append(f"Included row has no crop_id (label {row.get('label_id')}).")
                else:
                    crop_splits[crop_id].add(split)
                crop_hash = (row.get("crop_hash") or "").strip()
                if crop_hash:
                    crop_splits[f"hash:{crop_hash}"].add(split)
                    # Exact same crop must appear at most once, even inside one split.
                    if len(crop_splits[f"hash:{crop_hash}"]) == 1:
                        crop_count[f"seen_hash:{crop_hash}"] += 1
                        if crop_count[f"seen_hash:{crop_hash}"] > 1:
                            issues.append(f"Duplicate crop hash {crop_hash[:12]} is included more than once.")
    split_folders = {"train": "train", "val": "validation", "test": "test"}
    for split, folder in split_folders.items():
        labels_path = root / folder / "labels.csv"
        if not labels_path.is_file():
            continue  # Legacy class-directory datasets remain readable.
        with labels_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                grade = str(row.get("icdas_score") or "")
                if grade not in counts[split]:
                    issues.append(f"{folder}/labels.csv contains an invalid ICDAS label {grade!r}.")
                    continue
                if not (row.get("patient_id") or "").strip():
                    issues.append(f"{folder}/labels.csv contains a crop without patient_id.")
                for field in ("source_photo_id", "crop_id"):
                    if not (row.get(field) or "").strip():
                        issues.append(f"{folder}/labels.csv contains a crop without {field}.")
                filename = (row.get("filename") or "").strip()
                if not filename or not (root / folder / filename).is_file():
                    issues.append(f"{folder}/labels.csv references a missing tooth crop: {filename!r}.")
    for patient_id, splits in patient_splits.items():
        if len(splits) > 1:
            issues.append(f"Patient {patient_id} leaked across splits {sorted(splits)}")
    for image_id, splits in image_splits.items():
        if len(splits) > 1:
            issues.append(f"Original image {image_id} leaked across splits {sorted(splits)}")
    for sid, splits in source_splits.items():
        if len(splits) > 1:
            issues.append(f"Source image {sid} leaked across splits {sorted(splits)}")
    for digest, splits in image_hash_splits.items():
        if len(splits) > 1:
            issues.append(f"Duplicate original image hash {digest[:12]} appears across splits {sorted(splits)}")
    for crop_id, splits in crop_splits.items():
        if len(splits) > 1:
            issues.append(f"Tooth crop {crop_id} leaked across splits {sorted(splits)}")
    for split in ("train", "val", "test"):
        if not image_count[split] or not patient_count[split] or sum(counts[split].values()) == 0:
            issues.append(f"Split {split} is empty")
    missing_train = [g for g in range(5) if counts["train"][str(g)] == 0]
    missing_val = [g for g in range(5) if counts["val"][str(g)] == 0]
    missing_test = [g for g in range(5) if counts["test"][str(g)] == 0]
    return {
        "ok": not issues and not missing_train and not missing_val and not missing_test,
        "issues": issues,
        "missing_classes_in_train": missing_train,
        "missing_classes_in_validation": missing_val,
        "missing_classes_in_test": missing_test,
        "split_class_counts": counts,
        "split_patient_counts": {split: len(values) for split, values in patient_count.items()},
        "split_image_counts": {split: len(values) for split, values in image_count.items()},
        "split_crop_counts": {split: int(crop_count[split]) for split in counts},
        "patient_leakage_count": sum(1 for values in patient_splits.values() if len(values) > 1),
        "image_leakage_count": sum(1 for values in image_splits.values() if len(values) > 1),
    }


def _split_source_images(db: Session) -> list[TrainingImage]:
    return [
        image
        for image in db.query(TrainingImage).order_by(TrainingImage.id.asc()).all()
        if image.is_active is not False
        and not image.exclude_from_dataset
    ]


def _patient_identity_provenance(db: Session) -> dict[str, dict]:
    """Return the latest saved review evidence for each assigned patient group."""
    provenance: dict[str, dict] = {}
    events = (
        db.query(TrainingAuditEvent)
        .filter(TrainingAuditEvent.event_type == "patient_photo_group_review")
        .order_by(TrainingAuditEvent.id.asc())
        .all()
    )
    for event in events:
        try:
            details = json.loads(event.details_json or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        patient_id = str(details.get("assigned_patient_id") or "").strip()
        if not patient_id:
            continue
        method = str(details.get("method") or "").strip()
        confidence = str(details.get("confidence") or "").strip()
        decision = str(details.get("decision") or "").strip().casefold()
        if confidence == "approximate_not_verified":
            status = "APPROXIMATE_NOT_VERIFIED"
        elif method == "exact_sha256_and_verified_source_reference" and decision == "confirm":
            status = "SOURCE_REFERENCE_CONFIRMED"
        elif decision == "confirm":
            status = "HUMAN_CONFIRMED"
        else:
            status = "UNVERIFIED"
        provenance[patient_id] = {
            "status": status,
            "method": method or "unspecified_review",
            "evidence": str(details.get("evidence") or "").strip(),
            "decision": decision or "unspecified",
        }
    return provenance


def _patient_record_link_conflicts(images: list[TrainingImage]) -> list[int]:
    """Catch disagreement between the FK source of truth and exported ID field."""
    conflicts = []
    for image in images:
        if image.patient_record_id is None:
            continue
        patient = image.patient_record
        if patient is None or not image.patient_id or image.patient_id != patient.public_id:
            conflicts.append(int(image.id))
    return conflicts


def patient_assignment_inventory(db: Session, image_to_split: dict[int, str] | None = None) -> dict:
    """Build patient-wise assignment rows from stored photos, boxes, labels, and verification."""
    images = _split_source_images(db)
    grouped: dict[str, list[TrainingImage]] = defaultdict(list)
    for image in images:
        patient_id = (
            image.patient_record.public_id
            if image.patient_record is not None
            else (image.patient_id or "").strip()
        )
        key = patient_id if patient_id else f"MISSING_ID_PHOTO_{image.id}"
        grouped[key].append(image)

    rows = []
    for key, patient_images in sorted(grouped.items(), key=lambda item: item[0].casefold()):
        is_missing = key.startswith("MISSING_ID_PHOTO_")
        crop_count = labeled_count = verified_count = unverified_count = unlabeled_count = 0
        grades = Counter()
        duplicate_count = 0
        current_splits: set[str] = set()
        planned_splits: set[str] = set()
        image_ids = []
        for image in patient_images:
            image_ids.append(int(image.id))
            if image.split_name in {"train", "val", "test"}:
                current_splits.add(image.split_name)
            if image_to_split and image_to_split.get(int(image.id)) in {"train", "val", "test"}:
                planned_splits.add(image_to_split[int(image.id)])
            if (image.duplicate_status or STATUS_UNIQUE) != STATUS_UNIQUE:
                duplicate_count += 1
            for label in image.labels:
                if label.active is False:
                    continue
                crop_count += 1
                is_verified = bool(image.boxes_verified and label.box_verified)
                verified_count += int(is_verified)
                status = _effective_annotation_status(label)
                if status == "LABELED" and label.grade is not None:
                    labeled_count += 1
                    grades[int(label.grade)] += 1
                else:
                    unlabeled_count += 1
                unverified_count += int(not is_verified)

        current_assignment = next(iter(current_splits)) if len(current_splits) == 1 else "UNASSIGNED"
        planned_assignment = next(iter(planned_splits)) if len(planned_splits) == 1 else "UNASSIGNED"
        if is_missing:
            assignment_status = "MISSING PATIENT ID"
        elif len(current_splits) > 1:
            assignment_status = "CURRENT ASSIGNMENT CONFLICT"
        elif current_assignment == "UNASSIGNED":
            assignment_status = "UNASSIGNED"
        else:
            assignment_status = "ASSIGNED"
        if duplicate_count:
            assignment_status += f" · {duplicate_count} duplicate photo(s)"
        if unverified_count:
            assignment_status += f" · {unverified_count} unverified crop(s)"
        if unlabeled_count:
            assignment_status += f" · {unlabeled_count} unresolved crop(s)"
        rows.append(
            {
                "patient_key": key,
                "patient_id": "" if is_missing else key,
                "photo_count": len(patient_images),
                "training_image_ids": image_ids,
                "crop_count": crop_count,
                "labeled_crop_count": labeled_count,
                "verified_crop_count": verified_count,
                "unverified_crop_count": unverified_count,
                "unlabeled_crop_count": unlabeled_count,
                "duplicate_photo_count": duplicate_count,
                "grade_counts": {str(grade): int(grades.get(grade, 0)) for grade in range(5)},
                "current_assignment": current_assignment,
                "planned_assignment": planned_assignment,
                "assignment_status": assignment_status,
                "missing_patient_id": is_missing,
            }
        )
    return {
        "patient_count": sum(1 for row in rows if not row["missing_patient_id"]),
        "missing_patient_photo_count": sum(row["photo_count"] for row in rows if row["missing_patient_id"]),
        "patients": rows,
    }


def _patient_class_counts(images: list[TrainingImage], labels: list[TrainingLabel]) -> dict:
    allowed = {int(image.id) for image in images}
    counts: dict[str, Counter] = defaultdict(Counter)
    for label in labels:
        if label.image_id in allowed and label.grade is not None and label.image and label.image.patient_id:
            counts[str(label.image.patient_id)][int(label.grade)] += 1
    return {
        patient_id: {str(grade): int(counts[patient_id].get(grade, 0)) for grade in range(5)}
        for patient_id in sorted(counts)
    }


def _split_input_fingerprint(
    images: list[TrainingImage],
    plan: dict,
    image_to_split: dict,
    labels: list[TrainingLabel] | None = None,
) -> str:
    from .admin_portal_ops import content_digest_for_image

    rows = [
        {
            "id": int(image.id),
            "image_id": image.image_id,
            "patient_id": image.patient_id,
            "content_hash": content_digest_for_image(image),
            "duplicate_status": image.duplicate_status,
            "boxes_verified": bool(image.boxes_verified),
            "is_active": bool(image.is_active is not False),
            "exclude_from_dataset": bool(image.exclude_from_dataset),
            "split": image_to_split.get(int(image.id)),
        }
        for image in images
    ]
    label_rows = [
        {
            "id": int(label.id),
            "image_id": int(label.image_id),
            "crop_id": label.crop_id,
            "grade": label.grade,
            "annotation_status": label.annotation_status,
            "skipped": bool(label.skipped),
            "box_verified": bool(label.box_verified),
            "active": bool(label.active is not False),
            "crop_hash": label.crop_hash,
            "crop_duplicate_status": label.crop_duplicate_status,
            "crop_exists": bool(label.crop_path and Path(label.crop_path).is_file()),
        }
        for label in sorted(labels or [], key=lambda item: int(item.id))
    ]
    payload = json.dumps(
        {"plan": plan, "images": rows, "labels": label_rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preview_dataset_split(db: Session, plan: dict | None = None) -> dict:
    """Resolve a patient-wise split and its readiness warnings without writing a version."""
    from .admin_portal_ops import content_digest_for_image, resolve_image_splits, summarize_split

    refresh_crop_identities(db)
    db.flush()
    normalized_plan = dict(plan or {})
    normalized_plan.setdefault("strategy", "percentage")
    normalized_plan.setdefault("seed", SPLIT_SEED)
    images = _split_source_images(db)
    eligible_labels = list(_eligible_label_query(db).all())
    all_labels = db.query(TrainingLabel).all()
    class_counts = _patient_class_counts(images, eligible_labels)
    resolved = resolve_image_splits(images, class_counts, normalized_plan)
    summary = summarize_split(images, all_labels, resolved)
    image_to_split = resolved["image_to_split"]
    fingerprint = _split_input_fingerprint(images, normalized_plan, image_to_split, all_labels)
    patient_rows = patient_assignment_inventory(db, image_to_split)["patients"]
    identity_provenance = _patient_identity_provenance(db)
    approximate_patient_ids = sorted(
        patient_id
        for patient_id, details in identity_provenance.items()
        if details["status"] == "APPROXIMATE_NOT_VERIFIED"
        and patient_id in resolved["patient_to_split"]
    )

    assignments = []
    for image in images:
        assignments.append(
            {
                "training_image_id": int(image.id),
                "image_id": image.image_id,
                "filename": image.filename,
                "patient_id": image.patient_id,
                "sha256": content_digest_for_image(image),
                "split": image_to_split.get(int(image.id)),
                "current_split": image.split_name or "UNASSIGNED",
                "split_locked": bool(image.split_locked),
                "crop_count": sum(1 for label in image.labels if label.active is not False),
                "labeled_crops": sum(1 for label in image.labels if label.active is not False and label.grade is not None),
                "verified_crops": sum(
                    1
                    for label in image.labels
                    if label.active is not False and image.boxes_verified and label.box_verified
                ),
            }
        )

    inventory = dataset_inventory(db)
    blockers = []
    warnings = list(resolved.get("warnings") or [])
    if approximate_patient_ids:
        warnings.append(
            f"{len(approximate_patient_ids)} patient group(s) use a documented approximation and are not identity-verified. "
            "Leakage checks cover saved patient IDs and image hashes; real-person leakage remains possible until those groups are confirmed."
        )
    if inventory["photos"].get("unreviewed_images"):
        warnings.append(f"{inventory['photos']['unreviewed_images']} photo(s) still need tooth-box review; unresolved detections are excluded.")
    if inventory["photos"].get("unverified_crops"):
        warnings.append(f"{inventory['photos']['unverified_crops']} unverified tooth crop(s) are not box-verified and will be excluded.")
    if inventory["crops"].get("unlabeled"):
        warnings.append(f"{inventory['crops']['unlabeled']} verified crop(s) have no confirmed ICDAS grade and will be excluded.")
    duplicate_photo_count = sum(
        int(inventory["photos"].get(key, 0) or 0)
        for key in ("exact_duplicates", "likely_duplicates", "invalid")
    )
    if duplicate_photo_count:
        warnings.append(f"{duplicate_photo_count} duplicate or invalid photo record(s) are excluded from classifier images and retained in the manifest.")
    patient_conflicts = [
        row["patient_id"] or row["patient_key"]
        for row in patient_rows
        if row["assignment_status"].startswith("CURRENT ASSIGNMENT CONFLICT")
    ]
    if patient_conflicts:
        warnings.append(f"{len(patient_conflicts)} patient(s) have conflicting saved split values; this preview consolidates each patient before export.")
    min_message = dataset_min_crops_message(inventory["crops"].get("eligible", 0))
    if min_message:
        blockers.append(min_message)
    patient_link_conflicts = _patient_record_link_conflicts(images)
    if patient_link_conflicts:
        blockers.append(
            "Patient table links disagree with the stored patient ID on photo(s): "
            + ", ".join(str(value) for value in patient_link_conflicts[:10])
            + ". Correct the patient grouping before exporting."
        )
    known_patients = {str(image.patient_id).strip() for image in images if (image.patient_id or "").strip()}
    if len(known_patients) < 3:
        blockers.append("At least three real patient IDs are needed to populate train, validation, and test.")
    if any(not summary["split_summary"][split]["patient_count"] for split in ("train", "val", "test")):
        blockers.append("Train, validation, and test each need at least one assigned patient.")
    if any(not summary["split_summary"][split]["image_count"] for split in ("train", "val", "test")):
        blockers.append("Train, validation, and test each need at least one assigned source photo.")
    unassigned_known_patients = [
        row["patient_id"] for row in patient_rows
        if not row["missing_patient_id"] and row["planned_assignment"] == "UNASSIGNED"
    ]
    if unassigned_known_patients:
        warnings.append(f"{len(unassigned_known_patients)} patient(s) remain Unassigned and will be excluded from this version.")
    if resolved["unassigned_image_ids"]:
        unassigned_patient_ids = set(resolved["missing_patient_ids"])
        unresolved_known = [image_id for image_id in resolved["unassigned_image_ids"] if image_id not in unassigned_patient_ids]
        if unresolved_known:
            warnings.append(f"{len(unresolved_known)} photo(s) are unassigned and will not be exported.")
    blockers.extend(resolved["issues"])
    deviations = []
    for split, details in summary["split_summary"].items():
        deviation = details["deviation_percentage_points"]
        if abs(deviation) >= 0.01:
            deviations.append(
                f"{split.title()} is {details['actual_patient_percent']:.2f}% vs {details['target_patient_percent']:.2f}% target ({deviation:+.2f} percentage points)."
            )
    warnings.extend(deviations)
    requires_acknowledgment = bool(warnings)
    acknowledged = bool(normalized_plan.get("allow_exclusions"))
    if requires_acknowledgment and not acknowledged:
        blockers.append("Review the readiness warnings and explicitly acknowledge that excluded records will not enter training.")

    return {
        "plan": normalized_plan,
        "fingerprint": fingerprint,
        "strategy": resolved["strategy"],
        "seed": resolved["seed"],
        "assignments": assignments,
        "patient_assignments": patient_rows,
        "split_summary": summary["split_summary"],
        "issues": resolved["issues"],
        "warnings": list(dict.fromkeys(warnings)),
        "deviations": deviations,
        "unassigned_image_ids": resolved["unassigned_image_ids"],
        "manual_image_ids": resolved["manual_image_ids"],
        "missing_patient_ids": resolved["missing_patient_ids"],
        "patient_conflicts": patient_conflicts,
        "patient_record_link_conflicts": patient_link_conflicts,
        "unassigned_patients": unassigned_known_patients,
        "duplicate_photo_count": duplicate_photo_count,
        "unlabeled_crop_count": int(inventory["crops"].get("unlabeled", 0) or 0),
        "unverified_crop_count": int(inventory["photos"].get("unverified_crops", 0) or 0),
        "unreviewed_photo_count": int(inventory["photos"].get("unreviewed_images", 0) or 0),
        "approximate_patient_group_count": len(approximate_patient_ids),
        "requires_exclusion_acknowledgment": requires_acknowledgment,
        "exclusion_acknowledged": acknowledged,
        "build_blockers": list(dict.fromkeys(blockers)),
        "can_build": not blockers and summary["ok"],
        "note": "Only unique, box-verified ICDAS 0–4 crops are exported. Unassigned, missing-ID, duplicate, unverified, and unlabeled records stay out of training and remain documented in the manifest.",
    }


def bulk_assign_training_images(db: Session, selected_image_ids: list[int], split_name: str) -> dict:
    """Persist a manual bulk split lock for selected patient/hash groups."""
    from .admin_portal_ops import resolve_image_splits

    split_name = str(split_name or "").lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError("Split must be train, val, or test.")
    selected = sorted({int(value) for value in (selected_image_ids or [])})
    if not selected:
        raise ValueError("Select at least one photo to assign.")
    images = _split_source_images(db)
    by_id = {int(image.id): image for image in images}
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise ValueError(f"Selected photo ID(s) are not active UNIQUE photos: {missing[:10]}.")
    labels = list(_eligible_label_query(db).all())
    plan = {
        "strategy": "manual",
        "selected_image_ids": selected,
        "assignments": {str(image_id): split_name for image_id in selected},
        "seed": SPLIT_SEED,
    }
    resolved = resolve_image_splits(images, _patient_class_counts(images, labels), plan)
    selected_groups_missing_identity = set(selected) & set(resolved["missing_patient_ids"])
    if selected_groups_missing_identity:
        raise ValueError("Assign real pseudonymous patient IDs before locking these photo splits.")
    relevant_issues = [
        issue
        for issue in resolved["issues"]
        if not issue.startswith("Manual split leaves")
        and not issue.startswith(f"{len(resolved['missing_patient_ids'])} photo(s) have no patient ID")
    ]
    if relevant_issues:
        raise ValueError("Cannot lock this assignment: " + "; ".join(relevant_issues))
    assigned_ids = resolved["manual_image_ids"]
    if not assigned_ids or any(resolved["image_to_split"].get(image_id) != split_name for image_id in assigned_ids):
        raise ValueError("The selected photos do not form a safe patient/hash group assignment.")
    for image_id in assigned_ids:
        image = by_id[image_id]
        image.split_name = split_name
        image.split_locked = True
    _audit(
        db,
        "split_assignment_locked",
        "training_dataset",
        "manual",
        {"split": split_name, "training_image_ids": assigned_ids},
    )
    db.commit()
    return {"assigned_count": len(assigned_ids), "split": split_name, "locked_image_ids": assigned_ids}


def build_dataset_version(
    db: Session,
    split_plan: dict | None = None,
    expected_fingerprint: str | None = None,
) -> dict:
    """Create the next versioned split after an unchanged assignment preview."""
    from .admin_portal_ops import content_digest_for_image, resolve_image_splits

    plan = dict(split_plan or {})
    plan.setdefault("strategy", "percentage")
    plan.setdefault("seed", SPLIT_SEED)
    preview = preview_dataset_split(db, plan)
    identity_provenance = _patient_identity_provenance(db)
    if expected_fingerprint and expected_fingerprint != preview["fingerprint"]:
        raise ValueError("The dataset changed after split preview. Refresh the preview before confirming.")
    if preview["build_blockers"]:
        raise ValueError("Cannot build dataset: " + "; ".join(preview["build_blockers"][:8]))
    ensure_dirs()
    inv = dataset_inventory(db)
    source_images = _split_source_images(db)
    labeled = list(_eligible_label_query(db).all())
    for lab in labeled:
        validate_icdas_grade(lab.grade)
    patient_class_counts = _patient_class_counts(source_images, labeled)
    resolved = resolve_image_splits(source_images, patient_class_counts, plan)
    splits = {
        split: sorted(
            patient_id
            for patient_id, assigned in resolved["patient_to_split"].items()
            if assigned == split
        )
        for split in ("train", "val", "test")
    }
    patient_to_split = resolved["patient_to_split"]
    image_to_split = resolved["image_to_split"]
    if resolved["issues"]:
        raise ValueError("Split preview is no longer safe: " + "; ".join(resolved["issues"][:8]))
    if len(patient_to_split) < 3:
        raise ValueError("At least 3 distinct patient IDs are required to create non-empty train, validation, and test splits.")

    n_labeled = len(labeled)
    min_msg = dataset_min_crops_message(n_labeled)
    if min_msg:
        raise ValueError(min_msg)
    label_fingerprint = "|".join(
        f"{lab.id}:{lab.image_id}:{lab.image.patient_id}:{lab.grade}:{lab.annotation_status}:{lab.crop_hash or ''}"
        for lab in sorted(labeled, key=lambda x: int(x.id))
    )
    fingerprint = hashlib.sha256(
        f"{label_fingerprint}|split:{preview['fingerprint']}".encode("utf-8")
    ).hexdigest()
    latest_existing = db.query(DatasetVersion).order_by(DatasetVersion.id.desc()).first()
    if latest_existing and latest_existing.statistics_json:
        prev = json.loads(latest_existing.statistics_json)
        if prev.get("eligible_fingerprint") == fingerprint:
            raise ValueError(
                f"No new verified labeled crops since {latest_existing.name}; "
                "not creating a duplicate version."
            )
    name, version_number = next_dataset_version_name(db)
    root = ICDAS_DATASETS / name
    if root.exists():
        raise ValueError(f"Dataset folder {root} already exists and will not be overwritten.")
    root.mkdir(parents=True, exist_ok=False)

    manual_image_ids = set(resolved["manual_image_ids"])
    for image in source_images:
        image.dataset_version = name
        image.split_name = image_to_split.get(int(image.id), "UNASSIGNED")
        image.split_locked = int(image.id) in manual_image_ids
        if not image.content_hash:
            image.content_hash = content_digest_for_image(image) or None

    split_folders = {"train": "train", "val": "validation", "test": "test"}
    copied: dict[str, list[TrainingLabel]] = {"train": [], "val": [], "test": []}
    split_images: dict[str, set[int]] = {split: set() for split in copied}
    split_labels: dict[str, list[dict]] = {split: [] for split in copied}
    manifest_rows: list[dict] = []
    seen_crop_hash: set[str] = set()
    assigned_ids = set(image_to_split)
    for image in source_images:
        split = image_to_split.get(int(image.id), "")
        if split in split_images:
            split_images[split].add(int(image.id))
        active_labels = [lab for lab in image.labels if lab.active is not False]
        if not active_labels:
            manifest_rows.append(
                {
                    "split": split,
                    "grade": "",
                    "src": "",
                    "dest": "",
                    "source_image_id": image.id,
                    "source_photo_id": image.image_id,
                    "image_id": image.image_id,
                    "patient_id": image.patient_id or "",
                    "original_image_path": _manifest_path(Path(image.path)) if image.path else "",
                    "label_id": "",
                    "crop_id": "",
                    "annotation_status": "NO_CROPS",
                    "verification_status": "UNVERIFIED" if not image.boxes_verified else "VERIFIED",
                    "detector_version": image.detector_version or "Tooth Detector V2",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "crop_hash": "",
                    "content_hash": image.content_hash or "",
                    "status": "UNASSIGNED_PATIENT" if not split else "NO_DETECTED_CROPS",
                    "reason": "No active detected tooth crops are available for export." if split else "Photo has no safe patient assignment and is excluded.",
                }
            )
        for lab in image.labels:
            ann = _effective_annotation_status(lab)
            row = {
                "split": split,
                "grade": lab.grade if ann == "LABELED" else "",
                "src": _manifest_path(Path(lab.crop_path)) if lab.crop_path and Path(lab.crop_path).exists() else "",
                "dest": "",
                "source_image_id": image.id,
                "source_photo_id": image.image_id,
                "image_id": image.image_id,
                "patient_id": image.patient_id or "",
                "original_image_path": _manifest_path(Path(image.path)) if image.path else "",
                "label_id": lab.id,
                "crop_id": lab.crop_id or f"crop-{lab.id}",
                "annotation_status": ann,
                "verification_status": "VERIFIED" if image.boxes_verified and lab.box_verified else "UNVERIFIED",
                "detector_version": image.detector_version or "Tooth Detector V2",
                "x1": lab.x1,
                "y1": lab.y1,
                "x2": lab.x2,
                "y2": lab.y2,
                "crop_hash": lab.crop_hash or "",
                "content_hash": image.content_hash or "",
                "status": "",
                "reason": "",
            }
            if lab.active is False:
                row["status"] = "INACTIVE"
                row["reason"] = "Crop was deleted or deactivated during box review."
            elif not split:
                row["status"] = "UNASSIGNED_PATIENT"
                row["reason"] = "Missing patient identity or explicit Unassigned choice; crop is excluded."
            elif not (image.patient_id or "").strip():
                row["status"] = "MISSING_PATIENT_ID"
                row["reason"] = "The source photo has no patient ID; the crop is excluded even if its exact image hash is linked to a split."
            elif (image.duplicate_status or STATUS_UNIQUE) != STATUS_UNIQUE:
                row["status"] = EXCLUDED_DUPLICATE
                row["reason"] = f"Original photo status is {image.duplicate_status}; duplicate/invalid photos are excluded."
            elif not image.boxes_verified or not lab.box_verified:
                row["status"] = "UNVERIFIED_BOX"
                row["reason"] = "Both the source photo and this tooth box must be verified before training."
            elif ann in {"UNCERTAIN", "UNGRADABLE"}:
                row["status"] = ann
                row["reason"] = "Manually excluded from supervised training."
            elif ann != "LABELED" or lab.grade is None:
                row["status"] = "UNLABELED"
                row["reason"] = "No clinician-confirmed ICDAS 0–4 label."
            elif (lab.crop_duplicate_status or CROP_UNIQUE) != CROP_UNIQUE:
                row["status"] = EXCLUDED_DUPLICATE
                row["reason"] = f"crop_{(lab.crop_duplicate_status or '').lower()}"
            elif lab.crop_hash and lab.crop_hash in seen_crop_hash:
                row["status"] = EXCLUDED_DUPLICATE
                row["reason"] = "Repeated exact crop hash."
            elif not lab.crop_path or not Path(lab.crop_path).is_file():
                row["status"] = "MISSING_CROP"
                row["reason"] = "Crop file is missing."
            else:
                seen_crop_hash.add(lab.crop_hash) if lab.crop_hash else None
                crop_filename = f"{image.image_id}_{lab.crop_id or ('crop-' + str(lab.id))}.jpg"
                relative_dest = f"{split_folders[split]}/images/{crop_filename}"
                destination = root / relative_dest
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lab.crop_path, destination)
                row["dest"] = relative_dest
                row["status"] = INCLUDED
                copied[split].append(lab)
                split_labels[split].append(
                    {
                        "filename": f"images/{crop_filename}",
                        "icdas_score": int(lab.grade),
                        "patient_id": image.patient_id,
                        "source_photo_id": image.image_id,
                        "source_training_image_id": int(image.id),
                        "crop_id": lab.crop_id or f"crop-{lab.id}",
                        "label_id": int(lab.id),
                        "crop_hash": lab.crop_hash or "",
                        "source_photo_sha256": image.content_hash or "",
                        "annotation_status": ann,
                        "verification_status": "VERIFIED",
                        "patient_identity_status": identity_provenance.get(image.patient_id or "", {}).get("status", "UNVERIFIED"),
                        "patient_identity_method": identity_provenance.get(image.patient_id or "", {}).get("method", ""),
                    }
                )
            manifest_rows.append(row)

    # Keep records that cannot enter the current assignment visible in the
    # version manifest while never exporting their original photos or crops.
    for image in db.query(TrainingImage).order_by(TrainingImage.id.asc()).all():
        if int(image.id) in {int(row.id) for row in source_images}:
            continue
        if image.is_active is False or image.exclude_from_dataset or image.duplicate_status in {STATUS_EXACT, STATUS_LIKELY, STATUS_INVALID}:
            reason = "Training photo is deactivated or excluded from the active dataset."
            status = "REJECTED" if image.is_active is False else EXCLUDED_DUPLICATE
            labels_to_record = [lab for lab in image.labels if lab.active is not False] or [None]
            for lab in labels_to_record:
                manifest_rows.append(
                    {
                        "split": "",
                        "grade": lab.grade if lab and lab.grade is not None else "",
                        "src": _manifest_path(Path(lab.crop_path)) if lab and lab.crop_path and Path(lab.crop_path).is_file() else "",
                        "dest": "",
                        "source_image_id": image.id,
                        "source_photo_id": image.image_id,
                        "image_id": image.image_id,
                        "patient_id": image.patient_id or "",
                        "original_image_path": _manifest_path(Path(image.path)) if image.path else "",
                        "label_id": lab.id if lab else "",
                        "crop_id": (lab.crop_id or f"crop-{lab.id}") if lab else "",
                        "annotation_status": _effective_annotation_status(lab) if lab else "EXCLUDED",
                        "verification_status": "VERIFIED" if lab and image.boxes_verified and lab.box_verified else "UNVERIFIED",
                        "detector_version": image.detector_version or "",
                        "x1": lab.x1 if lab else "",
                        "y1": lab.y1 if lab else "",
                        "x2": lab.x2 if lab else "",
                        "y2": lab.y2 if lab else "",
                        "crop_hash": lab.crop_hash or "" if lab else "",
                        "content_hash": image.content_hash or "",
                        "status": status,
                        "reason": reason,
                    }
                )

    for row in manifest_rows:
        identity = identity_provenance.get(str(row.get("patient_id") or ""), {})
        row["patient_identity_status"] = identity.get("status", "UNVERIFIED")
        row["patient_identity_method"] = identity.get("method", "")
        row["patient_identity_evidence"] = identity.get("evidence", "")

    label_fields = [
        "filename", "icdas_score", "patient_id", "source_photo_id", "source_training_image_id",
        "crop_id", "label_id", "crop_hash", "source_photo_sha256", "annotation_status", "verification_status",
        "patient_identity_status", "patient_identity_method",
    ]
    for split, folder in split_folders.items():
        split_dir = root / folder
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        with (split_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=label_fields)
            writer.writeheader()
            writer.writerows(split_labels[split])

    man = root / "manifest.csv"
    fields = [
        "split",
        "grade",
        "src",
        "dest",
        "source_image_id",
        "source_photo_id",
        "image_id",
        "patient_id",
        "patient_identity_status",
        "patient_identity_method",
        "patient_identity_evidence",
        "original_image_path",
        "label_id",
        "crop_id",
        "annotation_status",
        "verification_status",
        "detector_version",
        "x1",
        "y1",
        "x2",
        "y2",
        "status",
        "reason",
        "crop_hash",
        "content_hash",
    ]
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)

    split_summary = {
        split: {
            "patient_count": len(splits[split]),
            "image_count": len(split_images[split]),
            "crop_count": len(copied[split]),
            "labeled_crop_count": len(copied[split]),
            "verified_crop_count": len(copied[split]),
            "class_counts": _class_counts(copied[split]),
            "actual_patient_percent": round(100.0 * len(splits[split]) / max(1, len(patient_to_split)), 2),
            "target_patient_percent": round(100.0 * resolved["fractions"][split], 2),
            "deviation_percentage_points": round(
                100.0 * len(splits[split]) / max(1, len(patient_to_split))
                - 100.0 * resolved["fractions"][split],
                2,
            ),
        }
        for split in ("train", "val", "test")
    }
    split_manifest = {
        "seed": resolved["seed"],
        "strategy": resolved["strategy"],
        "ratios": resolved["fractions"],
        "assignments": {
            patient_id: {
                "split": split,
                "identity_provenance": identity_provenance.get(patient_id, {"status": "UNVERIFIED"}),
            }
            for patient_id, split in patient_to_split.items()
        },
        "image_assignments": {
            image.image_id: {
                "patient_id": image.patient_id,
                "identity_provenance": identity_provenance.get(image.patient_id or "", {"status": "UNVERIFIED"}),
                "split": image_to_split.get(int(image.id), "UNASSIGNED"),
                "source_image_id": image.id,
                "sha256": image.content_hash,
                "split_locked": bool(image.split_locked),
            }
            for image in source_images
        },
    }
    (root / "split_assignments.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": name,
                "version_number": version_number,
                "source_fingerprint": preview["fingerprint"],
                "split_assignments": split_manifest["assignments"],
                "photo_assignments": split_manifest["image_assignments"],
                "records": manifest_rows,
                "label_files": {split: f"{folder}/labels.csv" for split, folder in split_folders.items()},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    class_mapping = {
        "target": "Project subset of ICDAS; model grades remain 0–4.",
        "excluded_full_system_grades": [5, 6],
        "classes": {str(i): f"ICDAS {i}" for i in range(5)},
    }
    (root / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")

    stats = {
        "name": name,
        "version_number": version_number,
        "split_seed": resolved["seed"],
        "split_strategy": resolved["strategy"],
        "eligible_fingerprint": fingerprint,
        "split_source_fingerprint": preview["fingerprint"],
        "split_fractions_target": resolved["fractions"],
        "split_method": (
            "deterministic whole-patient and exact-duplicate assignment; every assigned photo and crop inherits the patient split; "
            "only unique, box-verified ICDAS 0–4 tooth crops are exported"
        ),
        "split_limitations": (
            split_fraction_limitations(len(patient_to_split))
            + class_support_limitations(patient_class_counts)
        ),
        "total_patients": len(patient_to_split),
        "approximate_patient_group_count": preview.get("approximate_patient_group_count", 0),
        "total_source_images_included": len(assigned_ids),
        "total_source_images_excluded": len(source_images) - len(assigned_ids),
        "total_tooth_crops_included": sum(len(v) for v in copied.values()),
        "split_summary": split_summary,
        "patient_assignments": {patient_id: split for patient_id, split in sorted(patient_to_split.items())},
        "image_assignments": split_manifest["image_assignments"],
        "readiness_warnings": preview["warnings"],
        "exclusions_acknowledged": bool(plan.get("allow_exclusions")),
        "excluded_crop_counts": dict(Counter(row["status"] for row in manifest_rows if row["status"] != INCLUDED)),
        "class_mapping": class_mapping,
        "annotation_counts": {
            "uncertain": sum(1 for r in manifest_rows if r["status"] == "UNCERTAIN"),
            "ungradable": sum(1 for r in manifest_rows if r["status"] == "UNGRADABLE"),
            "rejected": sum(1 for r in manifest_rows if r["status"] == "REJECTED"),
        },
        "class_counts_overall": _class_counts(sum(copied.values(), [])),
        "train_counts": _class_counts(copied["train"]),
        "validation_counts": _class_counts(copied["val"]),
        "test_counts": _class_counts(copied["test"]),
        "n_train": len(copied["train"]),
        "n_valid": len(copied["val"]),
        "n_test": len(copied["test"]),
        "n_source_train": split_summary["train"]["image_count"],
        "n_source_val": split_summary["val"]["image_count"],
        "n_source_test": split_summary["test"]["image_count"],
        "n_patient_train": split_summary["train"]["patient_count"],
        "n_patient_val": split_summary["val"]["patient_count"],
        "n_patient_test": split_summary["test"]["patient_count"],
        "n_excluded_duplicate_rows": sum(
            1 for r in manifest_rows if r["status"] == EXCLUDED_DUPLICATE
        ),
        "imbalance_warning": False,
        "disclaimer": (
            "AI research/screening dataset. Not a clinical diagnosis. ICDAS 5–6 out of scope."
        ),
    }
    vals = [stats["class_counts_overall"][str(i)] for i in range(5)]
    if max(vals) > 0 and min(vals) * 4 < max(vals):
        stats["imbalance_warning"] = True
    stats["validation"] = validate_built_dataset(root)
    (root / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (root / "split_report.md").write_text(_split_report_markdown(stats), encoding="utf-8")

    rec = DatasetVersion(
        name=name,
        path=str(root),
        status="READY" if stats["validation"]["ok"] else "BUILT_WITH_WARNINGS",
        n_train=stats["n_train"],
        n_valid=stats["n_valid"],
        n_test=stats["n_test"],
        version_number=version_number,
        statistics_json=json.dumps(stats),
        split_seed=resolved["seed"],
    )
    db.add(rec)
    _audit(
        db,
        "dataset_assignment_confirmed",
        "dataset_version",
        name,
        {
            "version": name,
            "path": str(root),
            "strategy": resolved["strategy"],
            "seed": resolved["seed"],
            "source_fingerprint": preview["fingerprint"],
            "patient_assignment_count": len(patient_to_split),
            "assigned_photo_count": len(assigned_ids),
            "excluded_photo_count": len(source_images) - len(assigned_ids),
            "exclusions_acknowledged": bool(plan.get("allow_exclusions")),
        },
    )
    db.commit()
    db.refresh(rec)
    return {
        "name": name,
        "path": str(root),
        "statistics": stats,
        "status": rec.status,
        "note": "Previous dataset versions were left unchanged. Gold/Batch 01/02 were not modified.",
    }


def _write_portal_train_yaml(dataset_path: Path, model_dir: Path, dataset_name: str) -> Path:
    src_yaml = PROJECT_ROOT / "ml" / "configs" / "default.yaml"
    text = src_yaml.read_text(encoding="utf-8") if src_yaml.exists() else "num_classes: 5\n"
    rel_data = dataset_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    rel_out = model_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    text += (
        f"\n# portal-generated — Tooth Detector V2 is never trained here\n"
        f"dataset_root: {rel_data}\n"
        f"output_dir: {rel_out}\n"
        f"experiment_name: icdas_{dataset_name}\n"
        "overwrite_root_checkpoints: false\n"
        "ordinal_regression: false\n"
    )
    # The dataset version is immutable after build; keep runtime configuration
    # with the versioned model artifacts instead of writing back into it.
    yaml_path = model_dir / "training_config.yaml"
    yaml_path.write_text(text, encoding="utf-8")
    for artifact in ("split_assignments.json", "class_mapping.json", "statistics.json", "manifest.csv"):
        source = dataset_path / artifact
        if source.is_file():
            shutil.copy2(source, model_dir / artifact)
    provenance = {
        "dataset_version": dataset_name,
        "dataset_path": _manifest_path(dataset_path),
        "model_directory": _manifest_path(model_dir),
        "config_file": "training_config.yaml",
        "dataset_artifacts": {
            artifact: {
                "file": artifact,
                "sha256": sha256_file(dataset_path / artifact),
            }
            for artifact in ("manifest.csv", "split_assignments.json", "class_mapping.json", "statistics.json")
            if (dataset_path / artifact).is_file()
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    (model_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return yaml_path


def validate_dataset_for_train(db: Session, dataset_name: str | None) -> DatasetVersion:
    latest = (
        db.query(DatasetVersion).filter(DatasetVersion.name == dataset_name).first()
        if dataset_name
        else db.query(DatasetVersion).order_by(DatasetVersion.id.desc()).first()
    )
    if latest is None:
        raise ValueError("Build a dataset version before training.")
    conflicts = (
        db.query(TrainingLabel)
        .filter(TrainingLabel.crop_duplicate_status == CROP_CONFLICT, ACTIVE_LABEL)
        .count()
    )
    if conflicts:
        raise ValueError(CONFLICT_MESSAGE)
    root = Path(latest.path)
    check = validate_built_dataset(root)
    if check["missing_classes_in_train"]:
        raise ValueError(
            "All ICDAS classes 0–4 must exist in the train split. Missing: "
            + ", ".join(str(x) for x in check["missing_classes_in_train"])
        )
    if check.get("missing_classes_in_validation"):
        raise ValueError(
            "The validation split must contain labeled examples of ICDAS classes 0–4. Missing: "
            + ", ".join(str(x) for x in check["missing_classes_in_validation"])
        )
    if check.get("missing_classes_in_test"):
        raise ValueError(
            "The held-out test split must contain labeled examples of ICDAS classes 0–4. Missing: "
            + ", ".join(str(x) for x in check["missing_classes_in_test"])
        )
    if check["issues"]:
        raise ValueError("Dataset failed integrity, split, or leakage validation: " + "; ".join(check["issues"][:8]))
    stats = json.loads(latest.statistics_json) if latest.statistics_json else {}
    patient_count = int(
        stats.get("total_patients")
        or sum((check.get("split_patient_counts") or {}).values())
    )
    min_patients = get_settings().min_training_patients
    if patient_count < min_patients:
        raise ValueError(
            f"Training requires at least {min_patients} distinct patients; "
            f"dataset {latest.name} contains {patient_count}."
        )
    if int(stats.get("n_train") or latest.n_train or 0) < 5:
        raise ValueError("Train split needs at least 5 verified crops.")
    return latest


def _finish_training_job(job_id: int, returncode: int, model_dir: Path, dataset_name: str) -> None:
    from .database import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job is None:
            return
        job.updated_at = datetime.utcnow()
        if returncode != 0:
            job.status = "failed"
            job.message = f"ml/train.py exited with code {returncode}"
            db.commit()
            return
        from .portal_runtime import is_valid_icdas_checkpoint

        checkpoint = _find_keras(model_dir)
        if checkpoint is None or not is_valid_icdas_checkpoint(checkpoint):
            job.status = "failed"
            job.message = "Training output is not a genuine five-class softmax checkpoint; model was not registered."
            db.commit()
            return
        metrics_path = model_dir / "test_results.json"
        if not metrics_path.exists():
            job.status = "failed"
            job.message = "Training ended without a saved held-out test evaluation; model was not registered."
            db.add(
                TrainingAuditEvent(
                    event_type="training_evaluation_missing",
                    entity_type="training_job",
                    entity_id=str(job.id),
                    actor="admin",
                    details_json=json.dumps({"dataset_version": dataset_name, "model_directory": str(model_dir)}),
                )
            )
            db.commit()
            return
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not metrics.get("per_class") or not metrics.get("confusion_matrix"):
            job.status = "failed"
            job.message = "Held-out test evaluation is incomplete; model was not registered."
            db.commit()
            return
        db.add(
            ModelVersion(
                name=f"ICDAS {model_dir.name.upper()}",
                kind="icdas",
                path=str(model_dir),
                dataset_version=dataset_name,
                metrics_json=json.dumps(metrics) if metrics else None,
                is_active=False,
            )
        )
        job.status = "completed"
        job.message = (
            f"Training finished. Model saved at {model_dir}. "
            "Not set active. Research/screening model only — not declared clinically accurate."
        )
        db.add(
            TrainingAuditEvent(
                event_type="training_completed",
                entity_type="training_job",
                entity_id=str(job.id),
                actor="admin",
                details_json=json.dumps(
                    {
                        "dataset_version": dataset_name,
                        "model_directory": str(model_dir),
                        "test_accuracy": metrics.get("accuracy"),
                        "test_macro_f1": metrics.get("macro_f1"),
                        "per_class": metrics.get("per_class"),
                    },
                    sort_keys=True,
                ),
            )
        )
        db.commit()
    except Exception:
        logger.exception("Failed to finalize training job %s", job_id)
        db.rollback()
    finally:
        db.close()


def launch_icdas_training(db: Session, dataset_name: str | None) -> dict:
    import subprocess
    import sys

    settings = get_settings()
    if not settings.allow_icdas_train:
        job = TrainingJob(
            status="blocked",
            message="ICDAS training is disabled.",
            log_text="Set ICDAS_ALLOW_TRAIN=1 on the backend to enable. Default is disabled.",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return {
            "job_id": job.id,
            "status": job.status,
            "message": job.message,
            "log": job.log_text,
            "launched": False,
            "icdas_train_enabled": False,
        }
    latest = validate_dataset_for_train(db, dataset_name)
    ver, model_dir = next_icdas_model_dir()
    model_dir.mkdir(parents=True, exist_ok=False)
    yaml_path = _write_portal_train_yaml(Path(latest.path), model_dir, latest.name)
    log_path = model_dir / "train_job.log"
    job = TrainingJob(
        status="running",
        message=f"Launching ml/train.py → {model_dir} (dataset {latest.name})",
        dataset_name=latest.name,
        model_dir=str(model_dir),
        log_text=f"pid=pending\nlog={log_path}",
    )
    db.add(job)
    db.flush()
    _audit(
        db,
        "training_started",
        "training_job",
        job.id,
        {"dataset_version": latest.name, "model_directory": str(model_dir)},
    )
    db.commit()
    db.refresh(job)
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "ml" / "train.py"), "--config", str(yaml_path)],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        job.log_text = f"pid={proc.pid}\nlog={log_path}"
        db.commit()
    except Exception as exc:
        log_handle.close()
        job.status = "failed"
        job.message = str(exc)
        db.commit()
        return {
            "job_id": job.id,
            "status": job.status,
            "message": job.message,
            "log": job.log_text,
            "launched": False,
            "icdas_train_enabled": True,
        }

    job_id = job.id
    dataset_name_s = latest.name

    def _watch() -> None:
        code = proc.wait()
        try:
            log_handle.close()
        except Exception:
            pass
        _finish_training_job(job_id, code, model_dir, dataset_name_s)

    threading.Thread(target=_watch, daemon=True).start()
    return {
        "job_id": job.id,
        "status": job.status,
        "message": job.message,
        "log": job.log_text,
        "launched": True,
        "icdas_train_enabled": True,
        "model_dir": str(model_dir),
        "dataset": latest.name,
        "note": "Streamlit is not blocked. Training runs as a backend subprocess. YOLO is not retrained.",
    }


def apply_active_icdas_model(db: Session, model_id: int, active: bool) -> dict:
    from .portal_runtime import is_blocked_icdas_checkpoint, is_valid_icdas_checkpoint, reset_portal_runtime

    m = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if m is None:
        raise KeyError(model_id)
    if m.kind == "detection":
        if active:
            for other in db.query(ModelVersion).filter(ModelVersion.kind == "detection"):
                other.is_active = other.id == m.id
        else:
            m.is_active = False
        db.commit()
        return {
            "id": m.id,
            "is_active": m.is_active,
            "note": "Detection registry flag only. Tooth Detector V2 weights were not overwritten.",
        }
    if m.kind != "icdas":
        m.is_active = active
        db.commit()
        return {"id": m.id, "is_active": m.is_active}

    path_text = str(m.path).replace("\\", "/").lower()
    if "historical" in path_text or is_blocked_icdas_checkpoint(m.path):
        raise ValueError("Historical ICDAS checkpoints cannot be set active.")

    keras = _find_keras(Path(m.path)) if active else None
    if active and keras is None:
        raise ValueError(f"No keras checkpoint found under {m.path}")
    if active and keras is not None and not is_valid_icdas_checkpoint(keras):
        raise ValueError("Only a loadable five-class softmax ICDAS model can be promoted.")
    metrics = json.loads(m.metrics_json) if m.metrics_json else {}
    if active:
        if not all(key in metrics for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix")):
            raise ValueError("Review the saved held-out test metrics before promoting this ICDAS model.")
        missing_support = [
            str(grade)
            for grade in range(5)
            if int((metrics.get("per_class", {}).get(str(grade)) or {}).get("support", 0)) < 1
        ]
        if missing_support:
            raise ValueError(
                "The held-out test evaluation has no support for ICDAS class(es): "
                + ", ".join(missing_support)
                + ". This model cannot be promoted."
            )

    for other in db.query(ModelVersion).filter(ModelVersion.kind == "icdas"):
        other.is_active = False
    m.is_active = bool(active)
    _audit(
        db,
        "model_promotion_reviewed" if active else "model_deactivated",
        "model_version",
        m.id,
        {
            "name": m.name,
            "dataset_version": m.dataset_version,
            "active": bool(active),
            "metrics_present": bool(m.metrics_json),
            "test_accuracy": metrics.get("accuracy"),
            "test_macro_f1": metrics.get("macro_f1"),
        },
    )
    db.commit()
    current = MODELS_ICDAS / "current"
    current.mkdir(parents=True, exist_ok=True)
    slot = current / "deploy.keras"
    if active and keras is not None:
        shutil.copy2(keras, slot)
        (current / "ACTIVE.txt").write_text(
            f"active_model_id={m.id}\nname={m.name}\nsource={keras}\n",
            encoding="utf-8",
        )
    elif not active and slot.exists():
        slot.unlink()
    reset_portal_runtime()
    return {
        "id": m.id,
        "is_active": m.is_active,
        "keras": str(keras) if keras else None,
        "inference_slot": str(slot),
        "note": (
            "SET ACTIVE updates the local inference pointer only. "
            "Versioned weights under models/icdas/vN/ were not overwritten. Not a cloud deploy."
        ),
    }


def _find_keras(root: Path) -> Path | None:
    if root.is_file() and root.suffix == ".keras":
        return root
    for name in ("deploy.keras", "final.keras", "best.keras"):
        p = root / name
        if p.exists():
            return p
    found = sorted(root.rglob("*.keras"))
    return found[0] if found else None
