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
    ModelVersion,
    TrainingAuditEvent,
    TrainingImage,
    TrainingJob,
    TrainingLabel,
)
from .storage_paths import (
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
            "patient_id": validate_patient_id(normalized.get("patient_id")),
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
        if row.version_number:
            nums.append(int(row.version_number))
    if TRAIN_VERSIONS.exists():
        for p in TRAIN_VERSIONS.iterdir():
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
                nums.append(int(p.name[1:]))
    n = max(nums, default=0) + 1
    return f"v{n}", n


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


def ingest_training_image(
    db: Session,
    filename: str,
    data: bytes,
    decode_fn,
    *,
    patient_id: str | None = None,
    image_id: str | None = None,
) -> dict:
    """Store one admin training photo. Never writes to patient folders."""
    ensure_dirs()
    patient_id = validate_patient_id(patient_id) if patient_id else None
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

    # Do not persist a potentially identifying source filename. Use only the
    # pseudonymous image key and a validated image extension on private storage.
    stored_filename = f"{image_id}{Path(filename).suffix.lower()}"
    dest = TRAIN_UPLOADS / stored_filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    rec = TrainingImage(
        filename=stored_filename,
        path=str(dest),
        image_id=image_id,
        patient_id=patient_id,
        upload_timestamp=datetime.utcnow(),
        annotation_status="UNLABELED",
        split_name="UNASSIGNED",
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
    db.add(rec)
    db.flush()

    crops = []
    overlay_b64 = None
    original_b64 = None
    detected = False
    if status == STATUS_UNIQUE and rgb is not None:
        if not detector_available():
            raise RuntimeError(detector_error() or "Tooth Detector V2 unavailable.")
        crops, overlay = _detect_and_store_crops(db, rec, rgb)
        detected = True
        rec.review_status = "PENDING"
        rec.annotation_status = "UNLABELED"
        from .portal_service import _b64_rgb

        overlay_b64 = _b64_rgb(overlay)
        original_b64 = _b64_rgb(rgb)
    elif rgb is not None:
        from .portal_service import _b64_rgb

        original_b64 = _b64_rgb(rgb)

    db.commit()
    db.refresh(rec)
    return {
        "training_image_id": rec.id,
        "image_id": rec.image_id,
        "patient_id": rec.patient_id,
        "filename": rec.filename,
        "duplicate_status": rec.duplicate_status,
        "canonical_id": rec.canonical_id,
        "excluded_from_dataset": bool(rec.exclude_from_dataset),
        "invalid_reason": invalid_reason,
        "n_crops": rec.n_crops,
        "crop_count": rec.crop_count,
        "review_status": rec.review_status,
        "detector_ran": detected,
        "overlay_base64": overlay_b64,
        "original_base64": original_b64,
        "crops": crops,
        "note": (
            "Original upload was kept. Duplicate copies are flagged and excluded "
            "from the training dataset; they are not deleted."
            if status != STATUS_UNIQUE
            else "Canonical UNIQUE image. Tooth Detector V2 ran; boxes need admin review."
        ),
    }


def update_training_identity(
    db: Session,
    training_image_id: int,
    patient_id: str,
    image_id: str | None = None,
) -> dict:
    rec = db.query(TrainingImage).filter(TrainingImage.id == training_image_id).first()
    if rec is None:
        raise KeyError(training_image_id)
    patient_id = validate_patient_id(patient_id)
    new_image_id = validate_image_id(image_id) if image_id else rec.image_id
    duplicate = (
        db.query(TrainingImage)
        .filter(TrainingImage.image_id == new_image_id, TrainingImage.id != rec.id)
        .first()
    )
    if duplicate:
        raise ValueError(f"Image ID {new_image_id} already exists.")
    old = {"patient_id": rec.patient_id, "image_id": rec.image_id}
    rec.patient_id = patient_id
    rec.image_id = new_image_id
    # Identity edits invalidate the current, mutable assignment. Existing
    # DatasetVersion manifests remain immutable historical records.
    rec.dataset_version = None
    rec.split_name = "UNASSIGNED"
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
                "dataset_version": r.dataset_version,
                "annotation_status": r.annotation_status,
                "review_status": r.review_status,
                "upload_timestamp": r.upload_timestamp.isoformat() if r.upload_timestamp else None,
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
        "images": items,
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
    _audit(
        db,
        "detections_verified",
        "training_image",
        rec.image_id,
        {"training_image_id": rec.id, "patient_id": rec.patient_id, "active_crop_count": len(active)},
    )
    db.commit()
    return {"training_image_id": rec.id, "boxes_verified": True, "n_crops": len(active)}


def labeling_queue(
    db: Session,
    image_id: int | None = None,
    label_id: int | None = None,
    resume: bool = False,
    patient_id: str | None = None,
    split_name: str | None = None,
    grade: int | None = None,
    annotation_status: str | None = None,
) -> dict:
    from .portal_service import _b64_rgb, _read_rgb

    inventory = dataset_inventory(db)
    ordered = _ordered_verified_labels(
        db,
        patient_id=patient_id,
        split_name=split_name,
        grade=grade,
        annotation_status=annotation_status,
    )
    ids = [lab.id for lab in ordered]
    resume_label_id = None
    for lab in ordered:
        skipped = bool(lab.skipped)
        if (
            lab.grade is None
            and not skipped
            and _effective_annotation_status(lab) == "UNLABELED"
        ):
            resume_label_id = lab.id
            break
    current_lab = None
    if resume and resume_label_id:
        current_lab = next((l for l in ordered if l.id == resume_label_id), None)
    elif label_id:
        current_lab = next((l for l in ordered if l.id == int(label_id)), None)
    elif image_id:
        current_lab = next((l for l in ordered if l.image_id == int(image_id)), None)
    elif resume_label_id:
        current_lab = next((l for l in ordered if l.id == resume_label_id), None)
    elif ordered:
        current_lab = ordered[0]

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
        "note": "One verified UNIQUE crop at a time. Groq is not used. Missing 2/3 on a photo is valid.",
    }


def save_icdas_label(db: Session, label_id: int, grade: int) -> dict:
    grade = validate_icdas_grade(grade)
    lab = db.query(TrainingLabel).filter(TrainingLabel.id == label_id).first()
    if lab is None:
        raise KeyError(label_id)
    img = lab.image
    if img is None or (img.duplicate_status not in {STATUS_UNIQUE, None}):
        raise ValueError("ICDAS labels can only be saved on UNIQUE training photos.")
    if lab.active is False or not lab.box_verified or not img.boxes_verified:
        raise ValueError("Crop must be an active verified tooth box before labeling.")
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


def split_patients(patient_ids: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Deterministic 70/15/15 split over patient groups, never image/crop rows."""
    ids = sorted({str(value).strip() for value in patient_ids if str(value).strip()})
    rng = np.random.default_rng(seed)
    order = np.asarray(ids, dtype=object)
    rng.shuffle(order)
    n = len(order)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": [str(order[0])], "val": [], "test": []}
    if n == 2:
        return {"train": [str(order[0])], "val": [str(order[1])], "test": []}
    n_train = min(max(1, int(round(n * SPLIT_FRACS[0]))), n - 2)
    n_val = min(max(1, int(round(n * SPLIT_FRACS[1]))), n - n_train - 1)
    return {
        "train": [str(v) for v in order[:n_train]],
        "val": [str(v) for v in order[n_train : n_train + n_val]],
        "test": [str(v) for v in order[n_train + n_val :]],
    }


def _class_counts(labs: list[TrainingLabel]) -> dict[str, int]:
    c = Counter(int(l.grade) for l in labs)
    return {str(i): int(c.get(i, 0)) for i in range(5)}


def dataset_inventory(db: Session) -> dict:
    refresh_crop_identities(db)
    all_imgs = db.query(TrainingImage).all()
    active_imgs = [i for i in all_imgs if i.is_active is not False]
    dup_counts = Counter(i.duplicate_status or STATUS_UNIQUE for i in active_imgs)
    unique_imgs = [i for i in active_imgs if (i.duplicate_status or STATUS_UNIQUE) == STATUS_UNIQUE]
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
                issues.append(f"Split row has no patient_id (image {image_id or row.get('source_image_id')}).")
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
    missing_test = [g for g in range(5) if counts["test"][str(g)] == 0]
    return {
        "ok": not issues and not missing_train and not missing_test,
        "issues": issues,
        "missing_classes_in_train": missing_train,
        "missing_classes_in_test": missing_test,
        "split_class_counts": counts,
        "split_patient_counts": {split: len(values) for split, values in patient_count.items()},
        "split_image_counts": {split: len(values) for split, values in image_count.items()},
        "split_crop_counts": {split: int(crop_count[split]) for split in counts},
        "patient_leakage_count": sum(1 for values in patient_splits.values() if len(values) > 1),
        "image_leakage_count": sum(1 for values in image_splits.values() if len(values) > 1),
    }


def build_dataset_version(db: Session) -> dict:
    """Create the next versioned split. Never overwrites an existing version folder."""
    ensure_dirs()
    refresh_crop_identities(db)
    db.flush()
    inv = dataset_inventory(db)
    if inv["photos"]["unassigned_patient_ids"]:
        raise ValueError(
            f"Assign pseudonymous patient IDs to all {inv['photos']['unassigned_patient_ids']} UNIQUE photos before building."
        )
    if inv["photos"]["unreviewed_images"]:
        raise ValueError(
            f"Review detections for all active UNIQUE photos before building ({inv['photos']['unreviewed_images']} remain)."
        )
    if inv["photos"].get("unverified_crops", 0):
        raise ValueError(
            f"Verify all active tooth boxes before building ({inv['photos']['unverified_crops']} remain)."
        )
    if inv["crops"]["unlabeled"]:
        raise ValueError(
            f"Resolve all verified crops as ICDAS 0–4, uncertain, or ungradable before building ({inv['crops']['unlabeled']} remain)."
        )

    source_images = [
        image
        for image in db.query(TrainingImage).all()
        if (image.duplicate_status or STATUS_UNIQUE) == STATUS_UNIQUE
        and image.is_active is not False
        and not image.exclude_from_dataset
    ]
    if len({image.patient_id for image in source_images if image.patient_id}) < 3:
        raise ValueError("At least 3 distinct patient IDs are required to create non-empty train, validation, and test splits.")

    labeled = list(_eligible_label_query(db).all())
    unverified = [
        lab.id
        for lab in labeled
        if not lab.box_verified or not lab.image or not lab.image.boxes_verified
    ]
    if unverified:
        raise ValueError(
            f"Cannot build dataset because {len(unverified)} selected crops are unverified."
        )
    for lab in labeled:
        try:
            validate_icdas_grade(lab.grade)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    n_labeled = len(labeled)
    min_msg = dataset_min_crops_message(n_labeled)
    if min_msg:
        raise ValueError(min_msg)
    fingerprint_source = "|".join(
        f"{lab.id}:{lab.image_id}:{lab.image.patient_id}:{lab.grade}:{lab.annotation_status}:{lab.crop_hash or ''}"
        for lab in sorted(labeled, key=lambda x: int(x.id))
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    latest_existing = db.query(DatasetVersion).order_by(DatasetVersion.id.desc()).first()
    if latest_existing and latest_existing.statistics_json:
        prev = json.loads(latest_existing.statistics_json)
        if prev.get("eligible_fingerprint") == fingerprint:
            raise ValueError(
                f"No new verified labeled crops since {latest_existing.name}; "
                "not creating a duplicate version."
            )
    name, version_number = next_dataset_version_name(db)
    root = TRAIN_VERSIONS / name
    if root.exists():
        raise ValueError(f"Dataset folder {root} already exists and will not be overwritten.")
    root.mkdir(parents=True, exist_ok=False)

    patient_ids = [str(image.patient_id) for image in source_images]
    splits = split_patients(patient_ids, seed=SPLIT_SEED)
    patient_to_split = {patient_id: split for split, values in splits.items() for patient_id in values}
    image_to_split = {}
    for image in source_images:
        split = patient_to_split[str(image.patient_id)]
        image_to_split[image.id] = split
        image.dataset_version = name
        image.split_name = split

    copied: dict[str, list[TrainingLabel]] = {"train": [], "val": [], "test": []}
    split_images: dict[str, set[int]] = {split: set() for split in copied}
    manifest_rows: list[dict] = []
    seen_crop_hash: set[str] = set()
    for image in source_images:
        split = image_to_split[image.id]
        split_images[split].add(image.id)
        active_labels = [lab for lab in image.labels if lab.active is not False]
        for lab in active_labels:
            ann = _effective_annotation_status(lab)
            row = {
                "split": split,
                "grade": lab.grade if ann == "LABELED" else "",
                "src": _manifest_path(Path(lab.crop_path))
                if lab.crop_path and Path(lab.crop_path).exists()
                else "",
                "dest": "",
                "source_image_id": image.id,
                "image_id": image.image_id,
                "patient_id": image.patient_id,
                "original_image_path": _manifest_path(Path(image.path)),
                "label_id": lab.id,
                "crop_id": lab.crop_id,
                "annotation_status": ann,
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
            if ann in {"UNCERTAIN", "UNGRADABLE"}:
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
            elif not Path(lab.crop_path).exists():
                row["status"] = "MISSING_CROP"
                row["reason"] = "Crop file is missing."
            else:
                seen_crop_hash.add(lab.crop_hash) if lab.crop_hash else None
                destination = root / split / str(int(lab.grade)) / f"{image.image_id}_{lab.crop_id}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lab.crop_path, destination)
                row["dest"] = _manifest_path(destination)
                row["status"] = INCLUDED
                copied[split].append(lab)
            manifest_rows.append(row)

    # Keep rejected crops and duplicate photos visible in the manifest, while
    # never allowing them to contribute a training sample.
    for image in db.query(TrainingImage).all():
        if image.is_active is False:
            for lab in image.labels:
                manifest_rows.append(
                    {
                        "split": image_to_split.get(image.id, ""),
                        "grade": lab.grade if lab.grade is not None else "",
                        "src": "",
                        "dest": "",
                        "source_image_id": image.id,
                        "image_id": image.image_id,
                        "patient_id": image.patient_id or "",
                        "original_image_path": "",
                        "label_id": lab.id,
                        "crop_id": lab.crop_id,
                        "annotation_status": _effective_annotation_status(lab),
                        "detector_version": image.detector_version or "",
                        "x1": lab.x1,
                        "y1": lab.y1,
                        "x2": lab.x2,
                        "y2": lab.y2,
                        "crop_hash": lab.crop_hash or "",
                        "content_hash": image.content_hash or "",
                        "status": "REJECTED",
                        "reason": "Training image is deactivated.",
                    }
                )
        elif image.duplicate_status in {STATUS_EXACT, STATUS_LIKELY, STATUS_INVALID}:
            manifest_rows.append(
                {
                    "split": "",
                    "grade": "",
                    "src": "",
                    "dest": "",
                    "source_image_id": image.id,
                    "image_id": image.image_id,
                    "patient_id": image.patient_id or "",
                    "original_image_path": "",
                    "label_id": "",
                    "crop_id": "",
                    "annotation_status": "EXCLUDED",
                    "detector_version": image.detector_version or "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "crop_hash": "",
                    "content_hash": image.content_hash or "",
                    "status": EXCLUDED_DUPLICATE,
                    "reason": image.duplicate_status,
                }
            )

    man = root / "manifest.csv"
    fields = [
        "split",
        "grade",
        "src",
        "dest",
        "source_image_id",
        "image_id",
        "patient_id",
        "original_image_path",
        "label_id",
        "crop_id",
        "annotation_status",
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

    if (root / "val").exists():
        valid_alias = root / "valid"
        if not valid_alias.exists():
            shutil.copytree(root / "val", valid_alias)

    split_summary = {
        split: {
            "patient_count": len(splits[split]),
            "image_count": len(split_images[split]),
            "crop_count": len(copied[split]),
            "class_counts": _class_counts(copied[split]),
        }
        for split in ("train", "val", "test")
    }
    split_manifest = {
        "seed": SPLIT_SEED,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "assignments": {
            patient_id: {"split": split} for split, values in splits.items() for patient_id in values
        },
        "image_assignments": {
            image.image_id: {
                "patient_id": image.patient_id,
                "split": image_to_split[image.id],
                "source_image_id": image.id,
            }
            for image in source_images
        },
    }
    (root / "split_assignments.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    class_mapping = {
        "target": "Project subset of ICDAS; model grades remain 0–4.",
        "excluded_full_system_grades": [5, 6],
        "classes": {str(i): f"ICDAS {i}" for i in range(5)},
    }
    (root / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")

    stats = {
        "name": name,
        "version_number": version_number,
        "split_seed": SPLIT_SEED,
        "eligible_fingerprint": fingerprint,
        "split_fractions_target": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_method": "deterministic patient-level shuffle; all images and crops for a patient share one split",
        "total_patients": len(patient_to_split),
        "total_source_images_included": len(source_images),
        "total_tooth_crops_included": sum(len(v) for v in copied.values()),
        "split_summary": split_summary,
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

    rec = DatasetVersion(
        name=name,
        path=str(root),
        status="READY" if stats["validation"]["ok"] else "BUILT_WITH_WARNINGS",
        n_train=stats["n_train"],
        n_valid=stats["n_valid"],
        n_test=stats["n_test"],
        version_number=version_number,
        statistics_json=json.dumps(stats),
        split_seed=SPLIT_SEED,
    )
    db.add(rec)
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
        raise ValueError("Train split is too small.")
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
    from .portal_runtime import is_blocked_icdas_checkpoint, reset_portal_runtime

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
