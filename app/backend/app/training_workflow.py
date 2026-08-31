"""Admin ICDAS training-dataset workflow. Does not train YOLO or mix patient photos."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import threading
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
from .portal_db import DatasetVersion, ModelVersion, TrainingImage, TrainingJob, TrainingLabel
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
STATUS_UNIQUE = "UNIQUE"
STATUS_EXACT = "EXACT_DUPLICATE"
STATUS_LIKELY = "LIKELY_DUPLICATE"
STATUS_INVALID = "INVALID"
INCLUDED = "INCLUDED"
EXCLUDED_DUPLICATE = "EXCLUDED_DUPLICATE"

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
                "confidence": item.confidence,
                "x1": item.x1,
                "y1": item.y1,
                "x2": item.x2,
                "y2": item.y2,
                "grade": None,
            }
        )
    return crops, overlay


def ingest_training_image(db: Session, filename: str, data: bytes, decode_fn) -> dict:
    """Store one admin training photo. Never writes to patient folders."""
    ensure_dirs()
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

    dest = TRAIN_UPLOADS / f"{digest[:16]}_{Path(filename).name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    rec = TrainingImage(
        filename=Path(filename).name,
        path=str(dest),
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
        "filename": rec.filename,
        "duplicate_status": rec.duplicate_status,
        "canonical_id": rec.canonical_id,
        "excluded_from_dataset": bool(rec.exclude_from_dataset),
        "invalid_reason": invalid_reason,
        "n_crops": rec.n_crops,
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


def list_training_images(db: Session) -> dict:
    rows = db.query(TrainingImage).order_by(TrainingImage.id.asc()).all()
    items = []
    active_rows = [r for r in rows if r.is_active is not False]
    counts = Counter(r.duplicate_status or STATUS_UNIQUE for r in active_rows)
    for r in rows:
        items.append(
            {
                "training_image_id": r.id,
                "filename": r.filename,
                "duplicate_status": r.duplicate_status,
                "canonical_id": r.canonical_id,
                "n_crops": r.n_crops,
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
                "x1": lab.x1,
                "y1": lab.y1,
                "x2": lab.x2,
                "y2": lab.y2,
                "confidence": lab.confidence,
                "grade": lab.grade,
                "box_verified": bool(lab.box_verified),
                "crop_duplicate_status": lab.crop_duplicate_status or CROP_UNIQUE,
                "duplicate_of_label_id": lab.duplicate_of_label_id,
                "crop_status_caption": crop_status_caption(lab),
                "crop_base64": _b64_rgb(crop),
            }
        )
    return {
        "training_image_id": rec.id,
        "filename": rec.filename,
        "duplicate_status": rec.duplicate_status,
        "canonical_id": rec.canonical_id,
        "boxes_verified": bool(rec.boxes_verified),
        "is_active": bool(rec.is_active if rec.is_active is not None else True),
        "n_crops": rec.n_crops,
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
            lab = TrainingLabel(image_id=rec.id, grade=None, confidence=0.0, crop_path="")
            db.add(lab)
            db.flush()
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
    rec.boxes_verified = False
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
    for lab in active:
        lab.box_verified = True
    db.commit()
    return {"training_image_id": rec.id, "boxes_verified": True, "n_crops": len(active)}


def labeling_queue(
    db: Session,
    image_id: int | None = None,
    label_id: int | None = None,
    resume: bool = False,
) -> dict:
    from .portal_service import _b64_rgb, _read_rgb

    inventory = dataset_inventory(db)
    ordered = _ordered_verified_labels(db)
    ids = [lab.id for lab in ordered]
    resume_label_id = None
    for lab in ordered:
        skipped = bool(lab.skipped)
        if lab.grade is None and not skipped:
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
            "training_image_id": current_lab.image_id,
            "filename": current_lab.image.filename if current_lab.image else "",
            "grade": current_lab.grade,
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
    lab.grade = grade
    lab.skipped = False
    labeled_dir = TRAIN_LABELED / str(grade)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    dest = labeled_dir / f"{lab.id}_{Path(lab.crop_path).name}"
    if Path(lab.crop_path).exists():
        shutil.copy2(lab.crop_path, dest)
    apply_crop_identity(db, lab)
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
    lab.skipped = True
    lab.grade = None
    if lab.crop_hash:
        sync_exact_hash_group(db, lab.crop_hash)
    db.commit()
    return {"label_id": lab.id, "skipped": True, "grade": None, "saved": True}


def _ordered_verified_labels(db: Session) -> list[TrainingLabel]:
    return (
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
        .all()
    )


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


def _class_counts(labs: list[TrainingLabel]) -> dict[str, int]:
    c = Counter(int(l.grade) for l in labs)
    return {str(i): int(c.get(i, 0)) for i in range(5)}


def dataset_inventory(db: Session) -> dict:
    refresh_crop_identities(db)
    all_imgs = db.query(TrainingImage).all()
    active_imgs = [i for i in all_imgs if i.is_active is not False]
    dup_counts = Counter(i.duplicate_status or STATUS_UNIQUE for i in active_imgs)
    unique_imgs = [i for i in active_imgs if (i.duplicate_status or STATUS_UNIQUE) == STATUS_UNIQUE]
    eligible = list(_eligible_label_query(db).all())
    class_counts = _class_counts(eligible)
    labeled_n = sum(class_counts.values())

    detected = 0
    verified = 0
    skipped_n = 0
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
                if lab.skipped:
                    skipped_n += 1
                elif lab.grade is None:
                    unlabeled_n += 1
                else:
                    labeled_raw += 1
                if crop_is_resolved(lab.grade, bool(lab.skipped)):
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
    ready = dataset_build_ready(labeled_n, class_counts)
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
        },
        "crops": {
            "detected": detected,
            "verified": verified,
            "labeled": labeled_raw,
            "unlabeled": unlabeled_n,
            "skipped": skipped_n,
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
        "min_dataset_crops": MIN_DATASET_CROPS,
        "min_crops_message": min_msg,
        "classes_ready": dataset_classes_ready(class_counts),
        "crops_ready": labeled_n >= MIN_DATASET_CROPS,
        "dataset_ready": ready,
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
    issues = []
    counts = {split: {str(g): 0 for g in range(5)} for split in ("train", "val", "test")}
    source_in_split: dict[int, set[str]] = defaultdict(set)
    hashes_in_split: dict[str, dict[str, str]] = defaultdict(dict)
    man = root / "manifest.csv"
    if not man.exists():
        raise ValueError(f"Missing manifest.csv in {root}")
    with man.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status") or INCLUDED
            if status != INCLUDED:
                continue
            split = row.get("split") or ""
            grade = str(row.get("grade") or "")
            if split not in counts or grade not in counts[split]:
                issues.append(f"Bad split/grade row: {row}")
                continue
            counts[split][grade] += 1
            sid = row.get("source_image_id") or ""
            if sid.isdigit():
                source_in_split[int(sid)].add(split)
            ch = row.get("crop_hash") or ""
            if ch:
                prev = hashes_in_split.get(ch)
                if prev:
                    issues.append(
                        f"Duplicate crop hash {ch[:12]} included more than once "
                        f"({prev.get('split')} and {split})"
                    )
                hashes_in_split[ch] = {"split": split}
    for sid, splits in source_in_split.items():
        if len(splits) > 1:
            issues.append(f"Source image {sid} leaked across splits {sorted(splits)}")
    for split in ("train", "val", "test"):
        if sum(counts[split].values()) == 0:
            issues.append(f"Split {split} is empty")
    missing_train = [g for g in range(5) if counts["train"][str(g)] == 0]
    return {
        "ok": not issues and not missing_train,
        "issues": issues,
        "missing_classes_in_train": missing_train,
        "split_class_counts": counts,
    }


def build_dataset_version(db: Session) -> dict:
    """Create the next versioned split. Never overwrites an existing version folder."""
    ensure_dirs()
    refresh_crop_identities(db)
    db.flush()
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
    fingerprint = _eligible_fingerprint(labeled)
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

    excluded_rows = []
    included_by_image: dict[int, list[TrainingLabel]] = defaultdict(list)
    seen_crop_hash: dict[str, int] = {}
    for lab in labeled:
        img = lab.image
        status = lab.crop_duplicate_status or CROP_UNIQUE
        if status in {CROP_DUP_EXACT, CROP_POSSIBLE, CROP_CONFLICT}:
            excluded_rows.append(
                {
                    "label": lab,
                    "status": EXCLUDED_DUPLICATE,
                    "reason": f"crop_{status.lower()}",
                }
            )
            continue
        digest = lab.crop_hash
        if digest and digest in seen_crop_hash:
            excluded_rows.append(
                {
                    "label": lab,
                    "status": EXCLUDED_DUPLICATE,
                    "reason": f"duplicate_tooth_crop hash of label {seen_crop_hash[digest]}",
                }
            )
            continue
        if digest:
            seen_crop_hash[digest] = lab.id
        included_by_image[img.id].append(lab)

    for img in db.query(TrainingImage).all():
        if img.duplicate_status in {STATUS_EXACT, STATUS_LIKELY}:
            excluded_rows.append(
                {
                    "label": None,
                    "image": img,
                    "status": EXCLUDED_DUPLICATE,
                    "reason": img.duplicate_status,
                }
            )

    image_ids = sorted(included_by_image.keys())
    splits = split_source_images(image_ids, seed=SPLIT_SEED)
    image_to_split = {}
    for split, ids in splits.items():
        for iid in ids:
            image_to_split[iid] = split

    copied = {"train": [], "val": [], "test": []}
    manifest_rows = []
    for iid, labs in included_by_image.items():
        split = image_to_split[iid]
        for lab in labs:
            dest_dir = root / split / str(int(lab.grade))
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{lab.id}_{Path(lab.crop_path).name}"
            if Path(lab.crop_path).exists():
                shutil.copy2(lab.crop_path, dest)
            copied[split].append(lab)
            manifest_rows.append(
                {
                    "split": split,
                    "grade": lab.grade,
                    "src": lab.crop_path,
                    "dest": str(dest),
                    "source_image_id": iid,
                    "label_id": lab.id,
                    "status": INCLUDED,
                    "reason": "",
                    "crop_hash": lab.crop_hash or "",
                    "content_hash": lab.image.content_hash or "",
                }
            )

    for item in excluded_rows:
        img = item.get("image")
        lab = item.get("label")
        manifest_rows.append(
            {
                "split": "",
                "grade": lab.grade if lab is not None else "",
                "src": (lab.crop_path if lab is not None else (img.path if img else "")),
                "dest": "",
                "source_image_id": (lab.image_id if lab is not None else (img.id if img else "")),
                "label_id": lab.id if lab is not None else "",
                "status": item["status"],
                "reason": item["reason"],
                "crop_hash": (lab.crop_hash if lab is not None else "") or "",
                "content_hash": (
                    (lab.image.content_hash if lab is not None else (img.content_hash if img else ""))
                    or ""
                ),
            }
        )

    man = root / "manifest.csv"
    fields = [
        "split",
        "grade",
        "src",
        "dest",
        "source_image_id",
        "label_id",
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

    stats = {
        "name": name,
        "version_number": version_number,
        "split_seed": SPLIT_SEED,
        "eligible_fingerprint": fingerprint,
        "split_fractions_target": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_source_images_included": len(image_ids),
        "total_tooth_crops_included": sum(len(v) for v in copied.values()),
        "class_counts_overall": _class_counts(sum(copied.values(), [])),
        "train_counts": _class_counts(copied["train"]),
        "validation_counts": _class_counts(copied["val"]),
        "test_counts": _class_counts(copied["test"]),
        "n_train": len(copied["train"]),
        "n_valid": len(copied["val"]),
        "n_test": len(copied["test"]),
        "n_source_train": len(splits["train"]),
        "n_source_val": len(splits["val"]),
        "n_source_test": len(splits["test"]),
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
    yaml_path = dataset_path / "portal_train.yaml"
    yaml_path.write_text(text, encoding="utf-8")
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
    if check["issues"]:
        raise ValueError("Dataset failed duplicate/leakage validation: " + "; ".join(check["issues"][:8]))
    stats = json.loads(latest.statistics_json) if latest.statistics_json else {}
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
        metrics = None
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
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

    for other in db.query(ModelVersion).filter(ModelVersion.kind == "icdas"):
        other.is_active = False
    m.is_active = bool(active)
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
