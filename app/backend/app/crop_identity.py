"""Tooth-crop identity: exact SHA-256 plus a tight perceptual near-duplicate flag.

Normalization for exact hash (no resize):
1. Decode crop file with OpenCV (BGR) or accept an RGB array.
2. Convert to 8-bit RGB, 3 channels (grayscale is stacked).
3. Require a C-contiguous uint8 HWC array.
4. SHA-256 over ASCII prefix ``rgb8|{h}|{w}|3|`` plus raw pixel bytes.

This ignores JPEG file-container differences after decode. It does not resize,
histogram-match, or otherwise warp geometry, so a different angle or lighting
of the same physical tooth is not treated as an exact duplicate.

Near-duplicates use 8x8 average-hash with Hamming distance 0 (identical
perceptual hash, different SHA-256). They are never classified as exact
duplicates. This is intentionally tight so a different photograph of the
same physical tooth is not removed as a duplicate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from .image_phash import average_hash_hex, hamming_hex

CROP_UNIQUE = "UNIQUE"
CROP_DUP_EXACT = "DUPLICATE_EXACT"
CROP_POSSIBLE = "POSSIBLE_DUPLICATE"
CROP_CONFLICT = "CONFLICT"

CROP_NEAR_HAMMING = 0
CONFLICT_MESSAGE = "Duplicate crop has conflicting ICDAS labels. Manual review required."

ELIGIBLE_CROP_STATUSES = {CROP_UNIQUE, None}


def rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def crop_content_sha256(rgb: np.ndarray) -> str:
    import hashlib

    arr = rgb_uint8(rgb)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    payload = f"rgb8|{h}|{w}|3|".encode("ascii") + arr.tobytes()
    return hashlib.sha256(payload).hexdigest()


def decode_crop_rgb(path: str | Path) -> np.ndarray | None:
    import cv2

    p = Path(path)
    if not p.exists():
        return None
    bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_sha256_from_path(path: str | Path) -> str | None:
    rgb = decode_crop_rgb(path)
    if rgb is None:
        return None
    return crop_content_sha256(rgb)


def fill_crop_identity_fields(lab, rgb: np.ndarray | None = None) -> str | None:
    """Set crop_hash (SHA-256) and crop_phash from pixels. Does not classify."""
    arr = rgb
    if arr is None and lab.crop_path:
        arr = decode_crop_rgb(lab.crop_path)
    if arr is None:
        return lab.crop_hash
    digest = crop_content_sha256(arr)
    lab.crop_hash = digest
    lab.crop_phash = average_hash_hex(arr)
    return digest


def _active(lab) -> bool:
    return lab.active is not False and (lab.image is None or lab.image.is_active is not False)


def sync_exact_hash_group(db: Session, digest: str | None) -> None:
    """Index lookup on crop_hash. Marks UNIQUE / DUPLICATE_EXACT / CONFLICT."""
    from .portal_db import TrainingLabel

    if not digest:
        return
    members = (
        db.query(TrainingLabel)
        .filter(TrainingLabel.crop_hash == digest)
        .order_by(TrainingLabel.id.asc())
        .all()
    )
    members = [m for m in members if _active(m)]
    if not members:
        return
    grades = sorted(
        {
            int(m.grade)
            for m in members
            if m.grade is not None and not m.skipped
        }
    )
    canon = members[0]
    if len(grades) > 1:
        for m in members:
            m.crop_duplicate_status = CROP_CONFLICT
            m.duplicate_of_label_id = None if m.id == canon.id else canon.id
        return
    canon.crop_duplicate_status = CROP_UNIQUE
    canon.duplicate_of_label_id = None
    for m in members[1:]:
        m.crop_duplicate_status = CROP_DUP_EXACT
        m.duplicate_of_label_id = canon.id


def classify_near_duplicate(db: Session, lab) -> None:
    """Flag POSSIBLE_DUPLICATE using indexed phash rows; never as exact."""
    from .portal_db import TrainingLabel

    if lab.crop_duplicate_status in {CROP_DUP_EXACT, CROP_CONFLICT}:
        return
    phash = lab.crop_phash
    if not phash:
        return
    candidates = (
        db.query(TrainingLabel.id, TrainingLabel.crop_phash)
        .filter(
            TrainingLabel.id != lab.id,
            TrainingLabel.crop_phash.isnot(None),
            TrainingLabel.crop_duplicate_status == CROP_UNIQUE,
            TrainingLabel.active.isnot(False),
        )
        .all()
    )
    best_id = None
    best_d = 99
    for other_id, other_ph in candidates:
        d = hamming_hex(phash, other_ph)
        if d <= CROP_NEAR_HAMMING and d < best_d:
            best_d = d
            best_id = other_id
    if best_id is not None:
        lab.crop_duplicate_status = CROP_POSSIBLE
        lab.duplicate_of_label_id = best_id
    elif lab.crop_duplicate_status not in {CROP_DUP_EXACT, CROP_CONFLICT}:
        if lab.crop_duplicate_status != CROP_UNIQUE:
            lab.crop_duplicate_status = CROP_UNIQUE
            if not lab.duplicate_of_label_id:
                lab.duplicate_of_label_id = None


def apply_crop_identity(db: Session, lab, rgb: np.ndarray | None = None) -> None:
    digest = fill_crop_identity_fields(lab, rgb)
    db.flush()
    sync_exact_hash_group(db, digest)
    db.flush()
    classify_near_duplicate(db, lab)


def refresh_crop_identities(db: Session) -> None:
    """Recompute hashes for active crops missing identity, then sync groups."""
    from .portal_db import TrainingLabel

    rows = db.query(TrainingLabel).order_by(TrainingLabel.id.asc()).all()
    seen_hashes: set[str] = set()
    for lab in rows:
        if not _active(lab):
            continue
        if not lab.crop_hash or not lab.crop_phash or not lab.crop_duplicate_status:
            fill_crop_identity_fields(lab)
        if lab.crop_hash:
            seen_hashes.add(lab.crop_hash)
    db.flush()
    for digest in seen_hashes:
        sync_exact_hash_group(db, digest)
    db.flush()
    for lab in rows:
        if _active(lab) and lab.crop_duplicate_status == CROP_UNIQUE:
            classify_near_duplicate(db, lab)


def crop_status_caption(lab) -> str:
    status = lab.crop_duplicate_status or CROP_UNIQUE
    grade = f"ICDAS {lab.grade}" if lab.grade is not None else "unlabeled"
    if status == CROP_DUP_EXACT:
        of = lab.duplicate_of_label_id
        extra = f" — excluded from dataset. Duplicate of crop #{of}" if of else " — excluded from dataset"
        return f"Tooth crop #{lab.id}  {grade}  DUPLICATE{extra}"
    if status == CROP_POSSIBLE:
        of = lab.duplicate_of_label_id
        extra = f" of crop #{of}" if of else ""
        return f"Tooth crop #{lab.id}  {grade}  POSSIBLE_DUPLICATE{extra} — excluded from dataset until reviewed"
    if status == CROP_CONFLICT:
        return f"Tooth crop #{lab.id}  {grade}  CONFLICT — manual review required"
    return f"Tooth crop #{lab.id}  {grade}  UNIQUE"
