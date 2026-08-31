"""ICDAS v2 labeling/dataset helpers.

Does not map d/D or any detection class to ICDAS. Does not modify dataset/,
models/icdas_mobilenet_cbam*, or fdi_detection_dataset/images/selected/.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ICDAS = PROJECT_ROOT / "data" / "icdas" / "labeling_v2"
PUBLIC_RAW_CANDIDATES = [
    PROJECT_ROOT / "data_external" / "detection" / "raw",
    PROJECT_ROOT / "data_external" / "detection",
]
PERSONAL_420 = PROJECT_ROOT / "fdi_detection_dataset" / "images" / "selected"

ALLOWED_GRADES = {"0", "1", "2", "3", "4", "SKIP"}
TRAIN_GRADES = {"0", "1", "2", "3", "4"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SOURCE_MANIFEST_COLUMNS = [
    "source_id",
    "source_type",
    "original_path",
    "filename",
    "license_source",
    "width",
    "height",
    "format",
    "sha256",
    "status",
]

LABEL_COLUMNS = [
    "sample_id",
    "source_type",
    "source_image",
    "crop_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "icdas_grade",
    "status",
    "label_timestamp",
]

CROP_POOL_COLUMNS = [
    "sample_id",
    "source_type",
    "source_image",
    "crop_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "sha256",
    "width",
    "height",
    "status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_icdas_dirs() -> None:
    for p in [
        DATA_ICDAS / "manifest",
        DATA_ICDAS / "user_images",
        DATA_ICDAS / "crops",
        DATA_ICDAS / "final" / "train",
        DATA_ICDAS / "final" / "val",
        DATA_ICDAS / "final" / "test",
    ]:
        p.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for g in "01234":
            (DATA_ICDAS / "final" / split / g).mkdir(parents=True, exist_ok=True)
    headers = {
        DATA_ICDAS / "source_manifest.csv": SOURCE_MANIFEST_COLUMNS,
        DATA_ICDAS / "manifest" / "crop_pool.csv": CROP_POOL_COLUMNS,
        DATA_ICDAS / "manifest" / "icdas_labels.csv": LABEL_COLUMNS,
    }
    for path, cols in headers.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(cols)


def find_public_image_root() -> Path | None:
    for cand in PUBLIC_RAW_CANDIDATES:
        if not cand.exists():
            continue
        n = 0
        for p in cand.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                n += 1
                if n >= 10:
                    return cand
    return None


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def image_meta(path: Path) -> tuple[int, int, str] | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(w), int(h), path.suffix.lower().lstrip(".")


def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    return df[columns]


def save_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = ""
    out[columns].to_csv(path, index=False)


def source_manifest_path() -> Path:
    return DATA_ICDAS / "source_manifest.csv"


def labels_path() -> Path:
    return DATA_ICDAS / "manifest" / "icdas_labels.csv"


def crop_pool_path() -> Path:
    return DATA_ICDAS / "manifest" / "crop_pool.csv"


def is_valid_grade(value: str) -> bool:
    return str(value).strip() in ALLOWED_GRADES


def persist_label(row: dict) -> None:
    ensure_icdas_dirs()
    path = labels_path()
    df = load_csv(path, LABEL_COLUMNS)
    grade = str(row.get("icdas_grade", "")).strip()
    if not is_valid_grade(grade):
        raise ValueError(f"Invalid ICDAS grade: {grade}")
    if grade in {"5", "6"}:
        raise ValueError("ICDAS 5 and 6 are out of scope")
    sample_id = str(row["sample_id"])
    status = "skipped" if grade == "SKIP" else "labelled"
    payload = {c: str(row.get(c, "") or "") for c in LABEL_COLUMNS}
    payload["icdas_grade"] = grade
    payload["status"] = status
    payload["label_timestamp"] = utc_now()
    mask = df["sample_id"].astype(str) == sample_id
    if mask.any():
        for k, v in payload.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    save_csv(path, df, LABEL_COLUMNS)


def group_key(row: pd.Series) -> str:
    src = str(row.get("source_image") or "").strip()
    if src:
        return f"source:{Path(src).name}"
    return f"sample:{row.get('sample_id')}"


def stable_split(key: str, seed: int, train_r: float, val_r: float) -> str:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    if u < train_r:
        return "train"
    if u < train_r + val_r:
        return "val"
    return "test"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
