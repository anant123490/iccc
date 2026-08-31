"""Shared paths and helpers for the ICDAS dataset-preparation toolkit.

This module does not assign ICDAS grades. Annotation class names from a
public dataset are treated as region labels only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ICDAS_CLASSES = ("0", "1", "2", "3", "4")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CROP_CSV_COLUMNS = [
    "crop_id",
    "filename",
    "source_image",
    "annotation_id",
    "annotation_class",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
]

LABEL_CSV_COLUMNS = [
    "crop_id",
    "filename",
    "source_image",
    "icdas_grade",
]

QUALITY_CSV_COLUMNS = [
    "crop_id",
    "filename",
    "source_image",
    "issue",
    "details",
    "kept",
]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dataset_class_dirs(dataset_root: Path | None = None) -> Path:
    """Create dataset/train|val|test/0-4. Never creates a second hierarchy."""
    root = Path(dataset_root) if dataset_root else project_path("data", "icdas")
    for split in SPLITS:
        for grade in ICDAS_CLASSES:
            ensure_dir(root / split / grade)
    return root


def ensure_pipeline_dirs() -> None:
    ensure_dataset_class_dirs()
    ensure_dir(project_path("tools"))
    ensure_dir(project_path("data", "tooth_crops", "generated", "images"))
    ensure_dir(project_path("data", "icdas", "annotations", "labeling_studio"))
    ensure_dir(project_path("reports"))
    ensure_dir(project_path("models"))


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if is_image_file(p)]
    files.sort()
    return files


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image, overwrite: bool = False) -> bool:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        return False
    ext = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        ok, encoded = cv2.imencode(".jpg", image)
        path = path.with_suffix(".jpg")
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    x1 = int(round(max(0, min(width - 1, x1))))
    y1 = int(round(max(0, min(height - 1, y1))))
    x2 = int(round(max(0, min(width, x2))))
    y2 = int(round(max(0, min(height, y2))))
    return x1, y1, x2, y2


def box_is_valid(x1: int, y1: int, x2: int, y2: int, min_size: int = 20) -> tuple[bool, str]:
    if x2 <= x1 or y2 <= y1:
        return False, "non_positive_extent"
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return False, f"tiny_crop:{w}x{h}"
    return True, "ok"


def extreme_aspect_ratio(width: int, height: int, max_ratio: float = 8.0) -> bool:
    if width <= 0 or height <= 0:
        return True
    ratio = max(width / height, height / width)
    return ratio > max_ratio


def is_blank_image(image, std_threshold: float = 3.0) -> bool:
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if float(np.std(gray)) < std_threshold:
        return True
    mean = float(np.mean(gray))
    return mean < 2.0 or mean > 253.0


def unique_crop_id(source_stem: str, annotation_id: str, used: set[str]) -> str:
    base = f"{source_stem}_{annotation_id}"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
    candidate = safe
    n = 1
    while candidate in used:
        n += 1
        candidate = f"{safe}_{n}"
    used.add(candidate)
    return candidate


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def load_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty_frame(columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(columns) + [c for c in df.columns if c not in columns]]


def save_csv(path: Path, df: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    if columns is not None:
        cols = list(columns)
        for col in cols:
            if col not in out.columns:
                out[col] = pd.NA
        extra = [c for c in out.columns if c not in cols]
        out = out[cols + extra]
    out.to_csv(path, index=False)
