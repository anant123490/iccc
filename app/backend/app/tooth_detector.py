"""Tooth Detector V2 (YOLO whole-tooth boxes). Does not assign ICDAS or FDI."""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .storage_paths import TOOTH_V2_WEIGHTS

logger = logging.getLogger("icdas.tooth_detector")

_pipeline = None
_load_error: str | None = None

MIN_CROP_SIDE_PX = 8
DUPLICATE_IOU = 0.90


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def box_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom else 0.0


def validate_xyxy_box(
    x1: object,
    y1: object,
    x2: object,
    y2: object,
    img_w: int,
    img_h: int,
    *,
    min_side: int = MIN_CROP_SIDE_PX,
    confidence: object | None = None,
) -> tuple[int, int, int, int]:
    """Return clamped integer box or raise ValueError. Does not invent teeth."""
    if img_w < 2 or img_h < 2:
        raise ValueError("Invalid bounding box: image is too small.")
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not _finite_number(value):
            raise ValueError(f"Invalid bounding box: {name} is not a finite number.")
    if confidence is not None:
        if not _finite_number(confidence) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("Invalid bounding box: confidence must be between 0 and 1.")
    xi1, yi1, xi2, yi2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
    if xi1 >= xi2 or yi1 >= yi2:
        raise ValueError("Invalid bounding box: x1 < x2 and y1 < y2 are required.")
    xa = max(0, min(img_w - 1, xi1))
    ya = max(0, min(img_h - 1, yi1))
    xb = max(xa + 1, min(img_w, xi2))
    yb = max(ya + 1, min(img_h, yi2))
    if xb <= xa or yb <= ya:
        raise ValueError("Invalid bounding box: crop would be empty.")
    if (xb - xa) < min_side or (yb - ya) < min_side:
        raise ValueError(f"Invalid bounding box: crop is smaller than {min_side}px.")
    area = (xb - xa) * (yb - ya)
    if area < min_side * min_side:
        raise ValueError("Invalid bounding box: area is too small.")
    return xa, ya, xb, yb


def _filter_kept_detections(items: list, crops: list, img_w: int, img_h: int) -> tuple[list, list]:
    ranked = sorted(
        zip(items, crops),
        key=lambda pair: float(getattr(pair[0], "confidence", 0.0) or 0.0),
        reverse=True,
    )
    kept_items: list = []
    kept_crops: list = []
    for item, crop in ranked:
        try:
            box = validate_xyxy_box(
                item.x1,
                item.y1,
                item.x2,
                item.y2,
                img_w,
                img_h,
                confidence=item.confidence,
            )
        except ValueError:
            continue
        crop_arr = np.asarray(crop[1] if isinstance(crop, tuple) else crop)
        if crop_arr.size == 0 or crop_arr.shape[0] < MIN_CROP_SIDE_PX or crop_arr.shape[1] < MIN_CROP_SIDE_PX:
            continue
        if crop_arr.ndim != 3 or crop_arr.shape[2] < 3:
            continue
        candidate = (box[0], box[1], box[2], box[3])
        if any(
            box_iou_xyxy(candidate, (int(k.x1), int(k.y1), int(k.x2), int(k.y2))) >= DUPLICATE_IOU
            for k in kept_items
        ):
            continue
        item.x1, item.y1, item.x2, item.y2 = box
        kept_items.append(item)
        kept_crops.append(crop)
    return kept_items, kept_crops


def _ensure_ml_path() -> None:
    ml = str(PROJECT_ROOT / "ml")
    if ml not in sys.path:
        sys.path.insert(0, ml)


def detector_available() -> bool:
    return get_pipeline() is not None


def detector_error() -> str | None:
    get_pipeline()
    return _load_error


def get_pipeline():
    global _pipeline, _load_error
    if _pipeline is False:
        return None
    if _pipeline is not None:
        return _pipeline
    if not TOOTH_V2_WEIGHTS.exists():
        _load_error = f"Tooth Detector V2 weights missing: {TOOTH_V2_WEIGHTS}"
        logger.warning(_load_error)
        _pipeline = False
        return None
    try:
        _ensure_ml_path()
        from src.tooth_cropping import ToothCropPipeline

        _pipeline = ToothCropPipeline(weights=TOOTH_V2_WEIGHTS)
        _load_error = None
        logger.info("Tooth Detector V2 loaded: %s", TOOTH_V2_WEIGHTS)
        return _pipeline
    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Tooth Detector V2 unavailable: %s", exc)
        _pipeline = False
        return None


def detect_rgb(image_rgb: np.ndarray, source_name: str = "image.jpg") -> dict[str, Any]:
    pipe = get_pipeline()
    if pipe is None:
        raise RuntimeError(_load_error or "Tooth Detector V2 is not available.")
    bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    result = pipe.crop_bgr(bgr, source_name=source_name)
    overlay_rgb = None
    if result.overlay_bgr is not None:
        overlay_rgb = cv2.cvtColor(result.overlay_bgr, cv2.COLOR_BGR2RGB)
    crops = []
    for name, crop_bgr in result.crops_bgr:
        crops.append((name, cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
    kept = [it for it in result.items if it.kept]
    h, w = np.asarray(image_rgb).shape[:2]
    kept, crops = _filter_kept_detections(kept, crops, w, h)
    return {
        "n_raw": result.n_raw,
        "n_kept": len(kept),
        "overlay_rgb": overlay_rgb,
        "crops": crops,
        "items": kept,
        "mean_confidence": (
            round(sum(it.confidence for it in kept) / len(kept), 4) if kept else 0.0
        ),
    }


def crop_xyxy_rgb(image_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Slice a tooth crop from the original image. Clamps to bounds; never returns empty."""
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Invalid bounding box: original image must be RGB.")
    h, w = rgb.shape[:2]
    xa, ya, xb, yb = validate_xyxy_box(x1, y1, x2, y2, w, h)
    crop = rgb[ya:yb, xa:xb].copy()
    if crop.size == 0:
        raise ValueError("Invalid bounding box: crop would be empty.")
    return crop


def draw_boxes_rgb(image_rgb: np.ndarray, boxes: list[dict]) -> np.ndarray:
    overlay = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR).copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            overlay,
            f"#{i + 1}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
