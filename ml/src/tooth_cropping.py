"""Whole-tooth cropping from YOLO11n `best.pt`.

Reusable by CLI, FastAPI, and Streamlit. Does not train, does not assign
FDI or ICDAS, and does not read Batch_02 pseudo-labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "detection"
    / "tooth_detector_batch01"
    / "weights"
    / "best.pt"
)
DEFAULT_SOURCE = PROJECT_ROOT / "fdi_detection_dataset" / "images" / "selected"
DEFAULT_OUT = PROJECT_ROOT / "data" / "tooth_crops" / "generated"

MIN_SIDE_PX = 20
MAX_ASPECT = 8.0
MIN_AREA_FRAC = 0.0004
MAX_AREA_FRAC = 0.75
PAD_FRAC = 0.08
CONF_THRES = 0.25
IMGSZ = 640


@dataclass
class CropItem:
    source_image: str
    crop_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    crop_w: int
    crop_h: int
    kept: bool
    skip_reason: str = ""


@dataclass
class ImageCropResult:
    source_image: str
    overlay_name: str
    n_raw: int
    n_kept: int
    items: list[CropItem] = field(default_factory=list)
    overlay_bgr: np.ndarray | None = None
    crops_bgr: list[tuple[str, np.ndarray]] = field(default_factory=list)


def read_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    enc.tofile(str(path))


def _clamp(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1i = int(round(max(0, min(w - 1, x1))))
    y1i = int(round(max(0, min(h - 1, y1))))
    x2i = int(round(max(0, min(w, x2))))
    y2i = int(round(max(0, min(h, y2))))
    return x1i, y1i, x2i, y2i


def _pad_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    return _clamp(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def quality_check(
    crop: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    min_side: int = MIN_SIDE_PX,
) -> str:
    """Return empty string if OK, else a skip reason."""
    if x2 <= x1 or y2 <= y1:
        return "non_positive_extent"
    cw, ch = x2 - x1, y2 - y1
    if cw < min_side or ch < min_side:
        return f"tiny:{cw}x{ch}"
    ratio = max(cw / ch, ch / cw) if ch else 999.0
    if ratio > MAX_ASPECT:
        return f"extreme_aspect:{ratio:.1f}"
    area_frac = (cw * ch) / float(img_w * img_h)
    if area_frac < MIN_AREA_FRAC:
        return f"tiny_area_frac:{area_frac:.6f}"
    if area_frac > MAX_AREA_FRAC:
        return f"too_large_frac:{area_frac:.3f}"
    if crop is None or crop.size == 0:
        return "empty_crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    std = float(np.std(gray))
    mean = float(np.mean(gray))
    if std < 3.0:
        return "blank_low_std"
    if mean < 2.0 or mean > 253.0:
        return "blank_mean"
    return ""


class ToothCropPipeline:
    """Load YOLO `best.pt` once; crop teeth from BGR arrays or files."""

    def __init__(
        self,
        weights: Path | str | None = None,
        conf: float = CONF_THRES,
        imgsz: int = IMGSZ,
        device: str | None = None,
        pad: float = PAD_FRAC,
        min_side: int = MIN_SIDE_PX,
    ) -> None:
        from ultralytics import YOLO
        import torch

        self.weights = Path(weights) if weights else DEFAULT_WEIGHTS
        if not self.weights.exists():
            raise FileNotFoundError(f"missing detector weights: {self.weights}")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.pad = pad
        self.min_side = min_side
        self.model = YOLO(str(self.weights))

    def crop_bgr(self, image_bgr: np.ndarray, source_name: str = "image.jpg") -> ImageCropResult:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        overlay = image_bgr.copy()
        stem = Path(source_name).stem
        overlay_name = f"{stem}_overlay.jpg"
        items: list[CropItem] = []
        crops_bgr: list[tuple[str, np.ndarray]] = []
        raw_boxes: list[tuple[float, float, float, float, float]] = []

        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                cls = int(b.cls[0])
                if cls != 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0])
                raw_boxes.append((x1, y1, x2, y2, conf))

        raw_boxes.sort(key=lambda t: (t[0], t[1], -t[4]))
        n_raw = len(raw_boxes)
        kept_i = 0
        for x1, y1, x2, y2, conf in raw_boxes:
            px1, py1, px2, py2 = _pad_xyxy(x1, y1, x2, y2, w, h, self.pad)
            crop = image_bgr[py1:py2, px1:px2]
            reason = quality_check(crop, px1, py1, px2, py2, w, h, self.min_side)
            crop_name = f"{stem}_tooth_{kept_i:03d}.jpg"
            rec = CropItem(
                source_image=source_name,
                crop_name=crop_name if not reason else "",
                confidence=round(conf, 4),
                x1=px1,
                y1=py1,
                x2=px2,
                y2=py2,
                crop_w=max(0, px2 - px1),
                crop_h=max(0, py2 - py1),
                kept=not reason,
                skip_reason=reason,
            )
            color = (0, 255, 0) if rec.kept else (0, 0, 255)
            cv2.rectangle(overlay, (px1, py1), (px2, py2), color, 2)
            label = f"tooth {conf:.2f}" + (f" skip" if reason else "")
            cv2.putText(
                overlay,
                label,
                (px1, max(15, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            if rec.kept:
                rec.crop_name = f"{stem}_tooth_{kept_i:03d}.jpg"
                crops_bgr.append((rec.crop_name, crop.copy()))
                kept_i += 1
            items.append(rec)

        return ImageCropResult(
            source_image=source_name,
            overlay_name=overlay_name,
            n_raw=n_raw,
            n_kept=kept_i,
            items=items,
            overlay_bgr=overlay,
            crops_bgr=crops_bgr,
        )

    def crop_path(self, image_path: Path | str) -> ImageCropResult:
        path = Path(image_path)
        im = read_bgr(path)
        if im is None:
            raise RuntimeError(f"unreadable image: {path}")
        return self.crop_bgr(im, source_name=path.name)

    def save_result(self, result: ImageCropResult, out_dir: Path | str) -> None:
        out = Path(out_dir)
        img_dir = out / "images"
        ov_dir = out / "overlays"
        for name, crop in result.crops_bgr:
            write_jpg(img_dir / name, crop)
        if result.overlay_bgr is not None:
            write_jpg(ov_dir / result.overlay_name, result.overlay_bgr)


def manifest_rows(result: ImageCropResult) -> list[dict[str, Any]]:
    rows = []
    for it in result.items:
        if not it.kept:
            continue
        rows.append(
            {
                "image_name": it.source_image,
                "crop_name": it.crop_name,
                "confidence": it.confidence,
                "x1": it.x1,
                "y1": it.y1,
                "x2": it.x2,
                "y2": it.y2,
                "crop_w": it.crop_w,
                "crop_h": it.crop_h,
            }
        )
    return rows


def skipped_rows(result: ImageCropResult) -> list[dict[str, Any]]:
    rows = []
    for it in result.items:
        if it.kept:
            continue
        rows.append(
            {
                "image_name": it.source_image,
                "confidence": it.confidence,
                "x1": it.x1,
                "y1": it.y1,
                "x2": it.x2,
                "y2": it.y2,
                "skip_reason": it.skip_reason,
            }
        )
    return rows
