"""YOLO decay-region detector. Classes D/d only — not ICDAS, not FDI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("icdas.caries_detector")

from .config import PROJECT_ROOT

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "caries_detector" / "best.pt"
CLASS_NAMES = {0: "D", 1: "d"}
CLASS_MEANING = {
    "D": "permanent-tooth decay region",
    "d": "primary-tooth decay region",
}


class CariesDetector:
    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 320,
    ):
        self.weights = Path(weights or DEFAULT_WEIGHTS)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = None
        self.available = False
        if not self.weights.exists():
            logger.warning("Caries detector weights missing: %s", self.weights)
            return
        try:
            from ultralytics.models import YOLO

            self.model = YOLO(str(self.weights))
            self.available = True
        except Exception as exc:
            logger.exception("Could not load caries detector: %s", exc)
            self.available = False

    def predict(self, image_rgb: np.ndarray) -> list[dict]:
        if not self.available or self.model is None:
            return []
        h, w = image_rgb.shape[:2]
        results = self.model.predict(
            source=image_rgb,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        out: list[dict] = []
        if not results:
            return out
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return out
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            name = CLASS_NAMES.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": name,
                    "class_meaning": CLASS_MEANING.get(name, "decay region"),
                }
            )
        return out


_detector: Optional[CariesDetector] = None


def get_caries_detector() -> CariesDetector:
    global _detector
    if _detector is None:
        _detector = CariesDetector()
    return _detector
