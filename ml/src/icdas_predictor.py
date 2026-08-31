"""ICDAS 0–4 classifier for tooth crops (MobileNetV3 + CBAM).

Production weights live at `models/icdas/current/deploy.keras` (5-class softmax).
Historical stale ordinal files are not a valid production default.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

DEFAULT_DEPLOY = PROJECT_ROOT / "models" / "icdas" / "current" / "deploy.keras"
DEFAULT_BEST = PROJECT_ROOT / "models" / "icdas" / "current" / "best.keras"
IMAGE_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ("ICDAS 0", "ICDAS 1", "ICDAS 2", "ICDAS 3", "ICDAS 4")


def resolve_icdas_weights() -> Path:
    if DEFAULT_DEPLOY.exists():
        return DEFAULT_DEPLOY
    if DEFAULT_BEST.exists():
        return DEFAULT_BEST
    raise FileNotFoundError(
        "No approved ICDAS 5-class softmax model at "
        f"{DEFAULT_DEPLOY} or {DEFAULT_BEST}. "
        "The historical stale ordinal checkpoint is not used for production."
    )


def _custom_objects() -> dict:
    from src.attention import CBAM, ChannelAttention, SEBlock, SpatialAttention

    return {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "SEBlock": SEBlock,
    }


@dataclass
class IcdasPrediction:
    crop_name: str
    predicted_class: int
    class_name: str
    confidence: float
    prob_0: float
    prob_1: float
    prob_2: float
    prob_3: float
    prob_4: float


class IcdasCropClassifier:
    """Load production keras weights once; predict on BGR uint8 crops or files."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        import tensorflow as tf

        from src.preprocessing import preprocess_image

        self.model_path = Path(model_path) if model_path else resolve_icdas_weights()
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self._preprocess = preprocess_image
        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False,
            custom_objects=_custom_objects(),
        )
        if self.model.input_shape != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"unexpected input_shape {self.model.input_shape}")
        out = self.model.output_shape
        units = int(out[-1])
        if units == NUM_CLASSES:
            self.ordinal = False
        elif units == NUM_CLASSES - 1:
            raise ValueError(
                "4-output ordinal ICDAS checkpoints are not a valid production "
                "classifier. Deploy a 5-class softmax at models/icdas/current/."
            )
        else:
            raise ValueError(f"expected 5-class softmax (None, 5), got {out}")

    def preprocess_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._preprocess(
            image_bgr,
            target_size=IMAGE_SIZE,
            use_roi=False,
            use_clahe=False,
            use_specular=False,
            color_norm=False,
        )

    def predict_processed_batch(self, batch: np.ndarray) -> np.ndarray:
        from src.losses import ordinal_to_class_probabilities

        raw = self.model.predict(batch, verbose=0)
        probs = np.asarray(raw, dtype=np.float32)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if self.ordinal:
            return ordinal_to_class_probabilities(probs)
        probs = np.clip(probs, 0.0, 1.0)
        sums = probs.sum(axis=1, keepdims=True)
        sums = np.where(sums <= 0, 1.0, sums)
        return probs / sums

    def predict_bgr(self, image_bgr: np.ndarray, crop_name: str = "crop.jpg") -> IcdasPrediction:
        x = self.preprocess_bgr(image_bgr)
        probs = self.predict_processed_batch(np.expand_dims(x, 0))[0]
        return self._pack(crop_name, probs)

    def _pack(self, crop_name: str, probs: np.ndarray) -> IcdasPrediction:
        grade = int(np.argmax(probs))
        grade = int(np.clip(grade, 0, 4))
        conf = float(probs[grade])
        return IcdasPrediction(
            crop_name=crop_name,
            predicted_class=grade,
            class_name=CLASS_NAMES[grade],
            confidence=round(conf, 6),
            prob_0=round(float(probs[0]), 6),
            prob_1=round(float(probs[1]), 6),
            prob_2=round(float(probs[2]), 6),
            prob_3=round(float(probs[3]), 6),
            prob_4=round(float(probs[4]), 6),
        )
