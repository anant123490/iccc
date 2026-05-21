"""
Model loading and inference with Grad-CAM explainability.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Add ml src to path for shared preprocessing/gradcam
ML_SRC = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(ML_SRC))

from src.preprocessing import preprocess_image  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.model import ordinal_to_class  # noqa: E402


class InferenceEngine:
    """Singleton inference engine for API."""

    _instance: Optional["InferenceEngine"] = None

    def __init__(self, model_path: str, num_classes: int = 7, image_size: int = 224):
        self.num_classes = num_classes
        self.image_size = image_size
        self.model = None
        self.gradcam = None
        self.model_path = model_path
        self._load_model(model_path)

    def _load_model(self, path: str):
        path = Path(path)
        if not path.exists():
            # Demo mode: create untrained model for API structure testing
            from src.model import build_model
            self.model = build_model(num_classes=self.num_classes, image_size=self.image_size)
            print(f"WARNING: Model not found at {path}. Using untrained demo model.")
        else:
            self.model = tf.keras.models.load_model(str(path), compile=False)
        self.gradcam = GradCAM(self.model)

    @classmethod
    def get_instance(cls, model_path: str, **kwargs) -> "InferenceEngine":
        if cls._instance is None:
            cls._instance = cls(model_path, **kwargs)
        return cls._instance

    def preprocess_upload(self, image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Decode upload and preprocess."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(bgr, target_size=self.image_size)
        return original_rgb, processed

    def predict(self, processed: np.ndarray) -> dict:
        """Run inference and return ICDAS prediction."""
        batch = np.expand_dims(processed, axis=0)
        outputs = self.model.predict(batch, verbose=0)

        if isinstance(outputs, dict):
            if "ordinal" in outputs:
                ordinal_probs = outputs["ordinal"][0]
                grade = int(ordinal_to_class(np.expand_dims(ordinal_probs, 0)).numpy()[0])
            else:
                grade = int(np.argmax(outputs["class"][0]))
            probs = outputs.get("class", outputs["ordinal"])[0]
        else:
            probs = outputs[0]
            grade = int(np.argmax(probs))

        confidence = float(np.max(probs) if len(probs) == self.num_classes else np.mean(probs))
        return {
            "icdas_grade": grade,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs.tolist() if hasattr(probs, "tolist") else list(probs),
        }

    def explain(self, processed: np.ndarray, original_rgb: np.ndarray, class_idx: int) -> dict:
        """Generate Grad-CAM heatmap and lesion contours."""
        heatmap = self.gradcam.compute_heatmap(processed, class_idx=class_idx)
        overlay = self.gradcam.overlay_heatmap(processed, heatmap)
        mask, contours = self.gradcam.extract_lesion_contour(heatmap)
        contour_img = self.gradcam.draw_contours_on_image(processed, contours)

        return {
            "heatmap": self._encode_image(overlay),
            "overlay": self._encode_image(overlay),
            "contour": self._encode_image(contour_img),
            "heatmap_raw": heatmap.tolist(),
        }

    @staticmethod
    def _encode_image(rgb: np.ndarray) -> str:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", bgr)
        return base64.b64encode(buf).decode("utf-8")
