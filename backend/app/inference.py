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


# ============================================================
# Add ML source directory to Python path
# ============================================================

ML_SRC = Path(__file__).resolve().parents[2] / "ml"

if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))


from src.preprocessing import preprocess_image  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402


class InferenceEngine:
    """
    Singleton inference engine for ICDAS 0–4 classification.

    The trained model uses ordinal regression:

        output shape = (batch_size, 4)
        output name  = ordinal
        output[k]    = P(y > k)

    Class probabilities are reconstructed from those thresholds.
    A 7-class checkpoint is incompatible and must not be used.
    """

    _instance: Optional["InferenceEngine"] = None

    def __init__(
        self,
        model_path: str,
        num_classes: int = 5,
        image_size: int = 224,
        ordinal_regression: bool = True,
        confidence_threshold: float = 0.55,
    ):
        self.num_classes = num_classes
        self.image_size = image_size
        self.ordinal_regression = ordinal_regression
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.gradcam = None
        self.model_path = model_path

        self._load_model(model_path)

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self, path: str):
        """Load the trained Keras model."""

        path = Path(path)

        if not path.exists():

            print(
                f"WARNING: Model not found at {path}. "
                f"Using an untrained demo model."
            )

            from src.model import build_model

            self.model = build_model(
                num_classes=self.num_classes,
                image_size=self.image_size,
                ordinal_regression=self.ordinal_regression,
            )

        else:

            print(f"Loading model from: {path}")

            try:
                # Import custom layers BEFORE loading the model.
                import src.attention  # noqa: F401
                import src.model  # noqa: F401

                self.model = tf.keras.models.load_model(
                    str(path),
                    compile=False,
                )

                print("Model loaded successfully.")

                print(
                    f"Model output names: "
                    f"{self.model.output_names}"
                )

                print(
                    f"Model output shape: "
                    f"{self.model.output_shape}"
                )

                out_dim = int(self.model.output_shape[-1])
                expected_ordinal = self.num_classes - 1
                if self.ordinal_regression and out_dim not in {
                    expected_ordinal,
                    self.num_classes,
                }:
                    raise RuntimeError(
                        f"Incompatible checkpoint output size {out_dim}. "
                        f"Expected {expected_ordinal} ordinal thresholds "
                        f"(ICDAS 0–4). Do not use a 7-class model."
                    )

            except Exception as e:

                raise RuntimeError(
                    f"Failed to load model from {path}: {e}"
                ) from e

        # Create Grad-CAM after model is loaded.
        try:
            self.gradcam = GradCAM(self.model)
            print("Grad-CAM initialized successfully.")

        except Exception as e:

            print(
                f"WARNING: Grad-CAM initialization failed: {e}"
            )

            self.gradcam = None

    # ========================================================
    # SINGLETON
    # ========================================================

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        **kwargs,
    ) -> "InferenceEngine":

        if cls._instance is None:

            cls._instance = cls(
                model_path,
                **kwargs,
            )

        return cls._instance

    # ========================================================
    # IMAGE DECODING
    # ========================================================

    def preprocess_upload(
        self,
        image_bytes: bytes,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode uploaded image and preprocess it.

        Returns:

            original_rgb:
                Original image in RGB format.

            processed:
                Model-ready RGB float32 image
                with shape:

                    (224, 224, 3)

                and values in [0, 1].
        """

        if not image_bytes:

            raise ValueError(
                "Uploaded image is empty."
            )

        # ----------------------------------------------------
        # First try OpenCV
        # ----------------------------------------------------

        nparr = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        if nparr.size == 0:

            raise ValueError(
                "Could not read uploaded image bytes."
            )

        bgr = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR,
        )

        # ----------------------------------------------------
        # Fallback to PIL
        # ----------------------------------------------------

        if bgr is None:

            try:

                pil_image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                rgb_array = np.asarray(
                    pil_image,
                    dtype=np.uint8,
                )

                if rgb_array.size == 0:

                    raise ValueError(
                        "Uploaded image contains no pixels."
                    )

                bgr = cv2.cvtColor(
                    rgb_array,
                    cv2.COLOR_RGB2BGR,
                )

            except Exception as e:

                raise ValueError(
                    f"Could not decode uploaded image: {e}"
                ) from e

        # ----------------------------------------------------
        # Validate decoded image
        # ----------------------------------------------------

        if bgr is None:

            raise ValueError(
                "OpenCV could not decode the uploaded image."
            )

        if bgr.ndim != 3:

            raise ValueError(
                f"Expected 3-channel image, got shape "
                f"{bgr.shape}."
            )

        if bgr.shape[0] <= 0 or bgr.shape[1] <= 0:

            raise ValueError(
                "Uploaded image has invalid dimensions."
            )

        # ----------------------------------------------------
        # Convert BGR -> RGB for frontend/explainability
        # ----------------------------------------------------

        original_rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Run shared preprocessing
        # ----------------------------------------------------

        try:

            processed = preprocess_image(
                bgr,
                target_size=self.image_size,
            )

        except Exception as e:

            raise ValueError(
                f"Image preprocessing failed: {e}"
            ) from e

        # ----------------------------------------------------
        # Validate processed image
        # ----------------------------------------------------

        if processed is None:

            raise ValueError(
                "Preprocessing returned None."
            )

        expected_shape = (
            self.image_size,
            self.image_size,
            3,
        )

        if processed.shape != expected_shape:

            raise ValueError(
                f"Invalid processed image shape. "
                f"Expected {expected_shape}, "
                f"got {processed.shape}."
            )

        if not np.isfinite(processed).all():

            raise ValueError(
                "Processed image contains NaN or infinite values."
            )

        # Ensure float32
        processed = processed.astype(
            np.float32,
            copy=False,
        )

        # Ensure values are in [0, 1]
        processed = np.clip(
            processed,
            0.0,
            1.0,
        )

        return original_rgb, processed

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        processed: np.ndarray,
    ) -> dict:
        """
        Run ICDAS 0–4 classification.

        Ordinal models return 4 sigmoid thresholds. Softmax models return
        5 class probabilities. 7-class checkpoints are rejected.
        """
        from src.losses import ordinal_to_class_probabilities

        if processed is None:
            raise ValueError("Processed image is None.")

        if not isinstance(processed, np.ndarray):
            processed = np.asarray(processed, dtype=np.float32)

        if processed.ndim != 3:
            raise ValueError(
                f"Expected processed image with 3 dimensions "
                f"(H, W, C), got {processed.shape}."
            )

        expected_shape = (self.image_size, self.image_size, 3)
        if processed.shape != expected_shape:
            raise ValueError(
                f"Expected processed image shape {expected_shape}, "
                f"got {processed.shape}."
            )

        batch = np.expand_dims(processed, axis=0).astype(np.float32)

        try:
            outputs = self.model.predict(batch, verbose=0)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}") from e

        if isinstance(outputs, dict):
            if "ordinal" in outputs:
                raw = np.asarray(outputs["ordinal"][0], dtype=np.float32)
                is_ordinal = True
            elif "class" in outputs:
                raw = np.asarray(outputs["class"][0], dtype=np.float32)
                is_ordinal = False
            else:
                raise ValueError(
                    "Model returned dictionary output without 'ordinal' or 'class'. "
                    f"Available outputs: {list(outputs.keys())}"
                )
        else:
            raw = np.asarray(outputs[0], dtype=np.float32)
            is_ordinal = raw.shape[-1] == (self.num_classes - 1)

        if raw.ndim != 1:
            raw = raw.reshape(-1)

        if not np.isfinite(raw).all():
            raise ValueError("Model returned NaN or infinite values.")

        if is_ordinal:
            if raw.shape[0] != self.num_classes - 1:
                raise ValueError(
                    f"Ordinal model returned {raw.shape[0]} thresholds, "
                    f"expected {self.num_classes - 1} for ICDAS 0–4."
                )
            probs = ordinal_to_class_probabilities(raw)[0]
            grade = int(np.argmax(probs))
        else:
            if raw.shape[0] != self.num_classes:
                raise ValueError(
                    f"Model returned {raw.shape[0]} class probabilities, "
                    f"but expected {self.num_classes} (ICDAS 0–4). "
                    "A previous 7-class checkpoint cannot be used."
                )
            probs = np.clip(raw, 0.0, 1.0)
            total = float(np.sum(probs))
            if total > 0:
                probs = probs / total
            grade = int(np.argmax(probs))

        grade = int(np.clip(grade, 0, self.num_classes - 1))
        confidence = float(probs[grade])
        low_confidence = confidence < self.confidence_threshold

        return {
            "icdas_grade": grade,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                str(i): round(float(probs[i]), 6) for i in range(self.num_classes)
            },
            "low_confidence": low_confidence,
            "low_confidence_message": (
                "Low confidence prediction. Professional examination recommended."
                if low_confidence
                else None
            ),
        }

    # ========================================================
    # GRAD-CAM
    # ========================================================

    def explain(
        self,
        processed: np.ndarray,
        original_rgb: np.ndarray,
        class_idx: int,
    ) -> dict:
        """
        Generate Grad-CAM heatmap, overlay and lesion contour.
        """

        if self.gradcam is None:

            raise RuntimeError(
                "Grad-CAM is not available."
            )

        # ----------------------------------------------------
        # Validate class
        # ----------------------------------------------------

        if class_idx < 0 or class_idx >= self.num_classes:

            raise ValueError(
                f"Invalid class index: {class_idx}. "
                f"Expected 0-{self.num_classes - 1}."
            )

        # ----------------------------------------------------
        # Compute heatmap
        # ----------------------------------------------------

        try:

            heatmap = self.gradcam.compute_heatmap(
                processed,
                class_idx=class_idx,
            )

        except Exception as e:

            raise RuntimeError(
                f"Grad-CAM heatmap generation failed: {e}"
            ) from e

        if heatmap is None:

            raise RuntimeError(
                "Grad-CAM returned an empty heatmap."
            )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        try:

            overlay = self.gradcam.overlay_heatmap(
                processed,
                heatmap,
            )

        except Exception as e:

            raise RuntimeError(
                f"Grad-CAM overlay generation failed: {e}"
            ) from e

        # ----------------------------------------------------
        # Lesion contours
        # ----------------------------------------------------

        try:

            mask, contours = (
                self.gradcam.extract_lesion_contour(
                    heatmap
                )
            )

            contour_img = (
                self.gradcam.draw_contours_on_image(
                    processed,
                    contours,
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"Lesion contour generation failed: {e}"
            ) from e

        # ----------------------------------------------------
        # Return encoded images
        # ----------------------------------------------------

        return {
            "heatmap": self._encode_image(
                overlay
            ),

            "overlay": self._encode_image(
                overlay
            ),

            "contour": self._encode_image(
                contour_img
            ),

            "heatmap_raw": heatmap.tolist(),
        }

    # ========================================================
    # IMAGE ENCODING
    # ========================================================

    @staticmethod
    def _encode_image(
        image: np.ndarray,
    ) -> str:
        """
        Convert RGB NumPy image to base64 PNG.
        """

        if image is None:

            raise ValueError(
                "Cannot encode empty image."
            )

        image = np.asarray(
            image
        )

        if image.size == 0:

            raise ValueError(
                "Cannot encode an empty image."
            )

        # ----------------------------------------------------
        # Convert float image to uint8
        # ----------------------------------------------------

        if np.issubdtype(
            image.dtype,
            np.floating,
        ):

            if image.max() <= 1.0:

                image = (
                    image * 255.0
                ).clip(
                    0,
                    255,
                ).astype(
                    np.uint8
                )

            else:

                image = image.clip(
                    0,
                    255,
                ).astype(
                    np.uint8
                )

        else:

            image = image.astype(
                np.uint8
            )

        # ----------------------------------------------------
        # Handle grayscale
        # ----------------------------------------------------

        if image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )

        # ----------------------------------------------------
        # Handle RGBA
        # ----------------------------------------------------

        elif (
            image.ndim == 3
            and image.shape[2] == 4
        ):

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB,
            )

        # ----------------------------------------------------
        # Validate channels
        # ----------------------------------------------------

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):

            raise ValueError(
                f"Expected RGB image with shape "
                f"(H, W, 3), got {image.shape}."
            )

        # ----------------------------------------------------
        # RGB -> BGR for OpenCV PNG encoding
        # ----------------------------------------------------

        bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        success, buffer = cv2.imencode(
            ".png",
            bgr,
        )

        if not success:

            raise ValueError(
                "OpenCV failed to encode image as PNG."
            )

        return base64.b64encode(
            buffer.tobytes()
        ).decode(
            "utf-8"
        )