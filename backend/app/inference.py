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
    Singleton inference engine for ICDAS classification.

    The deployed model returns a single softmax output:

        (batch_size, 7)

    representing ICDAS classes:

        0, 1, 2, 3, 4, 5, 6
    """

    _instance: Optional["InferenceEngine"] = None

    def __init__(
        self,
        model_path: str,
        num_classes: int = 7,
        image_size: int = 224,
    ):
        self.num_classes = num_classes
        self.image_size = image_size
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
        Run ICDAS classification.

        IMPORTANT:

        deploy.keras was inspected and confirmed to return:

            output shape = (None, 7)
            output name  = ['class']

        Therefore this method treats the model as a
        standard 7-class softmax classifier.

        ICDAS prediction:

            grade = argmax(class probabilities)
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if processed is None:

            raise ValueError(
                "Processed image is None."
            )

        if not isinstance(
            processed,
            np.ndarray,
        ):

            processed = np.asarray(
                processed,
                dtype=np.float32,
            )

        if processed.ndim != 3:

            raise ValueError(
                f"Expected processed image with 3 dimensions "
                f"(H, W, C), got {processed.shape}."
            )

        expected_shape = (
            self.image_size,
            self.image_size,
            3,
        )

        if processed.shape != expected_shape:

            raise ValueError(
                f"Expected processed image shape "
                f"{expected_shape}, "
                f"got {processed.shape}."
            )

        # ----------------------------------------------------
        # Prepare batch
        # ----------------------------------------------------

        batch = np.expand_dims(
            processed,
            axis=0,
        ).astype(
            np.float32,
        )

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        try:

            outputs = self.model.predict(
                batch,
                verbose=0,
            )

        except Exception as e:

            raise RuntimeError(
                f"Model prediction failed: {e}"
            ) from e

        # ----------------------------------------------------
        # Extract class probabilities
        # ----------------------------------------------------

        if isinstance(outputs, dict):

            if "class" not in outputs:

                raise ValueError(
                    "Model returned dictionary output, "
                    "but 'class' output was not found. "
                    f"Available outputs: "
                    f"{list(outputs.keys())}"
                )

            probs = np.asarray(
                outputs["class"][0],
                dtype=np.float32,
            )

        else:

            # Current deploy.keras uses this path.
            #
            # outputs shape:
            #
            # (1, 7)
            #
            probs = np.asarray(
                outputs[0],
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Validate probabilities
        # ----------------------------------------------------

        if probs.ndim != 1:

            probs = probs.reshape(-1)

        if probs.shape[0] != self.num_classes:

            raise ValueError(
                f"Model returned {probs.shape[0]} "
                f"class probabilities, but expected "
                f"{self.num_classes}."
            )

        if not np.isfinite(probs).all():

            raise ValueError(
                "Model returned NaN or infinite probabilities."
            )

        # ----------------------------------------------------
        # Normalize probabilities if necessary
        # ----------------------------------------------------

        probs = np.clip(
            probs,
            0.0,
            1.0,
        )

        probability_sum = float(
            np.sum(probs)
        )

        if probability_sum > 0:

            probs = probs / probability_sum

        # ----------------------------------------------------
        # ICDAS prediction
        # ----------------------------------------------------

        grade = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[grade]
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "icdas_grade": grade,

            # Percentage
            "confidence": round(
                confidence * 100,
                2,
            ),

            "probabilities": [
                round(
                    float(p),
                    6,
                )
                for p in probs
            ],
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