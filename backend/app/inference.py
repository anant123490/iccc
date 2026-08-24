"""
ICDAS 0-4 inference engine.

Supports:
    - ICDAS 0-4 only
    - 5-class softmax models
    - Optional ordinal models
    - MobileNetV3 + CBAM custom models
    - Image preprocessing
    - Prediction probabilities
    - Confidence score
    - Low-confidence detection
    - Grad-CAM support when available

IMPORTANT:
    The current production model is:

        models/deploy.keras

    Expected output:

        (None, 5)

    Classes:

        0 -> ICDAS 0
        1 -> ICDAS 1
        2 -> ICDAS 2
        3 -> ICDAS 3
        4 -> ICDAS 4
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"


# Make ml/src importable.
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("icdas.api")


# ============================================================
# CONSTANTS
# ============================================================

NUM_CLASSES = 5
IMAGE_SIZE = 224

VALID_GRADES = (0, 1, 2, 3, 4)

CLASS_NAMES = {
    0: "ICDAS 0",
    1: "ICDAS 1",
    2: "ICDAS 2",
    3: "ICDAS 3",
    4: "ICDAS 4",
}


# ============================================================
# INFERENCE ENGINE
# ============================================================

class InferenceEngine:
    """
    Production inference engine for ICDAS 0-4.

    The constructor intentionally accepts ordinal_regression
    because main.py may pass this value from Settings.
    """

    _instance: Optional["InferenceEngine"] = None

    def __init__(
        self,
        model_path: str,
        num_classes: int = NUM_CLASSES,
        image_size: int = IMAGE_SIZE,
        ordinal_regression: bool = False,
        confidence_threshold: float = 0.50,
        use_roi: bool = False,
        use_clahe: bool = False,
        use_specular: bool = False,
        color_norm: bool = True,
        **kwargs,
    ):
        """
        Initialize inference engine.

        Parameters
        ----------
        model_path:
            Path to .keras model.

        num_classes:
            Must be 5 for ICDAS 0-4.

        image_size:
            Input image size. Current model uses 224.

        ordinal_regression:
            False for the current 5-class softmax model.

        confidence_threshold:
            Predictions below this probability are marked low confidence.

        use_roi:
            Optional ROI preprocessing.

        use_clahe:
            Optional CLAHE preprocessing.

        use_specular:
            Optional specular highlight reduction.

        color_norm:
            Optional color normalization.
        """

        self.model_path = str(model_path)

        self.num_classes = int(num_classes)

        self.image_size = int(image_size)

        self.ordinal_regression = bool(
            ordinal_regression
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.use_roi = bool(use_roi)
        self.use_clahe = bool(use_clahe)
        self.use_specular = bool(use_specular)
        self.color_norm = bool(color_norm)

        self.model = None
        self.gradcam = None

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if self.num_classes != 5:
            raise ValueError(
                "This inference engine supports ICDAS 0-4 only. "
                f"Received num_classes={self.num_classes}."
            )

        if self.image_size <= 0:
            raise ValueError(
                f"Invalid image_size={self.image_size}."
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self._load_model(self.model_path)

        logger.info(
            "Inference engine initialized."
        )

        logger.info(
            "Model: %s",
            self.model_path,
        )

        logger.info(
            "Classes: %d (ICDAS 0-4)",
            self.num_classes,
        )

        logger.info(
            "Image size: %d",
            self.image_size,
        )

        logger.info(
            "Ordinal regression: %s",
            self.ordinal_regression,
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(
        self,
        path: str,
    ):
        """
        Load trained Keras model.

        The model must output exactly 5 values for
        the current ICDAS 0-4 softmax configuration.
        """

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"ICDAS model not found: {model_path}"
            )

        logger.info(
            "Loading model from %s",
            model_path,
        )

        # ----------------------------------------------------
        # Import custom model objects
        # ----------------------------------------------------

        custom_objects = {}

        try:
            from src.model import get_custom_objects

            custom_objects.update(
                get_custom_objects()
            )

            logger.info(
                "Loaded custom model objects from src.model."
            )

        except Exception as exc:
            logger.warning(
                "Could not load custom objects from src.model: %s",
                exc,
            )

        # ----------------------------------------------------
        # Load Keras model
        # ----------------------------------------------------

        try:
            self.model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects=custom_objects,
            )

        except Exception as exc:
            logger.exception(
                "Failed to load ICDAS model."
            )

            raise RuntimeError(
                f"Unable to load ICDAS model: {model_path}. "
                f"Error: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        output_shape = self.model.output_shape

        logger.info(
            "Model output shape: %s",
            output_shape,
        )

        # ----------------------------------------------------
        # Handle single-output model
        # ----------------------------------------------------

        if isinstance(output_shape, tuple):

            if len(output_shape) < 2:
                raise ValueError(
                    f"Invalid model output shape: {output_shape}"
                )

            output_units = output_shape[-1]

            if output_units is None:
                raise ValueError(
                    "Model output dimension is undefined."
                )

            output_units = int(output_units)

            # ------------------------------------------------
            # Current model = 5-class softmax
            # ------------------------------------------------

            if output_units == 5:

                logger.info(
                    "Detected 5-class ICDAS softmax model."
                )

                self.detected_ordinal = False

            # ------------------------------------------------
            # Optional ordinal model = 4 thresholds
            # ------------------------------------------------

            elif output_units == 4:

                logger.info(
                    "Detected 4-threshold ordinal model."
                )

                self.detected_ordinal = True

            # ------------------------------------------------
            # Reject old 7-class model
            # ------------------------------------------------

            elif output_units == 7:

                raise ValueError(
                    "The loaded model is a 7-output model. "
                    "This project now supports ICDAS 0-4 only. "
                    "Use models/deploy.keras containing a 5-output model."
                )

            else:

                raise ValueError(
                    f"Unsupported model output dimension: "
                    f"{output_units}. "
                    "Expected 5 for ICDAS 0-4 softmax "
                    "or 4 for ordinal regression."
                )

        else:

            raise ValueError(
                f"Unsupported model output structure: "
                f"{output_shape}"
            )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        logger.info(
            "Final model output shape: %s",
            self.model.output_shape,
        )

    # ========================================================
    # SINGLETON
    # ========================================================

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        **kwargs,
    ) -> "InferenceEngine":
        """
        Return singleton inference engine.

        This accepts **kwargs so main.py can safely pass:

            ordinal_regression
            confidence_threshold
            preprocessing options
        """

        if cls._instance is None:

            cls._instance = cls(
                model_path=model_path,
                **kwargs,
            )

        return cls._instance

    # ========================================================
    # RESET SINGLETON
    # ========================================================

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton.

        Useful during development/testing.
        """

        cls._instance = None

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    def load_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Load image from disk.

        Returns
        -------
        RGB uint8 image.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return image

    # ========================================================
    # RESIZE
    # ========================================================

    def resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize image to model input size.
        """

        if image is None:
            raise ValueError(
                "Image is None."
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected RGB image with 3 dimensions. "
                f"Got {image.shape}."
            )

        return cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

    # ========================================================
    # CLAHE
    # ========================================================

    def apply_clahe(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply CLAHE to luminance channel.

        This is optional and disabled by default during
        inference unless enabled in Settings.
        """

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB,
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        l_channel = clahe.apply(
            l_channel
        )

        result = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel,
            )
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_LAB2RGB,
        )

        return result

    # ========================================================
    # SPECULAR REDUCTION
    # ========================================================

    def reduce_specular(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Reduce very bright specular highlights.

        This is intentionally conservative.
        """

        img = image.copy()

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY,
        )

        mask = gray > 245

        if not np.any(mask):
            return img

        blurred = cv2.GaussianBlur(
            img,
            (5, 5),
            0,
        )

        img[mask] = blurred[mask]

        return img

    # ========================================================
    # COLOR NORMALIZATION
    # ========================================================

    def normalize_color(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize image values.

        Keeps image in uint8 RGB format.
        """

        img = image.astype(
            np.float32
        )

        mean = np.mean(
            img,
            axis=(0, 1),
            keepdims=True,
        )

        std = np.std(
            img,
            axis=(0, 1),
            keepdims=True,
        )

        std = np.maximum(
            std,
            1.0,
        )

        normalized = (
            img - mean
        ) / std

        normalized = (
            normalized * 32.0
            + 128.0
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        )

        return normalized.astype(
            np.uint8
        )

    # ========================================================
    # ROI
    # ========================================================

    def apply_roi(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Conservative central ROI.

        Disabled by default because dental images can contain
        important tooth information outside the center.
        """

        height, width = image.shape[:2]

        # Keep 90% of the image.
        margin_y = int(
            height * 0.05
        )

        margin_x = int(
            width * 0.05
        )

        cropped = image[
            margin_y:height - margin_y,
            margin_x:width - margin_x,
        ]

        if cropped.size == 0:
            return image

        return cropped

    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess_image(
        self,
        image: np.ndarray,
    ):
        """
        Preprocess image for model inference.

        Returns
        -------
        original_rgb, processed_float32
        """

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            image = np.asarray(
                image
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected image shape (H,W,C), "
                f"got {image.shape}."
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image, "
                f"got {image.shape}."
            )

        # ----------------------------------------------------
        # Convert to uint8 if necessary
        # ----------------------------------------------------

        if image.dtype != np.uint8:

            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )

        original_rgb = image.copy()

        processed = image.copy()

        # ----------------------------------------------------
        # Optional ROI
        # ----------------------------------------------------

        if self.use_roi:

            processed = self.apply_roi(
                processed
            )

        # ----------------------------------------------------
        # Optional CLAHE
        # ----------------------------------------------------

        if self.use_clahe:

            processed = self.apply_clahe(
                processed
            )

        # ----------------------------------------------------
        # Optional specular reduction
        # ----------------------------------------------------

        if self.use_specular:

            processed = self.reduce_specular(
                processed
            )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        processed = self.resize_image(
            processed
        )

        # ----------------------------------------------------
        # Optional color normalization
        # ----------------------------------------------------

        if self.color_norm:

            processed = self.normalize_color(
                processed
            )

        # ----------------------------------------------------
        # Convert to float
        # ----------------------------------------------------

        processed = (
            processed.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # Final safety checks
        # ----------------------------------------------------

        processed = np.clip(
            processed,
            0.0,
            1.0,
        )

        expected_shape = (
            self.image_size,
            self.image_size,
            3,
        )

        if processed.shape != expected_shape:

            raise ValueError(
                f"Final processed image has wrong shape: "
                f"{processed.shape}. "
                f"Expected {expected_shape}."
            )

        return original_rgb, processed

    # ========================================================
    # ORDINAL CONVERSION
    # ========================================================

    def _ordinal_to_probabilities(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:
        """
        Convert ordinal threshold probabilities to
        5-class probabilities.

        Expected raw shape:

            (4,)
        """

        raw = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(-1)

        if raw.shape[0] != 4:

            raise ValueError(
                "Ordinal model must return exactly "
                "4 threshold values for ICDAS 0-4."
            )

        # Sigmoid if values are logits.
        if np.any(raw < 0.0) or np.any(raw > 1.0):

            raw = 1.0 / (
                1.0 + np.exp(-raw)
            )

        raw = np.clip(
            raw,
            0.0,
            1.0,
        )

        # Ensure monotonically decreasing cumulative
        # probabilities.
        cumulative = np.maximum.accumulate(
            raw[::-1]
        )[::-1]

        p0 = 1.0 - cumulative[0]

        p1 = (
            cumulative[0]
            - cumulative[1]
        )

        p2 = (
            cumulative[1]
            - cumulative[2]
        )

        p3 = (
            cumulative[2]
            - cumulative[3]
        )

        p4 = cumulative[3]

        probs = np.array(
            [
                p0,
                p1,
                p2,
                p3,
                p4,
            ],
            dtype=np.float32,
        )

        probs = np.clip(
            probs,
            0.0,
            1.0,
        )

        total = float(
            np.sum(probs)
        )

        if total <= 0:

            return np.ones(
                5,
                dtype=np.float32,
            ) / 5.0

        return probs / total

    # ========================================================
    # SOFTMAX SAFETY
    # ========================================================

    def _softmax_probabilities(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:
        """
        Convert model output into stable 5-class probabilities.
        """

        raw = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(-1)

        if raw.shape[0] != self.num_classes:

            raise ValueError(
                f"Model returned {raw.shape[0]} outputs, "
                f"but ICDAS 0-4 requires exactly "
                f"{self.num_classes}."
            )

        if not np.isfinite(raw).all():

            raise ValueError(
                "Model returned NaN or infinite values."
            )

        # ----------------------------------------------------
        # If already probabilities
        # ----------------------------------------------------

        if (
            np.all(raw >= 0.0)
            and np.all(raw <= 1.0)
            and abs(
                float(np.sum(raw)) - 1.0
            ) < 0.05
        ):

            probs = raw.copy()

        else:

            # Treat output as logits.
            shifted = (
                raw
                - np.max(raw)
            )

            exp_values = np.exp(
                shifted
            )

            total = float(
                np.sum(exp_values)
            )

            if total <= 0:

                raise ValueError(
                    "Invalid model probability output."
                )

            probs = (
                exp_values / total
            )

        # Final normalization.
        total = float(
            np.sum(probs)
        )

        if total <= 0:

            raise ValueError(
                "Probability sum is zero."
            )

        probs = probs / total

        return probs.astype(
            np.float32
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        processed: np.ndarray,
    ) -> dict:
        """
        Run ICDAS 0-4 prediction.

        Current production model:

            Input  -> (1, 224, 224, 3)
            Output -> (1, 5)

        Therefore prediction uses:

            np.argmax(probabilities)

        for ICDAS 0-4.
        """

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
                "Expected processed image with "
                "3 dimensions (H,W,C). "
                f"Got {processed.shape}."
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
            np.float32
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        try:

            outputs = self.model.predict(
                batch,
                verbose=0,
            )

        except Exception as exc:

            logger.exception(
                "Model prediction failed."
            )

            raise RuntimeError(
                f"Model prediction failed: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Extract output
        # ----------------------------------------------------

        if isinstance(
            outputs,
            dict,
        ):

            if "class" in outputs:

                raw = np.asarray(
                    outputs["class"][0],
                    dtype=np.float32,
                )

                detected_ordinal = False

            elif "ordinal" in outputs:

                raw = np.asarray(
                    outputs["ordinal"][0],
                    dtype=np.float32,
                )

                detected_ordinal = True

            else:

                raise ValueError(
                    "Model returned a dictionary but "
                    "no 'class' or 'ordinal' output was found. "
                    f"Outputs: {list(outputs.keys())}"
                )

        else:

            outputs_array = np.asarray(
                outputs,
                dtype=np.float32,
            )

            if outputs_array.ndim == 1:

                raw = outputs_array

            else:

                raw = outputs_array[0]

            raw = raw.reshape(-1)

            detected_ordinal = (
                raw.shape[0]
                == self.num_classes - 1
            )

        # ----------------------------------------------------
        # Validate finite values
        # ----------------------------------------------------

        if not np.isfinite(
            raw
        ).all():

            raise ValueError(
                "Model returned NaN or infinite values."
            )

        # ----------------------------------------------------
        # Convert output to probabilities
        # ----------------------------------------------------

        if detected_ordinal:

            probs = self._ordinal_to_probabilities(
                raw
            )

        else:

            probs = self._softmax_probabilities(
                raw
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        grade = int(
            np.argmax(
                probs
            )
        )

        grade = int(
            np.clip(
                grade,
                0,
                self.num_classes - 1,
            )
        )

        confidence = float(
            probs[grade]
        )

        low_confidence = (
            confidence
            < self.confidence_threshold
        )

        # ----------------------------------------------------
        # Probability dictionary
        # ----------------------------------------------------

        probabilities = {
            str(i): round(
                float(probs[i]),
                6,
            )
            for i in range(
                self.num_classes
            )
        }

        # ----------------------------------------------------
        # Human-readable class name
        # ----------------------------------------------------

        class_name = CLASS_NAMES.get(
            grade,
            f"ICDAS {grade}",
        )

        # ----------------------------------------------------
        # Log prediction
        # ----------------------------------------------------

        logger.info(
            "Prediction: %s | confidence=%.2f%% | probabilities=%s",
            class_name,
            confidence * 100.0,
            probabilities,
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "icdas_grade": grade,
            "class_name": class_name,
            "confidence": round(
                confidence * 100.0,
                2,
            ),
            "probabilities": probabilities,
            "low_confidence": low_confidence,
            "low_confidence_message": (
                "Low confidence prediction. "
                "Professional dental examination recommended."
                if low_confidence
                else None
            ),
        }

    # ========================================================
    # PREDICT FROM RGB IMAGE
    # ========================================================

    def predict_image(
        self,
        image: np.ndarray,
    ) -> dict:
        """
        Complete pipeline:

            RGB image
                ↓
            preprocessing
                ↓
            model
                ↓
            ICDAS prediction
        """

        original_rgb, processed = (
            self.preprocess_image(
                image
            )
        )

        result = self.predict(
            processed
        )

        # Include original image information.
        result["image_shape"] = list(
            original_rgb.shape
        )

        return result

    # ========================================================
    # PREDICT FROM FILE
    # ========================================================

    def predict_file(
        self,
        image_path: str | Path,
    ) -> dict:
        """
        Load image from disk and predict.
        """

        image = self.load_image(
            image_path
        )

        return self.predict_image(
            image
        )

    # ========================================================
    # GET MODEL INFO
    # ========================================================

    def get_model_info(self) -> dict:
        """
        Return model information for debugging/API.
        """

        return {
            "model_path": str(
                self.model_path
            ),
            "model_exists": Path(
                self.model_path
            ).exists(),
            "input_shape": str(
                self.model.input_shape
            ),
            "output_shape": str(
                self.model.output_shape
            ),
            "num_classes": self.num_classes,
            "icdas_mode": "0-4",
            "ordinal_regression_config": (
                self.ordinal_regression
            ),
            "detected_ordinal": getattr(
                self,
                "detected_ordinal",
                None,
            ),
            "image_size": self.image_size,
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "class_names": CLASS_NAMES,
        }

    # ========================================================
    # GRAD-CAM
    # ========================================================

    def generate_gradcam(
        self,
        processed: np.ndarray,
        layer_name: str | None = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        This is optional. If the model architecture does not
        expose a suitable convolutional layer, a clear error
        is returned.
        """

        if processed is None:

            raise ValueError(
                "Processed image is None."
            )

        image = np.expand_dims(
            processed,
            axis=0,
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Find convolutional layer
        # ----------------------------------------------------

        target_layer = None

        if layer_name:

            try:
                target_layer = (
                    self.model.get_layer(
                        layer_name
                    )
                )

            except Exception as exc:

                raise ValueError(
                    f"Grad-CAM layer '{layer_name}' "
                    f"was not found."
                ) from exc

        else:

            for layer in reversed(
                self.model.layers
            ):

                try:

                    output_shape = (
                        layer.output.shape
                    )

                    if (
                        len(output_shape) == 4
                    ):

                        target_layer = layer
                        break

                except Exception:
                    continue

        if target_layer is None:

            raise RuntimeError(
                "Could not find a suitable 4D "
                "convolutional layer for Grad-CAM."
            )

        # ----------------------------------------------------
        # Build Grad-CAM model
        # ----------------------------------------------------

        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                target_layer.output,
                self.model.output,
            ],
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                grad_model(
                    image,
                    training=False,
                )
            )

            if isinstance(
                predictions,
                dict,
            ):

                if "class" in predictions:

                    predictions = (
                        predictions["class"]
                    )

                elif "ordinal" in predictions:

                    predictions = (
                        predictions["ordinal"]
                    )

            predictions = tf.convert_to_tensor(
                predictions
            )

            # For softmax, select highest class.
            if (
                len(
                    predictions.shape
                ) == 2
                and predictions.shape[-1]
                == self.num_classes
            ):

                class_index = tf.argmax(
                    predictions[0]
                )

                class_score = predictions[
                    0,
                    class_index,
                ]

            else:

                class_score = tf.reduce_max(
                    predictions[0]
                )

        # ----------------------------------------------------
        # Gradients
        # ----------------------------------------------------

        gradients = tape.gradient(
            class_score,
            conv_outputs,
        )

        if gradients is None:

            raise RuntimeError(
                "Could not calculate Grad-CAM gradients."
            )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2),
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs
            * pooled_gradients,
            axis=-1,
        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        max_value = tf.reduce_max(
            heatmap
        )

        if float(
            max_value
        ) > 0:

            heatmap = (
                heatmap
                / max_value
            )

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (
                self.image_size,
                self.image_size,
            ),
        )

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0,
        )

        return heatmap

    # ========================================================
    # HEATMAP OVERLAY
    # ========================================================

    def create_gradcam_overlay(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.40,
    ) -> np.ndarray:
        """
        Create RGB Grad-CAM overlay.
        """

        if original_rgb is None:

            raise ValueError(
                "Original image is None."
            )

        if heatmap is None:

            raise ValueError(
                "Heatmap is None."
            )

        heatmap_uint8 = np.uint8(
            np.clip(
                heatmap,
                0.0,
                1.0,
            )
            * 255
        )

        colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        colored = cv2.cvtColor(
            colored,
            cv2.COLOR_BGR2RGB,
        )

        original = cv2.resize(
            original_rgb,
            (
                colored.shape[1],
                colored.shape[0],
            ),
        )

        overlay = cv2.addWeighted(
            original,
            1.0 - alpha,
            colored,
            alpha,
            0,
        )

        return overlay

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(self) -> dict:
        """
        Check whether the model is loaded and usable.
        """

        loaded = (
            self.model is not None
        )

        return {
            "status": (
                "healthy"
                if loaded
                else "unhealthy"
            ),
            "model_loaded": loaded,
            "model_path": str(
                self.model_path
            ),
            "model_exists": Path(
                self.model_path
            ).exists(),
            "num_classes": self.num_classes,
            "icdas_mode": "0-4",
            "input_shape": (
                str(
                    self.model.input_shape
                )
                if loaded
                else None
            ),
            "output_shape": (
                str(
                    self.model.output_shape
                )
                if loaded
                else None
            ),
        }


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "InferenceEngine",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "CLASS_NAMES",
]