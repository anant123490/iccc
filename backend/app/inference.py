"""
ICDAS 0-4 inference engine.

Production configuration:
    - 5 ICDAS classes: 0, 1, 2, 3, 4
    - MobileNetV3Small + CBAM
    - 224x224 RGB input
    - 5-class softmax output
    - Optional Grad-CAM
    - Confidence threshold support

Expected model output:
    (None, 5)

Class mapping:
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

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("icdas.inference")


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
    Inference engine for ICDAS 0-4 classification.

    The current production model is a 5-class softmax model.

    Input:
        (1, 224, 224, 3)

    Output:
        (1, 5)
    """

    _instance: Optional["InferenceEngine"] = None

    def __init__(
        self,
        model_path: str,
        num_classes: int = NUM_CLASSES,
        image_size: int = IMAGE_SIZE,
        ordinal_regression: bool = False,
        confidence_threshold: float = 0.55,
        use_roi: bool = False,
        use_clahe: bool = False,
        use_specular: bool = False,
        color_norm: bool = False,
        **kwargs,
    ):

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
        # VALIDATION
        # ----------------------------------------------------

        if self.num_classes != 5:
            raise ValueError(
                "ICDAS inference supports exactly 5 classes: "
                "ICDAS 0-4."
            )

        if self.image_size <= 0:
            raise ValueError(
                f"Invalid image size: {self.image_size}"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        self._load_model(self.model_path)

        logger.info(
            "=============================================="
        )
        logger.info(
            "ICDAS INFERENCE ENGINE INITIALIZED"
        )
        logger.info(
            "Model: %s",
            self.model_path,
        )
        logger.info(
            "Classes: ICDAS 0-4"
        )
        logger.info(
            "Input size: %dx%d",
            self.image_size,
            self.image_size,
        )
        logger.info(
            "Ordinal regression config: %s",
            self.ordinal_regression,
        )
        logger.info(
            "Confidence threshold: %.2f",
            self.confidence_threshold,
        )
        logger.info(
            "Color normalization: %s",
            self.color_norm,
        )
        logger.info(
            "=============================================="
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self, path: str):

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"ICDAS model not found:\n{model_path}"
            )

        logger.info(
            "Loading trained ICDAS model from %s",
            model_path,
        )

        # ----------------------------------------------------
        # CUSTOM OBJECTS
        # ----------------------------------------------------

        custom_objects = {}

        try:

            from src.model import get_custom_objects

            custom_objects.update(
                get_custom_objects()
            )

            logger.info(
                "Custom model objects loaded."
            )

        except Exception as exc:

            logger.warning(
                "Could not load custom objects: %s",
                exc,
            )

        # ----------------------------------------------------
        # LOAD KERAS MODEL
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
                f"Unable to load ICDAS model:\n"
                f"{model_path}\n\n"
                f"Error: {exc}"
            ) from exc

        # ----------------------------------------------------
        # VALIDATE OUTPUT
        # ----------------------------------------------------

        output_shape = self.model.output_shape

        logger.info(
            "Loaded model output shape: %s",
            output_shape,
        )

        if isinstance(output_shape, list):

            raise ValueError(
                "Multi-output models are not supported "
                "by the current production inference pipeline."
            )

        if not isinstance(output_shape, tuple):

            raise ValueError(
                f"Unsupported model output shape: "
                f"{output_shape}"
            )

        if len(output_shape) < 2:

            raise ValueError(
                f"Invalid model output shape: "
                f"{output_shape}"
            )

        output_units = output_shape[-1]

        if output_units is None:

            raise ValueError(
                "Model output dimension is undefined."
            )

        output_units = int(output_units)

        # ----------------------------------------------------
        # 5 CLASS SOFTMAX
        # ----------------------------------------------------

        if output_units == 5:

            self.detected_ordinal = False

            logger.info(
                "Detected 5-class softmax model."
            )

        # ----------------------------------------------------
        # ORDINAL MODEL
        # ----------------------------------------------------

        elif output_units == 4:

            self.detected_ordinal = True

            logger.info(
                "Detected 4-threshold ordinal model."
            )

        # ----------------------------------------------------
        # OLD 7 CLASS MODEL
        # ----------------------------------------------------

        elif output_units == 7:

            raise ValueError(
                "A 7-class ICDAS model was loaded.\n"
                "This project supports ICDAS 0-4 only.\n"
                "Use a trained 5-class model."
            )

        else:

            raise ValueError(
                f"Unsupported model output dimension: "
                f"{output_units}.\n"
                f"Expected 5-class softmax output."
            )

        # ----------------------------------------------------
        # VALIDATE INPUT
        # ----------------------------------------------------

        input_shape = self.model.input_shape

        logger.info(
            "Loaded model input shape: %s",
            input_shape,
        )

        logger.info(
            "Model successfully loaded."
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

        if cls._instance is None:

            cls._instance = cls(
                model_path=model_path,
                **kwargs,
            )

        return cls._instance

    # ========================================================
    # RESET
    # ========================================================

    @classmethod
    def reset_instance(cls):

        cls._instance = None

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:

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

        if image is None:
            raise ValueError("Image is None.")

        if image.ndim != 3:
            raise ValueError(
                f"Expected H,W,C image. "
                f"Received {image.shape}"
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

        height, width = image.shape[:2]

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
    # PREPROCESS IMAGE
    # ========================================================

    def preprocess_image(
        self,
        image: np.ndarray,
    ):

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            image = np.asarray(image)

        if image.ndim != 3:

            raise ValueError(
                f"Expected image shape (H,W,C), "
                f"got {image.shape}"
            )

        if image.shape[2] != 3:

            raise ValueError(
                f"Expected 3-channel RGB image, "
                f"got {image.shape}"
            )

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
        # OPTIONAL PREPROCESSING
        # ----------------------------------------------------

        if self.use_roi:

            processed = self.apply_roi(
                processed
            )

        if self.use_clahe:

            processed = self.apply_clahe(
                processed
            )

        if self.use_specular:

            processed = self.reduce_specular(
                processed
            )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        processed = self.resize_image(
            processed
        )

        # ----------------------------------------------------
        # COLOR NORMALIZATION
        # ----------------------------------------------------

        if self.color_norm:

            processed = self.normalize_color(
                processed
            )

        # ----------------------------------------------------
        # FLOAT NORMALIZATION
        # ----------------------------------------------------

        processed = (
            processed.astype(
                np.float32
            )
            / 255.0
        )

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
                f"Final image shape is "
                f"{processed.shape}.\n"
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

        raw = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(-1)

        if raw.shape[0] != 4:

            raise ValueError(
                "Ordinal model must return "
                "exactly 4 threshold values."
            )

        if np.any(raw < 0.0) or np.any(raw > 1.0):

            raw = 1.0 / (
                1.0 + np.exp(-raw)
            )

        raw = np.clip(
            raw,
            0.0,
            1.0,
        )

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
    # SOFTMAX
    # ========================================================

    def _softmax_probabilities(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:

        raw = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(-1)

        if raw.shape[0] != self.num_classes:

            raise ValueError(
                f"Model returned {raw.shape[0]} outputs. "
                f"Expected {self.num_classes}."
            )

        if not np.isfinite(raw).all():

            raise ValueError(
                "Model returned NaN or infinite values."
            )

        # Already probabilities
        if (
            np.all(raw >= 0.0)
            and np.all(raw <= 1.0)
            and abs(
                float(np.sum(raw)) - 1.0
            ) < 0.05
        ):

            probs = raw.copy()

        else:

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
                    "Invalid model output."
                )

            probs = (
                exp_values / total
            )

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
    # PREDICT
    # ========================================================

    def predict(
        self,
        processed: np.ndarray,
    ) -> dict:

        if processed is None:

            raise ValueError(
                "Processed image is None."
            )

        processed = np.asarray(
            processed,
            dtype=np.float32,
        )

        expected_shape = (
            self.image_size,
            self.image_size,
            3,
        )

        if processed.shape != expected_shape:

            raise ValueError(
                f"Expected {expected_shape}, "
                f"got {processed.shape}"
            )

        batch = np.expand_dims(
            processed,
            axis=0,
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
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
        # EXTRACT OUTPUT
        # ----------------------------------------------------

        if isinstance(outputs, dict):

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
                    "Unsupported dictionary model output."
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
        # PROBABILITIES
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
        # CLASS
        # ----------------------------------------------------

        grade = int(
            np.argmax(probs)
        )

        grade = int(
            np.clip(
                grade,
                0,
                4,
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
        # PROBABILITY DICTIONARY
        # ----------------------------------------------------

        probabilities = {
            str(i): round(
                float(probs[i]),
                6,
            )
            for i in range(NUM_CLASSES)
        }

        class_name = CLASS_NAMES[grade]

        logger.info(
            "Prediction=%s | confidence=%.2f%% | probabilities=%s",
            class_name,
            confidence * 100.0,
            probabilities,
        )

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
    # PREDICT IMAGE
    # ========================================================

    def predict_image(
        self,
        image: np.ndarray,
    ) -> dict:

        original_rgb, processed = (
            self.preprocess_image(image)
        )

        result = self.predict(
            processed
        )

        result["image_shape"] = list(
            original_rgb.shape
        )

        return result

    # ========================================================
    # PREDICT FILE
    # ========================================================

    def predict_file(
        self,
        image_path: str | Path,
    ) -> dict:

        image = self.load_image(
            image_path
        )

        return self.predict_image(
            image
        )

    # ========================================================
    # MODEL INFO
    # ========================================================

    def get_model_info(self) -> dict:

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
                False,
            ),
            "image_size": self.image_size,
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "use_roi": self.use_roi,
            "use_clahe": self.use_clahe,
            "use_specular": self.use_specular,
            "color_norm": self.color_norm,
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

        if processed is None:

            raise ValueError(
                "Processed image is None."
            )

        processed = np.asarray(
            processed,
            dtype=np.float32,
        )

        image = np.expand_dims(
            processed,
            axis=0,
        )

        # ----------------------------------------------------
        # FIND TARGET LAYER
        # ----------------------------------------------------

        target_layer = None

        if layer_name:

            try:

                target_layer = self.model.get_layer(
                    layer_name
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

                    shape = layer.output.shape

                    if (
                        len(shape) == 4
                        and shape[1] is not None
                        and shape[2] is not None
                    ):

                        target_layer = layer
                        break

                except Exception:
                    continue

        if target_layer is None:

            raise RuntimeError(
                "Could not find a suitable "
                "4D convolutional layer for Grad-CAM."
            )

        logger.info(
            "Grad-CAM target layer: %s",
            target_layer.name,
        )

        # ----------------------------------------------------
        # GRADIENT MODEL
        # ----------------------------------------------------

        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                target_layer.output,
                self.model.output,
            ],
        )

        # ----------------------------------------------------
        # FORWARD PASS
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

                    predictions = predictions[
                        "class"
                    ]

                elif "ordinal" in predictions:

                    predictions = predictions[
                        "ordinal"
                    ]

            predictions = tf.convert_to_tensor(
                predictions
            )

            if (
                len(predictions.shape) == 2
                and predictions.shape[-1] == 5
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
        # GRADIENTS
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
        # GLOBAL AVERAGE POOLING
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

        if float(max_value) > 0:

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
            interpolation=cv2.INTER_LINEAR,
        )

        return np.clip(
            heatmap,
            0.0,
            1.0,
        )

    # ========================================================
    # GRAD-CAM OVERLAY
    # ========================================================

    def create_gradcam_overlay(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.40,
    ) -> np.ndarray:

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
            interpolation=cv2.INTER_AREA,
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
# EXPORTS
# ============================================================

__all__ = [
    "InferenceEngine",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "CLASS_NAMES",
]