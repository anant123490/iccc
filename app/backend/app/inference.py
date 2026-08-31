"""
Production ICDAS 0-4 inference engine.

MODEL CONTRACT
--------------

Production model (5-class softmax only, when approved):
    models/icdas/current/deploy.keras

Historical 4-output ordinal checkpoints under
    models/icdas/historical/stale_ordinal_4output/
must never be loaded as the production ICDAS classifier.

Expected model:
    Input  : (None, 224, 224, 3)
    Output : (None, 5)  # 5-class softmax only

Classes:
    0 -> ICDAS 0
    1 -> ICDAS 1
    2 -> ICDAS 2
    3 -> ICDAS 3
    4 -> ICDAS 4

Ordinal (4-output) checkpoints are not used in production.

IMPORTANT INPUT SCALE
---------------------

The MobileNetV3Small model is built with:

    include_preprocessing=True

and ImageNet pretrained weights.

Therefore the model expects image pixels in:

    [0, 255]

NOT:

    [0, 1]

Training pipeline:
    image
       ↓
    RGB [0,1]
       ↓
    * 255
       ↓
    MobileNetV3
       ↓
    softmax

Inference pipeline:
    upload
       ↓
    RGB
       ↓
    resize 224x224
       ↓
    float32 [0,255]
       ↓
    MobileNetV3
       ↓
    softmax
"""

from __future__ import annotations

import base64
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

from .config import PROJECT_ROOT, BACKEND_DIR

ML_DIR = PROJECT_ROOT / "ml"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "icdas.api"
)


# ============================================================
# CONSTANTS
# ============================================================

NUM_CLASSES = 5

IMAGE_SIZE = 224

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
    Production ICDAS 0-4 inference engine.
    """

    _instance: Optional[
        "InferenceEngine"
    ] = None

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
        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.model_path = str(
            model_path
        )

        self.num_classes = int(
            num_classes
        )

        self.image_size = int(
            image_size
        )

        self.ordinal_regression = bool(
            ordinal_regression
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        # IMPORTANT:
        # These are OFF because training config uses them OFF.
        self.use_roi = bool(
            use_roi
        )

        self.use_clahe = bool(
            use_clahe
        )

        self.use_specular = bool(
            use_specular
        )

        self.color_norm = bool(
            color_norm
        )

        self.model = None

        self.detected_ordinal = False

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if self.num_classes != 5:
            raise ValueError(
                "This project supports "
                "ICDAS 0-4 only."
            )

        if self.image_size != 224:
            raise ValueError(
                "Production model expects "
                "image_size=224."
            )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must "
                "be between 0 and 1."
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self._load_model(
            self.model_path
        )

        logger.info(
            "========================================"
        )

        logger.info(
            "ICDAS INFERENCE ENGINE INITIALIZED"
        )

        logger.info(
            "Model       : %s",
            self.model_path,
        )

        logger.info(
            "Input       : %s",
            self.model.input_shape,
        )

        logger.info(
            "Output      : %s",
            self.model.output_shape,
        )

        logger.info(
            "Classes     : 5 (ICDAS 0-4)"
        )

        logger.info(
            "Input range : [0,255]"
        )

        logger.info(
            "ROI         : %s",
            self.use_roi,
        )

        logger.info(
            "CLAHE       : %s",
            self.use_clahe,
        )

        logger.info(
            "Specular    : %s",
            self.use_specular,
        )

        logger.info(
            "Color norm  : %s",
            self.color_norm,
        )

        logger.info(
            "Confidence  : %.2f",
            self.confidence_threshold,
        )

        logger.info(
            "========================================"
        )


    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(
        self,
        path: str,
    ) -> None:

        model_path = Path(
            path
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"ICDAS model not found: "
                f"{model_path}"
            )

        logger.info(
            "Loading model from %s",
            model_path,
        )

        # ----------------------------------------------------
        # Explicitly register CBAM
        # ----------------------------------------------------

        custom_objects = {}

        try:
            from src.attention import (
                CBAM,
                ChannelAttention,
                SpatialAttention,
                SEBlock,
            )

            custom_objects.update(
                {
                    "CBAM": CBAM,
                    "ChannelAttention":
                        ChannelAttention,
                    "SpatialAttention":
                        SpatialAttention,
                    "SEBlock":
                        SEBlock,
                }
            )

        except Exception as exc:
            raise RuntimeError(
                "Unable to import CBAM "
                "custom objects: "
                f"{exc}"
            ) from exc

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        try:
            self.model = (
                tf.keras.models.load_model(
                    str(model_path),
                    compile=False,
                    custom_objects=custom_objects,
                )
            )

        except Exception as exc:

            logger.exception(
                "Failed to load ICDAS model."
            )

            raise RuntimeError(
                f"Unable to load model "
                f"{model_path}: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        input_shape = (
            self.model.input_shape
        )

        if input_shape != (
            None,
            224,
            224,
            3,
        ):
            raise ValueError(
                "Unexpected model input shape: "
                f"{input_shape}. "
                "Expected "
                "(None, 224, 224, 3)."
            )

        # ----------------------------------------------------
        # Output validation
        # ----------------------------------------------------

        output_shape = (
            self.model.output_shape
        )

        if isinstance(
            output_shape,
            list,
        ):
            raise ValueError(
                "Production model must have "
                "one output. "
                f"Got: {output_shape}"
            )

        output_units = int(
            output_shape[-1]
        )

        if output_units == 5:

            self.detected_ordinal = False

            logger.info(
                "Detected 5-class softmax model."
            )

        elif output_units == 4:

            raise ValueError(
                "4-output ordinal model detected. "
                "Production inference requires a "
                "5-class softmax model "
                "(output shape (None, 5))."
            )

        elif output_units == 7:

            raise ValueError(
                "7-output ICDAS model detected. "
                "ICDAS 5 and 6 are unsupported."
            )

        else:

            raise ValueError(
                f"Unsupported model output size: "
                f"{output_units}. "
                "Expected 5."
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


    @classmethod
    def reset_instance(
        cls,
    ) -> None:

        cls._instance = None


    # ========================================================
    # IMAGE DECODING
    # ========================================================

    def _decode_image_bytes(
        self,
        content: bytes,
    ) -> np.ndarray:

        if not content:
            raise ValueError(
                "Image content is empty."
            )

        array = np.frombuffer(
            content,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Could not decode uploaded image."
            )

        return image


    # ========================================================
    # PREPROCESS UPLOAD
    # ========================================================

    def preprocess_upload(
        self,
        content: bytes,
    ):
        """
        Decode uploaded bytes and preprocess.

        Returns:

            original_rgb
            processed

        processed is float32 [0,255].
        """

        image_bgr = (
            self._decode_image_bytes(
                content
            )
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return self.preprocess_image(
            image_rgb
        )


    # ========================================================
    # LOAD IMAGE FILE
    # ========================================================

    def load_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:

        path = Path(
            image_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )


    # ========================================================
    # RESIZE
    # ========================================================

    def resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        if image is None:
            raise ValueError(
                "Image is None."
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected HWC image. "
                f"Got {image.shape}"
            )

        if image.shape[-1] != 3:
            raise ValueError(
                "Expected 3-channel image."
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
    # OPTIONAL ROI
    # ========================================================

    def apply_roi(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        height, width = (
            image.shape[:2]
        )

        margin_x = int(
            width * 0.05
        )

        margin_y = int(
            height * 0.05
        )

        cropped = image[
            margin_y:
            height - margin_y,
            margin_x:
            width - margin_x,
        ]

        if cropped.size == 0:
            return image

        return cropped


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

        l, a, b = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        l = clahe.apply(
            l
        )

        result = cv2.merge(
            [l, a, b]
        )

        return cv2.cvtColor(
            result,
            cv2.COLOR_LAB2RGB,
        )


    # ========================================================
    # SPECULAR
    # ========================================================

    def reduce_specular(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        mask = (
            gray > 245
        )

        if not np.any(mask):
            return image

        blurred = cv2.GaussianBlur(
            image,
            (5, 5),
            0,
        )

        result = image.copy()

        result[mask] = (
            blurred[mask]
        )

        return result


    # ========================================================
    # COLOR NORMALIZATION
    # ========================================================

    def normalize_color(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        image_float = (
            image.astype(
                np.float32
            )
        )

        mean = image_float.mean(
            axis=(0, 1),
            keepdims=True,
        )

        std = image_float.std(
            axis=(0, 1),
            keepdims=True,
        )

        std = np.maximum(
            std,
            1.0,
        )

        normalized = (
            (
                image_float
                - mean
            )
            / std
            * 32.0
            + 128.0
        )

        return np.clip(
            normalized,
            0,
            255,
        ).astype(
            np.uint8
        )


    # ========================================================
    # COMPLETE PREPROCESSING
    # ========================================================

    def preprocess_image(
        self,
        image: np.ndarray,
    ):
        """
        Prepare image for MobileNetV3.

        IMPORTANT:
            final output = float32 [0,255]
        """

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        image = np.asarray(
            image
        )

        if image.ndim != 3:
            raise ValueError(
                f"Expected image with "
                f"shape (H,W,3), "
                f"got {image.shape}"
            )

        if image.shape[-1] != 3:
            raise ValueError(
                f"Expected 3 channels, "
                f"got {image.shape}"
            )

        # ----------------------------------------------------
        # Convert to uint8
        # ----------------------------------------------------

        if image.dtype != np.uint8:

            image = np.asarray(
                image,
                dtype=np.float32,
            )

            # If image is normalized.
            if (
                image.max() <= 1.0
                and image.min() >= 0.0
            ):
                image *= 255.0

            image = np.clip(
                image,
                0.0,
                255.0,
            ).astype(
                np.uint8
            )

        original_rgb = (
            image.copy()
        )

        processed = (
            image.copy()
        )

        # ----------------------------------------------------
        # Optional preprocessing
        # ----------------------------------------------------

        if self.use_roi:
            processed = (
                self.apply_roi(
                    processed
                )
            )

        if self.use_specular:
            processed = (
                self.reduce_specular(
                    processed
                )
            )

        if self.use_clahe:
            processed = (
                self.apply_clahe(
                    processed
                )
            )

        if self.color_norm:
            processed = (
                self.normalize_color(
                    processed
                )
            )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        processed = (
            self.resize_image(
                processed
            )
        )

        # ----------------------------------------------------
        # CRITICAL:
        #
        # DO NOT divide by 255 here.
        #
        # Model expects [0,255].
        # ----------------------------------------------------

        processed = (
            processed.astype(
                np.float32
            )
        )

        processed = np.clip(
            processed,
            0.0,
            255.0,
        )

        processed = (
            np.ascontiguousarray(
                processed,
                dtype=np.float32,
            )
        )

        expected_shape = (
            self.image_size,
            self.image_size,
            3,
        )

        if processed.shape != (
            expected_shape
        ):

            raise ValueError(
                f"Unexpected processed "
                f"shape: {processed.shape}; "
                f"expected {expected_shape}"
            )

        return (
            original_rgb,
            processed,
        )


    # ========================================================
    # SOFTMAX PROBABILITIES
    # ========================================================

    def _get_probabilities(
        self,
        raw: np.ndarray,
    ) -> np.ndarray:

        raw = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(-1)

        if raw.shape[0] != 5:

            raise ValueError(
                f"Expected 5 class outputs. "
                f"Received {raw.shape[0]}."
            )

        if not np.isfinite(
            raw
        ).all():

            raise ValueError(
                "Model returned NaN "
                "or infinite values."
            )

        # Because model output layer is:
        #
        # Dense(5, activation="softmax")
        #
        # raw should already be probabilities.

        clipped = np.clip(
            raw,
            0.0,
            1.0,
        )

        total = float(
            clipped.sum()
        )

        if total <= 0:
            raise ValueError(
                "Probability sum is zero."
            )

        return (
            clipped / total
        ).astype(
            np.float32
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        processed: np.ndarray,
    ) -> dict:
        """
        Predict ICDAS 0-4.
        """

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

        if processed.shape != (
            expected_shape
        ):

            raise ValueError(
                f"Expected processed "
                f"shape {expected_shape}; "
                f"got {processed.shape}"
            )

        if not np.isfinite(
            processed
        ).all():

            raise ValueError(
                "Processed image contains "
                "NaN or infinite values."
            )

        minimum = float(
            processed.min()
        )

        maximum = float(
            processed.max()
        )

        # ----------------------------------------------------
        # CRITICAL RANGE CHECK
        # ----------------------------------------------------

        if (
            minimum < 0.0
            or maximum > 255.0
        ):

            raise ValueError(
                "MobileNetV3 input must "
                "be in [0,255]. "
                f"Got min={minimum:.4f}, "
                f"max={maximum:.4f}"
            )

        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        batch = np.expand_dims(
            processed,
            axis=0,
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        try:

            outputs = (
                self.model.predict(
                    batch,
                    verbose=0,
                )
            )

        except Exception as exc:

            logger.exception(
                "Model prediction failed."
            )

            raise RuntimeError(
                f"Model prediction failed: "
                f"{exc}"
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

            else:

                raise ValueError(
                    "Expected softmax "
                    "'class' output."
                )

        else:

            outputs = np.asarray(
                outputs,
                dtype=np.float32,
            )

            if outputs.ndim == 1:

                raw = outputs

            elif outputs.ndim == 2:

                raw = outputs[0]

            else:

                raise ValueError(
                    f"Unexpected output shape: "
                    f"{outputs.shape}"
                )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = (
            self._get_probabilities(
                raw
            )
        )

        # ----------------------------------------------------
        # Grade
        # ----------------------------------------------------

        grade = int(
            np.argmax(
                probabilities
            )
        )

        grade = int(
            np.clip(
                grade,
                0,
                4,
            )
        )

        confidence = float(
            probabilities[grade]
        )

        low_confidence = (
            confidence
            < self.confidence_threshold
        )

        probability_dict = {
            str(i): round(
                float(
                    probabilities[i]
                ),
                6,
            )
            for i in range(5)
        }

        logger.info(
            "Prediction: ICDAS %d | "
            "confidence %.2f%% | "
            "probabilities=%s",
            grade,
            confidence * 100.0,
            probability_dict,
        )

        return {
            "icdas_grade": grade,

            "class_name": CLASS_NAMES[
                grade
            ],

            "confidence": round(
                confidence * 100.0,
                2,
            ),

            "probabilities":
                probability_dict,

            "low_confidence":
                low_confidence,

            "low_confidence_message": (
                "Low confidence prediction. "
                "Professional dental "
                "examination recommended."
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

        original, processed = (
            self.preprocess_image(
                image
            )
        )

        result = self.predict(
            processed
        )

        result[
            "image_shape"
        ] = list(
            original.shape
        )

        return result


    # ========================================================
    # PREDICT FILE
    # ========================================================

    def predict_file(
        self,
        image_path: str | Path,
    ) -> dict:

        image = (
            self.load_image(
                image_path
            )
        )

        return self.predict_image(
            image
        )


    # ========================================================
    # MODEL INFO
    # ========================================================

    def get_model_info(
        self,
    ) -> dict:

        return {
            "model_path":
                self.model_path,

            "model_exists":
                Path(
                    self.model_path
                ).exists(),

            "input_shape":
                str(
                    self.model.input_shape
                ),

            "output_shape":
                str(
                    self.model.output_shape
                ),

            "num_classes":
                5,

            "icdas_mode":
                "0-4",

            "model_type":
                (
                    "ordinal"
                    if self.detected_ordinal
                    else "5-class softmax"
                ),

            "image_size":
                224,

            "confidence_threshold":
                self.confidence_threshold,

            "model_input_range":
                "0-255",

            "preprocessing": {
                "roi":
                    self.use_roi,

                "clahe":
                    self.use_clahe,

                "specular_reduction":
                    self.use_specular,

                "color_normalization":
                    self.color_norm,
            },

            "class_names":
                CLASS_NAMES,
        }


    # ========================================================
    # GRAD-CAM
    # ========================================================

    def generate_gradcam(
        self,
        processed: np.ndarray,
        layer_name: str | None = None,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM for the given ICDAS class (0-4).

        If class_idx is omitted, the softmax argmax is used.
        """

        processed = np.asarray(
            processed,
            dtype=np.float32,
        )

        batch = np.expand_dims(
            processed,
            axis=0,
        )

        # ----------------------------------------------------
        # Find target layer
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
                    f"Grad-CAM layer "
                    f"'{layer_name}' "
                    "not found."
                ) from exc

        else:

            # Prefer layers with 4D outputs.
            for layer in reversed(
                self.model.layers
            ):

                try:

                    output_shape = (
                        layer.output.shape
                    )

                    if (
                        len(output_shape)
                        == 4
                    ):

                        target_layer = (
                            layer
                        )

                        break

                except Exception:

                    continue

        if target_layer is None:

            raise RuntimeError(
                "No suitable convolutional "
                "layer found for Grad-CAM."
            )

        # ----------------------------------------------------
        # Gradient model
        # ----------------------------------------------------

        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                target_layer.output,
                self.model.output,
            ],
        )

        # ----------------------------------------------------
        # Forward + gradient
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                grad_model(
                    batch,
                    training=False,
                )
            )

            if isinstance(
                predictions,
                dict,
            ):

                predictions = (
                    predictions.get(
                        "class",
                        next(
                            iter(
                                predictions.values()
                            )
                        ),
                    )
                )

            n_outputs = int(
                predictions.shape[-1]
            )

            if class_idx is None:
                class_index = tf.argmax(
                    predictions[0]
                )
            else:
                safe = min(
                    max(int(class_idx), 0),
                    max(n_outputs - 1, 0),
                )
                class_index = tf.constant(
                    safe,
                    dtype=tf.int64,
                )

            class_score = (
                predictions[
                    0,
                    class_index,
                ]
            )

        gradients = tape.gradient(
            class_score,
            conv_outputs,
        )

        if gradients is None:

            raise RuntimeError(
                "Could not calculate "
                "Grad-CAM gradients."
            )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        pooled_gradients = (
            tf.reduce_mean(
                gradients,
                axis=(0, 1, 2),
            )
        )

        conv_outputs = (
            conv_outputs[0]
        )

        heatmap = tf.reduce_sum(
            conv_outputs
            * pooled_gradients,
            axis=-1,
        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        maximum = tf.reduce_max(
            heatmap
        )

        if float(maximum) > 0:

            heatmap = (
                heatmap
                / maximum
            )

        heatmap = (
            heatmap.numpy()
        )

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

        heatmap_uint8 = (
            np.uint8(
                np.clip(
                    heatmap,
                    0.0,
                    1.0,
                )
                * 255
            )
        )

        colored = (
            cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET,
            )
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

        return cv2.addWeighted(
            original,
            1.0 - alpha,
            colored,
            alpha,
            0,
        )


    # ========================================================
    # EXPLAIN
    # ========================================================

    def explain(
        self,
        processed: np.ndarray,
        original_rgb: np.ndarray,
        predicted_grade: int,
    ) -> dict:

        heatmap = (
            self.generate_gradcam(
                processed,
                class_idx=int(
                    predicted_grade
                ),
            )
        )

        overlay = (
            self.create_gradcam_overlay(
                original_rgb,
                heatmap,
            )
        )

        # ----------------------------------------------------
        # Heatmap image
        # ----------------------------------------------------

        heatmap_uint8 = (
            np.uint8(
                np.clip(
                    heatmap,
                    0.0,
                    1.0,
                )
                * 255
            )
        )

        heatmap_color = (
            cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET,
            )
        )

        # ----------------------------------------------------
        # Contour
        # ----------------------------------------------------

        threshold_mask = (
            np.uint8(
                heatmap > 0.50
            )
            * 255
        )

        contours, _ = (
            cv2.findContours(
                threshold_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )

        contour_image = cv2.resize(
            original_rgb,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_AREA,
        ).copy()

        cv2.drawContours(
            contour_image,
            contours,
            -1,
            (255, 0, 0),
            2,
        )

        # ----------------------------------------------------
        # Encode images
        # ----------------------------------------------------

        _, heatmap_encoded = (
            cv2.imencode(
                ".png",
                heatmap_color,
            )
        )

        _, overlay_encoded = (
            cv2.imencode(
                ".png",
                cv2.cvtColor(
                    overlay,
                    cv2.COLOR_RGB2BGR,
                ),
            )
        )

        _, contour_encoded = (
            cv2.imencode(
                ".png",
                cv2.cvtColor(
                    contour_image,
                    cv2.COLOR_RGB2BGR,
                ),
            )
        )

        return {
            "heatmap": (
                base64.b64encode(
                    heatmap_encoded
                ).decode(
                    "utf-8"
                )
            ),

            "overlay": (
                base64.b64encode(
                    overlay_encoded
                ).decode(
                    "utf-8"
                )
            ),

            "contour": (
                base64.b64encode(
                    contour_encoded
                ).decode(
                    "utf-8"
                )
            ),

            "predicted_grade":
                int(
                    predicted_grade
                ),
        }


    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> dict:

        loaded = (
            self.model is not None
        )

        return {
            "status":
                (
                    "healthy"
                    if loaded
                    else "unhealthy"
                ),

            "model_loaded":
                loaded,

            "model_path":
                self.model_path,

            "model_exists":
                Path(
                    self.model_path
                ).exists(),

            "num_classes":
                5,

            "icdas_mode":
                "0-4",

            "input_shape":
                (
                    str(
                        self.model.input_shape
                    )
                    if loaded
                    else None
                ),

            "output_shape":
                (
                    str(
                        self.model.output_shape
                    )
                    if loaded
                    else None
                ),

            "model_input_range":
                "0-255",
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