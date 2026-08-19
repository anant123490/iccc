"""
Grad-CAM explainability and lesion contour extraction.

Uses PIL for heatmap resizing instead of cv2.resize()
to avoid OpenCV resize compatibility issues.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ICDAS model."""

    def __init__(
        self,
        model: keras.Model,
        layer_name: Optional[str] = None,
    ):
        self.model = model

        self.layer_name = (
            layer_name
            or self._find_target_layer()
        )

        self.grad_model = self._build_grad_model()

    # ========================================================
    # FIND TARGET CONVOLUTIONAL LAYER
    # ========================================================

    def _find_target_layer(self) -> str:
        """Locate the last Conv2D layer in the model."""

        for layer in reversed(self.model.layers):

            if isinstance(
                layer,
                layers.Conv2D,
            ):
                return layer.name

        # Fallback
        if len(self.model.layers) >= 4:
            return self.model.layers[-4].name

        raise ValueError(
            "Could not find a suitable convolutional layer."
        )

    # ========================================================
    # BUILD GRADIENT MODEL
    # ========================================================

    def _build_grad_model(self) -> keras.Model:
        """
        Create a model that returns:

        1. Last convolutional feature maps
        2. Classification predictions
        """

        target_layer = None

        for layer in reversed(
            self.model.layers
        ):

            if isinstance(
                layer,
                layers.Conv2D,
            ):
                target_layer = layer
                break

        if target_layer is None:
            raise ValueError(
                "Could not find a Conv2D layer in model."
            )

        # ----------------------------------------------------
        # Determine model output
        # ----------------------------------------------------

        model_output = self.model.output

        if isinstance(model_output, dict):
            if "ordinal" in model_output:
                class_output = model_output["ordinal"]
            elif "class" in model_output:
                class_output = model_output["class"]
            else:
                raise ValueError(
                    "Model output dictionary must contain 'ordinal' or 'class'. "
                    f"Available: {list(model_output.keys())}"
                )
        else:
            class_output = model_output

        # ----------------------------------------------------
        # Build gradient model
        # ----------------------------------------------------

        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                target_layer.output,
                class_output,
            ],
        )

        return grad_model

    # ========================================================
    # COMPUTE HEATMAP
    # ========================================================

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Input:

            image:
                (H, W, 3)

            OR:

                (1, H, W, 3)

        Expected dtype:

            float32

        Expected range:

            [0, 1]

        Returns:

            2D heatmap normalized to [0, 1].
        """

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        if image.size == 0:
            raise ValueError(
                "GradCAM received an empty image."
            )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        if image.ndim == 3:

            img_batch = np.expand_dims(
                image,
                axis=0,
            )

        elif image.ndim == 4:

            img_batch = image

        else:

            raise ValueError(
                f"Invalid image shape for GradCAM: "
                f"{image.shape}"
            )

        # ----------------------------------------------------
        # Make contiguous
        # ----------------------------------------------------

        img_batch = np.ascontiguousarray(
            img_batch,
            dtype=np.float32,
        )

        img_tensor = tf.convert_to_tensor(
            img_batch,
            dtype=tf.float32,
        )

        # ----------------------------------------------------
        # Forward pass + gradient
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                self.grad_model(
                    img_tensor,
                    training=False,
                )
            )

            # ------------------------------------------------
            # Determine target class
            # ------------------------------------------------

            n_outputs = int(predictions.shape[-1])
            is_ordinal = n_outputs == 4 or (
                hasattr(self.model, "output_names")
                and "ordinal" in list(self.model.output_names)
            )

            if class_idx is None:
                if is_ordinal:
                    class_idx = int(
                        tf.reduce_sum(
                            tf.cast(predictions[0] >= 0.5, tf.int32)
                        ).numpy()
                    )
                else:
                    class_idx = int(tf.argmax(predictions[0]).numpy())
            else:
                class_idx = int(class_idx)

            # Class score used for Grad-CAM.
            # Softmax: the selected class logit/probability.
            # Ordinal (K-1 thresholds): differentiable score for class k
            #   k = 0     -> 1 - P(y > 0)
            #   0 < k < K -> P(y > k-1) - P(y > k)  (last class: P(y > K-2))
            if is_ordinal:
                k = class_idx
                last = n_outputs - 1
                if k <= 0:
                    loss = 1.0 - predictions[:, 0]
                elif k >= n_outputs:
                    loss = predictions[:, last]
                else:
                    loss = predictions[:, k - 1] - predictions[:, k]
            else:
                safe_idx = min(max(class_idx, 0), n_outputs - 1)
                loss = predictions[:, safe_idx]

        # ----------------------------------------------------
        # Calculate gradients
        # ----------------------------------------------------

        grads = tape.gradient(
            loss,
            conv_outputs,
        )

        if grads is None:

            raise ValueError(
                "GradCAM could not calculate gradients."
            )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2),
        )

        conv_outputs = conv_outputs[0]

        # ----------------------------------------------------
        # Weighted feature maps
        # ----------------------------------------------------

        heatmap = (
            conv_outputs
            @ pooled_grads[
                ...,
                tf.newaxis,
            ]
        )

        heatmap = tf.squeeze(
            heatmap
        ).numpy()

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        heatmap = np.maximum(
            heatmap,
            0,
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        max_value = float(
            heatmap.max()
        )

        if max_value > 0:

            heatmap = (
                heatmap
                / max_value
            )

        else:

            heatmap = np.zeros_like(
                heatmap,
                dtype=np.float32,
            )

        return np.asarray(
            heatmap,
            dtype=np.float32,
        )

    # ========================================================
    # SAFE HEATMAP RESIZE
    # ========================================================

    @staticmethod
    def _resize_heatmap(
        heatmap: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Resize heatmap using PIL.

        This intentionally avoids cv2.resize().
        """

        if heatmap is None:
            raise ValueError(
                "Heatmap is None."
            )

        heatmap = np.asarray(
            heatmap,
            dtype=np.float32,
        )

        if heatmap.size == 0:
            raise ValueError(
                "Heatmap is empty."
            )

        if heatmap.ndim != 2:
            raise ValueError(
                f"Expected 2D heatmap, "
                f"got shape {heatmap.shape}"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid resize dimensions: "
                f"{width}x{height}"
            )

        # ----------------------------------------------------
        # Convert [0,1] -> [0,255]
        # ----------------------------------------------------

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0,
        )

        heatmap_uint8 = (
            heatmap * 255.0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # PIL resize
        # ----------------------------------------------------

        pil_heatmap = Image.fromarray(
            heatmap_uint8,
            mode="L",
        )

        pil_heatmap = pil_heatmap.resize(
            (
                int(width),
                int(height),
            ),
            Image.Resampling.BILINEAR,
        )

        resized = np.asarray(
            pil_heatmap,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Back to [0,1]
        # ----------------------------------------------------

        resized /= 255.0

        return np.ascontiguousarray(
            resized,
            dtype=np.float32,
        )

    # ========================================================
    # OVERLAY HEATMAP
    # ========================================================

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on image.

        Returns RGB uint8 image.
        """

        image = np.asarray(
            image
        )

        if image.size == 0:
            raise ValueError(
                "Image for heatmap overlay is empty."
            )

        # ----------------------------------------------------
        # Convert image to uint8
        # ----------------------------------------------------

        if image.dtype != np.uint8:

            if image.max() <= 1.0:

                display = (
                    image * 255.0
                ).clip(
                    0,
                    255,
                ).astype(
                    np.uint8
                )

            else:

                display = (
                    image.clip(
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )
                )

        else:

            display = image.copy()

        # ----------------------------------------------------
        # Validate image shape
        # ----------------------------------------------------

        if display.ndim != 3:
            raise ValueError(
                f"Expected RGB image, "
                f"got shape {display.shape}"
            )

        if display.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel image, "
                f"got shape {display.shape}"
            )

        height, width = display.shape[:2]

        # ----------------------------------------------------
        # Resize heatmap WITHOUT cv2.resize
        # ----------------------------------------------------

        heatmap_resized = GradCAM._resize_heatmap(
            heatmap,
            width,
            height,
        )

        # ----------------------------------------------------
        # Convert heatmap to uint8
        # ----------------------------------------------------

        heatmap_uint8 = (
            heatmap_resized * 255.0
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Apply OpenCV colormap
        # ----------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            colormap,
        )

        # OpenCV gives BGR
        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Ensure same shape
        # ----------------------------------------------------

        if heatmap_color.shape != display.shape:

            raise ValueError(
                "Heatmap and image dimensions do not match: "
                f"{heatmap_color.shape} vs {display.shape}"
            )

        # ----------------------------------------------------
        # Blend
        # ----------------------------------------------------

        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )

        overlay = cv2.addWeighted(
            display,
            1.0 - alpha,
            heatmap_color,
            alpha,
            0,
        )

        return overlay

    # ========================================================
    # EXTRACT LESION CONTOURS
    # ========================================================

    @staticmethod
    def extract_lesion_contour(
        heatmap: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 50,
    ) -> Tuple[np.ndarray, list]:
        """
        Extract lesion contours from Grad-CAM heatmap.

        Returns:

            binary mask
            filtered contours
        """

        heatmap = np.asarray(
            heatmap,
            dtype=np.float32,
        )

        if heatmap.size == 0:
            raise ValueError(
                "Heatmap is empty."
            )

        if heatmap.ndim != 2:
            raise ValueError(
                f"Expected 2D heatmap, "
                f"got {heatmap.shape}"
            )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        threshold = float(
            np.clip(
                threshold,
                0.0,
                1.0,
            )
        )

        binary = (
            heatmap >= threshold
        ).astype(
            np.uint8
        ) * 255

        # ----------------------------------------------------
        # Morphological cleanup
        # ----------------------------------------------------

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # ----------------------------------------------------
        # Find contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # ----------------------------------------------------
        # Filter small contours
        # ----------------------------------------------------

        filtered = [
            contour
            for contour in contours
            if cv2.contourArea(contour)
            >= min_area
        ]

        # ----------------------------------------------------
        # Create mask
        # ----------------------------------------------------

        height, width = heatmap.shape

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        if filtered:

            cv2.drawContours(
                mask,
                filtered,
                -1,
                255,
                -1,
            )

        return mask, filtered

    # ========================================================
    # DRAW CONTOURS
    # ========================================================

    def draw_contours_on_image(
        self,
        image: np.ndarray,
        contours: list,
        color: Tuple[int, int, int] = (
            0,
            255,
            128,
        ),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw lesion contours on image.

        Input image is expected to be RGB.
        Output image is RGB uint8.
        """

        image = np.asarray(
            image
        )

        if image.size == 0:
            raise ValueError(
                "Image is empty."
            )

        # ----------------------------------------------------
        # Convert to uint8
        # ----------------------------------------------------

        if image.dtype != np.uint8:

            if image.max() <= 1.0:

                out = (
                    image * 255.0
                ).clip(
                    0,
                    255,
                ).astype(
                    np.uint8
                )

            else:

                out = image.clip(
                    0,
                    255,
                ).astype(
                    np.uint8
                )

        else:

            out = image.copy()

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if out.ndim != 3 or out.shape[2] != 3:

            raise ValueError(
                f"Expected RGB image with 3 channels, "
                f"got {out.shape}"
            )

        # ----------------------------------------------------
        # Convert RGB color to BGR
        # ----------------------------------------------------

        bgr_color = (
            int(color[2]),
            int(color[1]),
            int(color[0]),
        )

        # ----------------------------------------------------
        # Draw contours
        # ----------------------------------------------------

        if contours:

            cv2.drawContours(
                out,
                contours,
                -1,
                bgr_color,
                int(thickness),
            )

        return out