"""
Grad-CAM explainability and lesion contour extraction.
"""

from __future__ import annotations

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ICDAS model."""

    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self.grad_model = self._build_grad_model()

    def _find_target_layer(self) -> str:
        """Locate last convolutional layer in MobileNetV3 backbone."""
        conv_layers = [
            l.name
            for l in self.model.layers
            if isinstance(l, layers.Conv2D) or "conv" in l.name.lower()
        ]
        # Prefer deep features from backbone
        for layer in reversed(self.model.layers):
            if "mobilenet" in layer.name.lower() and hasattr(layer, "layers"):
                sub_convs = [
                    sl.name
                    for sl in layer.layers
                    if isinstance(sl, keras.layers.Conv2D)
                ]
                if sub_convs:
                    return f"{layer.name}/{sub_convs[-1]}"
        if conv_layers:
            return conv_layers[-1]
        return self.model.layers[-4].name

    def _build_grad_model(self) -> keras.Model:
        """Model that outputs conv feature maps and predictions."""
        # Use intermediate backbone output
        backbone = None
        for layer in self.model.layers:
            if "mobilenet" in layer.name.lower():
                backbone = layer
                break

        if backbone is None:
            raise ValueError("Could not find MobileNet backbone in model")

        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                backbone.output,
                self.model.output["class"]
                if isinstance(self.model.output, dict)
                else self.model.output,
            ],
        )
        return grad_model

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for input image (preprocessed, batch dim optional).

        Args:
            image: (H, W, 3) or (1, H, W, 3) float32 [0,1]
            class_idx: Target class; None = predicted class
        """
        if image.ndim == 3:
            img_batch = np.expand_dims(image, axis=0)
        else:
            img_batch = image

        img_tensor = tf.constant(img_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-8
        return heatmap

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap on original RGB image (uint8 or float)."""
        if image.max() <= 1.0:
            display = (image * 255).astype(np.uint8)
        else:
            display = image.astype(np.uint8)

        h, w = display.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(display, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

    @staticmethod
    def extract_lesion_contour(
        heatmap: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 50,
    ) -> Tuple[np.ndarray, list]:
        """
        Extract lesion contours from heatmap.
        Returns binary mask and list of contours.
        """
        h, w = heatmap.shape
        binary = (heatmap >= threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, filtered, -1, 255, -1)
        return mask, filtered

    def draw_contours_on_image(
        self,
        image: np.ndarray,
        contours: list,
        color: Tuple[int, int, int] = (0, 255, 128),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw lesion contours on image."""
        if image.max() <= 1.0:
            out = (image * 255).astype(np.uint8).copy()
        else:
            out = image.copy()
        # OpenCV uses BGR
        bgr_color = (color[2], color[1], color[0])
        cv2.drawContours(out, contours, -1, bgr_color, thickness)
        return out
