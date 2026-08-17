"""
Image preprocessing for intraoral dental photos.
Includes ROI detection, CLAHE, specular reduction, and normalization.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def detect_mouth_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect mouth region using skin-tone heuristic and contour analysis.
    Returns (x, y, w, h) bounding box.
    """
    h, w = image.shape[:2]
    # Fallback: center crop 80% if detection fails
    default = (int(w * 0.1), int(h * 0.15), int(w * 0.8), int(h * 0.7))

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Skin/mucosa range (approximate)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return default
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        if bw * bh < 0.05 * w * h:
            return default
        pad = 0.05
        x = max(0, int(x - bw * pad))
        y = max(0, int(y - bh * pad))
        bw = min(w - x, int(bw * (1 + 2 * pad)))
        bh = min(h - y, int(bh * (1 + 2 * pad)))
        return (x, y, bw, bh)
    except Exception:
        return default


def crop_mouth(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Crop image to mouth ROI."""
    if bbox is None:
        bbox = detect_mouth_roi(image)
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def reduce_specular_reflection(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    """Inpaint bright specular highlights common in intraoral photos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if np.sum(mask) < 100:
        return image
    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def color_normalize(image: np.ndarray) -> np.ndarray:
    """Simple color normalization to reduce lighting variance."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_mean, l_std = lab[:, :, 0].mean(), lab[:, :, 0].std() + 1e-6
    lab[:, :, 0] = (lab[:, :, 0] - l_mean) / l_std * 32 + 128
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(
    image: np.ndarray,
    target_size: int = 224,
    use_roi: bool = True,
    use_clahe: bool = True,
    use_specular: bool = True,
    color_norm: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for training and inference.
    Returns RGB float32 array normalized to [0, 1].
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if use_roi:
        cropped = crop_mouth(image)
        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            image = cropped
        # else: degenerate ROI crop (zero width/height) — keep full image instead
    if use_specular:
        image = reduce_specular_reflection(image)
    if use_clahe:
        image = apply_clahe(image)
    if color_norm:
        image = color_normalize(image)

    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Preprocessing produced an empty image; check the uploaded photo.")
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def preprocess_from_path(path: str, **kwargs) -> np.ndarray:
    """Load and preprocess image from file path."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return preprocess_image(image, **kwargs)
