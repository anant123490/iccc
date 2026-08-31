"""
Image preprocessing for intraoral dental photos.

Pipeline:
1. Validate uploaded image
2. Convert to BGR uint8
3. Detect mouth ROI
4. Crop ROI safely
5. Reduce specular reflections
6. Apply CLAHE
7. Color normalization
8. Resize using PIL
9. Convert BGR -> RGB
10. Keep RGB float32 in [0, 255] for MobileNetV3
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================
# IMAGE VALIDATION
# ============================================================

def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert an image into a valid OpenCV BGR uint8 image.

    Expected final shape:
        (height, width, 3)

    Expected dtype:
        uint8
    """

    if image is None:
        raise ValueError("Image is None.")

    image = np.asarray(image)

    if image.size == 0:
        raise ValueError("Image is empty.")

    # --------------------------------------------------------
    # Grayscale -> BGR
    # --------------------------------------------------------

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    # --------------------------------------------------------
    # RGBA/BGRA -> BGR
    # --------------------------------------------------------

    elif image.ndim == 3 and image.shape[2] == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )

    # --------------------------------------------------------
    # Already BGR
    # --------------------------------------------------------

    elif image.ndim == 3 and image.shape[2] == 3:
        pass

    else:

        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    # --------------------------------------------------------
    # Convert dtype to uint8
    # --------------------------------------------------------

    if image.dtype != np.uint8:

        image = image.astype(np.float32)

        # If image is normalized [0,1]
        if image.size > 0 and image.max() <= 1.0:
            image = image * 255.0

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Make memory contiguous
    # --------------------------------------------------------

    image = np.ascontiguousarray(
        image,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:

        raise ValueError(
            f"Invalid image dimensions: {image.shape}"
        )

    return image


# ============================================================
# SAFE PIL RESIZE
# ============================================================

def resize_image(
    image: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """
    Resize image using PIL.

    PIL is intentionally used here instead of cv2.resize()
    because the OpenCV build currently installed on the
    machine is throwing:

        cv2.resize()
        (-215:Assertion failed)
        func != 0
    """

    if target_size <= 0:
        raise ValueError(
            "target_size must be greater than 0."
        )

    image = ensure_bgr_uint8(image)

    try:

        # BGR -> RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    except cv2.error as e:

        raise ValueError(
            f"Could not convert BGR image to RGB: {e}"
        )

    try:

        pil_image = Image.fromarray(
            rgb,
            mode="RGB",
        )

        pil_image = pil_image.resize(
            (
                int(target_size),
                int(target_size),
            ),
            Image.Resampling.LANCZOS,
        )

        rgb_resized = np.asarray(
            pil_image,
            dtype=np.uint8,
        )

    except Exception as e:

        raise ValueError(
            f"PIL image resize failed: {e}"
        )

    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    try:

        bgr_resized = cv2.cvtColor(
            rgb_resized,
            cv2.COLOR_RGB2BGR,
        )

    except cv2.error as e:

        raise ValueError(
            f"Could not convert resized image to BGR: {e}"
        )

    return np.ascontiguousarray(
        bgr_resized,
        dtype=np.uint8,
    )


# ============================================================
# MOUTH ROI DETECTION
# ============================================================

def detect_mouth_roi(
    image: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Detect approximate mouth region.

    Returns:

        x, y, width, height

    If detection fails, a safe center crop is returned.
    """

    image = ensure_bgr_uint8(image)

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Safe fallback ROI
    # --------------------------------------------------------

    fallback_x = max(
        0,
        int(width * 0.10),
    )

    fallback_y = max(
        0,
        int(height * 0.15),
    )

    fallback_w = max(
        1,
        int(width * 0.80),
    )

    fallback_h = max(
        1,
        int(height * 0.70),
    )

    fallback = (
        fallback_x,
        fallback_y,
        fallback_w,
        fallback_h,
    )

    try:

        # ----------------------------------------------------
        # BGR -> HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        # Approximate skin/mucosa range
        lower = np.array(
            [0, 20, 70],
            dtype=np.uint8,
        )

        upper = np.array(
            [25, 255, 255],
            dtype=np.uint8,
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper,
        )

        # ----------------------------------------------------
        # Morphological cleanup
        # ----------------------------------------------------

        kernel = np.ones(
            (15, 15),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # ----------------------------------------------------
        # Find contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return fallback

        largest = max(
            contours,
            key=cv2.contourArea,
        )

        x, y, roi_w, roi_h = cv2.boundingRect(
            largest
        )

        # Ignore extremely small regions
        if roi_w * roi_h < 0.05 * width * height:
            return fallback

        # ----------------------------------------------------
        # Add padding
        # ----------------------------------------------------

        padding = 0.05

        x = int(
            x - roi_w * padding
        )

        y = int(
            y - roi_h * padding
        )

        roi_w = int(
            roi_w * (1 + 2 * padding)
        )

        roi_h = int(
            roi_h * (1 + 2 * padding)
        )

        # ----------------------------------------------------
        # Clamp ROI to image
        # ----------------------------------------------------

        x = max(
            0,
            x,
        )

        y = max(
            0,
            y,
        )

        roi_w = min(
            roi_w,
            width - x,
        )

        roi_h = min(
            roi_h,
            height - y,
        )

        if roi_w <= 0 or roi_h <= 0:
            return fallback

        return (
            x,
            y,
            roi_w,
            roi_h,
        )

    except Exception:

        return fallback


# ============================================================
# CROP MOUTH
# ============================================================

def crop_mouth(
    image: np.ndarray,
    bbox: Optional[
        Tuple[int, int, int, int]
    ] = None,
) -> np.ndarray:
    """
    Crop image using mouth ROI.
    """

    image = ensure_bgr_uint8(image)

    height, width = image.shape[:2]

    if bbox is None:

        bbox = detect_mouth_roi(
            image
        )

    x, y, roi_w, roi_h = bbox

    # --------------------------------------------------------
    # Clamp coordinates
    # --------------------------------------------------------

    x = max(
        0,
        min(
            int(x),
            width - 1,
        ),
    )

    y = max(
        0,
        min(
            int(y),
            height - 1,
        ),
    )

    roi_w = max(
        1,
        min(
            int(roi_w),
            width - x,
        ),
    )

    roi_h = max(
        1,
        min(
            int(roi_h),
            height - y,
        ),
    )

    cropped = image[
        y:y + roi_h,
        x:x + roi_w,
    ]

    if cropped.size == 0:
        return image

    return np.ascontiguousarray(
        cropped,
        dtype=np.uint8,
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Apply CLAHE on the L channel.
    """

    image = ensure_bgr_uint8(image)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    )

    l_channel = clahe.apply(
        l_channel
    )

    merged = cv2.merge(
        [
            l_channel,
            a_channel,
            b_channel,
        ]
    )

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================
# SPECULAR REFLECTION REDUCTION
# ============================================================

def reduce_specular_reflection(
    image: np.ndarray,
    threshold: int = 220,
) -> np.ndarray:
    """
    Reduce bright reflections commonly found
    in intraoral photographs.
    """

    image = ensure_bgr_uint8(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    # Nothing significant to remove
    if np.sum(mask) < 100:
        return image

    try:

        return cv2.inpaint(
            image,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )

    except cv2.error:

        # If inpainting fails, don't break
        # the entire prediction pipeline.
        return image


# ============================================================
# COLOR NORMALIZATION
# ============================================================

def color_normalize(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize lighting using LAB color space.
    """

    image = ensure_bgr_uint8(image)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    l_channel = lab[:, :, 0]

    mean = float(
        l_channel.mean()
    )

    std = float(
        l_channel.std()
    )

    std = max(
        std,
        1e-6,
    )

    lab[:, :, 0] = (
        (l_channel - mean)
        / std
        * 32.0
        + 128.0
    )

    lab = np.clip(
        lab,
        0,
        255,
    ).astype(
        np.uint8
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(
    image: np.ndarray,
    target_size: int = 224,
    use_roi: bool = False,
    use_clahe: bool = False,
    use_specular: bool = False,
    color_norm: bool = False,
) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Input:
        BGR uint8 NumPy image.

    Output:
        RGB float32 image in [0, 255].

    This matches Keras MobileNetV3Small with
    include_preprocessing=True (ImageNet).

    Output shape:
        (target_size, target_size, 3)
    """

    # --------------------------------------------------------
    # 1. Validate original image
    # --------------------------------------------------------

    image = ensure_bgr_uint8(
        image
    )

    # --------------------------------------------------------
    # 2. ROI
    # --------------------------------------------------------

    if use_roi:

        cropped = crop_mouth(
            image
        )

        if (
            cropped is not None
            and cropped.size > 0
            and cropped.shape[0] > 0
            and cropped.shape[1] > 0
        ):

            image = cropped

    # --------------------------------------------------------
    # 3. Specular reflection
    # --------------------------------------------------------

    if use_specular:

        image = reduce_specular_reflection(
            image
        )

    # --------------------------------------------------------
    # 4. CLAHE
    # --------------------------------------------------------

    if use_clahe:

        image = apply_clahe(
            image
        )

    # --------------------------------------------------------
    # 5. Color normalization
    # --------------------------------------------------------

    if color_norm:

        image = color_normalize(
            image
        )

    # --------------------------------------------------------
    # 6. Validate before resize
    # --------------------------------------------------------

    image = ensure_bgr_uint8(
        image
    )

    # --------------------------------------------------------
    # 7. Resize
    #
    # IMPORTANT:
    # We DO NOT use cv2.resize here.
    # PIL performs the resize.
    # --------------------------------------------------------

    image = resize_image(
        image,
        target_size,
    )

    # --------------------------------------------------------
    # 8. BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # 9. RGB float32 in [0, 255]
    #
    # Do NOT divide by 255. MobileNetV3 built-in
    # preprocessing expects this range.
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )

    image = np.clip(
        image,
        0.0,
        255.0,
    )

    # --------------------------------------------------------
    # 10. Final contiguous array
    # --------------------------------------------------------

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_shape = (
        int(target_size),
        int(target_size),
        3,
    )

    if image.shape != expected_shape:

        raise ValueError(
            f"Unexpected preprocessed image shape: "
            f"{image.shape}. "
            f"Expected: {expected_shape}"
        )

    return image


# ============================================================
# PREPROCESS IMAGE FROM PATH
# ============================================================

def preprocess_from_path(
    path: str,
    **kwargs,
) -> np.ndarray:
    """
    Load image from disk and preprocess it.
    """

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise ValueError(
            f"Could not read image: {path}"
        )

    return preprocess_image(
        image,
        **kwargs,
    )