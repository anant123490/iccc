"""Lightweight RGB intraoral image-quality checks (not a clinical device)."""

from __future__ import annotations

import cv2
import numpy as np

BLUR_VAR_MIN = 45.0
BRIGHT_MIN = 18.0
BRIGHT_MAX = 242.0


def assess_image_quality(image_rgb: np.ndarray) -> dict:
    if image_rgb is None or image_rgb.size == 0:
        return {
            "ok": False,
            "status": "invalid",
            "verdict": "FAIL",
            "message": "Image could not be decoded.",
            "sharpness": 0.0,
            "brightness": 0.0,
            "flags": ["invalid"],
            "warnings": [],
        }
    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[-1] != 3:
        return {
            "ok": False,
            "status": "invalid",
            "verdict": "FAIL",
            "message": "Expected an RGB image.",
            "sharpness": 0.0,
            "brightness": 0.0,
            "flags": ["invalid"],
            "warnings": [],
        }
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    reasons = []
    if sharpness < BLUR_VAR_MIN:
        reasons.append("blurry")
    if brightness < BRIGHT_MIN:
        reasons.append("too_dark")
    if brightness > BRIGHT_MAX:
        reasons.append("too_bright")
    ok = not reasons
    if img.shape[0] < 80 or img.shape[1] < 80:
        reasons.append("too_small")
        ok = False
    if not reasons:
        verdict = "PASS"
        status = "ok"
        message = "Image quality acceptable for AI-assisted screening."
    elif "too_small" in reasons:
        verdict = "FAIL"
        status = "fail"
        message = "Image is too small to analyze. Capture a larger photo."
    else:
        verdict = "WARNING"
        status = "low_quality"
        message = (
            "Low image quality ("
            + ", ".join(reasons)
            + "). Capture another photo if possible."
        )
    warnings = []
    if "blurry" in reasons:
        warnings.append("Blur")
    if "too_dark" in reasons:
        warnings.append("Dark image")
    if "too_bright" in reasons:
        warnings.append("Bright image")
    return {
        "ok": ok,
        "status": status,
        "verdict": verdict,
        "message": message,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "flags": reasons,
        "warnings": warnings,
    }
