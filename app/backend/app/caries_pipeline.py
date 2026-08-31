"""
RGB → quality → decay-region detector → ICDAS per ROI → Grad-CAM.

Does not remap d/D to ICDAS. Does not assign FDI.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from .caries_detector import get_caries_detector
from .groq_service import generate_report
from .icdas_actions import get_clinical_action
from .image_quality import assess_image_quality
from .inference import InferenceEngine

PAD = 0.08
NO_DET_MSG = "No sufficiently localized dental region detected."
LOW_CONF_MSG = "Low-confidence AI prediction — please capture another image."
AI_ASSISTED = "AI-assisted assessment — not a definitive clinical diagnosis."


def _b64_png_rgb(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def _crop(image: np.ndarray, box: dict) -> np.ndarray:
    h, w = image.shape[:2]
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    px = int(bw * PAD)
    py = int(bh * PAD)
    x1 = max(0, box["x1"] - px)
    y1 = max(0, box["y1"] - py)
    x2 = min(w, box["x2"] + px)
    y2 = min(h, box["y2"] + py)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    return crop


def _annotate(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    vis = image.copy()
    for i, b in enumerate(boxes, start=1):
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (16, 185, 129), 3)
        label = f"R{i} {b['class_name']} {b['confidence']:.2f}"
        cv2.putText(
            vis,
            label,
            (b["x1"], max(20, b["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (16, 185, 129),
            2,
            cv2.LINE_AA,
        )
    return vis


def run_localized_pipeline(
    engine: InferenceEngine,
    image_rgb: np.ndarray,
    include_explainability: bool = True,
    allow_whole_image_fallback: bool = False,
) -> dict[str, Any]:
    quality = assess_image_quality(image_rgb)
    detector = get_caries_detector()
    detections = detector.predict(image_rgb) if detector.available else []
    annotated = _annotate(image_rgb, detections) if detections else image_rgb

    regions: list[dict] = []
    for i, box in enumerate(detections, start=1):
        crop = _crop(image_rgb, box)
        original, processed = engine.preprocess_image(crop)
        pred = engine.predict(processed)
        action = get_clinical_action(pred["icdas_grade"])
        explain = None
        if include_explainability:
            try:
                explain = engine.explain(
                    processed=processed,
                    original_rgb=original,
                    predicted_grade=pred["icdas_grade"],
                )
            except Exception:
                explain = None
        roi_note = None
        if pred["low_confidence"]:
            roi_note = LOW_CONF_MSG
        regions.append(
            {
                "region_id": i,
                "detection_class": box["class_name"],
                "detection_meaning": box["class_meaning"],
                "detection_confidence": box["confidence"],
                "box": {k: box[k] for k in ("x1", "y1", "x2", "y2")},
                "icdas_grade": pred["icdas_grade"],
                "icdas_class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "low_confidence": pred["low_confidence"],
                "low_confidence_message": roi_note,
                "finding": action["finding"],
                "label": action["label"],
                "heatmap_base64": None if not explain else explain.get("overlay"),
                "roi_base64": _b64_png_rgb(original),
            }
        )

    mode = "localized"
    fallback_used = False
    message = None
    primary = None

    if not detector.available:
        mode = "detector_unavailable"
        message = (
            "Caries localization weights are not loaded. "
            "Whole-image ICDAS is available only as an explicit fallback."
        )
        if allow_whole_image_fallback:
            mode = "whole_image_fallback"
            fallback_used = True
            original, processed = engine.preprocess_image(image_rgb)
            pred = engine.predict(processed)
            action = get_clinical_action(pred["icdas_grade"])
            explain = None
            if include_explainability:
                try:
                    explain = engine.explain(
                        processed=processed,
                        original_rgb=original,
                        predicted_grade=pred["icdas_grade"],
                    )
                except Exception:
                    explain = None
            primary = pred
            primary["action"] = action
            primary["explain"] = explain
            message = (
                "FALLBACK: whole-image ICDAS (no detector). "
                "Not a localized decay-region analysis."
            )
    elif not detections:
        mode = "no_detection"
        message = NO_DET_MSG
        if allow_whole_image_fallback:
            mode = "whole_image_fallback"
            fallback_used = True
            original, processed = engine.preprocess_image(image_rgb)
            pred = engine.predict(processed)
            action = get_clinical_action(pred["icdas_grade"])
            explain = None
            if include_explainability:
                try:
                    explain = engine.explain(
                        processed=processed,
                        original_rgb=original,
                        predicted_grade=pred["icdas_grade"],
                    )
                except Exception:
                    explain = None
            primary = pred
            primary["action"] = action
            primary["explain"] = explain
            message = (
                "FALLBACK: whole-image ICDAS because no decay region was localized. "
                "This is not a lesion-specific grade."
            )
    else:
        primary = {
            "icdas_grade": regions[0]["icdas_grade"],
            "confidence": regions[0]["confidence"],
            "probabilities": regions[0]["probabilities"],
            "low_confidence": any(r["low_confidence"] for r in regions),
            "class_name": regions[0]["icdas_class_name"],
            "action": get_clinical_action(regions[0]["icdas_grade"]),
            "explain": {
                "overlay": regions[0].get("heatmap_base64"),
                "heatmap": None,
                "contour": None,
            },
        }

    report_data = None
    if primary and mode != "no_detection":
        try:
            report_data = generate_report(
                icdas_grade=primary["icdas_grade"],
                confidence=primary["confidence"],
                finding=primary["action"]["finding"],
                urgency=primary["action"]["urgency"],
                detected_region_count=len(regions),
                quality_status=quality["status"],
                gradcam_available=bool(
                    primary.get("explain") and primary["explain"].get("overlay")
                ),
                localization_mode=mode,
            )
        except Exception:
            report_data = None

    action = (primary or {}).get("action") or {
        "label": "No localized region",
        "action": "Recapture",
        "description": NO_DET_MSG,
        "finding": NO_DET_MSG,
        "recommendation": "Capture a sharper intraoral photo showing the tooth surface.",
        "urgency": "LOW",
    }

    low_conf = bool(primary and primary.get("low_confidence")) if primary else False
    explain = (primary or {}).get("explain") or {}

    return {
        "mode": mode,
        "fallback_used": fallback_used,
        "detector_available": detector.available,
        "quality": quality,
        "region_count": len(regions),
        "regions": regions,
        "annotated_image_base64": _b64_png_rgb(annotated),
        "message": message,
        "ai_assisted_note": AI_ASSISTED,
        "icdas_grade": None if primary is None else primary["icdas_grade"],
        "confidence": 0.0 if primary is None else primary["confidence"],
        "probabilities": {} if primary is None else primary["probabilities"],
        "low_confidence": low_conf,
        "low_confidence_message": LOW_CONF_MSG if low_conf else None,
        "label": action["label"],
        "action": action["action"],
        "description": action["description"],
        "finding": (report_data or {}).get("finding", action["finding"]),
        "recommendation": (report_data or {}).get(
            "recommendation", action["recommendation"]
        ),
        "urgency": (report_data or {}).get("urgency", action["urgency"]),
        "report": (report_data or {}).get("report"),
        "heatmap_base64": explain.get("heatmap"),
        "overlay_base64": explain.get("overlay"),
        "contour_base64": explain.get("contour"),
    }
