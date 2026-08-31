"""Smoke tests for quality checks and localized pipeline (no 420-image use)."""

from __future__ import annotations

import numpy as np
import pytest

from app.image_quality import assess_image_quality
from app.caries_pipeline import NO_DET_MSG, run_localized_pipeline


class _FakeEngine:
    def preprocess_image(self, image):
        proc = np.zeros((224, 224, 3), dtype=np.float32)
        return image, proc

    def predict(self, processed):
        return {
            "icdas_grade": 2,
            "class_name": "ICDAS 2",
            "confidence": 84.0,
            "probabilities": {"0": 0.02, "1": 0.04, "2": 0.84, "3": 0.07, "4": 0.03},
            "low_confidence": False,
            "low_confidence_message": None,
        }

    def explain(self, processed, original_rgb, predicted_grade):
        return {"heatmap": None, "overlay": None, "contour": None}


def test_invalid_image_quality():
    q = assess_image_quality(None)
    assert q["ok"] is False
    assert q["status"] == "invalid"


def test_blurry_image_quality():
    img = np.full((160, 160, 3), 128, dtype=np.uint8)
    q = assess_image_quality(img)
    assert q["status"] in {"ok", "low_quality", "fail"}
    assert q["verdict"] in {"PASS", "WARNING", "FAIL"}
    assert "sharpness" in q


def test_no_detection_without_fallback(monkeypatch):
    import app.caries_pipeline as cp

    class Det:
        available = True

        def predict(self, image_rgb):
            return []

    monkeypatch.setattr(cp, "get_caries_detector", lambda: Det())
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[10:20, 10:20] = 200
    out = run_localized_pipeline(_FakeEngine(), rgb, include_explainability=False)
    assert out["mode"] == "no_detection"
    assert out["icdas_grade"] is None
    assert NO_DET_MSG in (out["message"] or "")
    assert out["region_count"] == 0


def test_one_and_multiple_detections(monkeypatch):
    import app.caries_pipeline as cp

    class Det:
        available = True

        def predict(self, image_rgb):
            return [
                {
                    "x1": 2,
                    "y1": 2,
                    "x2": 30,
                    "y2": 30,
                    "confidence": 0.9,
                    "class_id": 0,
                    "class_name": "D",
                    "class_meaning": "permanent-tooth decay region",
                },
                {
                    "x1": 40,
                    "y1": 40,
                    "x2": 70,
                    "y2": 70,
                    "confidence": 0.8,
                    "class_id": 1,
                    "class_name": "d",
                    "class_meaning": "primary-tooth decay region",
                },
            ]

    monkeypatch.setattr(cp, "get_caries_detector", lambda: Det())
    rgb = np.zeros((90, 90, 3), dtype=np.uint8)
    rgb[:] = 40
    out = run_localized_pipeline(_FakeEngine(), rgb, include_explainability=True)
    assert out["mode"] == "localized"
    assert out["region_count"] == 2
    assert out["regions"][0]["icdas_grade"] == 2
    assert out["icdas_grade"] == 2


def test_report_mentions_ai_assisted():
    from app.groq_service import _local_report

    text = _local_report(2, 80.0, "finding", "MODERATE")["report"]
    assert "AI-assisted" in text or "not a definitive" in text.lower()
