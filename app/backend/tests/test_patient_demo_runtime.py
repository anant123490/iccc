from __future__ import annotations

import json

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import portal_runtime, portal_service
from app.portal_db import (
    Base,
    ICDASPrediction,
    Patient,
    ToothCrop,
    ToothDetection,
    UploadedImage,
    Visit,
)


def _db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _visit_with_saved_crop(db: Session):
    patient = Patient(public_id="TEST-PORTAL", name="Test", is_anonymous=True)
    visit = Visit(patient=patient, visit_date="2026-11-25")
    image = UploadedImage(
        visit=visit,
        filename="sample.jpg",
        path="original.jpg",
        content_hash="a" * 64,
        quality_json="{}",
    )
    detection = ToothDetection(
        image=image,
        index_in_image=0,
        confidence=0.91,
        x1=0,
        y1=0,
        x2=32,
        y2=32,
    )
    crop = ToothCrop(detection=detection, path="crop.jpg")
    prediction = ICDASPrediction(
        crop=crop,
        grade=2,
        confidence=63.5,
        probabilities_json=json.dumps({str(i): 0.2 for i in range(5)}),
    )
    db.add(visit)
    db.commit()
    return visit, image, detection, crop, prediction


def test_demo_checkpoint_resolution_requires_explicit_demo_mode(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.keras"
    checkpoint.write_bytes(b"candidate")
    monkeypatch.setattr(portal_runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(portal_runtime, "is_valid_icdas_checkpoint", lambda _: True)
    monkeypatch.setattr(
        portal_runtime,
        "get_settings",
        lambda: type("Settings", (), {"local_demo_mode": True, "demo_model_path": "best.keras"})(),
    )
    assert portal_runtime.resolve_demo_icdas_checkpoint() == checkpoint.resolve()

    monkeypatch.setattr(
        portal_runtime,
        "get_settings",
        lambda: type("Settings", (), {"local_demo_mode": False, "demo_model_path": "best.keras"})(),
    )
    assert portal_runtime.resolve_demo_icdas_checkpoint() is None


def test_repeated_detection_reuses_saved_crops_without_deleting_them(monkeypatch):
    db = _db()
    try:
        visit, image, detection, crop, _ = _visit_with_saved_crop(db)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        monkeypatch.setattr(portal_service, "_read_rgb", lambda _: rgb)
        monkeypatch.setattr(portal_service, "_b64_rgb", lambda _: "image")
        monkeypatch.setattr(portal_service, "detect_rgb", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("detector reran")))

        output = portal_service.run_detection_for_image(db, image)

        assert output["teeth_detected"] == 1
        assert len(output["crops"]) == 1
        assert db.query(ToothDetection).filter(ToothDetection.image_id == image.id).count() == 1
        assert db.query(ToothCrop).filter(ToothCrop.detection_id == detection.id).count() == 1
        assert db.query(ICDASPrediction).filter(ICDASPrediction.crop_id == crop.id).count() == 1
    finally:
        db.close()


def test_cached_classification_is_not_overwritten_by_demo_candidate(monkeypatch):
    db = _db()
    try:
        visit, _, _, crop, prediction = _visit_with_saved_crop(db)
        original_probs = prediction.probabilities_json

        class Runtime:
            prediction_mode = "DEMO"

            @staticmethod
            def icdas_ok():
                return True

            @staticmethod
            def predict_crop_rgb(_):
                raise AssertionError("cached prediction must not be recomputed")

            @staticmethod
            def explain_crop(_):
                raise AssertionError("cached Grad-CAM should not be recomputed")

        monkeypatch.setattr(portal_runtime, "get_runtime", lambda: Runtime())
        monkeypatch.setattr(portal_service, "_read_rgb", lambda _: np.zeros((32, 32, 3), dtype=np.uint8))
        monkeypatch.setattr(portal_service, "_b64_rgb", lambda _: "image")

        output = portal_service.classify_visit_crops(db, visit, include_heatmaps=True, persist=True)

        db.refresh(prediction)
        assert output["teeth"][0]["icdas_grade"] == 2
        assert output["teeth"][0]["prediction_mode"] == "SAVED_RESULT_MODEL_VERSION_UNAVAILABLE"
        assert prediction.grade == 2
        assert prediction.probabilities_json == original_probs
        assert prediction.crop_id == crop.id
    finally:
        db.close()
