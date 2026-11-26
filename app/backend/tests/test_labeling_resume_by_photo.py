"""Photo lookup and resume behavior use disposable SQLite data and crop files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.portal_db import LabelingProgress, Patient, TrainingAuditEvent, TrainingImage, TrainingLabel
from app.training_workflow import (
    labeling_queue,
    training_crop_listing,
    training_labeling_photo_inventory,
    save_icdas_label,
)


def _jpg(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((48, 48, 3), 30 + seed * 20, dtype=np.uint8)
    rgb[8:40, 8:40] = (seed * 37 % 255, seed * 61 % 255, seed * 83 % 255)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    path.write_bytes(encoded.tobytes())


def _session(tmp_path: Path):
    import app.db_models  # noqa: F401
    import app.portal_db  # noqa: F401

    engine = create_engine(f"sqlite:///{(tmp_path / 'resume-labeling.db').as_posix()}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def _photo(db, tmp_path: Path, number: int, *, original_filename: str = "front-view.jpg"):
    patient = db.query(Patient).filter_by(public_id="PAT-001").first()
    if patient is None:
        patient = Patient(
            public_id="PAT-001",
            name="Anonymous Patient",
            is_anonymous=True,
            anonymous_sequence=1,
        )
        db.add(patient)
        db.flush()
    image = TrainingImage(
        image_id=f"reference-{number:03d}",
        patient_id=patient.public_id,
        patient_record=patient,
        filename=f"safe-storage-{number:03d}.jpg",
        original_filename=original_filename,
        path=str(tmp_path / f"safe-storage-{number:03d}.jpg"),
        processing_status="COMPLETED",
        duplicate_status="UNIQUE",
        is_active=True,
        boxes_verified=True,
        split_name="UNASSIGNED",
    )
    db.add(image)
    db.flush()
    return image


def _crop(db, tmp_path: Path, image: TrainingImage, *, idx: int, grade: int | None, status="UNLABELED"):
    crop_id = f"CROP-{image.id:03d}-{idx:03d}"
    crop_path = tmp_path / "detected" / str(image.id) / f"crop-{idx:03d}.jpg"
    _jpg(crop_path, idx + image.id)
    label = TrainingLabel(
        image_id=image.id,
        crop_id=crop_id,
        crop_path=str(crop_path),
        grade=grade,
        annotation_status="LABELED" if grade is not None else status,
        index_in_image=idx,
        box_verified=True,
        active=True,
        skipped=status == "UNCERTAIN",
    )
    db.add(label)
    db.flush()
    return label


def test_photo_inventory_searches_photo_number_reference_and_filename(tmp_path):
    db, engine = _session(tmp_path)
    try:
        image = _photo(db, tmp_path, 7, original_filename="upper-left-molar.jpg")
        _crop(db, tmp_path, image, idx=0, grade=2)
        _crop(db, tmp_path, image, idx=1, grade=None)
        db.commit()

        inventory = training_labeling_photo_inventory(db)
        photo = inventory["photos"][0]
        assert photo["photo_id"] == f"PHOTO-{image.id:03d}"
        assert photo["reference_number"] == "reference-007"
        assert photo["original_filename"] == "upper-left-molar.jpg"
        assert photo["patient_id"] == "PAT-001"
        assert (photo["eligible_crops"], photo["labeled_crops"], photo["remaining_unlabeled_crops"]) == (2, 1, 1)
        for search in (photo["photo_id"], str(image.id), "reference-007", "upper-left-molar.jpg"):
            result = training_labeling_photo_inventory(db, search=search)
            assert [row["training_image_id"] for row in result["photos"]] == [image.id]
        assert training_crop_listing(db, training_image_id=image.id)["count"] == 2
    finally:
        db.close()
        engine.dispose()


def test_legacy_photo_without_original_filename_displays_safe_stored_name(tmp_path):
    db, engine = _session(tmp_path)
    try:
        image = _photo(db, tmp_path, 8, original_filename=None)
        image.original_filename = None
        _crop(db, tmp_path, image, idx=0, grade=None)
        db.commit()

        photo = training_labeling_photo_inventory(db)["photos"][0]
        assert photo["original_filename"] is None
        assert photo["original_filename_available"] is False
        assert photo["filename"] == image.filename
        assert training_labeling_photo_inventory(db, search=image.filename)["count"] == 1
    finally:
        db.close()
        engine.dispose()


def test_resume_selects_first_unlabeled_crop_for_selected_photo_and_keeps_existing_grades(tmp_path, monkeypatch):
    import app.training_workflow as workflow

    db, engine = _session(tmp_path)
    labeled_dir = tmp_path / "labeled"
    monkeypatch.setattr(workflow, "TRAIN_LABELED", labeled_dir)
    try:
        image = _photo(db, tmp_path, 3)
        existing = _crop(db, tmp_path, image, idx=0, grade=1)
        first_unlabeled = _crop(db, tmp_path, image, idx=1, grade=None)
        later_unlabeled = _crop(db, tmp_path, image, idx=2, grade=None)
        other_photo = _photo(db, tmp_path, 4, original_filename="different-photo.jpg")
        _crop(db, tmp_path, other_photo, idx=0, grade=None)
        db.commit()

        queue = labeling_queue(db, image_id=image.id, mode="first_unlabeled", track="initial")
        assert queue["current"]["label_id"] == first_unlabeled.id
        assert queue["current"]["crop_id"] == first_unlabeled.crop_id
        assert queue["current"]["photo_id"] == f"PHOTO-{image.id:03d}"

        count_before = db.query(TrainingLabel).count()
        saved = save_icdas_label(db, first_unlabeled.id, 4, track="initial")
        assert saved["saved"] is True
        assert db.query(TrainingLabel).count() == count_before
        db.refresh(existing)
        db.refresh(first_unlabeled)
        assert existing.grade == 1
        assert first_unlabeled.grade == 4
        assert existing.patient_id == first_unlabeled.patient_id == "PAT-001"
        assert existing.image_id == first_unlabeled.image_id == image.id
        assert db.query(TrainingAuditEvent).filter_by(event_type="label_saved").count() == 1

        refreshed = training_labeling_photo_inventory(db)["photos"]
        photo = next(row for row in refreshed if row["training_image_id"] == image.id)
        assert photo["labeled_crops"] == 2
        assert photo["remaining_unlabeled_crops"] == 1
        assert photo["grade_counts"] == {"0": 0, "1": 1, "2": 0, "3": 0, "4": 1}
        assert training_labeling_photo_inventory(db)["last_saved_label"]["crop_id"] == first_unlabeled.crop_id
        next_queue = labeling_queue(db, image_id=image.id, mode="first_unlabeled", track="initial")
        assert next_queue["current"]["label_id"] == later_unlabeled.id
        assert next_queue["current"]["photo_id"] == f"PHOTO-{image.id:03d}"
    finally:
        db.close()
        engine.dispose()


def test_all_labeled_photo_does_not_restart_and_saved_resume_cursor_is_read_from_db(tmp_path):
    db, engine = _session(tmp_path)
    try:
        image = _photo(db, tmp_path, 9, original_filename="completed.jpg")
        first = _crop(db, tmp_path, image, idx=0, grade=0)
        second = _crop(db, tmp_path, image, idx=1, grade=3)
        db.commit()

        inventory = training_labeling_photo_inventory(db)
        photo = inventory["photos"][0]
        assert photo["labeling_status"] == "FULLY LABELED"
        assert photo["eligible_crops"] == photo["labeled_crops"] == 2
        queue = labeling_queue(db, image_id=image.id, mode="first_unlabeled", track="initial")
        assert queue["current"] is None
        assert queue["resume_label_id"] is None

        review = labeling_queue(db, image_id=image.id, label_id=second.id, mode="photo", track="review")
        assert review["current"]["label_id"] == second.id
        assert review["current"]["grade"] == 3
        db.add(LabelingProgress(track="initial", label_id=second.id, image_id=image.id))
        db.commit()
        cursor = training_labeling_photo_inventory(db)["saved_resume_position"]
        assert cursor["photo_id"] == f"PHOTO-{image.id:03d}"
        assert cursor["crop_id"] == second.crop_id
        assert first.grade == 0
    finally:
        db.close()
        engine.dispose()


def test_labeling_photo_and_crop_lookup_api_is_read_only_and_searchable(tmp_path):
    from app.portal_auth import require_admin
    from app.portal_routes import get_db, router

    db, engine = _session(tmp_path)
    try:
        image = _photo(db, tmp_path, 12, original_filename="right-side.jpg")
        label = _crop(db, tmp_path, image, idx=0, grade=2)
        db.commit()
        count_before = (db.query(TrainingImage).count(), db.query(TrainingLabel).count())

        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield db

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[require_admin] = lambda: None
        with TestClient(app) as client:
            response = client.get("/api/v1/admin/training/labeling/photos", params={"search": "right-side.jpg"})
            assert response.status_code == 200
            assert response.json()["photos"][0]["photo_id"] == f"PHOTO-{image.id:03d}"
            crops = client.get(
                "/api/v1/admin/training/crops",
                params={"training_image_id": image.id},
            )
            assert crops.status_code == 200
            assert crops.json()["crops"][0]["crop_id"] == label.crop_id
        assert (db.query(TrainingImage).count(), db.query(TrainingLabel).count()) == count_before
    finally:
        db.close()
        engine.dispose()
