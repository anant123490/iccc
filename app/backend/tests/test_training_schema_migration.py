"""Additive legacy training-table migration checks."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def test_legacy_training_columns_are_added_before_indexes(monkeypatch):
    from app import database
    from app.portal_db import TrainingImage, TrainingLabel

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE training_images ("
                "id INTEGER PRIMARY KEY, filename VARCHAR(256) NOT NULL, path VARCHAR(512) NOT NULL, "
                "overlay_path VARCHAR(512), n_crops INTEGER DEFAULT 0, content_hash VARCHAR(64), "
                "phash VARCHAR(32), duplicate_status VARCHAR(32), canonical_id INTEGER, "
                "boxes_verified BOOLEAN DEFAULT 1, exclude_from_dataset BOOLEAN DEFAULT 0, "
                "is_active BOOLEAN DEFAULT 1, created_at TIMESTAMP, image_id VARCHAR(64))"
            )
        )
        conn.execute(
            text(
                "INSERT INTO training_images (id, filename, path, n_crops, duplicate_status, "
                "boxes_verified, is_active, created_at, image_id) VALUES "
                "(1, 'old1.jpg', 'old1.jpg', 1, 'UNIQUE', 1, 1, '2026-01-01', 'legacy-id'), "
                "(2, 'old2.jpg', 'old2.jpg', 1, 'UNIQUE', 1, 1, '2026-01-02', 'legacy-id')"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE training_labels ("
                "id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, crop_path VARCHAR(512) NOT NULL, "
                "grade INTEGER, x1 INTEGER DEFAULT 0, y1 INTEGER DEFAULT 0, x2 INTEGER DEFAULT 0, "
                "y2 INTEGER DEFAULT 0, confidence FLOAT DEFAULT 0, index_in_image INTEGER DEFAULT 0, "
                "crop_hash VARCHAR(64), crop_phash VARCHAR(32), crop_duplicate_status VARCHAR(32), "
                "duplicate_of_label_id INTEGER, box_verified BOOLEAN DEFAULT 1, active BOOLEAN DEFAULT 1, "
                "skipped BOOLEAN DEFAULT 0, crop_id VARCHAR(64))"
            )
        )
        conn.execute(
            text(
                "INSERT INTO training_labels (id, image_id, crop_path, grade, crop_id) VALUES "
                "(1, 1, 'crop1.jpg', 2, 'legacy-crop'), (2, 2, 'crop2.jpg', 1, 'legacy-crop')"
            )
        )

    monkeypatch.setattr(database, "engine", engine)
    database.init_db()
    database.init_db()

    db = sessionmaker(bind=engine)()
    try:
        images = db.query(TrainingImage).order_by(TrainingImage.id).all()
        labels = db.query(TrainingLabel).order_by(TrainingLabel.id).all()
        assert len({image.image_id for image in images}) == 2
        assert all(image.patient_id is None for image in images)
        assert all(image.upload_timestamp is not None for image in images)
        assert all(image.review_status == "APPROVED" for image in images)
        assert len({label.crop_id for label in labels}) == 2
        assert [label.annotation_status for label in labels] == ["LABELED", "LABELED"]
    finally:
        db.close()
        engine.dispose()
