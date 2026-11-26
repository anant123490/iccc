"""Additive legacy training-table migration checks."""

from __future__ import annotations

from types import SimpleNamespace

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
    monkeypatch.setattr(database, "settings", SimpleNamespace(database_url="sqlite://"))
    database.init_db()
    database.init_db()

    db = sessionmaker(bind=engine)()
    try:
        images = db.query(TrainingImage).order_by(TrainingImage.id).all()
        labels = db.query(TrainingLabel).order_by(TrainingLabel.id).all()
        assert len({image.image_id for image in images}) == 2
        assert all(image.patient_id is None for image in images)
        assert all(image.original_filename is None for image in images)
        assert all(image.upload_timestamp is not None for image in images)
        assert all(image.review_status == "APPROVED" for image in images)
        assert len({label.crop_id for label in labels}) == 2
        assert [label.annotation_status for label in labels] == ["LABELED", "LABELED"]
    finally:
        db.close()
        engine.dispose()


def test_legacy_sqlite_backup_is_created_before_schema_change(tmp_path, monkeypatch):
    from app import database
    from app import portal_db  # noqa: F401

    db_path = tmp_path / "legacy.db"
    legacy_engine = create_engine(f"sqlite:///{db_path}")
    with legacy_engine.begin() as connection:
        connection.execute(text("CREATE TABLE legacy_record (id INTEGER PRIMARY KEY, value TEXT)"))
        connection.execute(text("INSERT INTO legacy_record (id, value) VALUES (1, 'preserve-me')"))

    monkeypatch.setattr(database, "engine", legacy_engine)
    monkeypatch.setattr(database, "settings", SimpleNamespace(database_url=f"sqlite:///{db_path.as_posix()}"))
    database._backup_sqlite_before_migration()

    backups = list((tmp_path / "data" / "training" / "backups").glob("legacy_before_migration_*.db"))
    assert len(backups) == 1
    with create_engine(f"sqlite:///{backups[0].as_posix()}").connect() as connection:
        value = connection.execute(text("SELECT value FROM legacy_record WHERE id = 1")).scalar_one()
    assert value == "preserve-me"
    legacy_engine.dispose()


def test_existing_training_patient_ids_backfill_patient_foreign_keys(monkeypatch):
    from app import database
    from app.portal_db import TrainingImage

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE patients (id INTEGER PRIMARY KEY, public_id VARCHAR(32) UNIQUE, "
                "name VARCHAR(256), age INTEGER, gender VARCHAR(32), phone VARCHAR(64), created_at TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO patients (id, public_id, name) VALUES (1, 'P-123', 'Existing patient')"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE training_images (id INTEGER PRIMARY KEY, filename VARCHAR(256) NOT NULL, "
                "path VARCHAR(512) NOT NULL, overlay_path VARCHAR(512), n_crops INTEGER DEFAULT 0, "
                "created_at TIMESTAMP, image_id VARCHAR(64), patient_id VARCHAR(64))"
            )
        )
        conn.execute(
            text(
                "INSERT INTO training_images (id, filename, path, n_crops, image_id, patient_id) "
                "VALUES (1, 'existing.jpg', 'existing.jpg', 1, 'existing-image', 'P-123')"
            )
        )

    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "settings", SimpleNamespace(database_url="sqlite://"))
    database.init_db()
    db = sessionmaker(bind=engine)()
    try:
        image = db.query(TrainingImage).one()
        assert image.patient_record_id == 1
        assert image.patient_record.public_id == image.patient_id == "P-123"
    finally:
        db.close()
        engine.dispose()
