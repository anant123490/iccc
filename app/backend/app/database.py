"""SQLAlchemy engine and session for prediction history."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import get_settings

logger = logging.getLogger("icdas.db")


class Base(DeclarativeBase):
    pass


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


settings = get_settings()
engine = create_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from sqlalchemy import text

    from . import db_models  # noqa: F401
    from . import portal_db  # noqa: F401

    # The dental app predates a formal migration framework. Keep this small,
    # additive migration idempotent so existing SQLite databases can be opened
    # without dropping patient, model, or training records. Run it before
    # create_all so metadata indexes that reference newly added columns are not
    # attempted against an older table shape.
    if inspect(engine).has_table("training_images"):
        alters = {
            "visits": [
                ("test_only", "BOOLEAN DEFAULT 0"),
            ],
            "training_images": [
                ("content_hash", "VARCHAR(64)"),
                ("phash", "VARCHAR(32)"),
                ("duplicate_status", "VARCHAR(32) DEFAULT 'UNIQUE'"),
                ("canonical_id", "INTEGER"),
                ("boxes_verified", "BOOLEAN DEFAULT 0"),
                ("exclude_from_dataset", "BOOLEAN DEFAULT 0"),
                ("is_active", "BOOLEAN DEFAULT 1"),
                ("image_id", "VARCHAR(64)"),
                ("patient_id", "VARCHAR(64)"),
                ("upload_timestamp", "TIMESTAMP"),
                ("dataset_version", "VARCHAR(64)"),
                ("annotation_status", "VARCHAR(32) DEFAULT 'UNLABELED'"),
                ("split_name", "VARCHAR(16) DEFAULT 'UNASSIGNED'"),
                ("crop_count", "INTEGER DEFAULT 0"),
                ("review_status", "VARCHAR(32) DEFAULT 'PENDING'"),
                ("detector_version", "VARCHAR(64) DEFAULT 'Tooth Detector V2'"),
            ],
            "training_labels": [
                ("index_in_image", "INTEGER DEFAULT 0"),
                ("crop_hash", "VARCHAR(64)"),
                ("crop_phash", "VARCHAR(32)"),
                ("crop_duplicate_status", "VARCHAR(32)"),
                ("duplicate_of_label_id", "INTEGER"),
                ("box_verified", "BOOLEAN DEFAULT 0"),
                ("active", "BOOLEAN DEFAULT 1"),
                ("skipped", "BOOLEAN DEFAULT 0"),
                ("crop_id", "VARCHAR(64)"),
                ("annotation_status", "VARCHAR(32) DEFAULT 'UNLABELED'"),
            ],
            "dataset_versions": [
                ("version_number", "INTEGER"),
                ("statistics_json", "TEXT"),
                ("split_seed", "INTEGER DEFAULT 42"),
            ],
            "training_jobs": [
                ("dataset_name", "VARCHAR(64)"),
                ("model_dir", "VARCHAR(512)"),
            ],
        }
        with engine.begin() as conn:
            inspector = inspect(conn)
            for table, cols in alters.items():
                if not inspector.has_table(table):
                    continue
                existing = {column["name"] for column in inspector.get_columns(table)}
                for name, ddl in cols:
                    if name not in existing:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

            if inspector.has_table("training_images"):
                # Assign stable internal image/crop identifiers to pre-migration
                # rows. Patient identity deliberately remains NULL until the
                # administrator supplies a pseudonymous identifier.
                rows = conn.execute(
                    text("SELECT id FROM training_images WHERE image_id IS NULL OR image_id = ''")
                ).fetchall()
                for row in rows:
                    conn.execute(
                        text("UPDATE training_images SET image_id = :value WHERE id = :id"),
                        {"value": uuid.uuid4().hex, "id": row[0]},
                    )
                duplicates = conn.execute(
                    text(
                        "SELECT image_id FROM training_images WHERE image_id IS NOT NULL "
                        "GROUP BY image_id HAVING COUNT(*) > 1"
                    )
                ).fetchall()
                for (image_id,) in duplicates:
                    ids = conn.execute(
                        text("SELECT id FROM training_images WHERE image_id = :value ORDER BY id"),
                        {"value": image_id},
                    ).fetchall()
                    for row in ids[1:]:
                        conn.execute(
                            text("UPDATE training_images SET image_id = :value WHERE id = :id"),
                            {"value": uuid.uuid4().hex, "id": row[0]},
                        )
                conn.execute(
                    text(
                        "UPDATE training_images SET upload_timestamp = COALESCE(upload_timestamp, created_at) "
                        "WHERE upload_timestamp IS NULL"
                    )
                )
                conn.execute(
                    text(
                        "UPDATE training_images SET crop_count = COALESCE(n_crops, 0) "
                        "WHERE crop_count IS NULL OR crop_count < COALESCE(n_crops, 0)"
                    )
                )
                conn.execute(
                    text(
                        "UPDATE training_images SET review_status = "
                        "CASE WHEN duplicate_status = 'INVALID' OR duplicate_status IN "
                        "('EXACT_DUPLICATE', 'LIKELY_DUPLICATE') THEN 'FLAGGED' "
                        "WHEN is_active = 0 THEN 'REJECTED' "
                        "WHEN boxes_verified = 1 THEN 'APPROVED' ELSE 'PENDING' END"
                    )
                )
                conn.execute(
                    text("CREATE UNIQUE INDEX IF NOT EXISTS ux_training_images_image_id ON training_images (image_id)")
                )
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_training_images_patient_id ON training_images (patient_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_training_images_split_name ON training_images (split_name)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_training_images_dataset_version ON training_images (dataset_version)"))

            if inspector.has_table("training_labels"):
                rows = conn.execute(
                    text("SELECT id FROM training_labels WHERE crop_id IS NULL OR crop_id = ''")
                ).fetchall()
                for row in rows:
                    conn.execute(
                        text("UPDATE training_labels SET crop_id = :value WHERE id = :id"),
                        {"value": uuid.uuid4().hex, "id": row[0]},
                    )
                duplicates = conn.execute(
                    text(
                        "SELECT crop_id FROM training_labels WHERE crop_id IS NOT NULL "
                        "GROUP BY crop_id HAVING COUNT(*) > 1"
                    )
                ).fetchall()
                for (crop_id,) in duplicates:
                    ids = conn.execute(
                        text("SELECT id FROM training_labels WHERE crop_id = :value ORDER BY id"),
                        {"value": crop_id},
                    ).fetchall()
                    for row in ids[1:]:
                        conn.execute(
                            text("UPDATE training_labels SET crop_id = :value WHERE id = :id"),
                            {"value": uuid.uuid4().hex, "id": row[0]},
                        )
                conn.execute(
                    text(
                        "UPDATE training_labels SET annotation_status = CASE "
                        "WHEN grade IS NOT NULL THEN 'LABELED' "
                        "WHEN skipped = 1 THEN 'UNCERTAIN' ELSE 'UNLABELED' END"
                    )
                )
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_training_labels_crop_id ON training_labels (crop_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_training_labels_crop_hash ON training_labels (crop_hash)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_training_labels_crop_phash ON training_labels (crop_phash)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_training_labels_crop_dup_status ON training_labels (crop_duplicate_status)"))
    Base.metadata.create_all(bind=engine)
    logger.info("Database ready (%s)", settings.database_url.split("://", 1)[0])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
