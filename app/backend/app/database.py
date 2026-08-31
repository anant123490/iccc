"""SQLAlchemy engine and session for prediction history."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
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

    Base.metadata.create_all(bind=engine)
    if settings.database_url.startswith("sqlite"):
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
            for table, cols in alters.items():
                existing = {
                    row[1]
                    for row in conn.execute(text(f"PRAGMA table_info({table})"))
                }
                if not existing:
                    continue
                for name, ddl in cols:
                    if name not in existing:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
            label_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(training_labels)"))
            }
            if label_cols:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_training_labels_crop_hash "
                        "ON training_labels (crop_hash)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_training_labels_crop_phash "
                        "ON training_labels (crop_phash)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_training_labels_crop_dup_status "
                        "ON training_labels (crop_duplicate_status)"
                    )
                )
    logger.info("Database ready (%s)", settings.database_url.split("://", 1)[0])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
