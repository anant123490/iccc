"""Scoped Admin training-data reset. Never deletes models/, source, or patient data."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from .config import PROJECT_ROOT, get_settings
from .portal_db import DatasetVersion, TrainingImage, TrainingJob, TrainingLabel
from .storage_paths import (
    MODELS_DET,
    MODELS_ICDAS,
    PATIENT_ROOT,
    TRAIN_DETECTED,
    TRAIN_LABELED,
    TRAIN_UPLOADS,
    TRAIN_VERSIONS,
    TRAINING_ROOT,
)

RESET_CONFIRMATION_TEXT = (
    "I understand that this permanently deletes the selected training data."
)
SCOPE_DATASET = "dataset"
SCOPE_FULL = "full"
ALLOWED_SCOPES = {SCOPE_DATASET, SCOPE_FULL}

def reset_dir_specs() -> tuple[tuple[str, Path, str], ...]:
    return (
        ("uploads", TRAIN_UPLOADS, "uploaded training photographs and duplicate copies"),
        ("detected", TRAIN_DETECTED, "detector overlays and generated tooth crops"),
        ("labeled", TRAIN_LABELED, "ICDAS 0–4 labeled crop copies"),
        ("versions", TRAIN_VERSIONS, "dataset manifests and train/val/test splits"),
    )


def _resolved(path: Path) -> Path:
    return path.resolve()


def _is_allowed_reset_dir(path: Path) -> bool:
    root = _resolved(TRAINING_ROOT)
    target = _resolved(path)
    if target == root:
        return False
    try:
        target.relative_to(root)
    except ValueError:
        return False
    allowed = {_resolved(p) for _, p, _ in reset_dir_specs()}
    return target in allowed


def _empty_allowed_dir(path: Path) -> dict:
    """Remove contents of one training subdirectory. Keep the directory itself."""
    import shutil

    if not _is_allowed_reset_dir(path):
        raise RuntimeError(f"Refusing to delete path outside the training reset scope: {path}")
    removed_entries = 0
    path.mkdir(parents=True, exist_ok=True)
    for child in list(path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed_entries += 1
    return {"path": str(path), "removed_entries": removed_entries}


def validate_reset_request(scope: str, confirm: bool, confirmation_text: str) -> str:
    normalized = (scope or "").strip().lower()
    if normalized not in ALLOWED_SCOPES:
        raise ValueError("Reset scope must be 'dataset' or 'full'.")
    if not confirm:
        raise ValueError("Reset confirmation is required.")
    typed = (confirmation_text or "").strip()
    if typed != RESET_CONFIRMATION_TEXT:
        raise ValueError("Confirmation text does not match.")
    return normalized


def reset_plan(scope: str) -> dict:
    normalized = (scope or SCOPE_DATASET).strip().lower()
    if normalized not in ALLOWED_SCOPES:
        raise ValueError("Reset scope must be 'dataset' or 'full'.")
    tables = ["training_labels", "training_images", "dataset_versions"]
    if normalized == SCOPE_FULL:
        tables.append("training_jobs")
    return {
        "scope": normalized,
        "confirmation_text": RESET_CONFIRMATION_TEXT,
        "paths": [
            {
                "key": key,
                "path": str(path),
                "description": desc,
                "exists": path.exists(),
            }
            for key, path, desc in reset_dir_specs()
        ],
        "database_tables": tables,
        "will_not_delete": [
            str(PROJECT_ROOT / "app"),
            str(PROJECT_ROOT / "ml"),
            str(MODELS_DET),
            str(MODELS_ICDAS),
            str(PATIENT_ROOT),
            str(PROJECT_ROOT / ".git"),
            str(PROJECT_ROOT / ".env"),
            str(PROJECT_ROOT / "app" / "backend" / ".env"),
            "patients",
            "visits",
            "uploaded_images",
            "tooth_detections",
            "tooth_crops",
            "icdas_predictions",
            "clinical_reports",
            "predictions",
            "model_versions",
        ],
        "notes": [
            "Does not DROP tables or recreate the database.",
            "Does not delete Tooth Detector V2 or ICDAS model weights.",
            "Does not start ICDAS training.",
            "ICDAS labels remain 0–4 in schema and labeling UI.",
        ],
    }


def _inventory_summary(inv: dict) -> dict:
    photos = inv.get("photos") or {}
    crops = inv.get("crops") or {}
    enabled = bool(inv.get("icdas_train_enabled"))
    return {
        "uploads": int(photos.get("uploaded", 0) or 0),
        "duplicates": int(photos.get("exact_duplicates", 0) or 0)
        + int(photos.get("likely_duplicates", 0) or 0),
        "crops": int(crops.get("detected", 0) or 0),
        "labeled": int(crops.get("labeled", 0) or 0),
        "unlabeled": int(crops.get("unlabeled", 0) or 0),
        "dataset": inv.get("status") or ("READY" if inv.get("dataset_ready") else "NOT READY"),
        "training": "DISABLED" if not enabled else "IDLE",
        "dataset_ready": bool(inv.get("dataset_ready")),
        "icdas_train_enabled": enabled,
        "launched": False,
    }


def execute_training_reset(
    db: Session,
    scope: str,
    confirm: bool,
    confirmation_text: str,
) -> dict:
    """Delete only Admin training-workflow files and rows. No model training."""
    from .training_workflow import dataset_inventory

    normalized = validate_reset_request(scope, confirm, confirmation_text)
    plan = reset_plan(normalized)
    disk = [_empty_allowed_dir(path) for _, path, _ in reset_dir_specs()]
    for _, path, _ in reset_dir_specs():
        path.mkdir(parents=True, exist_ok=True)

    n_labels = db.query(TrainingLabel).delete()
    n_images = db.query(TrainingImage).delete()
    n_versions = db.query(DatasetVersion).delete()
    n_jobs = 0
    if normalized == SCOPE_FULL:
        n_jobs = db.query(TrainingJob).delete()
    db.commit()

    inv = dataset_inventory(db)
    summary = _inventory_summary(inv)
    settings = get_settings()
    if not settings.allow_icdas_train:
        summary["training"] = "DISABLED"
    summary.update(
        {
            "ok": True,
            "scope": normalized,
            "plan": plan,
            "disk": disk,
            "deleted_rows": {
                "training_labels": int(n_labels or 0),
                "training_images": int(n_images or 0),
                "dataset_versions": int(n_versions or 0),
                "training_jobs": int(n_jobs or 0),
            },
        }
    )
    return summary
