"""Scoped Admin training-data reset. Never deletes models/, source, or patient data."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import uuid
import zipfile

from sqlalchemy.orm import Session
from sqlalchemy.inspection import inspect as sa_inspect

from .config import PROJECT_ROOT, get_settings
from .portal_db import (
    DatasetVersion,
    TrainingAuditEvent,
    TrainingImage,
    TrainingJob,
    TrainingLabel,
)
from .storage_paths import (
    MODELS_DET,
    MODELS_ICDAS,
    PATIENT_ROOT,
    TRAIN_DETECTED,
    TRAIN_LABELED,
    TRAIN_UPLOADS,
    TRAIN_VERSIONS,
    TRAIN_BACKUPS,
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


def _row_snapshot(db: Session, model) -> list[dict]:
    fields = [attr.key for attr in sa_inspect(model).column_attrs]
    return [{name: getattr(row, name) for name in fields} for row in db.query(model).all()]


def _count_regular_files(path: Path) -> int:
    """Count regular files without resolving every path entry."""
    if not path.exists():
        return 0
    pending = [path]
    count = 0
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            count += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return count


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _create_recovery_archive(db: Session, scope: str, plan: dict) -> Path:
    """Archive reset-scoped files and rows before any deletion takes place."""
    root = _resolved(TRAINING_ROOT)
    backup_root = _resolved(TRAIN_BACKUPS)
    try:
        backup_root.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Refusing to write recovery archive outside the training data root.") from exc
    if any(backup_root == _resolved(path) for _, path, _ in reset_dir_specs()):
        raise RuntimeError("Recovery archive directory overlaps a reset target.")
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = backup_root / f"dataset_reset_{stamp}_{uuid.uuid4().hex[:8]}.zip"
    partial = archive.with_suffix(".zip.partial")
    payload = {
        "scope": scope,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": plan.get("affected_counts", {}),
        "tables": {
            "training_images": _row_snapshot(db, TrainingImage),
            "training_labels": _row_snapshot(db, TrainingLabel),
            "dataset_versions": _row_snapshot(db, DatasetVersion),
            "training_jobs": _row_snapshot(db, TrainingJob) if scope == SCOPE_FULL else [],
        },
    }
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("reset_snapshot.json", json.dumps(payload, indent=2, default=_json_default))
            for key, path, _ in reset_dir_specs():
                if not path.exists():
                    continue
                target_root = _resolved(path)
                for item in path.rglob("*"):
                    if item.is_symlink() or not item.is_file():
                        continue
                    resolved = _resolved(item)
                    try:
                        relative = resolved.relative_to(target_root)
                    except ValueError:
                        continue
                    zf.write(item, (Path(key) / relative).as_posix())
        with zipfile.ZipFile(partial, "r") as check:
            corrupted_member = check.testzip()
            if corrupted_member:
                raise RuntimeError(f"Recovery archive verification failed at {corrupted_member}.")
        os.replace(partial, archive)
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return archive


def reset_plan(scope: str, db: Session | None = None) -> dict:
    normalized = (scope or SCOPE_DATASET).strip().lower()
    if normalized not in ALLOWED_SCOPES:
        raise ValueError("Reset scope must be 'dataset' or 'full'.")
    tables = ["training_labels", "training_images", "dataset_versions"]
    if normalized == SCOPE_FULL:
        tables.append("training_jobs")
    affected_counts = {
        "training_images": int(db.query(TrainingImage).count()) if db is not None else None,
        "tooth_crops": int(db.query(TrainingLabel).count()) if db is not None else None,
        "annotations": (
            int(db.query(TrainingLabel).filter(TrainingLabel.grade.isnot(None)).count())
            if db is not None
            else None
        ),
        "dataset_records": int(db.query(DatasetVersion).count()) if db is not None else None,
        "training_jobs": (
            int(db.query(TrainingJob).count()) if db is not None and normalized == SCOPE_FULL else 0
        ),
        "running_training_jobs": (
            int(db.query(TrainingJob).filter(TrainingJob.status == "running").count())
            if db is not None
            else None
        ),
        "files": sum(
            _count_regular_files(path)
            for _, path, _ in reset_dir_specs()
        ),
    }
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
        "affected_counts": affected_counts,
        "recovery_archive_directory": str(TRAIN_BACKUPS),
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
            str(TRAIN_BACKUPS),
            "training_audit_events",
        ],
        "notes": [
            "A ZIP recovery archive is created before any files or rows are deleted.",
            "Reset actions are recorded in training_audit_events; backups survive later resets.",
            "Reset is blocked while an ICDAS training subprocess is running.",
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
    plan = reset_plan(normalized, db)
    if plan["affected_counts"].get("running_training_jobs", 0):
        raise ValueError("Wait for the active ICDAS training job to finish before resetting training data.")
    # Keep a complete recovery point before touching either files or rows.
    # If archive creation fails, the reset stops without deleting anything.
    archive = _create_recovery_archive(db, normalized, plan)
    disk = [_empty_allowed_dir(path) for _, path, _ in reset_dir_specs()]
    for _, path, _ in reset_dir_specs():
        path.mkdir(parents=True, exist_ok=True)

    n_labels = db.query(TrainingLabel).delete()
    n_images = db.query(TrainingImage).delete()
    n_versions = db.query(DatasetVersion).delete()
    n_jobs = 0
    if normalized == SCOPE_FULL:
        n_jobs = db.query(TrainingJob).delete()
    audit = TrainingAuditEvent(
        event_type="training_reset",
        entity_type="training_dataset",
        entity_id=normalized,
        actor="admin",
        details_json=json.dumps(
            {
                "scope": normalized,
                "recovery_archive": str(archive),
                "affected_counts": plan["affected_counts"],
                "deleted_rows": {
                    "training_labels": int(n_labels or 0),
                    "training_images": int(n_images or 0),
                    "dataset_versions": int(n_versions or 0),
                    "training_jobs": int(n_jobs or 0),
                },
            },
            sort_keys=True,
        ),
    )
    db.add(audit)
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
            "recovery_archive": str(archive),
            "audit_event_id": audit.id,
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
