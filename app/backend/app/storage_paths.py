"""Patient vs training storage. Never writes into gold or ICDAS train folders."""

from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
PATIENT_ROOT = DATA / "patient"
TRAINING_ROOT = DATA / "training"
TRAIN_UPLOADS = TRAINING_ROOT / "uploads"
TRAIN_DETECTED = TRAINING_ROOT / "detected"
TRAIN_LABELED = TRAINING_ROOT / "labeled"
TRAIN_VERSIONS = TRAINING_ROOT / "versions"
REPORTS_PATIENT = PROJECT_ROOT / "reports" / "patient"
HEATMAPS = PROJECT_ROOT / "reports" / "heatmaps"
MODELS_DET = PROJECT_ROOT / "models" / "detection"
MODELS_ICDAS = PROJECT_ROOT / "models" / "icdas"

TOOTH_V2_WEIGHTS = MODELS_DET / "tooth_detector_v2" / "weights" / "best.pt"
TOOTH_V1_WEIGHTS = MODELS_DET / "tooth_detector_batch01" / "weights" / "best.pt"


def ensure_dirs() -> None:
    for p in (
        PATIENT_ROOT,
        TRAIN_UPLOADS,
        TRAIN_DETECTED,
        TRAIN_LABELED,
        TRAIN_VERSIONS,
        REPORTS_PATIENT,
        HEATMAPS,
    ):
        p.mkdir(parents=True, exist_ok=True)
