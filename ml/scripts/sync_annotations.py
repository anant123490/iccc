#!/usr/bin/env python3
"""
Rebuild data/icdas/annotations/annotations.csv from train/val/test folder layout.

Only ICDAS 0–4 are accepted. Extra class folders fail validation.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.dataset import DatasetValidationError, discover_images_from_folders, validate_dataset_layout
from src.icdas import NUM_CLASSES

DATASET_ROOT = ROOT / "data" / "icdas"


def main():
    print(f"Scanning {DATASET_ROOT} ...")
    try:
        validate_dataset_layout(
            str(DATASET_ROOT),
            num_classes=NUM_CLASSES,
            require_all_splits=False,
            require_all_classes=False,
        )
        df = discover_images_from_folders(str(DATASET_ROOT), num_classes=NUM_CLASSES)
    except DatasetValidationError as exc:
        print(f"FAILED: {exc}")
        return 1

    if df.empty:
        print("No images found under data/icdas/train|val|test/<0-4>/.")
        print("Add images first, then run this script again.")
        print("Existing data/icdas/annotations/annotations.csv was not overwritten.")
        return 0

    out = DATASET_ROOT / "annotations" / "annotations.csv"
    df.to_csv(out, index=False)
    counts = df.groupby(["split", "icdas_score"]).size().unstack(fill_value=0)
    print(f"Wrote {len(df)} rows to {out}")
    print("\nCounts by split and ICDAS score:")
    print(counts.to_string())
    print("\nNext: python ml/scripts/validate_dataset.py")
    print("Then: python ml/train.py --config ml/configs/default.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
