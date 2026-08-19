#!/usr/bin/env python3
"""Validate ICDAS 0–4 dataset layout and print per-class counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.dataset import DatasetValidationError, format_dataset_report, validate_dataset_layout
from src.icdas import NUM_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ICDAS 0–4 dataset folders")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2] / "dataset"),
    )
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not require every class to have at least one image",
    )
    args = parser.parse_args()

    try:
        report = validate_dataset_layout(
            args.root,
            num_classes=args.num_classes,
            require_all_splits=True,
            require_all_classes=not args.allow_empty,
        )
    except DatasetValidationError as exc:
        print(f"DATASET VALIDATION FAILED: {exc}")
        return 1

    print(format_dataset_report(report))
    print("Dataset layout OK (ICDAS 0–4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
