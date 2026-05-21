#!/usr/bin/env python3
"""Create dataset folder structure and sample annotations.csv."""

import os
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"

STRUCTURE = """
dataset/
├── train/
│   ├── 0/   # ICDAS 0 - sound
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   └── 6/
├── val/
│   ├── 0/ ... 6/
├── test/
│   ├── 0/ ... 6/
├── raw/          # Unprocessed downloads
└── annotations.csv
"""

ANNOTATIONS_TEMPLATE = """filename,icdas_score,split,patient_id,notes
train/0/example.jpg,0,train,P001,sound tooth
"""


def main():
    print("Creating ICDAS dataset structure...")
    for split in ["train", "val", "test"]:
        for grade in range(7):
            (DATASET_ROOT / split / str(grade)).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "raw").mkdir(parents=True, exist_ok=True)

    ann_path = DATASET_ROOT / "annotations.csv"
    if not ann_path.exists():
        ann_path.write_text(ANNOTATIONS_TEMPLATE, encoding="utf-8")
        print(f"Created template: {ann_path}")

    readme = DATASET_ROOT / "README.md"
    readme.write_text(
        "# Dataset Directory\n\n"
        "Place intraoral images in class subfolders (0-6) or update annotations.csv.\n\n"
        f"```\n{STRUCTURE}\n```\n",
        encoding="utf-8",
    )
    print(f"Dataset root: {DATASET_ROOT}")
    print("Done. Add images to train/val/test/0..6/, then run:")
    print("  python ml/scripts/sync_annotations.py")
    print("  python scripts/download_datasets.py  # optional public data")


if __name__ == "__main__":
    main()
