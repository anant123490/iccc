#!/usr/bin/env python3
"""Create ICDAS 0–4 dataset folder structure and sample annotations.csv."""

from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "icdas"

STRUCTURE = """
data/icdas/
├── train/0..4/
├── val/0..4/
├── test/0..4/
├── excluded/     # ICDAS 5/6 originals (never remapped to 4)
├── raw/
├── images/
└── annotations/annotations.csv
"""

ANNOTATIONS_TEMPLATE = """filename,icdas_score,split,patient_id,notes
train/0/example.jpg,0,train,P001,sound tooth
"""


def main():
    print("Creating ICDAS 0–4 dataset structure...")
    for split in ["train", "val", "test"]:
        for grade in range(5):
            (DATASET_ROOT / split / str(grade)).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "excluded").mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "annotations").mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "images").mkdir(parents=True, exist_ok=True)

    ann_path = DATASET_ROOT / "annotations" / "annotations.csv"
    if not ann_path.exists():
        ann_path.write_text(ANNOTATIONS_TEMPLATE, encoding="utf-8")
        print(f"Created template: {ann_path}")
    else:
        print(f"Kept existing labels: {ann_path}")

    readme = DATASET_ROOT / "README.md"
    if not readme.exists():
        readme.write_text(
            "# ICDAS dataset\n\n"
            "Place clinician-confirmed tooth images in class subfolders (0–4).\n\n"
            "ICDAS 5 and 6 images must **not** be copied into class 4. "
            "Keep them under `excluded/`.\n\n"
            f"```\n{STRUCTURE}\n```\n",
            encoding="utf-8",
        )
    print(f"Dataset root: {DATASET_ROOT}")
    print("Done. Add images to train/val/test/0..4/, then run:")
    print("  python ml/scripts/sync_annotations.py")
    print("  python ml/scripts/validate_dataset.py")


if __name__ == "__main__":
    main()
