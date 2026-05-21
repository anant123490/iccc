#!/usr/bin/env python3
"""
Rebuild dataset/annotations.csv from train/val/test folder layout.

Run after copying images into class folders (e.g. 2000 labeled images):
  python ml/scripts/sync_annotations.py
"""

from pathlib import Path

import pandas as pd

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_records(root: Path) -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            label = int(class_dir.name)
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_SUFFIXES:
                    records.append(
                        {
                            "filename": f"{split}/{label}/{img_path.name}",
                            "icdas_score": label,
                            "split": split,
                        }
                    )
    return records


def main():
    print(f"Scanning {DATASET_ROOT} ...")
    records = discover_records(DATASET_ROOT)
    if not records:
        print("No images found under dataset/train|val|test/<0-6>/.")
        print("Add images first, then run this script again.")
        return

    df = pd.DataFrame(records)
    out = DATASET_ROOT / "annotations.csv"
    df.to_csv(out, index=False)

    counts = df.groupby(["split", "icdas_score"]).size().unstack(fill_value=0)
    print(f"Wrote {len(df)} rows to {out}")
    print("\nCounts by split and ICDAS score:")
    print(counts.to_string())
    print("\nNext: cd ml && python train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
