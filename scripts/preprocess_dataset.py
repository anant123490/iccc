#!/usr/bin/env python3
"""Batch preprocess raw images into train/val/test splits."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocessing import preprocess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../dataset/raw")
    parser.add_argument("--output", default="../dataset")
    parser.add_argument("--csv", default=None, help="annotations.csv with labels")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    records = []
    if args.csv:
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            records.append({
                "path": input_dir / row["filename"],
                "label": int(row["icdas_score"]),
            })
    else:
        for img in input_dir.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                # Try parent folder name as label
                if img.parent.name.isdigit():
                    records.append({"path": img, "label": int(img.parent.name)})

    if not records:
        print("No images found.")
        return

    paths = [r["path"] for r in records]
    labels = [r["label"] for r in records]

    train_idx, test_idx = train_test_split(
        range(len(records)), test_size=args.test_size, stratify=labels, random_state=42
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_size / (1 - args.test_size), stratify=train_labels, random_state=42
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    ann_rows = []

    for split_name, indices in splits.items():
        for i in tqdm(indices, desc=split_name):
            src = Path(paths[i])
            label = labels[i]
            dest_dir = output_dir / split_name / str(label)
            dest_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(src))
            if img is None:
                continue
            processed = (preprocess_image(img) * 255).astype("uint8")
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            dest = dest_dir / src.name
            cv2.imwrite(str(dest), processed_bgr)
            ann_rows.append({
                "filename": f"{split_name}/{label}/{src.name}",
                "icdas_score": label,
                "split": split_name,
            })

    pd.DataFrame(ann_rows).to_csv(output_dir / "annotations.csv", index=False)
    print(f"Done. {len(ann_rows)} images processed → {output_dir}")


if __name__ == "__main__":
    main()
