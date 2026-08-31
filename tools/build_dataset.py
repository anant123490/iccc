#!/usr/bin/env python3
"""Build dataset/train|val|test/0-4 from human ICDAS labels.

Splits by patient_id when that column exists, otherwise by source_image,
so crops from the same mouth never leak across splits.

Does not invent ICDAS grades. Reads labels/labels.csv only.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import (  # noqa: E402
    ICDAS_CLASSES,
    LABEL_CSV_COLUMNS,
    PROJECT_ROOT,
    SPLITS,
    ensure_dataset_class_dirs,
    ensure_pipeline_dirs,
    load_csv,
    project_path,
    read_image,
    save_csv,
    write_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy human-labeled crops into dataset/train|val|test/0-4."
    )
    parser.add_argument("--labels", default="labels/labels.csv")
    parser.add_argument("--crops-dir", default="data/tooth_crops/generated/images")
    parser.add_argument("--crops-csv", default="data/tooth_crops/generated/crops.csv")
    parser.add_argument("--dataset", default="data/icdas", help="Existing dataset root (do not duplicate).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--resize", type=int, default=224, help="0 keeps the crop size as stored.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def group_key(row: pd.Series) -> str:
    for col in ("patient_id", "patient", "case_id"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return f"patient:{row[col]}"
    source = str(row.get("source_image") or "").strip()
    if source:
        return f"source:{source}"
    return f"crop:{row.get('crop_id')}"


def stable_bucket(key: str, seed: int) -> float:
    payload = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_split(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    u = stable_bucket(key, seed)
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def find_crop_file(crops_dir: Path, filename: str, crop_id: str) -> Path | None:
    direct = crops_dir / str(filename)
    if direct.exists():
        return direct
    matches = list(crops_dir.glob(f"{crop_id}.*"))
    return matches[0] if matches else None


def main():
    args = parse_args()
    ensure_pipeline_dirs()
    dataset_root = resolve(args.dataset)
    ensure_dataset_class_dirs(dataset_root)

    labels_path = resolve(args.labels)
    crops_dir = resolve(args.crops_dir)
    crops_csv = resolve(args.crops_csv)

    labels = load_csv(labels_path, LABEL_CSV_COLUMNS)
    if labels.empty:
        print(f"No labels found at {labels_path}")
        print("Run: streamlit run tools/label_icdas.py")
        print("Created empty class folders under", dataset_root)
        return 1

    crops_meta = load_csv(crops_csv, ["crop_id", "filename", "source_image"])
    if not crops_meta.empty:
        labels = labels.merge(
            crops_meta,
            on="crop_id",
            how="left",
            suffixes=("", "_crop"),
        )
        if "source_image" in labels.columns and "source_image_crop" in labels.columns:
            labels["source_image"] = labels["source_image"].fillna(labels["source_image_crop"])
        if "filename" in labels.columns and "filename_crop" in labels.columns:
            labels["filename"] = labels["filename"].fillna(labels["filename_crop"])

    valid_rows = []
    invalid = 0
    for _, row in labels.iterrows():
        try:
            grade = int(row["icdas_grade"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        if grade < 0 or grade > 4:
            invalid += 1
            continue
        valid_rows.append(row)
    if invalid:
        print(f"Skipped {invalid} rows with invalid ICDAS grades (not remapped).")
    if not valid_rows:
        print("No valid ICDAS 0–4 labels.")
        return 1

    work = pd.DataFrame(valid_rows)
    work["group"] = work.apply(group_key, axis=1)
    group_to_split = {
        key: assign_split(key, args.seed, args.train_ratio, args.val_ratio)
        for key in sorted(work["group"].unique())
    }
    work["split"] = work["group"].map(group_to_split)

    copied = 0
    missing = 0
    skipped = 0
    manifest_rows = []

    for _, row in work.iterrows():
        crop_id = str(row["crop_id"])
        filename = str(row.get("filename") or f"{crop_id}.jpg")
        split = row["split"]
        grade = int(row["icdas_grade"])
        src = find_crop_file(crops_dir, filename, crop_id)
        dest_dir = dataset_root / split / str(grade)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{crop_id}{Path(filename).suffix or '.jpg'}"
        dest = dest_dir / dest_name
        if src is None or not src.exists():
            missing += 1
            continue
        if dest.exists() and not args.overwrite:
            skipped += 1
            manifest_rows.append(
                {
                    "crop_id": crop_id,
                    "filename": dest_name,
                    "source_image": row.get("source_image"),
                    "group": row["group"],
                    "split": split,
                    "icdas_grade": grade,
                    "dest": str(dest.relative_to(dataset_root)).replace("\\", "/"),
                }
            )
            continue
        if args.resize and args.resize > 0:
            image = read_image(src)
            if image is None:
                missing += 1
                continue
            image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
            write_image(dest, image, overwrite=True)
        elif args.copy_mode == "hardlink":
            if dest.exists():
                dest.unlink()
            try:
                dest.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dest)
        else:
            shutil.copy2(src, dest)
        copied += 1
        manifest_rows.append(
            {
                "crop_id": crop_id,
                "filename": dest_name,
                "source_image": row.get("source_image"),
                "group": row["group"],
                "split": split,
                "icdas_grade": grade,
                "dest": str(dest.relative_to(dataset_root)).replace("\\", "/"),
            }
        )

    manifest_path = project_path("reports", "split_manifest.csv")
    save_csv(manifest_path, pd.DataFrame(manifest_rows))

    print(f"Copied:   {copied}")
    print(f"Skipped existing: {skipped}")
    print(f"Missing crops: {missing}")
    print(f"Dataset root: {dataset_root}")
    for split in SPLITS:
        n = int((work["split"] == split).sum())
        print(f"  {split}: {n} labeled crops")
    for grade in ICDAS_CLASSES:
        n = int((work["icdas_grade"].astype(int) == int(grade)).sum())
        print(f"  class {grade}: {n}")
    print("Leakage prevention: same source_image/patient stays in one split.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
