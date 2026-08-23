#!/usr/bin/env python3
"""Validate the ICDAS 0–4 dataset used by ml/train.py."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import (  # noqa: E402
    ICDAS_CLASSES,
    PROJECT_ROOT,
    QUALITY_CSV_COLUMNS,
    SPLITS,
    ensure_dataset_class_dirs,
    ensure_pipeline_dirs,
    is_blank_image,
    is_image_file,
    load_csv,
    project_path,
    read_image,
    save_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check ICDAS dataset layout, labels, and leakage.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--labels", default="labels/labels.csv")
    parser.add_argument("--manifest", default="reports/split_manifest.csv")
    parser.add_argument("--create-missing-dirs", action="store_true", default=True)
    parser.add_argument("--no-create-missing-dirs", action="store_false", dest="create_missing_dirs")
    parser.add_argument("--imbalance-ratio", type=float, default=3.0)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def list_split_images(dataset_root: Path):
    records = []
    missing_dirs = []
    for split in SPLITS:
        for grade in ICDAS_CLASSES:
            folder = dataset_root / split / grade
            if not folder.exists():
                missing_dirs.append(str(folder.relative_to(dataset_root)).replace("\\", "/"))
                continue
            for path in folder.iterdir():
                if not is_image_file(path):
                    continue
                records.append(
                    {
                        "split": split,
                        "icdas_grade": int(grade),
                        "path": path,
                        "filename": path.name,
                        "stem": path.stem,
                    }
                )
    return records, missing_dirs


def main():
    args = parse_args()
    ensure_pipeline_dirs()
    dataset_root = resolve(args.dataset)
    if args.create_missing_dirs:
        ensure_dataset_class_dirs(dataset_root)
        created = True
    else:
        created = False

    records, missing_dirs = list_split_images(dataset_root)
    df = pd.DataFrame(records)

    print("=== ICDAS dataset check ===")
    print(f"Dataset root: {dataset_root}")
    if created:
        print("Ensured train/val/test class directories 0-4 exist.")
    if missing_dirs and not created:
        print("MISSING DIRECTORIES:")
        for item in missing_dirs:
            print("  ", item)

    total = len(df)
    print(f"total images: {total}")
    if df.empty:
        print("train count: 0")
        print("validation count: 0")
        print("test count: 0")
        for grade in ICDAS_CLASSES:
            print(f"class {grade} count: 0")
        print("No images yet. Label crops, then run build_dataset.py.")
        return 0

    for split, label in (("train", "train count"), ("val", "validation count"), ("test", "test count")):
        print(f"{label}: {int((df['split'] == split).sum())}")
    for grade in ICDAS_CLASSES:
        print(f"class {grade} count: {int((df['icdas_grade'] == int(grade)).sum())}")

    corrupt = []
    blank = []
    for rec in records:
        image = read_image(rec["path"])
        if image is None:
            corrupt.append(rec["path"].as_posix())
        elif is_blank_image(image):
            blank.append(rec["path"].as_posix())
    print(f"corrupt images: {len(corrupt)}")
    for item in corrupt[:20]:
        print("  ", item)
    print(f"blank images: {len(blank)}")

    filename_counts = df["filename"].value_counts()
    dup_names = filename_counts[filename_counts > 1]
    print(f"duplicate filenames: {len(dup_names)}")
    for name, count in dup_names.head(20).items():
        print(f"  {name}: {count}")

    id_counts = df["stem"].value_counts()
    dup_ids = id_counts[id_counts > 1]
    print(f"duplicate IDs: {len(dup_ids)}")
    for name, count in dup_ids.head(20).items():
        print(f"  {name}: {count}")

    labels = load_csv(resolve(args.labels), ["crop_id", "filename", "source_image", "icdas_grade"])
    invalid_labels = []
    if not labels.empty:
        for _, row in labels.iterrows():
            try:
                grade = int(row["icdas_grade"])
            except (TypeError, ValueError):
                invalid_labels.append(str(row.get("crop_id")))
                continue
            if grade < 0 or grade > 4:
                invalid_labels.append(f"{row.get('crop_id')}={row.get('icdas_grade')}")
    print(f"invalid labels: {len(invalid_labels)}")
    for item in invalid_labels[:20]:
        print("  ", item)

    labeled_ids = set(labels["crop_id"].astype(str)) if not labels.empty else set()
    on_disk_ids = set(df["stem"].astype(str))
    missing_images = sorted(labeled_ids - on_disk_ids) if labeled_ids else []
    print(f"missing images (labeled but not in dataset/): {len(missing_images)}")
    for item in missing_images[:20]:
        print("  ", item)

    manifest = load_csv(resolve(args.manifest), ["crop_id", "source_image", "group", "split", "icdas_grade"])
    leakage_groups = []
    if not manifest.empty and "group" in manifest.columns:
        grouped = manifest.groupby("group")["split"].nunique()
        leakage_groups = grouped[grouped > 1].index.tolist()
    elif not labels.empty and "source_image" in labels.columns:
        source_to_splits = defaultdict(set)
        id_to_split = dict(zip(df["stem"].astype(str), df["split"]))
        for _, row in labels.iterrows():
            cid = str(row["crop_id"])
            if cid in id_to_split:
                source_to_splits[str(row.get("source_image"))].add(id_to_split[cid])
        leakage_groups = [src for src, splits in source_to_splits.items() if src and len(splits) > 1]

    print(f"source-image leakage groups: {len(leakage_groups)}")
    for item in leakage_groups[:20]:
        print("  ", item)

    class_counts = [int((df["icdas_grade"] == int(g)).sum()) for g in ICDAS_CLASSES]
    nonzero = [c for c in class_counts if c > 0]
    imbalance = False
    if len(nonzero) >= 2:
        ratio = max(nonzero) / max(min(nonzero), 1)
        imbalance = ratio >= args.imbalance_ratio
        print(f"class imbalance ratio (max/min): {ratio:.2f}")
    print(f"class imbalance warning: {imbalance}")

    extra_dirs = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name not in ICDAS_CLASSES:
                extra_dirs.append(f"{split}/{child.name}")
    if extra_dirs:
        print("unsupported class directories (not 0–4):")
        for item in extra_dirs:
            print("  ", item)

    issues = []
    for path in corrupt:
        issues.append({"crop_id": Path(path).stem, "filename": Path(path).name, "source_image": "", "issue": "corrupt_image", "details": path, "kept": True})
    for path in blank:
        issues.append({"crop_id": Path(path).stem, "filename": Path(path).name, "source_image": "", "issue": "blank_image", "details": path, "kept": True})
    for name in dup_ids.index:
        issues.append({"crop_id": name, "filename": "", "source_image": "", "issue": "duplicate_crop_id", "details": str(int(dup_ids[name])), "kept": True})
    for group in leakage_groups:
        issues.append({"crop_id": "", "filename": "", "source_image": str(group), "issue": "source_image_leakage", "details": str(group), "kept": True})

    report_path = project_path("reports", "dataset_quality_report.csv")
    existing = load_csv(report_path, QUALITY_CSV_COLUMNS)
    new = pd.DataFrame(issues)
    if new.empty:
        if existing.empty:
            save_csv(report_path, pd.DataFrame(columns=QUALITY_CSV_COLUMNS), QUALITY_CSV_COLUMNS)
    else:
        combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
        save_csv(report_path, combined, QUALITY_CSV_COLUMNS)
    print(f"quality report: {report_path}")

    problems = len(corrupt) + len(dup_ids) + len(leakage_groups) + len(invalid_labels) + len(extra_dirs)
    if problems:
        print(f"RESULT: issues found ({problems})")
        return 1
    print("RESULT: dataset layout OK for ICDAS 0-4 training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
