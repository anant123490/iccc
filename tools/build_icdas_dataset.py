#!/usr/bin/env python3
"""Build data_icdas/final from dentist labels only.

Never writes to dataset/. SKIP rows never enter train/val/test.
Splits by original source image to avoid crop leakage.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    DATA_ICDAS,
    LABEL_COLUMNS,
    PROJECT_ROOT,
    TRAIN_GRADES,
    ensure_icdas_dirs,
    group_key,
    labels_path,
    load_csv,
    save_csv,
    stable_split,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_icdas_dirs()
    final_root = DATA_ICDAS / "final"
    labels = load_csv(labels_path(), LABEL_COLUMNS)
    if labels.empty:
        print("NO_LABELS: label crops in ICDAS Labeling Studio first.")
        write_json(
            PROJECT_ROOT / "reports" / "icdas_dataset_final_summary.json",
            {"status": "NO_LABELS", "counts": {}},
        )
        sys.exit(0)

    train_rows = labels[
        (labels["status"] == "labelled")
        & (labels["icdas_grade"].isin(sorted(TRAIN_GRADES)))
    ].copy()
    skipped = labels[labels["icdas_grade"] == "SKIP"]
    invalid = labels[~labels["icdas_grade"].isin(list(TRAIN_GRADES) + ["SKIP"])]

    if args.overwrite:
        for split in ("train", "val", "test"):
            for g in "01234":
                d = final_root / split / g
                if d.exists():
                    for f in d.glob("*"):
                        if f.is_file():
                            f.unlink()

    assigned = []
    missing = 0
    copied = 0
    for _, row in train_rows.iterrows():
        src = Path(str(row["crop_path"]))
        if not src.exists():
            missing += 1
            continue
        split = stable_split(group_key(row), args.seed, args.train_ratio, args.val_ratio)
        grade = str(row["icdas_grade"])
        dest = final_root / split / grade / f"{row['sample_id']}{src.suffix.lower() or '.jpg'}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        assigned.append({**row.to_dict(), "split": split, "dest": str(dest)})
        copied += 1

    manifest = pd.DataFrame(assigned)
    out_csv = PROJECT_ROOT / "reports" / "icdas_v2_split_manifest.csv"
    if not manifest.empty:
        manifest.to_csv(out_csv, index=False)
        leak = 0
        for src_name, grp in manifest.groupby(manifest["source_image"].map(lambda x: Path(str(x)).name if str(x) else "")):
            if not src_name:
                continue
            if grp["split"].nunique() > 1:
                leak += 1
    else:
        leak = 0

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for g in "01234":
            counts[split][g] = len(list((final_root / split / g).glob("*")))

    summary = {
        "labelled_kept": int(len(train_rows)),
        "copied": copied,
        "missing_files": missing,
        "skipped_excluded": int(len(skipped)),
        "invalid_excluded": int(len(invalid)),
        "leakage_source_images": leak,
        "counts": counts,
        "final_root": str(final_root),
        "dataset_dir_untouched": True,
    }
    write_json(PROJECT_ROOT / "reports" / "icdas_dataset_final_summary.json", summary)

    md = PROJECT_ROOT / "reports" / "ICDAS_DATASET_FINAL.md"
    lines = [
        "# ICDAS v2 dataset build",
        "",
        "Labels are **project-generated** via ICDAS Labeling Studio, not original public d/D.",
        "",
        f"- Copied: {copied}",
        f"- SKIP excluded: {len(skipped)}",
        f"- Missing crops: {missing}",
        f"- Source-image leakage groups: {leak}",
        f"- Existing `dataset/` was not modified.",
        "",
        "## Counts",
        "",
        "```",
        str(counts),
        "```",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(summary)
    if copied == 0:
        print("DATASET_EMPTY: not ready to train.")


if __name__ == "__main__":
    main()
