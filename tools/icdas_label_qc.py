#!/usr/bin/env python3
"""QC for ICDAS v2 labels and crop pool. Does not delete files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    CROP_POOL_COLUMNS,
    LABEL_COLUMNS,
    PROJECT_ROOT,
    TRAIN_GRADES,
    crop_pool_path,
    ensure_icdas_dirs,
    image_meta,
    labels_path,
    load_csv,
    sha256_file,
)


def main():
    ensure_icdas_dirs()
    pool = load_csv(crop_pool_path(), CROP_POOL_COLUMNS)
    labels = load_csv(labels_path(), LABEL_COLUMNS)
    rows = []
    issues = {
        "missing_crop_file": 0,
        "unreadable": 0,
        "tiny_crop": 0,
        "invalid_grade": 0,
        "skipped": 0,
        "duplicate_sha": 0,
        "invalid_bbox": 0,
    }
    hashes = {}
    for _, r in pool.iterrows():
        p = Path(str(r["crop_path"]))
        issue = ""
        if not p.exists():
            issues["missing_crop_file"] += 1
            issue = "missing_crop_file"
        else:
            meta = image_meta(p)
            if meta is None:
                issues["unreadable"] += 1
                issue = "unreadable"
            else:
                w, h, _ = meta
                if w < 16 or h < 16:
                    issues["tiny_crop"] += 1
                    issue = "tiny_crop"
                digest = sha256_file(p)
                hashes.setdefault(digest, []).append(r["sample_id"])
        for b in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            v = str(r.get(b, "")).strip()
            if v:
                try:
                    float(v)
                except ValueError:
                    issues["invalid_bbox"] += 1
                    issue = "invalid_bbox"
        rows.append({"sample_id": r["sample_id"], "crop_path": r["crop_path"], "issue": issue or "ok"})
    for h, ids in hashes.items():
        if len(ids) > 1:
            issues["duplicate_sha"] += len(ids)
    for _, r in labels.iterrows():
        g = str(r["icdas_grade"])
        if g == "SKIP":
            issues["skipped"] += 1
        elif g not in TRAIN_GRADES:
            issues["invalid_grade"] += 1
            rows.append({"sample_id": r["sample_id"], "crop_path": r.get("crop_path"), "issue": f"invalid_grade:{g}"})
    qc = PROJECT_ROOT / "reports" / "icdas_label_qc.csv"
    pd.DataFrame(rows).to_csv(qc, index=False)
    md = PROJECT_ROOT / "reports" / "ICDAS_LABEL_QC.md"
    md.write_text(
        "# ICDAS label QC\n\nNothing was deleted.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in issues.items())
        + "\n\nd/D lesion classes were not read as ICDAS.\n",
        encoding="utf-8",
    )
    print(issues)
    print(f"wrote {qc} and {md}")


if __name__ == "__main__":
    main()
