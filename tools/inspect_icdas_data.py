#!/usr/bin/env python3
"""Inspect ICDAS v2 data locations. Does not assign grades or modify sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    DATA_ICDAS,
    PERSONAL_420,
    PROJECT_ROOT,
    crop_pool_path,
    ensure_icdas_dirs,
    find_public_image_root,
    labels_path,
    load_csv,
    LABEL_COLUMNS,
    CROP_POOL_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    source_manifest_path,
)


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            n += 1
    return n


def main():
    ensure_icdas_dirs()
    public = find_public_image_root()
    crops_dir = DATA_ICDAS / "crops"
    user_dir = DATA_ICDAS / "user_images"
    labels = load_csv(labels_path(), LABEL_COLUMNS)
    pool = load_csv(crop_pool_path(), CROP_POOL_COLUMNS)
    src = load_csv(source_manifest_path(), SOURCE_MANIFEST_COLUMNS)
    labelled = labels[labels["status"] == "labelled"] if not labels.empty else labels
    skipped = labels[labels["status"] == "skipped"] if not labels.empty else labels
    report = {
        "public_image_root": str(public) if public else None,
        "public_image_count_on_disk": count_images(public) if public else 0,
        "personal_420_dir": str(PERSONAL_420),
        "personal_420_count": count_images(PERSONAL_420),
        "user_images_count": count_images(user_dir),
        "registered_sources": int(len(src)),
        "crop_pool_rows": int(len(pool)),
        "crop_files_on_disk": count_images(crops_dir),
        "labels_total_rows": int(len(labels)),
        "labelled_0_4": int(len(labelled)),
        "skipped": int(len(skipped)),
        "tooth_detector_for_icdas": "UNAVAILABLE — no whole-tooth detector; d/D boxes must not be used as ICDAS crops",
        "existing_icdas_dataset_dir": str(PROJECT_ROOT / "data" / "icdas"),
        "note": "d/D lesion boxes are not ICDAS. Unannotated public images are not ICDAS 0.",
    }
    print(json.dumps(report, indent=2))
    out = PROJECT_ROOT / "reports" / "icdas_v2_inspect.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
