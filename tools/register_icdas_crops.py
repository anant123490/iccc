#!/usr/bin/env python3
"""Register existing tooth-crop files into the ICDAS labeling pool.

Does not invent boxes. Does not crop from d/D lesion XML.
Place human or detector crops as JPG/PNG under data/icdas/labeling_v2/crops/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    CROP_POOL_COLUMNS,
    DATA_ICDAS,
    crop_pool_path,
    ensure_icdas_dirs,
    image_meta,
    load_csv,
    save_csv,
    sha256_file,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops-dir", default=None)
    p.add_argument("--source-type", default="unknown", choices=("public", "user", "unknown"))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_icdas_dirs()
    crops_dir = Path(args.crops_dir) if args.crops_dir else DATA_ICDAS / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in crops_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    df = load_csv(crop_pool_path(), CROP_POOL_COLUMNS)
    known = set(df["crop_path"].astype(str)) if not df.empty else set()
    rows = [] if df.empty else df.to_dict("records")
    added = 0
    for fp in files:
        rel = str(fp)
        if rel in known:
            continue
        meta = image_meta(fp)
        if meta is None:
            continue
        w, h, _ = meta
        digest = sha256_file(fp)
        sid = f"crop_{digest[:16]}"
        rows.append(
            {
                "sample_id": sid,
                "source_type": args.source_type,
                "source_image": "",
                "crop_path": rel,
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
                "sha256": digest,
                "width": w,
                "height": h,
                "status": "unlabelled",
            }
        )
        known.add(rel)
        added += 1
    import pandas as pd

    save_csv(crop_pool_path(), pd.DataFrame(rows), CROP_POOL_COLUMNS)
    print(f"crops_dir={crops_dir} files={len(files)} added={added}")
    if not files:
        print(
            "TOOTH_CROPS_UNAVAILABLE: put crop images in data/icdas/labeling_v2/crops/ then re-run. "
            "Do not use lesion d/D boxes as ICDAS crops."
        )


if __name__ == "__main__":
    main()
