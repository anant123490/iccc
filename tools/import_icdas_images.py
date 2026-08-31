#!/usr/bin/env python3
"""Copy user RGB images into data_icdas/user_images without renaming originals.

Does not modify the input folder. Does not assign ICDAS grades.
Does not import fdi_detection_dataset/images/selected unless --input points there
(still copies, never moves).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    DATA_ICDAS,
    SOURCE_MANIFEST_COLUMNS,
    ensure_icdas_dirs,
    image_meta,
    load_csv,
    save_csv,
    sha256_file,
    source_manifest_path,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder of JPG/JPEG/PNG to import (copied)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_icdas_dirs()
    src = Path(args.input)
    if not src.exists():
        print(f"INPUT_MISSING: {src}")
        sys.exit(1)
    dest = DATA_ICDAS / "user_images"
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    df = load_csv(source_manifest_path(), SOURCE_MANIFEST_COLUMNS)
    hashes = set(df["sha256"].astype(str)) if not df.empty else set()
    rows = [] if df.empty else df.to_dict("records")
    copied = 0
    dup = 0
    bad = 0
    for fp in files:
        meta = image_meta(fp)
        if meta is None:
            bad += 1
            continue
        digest = sha256_file(fp)
        if digest in hashes:
            dup += 1
            continue
        w, h, fmt = meta
        out_name = f"user_{digest[:16]}{fp.suffix.lower()}"
        out = dest / out_name
        shutil.copy2(fp, out)
        rows.append(
            {
                "source_id": f"user_{digest[:16]}",
                "source_type": "user",
                "original_path": str(fp.resolve()),
                "filename": out_name,
                "license_source": "user_provided",
                "width": w,
                "height": h,
                "format": fmt,
                "sha256": digest,
                "status": "registered_unlabelled",
            }
        )
        hashes.add(digest)
        copied += 1
    import pandas as pd

    save_csv(source_manifest_path(), pd.DataFrame(rows), SOURCE_MANIFEST_COLUMNS)
    print(f"copied={copied} duplicates_skipped={dup} unreadable={bad} dest={dest}")
    print("Originals were not modified. No ICDAS grades assigned.")


if __name__ == "__main__":
    main()
