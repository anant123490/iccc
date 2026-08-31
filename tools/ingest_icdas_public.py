#!/usr/bin/env python3
"""Register public RGB intraoral files into data_icdas/source_manifest.csv.

Does not copy or modify source images. Does not create ICDAS labels.
Does not treat d/D or missing boxes as ICDAS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    SOURCE_MANIFEST_COLUMNS,
    ensure_icdas_dirs,
    find_public_image_root,
    image_meta,
    load_csv,
    save_csv,
    sha256_file,
    source_manifest_path,
)

LICENSE = "Zenodo 10.5281/zenodo.14827784 (public RGB intraoral; lesion d/D are NOT ICDAS)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="Override public image root")
    p.add_argument("--limit", type=int, default=0, help="Debug cap; 0 = all")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_icdas_dirs()
    root = Path(args.root) if args.root else find_public_image_root()
    if root is None or not root.exists():
        print("PUBLIC_IMAGES_MISSING: no local public RGB folder found.")
        sys.exit(1)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        files = files[: args.limit]
    path = source_manifest_path()
    df = load_csv(path, SOURCE_MANIFEST_COLUMNS)
    known = set(df["original_path"].astype(str)) if not df.empty else set()
    rows = [] if df.empty else df.to_dict("records")
    added = 0
    skipped = 0
    bad = 0
    for i, fp in enumerate(files, 1):
        rel = str(fp)
        if rel in known:
            skipped += 1
            continue
        meta = image_meta(fp)
        if meta is None:
            bad += 1
            rows.append(
                {
                    "source_id": f"pub_{fp.stem}_{i}",
                    "source_type": "public",
                    "original_path": rel,
                    "filename": fp.name,
                    "license_source": LICENSE,
                    "width": "",
                    "height": "",
                    "format": fp.suffix.lower().lstrip("."),
                    "sha256": "",
                    "status": "unreadable",
                }
            )
            continue
        w, h, fmt = meta
        digest = sha256_file(fp)
        rows.append(
            {
                "source_id": f"pub_{digest[:16]}",
                "source_type": "public",
                "original_path": rel,
                "filename": fp.name,
                "license_source": LICENSE,
                "width": w,
                "height": h,
                "format": fmt,
                "sha256": digest,
                "status": "registered_unlabelled",
            }
        )
        known.add(rel)
        added += 1
        if added % 200 == 0:
            print(f"hashed {added} new / scanned {i}/{len(files)}")
    import pandas as pd

    out = pd.DataFrame(rows)
    save_csv(path, out, SOURCE_MANIFEST_COLUMNS)
    print(
        f"root={root}\nfiles_found={len(files)}\nadded={added}\n"
        f"already_listed={skipped}\nunreadable={bad}\nmanifest={path}"
    )
    print("ICDAS grades were NOT assigned.")


if __name__ == "__main__":
    main()
