#!/usr/bin/env python3
"""ICDAS inference helper. Does not write ground-truth labels.

Does not modify fdi_detection_dataset/images/selected/.
Copies 420 only when --copy-working is set, into data_working/personal_420_inference/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import PERSONAL_420, PROJECT_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None)
    p.add_argument("--copy-working", action="store_true", help="Copy 420 sources to data_working (never move)")
    p.add_argument("--run-personal-420", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    args = parse_args()
    model = PROJECT_ROOT / "models" / "icdas" / "current" / "icdas_mobilenet_cbam_v2" / "best.keras"
    md = PROJECT_ROOT / "reports" / "PERSONAL_420_INFERENCE.md"
    if args.copy_working or args.run_personal_420:
        dest = PROJECT_ROOT / "data_working" / "personal_420_inference" / "copies"
        dest.mkdir(parents=True, exist_ok=True)
        srcs = sorted(PERSONAL_420.glob("*.jpg"))
        before = {p.name: sha256_file(p) for p in srcs}
        copied = 0
        for p in srcs:
            shutil.copy2(p, dest / p.name)
            copied += 1
        after = {p.name: sha256_file(p) for p in srcs}
        if before != after:
            raise SystemExit("ABORT: 420 originals changed — this must never happen")
        md.write_text(
            "# Personal 420 inference\n\n"
            "**Independent inference/demo images — NOT test-set ground truth.**\n\n"
            f"Copies written: {copied} to `{dest}`\n\n"
            f"Originals unchanged: yes (sha256 verified)\n\n"
            f"v2 model present: {model.exists()}\n\n"
            "ICDAS grades were **not** auto-saved as labels.\n"
            "Tooth crops were **not** invented (no whole-tooth detector).\n",
            encoding="utf-8",
        )
        print(md.read_text(encoding="utf-8"))
        if not model.exists():
            print("NO_V2_MODEL: copies only; no predictions.")
        return
    print("Pass --copy-working to copy 420 images to data_working without changing originals.")
    print(f"v2 model exists: {model.exists()}")


if __name__ == "__main__":
    main()
