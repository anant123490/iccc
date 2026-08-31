"""
Build an isolated YOLO layout from the existing Zenodo dump.

Does not modify data_external/detection/raw or original labels.
Does not download. Does not map d/D to ICDAS.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_RAW = ROOT / "data_external" / "detection" / "raw"
SRC_YOLO = ROOT / "data_external" / "detection" / "annotations" / "yolo"
OUT = ROOT / "data_external" / "detection" / "public_caries"
SEED = 42
# YOLO ids verified against VOC in Stage 2C
NAMES = {0: "D", 1: "d"}


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cid = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if cid not in NAMES:
        return None
    if any(not (v == v) or v in (float("inf"), float("-inf")) for v in vals):
        return None
    xc, yc, w, h = vals
    if w <= 0 or h <= 0:
        return None
    # clip to [0,1] if slightly out; skip if wildly invalid
    if min(xc, yc, w, h) < -0.05 or max(xc, yc, w, h) > 1.05:
        return None
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 1e-6), 1.0)
    h = min(max(h, 1e-6), 1.0)
    return cid, xc, yc, w, h


def split_name(stem: str) -> str:
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def main() -> None:
    random.seed(SEED)
    stats = Counter()
    class_counts = Counter()
    rows = []
    skipped_empty = 0
    skipped_invalid = 0
    skipped_no_img = 0

    label_files = list(SRC_YOLO.rglob("*.txt"))
    for lp in label_files:
        rel = lp.relative_to(SRC_YOLO)
        img = SRC_RAW / rel.with_suffix(".jpg")
        if not img.exists():
            skipped_no_img += 1
            continue
        text = lp.read_text(encoding="utf-8", errors="ignore")
        boxes = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = parse_yolo_line(line)
            if parsed is None:
                skipped_invalid += 1
                continue
            boxes.append(parsed)
            class_counts[NAMES[parsed[0]]] += 1
        if not boxes:
            skipped_empty += 1
            continue
        split = split_name(img.stem)
        stats[split] += 1
        dest_img = OUT / "images" / split / img.name
        dest_lbl = OUT / "labels" / split / f"{img.stem}.txt"
        # YOLO layout also mirrored under yolo/
        yolo_img = OUT / "yolo" / split / "images" / img.name
        yolo_lbl = OUT / "yolo" / split / "labels" / f"{img.stem}.txt"
        link_or_copy(img, dest_img)
        link_or_copy(img, yolo_img)
        body = "\n".join(
            f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cid, xc, yc, w, h in boxes
        ) + "\n"
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        yolo_lbl.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.write_text(body, encoding="utf-8")
        yolo_lbl.write_text(body, encoding="utf-8")
        rows.append(
            {
                "filename": img.name,
                "split": split,
                "n_boxes": len(boxes),
                "source_image": str(img.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    yaml_text = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: D\n"
        "  1: d\n"
        "# D = permanent-tooth decay region (NOT ICDAS)\n"
        "# d = primary-tooth decay region (NOT ICDAS)\n"
    )
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (OUT / "yolo" / "data.yaml").write_text(
        "path: "
        + (OUT / "yolo").as_posix()
        + "\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 2\nnames:\n  0: D\n  1: d\n",
        encoding="utf-8",
    )

    readme = """# Isolated public caries detector set

Source: existing `data_external/detection/` (Zenodo 10.5281/zenodo.14827784).
Original `raw/` and `annotations/` were not overwritten.

Classes (dataset original labels, NOT ICDAS):
- `D` YOLO id 0 — permanent-tooth decay region
- `d` YOLO id 1 — primary-tooth decay region

Images are hardlinked or copied. Labels are new clipped YOLO files.
Unlabeled JPGs from the dump are excluded (not treated as ICDAS 0).
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "source": "data_external/detection (already downloaded)",
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
        "total_images_used": sum(stats.values()),
        "boxes": dict(class_counts),
        "skipped_empty_or_all_invalid": skipped_empty,
        "invalid_lines_dropped": skipped_invalid,
        "labels_missing_image": skipped_no_img,
        "seed": SEED,
        "note": "d/D are detection classes only. Not ICDAS.",
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "split", "n_boxes", "source_image"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
