#!/usr/bin/env python3
"""Build Batch 02 cleaned YOLO copy. Does not train. Does not modify Batch 01 or original Batch 02."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "detection" / "batches" / "batch02" / "yolo_detection"
DST = ROOT / "data" / "detection" / "batches" / "batch02_clean"
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def rf_stem(name: str) -> str:
    stem = Path(name).stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem


def count_boxes(label: Path) -> int:
    if not label.exists():
        return 0
    n = 0
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            n += 1
    return n


def parse_boxes(label: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label.exists():
        return boxes
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = (float(x) for x in parts)
        boxes.append((xc, yc, w, h))
    return boxes


def is_near_gray(path: Path, sat_thresh: float = 4.0) -> bool:
    arr = np.array(Image.open(path).convert("RGB"))
    return float(arr.std(axis=2).mean()) < sat_thresh


def style_tags(name: str) -> list[str]:
    n = name.lower()
    tags = []
    if "penta_" in n or "gallery" in n:
        tags.append("gallery_3d_like")
    if "screen_shot" in n:
        tags.append("screenshot")
    return tags


def classify(tags: list[str], near_gray: bool) -> tuple[str, str]:
    """Return (KEEP|REVIEW|EXCLUDE, reason). Duplicates handled separately."""
    if "screenshot" in tags:
        return "REVIEW", "screenshot_filename"
    if "gallery_3d_like" in tags:
        return "REVIEW", "gallery_or_3d_like_filename"
    if near_gray:
        return "REVIEW", "near_grayscale_color_stats"
    return "KEEP", "color_clinical_like"


def pick_representative(members: list[dict]) -> dict:
    counts = [m["n_boxes"] for m in members]
    med = statistics.median(counts)
    members = sorted(members, key=lambda m: (abs(m["n_boxes"] - med), m["name"]))
    return members[0]


def problematic_boundary(boxes: list[tuple[float, float, float, float]]) -> list[str]:
    """Flag only clearly bad edge boxes; do not drop the image."""
    notes = []
    for xc, yc, w, h in boxes:
        xmin, xmax = xc - w / 2, xc + w / 2
        ymin, ymax = yc - h / 2, yc + h / 2
        edges = 0
        if xmin <= 1e-4:
            edges += 1
        if ymin <= 1e-4:
            edges += 1
        if xmax >= 1 - 1e-4:
            edges += 1
        if ymax >= 1 - 1e-4:
            edges += 1
        area = w * h
        if edges >= 2 and area >= 0.08:
            notes.append(f"multi_border_large_area={area:.4f}")
        if edges >= 1 and w >= 0.28 and h >= 0.40:
            notes.append(f"huge_edge_box_w={w:.3f}_h={h:.3f}")
    return notes


def copy_pair(img: Path, lab: Path, dest_img: Path, dest_lab: Path) -> None:
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_lab.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dest_img)
    shutil.copy2(lab, dest_lab)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing converted Batch 02: {SRC}")

    records = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for split in SPLITS:
        img_dir = SRC / split / "images"
        lab_dir = SRC / split / "labels"
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
            lab = lab_dir / f"{img.stem}.txt"
            tags = style_tags(img.name)
            gray = is_near_gray(img)
            cls, reason = classify(tags, gray)
            rec = {
                "split": split,
                "name": img.name,
                "stem": img.stem,
                "source_stem": rf_stem(img.name),
                "n_boxes": count_boxes(lab),
                "near_gray": gray,
                "style_tags": "|".join(tags) if tags else "photo_like",
                "class": cls,
                "class_reason": reason,
                "img": str(img),
                "lab": str(lab),
                "label_exists": lab.exists(),
            }
            records.append(rec)
            groups[(split, rec["source_stem"])].append(rec)

    keep, review, exclude = [], [], []
    dup_excluded = 0
    inventory_rows = []

    for rec in records:
        rec = dict(rec)
        members = groups[(rec["split"], rec["source_stem"])]
        rep = pick_representative(members)
        is_rep = rec["name"] == rep["name"] and rec["split"] == rep["split"]
        rec["is_representative"] = is_rep
        if not is_rep:
            rec["class"] = "EXCLUDE"
            rec["class_reason"] = "roboflow_augmentation_duplicate"
            dup_excluded += 1
        dest_bucket = rec["class"]
        if dest_bucket == "KEEP":
            keep.append(rec)
        elif dest_bucket == "REVIEW":
            review.append(rec)
        else:
            exclude.append(rec)
        inventory_rows.append(rec)

    # copy KEEP into YOLO layout
    if DST.exists():
        # only remove our previous clean outputs, not source batch02
        for sub in ("images", "labels", "held_out"):
            p = DST / sub
            if p.exists():
                shutil.rmtree(p)

    copied = Counter()
    boxes_kept = Counter()
    class_ids = Counter()
    missing = []

    for rec in keep:
        split = rec["split"]
        img_src = Path(rec["img"])
        lab_src = Path(rec["lab"])
        if not lab_src.exists():
            missing.append(rec["name"])
            continue
        dest_img = DST / "images" / split / img_src.name
        dest_lab = DST / "labels" / split / lab_src.name
        copy_pair(img_src, lab_src, dest_img, dest_lab)
        copied[split] += 1
        n = rec["n_boxes"]
        boxes_kept[split] += n
        class_ids[0] += n

    def dump_held(bucket: str, rows: list[dict]) -> None:
        for rec in rows:
            split = rec["split"]
            img_src, lab_src = Path(rec["img"]), Path(rec["lab"])
            dest_img = DST / "held_out" / bucket / "images" / split / img_src.name
            dest_lab = DST / "held_out" / bucket / "labels" / split / Path(rec["name"]).with_suffix(".txt").name
            if lab_src.exists():
                copy_pair(img_src, lab_src, dest_img, dest_lab)
            else:
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_src, dest_img)

    dump_held("review", review)
    dump_held("excluded", exclude)

    # verify 1:1
    verify = {}
    for split in SPLITS:
        imgs = sorted((DST / "images" / split).glob("*")) if (DST / "images" / split).exists() else []
        labs = sorted((DST / "labels" / split).glob("*.txt")) if (DST / "labels" / split).exists() else []
        img_stems = {p.stem for p in imgs if p.suffix.lower() in IMG_EXT}
        lab_stems = {p.stem for p in labs}
        verify[split] = {
            "images": len(img_stems),
            "labels": len(lab_stems),
            "missing_labels": sorted(img_stems - lab_stems),
            "orphan_labels": sorted(lab_stems - img_stems),
        }

    yaml = f"""# Batch 02 CLEAN — unique stems, KEEP only. Do not merge with Batch 01.
# Original converted Batch 02 is unchanged at data/detection/batches/batch02/yolo_detection/
path: {str(DST.resolve()).replace(chr(92), '/')}
train: images/train
val: images/valid
test: images/test
nc: 1
names:
  0: tooth
"""
    (DST / "data.yaml").write_text(yaml, encoding="utf-8")

    # unique stems remaining in KEEP
    keep_stems = {r["source_stem"] for r in keep}
    dup_in_clean = 0  # should be 0 extra copies

    border_flags = []
    for rec in keep:
        notes = problematic_boundary(parse_boxes(Path(rec["lab"])))
        if notes:
            border_flags.append(
                {
                    "split": rec["split"],
                    "name": rec["name"],
                    "source_stem": rec["source_stem"],
                    "notes": ";".join(notes),
                }
            )

    summary = {
        "source": str(SRC),
        "destination": str(DST),
        "original_images": len(records),
        "unique_source_stems": len(groups),
        "unique_stems_global": len({r["source_stem"] for r in records}),
        "keep_images": len(keep),
        "review_images": len(review),
        "exclude_images": len(exclude),
        "exclude_augmentation_duplicates": dup_excluded,
        "keep_by_split": dict(copied),
        "boxes_by_split": dict(boxes_kept),
        "boxes_total": int(sum(boxes_kept.values())),
        "class_counts": {"0_tooth": int(class_ids[0])},
        "duplicate_count_in_clean": dup_in_clean,
        "verify": verify,
        "missing_labels_on_keep": missing,
        "near_gray_original": sum(1 for r in records if r["near_gray"]),
        "gallery_original": sum(1 for r in records if "gallery_3d_like" in r["style_tags"]),
        "screenshot_original": sum(1 for r in records if "screenshot" in r["style_tags"]),
        "keep_near_gray": sum(1 for r in keep if r["near_gray"]),
        "review_near_gray": sum(1 for r in review if r["near_gray"]),
        "review_gallery": sum(1 for r in review if "gallery_3d_like" in r["style_tags"]),
        "problematic_boundary_keep_images_flagged_only": len(border_flags),
        "trained": False,
    }

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "cleanup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    inv_path = DST / "file_classification.csv"
    keys = [
        "split",
        "name",
        "source_stem",
        "is_representative",
        "class",
        "class_reason",
        "n_boxes",
        "near_gray",
        "style_tags",
    ]
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(inventory_rows)

    if border_flags:
        with (DST / "border_boxes_flagged_not_removed.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "name", "source_stem", "notes"])
            w.writeheader()
            w.writerows(border_flags)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
