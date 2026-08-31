#!/usr/bin/env python3
"""Convert YOLOv8 tooth *polygons* to axis-aligned YOLO *boxes*.

Does not modify source polygon .txt files.
Does not train. Does not touch Batch 01 or ICDAS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp01(v: float) -> float:
    if v != v or math.isinf(v):
        raise ValueError("NaN/Inf")
    return max(0.0, min(1.0, v))


def polygon_to_xywh(coords: list[float]) -> tuple[float, float, float, float, float]:
    """Return x_center, y_center, width, height, polygon_area (shoelace, abs)."""
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError(f"need >=3 points, even count; got {len(coords)}")
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmin, xmax = clamp01(xmin), clamp01(xmax)
    ymin, ymax = clamp01(ymin), clamp01(ymax)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive box {w}x{h}")
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    poly_area = abs(area) / 2.0
    return xc, yc, w, h, poly_area


def parse_polygon_line(line: str) -> tuple[int, list[float]]:
    parts = line.split()
    if len(parts) < 7:
        raise ValueError(f"too few fields ({len(parts)})")
    cid = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    return cid, coords


def convert_label_file(src: Path, dst: Path) -> list[dict]:
    records = []
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    out_lines: list[str] = []
    for i, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        rec: dict = {"src": src.name, "line": i, "ok": False, "error": ""}
        try:
            cid, coords = parse_polygon_line(text)
            if cid != 0:
                raise ValueError(f"class_id {cid} != 0")
            xc, yc, w, h, poly_area = polygon_to_xywh(coords)
            box_area = w * h
            rec.update(
                {
                    "ok": True,
                    "class_id": cid,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                    "poly_area": poly_area,
                    "box_area": box_area,
                    "n_pts": len(coords) // 2,
                    "xmin": xc - w / 2,
                    "xmax": xc + w / 2,
                    "ymin": yc - h / 2,
                    "ymax": yc + h / 2,
                }
            )
            out_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        except Exception as exc:  # noqa: BLE001 — collect conversion errors
            rec["error"] = str(exc)
            records.append(rec)
            continue
        records.append(rec)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return records


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
    bx1, by1, bx2, by2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["box_area"] + b["box_area"] - inter
    return inter / union if union > 0 else 0.0


def rf_stem(name: str) -> str:
    stem = Path(name).stem
    marker = ".rf."
    if marker in stem:
        return stem.split(marker)[0]
    return stem


def copy_images(src_split: Path, dst_split: Path) -> int:
    dst_split.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in list_images(src_split):
        dest = dst_split / p.name
        if dest.exists():
            n += 1
            continue
        shutil.copy2(p, dest)
        n += 1
    return n


def write_data_yaml(dst: Path, copy_images: bool, src_root: Path) -> None:
    if copy_images:
        path_line = str(dst.resolve()).replace("\\", "/")
        train, val, test = "train/images", "valid/images", "test/images"
        note = "Images copied next to converted labels."
    else:
        path_line = str(src_root.resolve()).replace("\\", "/")
        train, val, test = "train/images", "valid/images", "test/images"
        note = (
            "Images stay at the original extract. Converted labels are NOT in that tree. "
            "Copy or link images into this yolo_detection folder before training, or pass "
            "an explicit labels path. See SOURCE.md."
        )
    text = f"""# Batch 02 converted whole-tooth boxes (class 0 = tooth).
# Original Roboflow polygons were NOT modified.
# {note}
path: {path_line}
train: {train}
val: {val}
test: {test}
nc: 1
names:
  0: tooth
"""
    (dst / "data.yaml").write_text(text, encoding="utf-8")


def overlay_sample(
    src_root: Path,
    dst_labels: Path,
    out_dir: Path,
    n_train: int = 10,
    n_valid: int = 5,
    n_test: int = 5,
) -> list[str]:
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    rng = random.Random(42)

    def draw_split(split: str, k: int) -> None:
        img_dir = src_root / split / "images"
        lab_dir = dst_labels / split / "labels"
        images = list_images(img_dir)
        rng.shuffle(images)
        for p in images[:k]:
            lab = lab_dir / f"{p.stem}.txt"
            data = np.fromfile(str(p), dtype=np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            if lab.exists():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, xc, yc, bw, bh = (float(x) for x in parts)
                    x1 = int(round((xc - bw / 2) * w))
                    y1 = int(round((yc - bh / 2) * h))
                    x2 = int(round((xc + bw / 2) * w))
                    y2 = int(round((yc + bh / 2) * h))
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dest = out_dir / f"{split}_{p.stem[:80]}.jpg"
            ok, enc = cv2.imencode(".jpg", im)
            if ok:
                enc.tofile(str(dest))
                written.append(str(dest))

    draw_split("train", n_train)
    draw_split("valid", n_valid)
    draw_split("test", n_test)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Original polygon YOLO root (data.yaml + splits)")
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination yolo_detection folder (converted labels only unless --copy-images)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy JPGs into dst (do not use if it would duplicate several GB)",
    )
    parser.add_argument("--qc-dir", default="", help="Write QC CSV/JSON/overlays here")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "data.yaml").exists():
        raise SystemExit(f"missing data.yaml in {src}")

    src_resolved = str(src)
    dst_resolved = str(dst)
    if dst_resolved == src_resolved or dst_resolved.startswith(src_resolved + "\\") or dst_resolved.startswith(src_resolved + "/"):
        raise SystemExit("Refusing to write converted labels inside the original polygon tree.")

    conversion_errors: list[dict] = []
    per_split = {}
    all_ok: list[dict] = []

    for split in SPLITS:
        img_dir = src / split / "images"
        poly_dir = src / split / "labels"
        box_dir = dst / split / "labels"
        images = list_images(img_dir)
        polys = sorted(poly_dir.glob("*.txt")) if poly_dir.exists() else []
        img_stems = {p.stem for p in images}
        poly_stems = {p.stem for p in polys}
        missing_labels = sorted(img_stems - poly_stems)
        orphan_labels = sorted(poly_stems - img_stems)
        n_boxes = 0
        n_files = 0
        for poly in polys:
            recs = convert_label_file(poly, box_dir / poly.name)
            n_files += 1
            for r in recs:
                r["split"] = split
                r["stem"] = poly.stem
                if r["ok"]:
                    n_boxes += 1
                    all_ok.append(r)
                else:
                    conversion_errors.append(r)
        # empty converted files for images that had empty polygons? already 1:1
        per_split[split] = {
            "images": len(images),
            "polygon_label_files": len(polys),
            "converted_label_files": n_files,
            "converted_boxes": n_boxes,
            "images_missing_polygon_labels": len(missing_labels),
            "polygon_labels_missing_images": len(orphan_labels),
            "missing_label_examples": missing_labels[:10],
            "orphan_label_examples": orphan_labels[:10],
        }
        if args.copy_images:
            copy_images(img_dir, dst / split / "images")

    write_data_yaml(dst, args.copy_images, src)

    # Validate converted files
    val = {"splits": {}, "bad_lines": [], "total_boxes": 0, "total_label_files": 0}
    for split in SPLITS:
        img_dir = src / split / "images"
        box_dir = dst / split / "labels"
        images = list_images(img_dir)
        labels = sorted(box_dir.glob("*.txt")) if box_dir.exists() else []
        img_stems = {p.stem for p in images}
        lab_stems = {p.stem for p in labels}
        n_boxes = 0
        bad = []
        for lab in labels:
            for i, line in enumerate(lab.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad.append({"file": lab.name, "line": i, "reason": f"n_fields={len(parts)}"})
                    continue
                try:
                    cid = int(float(parts[0]))
                    nums = [float(x) for x in parts[1:]]
                except ValueError:
                    bad.append({"file": lab.name, "line": i, "reason": "parse"})
                    continue
                if cid != 0:
                    bad.append({"file": lab.name, "line": i, "reason": f"class={cid}"})
                if any(x != x or math.isinf(x) for x in nums):
                    bad.append({"file": lab.name, "line": i, "reason": "nan/inf"})
                xc, yc, w, h = nums
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad.append({"file": lab.name, "line": i, "reason": "range", "vals": nums})
                if xc - w / 2 < -1e-6 or yc - h / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc + h / 2 > 1 + 1e-6:
                    bad.append({"file": lab.name, "line": i, "reason": "extends_outside"})
                n_boxes += 1
        val["splits"][split] = {
            "images": len(images),
            "converted_labels": len(labels),
            "boxes": n_boxes,
            "missing_labels": len(img_stems - lab_stems),
            "orphan_labels": len(lab_stems - img_stems),
            "bad_lines": len(bad),
        }
        val["bad_lines"].extend(bad[:50])
        val["total_boxes"] += n_boxes
        val["total_label_files"] += len(labels)
    val["conversion_errors"] = len(conversion_errors)
    val["expected_boxes"] = {"train": 42779, "valid": 4153, "test": 2010, "total": 48942}

    # QC flags (do not delete)
    flags: list[dict] = []
    overlap_pairs = 0
    boundary = 0
    by_image: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_ok:
        by_image[(r["split"], r["stem"])].append(r)
        ratio = (r["poly_area"] / r["box_area"]) if r["box_area"] > 0 else 0.0
        r["poly_box_ratio"] = ratio
        aspect = max(r["w"] / r["h"], r["h"] / r["w"]) if r["h"] > 0 and r["w"] > 0 else 999.0
        r["aspect"] = aspect
        touch = r["xmin"] <= 1e-4 or r["ymin"] <= 1e-4 or r["xmax"] >= 1 - 1e-4 or r["ymax"] >= 1 - 1e-4
        r["boundary"] = touch
        if touch:
            boundary += 1
        reasons = []
        if r["box_area"] > 0.12:
            reasons.append("extremely_large")
        if r["box_area"] < 0.001:
            reasons.append("extremely_small")
        if aspect > 5:
            reasons.append("extreme_aspect")
        if ratio < 0.45:
            reasons.append("low_polygon_to_box_ratio")
        if touch:
            reasons.append("touches_boundary")
        if reasons:
            flags.append(
                {
                    "split": r["split"],
                    "stem": r["stem"],
                    "reasons": ";".join(reasons),
                    "box_area": round(r["box_area"], 6),
                    "poly_box_ratio": round(ratio, 4),
                    "aspect": round(aspect, 3),
                    "w": round(r["w"], 4),
                    "h": round(r["h"], 4),
                }
            )

    heavy_overlap_images = 0
    max_iou_all = 0.0
    for (_split, _stem), boxes in by_image.items():
        hit = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                v = iou(boxes[i], boxes[j])
                max_iou_all = max(max_iou_all, v)
                if v >= 0.7:
                    overlap_pairs += 1
                    hit = True
                    flags.append(
                        {
                            "split": _split,
                            "stem": _stem,
                            "reasons": "heavy_overlap_iou>=0.7",
                            "iou": round(v, 4),
                        }
                    )
        if hit:
            heavy_overlap_images += 1

    # near-duplicate boxes on same image
    dup_box = 0
    for (_split, _stem), boxes in by_image.items():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) >= 0.92:
                    dup_box += 1

    areas = [r["box_area"] for r in all_ok]
    widths = [r["w"] for r in all_ok]
    heights = [r["h"] for r in all_ok]
    ratios = [r["poly_box_ratio"] for r in all_ok]
    aspects = [r["aspect"] for r in all_ok]

    def summ(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {
            "min": min(xs),
            "max": max(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
        }

    # duplicates / leakage
    stem_splits: dict[str, set[str]] = defaultdict(set)
    stem_count: Counter[str] = Counter()
    for split in SPLITS:
        for p in list_images(src / split / "images"):
            base = rf_stem(p.name)
            stem_splits[base].add(split)
            stem_count[base] += 1
    leak = {k: sorted(v) for k, v in stem_splits.items() if len(v) > 1}
    n_unique = len(stem_count)
    n_aug = sum(1 for c in stem_count.values() if c > 1)
    extra_copies = sum(c - 1 for c in stem_count.values() if c > 1)

    qc = {
        "source": str(src),
        "destination": str(dst),
        "copied_images": bool(args.copy_images),
        "per_split_conversion": per_split,
        "validation": val,
        "box_quality": {
            "n_boxes": len(all_ok),
            "normalized_area": summ(areas),
            "width": summ(widths),
            "height": summ(heights),
            "polygon_to_box_ratio": summ(ratios),
            "aspect_ratio": summ(aspects),
            "boundary_touching_boxes": boundary,
            "heavy_overlap_pairs_iou_ge_0.7": overlap_pairs,
            "images_with_heavy_overlap": heavy_overlap_images,
            "near_duplicate_box_pairs_iou_ge_0.92": dup_box,
            "max_pairwise_iou": max_iou_all,
            "flagged_rows": len(flags),
        },
        "duplicates": {
            "total_images": sum(per_split[s]["images"] for s in SPLITS),
            "unique_stems_before_rf_hash": n_unique,
            "stems_with_more_than_one_file": n_aug,
            "extra_augmented_copies": extra_copies,
            "stems_leaking_across_splits": len(leak),
            "leak_examples": {k: leak[k] for k in list(leak)[:25]},
        },
        "conversion_error_examples": conversion_errors[:20],
    }

    (dst / "conversion_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    if args.qc_dir:
        qdir = Path(args.qc_dir)
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "conversion_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        if flags:
            keys = sorted({k for row in flags for k in row})
            with (qdir / "suspicious_boxes.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(flags)
        if leak:
            with (qdir / "split_leakage_stems.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["source_stem", "splits", "n_files"])
                for k, splits in sorted(leak.items()):
                    w.writerow([k, "|".join(splits), stem_count[k]])
        overlay_sample(src, dst, qdir / "overlays")

    print(json.dumps({"validation": val, "expected_match": val["total_boxes"] == 48942}, indent=2))
    return 0 if val["total_boxes"] == 48942 and not conversion_errors and not val["bad_lines"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
