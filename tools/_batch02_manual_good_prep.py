#!/usr/bin/env python3
"""Prepare batch02_manual_good candidate from visual audit. Does not train or rewrite source labels."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "detection" / "batches" / "batch02_clean"
DEST = ROOT / "data" / "detection" / "batches" / "batch02_manual_good"
REVIEWS = ROOT / "reports" / "tooth_detection_batch02_qc" / "anterior_audit" / "manual_reviews.json"
CATALOG = ROOT / "reports" / "tooth_detection_batch02_qc" / "anterior_audit" / "selection.json"
OUT_JSON = ROOT / "reports" / "tooth_detection_batch02_qc" / "anterior_audit" / "manual_qc_analysis.json"


def iou(a, b) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def parse_label(path: Path) -> tuple[list[tuple], list[str]]:
    boxes = []
    errors = []
    if not path.exists():
        return boxes, ["missing_label"]
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"row {i}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cid, xc, yc, w, h = (float(x) for x in parts)
        except ValueError:
            errors.append(f"row {i}: non-numeric")
            continue
        if int(cid) != 0:
            errors.append(f"row {i}: class {cid} != 0")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            errors.append(f"row {i}: center out of [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"row {i}: non-positive w/h")
        if xc - w / 2 < -1e-4 or yc - h / 2 < -1e-4 or xc + w / 2 > 1 + 1e-4 or yc + h / 2 > 1 + 1e-4:
            errors.append(f"row {i}: box extends outside image")
        boxes.append((int(cid), xc, yc, w, h))
    return boxes, errors


def geom(boxes) -> dict:
    xywh = [(xc, yc, w, h) for _, xc, yc, w, h in boxes]
    n = len(xywh)
    areas = [w * h for _, _, w, h in xywh]
    med = sorted(areas)[n // 2] if areas else 0.0
    max_iou = 0.0
    n50 = n70 = n_ant = n_border = n_ext = n_tiny = n_large = 0
    max_area = max(areas) if areas else 0.0
    for xc, yc, w, h in xywh:
        xmin, xmax = xc - w / 2, xc + w / 2
        ymin, ymax = yc - h / 2, yc + h / 2
        if xmin <= 1e-4 or ymin <= 1e-4 or xmax >= 1 - 1e-4 or ymax >= 1 - 1e-4:
            n_border += 1
        aspect = max(w / h, h / w) if w > 0 and h > 0 else 999
        if aspect > 5:
            n_ext += 1
        if w * h < 0.001:
            n_tiny += 1
        if med > 0 and w * h > 2.5 * med:
            n_large += 1
    for i in range(n):
        for j in range(i + 1, n):
            v = iou(xywh[i], xywh[j])
            max_iou = max(max_iou, v)
            if v >= 0.5:
                n50 += 1
            if v >= 0.7:
                n70 += 1
            axi, ayi = xywh[i][0], xywh[i][1]
            axj, ayj = xywh[j][0], xywh[j][1]
            if v >= 0.35 and 0.28 <= axi <= 0.72 and 0.28 <= axj <= 0.72:
                n_ant += 1
    return {
        "n_boxes": n,
        "max_iou": round(max_iou, 4),
        "n_iou50": n50,
        "n_iou70": n70,
        "n_anterior_overlap": n_ant,
        "n_border": n_border,
        "n_extreme": n_ext,
        "n_tiny": n_tiny,
        "n_large_vs_median": n_large,
        "max_area": round(max_area, 4),
    }


def classify(row: dict, g: dict, note: str) -> list[str]:
    tags = []
    reason = row.get("select_reason", "")
    name = row["name"].lower()
    if g["n_iou70"] >= 1:
        tags.append("duplicate boxes")
    if g["n_large_vs_median"] >= 1 and g["max_area"] >= 0.045:
        tags.append("two teeth inside one box")
    elif g["n_large_vs_median"] >= 1:
        tags.append("oversized box")
    if g["n_large_vs_median"] >= 1 and g["n_extreme"] >= 1:
        tags.append("excessive gum/palate/background")
    if g["n_border"] >= 1:
        tags.append("border-truncated tooth")
    if reason == "anterior_crowded" or g["n_anterior_overlap"] >= 3:
        tags.append("anterior crowding")
    if g["n_iou50"] >= 1:
        tags.append("overlapping boxes")
    if g["n_extreme"] >= 1 and g["n_iou70"] == 0:
        tags.append("polygon-to-rectangle conversion issue")
    if any(x in name for x in ("crop_", "flip_", "gallery", "penta_", "screen_shot", "smile")):
        tags.append("source-image issue")
    if "wrong" in note.lower() or "associat" in note.lower():
        tags.append("wrong tooth association")
    if not tags:
        tags.append("other")
    return tags


def recorded_reason(row: dict, g: dict, note: str, tags: list[str]) -> str:
    parts = []
    if note.strip():
        parts.append(f"note: {note.strip()}")
    else:
        parts.append("no free-text note")
    parts.append(f"sample={row.get('select_reason')}")
    parts.append(
        f"maxIoU={g['max_iou']} IoU50={g['n_iou50']} IoU70={g['n_iou70']} "
        f"antOv={g['n_anterior_overlap']} border={g['n_border']} "
        f"large={g['n_large_vs_median']} extreme={g['n_extreme']}"
    )
    parts.append("tags: " + "; ".join(tags))
    return " | ".join(parts)


def main() -> None:
    reviews = json.loads(REVIEWS.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in catalog}
    ratings = reviews.get("ratings", {})

    marks = Counter(v.get("mark") for v in ratings.values())
    catalog_names = {r["name"] for r in catalog}
    rated_names = set(ratings)
    missing = sorted(catalog_names - rated_names)
    extra = sorted(rated_names - catalog_names)
    unrated = [n for n in catalog_names if ratings.get(n, {}).get("mark") not in {"GOOD", "QUESTIONABLE", "BAD"}]

    groups = {"GOOD": [], "QUESTIONABLE": [], "BAD": []}
    problem_counts = defaultdict(lambda: {"QUESTIONABLE": 0, "BAD": 0})
    q_list = []
    bad_list = []
    good_val = []

    for name, rec in ratings.items():
        mark = rec.get("mark")
        row = by_name.get(name)
        if row is None:
            continue
        lab = Path(row["label_path"])
        img = Path(row["abs_path"])
        boxes, errors = parse_label(lab)
        g = geom(boxes) if boxes else {
            "n_boxes": 0, "max_iou": 0, "n_iou50": 0, "n_iou70": 0,
            "n_anterior_overlap": 0, "n_border": 0, "n_extreme": 0,
            "n_tiny": 0, "n_large_vs_median": 0, "max_area": 0,
        }
        note = rec.get("note") or ""
        tags = classify(row, g, note)
        reason = recorded_reason(row, g, note, tags)
        item = {
            "name": name,
            "split": rec.get("split") or row["split"],
            "mark": mark,
            "n_boxes": len(boxes),
            "note": note,
            "select_reason": row.get("select_reason"),
            "tags": tags,
            "recorded_reason": reason,
            "geometry": g,
            "label_errors": errors,
            "image_exists": img.exists(),
            "label_exists": lab.exists(),
            "src_image": str(img),
            "src_label": str(lab),
        }
        groups[mark].append(item)
        if mark in ("QUESTIONABLE", "BAD"):
            for t in tags:
                problem_counts[t][mark] += 1
            (q_list if mark == "QUESTIONABLE" else bad_list).append(item)
        if mark == "GOOD":
            suspicious = []
            if g["n_iou70"] >= 1:
                suspicious.append("duplicate_iou70")
            if g["n_large_vs_median"] >= 1:
                suspicious.append("oversized_vs_median")
            if g["n_extreme"] >= 1:
                suspicious.append("extreme_aspect")
            if g["n_border"] >= 3:
                suspicious.append("severe_border")
            elif g["n_border"] >= 1:
                suspicious.append("mild_border")
            if errors:
                suspicious.append("malformed")
            item["suspicious"] = suspicious
            good_val.append(item)

    # copy GOOD only
    if DEST.exists():
        shutil.rmtree(DEST)
    copied = 0
    box_total = 0
    invalid = 0
    for item in good_val:
        split = item["split"]
        img_dst = DEST / "images" / split / item["name"]
        lab_src = Path(item["src_label"])
        lab_dst = DEST / "labels" / split / (Path(item["name"]).stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["src_image"], img_dst)
        shutil.copy2(lab_src, lab_dst)
        copied += 1
        box_total += item["n_boxes"]
        if item["label_errors"]:
            invalid += 1

    yaml = (
        "# Batch 02 MANUAL GOOD — candidate subset only. Do not replace batch02_clean.\n"
        "# Source: data/detection/batches/batch02_clean (copied, not rewritten).\n"
        f"path: {DEST.as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['tooth']\n"
    )
    (DEST / "data.yaml").write_text(yaml, encoding="utf-8")
    (DEST / "README.md").write_text(
        "Candidate training subset: 57 images manually marked GOOD in the anterior/overlap visual audit.\n"
        "Copied from batch02_clean. Original Batch 02 was not modified. Do not train unless requested.\n",
        encoding="utf-8",
    )

    analysis = {
        "verify": {
            "ratings_keys": len(ratings),
            "catalog": len(catalog),
            "marks": dict(marks),
            "missing_from_catalog": missing,
            "extra_not_in_catalog": extra,
            "unrated": unrated,
            "all_100_rated": len(catalog) == 100 and not missing and not unrated,
        },
        "notes_nonempty": sum(1 for v in ratings.values() if (v.get("note") or "").strip()),
        "problem_counts": {k: dict(v) | {"Total": v["QUESTIONABLE"] + v["BAD"]} for k, v in problem_counts.items()},
        "questionable": [
            {"name": x["name"], "split": x["split"], "n_boxes": x["n_boxes"],
             "select_reason": x["select_reason"], "tags": x["tags"], "recorded_reason": x["recorded_reason"]}
            for x in sorted(q_list, key=lambda z: z["name"])
        ],
        "bad": [
            {"name": x["name"], "split": x["split"], "n_boxes": x["n_boxes"],
             "select_reason": x["select_reason"], "tags": x["tags"], "recorded_reason": x["recorded_reason"]}
            for x in sorted(bad_list, key=lambda z: z["name"])
        ],
        "good": {
            "n_images": len(good_val),
            "n_boxes": box_total,
            "copied": copied,
            "invalid_label_files": invalid,
            "splits": dict(Counter(x["split"] for x in good_val)),
            "suspicious_images": [x for x in good_val if x.get("suspicious")],
            "filenames": sorted(x["name"] for x in good_val),
            "per_image": [
                {
                    "name": x["name"],
                    "split": x["split"],
                    "n_boxes": x["n_boxes"],
                    "image_exists": x["image_exists"],
                    "label_exists": x["label_exists"],
                    "label_errors": x["label_errors"],
                    "suspicious": x.get("suspicious", []),
                    "geometry": x["geometry"],
                }
                for x in sorted(good_val, key=lambda z: z["name"])
            ],
        },
        "dest": str(DEST),
    }
    OUT_JSON.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis["verify"], indent=2))
    print("GOOD images", analysis["good"]["n_images"], "boxes", analysis["good"]["n_boxes"])
    print("invalid", invalid, "suspicious", len(analysis["good"]["suspicious_images"]))
    print("problems", analysis["problem_counts"])
    print("wrote", DEST, OUT_JSON)


if __name__ == "__main__":
    main()
