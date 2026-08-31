#!/usr/bin/env python3
"""Batch 02 Round 2 visual audit. Read-only overlays. Ratings never write CLEAN labels."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
QC = HERE.parent
ROOT = HERE.parents[2]
CLEAN = ROOT / "data" / "detection" / "batches" / "batch02_clean"
DEST = ROOT / "data" / "detection" / "batches" / "batch02_manual_round2"
ROUND1_REVIEWS = QC / "anterior_audit" / "manual_reviews.json"
ROUND1_CATALOG = QC / "anterior_audit" / "selection.json"
CATALOG = QC / "round2_catalog.json"
REVIEWS = HERE / "manual_reviews_round2.json"
REPORT = ROOT / "reports" / "BATCH02_MANUAL_QC_ROUND2_REPORT.md"
SPLITS = ("train", "valid", "test")
N_TARGET = 100
SEED = 20260827
B01_LABELS = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01" / "labels"


def yolo_norm(label: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label.exists():
        return boxes
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = (float(x) for x in parts)
        boxes.append((xc, yc, bw, bh))
    return boxes


def parse_yolo_px(label: Path, w: int, h: int):
    out = []
    for xc, yc, bw, bh in yolo_norm(label):
        x1 = int(round((xc - bw / 2) * w))
        y1 = int(round((yc - bh / 2) * h))
        x2 = int(round((xc + bw / 2) * w))
        y2 = int(round((yc + bh / 2) * h))
        out.append((x1, y1, x2, y2))
    return out


def parse_label(path: Path) -> tuple[list[tuple], list[str]]:
    boxes, errors = [], []
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
        boxes.append((int(cid), xc, yc, w, h))
    return boxes, errors


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


def analyze(boxes: list[tuple[float, float, float, float]]) -> dict:
    n = len(boxes)
    areas = [w * h for _, _, w, h in boxes]
    med = float(np.median(areas)) if areas else 0.0
    max_iou = 0.0
    n_iou50 = n_iou70 = n_ant_ov = n_border = n_extreme = n_tiny = n_large = 0
    n_post = 0
    for xc, yc, w, h in boxes:
        xmin, xmax = xc - w / 2, xc + w / 2
        ymin, ymax = yc - h / 2, yc + h / 2
        if xmin <= 1e-4 or ymin <= 1e-4 or xmax >= 1 - 1e-4 or ymax >= 1 - 1e-4:
            n_border += 1
        aspect = max(w / h, h / w) if w > 0 and h > 0 else 999
        if aspect > 5:
            n_extreme += 1
        if w * h < 0.001:
            n_tiny += 1
        if med > 0 and (w * h) > 2.5 * med:
            n_large += 1
        if xc <= 0.32 or xc >= 0.68:
            n_post += 1
    for i in range(n):
        for j in range(i + 1, n):
            v = iou(boxes[i], boxes[j])
            max_iou = max(max_iou, v)
            if v >= 0.5:
                n_iou50 += 1
            if v >= 0.7:
                n_iou70 += 1
            axi = boxes[i][0]
            axj = boxes[j][0]
            if v >= 0.35 and 0.28 <= axi <= 0.72 and 0.28 <= axj <= 0.72:
                n_ant_ov += 1
    calm = int(n_iou50 == 0 and n_ant_ov == 0 and n_iou70 == 0 and n_extreme == 0)
    return {
        "n_boxes": n,
        "max_iou": round(max_iou, 4),
        "n_iou50": n_iou50,
        "n_iou70": n_iou70,
        "n_anterior_overlap": n_ant_ov,
        "n_border": n_border,
        "n_extreme": n_extreme,
        "n_tiny": n_tiny,
        "n_large_vs_median": n_large,
        "n_posterior": n_post,
        "calm": calm,
    }


def round1_names() -> set[str]:
    names: set[str] = set()
    if ROUND1_REVIEWS.exists():
        data = json.loads(ROUND1_REVIEWS.read_text(encoding="utf-8"))
        names.update(data.get("ratings", {}).keys())
        for v in data.get("ratings", {}).values():
            if v.get("name"):
                names.add(v["name"])
    if ROUND1_CATALOG.exists():
        for row in json.loads(ROUND1_CATALOG.read_text(encoding="utf-8")):
            names.add(row["name"])
    return names


def scan_pool() -> list[dict]:
    skip = round1_names()
    rows = []
    for split in SPLITS:
        img_dir = CLEAN / "images" / split
        lab_dir = CLEAN / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if img.name in skip:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            stats = analyze(yolo_norm(lab))
            rows.append(
                {
                    "path": str(img.relative_to(ROOT)).replace("\\", "/"),
                    "abs_path": str(img),
                    "label_path": str(lab),
                    "split": split,
                    "name": img.name,
                    **stats,
                }
            )
    return rows


def take_top(pool: list[dict], key: str, n: int, seen: set[str], reason: str) -> list[dict]:
    ranked = sorted(pool, key=lambda r: r[key], reverse=True)
    out = []
    for r in ranked:
        if r["name"] in seen:
            continue
        if key not in {"n_boxes", "n_posterior"} and r[key] <= 0:
            continue
        item = dict(r)
        item["select_reason"] = reason
        out.append(item)
        seen.add(r["name"])
        if len(out) >= n:
            break
    return out


def take_random(pool: list[dict], n: int, seen: set[str], reason: str, rng: random.Random) -> list[dict]:
    cand = [r for r in pool if r["name"] not in seen]
    rng.shuffle(cand)
    out = []
    for r in cand:
        item = dict(r)
        item["select_reason"] = reason
        out.append(item)
        seen.add(r["name"])
        if len(out) >= n:
            break
    return out


def build_selection() -> list[dict]:
    pool = scan_pool()
    rng = random.Random(SEED)
    seen: set[str] = set()
    selected: list[dict] = []
    selected += take_top(pool, "n_anterior_overlap", 12, seen, "anterior_crowded")
    selected += take_top(pool, "n_iou50", 12, seen, "overlap_pairs")
    selected += take_top(pool, "n_tiny", 8, seen, "tiny_packed")
    selected += take_top(pool, "n_boxes", 10, seen, "many_detections")
    selected += take_top(pool, "n_border", 12, seen, "border_touching")
    selected += take_top(pool, "n_large_vs_median", 10, seen, "oversized_vs_neighbors")
    selected += take_top(pool, "n_extreme", 8, seen, "extreme_aspect")
    post_pool = [r for r in pool if r["n_posterior"] >= 4 and r["n_anterior_overlap"] == 0]
    selected += take_top(post_pool, "n_posterior", 14, seen, "normal_posterior")
    calm = [r for r in pool if r["calm"] == 1 and 8 <= r["n_boxes"] <= 24]
    selected += take_random(calm, 14, seen, "random_normal", rng)
    if len(selected) < N_TARGET:
        selected += take_random(pool, N_TARGET - len(selected), seen, "random_fill", rng)
    selected = selected[:N_TARGET]
    for i, row in enumerate(selected, start=1):
        row["index"] = i
        row["total"] = len(selected)
        row["round"] = 2
    return selected


def load_catalog() -> list[dict]:
    if CATALOG.exists():
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = build_selection()
    CATALOG.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def load_reviews() -> dict:
    if REVIEWS.exists():
        return json.loads(REVIEWS.read_text(encoding="utf-8"))
    return {"updated": None, "ratings": {}}


def save_reviews(data: dict) -> None:
    data["updated"] = datetime.now(timezone.utc).isoformat()
    REVIEWS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rating_counts(data: dict, catalog: list[dict]) -> dict[str, int]:
    c = {"GOOD": 0, "QUESTIONABLE": 0, "BAD": 0}
    for v in data.get("ratings", {}).values():
        m = v.get("mark", "")
        if m in c:
            c[m] += 1
    rated = c["GOOD"] + c["QUESTIONABLE"] + c["BAD"]
    c["REMAINING"] = len(catalog) - rated
    return c


def count_boxes(label: Path) -> int:
    boxes, _ = parse_label(label)
    return len(boxes)


def source_stem(name: str) -> str:
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem


def batch01_stats() -> dict:
    n_img = n_box = 0
    names = []
    if B01_LABELS.exists():
        for p in B01_LABELS.rglob("*.txt"):
            n_img += 1
            names.append(p.name.replace(".txt", ""))
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    n_box += 1
    return {"images": n_img, "boxes": n_box, "stems": names}


def export_if_complete(catalog: list[dict], reviews: dict) -> dict | None:
    names = {r["name"] for r in catalog}
    rated = {
        k: v
        for k, v in reviews.get("ratings", {}).items()
        if k in names and v.get("mark") in {"GOOD", "QUESTIONABLE", "BAD"}
    }
    if len(rated) < len(catalog) or len(catalog) != N_TARGET:
        return None
    if DEST.exists():
        shutil.rmtree(DEST)
    buckets = {"GOOD": "good", "QUESTIONABLE": "questionable", "BAD": "bad"}
    copied = {k: [] for k in buckets}
    box_counts = {k: 0 for k in buckets}
    good_errors = []
    by_name = {r["name"]: r for r in catalog}
    for name, rec in rated.items():
        row = by_name[name]
        folder = buckets[rec["mark"]]
        img_src = Path(row["abs_path"])
        lab_src = Path(row["label_path"])
        img_dst = DEST / folder / "images" / name
        lab_dst = DEST / folder / "labels" / f"{Path(name).stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lab_src, lab_dst)
        n_boxes, errs = parse_label(lab_src)
        copied[rec["mark"]].append(name)
        box_counts[rec["mark"]] += len(n_boxes)
        if rec["mark"] == "GOOD":
            if not img_src.exists() or not lab_src.exists() or errs:
                good_errors.append({"name": name, "errors": errs})
    for k in copied:
        copied[k].sort()
    summary = {
        "exported": True,
        "dest": str(DEST),
        "counts": {k: len(copied[k]) for k in copied},
        "boxes": box_counts,
        "filenames": copied,
        "good_invalid": good_errors,
        "good_valid": len(copied["GOOD"]) - len(good_errors),
    }
    write_report(catalog, reviews, summary)
    (DEST / "README.md").write_text(
        "Round 2 manual ratings copied from batch02_clean. Originals not moved or rewritten.\n",
        encoding="utf-8",
    )
    return summary


def write_report(catalog: list[dict], reviews: dict, summary: dict | None) -> None:
    r1 = json.loads(ROUND1_REVIEWS.read_text(encoding="utf-8")) if ROUND1_REVIEWS.exists() else {"ratings": {}}
    r1c = Counter(v.get("mark") for v in r1.get("ratings", {}).values())
    counts = rating_counts(reviews, catalog)
    reasons = Counter(r.get("select_reason") for r in catalog)
    overlap = sorted({r["name"] for r in catalog} & round1_names())
    lines = [
        "# Batch 02 Manual QC Round 2",
        "",
        "**Mode:** Review preparation / copy after complete ratings. **No training.** "
        "Round 1 files, Batch 01, ICDAS, and Batch 02 annotations were not overwritten.",
        "",
        "## Catalog",
        "",
        f"- File: `reports/tooth_detection_batch02_qc/round2_catalog.json`",
        f"- Images: **{len(catalog)}** unique, excluded Round 1 ({len(round1_names())} names).",
        f"- Overlap with Round 1: **{len(overlap)}** (must be 0).",
        f"- Viewer: `reports/tooth_detection_batch02_qc/round2_browser_audit/app.py`",
        f"- Ratings file: `reports/tooth_detection_batch02_qc/round2_browser_audit/manual_reviews_round2.json`",
        "",
        "### Sampling mix",
        "",
        "| Reason | Count |",
        "|--------|------:|",
    ]
    for k, v in sorted(reasons.items()):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Manual ratings",
        "",
        f"| Mark | Count |",
        f"|------|------:|",
        f"| GOOD | {counts['GOOD']} |",
        f"| QUESTIONABLE | {counts['QUESTIONABLE']} |",
        f"| BAD | {counts['BAD']} |",
        f"| Remaining | {counts['REMAINING']} |",
        "",
    ]
    if summary is None:
        lines += [
            "Round 2 ratings are **not complete**. Folder segregation has **not** run.",
            "Common issues cannot be summarized from human marks yet. Catalog is biased toward a mix of hard and normal views.",
            "",
            "## Comparison with Round 1",
            "",
            f"Round 1 (100 images): GOOD {r1c.get('GOOD', 0)}, QUESTIONABLE {r1c.get('QUESTIONABLE', 0)}, BAD {r1c.get('BAD', 0)}.",
            "Round 2: pending.",
            "",
            "## Recommendation",
            "",
            "**MORE REVIEW REQUIRED** — rate all 100 images in the Round 2 viewer. Do not merge gold sets yet.",
            "",
        ]
    else:
        lines += [
            f"Exported (copy only) to `{DEST.as_posix()}/`.",
            "",
            "| Category | Images | Tooth boxes |",
            "|----------|-------:|------------:|",
            f"| GOOD | {summary['counts']['GOOD']} | {summary['boxes']['GOOD']} |",
            f"| QUESTIONABLE | {summary['counts']['QUESTIONABLE']} | {summary['boxes']['QUESTIONABLE']} |",
            f"| BAD | {summary['counts']['BAD']} | {summary['boxes']['BAD']} |",
            "",
            f"GOOD validation: {summary['good_valid']}/{summary['counts']['GOOD']} files valid. "
            f"Invalid: {len(summary['good_invalid'])}.",
            "",
            "### GOOD filenames",
            "",
        ]
        for n in summary["filenames"]["GOOD"]:
            lines.append(f"- `{n}`")
        lines += ["", "### QUESTIONABLE filenames", ""]
        for n in summary["filenames"]["QUESTIONABLE"]:
            lines.append(f"- `{n}`")
        lines += ["", "### BAD filenames", ""]
        for n in summary["filenames"]["BAD"]:
            lines.append(f"- `{n}`")
        lines += [
            "",
            "## Recurring annotation issues (Round 2 BAD/QUESTIONABLE sample reasons)",
            "",
        ]
        by_name = {r["name"]: r for r in catalog}
        issue = Counter()
        for name, rec in reviews.get("ratings", {}).items():
            if rec.get("mark") in {"QUESTIONABLE", "BAD"} and name in by_name:
                issue[by_name[name].get("select_reason", "other")] += 1
        lines.append("| Sample reason among Q+BAD | Count |")
        lines.append("|---------------------------|------:|")
        for k, v in issue.most_common():
            lines.append(f"| `{k}` | {v} |")
        lines += [
            "",
            "## Comparison with Round 1",
            "",
            f"Round 1: GOOD {r1c.get('GOOD', 0)} / Q {r1c.get('QUESTIONABLE', 0)} / BAD {r1c.get('BAD', 0)} "
            f"(57 GOOD → 1415 boxes in `batch02_manual_good`).",
            f"Round 2: GOOD {summary['counts']['GOOD']} / Q {summary['counts']['QUESTIONABLE']} / "
            f"BAD {summary['counts']['BAD']}.",
            "",
        ]
        rec = (
            "READY FOR GOLD DATASET MERGE"
            if counts["REMAINING"] == 0 and summary["counts"]["GOOD"] >= 1
            else "MORE REVIEW REQUIRED"
        )
        # Merge readiness: complete review, but still don't merge in this task
        lines += [
            "## Recommendation",
            "",
            f"**{rec}** (compute only — datasets were **not** merged; **no training**).",
            "",
        ]
    gold = gold_summary_text(summary)
    lines += ["## Gold dataset summary (no merge)", "", gold]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gold_summary_text(summary: dict | None) -> str:
    b01 = batch01_stats()
    r1_good_names = []
    r1_boxes = 1415
    if ROUND1_REVIEWS.exists():
        data = json.loads(ROUND1_REVIEWS.read_text(encoding="utf-8"))
        r1_good_names = [k for k, v in data.get("ratings", {}).items() if v.get("mark") == "GOOD"]
    r2_good = summary["filenames"]["GOOD"] if summary else []
    r2_boxes = summary["boxes"]["GOOD"] if summary else 0
    all_names = r1_good_names + r2_good
    dup_names = len(all_names) - len(set(all_names))
    stems = [source_stem(n) for n in all_names]
    dup_stems = len(stems) - len(set(stems))
    b01_stems = set(b01["stems"])
    overlap_b01 = sorted(set(stems) & b01_stems)
    pot_img = b01["images"] + len(set(r1_good_names)) + len(set(r2_good))
    pot_box = b01["boxes"] + r1_boxes + r2_boxes
    rec = "MORE REVIEW REQUIRED" if summary is None else "READY FOR GOLD DATASET MERGE"
    return "\n".join(
        [
            f"Batch 01 human GT:",
            f"Images = {b01['images']}",
            f"Boxes = {b01['boxes']}",
            "",
            "Round 1 GOOD:",
            "Images = 57",
            "Boxes = 1415",
            "",
            "Round 2 GOOD:",
            f"Images = {len(r2_good)}",
            f"Boxes = {r2_boxes}",
            "",
            f"Duplicate images (R1 GOOD ∩ R2 GOOD filenames) = {dup_names}",
            f"Duplicate source stems (R1+R2 GOOD) = {dup_stems}",
            f"Filename/stem overlap with Batch 01 = {len(overlap_b01)}",
            "",
            "Potential Gold Detector Dataset (union, not created):",
            f"Images = {pot_img}",
            f"Boxes = {pot_box}",
            "",
            "Recommendation:",
            rec,
        ]
    )


def draw(img: Image.Image, boxes) -> Image.Image:
    im = img.convert("RGB")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 70, 255), width=4)
        d.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(0, 0, 0, 200), width=1)
    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Batch 02 Round 2 audit", layout="wide")
    st.title("Batch 02 — Round 2 visual audit")
    st.caption(
        "100 NEW images (not in Round 1). Existing YOLO rectangles only. "
        f"Ratings → `{REVIEWS.name}`. After 100/100, copies go to batch02_manual_round2/. "
        "CLEAN labels are never rewritten. No training."
    )
    catalog = load_catalog()
    reviews = load_reviews()
    counts = rating_counts(reviews, catalog)

    st.sidebar.markdown("**Progress**")
    st.sidebar.write(f"GOOD: {counts['GOOD']}")
    st.sidebar.write(f"QUESTIONABLE: {counts['QUESTIONABLE']}")
    st.sidebar.write(f"BAD: {counts['BAD']}")
    st.sidebar.write(f"Remaining: {counts['REMAINING']}")
    reasons = ["All"] + sorted({r["select_reason"] for r in catalog})
    filt = st.sidebar.selectbox("Filter by sample reason", reasons)
    view = catalog if filt == "All" else [r for r in catalog if r["select_reason"] == filt]
    if not view:
        view = catalog

    if counts["REMAINING"] == 0:
        summary = export_if_complete(catalog, reviews)
        if summary:
            st.sidebar.success("100/100 exported (copy only).")
        if st.sidebar.button("Re-export folders"):
            export_if_complete(catalog, reviews)
            st.rerun()

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if st.session_state.idx >= len(view):
        st.session_state.idx = 0

    top = st.columns([1, 1, 2])
    with top[0]:
        if st.button("Previous"):
            st.session_state.idx = (st.session_state.idx - 1) % len(view)
            st.rerun()
    with top[1]:
        if st.button("Next"):
            st.session_state.idx = (st.session_state.idx + 1) % len(view)
            st.rerun()

    row = view[st.session_state.idx]
    image = Image.open(row["abs_path"])
    w, h = image.size
    boxes = parse_yolo_px(Path(row["label_path"]), w, h)

    st.subheader(f"Image {st.session_state.idx + 1} / {len(view)}  (catalog {row['index']} / {row['total']})")
    a, b, c, d = st.columns(4)
    a.metric("Split", row["split"])
    b.metric("Tooth boxes", row["n_boxes"])
    c.metric("Class", "tooth")
    d.metric("Max IoU", row["max_iou"])
    st.write(f"**Filename:** `{row['name']}`")
    st.write(f"**Sample reason:** `{row['select_reason']}`")
    st.image(np.array(draw(image, boxes)), caption="Existing YOLO rectangles (green)", width="stretch")

    key = row["name"]
    prev = reviews.get("ratings", {}).get(key, {})
    st.markdown("### Manual rating (does not change labels)")
    note = st.text_input("Optional short note", value=prev.get("note", ""), key=f"note_{key}")

    def mark(val: str) -> None:
        data = load_reviews()
        data.setdefault("ratings", {})[key] = {
            "mark": val,
            "note": note,
            "split": row["split"],
            "name": row["name"],
            "n_boxes": row["n_boxes"],
            "select_reason": row["select_reason"],
        }
        save_reviews(data)
        catalog_local = load_catalog()
        export_if_complete(catalog_local, data)
        st.session_state.idx = (st.session_state.idx + 1) % len(view)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("GOOD"):
            mark("GOOD")
            st.rerun()
    with b2:
        if st.button("QUESTIONABLE"):
            mark("QUESTIONABLE")
            st.rerun()
    with b3:
        if st.button("BAD"):
            mark("BAD")
            st.rerun()
    if prev.get("mark"):
        st.info(f"Current mark: **{prev['mark']}**")


if __name__ == "__main__":
    main()
