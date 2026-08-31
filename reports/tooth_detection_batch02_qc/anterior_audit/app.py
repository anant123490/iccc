#!/usr/bin/env python3
"""Targeted anterior/overlap visual audit. Read-only YOLO overlay. Notes only — no label writes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLEAN = ROOT / "data" / "detection" / "batches" / "batch02_clean"
CATALOG = HERE / "selection.json"
REVIEWS = HERE / "manual_reviews.json"
SPLITS = ("train", "valid", "test")
N_TARGET = 100


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
    n_iou50 = 0
    n_iou70 = 0
    n_ant_ov = 0
    n_border = 0
    n_extreme = 0
    n_tiny = 0
    n_large = 0
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
    for i in range(n):
        for j in range(i + 1, n):
            v = iou(boxes[i], boxes[j])
            max_iou = max(max_iou, v)
            if v >= 0.5:
                n_iou50 += 1
            if v >= 0.7:
                n_iou70 += 1
            axi, ayi, _, _ = boxes[i]
            axj, ayj, _, _ = boxes[j]
            if v >= 0.35 and 0.28 <= axi <= 0.72 and 0.28 <= axj <= 0.72:
                n_ant_ov += 1
    packing = n / (sum(areas) + 1e-6)
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
        "packing": round(packing, 3),
        "duplicate_looking": n_iou70 >= 1,
        "multi_tooth_candidate": n_large >= 1,
        "border_truncated": n_border >= 1,
        "excessive_nonttooth_proxy": n_large >= 1 and n_extreme >= 1,
    }


def scan_all() -> list[dict]:
    rows = []
    for split in SPLITS:
        img_dir = CLEAN / "images" / split
        lab_dir = CLEAN / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            stats = analyze(yolo_norm(lab))
            rec = {
                "path": str(img.relative_to(ROOT)).replace("\\", "/"),
                "abs_path": str(img),
                "label_path": str(lab),
                "split": split,
                "name": img.name,
                **stats,
            }
            rows.append(rec)
    return rows


def take_top(pool: list[dict], key: str, n: int, seen: set, reason: str, reverse: bool = True) -> list[dict]:
    ranked = sorted(pool, key=lambda r: r[key], reverse=reverse)
    out = []
    for r in ranked:
        if r["abs_path"] in seen:
            continue
        if key != "n_boxes" and r[key] <= 0 and key not in {"packing"}:
            continue
        item = dict(r)
        item["select_reason"] = reason
        out.append(item)
        seen.add(r["abs_path"])
        if len(out) >= n:
            break
    return out


def build_selection() -> list[dict]:
    pool = scan_all()
    seen: set[str] = set()
    selected: list[dict] = []
    selected += take_top(pool, "n_iou70", 12, seen, "high_iou_duplicate")
    selected += take_top(pool, "n_iou50", 18, seen, "overlap_pairs")
    selected += take_top(pool, "n_anterior_overlap", 15, seen, "anterior_crowded")
    selected += take_top(pool, "n_tiny", 10, seen, "tiny_packed")
    selected += take_top(pool, "n_boxes", 12, seen, "many_detections")
    selected += take_top(pool, "n_extreme", 10, seen, "extreme_aspect")
    selected += take_top(pool, "n_large_vs_median", 10, seen, "oversized_vs_neighbors")
    selected += take_top(pool, "n_border", 13, seen, "border_touching")
    if len(selected) < N_TARGET:
        selected += take_top(pool, "max_iou", N_TARGET - len(selected), seen, "max_iou_fill")
    selected = selected[:N_TARGET]
    for i, row in enumerate(selected, start=1):
        row["index"] = i
        row["total"] = len(selected)
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


def rating_counts(data: dict) -> dict[str, int]:
    c = {"GOOD": 0, "QUESTIONABLE": 0, "BAD": 0, "UNRATED": 0}
    for v in data.get("ratings", {}).values():
        m = v.get("mark", "")
        if m in c:
            c[m] += 1
    return c


def draw(img: Image.Image, boxes) -> Image.Image:
    im = img.convert("RGB")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 70, 255), width=4)
        d.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(0, 0, 0, 200), width=1)
    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Batch 02 anterior audit", layout="wide")
    st.title("Batch 02 — anterior / overlap visual audit")
    st.caption(
        "Existing YOLO rectangles only. GOOD / QUESTIONABLE / BAD write to "
        f"`{REVIEWS.name}` — never to dataset labels. No training."
    )

    catalog = load_catalog()
    reviews = load_reviews()
    counts = rating_counts(reviews)
    unrated = len(catalog) - sum(counts[k] for k in ("GOOD", "QUESTIONABLE", "BAD"))

    st.sidebar.write("Manual marks (notes file only)")
    st.sidebar.write(f"GOOD: {counts['GOOD']}")
    st.sidebar.write(f"QUESTIONABLE: {counts['QUESTIONABLE']}")
    st.sidebar.write(f"BAD: {counts['BAD']}")
    st.sidebar.write(f"Unrated in catalog: {unrated}")

    reasons = ["All"] + sorted({r["select_reason"] for r in catalog})
    filt = st.sidebar.selectbox("Filter by sample reason", reasons)
    view = catalog if filt == "All" else [r for r in catalog if r["select_reason"] == filt]
    if not view:
        view = catalog

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
    img_path = Path(row["abs_path"])
    lab_path = Path(row["label_path"])
    image = Image.open(img_path)
    w, h = image.size
    boxes = parse_yolo_px(lab_path, w, h)

    st.subheader(f"Image {st.session_state.idx + 1} / {len(view)}  (catalog {row['index']} / {row['total']})")
    a, b, c, d = st.columns(4)
    a.metric("Split", row["split"])
    b.metric("Tooth boxes", row["n_boxes"])
    c.metric("Class", "tooth")
    d.metric("Max IoU", row["max_iou"])
    st.write(f"**Filename:** `{row['name']}`")
    st.write(
        f"**Sample reason:** `{row['select_reason']}` · "
        f"IoU≥0.5 pairs={row['n_iou50']} · IoU≥0.7 pairs={row['n_iou70']} · "
        f"anterior-overlap pairs={row['n_anterior_overlap']} · "
        f"border boxes={row['n_border']} · oversized vs median={row['n_large_vs_median']}"
    )
    st.image(np.array(draw(image, boxes)), caption="Existing YOLO rectangles (green)", use_container_width=True)

    key = row["name"]
    prev = reviews.get("ratings", {}).get(key, {})
    note_default = prev.get("note", "")
    st.markdown("### Manual audit (does not change labels)")
    note = st.text_input("Short note", value=note_default, key=f"note_{key}")
    b1, b2, b3 = st.columns(3)
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
        st.session_state.idx = (st.session_state.idx + 1) % len(view)

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
        st.info(f"Current mark: **{prev['mark']}** — {prev.get('note', '')}")


if __name__ == "__main__":
    main()
