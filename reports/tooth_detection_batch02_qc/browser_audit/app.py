#!/usr/bin/env python3
"""Streamlit viewer for existing Batch 02 CLEAN YOLO rectangles. Audit only — no training, no label writes."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLEAN = ROOT / "data" / "detection" / "batches" / "batch02_clean"
CLASS_CSV = CLEAN / "file_classification.csv"
CATALOG = HERE / "selection.json"
SPLITS = ("train", "valid", "test")
N_TARGET = 100
SEED = 42


def parse_yolo(label: Path, w: int, h: int) -> list[tuple[int, int, int, int, float, float]]:
    """Return pixel xyxy plus width, height (normalized) from existing label file."""
    boxes = []
    if not label.exists():
        return boxes
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, xc, yc, bw, bh = (float(x) for x in parts)
        if int(cid) != 0:
            continue
        x1 = int(round((xc - bw / 2) * w))
        y1 = int(round((yc - bh / 2) * h))
        x2 = int(round((xc + bw / 2) * w))
        y2 = int(round((yc + bh / 2) * h))
        boxes.append((x1, y1, x2, y2, bw, bh))
    return boxes


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


def flags_for_boxes(boxes: list[tuple[float, float, float, float]]) -> set[str]:
    f: set[str] = set()
    for xc, yc, w, h in boxes:
        xmin, xmax = xc - w / 2, xc + w / 2
        ymin, ymax = yc - h / 2, yc + h / 2
        if xmin <= 1e-4 or ymin <= 1e-4 or xmax >= 1 - 1e-4 or ymax >= 1 - 1e-4:
            f.add("border")
        area = w * h
        if area < 0.001:
            f.add("tiny")
        aspect = max(w / h, h / w) if w > 0 and h > 0 else 999
        if aspect > 5:
            f.add("extreme_aspect")
            f.add("low_fill")  # AABB on rotated teeth; no polygon rewrite
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) >= 0.5:
                f.add("overlap")
                break
    return f


def scan_split(base: Path, split: str, bucket: str) -> list[dict]:
    img_dir = base / "images" / split
    lab_dir = base / "labels" / split
    if not img_dir.exists():
        return []
    rows = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lab = lab_dir / f"{img.stem}.txt"
        boxes = yolo_norm(lab)
        fl = flags_for_boxes(boxes)
        rows.append(
            {
                "path": str(img.relative_to(ROOT)).replace("\\", "/"),
                "abs_path": str(img),
                "label_path": str(lab),
                "split": split,
                "name": img.name,
                "n_boxes": len(boxes),
                "bucket": bucket,
                "flags": sorted(fl),
            }
        )
    return rows


def load_class_map() -> dict[str, dict]:
    import csv

    out = {}
    if not CLASS_CSV.exists():
        return out
    with CLASS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = row
    return out


def build_selection() -> list[dict]:
    rng = random.Random(SEED)
    keep = []
    for split in SPLITS:
        keep.extend(scan_split(CLEAN, split, "keep"))
    review = []
    rev_root = CLEAN / "held_out" / "review"
    for split in SPLITS:
        review.extend(scan_split(rev_root, split, "review"))

    cmap = load_class_map()
    for r in keep + review:
        meta = cmap.get(r["name"], {})
        tags = (meta.get("style_tags") or "").lower()
        if meta.get("near_gray", "").lower() in {"true", "1"}:
            r["flags"] = sorted(set(r["flags"]) | {"near_gray"})
        if "gallery" in tags or "3d" in tags:
            r["flags"] = sorted(set(r["flags"]) | {"gallery"})
        r["class_csv"] = meta.get("class", "")

    by_flag: dict[str, list[dict]] = {
        "many": [x for x in keep if x["n_boxes"] >= 22],
        "few": [x for x in keep if x["n_boxes"] <= 8],
        "border": [x for x in keep if "border" in x["flags"]],
        "extreme_aspect": [x for x in keep if "extreme_aspect" in x["flags"]],
        "tiny": [x for x in keep if "tiny" in x["flags"]],
        "low_fill": [x for x in keep if "low_fill" in x["flags"]],
        "overlap": [x for x in keep if "overlap" in x["flags"]],
        "near_gray": [x for x in review if "near_gray" in x["flags"]],
        "gallery": [x for x in review if "gallery" in x["flags"]],
        "random": list(keep),
    }

    selected: list[dict] = []
    seen = set()

    def take(pool: list[dict], n: int, tag: str) -> None:
        pool = [p for p in pool if p["abs_path"] not in seen]
        rng.shuffle(pool)
        for p in pool[:n]:
            p = dict(p)
            p["select_reason"] = tag
            selected.append(p)
            seen.add(p["abs_path"])

    take(by_flag["many"], 12, "many_teeth")
    take(by_flag["few"], 10, "few_teeth")
    take(by_flag["border"], 15, "border")
    take(by_flag["extreme_aspect"], 10, "extreme_aspect")
    take(by_flag["tiny"], 8, "tiny")
    take(by_flag["low_fill"], 8, "low_fill")
    take(by_flag["overlap"], 12, "overlap")
    take(by_flag["near_gray"], 8, "near_gray")
    take(by_flag["gallery"], 7, "gallery")
    take(by_flag["random"], N_TARGET, "random")

    # trim to ~100 preferring diversity already added
    if len(selected) > N_TARGET:
        selected = selected[:N_TARGET]
    while len(selected) < N_TARGET and by_flag["random"]:
        take(by_flag["random"], 1, "random")

    for i, row in enumerate(selected, start=1):
        row["index"] = i
        row["total"] = len(selected)
    return selected


def get_catalog() -> list[dict]:
    if CATALOG.exists():
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = build_selection()
    CATALOG.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def draw_boxes(img: Image.Image, boxes) -> Image.Image:
    im = img.convert("RGB")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2, _bw, _bh in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 80, 255), width=4)
        draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(0, 0, 0, 180), width=1)
    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Batch 02 visual annotation audit", layout="wide")
    st.title("Batch 02 CLEAN — visual annotation audit")
    st.caption(
        "Existing YOLO rectangles only. No training. No label edits. "
        f"Dataset: `{CLEAN}`"
    )

    catalog = get_catalog()
    filters = [
        "All",
        "Random",
        "Many teeth",
        "Few teeth",
        "Border",
        "Extreme aspect",
        "Tiny",
        "Low fill",
        "Overlap",
        "Near-gray",
        "Gallery/3D",
    ]
    filt = st.sidebar.selectbox("Filter", filters)
    reason_map = {
        "All": None,
        "Random": "random",
        "Many teeth": "many_teeth",
        "Few teeth": "few_teeth",
        "Border": "border",
        "Extreme aspect": "extreme_aspect",
        "Tiny": "tiny",
        "Low fill": "low_fill",
        "Overlap": "overlap",
        "Near-gray": "near_gray",
        "Gallery/3D": "gallery",
    }
    want = reason_map[filt]
    view = catalog if want is None else [r for r in catalog if r["select_reason"] == want]
    if not view:
        view = catalog
        st.sidebar.warning("No images in this filter; showing All.")

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if st.session_state.idx >= len(view):
        st.session_state.idx = 0

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("Previous"):
            st.session_state.idx = (st.session_state.idx - 1) % len(view)
    with c3:
        if st.button("Next"):
            st.session_state.idx = (st.session_state.idx + 1) % len(view)

    row = view[st.session_state.idx]
    img_path = Path(row["abs_path"])
    lab_path = Path(row["label_path"])
    image = Image.open(img_path)
    w, h = image.size
    boxes = parse_yolo(lab_path, w, h)
    vis = draw_boxes(image, boxes)

    st.subheader(f"Image {st.session_state.idx + 1} / {len(view)}  (catalog {row['index']} / {row['total']})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Split", row["split"])
    m2.metric("Tooth boxes", len(boxes))
    m3.metric("Class", "tooth")
    m4.metric("Reason", row["select_reason"])
    st.write(f"**Filename:** `{row['name']}`")
    st.write(f"**Label file:** `{lab_path.name}`")
    st.write(f"**Flags:** {', '.join(row['flags']) if row['flags'] else '(none)'}")
    st.image(np.array(vis), caption="Existing YOLO rectangles (green)", use_container_width=True)
    st.sidebar.write(f"Catalog size: {len(catalog)}")
    st.sidebar.write(f"Filter size: {len(view)}")


if __name__ == "__main__":
    main()
