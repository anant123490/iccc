#!/usr/bin/env python3
"""Select Batch_02 (60) from remaining 360 using YOLO candidate stats + view diversity.

Does not modify fdi_detection_dataset/images/selected/, ICDAS data, or models.
YOLO boxes are review aids only — not ground truth.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "fdi_detection_dataset" / "images" / "selected"
META = ROOT / "fdi_detection_dataset" / "metadata" / "selected_images_stage3b.csv"
PRED_CSV = ROOT / "predictions" / "confidence_reports" / "per_image.csv"
PRED_LBL = ROOT / "predictions" / "labels"
PRED_VIZ = ROOT / "predictions" / "visualizations"
BATCH01 = ROOT / "annotation_batches" / "Batch_01"
BATCH02 = ROOT / "annotation_batches" / "Batch_02"
ARCHIVE = BATCH02 / "stage3b_round_robin_archive"
REPORT = ROOT / "reports" / "BATCH_02_SELECTION.md"

VIEWS = ["Frontal", "Left_Lateral", "Mandibular", "Maxillary_Occlusal", "Right_Lateral"]
PER_VIEW = 12
TARGET = 60


def view_of(name: str, mouth_view: str) -> str:
    if mouth_view in VIEWS:
        return mouth_view
    n = name.lower()
    if "frontal" in n:
        return "Frontal"
    if "left" in n and "lateral" in n:
        return "Left_Lateral"
    if "right" in n and "lateral" in n:
        return "Right_Lateral"
    if "mandibular" in n:
        return "Mandibular"
    if "maxillary" in n:
        return "Maxillary_Occlusal"
    return "Unknown"


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return inter / den if den else 0.0


def yolo_xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def label_stats(stem: str) -> dict:
    path = PRED_LBL / f"{stem}.txt"
    confs: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            boxes.append(yolo_xywh_to_xyxy(cx, cy, w, h))
            confs.append(float(p[5]) if len(p) > 5 else 0.0)
    n = len(boxes)
    overlaps = []
    for i, bi in enumerate(boxes):
        best = 0.0
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            best = max(best, iou(bi, bj))
        overlaps.append(best)
    mean_ov = sum(overlaps) / n if n else 0.0
    hi_ov = sum(1 for x in overlaps if x >= 0.3)
    mean_c = sum(confs) / n if n else 0.0
    if n > 1:
        var = sum((c - mean_c) ** 2 for c in confs) / n
        std_c = math.sqrt(var)
    else:
        std_c = 0.0
    return {
        "n_det": n,
        "mean_conf": mean_c,
        "min_conf": min(confs) if confs else 0.0,
        "std_conf": std_c,
        "frac_conf_lt_0_4": (sum(1 for c in confs if c < 0.4) / n) if n else 0.0,
        "mean_max_iou": mean_ov,
        "n_iou_ge_0_3": hi_ov,
    }


def load_meta() -> dict[str, dict]:
    with META.open(encoding="utf-8", newline="") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def protocol_of(name: str) -> str:
    if name.startswith("anonymous-") and "View-" in name.replace("_", ""):
        return "pilot"
    if name.startswith("anonymous-") and not name.startswith("anonymous_"):
        return "pilot"
    return "clinic"


def pick_view(cands: list[dict], already: set[str], used_patients: set[str], key, reverse: bool, k: int) -> list[dict]:
    ranked = sorted(cands, key=key, reverse=reverse)
    out = []
    for row in ranked:
        if len(out) >= k:
            break
        if row["filename"] in already:
            continue
        pid = row["patient"] or f"img:{row['filename']}"
        # allow same patient only if we would otherwise starve the view
        out.append(row)
        already.add(row["filename"])
        used_patients.add(pid)
    return out


def fill_view(cands: list[dict], already: set[str], used_patients: set[str], need: int) -> list[dict]:
    """Prefer new patients, then high composite hard-example score."""
    scored = sorted(
        cands,
        key=lambda r: (
            (r["patient"] in used_patients) if r["patient"] else False,
            -r["hard_score"],
            r["filename"],
        ),
    )
    out = []
    for row in scored:
        if len(out) >= need:
            break
        if row["filename"] in already:
            continue
        out.append(row)
        already.add(row["filename"])
        used_patients.add(row["patient"] or f"img:{row['filename']}")
    return out


def hard_score(s: dict) -> float:
    n = s["n_det"]
    count_ext = 0.0
    if n > 24:
        count_ext = 2.0 + (n - 24) * 0.1
    elif n < 10:
        count_ext = 1.5 + (10 - n) * 0.15
    else:
        count_ext = abs(n - 15) / 20.0
    return (
        (1.0 - s["mean_conf"]) * 2.0
        + s["frac_conf_lt_0_4"] * 1.5
        + s["std_conf"] * 1.2
        + s["mean_max_iou"] * 2.0
        + count_ext
    )


def main() -> None:
    b01 = {p.strip() for p in (BATCH01 / "cvat_upload_filenames.txt").read_text(encoding="utf-8").splitlines() if p.strip()}
    meta = load_meta()
    with PRED_CSV.open(encoding="utf-8", newline="") as f:
        pred_rows = list(csv.DictReader(f))

    pool: list[dict] = []
    for pr in pred_rows:
        name = pr["filename"]
        if name in b01:
            continue
        m = meta.get(name, {})
        st = label_stats(Path(name).stem)
        row = {
            "filename": name,
            "relative_path": f"fdi_detection_dataset/images/selected/{name}",
            "mouth_view": view_of(name, m.get("mouth_view") or ""),
            "width": int(m["width"]) if m.get("width") else 0,
            "height": int(m["height"]) if m.get("height") else 0,
            "annotation_status": "not_annotated",
            "patient_identifier_if_available": m.get("patient_identifier_if_available") or "",
            "patient": (m.get("patient_identifier_if_available") or "").strip(),
            "protocol": protocol_of(name),
            **st,
        }
        row["hard_score"] = hard_score(st)
        pool.append(row)

    by_view: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_view[r["mouth_view"]].append(r)

    selected: list[dict] = []
    already: set[str] = set()
    used_patients: set[str] = set()

    # Must-include: over-detection (>24)
    for r in pool:
        if r["n_det"] > 24 and r["filename"] not in already:
            selected.append(r)
            already.add(r["filename"])
            used_patients.add(r["patient"] or f"img:{r['filename']}")

    for view in VIEWS:
        cands = by_view[view]
        have = [r for r in selected if r["mouth_view"] == view]
        need = PER_VIEW - len(have)
        picked: list[dict] = []
        picked += pick_view(cands, already, used_patients, lambda r: r["n_det"], True, 2)
        picked += pick_view(cands, already, used_patients, lambda r: r["n_det"], False, 2)
        picked += pick_view(cands, already, used_patients, lambda r: r["mean_conf"], False, 2)
        picked += pick_view(cands, already, used_patients, lambda r: r["mean_max_iou"], True, 2)
        picked += pick_view(cands, already, used_patients, lambda r: r["frac_conf_lt_0_4"], True, 2)
        picked += fill_view(cands, already, used_patients, max(0, need - len(picked)))
        # trim if over (must-include ate a slot)
        extra = len(have) + len(picked) - PER_VIEW
        if extra > 0:
            # drop lowest hard_score from picked (not must-include >24)
            droppable = [p for p in picked if p["n_det"] <= 24]
            droppable.sort(key=lambda r: r["hard_score"])
            drop_set = {p["filename"] for p in droppable[:extra]}
            picked = [p for p in picked if p["filename"] not in drop_set]
            for name in drop_set:
                already.discard(name)
        selected.extend(picked)

    selected.sort(key=lambda r: (r["mouth_view"], r["filename"]))
    if len(selected) != TARGET:
        raise SystemExit(f"expected {TARGET} selected, got {len(selected)}")

    # Archive previous Stage 3B round-robin Batch_02 lists (never annotated).
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for fname in ("image_list.csv", "image_list.json", "README.md"):
        src = BATCH02 / fname
        if src.exists() and not (ARCHIVE / fname).exists():
            shutil.copy2(src, ARCHIVE / fname)

    fields = [
        "filename",
        "relative_path",
        "mouth_view",
        "width",
        "height",
        "annotation_status",
        "patient_identifier_if_available",
        "n_det",
        "mean_conf",
        "min_conf",
        "std_conf",
        "frac_conf_lt_0_4",
        "mean_max_iou",
        "n_iou_ge_0_3",
        "hard_score",
        "protocol",
        "selection_reason",
    ]

    reasons = {}
    for r in selected:
        bits = []
        if r["n_det"] > 24:
            bits.append("over_24_detections")
        if r["n_det"] <= 12:
            bits.append("low_count_possible_misses")
        if r["mean_conf"] < 0.48:
            bits.append("low_mean_confidence")
        if r["frac_conf_lt_0_4"] >= 0.25:
            bits.append("many_low_conf_boxes")
        if r["mean_max_iou"] >= 0.15:
            bits.append("overlapping_boxes")
        if r["protocol"] == "pilot":
            bits.append("pilot_naming_style")
        bits.append("view:" + r["mouth_view"])
        reasons[r["filename"]] = ";".join(bits)
        r["selection_reason"] = reasons[r["filename"]]

    csv_fields_out = [
        "filename",
        "relative_path",
        "mouth_view",
        "width",
        "height",
        "annotation_status",
        "patient_identifier_if_available",
    ]

    BATCH02.mkdir(parents=True, exist_ok=True)
    with (BATCH02 / "image_list.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields_out, extrasaction="ignore")
        w.writeheader()
        for r in selected:
            w.writerow(r)

    json_rows = [
        {
            "filename": r["filename"],
            "relative_path": r["relative_path"],
            "mouth_view": r["mouth_view"],
            "orientation": r["mouth_view"],
            "protocol": r["protocol"],
            "width": r["width"],
            "height": r["height"],
            "annotation_status": "not_annotated",
            "patient_identifier_if_available": r["patient_identifier_if_available"],
        }
        for r in selected
    ]
    (BATCH02 / "image_list.json").write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    (BATCH02 / "cvat_upload_filenames.txt").write_text(
        "\n".join(r["filename"] for r in selected) + "\n", encoding="utf-8"
    )

    with (BATCH02 / "selection_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in selected:
            w.writerow(r)

    seed = BATCH02 / "seed_60"
    overlays = BATCH02 / "yolo_overlays_for_review"
    cand_lbl = BATCH02 / "yolo_candidate_labels"
    for d in (seed, overlays, cand_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    missing = []
    for r in selected:
        src = SELECTED / r["filename"]
        if not src.exists():
            missing.append(r["filename"])
            continue
        shutil.copy2(src, seed / r["filename"])
        viz = PRED_VIZ / r["filename"]
        if viz.exists():
            shutil.copy2(viz, overlays / r["filename"])
        lbl = PRED_LBL / f"{Path(r['filename']).stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, cand_lbl / lbl.name)

    if missing:
        raise SystemExit("missing originals: " + ", ".join(missing))

    view_counts = {v: sum(1 for r in selected if r["mouth_view"] == v) for v in VIEWS}
    n_pat = len({r["patient"] for r in selected if r["patient"]})
    n_pilot = sum(1 for r in selected if r["protocol"] == "pilot")
    n_gt24 = sum(1 for r in selected if r["n_det"] > 24)
    n_le12 = sum(1 for r in selected if r["n_det"] <= 12)

    readme = f"""# Annotation Batch 02

Images: **{len(selected)}** (YOLO-informed selection from the 360 remaining after Batch_01).

Class: `tooth` (id 0). **No FDI.** **No ICDAS.** YOLO boxes are **not** ground truth.

The previous Stage 3B round-robin 60-file list is archived in `stage3b_round_robin_archive/` (it was never annotated). This batch replaces it so human QC targets detector failure modes.

## Mouth views in this batch

"""
    for v in VIEWS:
        readme += f"- {v}: {view_counts[v]}\n"
    readme += f"""
## Why these images

- {n_gt24} image(s) with **>24** YOLO detections (likely duplicates / false positives).
- {n_le12} image(s) with **≤12** detections (possible missed teeth).
- Mix of low mean confidence, high low-conf fraction, and overlapping boxes.
- {n_pilot} pilot-style filenames and {n_pat} distinct clinic patient IDs (plus unnamed pilots).
- Per-image scores: `selection_scores.csv`.

## How to annotate (CVAT)

1. Create task `iccc_batch_02` in project `iccc_whole_tooth_detection`.
2. Upload **only** the 60 originals in `seed_60/` (same pixels as `fdi_detection_dataset/images/selected/`; that folder was not modified).
3. Filename list: `cvat_upload_filenames.txt`.
4. Leave annotations **empty**. Draw whole-tooth rectangles from scratch (same as Batch_01).
5. Optionally keep `yolo_overlays_for_review/` open in a second window as a **checklist**, not as labels to copy.
6. Do **not** import `yolo_candidate_labels/` as ground truth.

Guidelines: `TOOTH_ANNOTATION_GUIDELINES.md`. Full Batch_01-style steps: `STAGE3C_MANUAL_ANNOTATION.md`.
"""
    (BATCH02 / "README.md").write_text(readme, encoding="utf-8")

    lines = [
        "# Batch 02 selection (human QC)",
        "",
        "Selected **60** images from the **360** remaining after Batch_01.",
        "Selection used YOLO candidate metadata (counts, confidence, overlap) plus view and patient diversity.",
        "Predictions are **not** accepted as labels.",
        "",
        "## Pool",
        "",
        f"- Remaining images scored: {len(pool)}",
        f"- Batch_01 excluded: {len(b01)}",
        f"- Target: {TARGET} ({PER_VIEW} per view)",
        "",
        "## Batch_02 composition",
        "",
        "| View | Count |",
        "| --- | ---: |",
    ]
    for v in VIEWS:
        lines.append(f"| {v} | {view_counts[v]} |")
    lines += [
        "",
        f"- Distinct clinic patient IDs: {n_pat}",
        f"- Pilot-style images: {n_pilot}",
        f"- >24 detections: {n_gt24}",
        f"- ≤12 detections: {n_le12}",
        f"- Mean YOLO detections in batch: {sum(r['n_det'] for r in selected)/len(selected):.2f}",
        f"- Mean YOLO confidence in batch: {sum(r['mean_conf'] for r in selected)/len(selected):.4f}",
        "",
        "## Must-include (>24 detections)",
        "",
        "```",
    ]
    for r in selected:
        if r["n_det"] > 24:
            lines.append(f"{r['filename']}  n={r['n_det']}  mean_conf={r['mean_conf']:.3f}")
    lines += [
        "```",
        "",
        "## Packaging (originals not modified)",
        "",
        "- Upload copies: `annotation_batches/Batch_02/seed_60/`",
        "- CVAT names: `annotation_batches/Batch_02/cvat_upload_filenames.txt`",
        "- Overlay review only: `annotation_batches/Batch_02/yolo_overlays_for_review/`",
        "- Candidate txt (do not treat as GT): `annotation_batches/Batch_02/yolo_candidate_labels/`",
        "- Prior round-robin list: `annotation_batches/Batch_02/stage3b_round_robin_archive/`",
        "",
        "## Annotate from scratch",
        "",
        "Same protocol as Batch_01: empty CVAT task, class `tooth` only, no FDI, no ICDAS.",
        "Use overlays as a second-screen QC aid.",
        "",
        "## Selected files",
        "",
        "| filename | view | n_det | mean_conf | overlap | reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in selected:
        lines.append(
            f"| `{r['filename']}` | {r['mouth_view']} | {r['n_det']} | {r['mean_conf']:.3f} | {r['mean_max_iou']:.3f} | {r['selection_reason']} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "views": view_counts, "gt24": n_gt24, "le12": n_le12}, indent=2))


if __name__ == "__main__":
    main()
