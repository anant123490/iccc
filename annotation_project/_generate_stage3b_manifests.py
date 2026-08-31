"""Stage 3B: generate annotation project manifests, batches, and splits.

Does not create tooth boxes or FDI labels. Does not modify ICDAS or originals.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"c:\Users\anant\OneDrive\Desktop\icdas project")
SEL = ROOT / "fdi_detection_dataset" / "images" / "selected"
SRC_CSV = ROOT / "reports" / "stage3a_selected_images.csv"
OUT_MANIFEST = ROOT / "annotation_project" / "manifests"
OUT_CVAT = ROOT / "annotation_project" / "cvat"
OUT_LS = ROOT / "annotation_project" / "label_studio"
OUT_BATCH = ROOT / "annotation_batches"
OUT_SPLITS = ROOT / "fdi_detection_dataset" / "splits"
YOLO_DIR = ROOT / "fdi_detection_dataset" / "annotations" / "yolo"
SEED_ORDER = True


def load_selected() -> list[dict]:
    with SRC_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_name = {r["current_filename"]: r for r in rows}
    files = sorted(p.name for p in SEL.glob("*.jpg"))
    if len(files) != 420:
        raise SystemExit(f"selected count mismatch: {len(files)}")
    out = []
    for name in files:
        src = by_name.get(name)
        if not src:
            raise SystemExit(f"missing Stage 3A row for {name}")
        rel = f"fdi_detection_dataset/images/selected/{name}"
        out.append(
            {
                "filename": name,
                "relative_path": rel.replace("\\", "/"),
                "mouth_view": src.get("view_folder") or src.get("orientation") or "",
                "orientation": src.get("orientation") or "",
                "protocol": src.get("protocol") or "",
                "width": int(src["width"]),
                "height": int(src["height"]),
                "annotation_status": "not_annotated",
                "patient_identifier_if_available": src.get("patient_identifier_if_available") or "",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def stratified_batches(rows: list[dict]) -> list[list[dict]]:
    views = ["Frontal", "Left_Lateral", "Mandibular", "Maxillary_Occlusal", "Right_Lateral"]
    buckets = {v: [] for v in views}
    other = []
    for r in rows:
        if r["mouth_view"] in buckets:
            buckets[r["mouth_view"]].append(r)
        else:
            other.append(r)
    for v in views:
        buckets[v].sort(key=lambda r: r["filename"])
    other.sort(key=lambda r: r["filename"])
    batches = [[] for _ in range(7)]
    # 12 per view per batch = 60
    for v in views:
        items = buckets[v]
        if len(items) != 84:
            # still distribute evenly
            pass
        for i, item in enumerate(items):
            batches[i % 7].append(item)
    for i, item in enumerate(other):
        batches[i % 7].append(item)
    for b in batches:
        b.sort(key=lambda r: (r["mouth_view"], r["filename"]))
    return batches


def patient_safe_splits(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        pid = r["patient_identifier_if_available"].strip()
        key = f"patient:{pid}" if pid else f"image:{r['filename']}"
        groups[key].append(r)
    # deterministic order: larger groups first, then key
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    targets = {"train": 294, "val": 63, "test": 63}
    counts = {"train": 0, "val": 0, "test": 0}
    assigned = {"train": [], "val": [], "test": []}
    for key, items in ordered:
        n = len(items)
        # pick split with most remaining capacity; tie-break train, val, test
        remain = {s: targets[s] - counts[s] for s in ("train", "val", "test")}
        # if all would overflow, pick the one with largest remaining (can go slightly over)
        split = max(("train", "val", "test"), key=lambda s: (remain[s], {"train": 3, "val": 2, "test": 1}[s]))
        assigned[split].extend(items)
        counts[split] += n
    for s in assigned:
        assigned[s].sort(key=lambda r: r["filename"])
    return assigned


def main() -> None:
    rows = load_selected()
    fields = [
        "filename",
        "relative_path",
        "mouth_view",
        "width",
        "height",
        "annotation_status",
        "patient_identifier_if_available",
    ]

    OUT_MANIFEST.mkdir(parents=True, exist_ok=True)
    OUT_CVAT.mkdir(parents=True, exist_ok=True)
    OUT_LS.mkdir(parents=True, exist_ok=True)
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)

    write_csv(OUT_MANIFEST / "selected_images.csv", rows, fields)
    (OUT_MANIFEST / "selected_images.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (OUT_MANIFEST / "selected_images.txt").write_text(
        "\n".join(r["relative_path"] for r in rows) + "\n", encoding="utf-8"
    )
    # also under dataset metadata
    meta = ROOT / "fdi_detection_dataset" / "metadata"
    write_csv(meta / "selected_images_stage3b.csv", rows, fields)

    # CVAT manifests
    write_csv(OUT_CVAT / "image_manifest.csv", rows, fields)
    (OUT_CVAT / "image_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT_CVAT / "dataset_manifest.csv").write_text(
        "name,path,images,label,note\n"
        "iccc_whole_tooth,fdi_detection_dataset/images/selected,420,tooth,"
        "Empty labels. Import images only. Do not import Zenodo d/D lesions.\n",
        encoding="utf-8",
    )

    # Label Studio tasks (local relative paths; user maps storage)
    tasks = []
    for i, r in enumerate(rows, start=1):
        tasks.append(
            {
                "id": i,
                "data": {
                    "image": f"/data/local-files/?d=fdi_detection_dataset/images/selected/{r['filename']}",
                    "filename": r["filename"],
                    "mouth_view": r["mouth_view"],
                },
                "meta": {
                    "width": r["width"],
                    "height": r["height"],
                    "patient_identifier_if_available": r["patient_identifier_if_available"],
                    "annotation_status": "not_annotated",
                },
            }
        )
    (OUT_LS / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (OUT_LS / "image_list.json").write_text(
        json.dumps([{"image": r["relative_path"], "filename": r["filename"]} for r in rows], indent=2),
        encoding="utf-8",
    )

    batches = stratified_batches(rows)
    if len(batches) != 7 or any(len(b) != 60 for b in batches):
        sizes = [len(b) for b in batches]
        raise SystemExit(f"batch size mismatch: {sizes}")

    for i, batch in enumerate(batches, start=1):
        d = OUT_BATCH / f"Batch_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        write_csv(d / "image_list.csv", batch, fields)
        (d / "image_list.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
        views = {}
        for r in batch:
            views[r["mouth_view"]] = views.get(r["mouth_view"], 0) + 1
        view_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(views.items()))
        (d / "README.md").write_text(
            f"# Annotation Batch {i:02d}\n\n"
            f"Images: **{len(batch)}** (manifests only; files stay in "
            f"`fdi_detection_dataset/images/selected/`).\n\n"
            f"Class: `tooth` (id 0). **No FDI.** **No pre-drawn boxes.**\n\n"
            f"## Mouth views in this batch\n\n{view_lines}\n\n"
            f"## How to use\n\n"
            f"1. Open CVAT or Label Studio using `annotation_project/` configs.\n"
            f"2. Import or filter this batch via `image_list.csv`.\n"
            f"3. Draw whole-tooth rectangles only.\n"
            f"4. Do not copy images into this folder.\n",
            encoding="utf-8",
        )

    splits = patient_safe_splits(rows)
    for name, items in splits.items():
        write_csv(OUT_SPLITS / f"{name}.csv", items, fields + ["protocol", "orientation"])
        (OUT_SPLITS / f"{name}.txt").write_text(
            "\n".join(f"images/selected/{r['filename']}" for r in items) + "\n",
            encoding="utf-8",
        )
        (OUT_SPLITS / f"{name}_repo_relative.txt").write_text(
            "\n".join(r["relative_path"] for r in items) + "\n", encoding="utf-8"
        )
        (OUT_SPLITS / f"{name}.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    # empty yolo if missing
    created = 0
    for r in rows:
        p = YOLO_DIR / (Path(r["filename"]).stem + ".txt")
        if not p.exists():
            p.write_text("", encoding="utf-8")
            created += 1
        elif p.read_text(encoding="utf-8").strip():
            raise SystemExit(f"non-empty YOLO placeholder: {p}")

    summary = {
        "selected": len(rows),
        "batches": {f"Batch_{i:02d}": len(b) for i, b in enumerate(batches, 1)},
        "splits": {k: len(v) for k, v in splits.items()},
        "split_unique_patient_keys": {
            k: len({r["patient_identifier_if_available"] or r["filename"] for r in v})
            for k, v in splits.items()
        },
        "yolo_placeholders_created_missing": created,
        "yolo_placeholders_total": len(list(YOLO_DIR.glob("*.txt"))),
    }
    (OUT_SPLITS / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
