#!/usr/bin/env python3
"""Inspect a local image/annotation tree. Does not assign ICDAS grades."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTS = {".txt", ".xml", ".json", ".csv"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def file_md5(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    data = path.read_bytes()
    array = cv2.imdecode(
        __import__("numpy").frombuffer(data, dtype="uint8"),
        cv2.IMREAD_COLOR,
    )
    if array is None:
        return {"ok": False, "width": None, "height": None, "channels": None}
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else int(array.shape[2])
    return {
        "ok": True,
        "width": int(width),
        "height": int(height),
        "channels": channels,
    }


def class_from_parent(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    parts = rel.parts
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset inspection report (no ICDAS remapping)."
    )
    parser.add_argument("--root", required=True, help="Folder to inspect.")
    parser.add_argument(
        "--annotations-csv",
        default=None,
        help="Optional CSV with a filename column.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write JSON report to this path.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    images = [p for p in root.rglob("*") if is_image(p)]
    label_files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in LABEL_EXTS
    ]

    class_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    corrupted: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)

    for path in images:
        rel = str(path.relative_to(root)).replace("\\", "/")
        class_counts[class_from_parent(path, root)] += 1
        info = inspect_image(path)
        if not info["ok"]:
            corrupted.append(rel)
            continue
        dimensions[f"{info['width']}x{info['height']}"] += 1
        hashes[file_md5(path)].append(rel)

    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}

    csv_rows = 0
    csv_missing = 0
    csv_present = 0
    csv_class: Counter[str] = Counter()
    csv_split: Counter[str] = Counter()
    if args.annotations_csv:
        csv_path = Path(args.annotations_csv)
        if csv_path.exists():
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            csv_rows = len(rows)
            for row in rows:
                filename = (row.get("filename") or "").replace("\\", "/")
                csv_class[str(row.get("icdas_score") or row.get("class") or "")] += 1
                csv_split[str(row.get("split") or "")] += 1
                candidate = root / filename
                if not candidate.exists():
                    candidate = csv_path.parent / filename
                if candidate.exists():
                    csv_present += 1
                else:
                    csv_missing += 1

    counts = list(class_counts.values())
    imbalance = None
    if counts:
        imbalance = {
            "min": min(counts),
            "max": max(counts),
            "ratio_max_over_min": round(max(counts) / max(min(counts), 1), 3),
        }

    report = {
        "root": str(root),
        "total_images": len(images),
        "annotation_sidecar_files": len(label_files),
        "class_or_folder_counts": dict(sorted(class_counts.items())),
        "corrupted_images": corrupted,
        "corrupted_count": len(corrupted),
        "dimension_histogram": dict(dimensions.most_common(20)),
        "duplicate_groups": len(duplicates),
        "duplicate_examples": list(duplicates.values())[:10],
        "class_imbalance": imbalance,
        "annotations_csv": {
            "path": args.annotations_csv,
            "rows": csv_rows,
            "files_present": csv_present,
            "files_missing": csv_missing,
            "icdas_or_class_counts": dict(csv_class),
            "split_counts": dict(csv_split),
        },
        "note": (
            "Folder names are not assumed to be ICDAS grades. "
            "CSV icdas_score values are reported as stored."
        ),
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
