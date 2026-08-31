#!/usr/bin/env python3
"""Build Batch_01 YOLO dataset from seed_60 + CVAT ZIP. Does not touch selected/420."""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "annotation_batches" / "Batch_01" / "seed_60"
ZIP = Path(
    r"C:\Users\anant\Downloads\task_2546747_annotations_2026_08_26_20_27_14_ultralytics yolo detection 1.0.zip"
)
LIST = ROOT / "annotation_batches" / "Batch_01" / "image_list.csv"
OUT = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01"
SEED_SPLIT = 42
TRAIN_R, VAL_R = 0.70, 0.15


def group_key(filename: str, patient: str) -> str:
    p = (patient or "").strip()
    if p:
        return f"patient:{p}"
    return f"image:{filename}"


def bucket(key: str) -> str:
    u = int(hashlib.md5(f"{SEED_SPLIT}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < TRAIN_R:
        return "train"
    if u < TRAIN_R + VAL_R:
        return "val"
    return "test"


def main():
    if not SEED.exists():
        raise SystemExit(f"missing seed_60: {SEED}")
    if not ZIP.exists():
        raise SystemExit(f"missing zip: {ZIP}")
    jpgs = {p.name: p for p in SEED.glob("*.jpg")}
    if len(jpgs) != 60:
        raise SystemExit(f"expected 60 jpgs in seed_60, got {len(jpgs)}")

    patients = {}
    with LIST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patients[row["filename"]] = row.get("patient_identifier_if_available") or ""

    with ZipFile(ZIP) as zf:
        labels = {
            Path(n).stem: zf.read(n)
            for n in zf.namelist()
            if n.startswith("labels/") and n.endswith(".txt")
        }
    if len(labels) != 60:
        raise SystemExit(f"expected 60 labels in zip, got {len(labels)}")

    missing = [n for n in jpgs if Path(n).stem not in labels]
    extra = [s for s in labels if f"{s}.jpg" not in jpgs]
    if missing or extra:
        raise SystemExit(f"pairing failed missing={missing} extra={extra}")

    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    for name, src in sorted(jpgs.items()):
        split = bucket(group_key(name, patients.get(name, "")))
        shutil.copy2(src, OUT / "images" / split / name)
        (OUT / "labels" / split / f"{Path(name).stem}.txt").write_bytes(labels[Path(name).stem])
        counts[split] += 1

    yaml = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: tooth\n"
    )
    (OUT / "data.yaml").write_text(yaml, encoding="utf-8")
    print(dict(counts), "total", sum(counts.values()))
    print("wrote", OUT / "data.yaml")


if __name__ == "__main__":
    main()
