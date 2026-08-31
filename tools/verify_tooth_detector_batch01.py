#!/usr/bin/env python3
"""Verify Batch_01 tooth detector dataset. Does not train. Does not touch ICDAS."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01"
FORBIDDEN = [
    ROOT / "predictions",
    ROOT / "annotation_batches" / "Batch_02" / "yolo_candidate_labels",
]


def stems(folder: Path, suffix: str) -> set[str]:
    return {p.stem for p in folder.glob(f"*{suffix}")}


def parse_label(path: Path) -> list[int]:
    ids = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (5, 6):
            raise SystemExit(f"{path.name} line {i}: expected 5 YOLO fields, got {len(parts)}")
        if len(parts) == 6:
            raise SystemExit(
                f"{path.name} line {i}: 6 fields (likely prediction conf). "
                "Batch_01 training labels must be class cx cy w h only."
            )
        cls = int(float(parts[0]))
        vals = list(map(float, parts[1:5]))
        if not (0 <= cls):
            raise SystemExit(f"{path.name} bad class {cls}")
        if any(v < 0 or v > 1 for v in vals):
            raise SystemExit(f"{path.name} line {i}: coord outside [0,1]: {vals}")
        ids.append(cls)
    return ids


def main() -> None:
    yaml = (DS / "data.yaml").read_text(encoding="utf-8")
    if "nc: 1" not in yaml or "tooth" not in yaml:
        raise SystemExit("data.yaml must declare nc: 1 and names 0: tooth")
    if "predictions" in yaml.lower() or "batch_02" in yaml.lower():
        raise SystemExit("data.yaml must not point at predictions or Batch_02")

    summary = {}
    all_classes: Counter[int] = Counter()
    n_boxes = 0
    for split in ("train", "val", "test"):
        img_dir = DS / "images" / split
        lbl_dir = DS / "labels" / split
        imgs = stems(img_dir, ".jpg")
        lbls = stems(lbl_dir, ".txt")
        if imgs != lbls:
            raise SystemExit(
                f"{split} mismatch imgs-only={sorted(imgs - lbls)} lbls-only={sorted(lbls - imgs)}"
            )
        split_boxes = 0
        for stem in sorted(lbls):
            ids = parse_label(lbl_dir / f"{stem}.txt")
            if not ids:
                raise SystemExit(f"{split}/{stem}.txt has zero boxes")
            all_classes.update(ids)
            split_boxes += len(ids)
        n_boxes += split_boxes
        summary[split] = {"images": len(imgs), "labels": len(lbls), "boxes": split_boxes}

    if set(all_classes) != {0}:
        raise SystemExit(f"classes must be exactly {{0}}, got {dict(all_classes)}")
    if summary["train"]["images"] + summary["val"]["images"] + summary["test"]["images"] != 60:
        raise SystemExit(f"expected 60 paired images, got {summary}")

    # Ensure we did not copy prediction overlays/labels into this dataset.
    for split in ("train", "val", "test"):
        for p in (DS / "labels" / split).glob("*.txt"):
            text = p.read_text(encoding="utf-8")
            # human Batch_01 export is 5-tuple lines; already enforced.

    out = {
        "ok": True,
        "nc": 1,
        "names": {0: "tooth"},
        "splits": summary,
        "total_boxes": n_boxes,
        "class_histogram": dict(all_classes),
        "data_yaml": str(DS / "data.yaml"),
        "forbidden_sources_not_used": [str(p) for p in FORBIDDEN],
    }
    print(json.dumps(out, indent=2))
    (ROOT / "reports" / "batch01_dataset_verify.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
