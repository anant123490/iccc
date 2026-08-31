"""Convert COCO json boxes to YOLO txt. Does not invent boxes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def coco_bbox_to_yolo(x: float, y: float, bw: float, bh: float, width: int, height: int) -> tuple[float, float, float, float]:
    cx = (x + bw / 2.0) / width
    cy = (y + bh / 2.0) / height
    return cx, cy, bw / width, bh / height


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id written for every box")
    args = ap.parse_args()

    data = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    by_image = defaultdict(list)
    for ann in data.get("annotations", []) or []:
        by_image[ann["image_id"]].append(ann)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ann = 0
    for img_id, im in images.items():
        stem = Path(im["file_name"]).stem
        w, h = int(im["width"]), int(im["height"])
        lines = []
        for ann in by_image.get(img_id, []):
            bbox = ann.get("bbox") or []
            if len(bbox) != 4 or w <= 0 or h <= 0:
                continue
            cx, cy, bw, bh = coco_bbox_to_yolo(*bbox, w, h)
            lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_ann += 1
        (args.out_dir / f"{stem}.txt").write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    print(f"wrote {len(images)} YOLO files, {n_ann} boxes (images with no anns stay empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
