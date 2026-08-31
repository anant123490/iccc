"""Convert YOLO txt labels to a COCO json. Does not invent boxes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def yolo_line_to_coco_bbox(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> list[float]:
    x = (cx - bw / 2.0) * width
    y = (cy - bh / 2.0) * height
    return [x, y, bw * width, bh * height]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--yolo-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--class-name", default="tooth")
    args = ap.parse_args()

    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    files = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.jpeg")) + sorted(args.images.glob("*.png"))
    for img_path in files:
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        lab = args.yolo_dir / (img_path.stem + ".txt")
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                bbox = yolo_line_to_coco_bbox(cx, cy, bw, bh, w, h)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1 if cls == 0 else cls,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        img_id += 1

    coco = {
        "info": {"description": "Converted from YOLO. Empty txt files yield no annotations."},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": args.class_name, "supercategory": "tooth"}],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"wrote {len(images)} images, {len(annotations)} annotations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
