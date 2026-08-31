"""Overlay COCO / YOLO / Pascal VOC boxes on RGB images.

No model inference. Empty annotation files produce the image with no boxes.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    return cls, cx, cy, w, h


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def load_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parsed = parse_yolo_line(line.strip())
        if parsed:
            boxes.append(parsed)
    return boxes


def load_voc(path: Path) -> list[tuple[str, int, int, int, int]]:
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or "tooth"
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = int(float(bb.findtext("xmin") or 0))
        y1 = int(float(bb.findtext("ymin") or 0))
        x2 = int(float(bb.findtext("xmax") or 0))
        y2 = int(float(bb.findtext("ymax") or 0))
        out.append((name, x1, y1, x2, y2))
    return out


def load_coco_index(coco_path: Path) -> dict[str, list[list[float]]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    id_to_name = {im["id"]: im["file_name"] for im in data.get("images", [])}
    by_file: dict[str, list[list[float]]] = {name: [] for name in id_to_name.values()}
    for ann in data.get("annotations", []) or []:
        name = id_to_name.get(ann["image_id"])
        if not name:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) == 4:
            by_file.setdefault(name, []).append(bbox)
    return by_file


def draw_xyxy(im: Image.Image, boxes_xyxy: list[tuple[int, int, int, int]], labels: list[str]) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
        draw.rectangle([x1, y1, x2, y2], outline=(230, 57, 70), width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), lab, fill=(230, 57, 70))
    return out


def resolve_image(images_dir: Path, filename: str) -> Path:
    p = images_dir / filename
    if p.exists():
        return p
    matches = list(images_dir.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(filename)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw existing boxes only. Does not invent annotations.")
    ap.add_argument("--images", type=Path, required=True, help="Directory of RGB images")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for overlays")
    ap.add_argument("--format", choices=("yolo", "coco", "voc"), required=True)
    ap.add_argument("--labels", type=Path, help="YOLO dir, COCO json, or VOC dir")
    ap.add_argument("--limit", type=int, default=0, help="Optional max images")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.jpeg")) + sorted(args.images.glob("*.png"))
    if args.limit:
        images = images[: args.limit]

    coco_index = None
    if args.format == "coco":
        if not args.labels:
            print("COCO requires --labels path to json", file=sys.stderr)
            return 2
        coco_index = load_coco_index(args.labels)

    n = 0
    empty = 0
    for img_path in images:
        with Image.open(img_path) as im:
            w, h = im.size
            xyxy: list[tuple[int, int, int, int]] = []
            labs: list[str] = []
            if args.format == "yolo":
                lab_dir = args.labels or (args.images.parent.parent / "annotations" / "yolo")
                recs = load_yolo(Path(lab_dir) / (img_path.stem + ".txt"))
                for rec in recs:
                    cls, cx, cy, bw, bh = rec
                    xyxy.append(yolo_to_xyxy(cx, cy, bw, bh, w, h))
                    labs.append(str(cls))
            elif args.format == "voc":
                lab_dir = args.labels or (args.images.parent.parent / "annotations" / "pascal_voc")
                recs = load_voc(Path(lab_dir) / (img_path.stem + ".xml"))
                for name, x1, y1, x2, y2 in recs:
                    xyxy.append((x1, y1, x2, y2))
                    labs.append(name)
            else:
                bboxes = (coco_index or {}).get(img_path.name, [])
                for bbox in bboxes:
                    x, y, bw, bh = bbox
                    xyxy.append((int(x), int(y), int(x + bw), int(y + bh)))
                    labs.append("tooth")
            if not xyxy:
                empty += 1
            vis = draw_xyxy(im, xyxy, labs)
            vis.save(args.out / img_path.name, quality=92)
            n += 1
    print(f"wrote {n} overlays ({empty} had empty/missing annotations) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
