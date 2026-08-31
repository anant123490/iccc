"""Convert Pascal VOC XML boxes to YOLO txt. Does not invent boxes."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_TO_ID = {"tooth": 0}


def voc_to_yolo_box(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def convert_one(xml_path: Path, out_path: Path, class_map: dict[str, int]) -> int:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        out_path.write_text("", encoding="utf-8")
        return 0
    w = int(float(size.findtext("width") or 0))
    h = int(float(size.findtext("height") or 0))
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_map:
            continue
        bb = obj.find("bndbox")
        if bb is None or w <= 0 or h <= 0:
            continue
        xmin = float(bb.findtext("xmin") or 0)
        ymin = float(bb.findtext("ymin") or 0)
        xmax = float(bb.findtext("xmax") or 0)
        ymax = float(bb.findtext("ymax") or 0)
        cx, cy, bw, bh = voc_to_yolo_box(xmin, ymin, xmax, ymax, w, h)
        lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--class-name", default="tooth")
    args = ap.parse_args()
    class_map = {args.class_name: 0, **CLASS_TO_ID}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_files = n_boxes = 0
    for xml_path in sorted(args.voc_dir.glob("*.xml")):
        n_boxes += convert_one(xml_path, args.out_dir / (xml_path.stem + ".txt"), class_map)
        n_files += 1
    print(f"converted {n_files} VOC files -> {n_boxes} YOLO lines (empty XML stay empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
