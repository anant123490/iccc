#!/usr/bin/env python3
"""Extract tooth/region crops from full-mouth images using existing boxes.

Annotation class names are NOT ICDAS grades. They only describe the region
that was boxed (tooth, caries, etc.). ICDAS 0–4 is assigned later by a
human annotator in tools/label_icdas.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import (  # noqa: E402
    CROP_CSV_COLUMNS,
    QUALITY_CSV_COLUMNS,
    PROJECT_ROOT,
    box_is_valid,
    clamp_box,
    ensure_dir,
    ensure_pipeline_dirs,
    extreme_aspect_ratio,
    is_blank_image,
    is_image_file,
    iter_images,
    load_csv,
    read_image,
    save_csv,
    unique_crop_id,
    write_image,
)


SUPPORTED_FORMATS = ("auto", "yolo", "coco", "voc", "labelme", "csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop annotated regions from full-mouth dental images. "
            "Does not assign ICDAS grades."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw public dataset (images + annotations).",
    )
    parser.add_argument(
        "--output",
        default="cropped_teeth",
        help="Output directory for crops.csv and images/ (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--format",
        default="auto",
        choices=SUPPORTED_FORMATS,
        help="Annotation format. Use auto to inspect the dataset.",
    )
    parser.add_argument(
        "--coco-json",
        default=None,
        help="Optional explicit COCO JSON path.",
    )
    parser.add_argument(
        "--boxes-csv",
        default=None,
        help="Optional CSV with filename,x1,y1,x2,y2[,annotation_class].",
    )
    parser.add_argument("--padding", type=float, default=0.08, help="Box padding as a fraction of box size.")
    parser.add_argument("--min-size", type=int, default=20, help="Reject crops smaller than this (pixels).")
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0)
    parser.add_argument("--resize", type=int, default=224, help="Resize crops to this square size. 0 keeps original.")
    parser.add_argument(
        "--keep-original-size",
        action="store_true",
        help="Do not resize crops (overrides --resize).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop files.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images to process (0 = all).")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def looks_like_coco(obj) -> bool:
    return isinstance(obj, dict) and "images" in obj and "annotations" in obj


def looks_like_labelme(obj) -> bool:
    return isinstance(obj, dict) and "shapes" in obj and ("imagePath" in obj or "imageData" in obj)


def detect_format(root: Path, coco_json: str | None, boxes_csv: str | None) -> str:
    if boxes_csv:
        return "csv"
    if coco_json:
        return "coco"

    json_files = list(root.rglob("*.json"))
    xml_files = list(root.rglob("*.xml"))
    txt_files = [p for p in root.rglob("*.txt") if p.name.lower() not in {"classes.txt", "obj.names"}]

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if looks_like_coco(obj):
            return "coco"
        if looks_like_labelme(obj):
            return "labelme"

    if xml_files:
        return "voc"
    if txt_files:
        return "yolo"
    return "none"


def yolo_class_names(root: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for candidate in [root / "classes.txt", root / "obj.names"]:
        if candidate.exists():
            for i, line in enumerate(candidate.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if line:
                    names[i] = line
            return names
    yaml_files = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    for path in yaml_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "names:" not in text:
            continue
        # Minimal YAML names parser (avoids adding PyYAML to tools extras).
        in_names = False
        idx = 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                rest = stripped[len("names:") :].strip()
                in_names = True
                if rest.startswith("["):
                    inner = rest.strip("[]")
                    for i, item in enumerate(inner.split(",")):
                        names[i] = item.strip().strip("'\"")
                    return names
                continue
            if in_names:
                if stripped.startswith("-"):
                    names[idx] = stripped[1:].strip().strip("'\"")
                    idx += 1
                elif stripped and not stripped.startswith("#") and ":" in stripped and not line.startswith(" "):
                    break
        if names:
            return names
    return names


def find_yolo_label(image_path: Path, root: Path) -> Path | None:
    stem = image_path.stem
    same_dir = image_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir
    rel = image_path.parent.relative_to(root)
    candidates = [
        root / "labels" / f"{stem}.txt",
        root / "labels" / rel / f"{stem}.txt",
        root / "Annotations" / f"{stem}.txt",
        root / "annotations" / f"{stem}.txt",
    ]
    parts = list(rel.parts)
    if parts and parts[0] in {"images", "Images", "img", "JPEGImages"}:
        rest = Path(*parts[1:]) if len(parts) > 1 else Path()
        candidates.append(root / "labels" / rest / f"{stem}.txt")
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob(f"{stem}.txt"))
    if len(matches) == 1:
        return matches[0]
    return None


def parse_yolo_line(line: str, img_w: int, img_h: int, names: dict[int, str]):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        nums = [float(x) for x in parts[1:5]]
    except ValueError:
        return None
    a, b, c, d = nums
    # YOLO normalized cx,cy,w,h vs already-absolute xyxy
    if max(a, b, c, d) <= 1.5:
        cx, cy, bw, bh = a * img_w, b * img_h, c * img_w, d * img_h
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
    else:
        x1, y1, x2, y2 = a, b, c, d
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
    label = names.get(cls_id, str(cls_id))
    return x1, y1, x2, y2, label


def collect_yolo(root: Path, images: list[Path], names: dict[int, str]):
    records = []
    for image_path in images:
        label_path = find_yolo_label(image_path, root)
        if label_path is None:
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        boxes = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            parsed = parse_yolo_line(line, 1, 1, names)
            # Width/height applied later when the image is loaded.
            if parsed is None:
                continue
            x1, y1, x2, y2, label = parsed
            boxes.append(
                {
                    "annotation_id": f"{label_path.stem}_{i}",
                    "annotation_class": label,
                    "norm_or_abs": (x1, y1, x2, y2),
                    "yolo_raw": line,
                }
            )
        records.append({"image": image_path, "missing_annotation": False, "boxes": boxes, "kind": "yolo"})
    return records


def find_coco_json(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        return path if path.exists() else None
    for path in root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if looks_like_coco(obj):
            return path
    return None


def collect_coco(root: Path, coco_path: Path):
    with coco_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    cat_map = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann.get("image_id"), []).append(ann)

    file_index = {p.name: p for p in iter_images(root)}
    records = []
    for img in data.get("images", []):
        file_name = img.get("file_name", "")
        path = root / file_name
        if not path.exists():
            path = file_index.get(Path(file_name).name)
        if path is None or not Path(path).exists():
            continue
        path = Path(path)
        boxes = []
        for i, ann in enumerate(anns_by_image.get(img["id"], [])):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = bbox[:4]
            cat = cat_map.get(ann.get("category_id"), str(ann.get("category_id", "")))
            boxes.append(
                {
                    "annotation_id": str(ann.get("id", f"{img['id']}_{i}")),
                    "annotation_class": cat,
                    "xyxy": (x, y, x + w, y + h),
                }
            )
        records.append({"image": path, "missing_annotation": False, "boxes": boxes, "kind": "abs"})
        images_by_id  # keep referenced
    return records


def collect_voc(root: Path, images: list[Path]):
    records = []
    xml_index = {p.stem: p for p in root.rglob("*.xml")}
    for image_path in images:
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():
            xml_path = xml_index.get(image_path.stem)
        if xml_path is None or not Path(xml_path).exists():
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        boxes = []
        for i, obj in enumerate(tree.getroot().findall("object")):
            name = (obj.findtext("name") or "region").strip()
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                x1 = float(bnd.findtext("xmin", "0"))
                y1 = float(bnd.findtext("ymin", "0"))
                x2 = float(bnd.findtext("xmax", "0"))
                y2 = float(bnd.findtext("ymax", "0"))
            except ValueError:
                continue
            boxes.append(
                {
                    "annotation_id": f"{Path(xml_path).stem}_{i}",
                    "annotation_class": name,
                    "xyxy": (x1, y1, x2, y2),
                }
            )
        records.append({"image": image_path, "missing_annotation": False, "boxes": boxes, "kind": "abs"})
    return records


def collect_labelme(root: Path, images: list[Path]):
    records = []
    json_index = {p.stem: p for p in root.rglob("*.json")}
    for image_path in images:
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            json_path = json_index.get(image_path.stem)
        if json_path is None:
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        try:
            with Path(json_path).open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except (OSError, json.JSONDecodeError):
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        if not looks_like_labelme(obj):
            records.append({"image": image_path, "missing_annotation": True, "boxes": []})
            continue
        boxes = []
        for i, shape in enumerate(obj.get("shapes", [])):
            pts = shape.get("points") or []
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append(
                {
                    "annotation_id": f"{Path(json_path).stem}_{i}",
                    "annotation_class": str(shape.get("label", "region")),
                    "xyxy": (min(xs), min(ys), max(xs), max(ys)),
                }
            )
        records.append({"image": image_path, "missing_annotation": False, "boxes": boxes, "kind": "abs"})
    return records


def collect_csv(root: Path, csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"filename", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"boxes CSV missing columns: {sorted(missing)}")
    file_index = {p.name: p for p in iter_images(root)}
    grouped: dict[Path, list] = {}
    for i, row in df.iterrows():
        name = str(row["filename"])
        path = root / name
        if not path.exists():
            path = file_index.get(Path(name).name)
        if path is None:
            continue
        path = Path(path)
        grouped.setdefault(path, []).append(
            {
                "annotation_id": str(row.get("annotation_id", i)),
                "annotation_class": str(row.get("annotation_class", row.get("class", "region"))),
                "xyxy": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
            }
        )
    return [
        {"image": path, "missing_annotation": False, "boxes": boxes, "kind": "abs"}
        for path, boxes in grouped.items()
    ]


def pad_box(x1, y1, x2, y2, img_w, img_h, padding: float):
    w, h = x2 - x1, y2 - y1
    px, py = w * padding, h * padding
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, img_w, img_h)


def materialize_box(box: dict, img_w: int, img_h: int, names: dict[int, str]):
    if "xyxy" in box:
        return box["xyxy"][0], box["xyxy"][1], box["xyxy"][2], box["xyxy"][3], box["annotation_class"]
    raw = box.get("yolo_raw")
    if raw:
        parsed = parse_yolo_line(raw, img_w, img_h, names)
        if parsed is None:
            return None
        return parsed
    return None


def main():
    args = parse_args()
    ensure_pipeline_dirs()

    input_root = Path(args.input)
    if not input_root.is_absolute():
        input_root = resolve_path(args.input)
    output_root = resolve_path(args.output)
    images_dir = ensure_dir(output_root / "images")
    crops_csv = output_root / "crops.csv"
    quality_csv = PROJECT_ROOT / "reports" / "dataset_quality_report.csv"

    if not input_root.exists():
        print(f"Input path does not exist yet: {input_root}")
        print("Place the public dataset there and re-run this command.")
        print("No ICDAS labels were created.")
        ensure_dir(images_dir)
        if not crops_csv.exists():
            save_csv(crops_csv, pd.DataFrame(columns=CROP_CSV_COLUMNS), CROP_CSV_COLUMNS)
        return 0

    images = [p for p in iter_images(input_root)]
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(input_root, args.coco_json, args.boxes_csv)
    print(f"Detected/selected annotation format: {fmt}")
    print(
        "Reminder: annotation_class is the dataset region label, NOT an ICDAS grade."
    )

    if fmt == "none":
        print("No supported annotations found (YOLO/COCO/VOC/LabelMe/CSV).")
        print("Images found:", len(images))
        print("Re-run with --format and/or --coco-json / --boxes-csv after inspecting the dataset.")
        if not crops_csv.exists():
            save_csv(crops_csv, pd.DataFrame(columns=CROP_CSV_COLUMNS), CROP_CSV_COLUMNS)
        return 1

    names = yolo_class_names(input_root)
    if fmt == "yolo":
        records = collect_yolo(input_root, images, names)
    elif fmt == "coco":
        coco_path = find_coco_json(input_root, args.coco_json)
        if coco_path is None:
            print("COCO JSON not found. Pass --coco-json.")
            return 1
        records = collect_coco(input_root, coco_path)
    elif fmt == "voc":
        records = collect_voc(input_root, images)
    elif fmt == "labelme":
        records = collect_labelme(input_root, images)
    elif fmt == "csv":
        csv_path = Path(args.boxes_csv) if args.boxes_csv else input_root / "boxes.csv"
        if not csv_path.is_absolute():
            csv_path = input_root / csv_path
        if not csv_path.exists():
            print(f"Boxes CSV not found: {csv_path}")
            return 1
        records = collect_csv(input_root, csv_path)
    else:
        print(f"Unsupported format: {fmt}")
        return 1

    used_ids: set[str] = set()
    existing = load_csv(crops_csv, CROP_CSV_COLUMNS)
    used_ids.update(existing["crop_id"].dropna().astype(str).tolist())

    crop_rows = [] if args.overwrite else existing.to_dict("records")
    quality_rows = []
    resize_to = 0 if args.keep_original_size else int(args.resize)

    processed_images = 0
    saved_crops = 0
    skipped_existing = 0

    for rec in records:
        image_path: Path = rec["image"]
        processed_images += 1
        image = read_image(image_path)
        rel_source = str(image_path.relative_to(input_root)).replace("\\", "/")
        if image is None:
            quality_rows.append(
                {
                    "crop_id": "",
                    "filename": "",
                    "source_image": rel_source,
                    "issue": "corrupt_image",
                    "details": "unreadable",
                    "kept": False,
                }
            )
            continue
        img_h, img_w = image.shape[:2]
        if rec.get("missing_annotation"):
            quality_rows.append(
                {
                    "crop_id": "",
                    "filename": "",
                    "source_image": rel_source,
                    "issue": "missing_annotation",
                    "details": image_path.name,
                    "kept": False,
                }
            )
            continue

        for box in rec.get("boxes", []):
            parsed = materialize_box(box, img_w, img_h, names)
            if parsed is None:
                quality_rows.append(
                    {
                        "crop_id": "",
                        "filename": "",
                        "source_image": rel_source,
                        "issue": "invalid_bounding_box",
                        "details": "unparseable",
                        "kept": False,
                    }
                )
                continue
            x1, y1, x2, y2, ann_class = parsed
            x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, img_w, img_h, args.padding)
            ok, reason = box_is_valid(x1, y1, x2, y2, min_size=args.min_size)
            crop_id = unique_crop_id(image_path.stem, str(box.get("annotation_id", "ann")), used_ids)
            if not ok:
                quality_rows.append(
                    {
                        "crop_id": crop_id,
                        "filename": "",
                        "source_image": rel_source,
                        "issue": "invalid_bounding_box" if "tiny" not in reason else "tiny_crop",
                        "details": reason,
                        "kept": False,
                    }
                )
                continue
            w, h = x2 - x1, y2 - y1
            if extreme_aspect_ratio(w, h, args.max_aspect_ratio):
                quality_rows.append(
                    {
                        "crop_id": crop_id,
                        "filename": "",
                        "source_image": rel_source,
                        "issue": "extreme_aspect_ratio",
                        "details": f"{w}x{h}",
                        "kept": False,
                    }
                )
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                quality_rows.append(
                    {
                        "crop_id": crop_id,
                        "filename": "",
                        "source_image": rel_source,
                        "issue": "invalid_bounding_box",
                        "details": "empty_slice",
                        "kept": False,
                    }
                )
                continue
            if is_blank_image(crop):
                quality_rows.append(
                    {
                        "crop_id": crop_id,
                        "filename": "",
                        "source_image": rel_source,
                        "issue": "blank_image",
                        "details": "low_variance",
                        "kept": False,
                    }
                )
                continue
            if resize_to > 0:
                crop = cv2.resize(crop, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
            filename = f"{crop_id}.jpg"
            dest = images_dir / filename
            if dest.exists() and not args.overwrite:
                skipped_existing += 1
                quality_rows.append(
                    {
                        "crop_id": crop_id,
                        "filename": filename,
                        "source_image": rel_source,
                        "issue": "existing_file_skipped",
                        "details": str(dest),
                        "kept": True,
                    }
                )
                continue
            if not write_image(dest, crop, overwrite=args.overwrite):
                quality_rows.append(
                    {
                        "crop_id": crop_id,
                        "filename": filename,
                        "source_image": rel_source,
                        "issue": "write_failed",
                        "details": str(dest),
                        "kept": False,
                    }
                )
                continue
            crop_rows.append(
                {
                    "crop_id": crop_id,
                    "filename": filename,
                    "source_image": rel_source,
                    "annotation_id": str(box.get("annotation_id", "")),
                    "annotation_class": str(ann_class),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": w,
                    "height": h,
                }
            )
            saved_crops += 1
            quality_rows.append(
                {
                    "crop_id": crop_id,
                    "filename": filename,
                    "source_image": rel_source,
                    "issue": "ok",
                    "details": "",
                    "kept": True,
                }
            )

    crop_df = pd.DataFrame(crop_rows)
    if not crop_df.empty:
        crop_df = crop_df.drop_duplicates(subset=["crop_id"], keep="last")
    else:
        crop_df = pd.DataFrame(columns=CROP_CSV_COLUMNS)
    save_csv(crops_csv, crop_df, CROP_CSV_COLUMNS)

    qdf = pd.DataFrame(quality_rows)
    if qdf.empty:
        qdf = pd.DataFrame(columns=QUALITY_CSV_COLUMNS)
    save_csv(quality_csv, qdf, QUALITY_CSV_COLUMNS)

    print(f"Images processed: {processed_images}")
    print(f"Crops saved:      {saved_crops}")
    print(f"Existing skipped: {skipped_existing}")
    print(f"crops.csv:        {crops_csv}")
    print(f"quality report:   {quality_csv}")
    print("ICDAS grades were NOT assigned from annotation classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
