"""Stage 3A: build whole-tooth detection dataset from existing RGB photos.

Read-only on originals. Does not touch dataset/, ml/, ICDAS, models, or app code.
Does not create tooth boxes or FDI labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat, UnidentifiedImageError

ROOT = Path(r"c:\Users\anant\OneDrive\Desktop\icdas project")
SRC_RAW = ROOT / "data_external" / "detection" / "raw"
SRC_ANN = ROOT / "data_external" / "detection" / "annotations"
SRC_YOLO = SRC_ANN / "yolo"
SRC_VOC = SRC_ANN / "pascal-voc"
SRC_COCO = SRC_ANN / "ms_coco"
SOURCE_TXT = ROOT / "data_external" / "detection" / "SOURCE.txt"

OUT = ROOT / "fdi_detection_dataset"
IMG_SEL = OUT / "images" / "selected"
IMG_REV = OUT / "images" / "review"
IMG_REJ = OUT / "images" / "rejected"
ANN_VOC = OUT / "annotations" / "pascal_voc"
ANN_COCO = OUT / "annotations" / "coco"
ANN_YOLO = OUT / "annotations" / "yolo"
ANN_FDI = ROOT / "archive" / "out_of_scope" / "fdi" / "fdi_mapping"
META = OUT / "metadata"
DS_REPORTS = OUT / "reports"
PROJ_REPORTS = ROOT / "reports"

ID_RE = re.compile(r"anonymous_(\d{3}-\d{3}-\d+)")
VIEW_MAP = {
    "Frontal": "Frontal",
    "Left_Lateral": "Left Buccal",
    "Right_Lateral": "Right Buccal",
    "Mandibular": "Mandibular",
    "Maxillary_Occlusal": "Maxillary",
}
WORKERS = 8
TARGET_SELECTED = 420
TARGET_REVIEW = 80
RNG = random.Random(3)


def ensure_dirs() -> None:
    for p in (
        IMG_SEL,
        IMG_REV,
        IMG_REJ,
        ANN_VOC,
        ANN_COCO,
        ANN_YOLO,
        ANN_FDI,
        META,
        DS_REPORTS,
        PROJ_REPORTS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def orientation_from_rel(rel: Path) -> dict:
    parts = rel.parts
    protocol = parts[0] if len(parts) >= 2 else "unknown"
    view_folder = parts[1] if len(parts) >= 2 else "Unknown"
    orientation = VIEW_MAP.get(view_folder, "Unknown")
    occlusal = view_folder in ("Mandibular", "Maxillary_Occlusal")
    return {
        "protocol": protocol,
        "view_folder": view_folder,
        "orientation": orientation,
        "occlusal": occlusal,
        "category_list": _categories(orientation, occlusal, view_folder),
    }


def _categories(orientation: str, occlusal: bool, view_folder: str) -> list[str]:
    cats = []
    if orientation == "Maxillary":
        cats.extend(["Maxillary", "Occlusal"])
    elif orientation == "Mandibular":
        cats.extend(["Mandibular", "Occlusal"])
    elif orientation == "Frontal":
        cats.append("Frontal")
    elif orientation == "Left Buccal":
        cats.append("Left Buccal")
    elif orientation == "Right Buccal":
        cats.append("Right Buccal")
    else:
        cats.append("Unknown")
    return cats


def visibility_from_view(view_folder: str) -> dict:
    if view_folder in ("Mandibular", "Maxillary_Occlusal"):
        return {
            "visibility_class": "full_arch",
            "arch": "full_arch",
            "region": "mixed",
            "visible_teeth_estimate": 14,
            "estimate_note": "ESTIMATE from occlusal view type; not counted per image; not FDI",
        }
    if view_folder == "Frontal":
        return {
            "visibility_class": "anterior_only",
            "arch": "partial_arch",
            "region": "anterior",
            "visible_teeth_estimate": 12,
            "estimate_note": "ESTIMATE from frontal view type; posteriors often out of focus; not FDI",
        }
    if view_folder in ("Left_Lateral", "Right_Lateral"):
        return {
            "visibility_class": "posterior_partial",
            "arch": "partial_arch",
            "region": "posterior",
            "visible_teeth_estimate": 8,
            "estimate_note": "ESTIMATE from lateral view type; overlapping crowns; not FDI",
        }
    return {
        "visibility_class": "unknown",
        "arch": "unknown",
        "region": "unknown",
        "visible_teeth_estimate": None,
        "estimate_note": "Unknown view folder",
    }


def ahash64(gray_small: Image.Image) -> int:
    g = gray_small.resize((8, 8), Image.BILINEAR)
    pix = np.asarray(g, dtype=np.float32).reshape(-1)
    avg = float(pix.mean())
    bits = (pix >= avg).astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def laplacian_var(arr: np.ndarray) -> float:
    x = arr.astype(np.float32)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # valid conv via slices
    p = x[0:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, 0:-2] + x[1:-1, 2:] - 4 * x[1:-1, 1:-1]
    return float(p.var()) if p.size else 0.0


def classify_quality(width: int, height: int, sharpness: float, brightness: float, contrast: float) -> str:
    mn = min(width, height)
    # REJECT only clearly unusable
    if mn < 80:
        return "REJECT"
    if brightness < 12 or brightness > 248:
        return "REJECT"
    if sharpness < 4.0:
        return "REJECT"
    if contrast < 3.0:
        return "REJECT"
    if sharpness < 25.0 or mn < 350 or brightness < 28 or brightness > 230:
        return "LOW QUALITY"
    if sharpness >= 80.0 and mn >= 800 and 45 <= brightness <= 210 and contrast >= 25:
        return "HIGH QUALITY"
    return "MEDIUM QUALITY"


def exposure_label(brightness: float) -> str:
    if brightness < 12:
        return "severe_underexposure"
    if brightness < 40:
        return "underexposed"
    if brightness > 248:
        return "severe_overexposure"
    if brightness > 220:
        return "overexposed"
    return "acceptable"


def patient_from_name(name: str) -> str:
    m = ID_RE.search(name)
    return m.group(1) if m else ""


def yolo_label_path(rel: Path) -> Path:
    return SRC_YOLO / rel.with_suffix(".txt")


def voc_label_path(rel: Path) -> Path:
    return SRC_VOC / rel.with_suffix(".xml")


def inspect_one(path: Path) -> dict:
    rel = path.relative_to(SRC_RAW)
    rec = {
        "original_filename": path.name,
        "source_path": str(path.as_posix()),
        "rel_posix": rel.as_posix(),
        "extension": path.suffix.lower(),
        "readable": False,
        "corrupted": False,
        "mode": "",
        "channels": 0,
        "width": 0,
        "height": 0,
        "md5": "",
        "ahash": 0,
        "sharpness": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_estimate": 0.0,
        "exposure": "unknown",
        "quality": "REJECT",
        "error": "",
    }
    rec.update(orientation_from_rel(rel))
    rec.update(visibility_from_view(rec["view_folder"]))
    rec["patient_identifier_if_available"] = patient_from_name(path.name)

    try:
        rec["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    except OSError as e:
        rec["corrupted"] = True
        rec["error"] = f"read:{e}"
        rec["quality"] = "REJECT"
        return rec

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            rgb = im.convert("RGB")
            thumb = rgb.copy()
            thumb.thumbnail((256, 256), Image.BILINEAR)
            arr = np.asarray(thumb.convert("L"), dtype=np.float32)
            rec["sharpness"] = round(laplacian_var(arr), 4)
            rec["blur_estimate"] = round(1.0 / (rec["sharpness"] + 1e-6), 6)
            rec["brightness"] = round(float(arr.mean()), 3)
            rec["contrast"] = round(float(arr.std()), 3)
            rec["ahash"] = ahash64(thumb.convert("L"))
            rec["channels"] = 3
            if im.mode in ("L", "LA", "1"):
                rec["channels"] = 1
            rec["readable"] = True
    except (UnidentifiedImageError, OSError, ValueError) as e:
        rec["corrupted"] = True
        rec["readable"] = False
        rec["error"] = str(e)
        rec["quality"] = "REJECT"
        return rec

    rec["exposure"] = exposure_label(rec["brightness"])
    rec["quality"] = classify_quality(
        rec["width"], rec["height"], rec["sharpness"], rec["brightness"], rec["contrast"]
    )
    if rec["channels"] == 1:
        rec["color"] = "grayscale"
    else:
        rec["color"] = "RGB"
    return rec


def lesion_stem_index() -> dict[str, dict]:
    """Map image stem (original jpg stem) to lesion annotation presence."""
    yolo_stems = {}
    for p in SRC_YOLO.rglob("*.txt"):
        yolo_stems[p.stem] = p
    voc_stems = {}
    for p in SRC_VOC.rglob("*.xml"):
        voc_stems[p.stem] = p
    return {"yolo": yolo_stems, "voc": voc_stems}


def count_xml_objects() -> dict:
    xml_files = list(SRC_VOC.rglob("*.xml"))
    labels = Counter()
    objects = 0
    files_with_obj = 0
    per_file = []
    parse_errors = 0
    for p in xml_files:
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            parse_errors += 1
            continue
        objs = root.findall("object")
        n = len(objs)
        per_file.append(n)
        objects += n
        if n:
            files_with_obj += 1
        for obj in objs:
            name = (obj.findtext("name") or "").strip()
            labels[name] += 1
    return {
        "xml_files": len(xml_files),
        "xml_parse_errors": parse_errors,
        "object_count": objects,
        "files_with_objects": files_with_obj,
        "unique_labels": dict(labels),
        "mean_objects_per_xml": round(objects / len(xml_files), 4) if xml_files else 0,
        "objects_per_file_histogram": dict(Counter(per_file)),
    }


def annotation_fields(rec: dict, yolo_stems: dict, voc_stems: dict) -> None:
    stem = Path(rec["original_filename"]).stem
    # stems in voc sometimes use underscores instead of hyphens
    alt = stem.replace("-", "_")
    has_yolo = stem in yolo_stems or alt in yolo_stems
    has_voc = stem in voc_stems or alt in voc_stems
    rec["annotation_exists"] = bool(has_yolo or has_voc)
    rec["annotation_type"] = "lesion_d_D" if rec["annotation_exists"] else "none"
    rec["lesion_annotation_note"] = (
        "Existing XML/YOLO boxes are decay lesions (d/D), NOT whole-tooth boxes. Unused for this dataset."
        if rec["annotation_exists"]
        else "No lesion annotation file matched this image stem."
    )


def mark_duplicates(rows: list[dict]) -> None:
    by_md5 = defaultdict(list)
    by_name = defaultdict(list)
    for i, r in enumerate(rows):
        by_name[r["original_filename"]].append(i)
        if r["md5"]:
            by_md5[r["md5"]].append(i)

    for r in rows:
        r["filename_duplicate"] = False
        r["hash_duplicate"] = False
        r["perceptual_duplicate"] = False
        r["duplicate_status"] = "unique"
        r["duplicate_group_id"] = ""
        r["canonical"] = True

    # filename collisions (same basename, different folders)
    for name, idxs in by_name.items():
        if len(idxs) > 1:
            for j, i in enumerate(idxs):
                rows[i]["filename_duplicate"] = True
                if j == 0:
                    rows[i]["duplicate_status"] = "filename_collision_canonical"
                else:
                    rows[i]["duplicate_status"] = "filename_collision"
                    rows[i]["canonical"] = False

    gid = 0
    for h, idxs in by_md5.items():
        if len(idxs) < 2 or not h:
            continue
        gid += 1
        group = f"md5_{h[:12]}"
        # canonical = first by path
        idxs_sorted = sorted(idxs, key=lambda i: rows[i]["rel_posix"])
        for j, i in enumerate(idxs_sorted):
            rows[i]["hash_duplicate"] = True
            rows[i]["duplicate_group_id"] = group
            if j == 0:
                rows[i]["duplicate_status"] = "hash_canonical"
                rows[i]["canonical"] = True
            else:
                rows[i]["duplicate_status"] = f"duplicate_of_{rows[idxs_sorted[0]]['original_filename']}"
                rows[i]["canonical"] = False

    # perceptual near-duplicates among unique-md5 HIGH/MEDIUM (bucket by ahash)
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        if not r["readable"] or r["hash_duplicate"]:
            continue
        buckets[r["ahash"]].append(i)
        # also nearby: skip extra buckets; exact ahash match first
    pgid = 0
    for ah, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        pgid += 1
        idxs_sorted = sorted(idxs, key=lambda i: rows[i]["rel_posix"])
        group = f"ahash_{ah:016x}"
        for j, i in enumerate(idxs_sorted):
            rows[i]["perceptual_duplicate"] = True
            if not rows[i]["hash_duplicate"]:
                rows[i]["duplicate_group_id"] = group
            if j == 0:
                if rows[i]["duplicate_status"] == "unique":
                    rows[i]["duplicate_status"] = "perceptual_canonical"
                rows[i]["canonical"] = True
            else:
                if not rows[i]["hash_duplicate"]:
                    rows[i]["duplicate_status"] = (
                        f"perceptual_duplicate_of_{rows[idxs_sorted[0]]['original_filename']}"
                    )
                    rows[i]["canonical"] = False


def select_diverse(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rejected = [r for r in rows if (not r["readable"]) or r["quality"] == "REJECT" or r["corrupted"]]
    pool = [
        r
        for r in rows
        if r["readable"]
        and r["quality"] != "REJECT"
        and r["canonical"]
        and not r["hash_duplicate"]
        or (r["canonical"] and r.get("duplicate_status") == "hash_canonical")
    ]
    # cleaner pool
    pool = [
        r
        for r in rows
        if r["readable"]
        and r["quality"] != "REJECT"
        and r["canonical"]
        and r["duplicate_status"]
        not in (
            None,
        )
        and not str(r["duplicate_status"]).startswith("duplicate_of_")
        and not str(r["duplicate_status"]).startswith("perceptual_duplicate_of_")
        and r["duplicate_status"] != "filename_collision"
    ]

    views = ["Frontal", "Left_Lateral", "Mandibular", "Maxillary_Occlusal", "Right_Lateral"]
    protocols = ["retractors", "no_retractors", "pilot"]
    # 420 = 5 views * 84; 84 = 37+37+10
    quotas = {
        "retractors": 37,
        "no_retractors": 37,
        "pilot": 10,
    }

    selected: list[dict] = []
    used_patients: set[str] = set()
    used_paths: set[str] = set()

    def qrank(r: dict) -> tuple:
        q = {"HIGH QUALITY": 0, "MEDIUM QUALITY": 1, "LOW QUALITY": 2}.get(r["quality"], 3)
        return (q, -r["sharpness"], -min(r["width"], r["height"]))

    for view in views:
        for proto in protocols:
            cand = [
                r
                for r in pool
                if r["view_folder"] == view
                and r["protocol"] == proto
                and r["rel_posix"] not in used_paths
            ]
            cand.sort(key=qrank)
            # prefer unused patients
            unused = [r for r in cand if r["patient_identifier_if_available"] and r["patient_identifier_if_available"] not in used_patients]
            usedp = [r for r in cand if r["patient_identifier_if_available"] in used_patients]
            nop = [r for r in cand if not r["patient_identifier_if_available"]]
            ordered = unused + nop + usedp
            take = quotas[proto]
            n = 0
            for r in ordered:
                if n >= take:
                    break
                selected.append(r)
                used_paths.add(r["rel_posix"])
                if r["patient_identifier_if_available"]:
                    used_patients.add(r["patient_identifier_if_available"])
                n += 1

    # fill to TARGET if short
    if len(selected) < TARGET_SELECTED:
        rest = [r for r in pool if r["rel_posix"] not in used_paths]
        rest.sort(key=qrank)
        for r in rest:
            if len(selected) >= TARGET_SELECTED:
                break
            selected.append(r)
            used_paths.add(r["rel_posix"])
            if r["patient_identifier_if_available"]:
                used_patients.add(r["patient_identifier_if_available"])

    selected = selected[:TARGET_SELECTED]
    selected_set = {r["rel_posix"] for r in selected}

    # review: 80 from remaining LOW / extremes / missing id / laterals
    remain = [r for r in pool if r["rel_posix"] not in selected_set]
    review_reasons: dict[str, str] = {}

    def review_score(r: dict) -> tuple:
        score = 0
        reasons = []
        if r["quality"] == "LOW QUALITY":
            score += 5
            reasons.append("poor_lighting_or_blur")
        if r["exposure"] in ("underexposed", "overexposed"):
            score += 3
            reasons.append(r["exposure"])
        if r["view_folder"] in ("Left_Lateral", "Right_Lateral"):
            score += 2
            reasons.append("lateral_occlusion_overlap")
        if not r["patient_identifier_if_available"]:
            score += 2
            reasons.append("missing_anonymous_id_pattern")
        if min(r["width"], r["height"]) < 500:
            score += 2
            reasons.append("low_resolution")
        if r["protocol"] == "no_retractors":
            score += 1
            reasons.append("no_retractors_possible_occlusion")
        aspect = r["width"] / max(r["height"], 1)
        if aspect < 0.7 or aspect > 2.2:
            score += 2
            reasons.append("unusual_aspect_ratio")
        review_reasons[r["rel_posix"]] = ";".join(reasons) if reasons else "borderline_quality"
        return (-score, qrank(r)[0])

    remain.sort(key=review_score)
    review = remain[:TARGET_REVIEW]
    for r in review:
        r["review_reason"] = review_reasons.get(r["rel_posix"], "human_inspection")

    for r in selected:
        r["selection_status"] = "selected"
        r["review_reason"] = ""
    for r in review:
        r["selection_status"] = "review"
    for r in rejected:
        r["selection_status"] = "rejected"
        r["reject_reason"] = (
            "corrupted_or_unreadable"
            if (r["corrupted"] or not r["readable"])
            else f"quality_REJECT:{r['exposure']}:sharpness={r['sharpness']}"
        )

    return selected, review, rejected


def copy_named(rows: list[dict], dest: Path, prefix_collision: bool = True) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    used = set()
    for r in rows:
        src = Path(r["source_path"])
        name = r["original_filename"]
        if name in used:
            # preserve original name when unique; if collision, prefix protocol_view
            stem, ext = os.path.splitext(name)
            name = f"{r['protocol']}__{r['view_folder']}__{stem}{ext}"
        used.add(name)
        r["current_filename"] = name
        shutil.copy2(src, dest / name)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    ensure_dirs()
    images = sorted(SRC_RAW.rglob("*.jpg")) + sorted(SRC_RAW.rglob("*.jpeg")) + sorted(SRC_RAW.rglob("*.png"))
    images = [p for p in images if p.is_file()]

    print(f"found_images={len(images)}", flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(inspect_one, p): p for p in images}
        n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            rows.append(rec)
            n += 1
            if n % 250 == 0:
                print(f"scanned {n}/{len(images)}", flush=True)

    rows.sort(key=lambda r: r["rel_posix"])
    print("scan_done", flush=True)

    yolo_stems = {p.stem: p for p in SRC_YOLO.rglob("*.txt")}
    voc_stems = {p.stem: p for p in SRC_VOC.rglob("*.xml")}
    # also index hyphen/underscore variants
    yolo_stems.update({k.replace("_", "-"): v for k, v in list(yolo_stems.items())})
    voc_stems.update({k.replace("_", "-"): v for k, v in list(voc_stems.items())})

    for r in rows:
        annotation_fields(r, yolo_stems, voc_stems)
        r["source_dataset"] = "Zenodo 10.5281/zenodo.14827784"

    mark_duplicates(rows)
    xml_stats = count_xml_objects()
    print("xml_stats", xml_stats["xml_files"], xml_stats["object_count"], flush=True)

    selected, review, rejected = select_diverse(rows)
    print(f"selected={len(selected)} review={len(review)} rejected={len(rejected)}", flush=True)

    # copy (never move originals)
    copy_named(selected, IMG_SEL)
    copy_named(review, IMG_REV)
    copy_named(rejected, IMG_REJ)

    # empty YOLO placeholders for selected only
    for r in selected:
        (ANN_YOLO / (Path(r["current_filename"]).stem + ".txt")).write_text("", encoding="utf-8")

    # empty VOC placeholders (no objects)
    for r in selected:
        stem = Path(r["current_filename"]).stem
        xml = (
            f'<annotation><folder>selected</folder><filename>{r["current_filename"]}</filename>'
            f'<size><width>{r["width"]}</width><height>{r["height"]}</height><depth>3</depth></size>'
            f"</annotation>\n"
        )
        (ANN_VOC / f"{stem}.xml").write_text(xml, encoding="utf-8")

    coco = {
        "info": {
            "description": "Stage 3A whole-tooth detection placeholder. Annotations array EMPTY. No tooth boxes. No FDI.",
            "version": "stage3a",
            "year": 2026,
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "contributor": "ICCC major project Stage 3A",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0 (source Zenodo 10.5281/zenodo.14827784)",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "placeholder_whole_tooth", "supercategory": "tooth"}],
    }
    for i, r in enumerate(selected, start=1):
        coco["images"].append(
            {
                "id": i,
                "file_name": r["current_filename"],
                "width": r["width"],
                "height": r["height"],
                "license": 1,
                "original_relpath": r["rel_posix"],
            }
        )
    (ANN_COCO / "instances_placeholder.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")

    manifest_fields = [
        "original_filename",
        "current_filename",
        "source_dataset",
        "source_path",
        "width",
        "height",
        "channels",
        "orientation",
        "quality",
        "duplicate_status",
        "selection_status",
        "annotation_exists",
        "annotation_type",
        "patient_identifier_if_available",
    ]
    for r in rows:
        if "current_filename" not in r:
            r["current_filename"] = r["original_filename"]
        if "selection_status" not in r:
            r["selection_status"] = "not_selected"

    sel_paths = {r["rel_posix"] for r in selected}
    rev_paths = {r["rel_posix"] for r in review}
    rej_paths = {r["rel_posix"] for r in rejected}
    for r in rows:
        if r["rel_posix"] in sel_paths:
            r["selection_status"] = "selected"
            # current_filename already set on selected objects (same dict refs)
        elif r["rel_posix"] in rev_paths:
            r["selection_status"] = "review"
        elif r["rel_posix"] in rej_paths:
            r["selection_status"] = "rejected"
        else:
            r["selection_status"] = "not_selected"
            r["current_filename"] = r["original_filename"]

    write_csv(META / "image_manifest.csv", rows, manifest_fields)
    write_csv(PROJ_REPORTS / "stage3a_selected_images.csv", selected, manifest_fields + [
        "protocol", "view_folder", "quality", "sharpness", "brightness", "rel_posix"
    ])
    dup_rows = [r for r in rows if r["duplicate_status"] != "unique"]
    write_csv(
        PROJ_REPORTS / "stage3a_duplicate_report.csv",
        dup_rows,
        [
            "original_filename",
            "rel_posix",
            "md5",
            "ahash",
            "duplicate_status",
            "duplicate_group_id",
            "canonical",
            "filename_duplicate",
            "hash_duplicate",
            "perceptual_duplicate",
        ],
    )
    write_csv(
        PROJ_REPORTS / "stage3a_quality_report.csv",
        rows,
        [
            "original_filename",
            "rel_posix",
            "width",
            "height",
            "channels",
            "color",
            "readable",
            "corrupted",
            "sharpness",
            "blur_estimate",
            "brightness",
            "contrast",
            "exposure",
            "quality",
            "error",
        ],
    )
    write_csv(
        PROJ_REPORTS / "stage3a_orientation_report.csv",
        rows,
        [
            "original_filename",
            "rel_posix",
            "protocol",
            "view_folder",
            "orientation",
            "occlusal",
            "visibility_class",
            "arch",
            "region",
            "visible_teeth_estimate",
            "selection_status",
        ],
    )

    # copy CSVs into dataset reports/
    for name in (
        "stage3a_selected_images.csv",
        "stage3a_duplicate_report.csv",
        "stage3a_quality_report.csv",
        "stage3a_orientation_report.csv",
    ):
        shutil.copy2(PROJ_REPORTS / name, DS_REPORTS / name)

    valid = [r for r in rows if r["readable"] and not r["corrupted"]]
    rgb = [r for r in valid if r.get("color") == "RGB"]
    gray = [r for r in valid if r.get("color") == "grayscale"]
    hash_dup_extra = sum(1 for r in rows if str(r["duplicate_status"]).startswith("duplicate_of_"))
    perc_dup_extra = sum(1 for r in rows if str(r["duplicate_status"]).startswith("perceptual_duplicate_of_"))

    def dist(key, subset=None):
        src = subset if subset is not None else rows
        return dict(Counter(r[key] for r in src))

    avg_w = round(sum(r["width"] for r in valid) / len(valid), 2) if valid else 0
    avg_h = round(sum(r["height"] for r in valid) / len(valid), 2) if valid else 0
    vis_est = [r["visible_teeth_estimate"] for r in selected if r["visible_teeth_estimate"] is not None]
    avg_vis = round(sum(vis_est) / len(vis_est), 2) if vis_est else 0

    patients_all = {r["patient_identifier_if_available"] for r in rows if r["patient_identifier_if_available"]}
    patients_sel = {r["patient_identifier_if_available"] for r in selected if r["patient_identifier_if_available"]}

    inventory = {
        "stage": "3A",
        "title": "RGB whole-tooth detection dataset construction",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "clinical_note": "No FDI generated. No tooth boxes generated. Lesion d/D unused. ICDAS unmodified.",
        "source": {
            "name": "Zenodo 10.5281/zenodo.14827784",
            "title": "Annotated intraoral image dataset for dental caries detection",
            "url": "https://zenodo.org/records/14827784",
            "license": "CC BY 4.0",
            "documented_in": str(SOURCE_TXT.as_posix()),
            "image_root": str(SRC_RAW.as_posix()),
            "annotation_root": str(SRC_ANN.as_posix()),
        },
        "counts": {
            "total_rgb_images_found": len(images),
            "valid_readable": len(valid),
            "rgb": len(rgb),
            "grayscale": len(gray),
            "unreadable_or_corrupted": sum(1 for r in rows if r["corrupted"] or not r["readable"]),
            "selected": len(selected),
            "review": len(review),
            "rejected_copied": len(rejected),
            "not_selected": sum(1 for r in rows if r["selection_status"] == "not_selected"),
            "xml_files": xml_stats["xml_files"],
            "xml_object_count": xml_stats["object_count"],
            "xml_unique_labels": xml_stats["unique_labels"],
            "tooth_annotations": 0,
            "fdi_annotations": 0,
            "yolo_placeholder_files": len(selected),
            "coco_placeholder_created": True,
        },
        "duplicates": {
            "filename_collisions": sum(1 for r in rows if r["filename_duplicate"]),
            "md5_extra_copies": hash_dup_extra,
            "perceptual_extra_copies": perc_dup_extra,
            "rows_not_unique_status": len(dup_rows),
        },
        "xml_inventory": xml_stats,
        "quality_distribution_all": dist("quality"),
        "orientation_distribution_all": dist("orientation"),
        "view_folder_distribution_all": dist("view_folder"),
        "protocol_distribution_all": dist("protocol"),
        "selected_view": dist("view_folder", selected),
        "selected_protocol": dist("protocol", selected),
        "selected_orientation": dist("orientation", selected),
        "selected_quality": dist("quality", selected),
        "patients_with_parsed_id_all": len(patients_all),
        "patients_with_parsed_id_selected": len(patients_sel),
        "average_width_valid": avg_w,
        "average_height_valid": avg_h,
        "average_visible_teeth_estimate_selected": avg_vis,
        "visible_teeth_estimate_is": "ESTIMATE from view type only; not per-image counts; not FDI",
        "outputs": {
            "dataset_root": str(OUT.as_posix()),
            "selected_images": str(IMG_SEL.as_posix()),
            "coco": str((ANN_COCO / "instances_placeholder.json").as_posix()),
            "yolo": str(ANN_YOLO.as_posix()),
        },
        "decisions": {
            "q1_ready_for_tooth_annotation": len(selected),
            "q2_rejected": len(rejected),
            "q3_sufficient_for_detection_dataset": True,
            "q4_icdas_modified": False,
            "q5_fdi_generated": False,
            "q6_tooth_boxes_generated": False,
            "q7_next_stage": "Stage 3B — Whole-Tooth Bounding Box Annotation Preparation",
        },
    }
    (PROJ_REPORTS / "stage3a_detection_dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    shutil.copy2(
        PROJ_REPORTS / "stage3a_detection_dataset_inventory.json",
        DS_REPORTS / "stage3a_detection_dataset_inventory.json",
    )

    review_csv_fields = manifest_fields + ["review_reason", "protocol", "view_folder"]
    write_csv(META / "review_set.csv", review, review_csv_fields)
    write_csv(META / "rejected_set.csv", rejected, manifest_fields + ["reject_reason"])

    print("DONE", json.dumps(inventory["counts"]), flush=True)


if __name__ == "__main__":
    main()
