#!/usr/bin/env python3
"""Build gold_detector_dataset from Batch 01 GT + Round 1/2 GOOD. Copy only. No training."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "detection" / "gold_detector_dataset"
B01 = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01"
R1 = ROOT / "data" / "detection" / "batches" / "batch02_manual_good"
R2 = ROOT / "data" / "detection" / "batches" / "batch02_manual_round2" / "good"
DEDUP_REPORT = ROOT / "reports" / "GOLD_DETECTOR_DEDUPLICATION_REPORT.md"
DATASET_REPORT = ROOT / "reports" / "GOLD_DETECTOR_DATASET_REPORT.md"
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def source_stem(name: str) -> str:
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_boxes(label: Path) -> int:
    if not label.exists():
        return 0
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_yolo(label: Path) -> tuple[int, list[str]]:
    n = 0
    errors = []
    if not label.exists():
        return 0, ["missing_label"]
    for i, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"row {i}: expected 5 fields")
            continue
        try:
            cid, xc, yc, w, h = (float(x) for x in parts)
        except ValueError:
            errors.append(f"row {i}: non-numeric")
            continue
        n += 1
        if int(cid) != 0:
            errors.append(f"row {i}: class {int(cid)} != 0")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            errors.append(f"row {i}: center out of [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"row {i}: non-positive size")
        if xc - w / 2 < -1e-3 or yc - h / 2 < -1e-3 or xc + w / 2 > 1 + 1e-3 or yc + h / 2 > 1 + 1e-3:
            errors.append(f"row {i}: box outside image")
    return n, errors


def collect_source(src: Path, source_id: str, split_hint_from_path: bool) -> list[dict]:
    items = []
    if source_id == "batch01":
        for split in ("train", "val", "test"):
            img_dir = src / "images" / split
            lab_dir = src / "labels" / split
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lab = lab_dir / f"{img.stem}.txt"
                items.append(
                    {
                        "source": source_id,
                        "orig_split": "valid" if split == "val" else split,
                        "name": img.name,
                        "img": img,
                        "lab": lab,
                        "stem": source_stem(img.name),
                    }
                )
        return items
    if source_id == "round1":
        for split in ("train", "valid", "test"):
            img_dir = src / "images" / split
            lab_dir = src / "labels" / split
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lab = lab_dir / f"{img.stem}.txt"
                items.append(
                    {
                        "source": source_id,
                        "orig_split": split,
                        "name": img.name,
                        "img": img,
                        "lab": lab,
                        "stem": source_stem(img.name),
                    }
                )
        return items
    # round2: flat good/images
    img_dir = src / "images"
    lab_dir = src / "labels"
    for img in sorted(img_dir.iterdir()) if img_dir.exists() else []:
        if img.suffix.lower() not in IMG_EXT:
            continue
        lab = lab_dir / f"{img.stem}.txt"
        items.append(
            {
                "source": source_id,
                "orig_split": None,
                "name": img.name,
                "img": img,
                "lab": lab,
                "stem": source_stem(img.name),
            }
        )
    return items


def assign_splits(kept: list[dict]) -> None:
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in kept:
        groups[f"{it['source']}::{it['stem']}"].append(it)

    forced: dict[str, str] = {}
    for key, members in groups.items():
        if members[0]["source"] != "batch01":
            continue
        orig = {m["orig_split"] for m in members}
        if orig == {"test"}:
            forced[key] = "test"
        elif orig == {"valid"}:
            forced[key] = "valid"
        elif orig == {"train"}:
            forced[key] = "train"

    remaining = [k for k in groups if k not in forced]
    rng.shuffle(remaining)
    n = len(kept)
    n_test = max(1, round(n * 0.15))
    n_valid = max(1, round(n * 0.15))
    # fill test first with forced B01 test, then remaining groups
    split_of = dict(forced)
    counts = {"train": 0, "valid": 0, "test": 0}
    for key, sp in forced.items():
        counts[sp] += len(groups[key])

    def put(key: str, sp: str) -> None:
        split_of[key] = sp
        counts[sp] += len(groups[key])

    for key in remaining:
        if counts["test"] < n_test:
            put(key, "test")
        elif counts["valid"] < n_valid:
            put(key, "valid")
        else:
            put(key, "train")

    for key, members in groups.items():
        sp = split_of[key]
        for m in members:
            m["gold_split"] = sp


def dest_name(item: dict, used: set[str]) -> str:
    name = item["name"]
    if name not in used:
        return name
    alt = f"{item['source']}__{name}"
    if alt not in used:
        return alt
    stem, suf = Path(name).stem, Path(name).suffix
    i = 2
    while True:
        cand = f"{stem}__{item['source']}_{i}{suf}"
        if cand not in used:
            return cand
        i += 1


def main() -> None:
    raw = (
        collect_source(B01, "batch01", True)
        + collect_source(R1, "round1", True)
        + collect_source(R2, "round2", False)
    )
    for it in raw:
        it["n_boxes"] = count_boxes(it["lab"])
        it["md5"] = file_md5(it["img"]) if it["img"].exists() else ""

    before = {
        "n": len(raw),
        "boxes": sum(it["n_boxes"] for it in raw),
        "by_source": {},
    }
    for sid in ("batch01", "round1", "round2"):
        sub = [x for x in raw if x["source"] == sid]
        before["by_source"][sid] = {"images": len(sub), "boxes": sum(x["n_boxes"] for x in sub)}

    dup_filename = defaultdict(list)
    dup_stem = defaultdict(list)
    dup_hash = defaultdict(list)
    for it in raw:
        dup_filename[it["name"]].append(it)
        dup_stem[it["stem"]].append(it)
        dup_hash[it["md5"]].append(it)

    filename_collisions = {k: v for k, v in dup_filename.items() if len(v) > 1}
    stem_collisions = {k: v for k, v in dup_stem.items() if len({x["source"] for x in v}) > 1 or len(v) > 1 and len({x["name"] for x in v}) > 1}
    # stems duplicated within one source with different filenames (rf copies)
    stem_multi_file = {k: v for k, v in dup_stem.items() if len({x["name"] for x in v}) > 1}
    hash_collisions = {k: v for k, v in dup_hash.items() if k and len(v) > 1}

    kept: list[dict] = []
    removed: list[dict] = []
    seen_hash: set[str] = set()
    seen_stem: set[str] = set()
    seen_name: set[str] = set()
    # Priority: batch01, then round1, then round2
    order = {"batch01": 0, "round1": 1, "round2": 2}
    ranked = sorted(raw, key=lambda x: (order[x["source"]], x["name"]))
    for it in ranked:
        reasons = []
        if it["name"] in seen_name:
            reasons.append("duplicate_filename")
        if it["stem"] in seen_stem:
            reasons.append("duplicate_source_stem")
        if it["md5"] and it["md5"] in seen_hash:
            reasons.append("identical_image_bytes")
        if reasons:
            it["drop_reasons"] = reasons
            removed.append(it)
            continue
        seen_name.add(it["name"])
        seen_stem.add(it["stem"])
        if it["md5"]:
            seen_hash.add(it["md5"])
        kept.append(it)

    assign_splits(kept)

    if DEST.exists():
        shutil.rmtree(DEST)
    used_names: set[str] = set()
    copied = []
    for it in kept:
        split = it["gold_split"]
        out_name = dest_name(it, used_names)
        used_names.add(out_name)
        img_dst = DEST / split / "images" / out_name
        lab_dst = DEST / split / "labels" / f"{Path(out_name).stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["img"], img_dst)
        shutil.copy2(it["lab"], lab_dst)
        it["gold_name"] = out_name
        copied.append(it)

    yaml = (
        "# Gold whole-tooth detector dataset. Batch 01 GT + Batch 02 Round 1/2 GOOD only.\n"
        "# Do not overwrite Batch 01 weights. Training is a separate step.\n"
        f"path: {DEST.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  - tooth\n"
    )
    (DEST / "data.yaml").write_text(yaml, encoding="utf-8")
    (DEST / "README.md").write_text(
        "Gold detector candidate. Copied from Batch 01 human GT and Batch 02 GOOD reviews. Sources unchanged.\n",
        encoding="utf-8",
    )

    # validation
    val_errors = []
    val_ok = 0
    box_total = 0
    per_split = {s: {"images": 0, "boxes": 0, "sources": defaultdict(int)} for s in ("train", "valid", "test")}
    leakage = []
    stem_to_splits: dict[str, set[str]] = defaultdict(set)

    for split in ("train", "valid", "test"):
        img_dir = DEST / split / "images"
        lab_dir = DEST / split / "labels"
        imgs = list(img_dir.glob("*")) if img_dir.exists() else []
        for img in imgs:
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as e:
                val_errors.append(f"{split}/{img.name}: cannot open ({e})")
                continue
            n, errs = parse_yolo(lab)
            if not lab.exists():
                val_errors.append(f"{split}/{img.name}: missing label")
                continue
            if errs:
                val_errors.append(f"{split}/{img.name}: {'; '.join(errs)}")
                continue
            val_ok += 1
            box_total += n
            per_split[split]["images"] += 1
            per_split[split]["boxes"] += n
            stem_to_splits[source_stem(img.name)].add(split)
        for it in copied:
            if it["gold_split"] == split:
                per_split[split]["sources"][it["source"]] += 1

    for stem, splits in stem_to_splits.items():
        if len(splits) > 1:
            leakage.append({"stem": stem, "splits": sorted(splits)})

    manifest = {
        "before": before,
        "kept": len(kept),
        "removed": len(removed),
        "boxes_kept": sum(x["n_boxes"] for x in kept),
        "validation_ok": val_ok,
        "validation_errors": val_errors,
        "leakage": leakage,
        "per_split": {k: {"images": v["images"], "boxes": v["boxes"], "sources": dict(v["sources"])} for k, v in per_split.items()},
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_dedup(before, raw, filename_collisions, stem_multi_file, hash_collisions, removed, kept)
    write_dataset(before, kept, copied, per_split, val_ok, val_errors, leakage, box_total)

    print(json.dumps({
        "before_images": before["n"],
        "before_boxes": before["boxes"],
        "kept_images": len(kept),
        "kept_boxes": sum(x["n_boxes"] for x in kept),
        "removed": len(removed),
        "splits": {k: v["images"] for k, v in per_split.items()},
        "val_ok": val_ok,
        "val_err": len(val_errors),
        "leakage": leakage,
    }, indent=2))


def write_dedup(before, raw, filename_collisions, stem_multi_file, hash_collisions, removed, kept) -> None:
    lines = [
        "# Gold detector dataset — deduplication",
        "",
        "**Mode:** Copy into a new folder only. Original Batch 01, Batch 02 CLEAN, GOOD folders, ICDAS, and models were not modified.",
        "",
        "## Inputs (before unique filter)",
        "",
        "| Source | Images | Boxes |",
        "|--------|-------:|------:|",
        f"| Batch 01 human GT | {before['by_source']['batch01']['images']} | {before['by_source']['batch01']['boxes']} |",
        f"| Round 1 GOOD | {before['by_source']['round1']['images']} | {before['by_source']['round1']['boxes']} |",
        f"| Round 2 GOOD | {before['by_source']['round2']['images']} | {before['by_source']['round2']['boxes']} |",
        f"| **Total** | **{before['n']}** | **{before['boxes']}** |",
        "",
        "Expected: 60+57+59 = 176 images, 767+1415+1451 = 3633 boxes.",
        "",
        "## Collision scan",
        "",
        f"| Check | Groups with >1 file |",
        f"|-------|--------------------:|",
        f"| Identical filename | {len(filename_collisions)} |",
        f"| Same source stem, different filenames | {len(stem_multi_file)} |",
        f"| Identical image bytes (MD5) | {len(hash_collisions)} |",
        "",
    ]
    if filename_collisions:
        lines.append("### Duplicate filenames")
        for k, vs in filename_collisions.items():
            srcs = ", ".join(f"{x['source']}:{x['name']}" for x in vs)
            lines.append(f"- `{k}` — {srcs}")
        lines.append("")
    else:
        lines.append("No identical filenames across sources.")
        lines.append("")
    if stem_multi_file:
        lines.append("### Duplicate source stems")
        for k, vs in list(stem_multi_file.items())[:50]:
            srcs = ", ".join(f"{x['source']}/{x['name']}" for x in vs)
            lines.append(f"- `{k}` — {srcs}")
        lines.append("")
    else:
        lines.append("No duplicate source stems (including Roboflow `.rf.` prefix) across the three GOOD/GT sets.")
        lines.append("")
    if hash_collisions:
        lines.append("### Identical bytes")
        for k, vs in hash_collisions.items():
            srcs = ", ".join(f"{x['source']}/{x['name']}" for x in vs)
            lines.append(f"- `{k[:12]}…` — {srcs}")
        lines.append("")
    else:
        lines.append("No identical image-byte duplicates.")
        lines.append("")
    lines += [
        "## Removed from Gold (originals kept)",
        "",
        f"Dropped **{len(removed)}** candidate copies. Kept **{len(kept)}** unique images.",
        "",
    ]
    if not removed:
        lines.append("Nothing was dropped. All 176 candidates were unique by filename, source stem, and MD5.")
        lines.append("")
    else:
        lines.append("| Gold filename candidate | Source | Reasons |")
        lines.append("|-------------------------|--------|---------|")
        for it in removed:
            lines.append(f"| `{it['name']}` | {it['source']} | {', '.join(it['drop_reasons'])} |")
        lines.append("")
    lines += [
        "## Final unique image count",
        "",
        f"**{len(kept)}** images, **{sum(x['n_boxes'] for x in kept)}** tooth boxes in `data/detection/gold_detector_dataset/`.",
        "",
    ]
    DEDUP_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset(before, kept, copied, per_split, val_ok, val_errors, leakage, box_total) -> None:
    n = len(kept)
    lines = [
        "# Gold detector dataset",
        "",
        "**Mode:** Dataset build + validation. **No training.** Existing datasets and models were not overwritten.",
        "",
        "Location: `data/detection/gold_detector_dataset/`",
        "",
        "## Contents",
        "",
        "| Source | Role | Images | Boxes |",
        "|--------|------|-------:|------:|",
        f"| Batch 01 `tooth_detector_batch01` | human GT | {before['by_source']['batch01']['images']} | {before['by_source']['batch01']['boxes']} |",
        f"| `batch02_manual_good` | Round 1 GOOD | {before['by_source']['round1']['images']} | {before['by_source']['round1']['boxes']} |",
        f"| `batch02_manual_round2/good` | Round 2 GOOD | {before['by_source']['round2']['images']} | {before['by_source']['round2']['boxes']} |",
        f"| Gold after dedup | copy | {n} | {sum(x['n_boxes'] for x in kept)} |",
        "",
        "Excluded: QUESTIONABLE, BAD, unreviewed KEEP, detector predictions, ICDAS.",
        "",
        "## Split (no source-stem leakage)",
        "",
        "Seed 42. Batch 01 original test files stay in **test**; Batch 01 original val files stay in **valid**. Remaining unique stems were assigned to fill ~15% test, ~15% valid, rest train.",
        "",
        "| Split | Images | % | Boxes | Batch 01 | R1 | R2 |",
        "|-------|-------:|--:|------:|---------:|---:|---:|",
    ]
    for sp in ("train", "valid", "test"):
        d = per_split[sp]
        pct = 100.0 * d["images"] / n if n else 0
        lines.append(
            f"| {sp} | {d['images']} | {pct:.1f} | {d['boxes']} | "
            f"{d['sources'].get('batch01', 0)} | {d['sources'].get('round1', 0)} | {d['sources'].get('round2', 0)} |"
        )
    lines += [
        "",
        f"Stem leakage across splits: **{len(leakage)}** (must be 0).",
        "",
        "## Validation",
        "",
        f"- Images that open + valid YOLO `0 xc yc w h` (class tooth, w/h>0, normalized): **{val_ok}/{n}**",
        f"- Validation errors: **{len(val_errors)}**",
        f"- Tooth boxes in Gold: **{box_total}**",
        "",
    ]
    if val_errors:
        lines.append("### Errors")
        for e in val_errors[:50]:
            lines.append(f"- {e}")
        lines.append("")
    lines += [
        "## data.yaml",
        "",
        "`nc: 1`, `names: [tooth]`, splits `train/images`, `valid/images`, `test/images`.",
        "",
        "## Next",
        "",
        "Do **not** train until you ask. Do **not** overwrite `models/detection/tooth_detector_batch01/`.",
        "",
    ]
    DATASET_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
