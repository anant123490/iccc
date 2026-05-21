#!/usr/bin/env python3
"""
Import WhatsApp clinical intraoral images into the ICDAS dataset.

Labels are assigned by most-severe visible lesion (image-level).
Review and correct labels with a licensed clinician before production training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Source: workspace assets/ (junction to Cursor-uploaded images)
ASSETS_DIR = PROJECT_ROOT / "assets"
CURSOR_ASSETS = Path(
    r"C:\Users\anant\.cursor\projects\c-Users-anant-OneDrive-Desktop-icdas-project\assets"
)
DATASET_ROOT = PROJECT_ROOT / "dataset"

# uuid suffix -> (icdas_score, notes)
# Based on visual assessment of provided clinical photos
IMAGE_LABELS: dict[str, tuple[int, str]] = {
    "f907cab9-cc86-40bd-87ca-e6fef6029031": (
        6,
        "Extensive occlusal cavitation, lower left molars",
    ),
    "4f99e4ef-edb6-44c5-955d-344e99f57689": (
        5,
        "Distinct cavity via mirror reflection, lower right",
    ),
    "434b7bb5-7304-41c0-9bfc-2b9dedc98825": (
        5,
        "Mirror view distinct occlusal cavity",
    ),
    "f4ef9d42-a356-4804-a8db-bd444d138562": (
        5,
        "Deep occlusal cavitation lower molar",
    ),
    "d6c6ca13-f4d5-4d50-ad9f-c75f1db62c85": (
        5,
        "Buccal-distal cavity with dentin visible",
    ),
    "bf5441fd-34cc-4ca8-a4d9-2e7c9b9fa108": (
        6,
        "Extensive posterior occlusal decay",
    ),
    "ce89b518-7752-4947-8e82-6f4aee199822": (
        5,
        "Dominant molar ICDAS 5, adjacent staining",
    ),
    "4c8ed055-0505-4a40-a5a1-9db08ba03102": (
        5,
        "Multi-tooth; worst lesion distinct cavity",
    ),
    "0cfeb970-adce-4f5f-a0a9-5c1346854a2d": (
        5,
        "Lower right occlusal distinct cavity",
    ),
    "adfec5d3-c421-4a5c-95d5-b1cd053217f6": (
        5,
        "Mirror reflection distinct cavity",
    ),
    "0f7720b8-47e8-4e05-83a8-a9352df3f014": (
        5,
        "Lower right multi-tooth, severe posterior",
    ),
    "b185d7e5-b13b-4e8b-86a7-84e747d928c8": (
        5,
        "Occlusal view with instrument, distinct cavities",
    ),
    "d757e9ab-e17d-4538-8b9f-9c7f88b341db": (
        5,
        "Lower quadrant occlusal caries",
    ),
    "25a3ec3a-ca66-45d5-894a-5a50ef6bcb22": (
        5,
        "Rubber dam interproximal distinct cavity",
    ),
    "5cf66728-f797-4f54-ad82-94614135d907": (
        5,
        "Anterior proximal distinct cavity, rubber dam",
    ),
}

# Stratified split: 10 train, 3 val, 2 test
SPLITS: dict[str, list[str]] = {
    "train": [
        "f907cab9-cc86-40bd-87ca-e6fef6029031",
        "4f99e4ef-edb6-44c5-955d-344e99f57689",
        "434b7bb5-7304-41c0-9bfc-2b9dedc98825",
        "f4ef9d42-a356-4804-a8db-bd444d138562",
        "d6c6ca13-f4d5-4d50-ad9f-c75f1db62c85",
        "bf5441fd-34cc-4ca8-a4d9-2e7c9b9fa108",
        "ce89b518-7752-4947-8e82-6f4aee199822",
        "4c8ed055-0505-4a40-a5a1-9db08ba03102",
        "0cfeb970-adce-4f5f-a0a9-5c1346854a2d",
        "adfec5d3-c421-4a5c-95d5-b1cd053217f6",
    ],
    "val": [
        "0f7720b8-47e8-4e05-83a8-a9352df3f014",
        "b185d7e5-b13b-4e8b-86a7-84e747d928c8",
        "25a3ec3a-ca66-45d5-894a-5a50ef6bcb22",
    ],
    "test": [
        "d757e9ab-e17d-4538-8b9f-9c7f88b341db",
        "5cf66728-f797-4f54-ad82-94614135d907",
    ],
}


def find_asset_file(uuid_key: str) -> Path | None:
    """Find asset file matching uuid suffix."""
    for f in ASSETS_DIR.glob("*.png"):
        if uuid_key in f.name:
            return f
    return None


def resolve_assets_dir() -> Path:
    """Prefer workspace assets/ junction; fall back to Cursor project assets."""
    if ASSETS_DIR.exists() and any(ASSETS_DIR.glob("*.png")):
        return ASSETS_DIR
    if CURSOR_ASSETS.exists():
        return CURSOR_ASSETS
    raise FileNotFoundError(
        f"No images found. Copy PNG files to: {ASSETS_DIR}\n"
        f"Or run: cmd /c mklink /J \"{ASSETS_DIR}\" \"{CURSOR_ASSETS}\""
    )


def main():
    global ASSETS_DIR
    ASSETS_DIR = resolve_assets_dir()
    print(f"Using assets from: {ASSETS_DIR}")

    raw_dir = DATASET_ROOT / "raw" / "whatsapp_clinical"
    raw_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    manifest = []
    idx = 0

    for split, uuid_list in SPLITS.items():
        for uuid_key in uuid_list:
            src = find_asset_file(uuid_key)
            if src is None:
                print(f"WARNING: Missing image for {uuid_key}")
                continue

            score, notes = IMAGE_LABELS[uuid_key]
            idx += 1
            short_name = f"whatsapp_{idx:03d}.png"

            # Raw archive (original copy)
            shutil.copy2(src, raw_dir / short_name)

            # Organized by split/class
            dest_dir = DATASET_ROOT / split / str(score)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / short_name
            shutil.copy2(src, dest_path)

            rel_path = f"{split}/{score}/{short_name}"
            annotations.append({
                "filename": rel_path,
                "icdas_score": score,
                "split": split,
                "patient_id": "whatsapp_clinical",
                "notes": notes,
                "source": "whatsapp_2026-05-20",
                "original_asset": src.name,
            })
            manifest.append({
                "short_name": short_name,
                "uuid": uuid_key,
                "split": split,
                "icdas_score": score,
            })
            print(f"  [{split}] ICDAS {score} <- {short_name}")

    # Write annotations.csv
    import pandas as pd

    df = pd.DataFrame(annotations)
    csv_path = DATASET_ROOT / "annotations.csv"
    df.to_csv(csv_path, index=False)

    manifest_path = DATASET_ROOT / "whatsapp_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nImported {len(annotations)} images")
    print(f"  train: {len(SPLITS['train'])}")
    print(f"  val:   {len(SPLITS['val'])}")
    print(f"  test:  {len(SPLITS['test'])}")
    print(f"  annotations: {csv_path}")
    print(f"  manifest:    {manifest_path}")
    print("\nNote: Labels are preliminary — verify with a dental expert before training.")


if __name__ == "__main__":
    main()
