#!/usr/bin/env python3
"""
Download and prepare public dental caries / oral imaging datasets.

Supported sources (manual download links — many require registration):
  - Mendeley Dental Caries datasets
  - OralScan / ToothSeg public releases
  - Weak supervision mode when ICDAS labels unavailable

Usage:
  python tools/ingest/download_datasets.py --dataset dental_caries --output data/icdas/raw
"""

import argparse
import os
import sys
from pathlib import Path

DATASET_SOURCES = {
    "dental_caries": {
        "name": "Dental Caries Detection (Mendeley)",
        "url": "https://data.mendeley.com/datasets/5vb5tvkjb5/1",
        "notes": "Download manually, extract to data/icdas/raw/dental_caries/",
    },
    "oral_images": {
        "name": "Oral Disease Image Dataset",
        "url": "https://www.kaggle.com/datasets/oral-disease",
        "notes": "Requires Kaggle API: kaggle datasets download -d <name>",
    },
    "icdas_weak": {
        "name": "Weak Supervision Mode",
        "url": None,
        "notes": "Use unlabeled images + python ml/src/advanced.py pseudo-labeling",
    },
}

ANNOTATION_FORMAT = """
# annotations.csv format
filename,icdas_score,split,patient_id,notes
train/0/image001.jpg,0,train,P001,sound
train/3/image002.jpg,3,train,P002,enamel breakdown
"""


def setup_raw_structure(output: Path):
    for name in DATASET_SOURCES:
        (output / name).mkdir(parents=True, exist_ok=True)
    readme = output / "README.md"
    readme.write_text(
        "# Raw Datasets\n\n"
        + "\n".join(
            f"## {info['name']}\n- URL: {info.get('url', 'N/A')}\n- {info['notes']}\n"
            for info in DATASET_SOURCES.values()
        )
        + f"\n## Annotation Format\n```csv\n{ANNOTATION_FORMAT}\n```\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_SOURCES.keys), default="dental_caries")
    parser.add_argument("--output", default="data/icdas/raw")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_raw_structure(output)

    info = DATASET_SOURCES[args.dataset]
    print(f"\n{'='*60}")
    print(f"Dataset: {info['name']}")
    print(f"Output:  {output / args.dataset}")
    if info.get("url"):
        print(f"URL:     {info['url']}")
    print(f"Notes:   {info['notes']}")
    print(f"{'='*60}\n")
    print("After download, organize into:")
    print("  data/icdas/train/0/ ... data/icdas/train/4/  (ICDAS 5–6 go to data/icdas/excluded/)")
    print("Or update data/icdas/annotations/annotations.csv")
    print("\nThen run preprocessing:")
    print("  python tools/ingest/preprocess_dataset.py")


if __name__ == "__main__":
    main()
