#!/usr/bin/env python3
"""
Download and prepare public dental caries / oral imaging datasets.

Supported sources (manual download links — many require registration):
  - Mendeley Dental Caries datasets
  - OralScan / ToothSeg public releases
  - Weak supervision mode when ICDAS labels unavailable

Usage:
  python scripts/download_datasets.py --dataset dental_caries --output ../dataset/raw
"""

import argparse
import os
import sys
from pathlib import Path

DATASET_SOURCES = {
    "dental_caries": {
        "name": "Dental Caries Detection (Mendeley)",
        "url": "https://data.mendeley.com/datasets/5vb5tvkjb5/1",
        "notes": "Download manually, extract to dataset/raw/dental_caries/",
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
    parser.add_argument("--output", default="../dataset/raw")
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
    print("  dataset/train/0/, dataset/train/1/, ... dataset/train/6/")
    print("Or update dataset/annotations.csv")
    print("\nThen run preprocessing:")
    print("  python scripts/preprocess_dataset.py --input dataset/raw --output dataset")


if __name__ == "__main__":
    main()
