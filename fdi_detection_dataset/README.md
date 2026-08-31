# Whole-tooth detection dataset

> **Do not rename this folder yet.** Cropping, YOLO `data.yaml`, and annotation JSON still use the path `fdi_detection_dataset/`. The name is historical. This is **not** an FDI labeling project. FDI numbering is out of scope. `annotations/fdi_mapping/` was moved to `archive/out_of_scope/fdi/fdi_mapping/`.

## Purpose

Curated RGB intraoral photographs plus **Batch 01** whole-tooth boxes.

This folder is **not** the ICDAS dataset.

Active pipeline:

RGB intraoral photograph → whole-tooth detection → tooth crop → ICDAS 0–4 → report

## What is here

- `images/selected/` — **420** original RGB JPGs
- `tooth_detector_batch01/` — 46/6/8 split, **767** verified `tooth` boxes
- Placeholder YOLO/VOC/COCO under `annotations/` (mostly empty; real Batch 01 labels live in `tooth_detector_batch01/labels/`)

New detection photos should go to `data/detection/raw_images/` so Batch 01 is not overwritten. See `data/detection/README.md`.


## Source

Zenodo **10.5281/zenodo.14827784**  
*Annotated intraoral image dataset for dental caries detection*  
https://zenodo.org/records/14827784  
License (documented in-repo): **CC BY 4.0**

Originals remain in `data_external/detection/raw/`. This tree contains **copies** of selected/review images only.

## Why lesion annotations are unused

Original Pascal VOC / YOLO / COCO labels use classes **`d`** and **`D`** (primary vs permanent **decay lesions**).

They are **not** whole-tooth boxes and **must not** be converted into tooth or FDI labels.

Placeholder files in `annotations/` have **empty** object lists / empty YOLO files.

## What is not this dataset

- No FDI numbers (out of scope; mapping stub archived)
- No ICDAS scores on these full-mouth copies
- Lesion d/D placeholders under `annotations/` are not tooth GT

## Folder structure

```
fdi_detection_dataset/
├── images/selected/              # 420 RGB photographs
├── tooth_detector_batch01/       # Batch 01 YOLO split + 767 boxes
├── annotations/                  # mostly placeholders
├── metadata/
└── reports/
```

## Future workflow

Add **new** photographs under `data/detection/raw_images/`. Do not merge this tree into `data/icdas/`.
