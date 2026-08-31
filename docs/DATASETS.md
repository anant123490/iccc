# Datasets

Full-mouth **detection** data and **ICDAS** tooth images are separate. See `docs/DATASET_WORKFLOW.md`.

## Detection (full RGB + tooth boxes)

| Asset | Path |
|-------|------|
| 420 originals | `fdi_detection_dataset/images/selected/` |
| Batch 01 YOLO | `fdi_detection_dataset/tooth_detector_batch01/` |
| New photos | `data/detection/raw_images/` |
| New boxes | `data/detection/annotations/` |

FDI numbering is not used. Folder name `fdi_detection_dataset` is historical.

## ICDAS (single tooth + grade 0–4)

```text
data/icdas/
├── train/0 .. 4/
├── val/0 .. 4/
├── test/0 .. 4/
├── excluded/5 and excluded/6
├── raw/
├── images/
└── annotations/annotations.csv
```

`annotations.csv` currently has ~643 rows; **image pixels are missing**. Restore files into the class folders; do not invent labels.

ICDAS 5 and 6 are never remapped to 4.

## Generated crops

`data/tooth_crops/generated/` — detector output, not ground truth.

## Public downloads

```bash
python tools/ingest/download_datasets.py --dataset dental_caries
```

Lesion `d`/`D` public boxes are **not** ICDAS and **not** whole-tooth GT.
