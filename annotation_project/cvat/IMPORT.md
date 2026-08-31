# CVAT import (no Docker required in this repo)

This folder is **import configuration**, not a running CVAT server.

You can use [CVAT online](https://app.cvat.ai/) or any existing CVAT install. This project does **not** require Docker in the ICCC repository.

## Project

- Name: `iccc_whole_tooth_detection`
- Labels: **one** rectangle class `tooth` (see `labels.json`)
- Images: `fdi_detection_dataset/images/selected/` (420 files)
- **Do not** import Zenodo `d`/`D` lesion XML/YOLO/COCO

## Import steps (CVAT UI)

1. Create a project named `iccc_whole_tooth_detection`.
2. Add label `tooth` of type **Rectangle** (paste `labels.json` if the UI allows JSON labels).
3. Create a task; attach images from `fdi_detection_dataset/images/selected/` (or upload a zip of those 420 files).
4. Leave annotations empty. Boxes are **HUMAN-GENERATED** (pretrained search closed).
5. **Start with Batch_01 only** (60 files): `annotation_batches/Batch_01/cvat_upload_filenames.txt`. Full steps: `STAGE3C_MANUAL_ANNOTATION.md`.
6. Then Batch_02 … Batch_07 as separate tasks.

## Export later (after humans annotate)

- COCO 1.0 → `fdi_detection_dataset/annotations/coco/`
- YOLO 1.1 → `fdi_detection_dataset/annotations/yolo/`
- Pascal VOC 1.1 → `fdi_detection_dataset/annotations/pascal_voc/`

Use `tools/coco_to_yolo.py` / `tools/yolo_to_coco.py` / `tools/voc_to_yolo.py` if you need format conversion. They do not invent boxes.

## Patient identifiers

Use only values already in filenames / manifests. Do not invent IDs.
