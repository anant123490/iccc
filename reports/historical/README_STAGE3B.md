# Stage 3B — Annotation project (whole-tooth detection)

Infrastructure only. **No tooth boxes drawn by software.** Stage 3C annotations (when you draw them) are **HUMAN-GENERATED**, not model-generated. **No FDI. No training. ICDAS `dataset/` unchanged.**

Start Batch_01: see `STAGE3C_MANUAL_ANNOTATION.md`. Progress: `annotation_project/annotation_progress.csv`.

## Dataset source

Selected RGB copies from Zenodo **10.5281/zenodo.14827784** (CC BY 4.0), curated in Stage 3A:

- `fdi_detection_dataset/images/selected/` — **420** images  
- `fdi_detection_dataset/images/review/` — **80** images (not in the 7 annotation batches)

Originals remain in `data_external/detection/raw/`. Lesion XML/`d`/`D` is **not** used.

## Folder structure

```
fdi_detection_dataset/
  images/selected/
  annotations/yolo/          # empty .txt (class 0 = tooth, later)
  annotations/coco/          # placeholder JSON, annotations: []
  annotations/pascal_voc/    # size-only XML
  classes.txt | classes.json | classes.yaml
  dataset.yaml
  splits/train|val|test.*    # manifests only; images not moved
annotation_project/
  cvat/                      # labels + import instructions (no Docker required)
  label_studio/
  manifests/selected_images.{csv,json,txt}
annotation_batches/Batch_01 … Batch_07/
tools/
  visualize_annotations.py
  voc_to_yolo.py
  yolo_to_coco.py
  coco_to_yolo.py
TOOTH_ANNOTATION_GUIDELINES.md
ANNOTATION_QC_CHECKLIST.md
```

## Class

One class: **`tooth`**, id **`0`**. No FDI classes.

## Annotation workflow

1. Read `TOOTH_ANNOTATION_GUIDELINES.md`.  
2. Take a batch from `annotation_batches/Batch_0N/` (CSV list, 60 images).  
3. Label in **CVAT** or **Label Studio** with rectangle `tooth`.  
4. QC with `ANNOTATION_QC_CHECKLIST.md`.  
5. Export COCO/YOLO into `fdi_detection_dataset/annotations/`.  
6. Optional overlays:  
   `python tools/visualize_annotations.py --format yolo --images fdi_detection_dataset/images/selected --labels fdi_detection_dataset/annotations/yolo --out tmp/vis`

## CVAT workflow

See `annotation_project/cvat/IMPORT.md`. Create a project, add rectangle label `tooth`, attach the 420 selected images (or one batch). Do not import lesion annotations. No Docker is required by this repository.

## YOLO workflow

- `fdi_detection_dataset/dataset.yaml` points at `splits/*.txt` and `names: 0: tooth`.  
- Each selected image already has an **empty** `annotations/yolo/<stem>.txt`.  
- After export, fill those files (or replace via `tools/coco_to_yolo.py`).  
- **Do not train** until boxes exist.

## COCO workflow

- `annotations/coco/instances_placeholder.json` lists 420 images; `annotations` is `[]`.  
- After CVAT/Label Studio export, replace or merge with real `annotations`.  
- Convert with `tools/yolo_to_coco.py` / `tools/coco_to_yolo.py` as needed.

## Future FDI stage

FDI is a **later** task (`annotations/fdi_mapping/`). Do not number teeth while drawing detection boxes unless a future protocol explicitly adds a second attribute — Stage 3B does **not**.

## Future training stage

Only after Stage 3C (or later) produces verified `tooth` boxes. Keep ICDAS training data in `dataset/` separate.
