# Stage 3B — Whole-tooth annotation project preparation

Date: 2026-08-26  
Scope: **annotation infrastructure only**. No boxes drawn, no FDI, no training, ICDAS `dataset/` not modified.

---

## Stage 3A verification

| Path | Status |
| --- | --- |
| `fdi_detection_dataset/images/selected` | **420** JPG (expected 420; **no mismatch**) |
| `fdi_detection_dataset/images/review` | **80** JPG |
| `fdi_detection_dataset/annotations/coco` | placeholder JSON present |
| `fdi_detection_dataset/annotations/yolo` | **420** empty `.txt` files |

---

## Class configuration

Exactly one detection class:

| Field | Value |
| --- | --- |
| Name | `tooth` |
| ID | `0` |
| FDI classes | **0** |

Files: `fdi_detection_dataset/classes.{txt,json,yaml}` (copied under `annotation_project/` as well).

---

## CVAT

`annotation_project/cvat/` contains `project.json`, `labels.json`, image/dataset manifests, and `IMPORT.md`.

**CVAT-ready: YES.** Import images from `images/selected/` (or per-batch CSV). **Docker is not required** by this repository. Do not import Zenodo lesion labels.

---

## Label Studio

`annotation_project/label_studio/` contains `labeling_config.xml` (single `RectangleLabels` value `tooth`), `tasks.json`, `image_list.json`, and README.

**Label Studio-ready: YES.**

---

## Manifests and batches

- `annotation_project/manifests/selected_images.{csv,json,txt}` — **420** rows  
- `annotation_batches/Batch_01` … `Batch_07` — **60** images each, manifests only (images **not** copied)  
- Each batch: 12 Frontal, 12 Left_Lateral, 12 Mandibular, 12 Maxillary_Occlusal, 12 Right_Lateral  

---

## Splits (manifests only)

Patient-safe greedy assignment, 70% / 15% / 15% of **420** images. Images were **not** moved.

| Split | Images |
| --- | --- |
| Train | 294 |
| Val | 63 |
| Test | 63 |

Parsed-ID overlap between splits: **0**. Files: `fdi_detection_dataset/splits/`.

`dataset.yaml` points at these lists and `nc: 1` / `tooth`.

---

## Tools

| Script | Role |
| --- | --- |
| `tools/visualize_annotations.py` | Overlay COCO / YOLO / VOC; empty labels → image only |
| `tools/voc_to_yolo.py` | Convert existing VOC |
| `tools/yolo_to_coco.py` | Convert existing YOLO |
| `tools/coco_to_yolo.py` | Convert existing COCO |

None of these generate boxes.

Guidelines: `TOOTH_ANNOTATION_GUIDELINES.md`  
QC: `ANNOTATION_QC_CHECKLIST.md`  
How-to: `README_STAGE3B.md`

---

## Final summary table

| Metric | Value |
| --- | --- |
| Selected images | 420 |
| Review images | 80 |
| Annotation batches | 7 |
| Images per batch | 60 |
| Detection classes | 1 (`tooth`) |
| FDI classes | 0 |
| Tooth boxes generated | **NO** |
| ICDAS modified | **NO** |
| Models trained | **NO** |
| CVAT readiness | YES |
| Label Studio readiness | YES |
| COCO readiness | YES (placeholder + converters) |
| YOLO readiness | YES (empty labels + `dataset.yaml`) |

---

## Final decision

1. **Ready for manual tooth annotation in CVAT?** **YES** (project/labels/manifests; human drawing is Stage 3C).  
2. **YOLO dataset structure ready?** **YES** (empty labels, class 0, splits, `dataset.yaml`).  
3. **COCO dataset structure ready?** **YES** (images registered, `annotations: []`).  
4. **FDI labels intentionally absent?** **YES**.  
5. **Next stage:** **Stage 3C — Whole-Tooth Bounding Box Annotation**

---

**STOP.** Do not draw boxes, assign FDI, train models, or change ICDAS in this stage.
