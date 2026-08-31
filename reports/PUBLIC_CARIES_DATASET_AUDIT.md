# Public caries dataset audit

Date: 2026-08-26

**DATASET_STATUS = ALREADY_DOWNLOADED** — no second download.

Source: Zenodo [10.5281/zenodo.14827784](https://zenodo.org/records/14827784)  
On disk: `data_external/detection/` (Stage 2C).  
Purpose: **decay-region localization only**. Classes **`d` / `D` are not ICDAS**.

---

## Original dump (Stage 2C + `inspection_summary.json`)

| Check | Exact number |
| --- | --- |
| Images | **6265** JPG (paper claimed 6313; 48 missing from `Dataset/Images`) |
| Format | JPEG, PIL mode RGB |
| RGB / grayscale / X-ray-like | **6265 / 0 / 0** |
| Corrupted | **0** |
| Width | 617–4080 |
| Height | 347–3206 |
| Duplicate stems / content groups | **0 / 0** |
| YOLO `.txt` | **2245** |
| Pascal VOC `.xml` | **2245** |
| COCO JSON | **3** |
| LabelMe | not extracted |
| Images with boxes | **2227** |
| Images without annotations | **4038** |
| Label files missing image | **36** (18 YOLO + 18 VOC stems) |
| Empty label files | **126** |
| Invalid YOLO boxes (outside [0,1] in Stage 2C) | **84** |
| Box count (primary) | **6728** |
| VOC class counts | **D = 6174**, **d = 554** |
| YOLO ids | **0 = D**, **1 = d** (verified vs VOC) |

Annotation formats available: **YOLO (working)**, VOC, COCO (train/valid vs **test.json id inversion** — do not use COCO ids blindly).

---

## Isolated derived set (`data_external/detection/public_caries/`)

Original `raw/` and `annotations/` **not overwritten**. Format chosen: **YOLO**, matching existing Darknet files.

| Split | Images |
| --- | --- |
| train | **1514** |
| val | **357** |
| test | **277** |
| total used | **2148** |

Boxes after clipping/drop: **D = 6134**, **d = 552**.  
Skipped empty/all-invalid labels: **79**. Invalid lines dropped: **42**. Labels missing image: **18**.

Split: MD5 of filename, seed 42, 70/15/15.

---

## Correspondence

Paired by relative path: `raw/<same>/<stem>.jpg` ↔ `annotations/yolo/<same>/<stem>.txt`.

Unlabeled 4038 images were **not** used as ICDAS 0 negatives.
