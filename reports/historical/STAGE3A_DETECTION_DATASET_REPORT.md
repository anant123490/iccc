# Stage 3A — RGB whole-tooth detection dataset construction

Date: 2026-08-26  
Scope: **dataset preparation only**. Originals untouched. `dataset/` (ICDAS), `ml/`, backend, frontend, and models **not modified**.  
**Tooth boxes generated: NO. FDI labels generated: NO.**

---

## 1. Source location (verified)

| Item | Path |
| --- | --- |
| Image root | `data_external/detection/raw/` |
| Annotation root | `data_external/detection/annotations/` |
| YOLO (lesion, unused) | `data_external/detection/annotations/yolo/` |
| Pascal VOC (lesion, unused) | `data_external/detection/annotations/pascal-voc/` |
| COCO (lesion, unused) | `data_external/detection/annotations/ms_coco/` |
| Source note | `data_external/detection/SOURCE.txt` |
| New dataset | `fdi_detection_dataset/` |

**Source:** Zenodo **10.5281/zenodo.14827784** — *Annotated intraoral image dataset for dental caries detection* — [https://zenodo.org/records/14827784](https://zenodo.org/records/14827784) — license documented as **CC BY 4.0**.

| Count | Value |
| --- | --- |
| RGB images found | **6,265** (`.jpg`) |
| XML files | **2,245** |

---

## 2. Image verification

Every file under the image root was opened.

| Check | Result |
| --- | --- |
| Readable | **6,265** |
| Unreadable / corrupted | **0** |
| RGB | **6,265** |
| Grayscale | **0** |
| Extensions | `.jpg` only |
| Width × height (valid) | mean **1635.67 × 1177.65** px (range from Stage 2C: 617–4080 × 347–3206) |

No originals were deleted. None met REJECT gates, so `images/rejected/` has **0** copies.

---

## 3. Quality analysis

Per image (256 px thumbnail): Laplacian variance (sharpness), mean luminance (brightness), std (contrast), `1/(sharpness+ε)` (blur), exposure band from brightness.

| Quality | Count |
| --- | --- |
| HIGH QUALITY | 3,921 |
| MEDIUM QUALITY | 2,337 |
| LOW QUALITY | 7 |
| REJECT | 0 |

LOW QUALITY frames were **not** rejected; they were prioritized for **review**.

---

## 4. Duplicates

| Method | Result |
| --- | --- |
| Filename collisions (same basename, different folders) | **0** |
| MD5 extra copies | **0** |
| Rows with non-`unique` status | **238** |
| Extra perceptual (identical 8×8 aHash, not canonical) | **125** |

**Exact pixel duplicates were not found.** Perceptual groups are **same aHash** (similar composition). Canonical members were kept eligible for selection; extra aHash matches were not copied into `selected/`. Originals were not deleted.

CSV: `reports/stage3a_duplicate_report.csv`

---

## 5. Selection (420 images)

Target **300–500**. Built **420** with stratified quotas:

- **84** per view folder: Frontal, Left_Lateral, Mandibular, Maxillary_Occlusal, Right_Lateral  
- Within each view: **37** retractors + **37** no_retractors + **10** pilot  
- Prefer unused parsed `anonymous_NNN-NNN-N` identifiers  
- Prefer HIGH QUALITY; skip non-canonical perceptual extras  

| Selected slice | Count |
| --- | --- |
| retractors / no_retractors / pilot | 185 / 185 / 50 |
| Parsed patient-like IDs in selected | **370** unique (empty string if pattern absent — **not invented**) |
| Selected quality | HIGH 410, MEDIUM 10 |

Copies live in `fdi_detection_dataset/images/selected/` with **original filenames**.

---

## 6. Dataset structure created

```
fdi_detection_dataset/
  images/selected/          420 JPG copies
  images/review/            80 JPG copies
  images/rejected/          0 copies (see README there)
  annotations/pascal_voc/   420 placeholder XML (no objects)
  annotations/coco/instances_placeholder.json
  annotations/yolo/         420 empty .txt
  annotations/fdi_mapping/  template only
  metadata/
  reports/
  README.md
```

---

## 7. Manifest

Full inventory: `fdi_detection_dataset/metadata/image_manifest.csv`  
Selected: `reports/stage3a_selected_images.csv`

Columns: `original_filename`, `current_filename`, `source_dataset`, `source_path`, `width`, `height`, `channels`, `orientation`, `quality`, `duplicate_status`, `selection_status`, `annotation_exists`, `annotation_type`, `patient_identifier_if_available`.

`annotation_type` is `lesion_d_D` or `none` — **never** tooth or FDI.

---

## 8. Lesion XML inventory (unused)

Recount of original Pascal VOC (not copied as tooth labels):

| Item | Value |
| --- | --- |
| XML files | 2,245 |
| Parse errors | 0 |
| Object count | **6,782** |
| Files with ≥1 object | 2,180 |
| Empty XML (0 objects) | 65 |
| Unique labels | **`D` = 6,228**, **`d` = 554** |
| Mean objects / XML | 3.02 |
| Images without a matching lesion file | 6,265 − matched stems (lesion labels cover a subset; Stage 2C: 2,227 images with COCO boxes) |
| Tooth annotations | **0** |
| FDI annotations | **0** |

Labels mean **decay lesions**, not whole teeth. They were **not** modified and **not** converted.

---

## 9. Tooth visibility (estimate, not FDI)

From **view folder**, not from counting teeth on every pixel:

| View | Visibility class | ESTIMATE teeth / image |
| --- | --- | --- |
| Maxillary_Occlusal / Mandibular | full_arch (single jaw) | 14 |
| Frontal | anterior / partial arch | 12 |
| Left/Right lateral | posterior partial | 8 |

Selected-set mean of those estimates: **11.2** (method: view-type constants only). **Not FDI. Not per-image counts.**

---

## 10. Orientation

From folder names (filenames unchanged):

| Category | All 6,265 | Selected 420 |
| --- | --- | --- |
| Frontal | 1,259 | 84 |
| Left Buccal (`Left_Lateral`) | 1,249 | 84 |
| Right Buccal (`Right_Lateral`) | 1,252 | 84 |
| Mandibular (also occlusal) | 1,251 | 84 |
| Maxillary (Maxillary_Occlusal) | 1,254 | 84 |
| Unknown | 0 | 0 |

CSV: `reports/stage3a_orientation_report.csv`

---

## 11. Review set (80)

Copied to `fdi_detection_dataset/images/review/`. Reasons include low sharpness, laterals (overlap), missing anonymous-id pattern, low resolution, no retractors, unusual aspect ratio. See `metadata/review_set.csv` column `review_reason`.

All **7** LOW QUALITY source frames were placed in review (none rejected).

---

## 12. Reject set

**0** images copied. No file was unreadable, empty of teeth by automated REJECT gates, or severely over/underexposed at the REJECT thresholds.

---

## 13. Detection dataset statistics

| Metric | Value |
| --- | --- |
| Total RGB images | 6,265 |
| Valid RGB images | 6,265 |
| Selected images | 420 |
| Review images | 80 |
| Rejected images | 0 |
| Duplicate extra MD5 copies | 0 |
| Unique parsed IDs (all / selected) | 889 / 370 |
| Average width / height | 1635.67 / 1177.65 |
| Average teeth visibility estimate (selected) | 11.2 (ESTIMATE) |

---

## 14. COCO placeholder

`fdi_detection_dataset/annotations/coco/instances_placeholder.json`

- `images`: **420** registered  
- `annotations`: **[]**  
- `categories`: placeholder `placeholder_whole_tooth` only  

**No tooth boxes.**

---

## 15. YOLO placeholder

`fdi_detection_dataset/annotations/yolo/*.txt` — **420** files, **all empty**. **No bounding boxes.**

Pascal VOC placeholders exist with `<size>` only and **no** `<object>` elements.

---

## 16. Future FDI directory

`fdi_detection_dataset/annotations/fdi_mapping/README.md` — template table only. **No mappings.**

---

## 17–18. Documentation and reports

- `fdi_detection_dataset/README.md`  
- `reports/stage3a_detection_dataset_inventory.json`  
- `STAGE3A_DETECTION_DATASET_REPORT.md` (this file)  
- `reports/stage3a_selected_images.csv`  
- `reports/stage3a_duplicate_report.csv`  
- `reports/stage3a_quality_report.csv`  
- `reports/stage3a_orientation_report.csv`  

---

## Final summary table

| Metric | Value |
| --- | --- |
| Total RGB images | 6,265 |
| Valid RGB images | 6,265 |
| Selected images | 420 |
| Review images | 80 |
| Rejected images | 0 |
| Duplicate images (MD5 extras) | 0 |
| XML annotations | 2,245 files / 6,782 lesion objects |
| Lesion annotations | 6,782 (`D` 6,228 + `d` 554) |
| Tooth annotations | **0** |
| FDI annotations | **0** |
| COCO placeholder created | **YES** |
| YOLO placeholder created | **YES** |

---

## Final decision

1. **How many RGB images are ready for tooth detection annotation?** **420** in `images/selected/` (plus 80 in review if a human accepts them later).  
2. **How many images were rejected?** **0**.  
3. **Are the selected images sufficient for a whole-tooth detection dataset?** **YES** as a **starter annotation set** (balanced views, many distinct parsed IDs, high quality). They are **not** yet labelled with tooth boxes.  
4. **Is any ICDAS data modified?** **NO.**  
5. **Were any FDI labels generated?** **NO.**  
6. **Were any tooth bounding boxes generated?** **NO.**  
7. **Recommended next stage:** **Stage 3B — Whole-Tooth Bounding Box Annotation Preparation**

---

**STOP.** Do not annotate, train, assign FDI, or change ICDAS resources in this stage.
