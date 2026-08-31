# Stage 2D-3 — Targeted RGB intraoral + whole-tooth + FDI search

Date: 2026-08-26  
Scope: search and original-source verification only. No large downloads, no API keys, no training, no application or `dataset/` / Zenodo / `.keras` changes.

> FDI numbering and tooth detection answer **which tooth**. ICDAS answers **caries severity**. Never map FDI, tooth number, caries, `d`/`D`, Black classes, or healthy/diseased to ICDAS 0–4.

> Zenodo 14827784 remains **lesion detection only**. It is not whole-tooth FDI data.

---

## 1. Search objective

Find a **real** public dataset for:

RGB intraoral photograph → whole-tooth detection → FDI number → (later) tooth crop → ICDAS 0–4.

Stage 2D-2 rejected Roboflow Prime (`prime-snf1v/teeth-detection-and-numbering-agi2i`) because samples are **panoramic X-rays**.

## 2. ICCC requirement

All of: **RGB intraoral photos**, **whole-tooth** boxes/masks (not lesion boxes), **FDI/WDF two-digit labels**, **verifiable source**, **documented license/access**.

Preferred: full-mouth views, many teeth per image, up to 32 permanent FDI classes, YOLO/COCO/VOC/CSV, open download.

## 3. Search sources

Targeted queries on Hugging Face, Kaggle, Roboflow Universe, Zenodo, Figshare (search), Mendeley Data, GitHub + papers, PhysioNet, and journal data statements. Titles were not trusted; panoramic/X-ray/CBCT/mesh/synthetic sets were rejected on modality.

## 4. Candidate table

| Rank | Name | RGB intraoral | Whole-tooth | FDI | Access | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **FDTooth** | VERIFIED (source) | VERIFIED (1800 boxes on photos) | VERIFIED (CSV two-digit) | Credentialed PhysioNet | Anterior only; 241 JPEGs |
| 2 | SegmentAnyTooth images | VERIFIED (paper) | VERIFIED (paper YOLO) | VERIFIED (paper) | Images **not public** | 5000 photos; MIT code only |
| 3 | DentalMate6v Intraoral Tooth Numbering FDI | UNVERIFIED | UNVERIFIED | listing 32 classes | Roboflow; Cloudflare | Must inspect samples |
| 4 | DigiLeap / TLNM | VERIFIED (paper) | VERIFIED (paper) | VERIFIED (paper) | Findata restricted | 1272 images |
| 5 | Yoon et al. 24,578 | VERIFIED (paper) | VERIFIED (paper) | tooth numbers (paper) | **Not public** | Caries stages ≠ ICDAS |

## 5. Evidence (serious candidates)

### Rank 1 — FDTooth (best available)

- **Source:** [PhysioNet fdtooth 1.0.0](https://physionet.org/content/fdtooth/1.0.0/), DOI [10.13026/v9xk-dy61](https://doi.org/10.13026/v9xk-dy61). Paper: Scientific Data [10.1038/s41597-025-05348-3](https://doi.org/10.1038/s41597-025-05348-3).
- **Modality:** Intraoral **JPEG** 5760×3840; paper Fig. 3(a) is an intraoral photograph; Fig. 5 shows original intraoral images with rectangular labels. Paired **CBCT** exists — **do not** use CBCT for the camera classifier.
- **Boxes:** 1,800 boxes on photographs (MakeSense JSON). Paper: boxes around anterior teeth, not caries lesions. 1,800 < 241×12, so **every-tooth coverage is UNVERIFIED** without opening files.
- **FDI:** CSV uses **FDI World Dental Federation** two-digit numbers for **12 anterior teeth** (incisors and canines, both arches). Exact exported list (e.g. 13–23 / 33–43) **not copied from CSV** this stage → treat exact codes as **UNVERIFIED**. Posterior molars **absent**.
- **JSON box classes** are described as FD vs no-FD colour, with FDI in the **CSV**. Not a 32-class YOLO pack.
- **Counts:** 241 photos, 241 CBCT, 2,892 tooth-level F/D/N labels.
- **License/access:** PhysioNet Credentialed Health Data License 1.5.0, DUA 1.5.0, CITI training. **No anonymous dump.** Files **not downloaded**.
- **Commercial use:** **UNVERIFIED** until DUA is read.

### Rank 2 — SegmentAnyTooth (protocol match, no images)

Paper (JDS / PMC11993027): 5,000 RGB intraoral photos, five views, dentists + CVAT, **FDI**, YOLO11 **tooth** boxes + SAM masks. GitHub [thangngoc89/SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth): MIT **code**; **weights** non-commercial email; **dataset not released**.

### Rank 3 — DentalMate6v listing

Workspace [dentalmate6v](https://universe.roboflow.com/dentalmate6v): “Intraoral Tooth Numbering FDI”, **1.21k images**, **32** classes. Live project HTML **Cloudflare-blocked**. **No sample JPEG inspected.** After Prime, this stays **UNVERIFIED**.

### Rank 4 — DigiLeap / TLNM

arXiv [2608.06275](https://arxiv.org/html/2608.06275v1): 1,272 smartphone photos, 14,736 FDI polygons, Mask R-CNN. Data under **Findata**. External test used Mendeley [6zsnhrds9t](https://data.mendeley.com/datasets/6zsnhrds9t/1) (9,562 RGB views) **without original FDI**.

### Rank 5 — Yoon et al.

[J Dent 104821](https://doi.org/10.1016/j.jdent.2023.104821): 24,578 DSLR intraoral views; tooth-number and caries-stage **boxes**. Scientific Data 2025: **not open source**. Stages 1–2–3 **are not ICDAS**.

## 6. Rejected (reason)

| Item | Why |
| --- | --- |
| Roboflow Prime agi2i | Panoramic X-ray (Stage 2D-2) |
| Zenodo 14827784 | Lesion `d`/`D`, not FDI teeth |
| DENTEX, PANDENT, TL-pano, panoramic Roboflow/Kaggle | OPG / X-ray |
| DenPAR | Periapical **radiographs** |
| Odontify V2 | ICDAS-like folders; no FDI boxes |
| Kaggle cavity/oral-disease YOLO | Lesion/disease, not FDI |
| Mendeley 6zsnhrds9t | RGB views only; no FDI boxes |
| mikhakdental / teeth-fdi-fhsfa / NTU dentistry | FDI names on listing; RGB samples **not** inspected; NTU text is X-ray-oriented |
| HF COde | Intraoral+text claim; FDI boxes **UNVERIFIED** |

## 7. Best candidate

**FDTooth (PhysioNet 1.0.0)** — only original source that jointly verifies RGB intraoral photos, whole-tooth boxes, and FDI numbering.

## 8. License / access

**VERIFIED** as credentialed PhysioNet (License 1.5.0 + DUA + CITI). **Not** CC-BY anonymous download. Commercial terms **UNVERIFIED**.

## 9. Exact FDI classes

FDTooth: **12 anterior** teeth with FDI two-digit IDs in CSV. **Exact numeric list UNVERIFIED** (files not opened). **Not** 11–48 full 32.

## 10. Image modality

FDTooth: **RGB intraoral photographs** (plus unused CBCT). SegmentAnyTooth/DigiLeap/Yoon: RGB in papers only.

## 11. Annotation format

FDTooth: JPEG + CSV (FDI + F/D/N) + MakeSense JSON boxes.

## 12. Image / annotation counts

FDTooth: **241** photos; **1,800** boxes; **2,892** tooth-level CSV rows.

## 13. ICCC compatibility

| Need | FDTooth |
| --- | --- |
| Camera RGB | Yes (photos only) |
| Whole-tooth crop | Partial (anterior boxes) |
| Full-mouth 32 FDI | **No** |
| Open download | **No** |
| ICDAS | **No** (F/D/N ≠ ICDAS) |

A 32-class open RGB dump was **not** sample-verified.

## 14. Remaining uncertainty

- FDTooth JSON vs CSV alignment; why 1,800 boxes vs 2,892 teeth.
- Exact 12 FDI codes.
- DUA commercial limits.
- Whether DentalMate samples are truly RGB (must view after Cloudflare).
- No Hugging Face/Kaggle **open** RGB+FDI+boxes dump verified.

## 15. Recommended next action

**Do not download in this stage.** Next human step: complete PhysioNet credentialing for **FDTooth** if anterior camera FDI/crops are acceptable, **or** open DentalMate “Intraoral Tooth Numbering FDI” in a normal browser and inspect **sample pixels** (reject if X-ray). Do not map any of these labels to ICDAS. Do not mix Zenodo `d`/`D` into tooth IDs.

---

**Final decision: B. CANDIDATE REQUIRES MORE VERIFICATION**

FDTooth is the strongest **verified** source but is credentialed and anterior-only; no fully open 32-class RGB intraoral FDI set passed sample checks.
