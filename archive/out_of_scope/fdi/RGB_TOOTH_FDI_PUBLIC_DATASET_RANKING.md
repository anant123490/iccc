# Public RGB datasets for whole-tooth detection + FDI

Date: 2026-08-26  
Project: CCC AI Dentist Camera 2.0  

Scope: discovery only. No CVAT. No training. No ICDAS / app / 420-image edits. No APIs, no author emails, no PhysioNet/SegmentAnyTooth access requests. No guessed suitability.

Target: RGB intraoral photo → whole-tooth detection (`0 = tooth`) → FDI (later) → crop → existing ICDAS 0–4.

Local camera set (untouched): `C:\Users\anant\OneDrive\Desktop\icdas project\fdi_detection_dataset\images\selected` (420 JPG).

---

## Verdict

**Rank A: none.**  
**Rank B: none verified.**  
**Rank C: none immediately downloadable.**

No public pack jointly verified as: RGB intraoral + whole-tooth boxes/masks + FDI object labels + immediate download + clear license, without permission systems.

The only paper-grade RGB + whole-tooth + FDI sources (FDTooth, SegmentAnyTooth images) require credentialing or weight/image requests — excluded by your rules.

---

## Rank definitions used

| Rank | Meaning |
| --- | --- |
| **A** | RGB intraoral + tooth boxes + FDI + public download + usable stated license |
| **B** | RGB intraoral + **whole-tooth** boxes + public download, no FDI |
| **C** | RGB intraoral + **whole-tooth** masks (boxes convertible) + public download now |
| **D** | Public RGB intraoral images that could later be combined / self-labeled |
| **X** | Excluded: permission, X-ray/3D-only, lesion/pathology-as-teeth, UNVERIFIED license used as if OK |

---

## Candidate sheets

### 1. FDTooth — Rank X (would be A after DUA; forbidden here)

- URL: https://physionet.org/content/fdtooth/1.0.0/ · paper https://doi.org/10.1038/s41597-025-05348-3  
- Modality: RGB intraoral JPEG + CBCT DICOM  
- Counts: 241 patients; 1800 photo boxes; 12 anterior teeth  
- RGB intraoral: yes  
- Views: intraoral photographs (paper)  
- Whole-tooth boxes: yes (MakeSense JSON)  
- Segmentation: no (paper)  
- FDI: yes (CSV 12 anterior; not 32-class)  
- Format: JPEG, JSON, CSV, DICOM  
- License: PhysioNet Credentialed Health Data License 1.5.0  
- Immediate download: **no**  
- Author/institutional permission: **yes (credentialing + DUA)**  
- API key: PhysioNet login  
- Academic: yes after DUA  
- Commercial: **no** (credentialed research license)  
- Train derivative: only under DUA  
- Usefulness: high technically, **not usable under current rules**

### 2. SegmentAnyTooth (images / weights) — Rank X

- URLs: https://github.com/thangngoc89/SegmentAnyTooth · https://doi.org/10.1016/j.jds.2025.01.003  
- Modality: RGB five standard intraoral views (paper)  
- Counts: 5000 photos, 1000 sets, 953 subjects (paper)  
- Boxes + FDI + masks: yes in paper (CVAT)  
- Exact FDI classes: paper excludes 5–8 on laterals; file list **UNVERIFIED**  
- Immediate download of images: **no**  
- Weights: email / non-commercial agreement  
- Usefulness: best pipeline match; **excluded**

### 3. IO150K / Teeth-SEG RGB0.8K — Rank C (blocked)

- URLs: https://zoubo9034.github.io/TeethSEG/ · arXiv 2404.01013  
- GitHub https://github.com/zoubo9034/TeethSEG → **404** (2026-08-26)  
- Claimed: ~800 RGB photos + 80k renders + 70k plaster; orthodontist instance labels; FDI-trained annotators  
- Majority **not** camera RGB (renders/plaster)  
- License: **UNVERIFIED** (no dump)  
- Immediate download: **no**  
- Do not treat as available

### 4. AlphaDent — not A/B/C (pathology, not whole-tooth / not FDI)

- URLs: https://huggingface.co/datasets/ZFTurbo/AlphaDent · https://zenodo.org/records/16582489 · https://github.com/ZFTurbo/AlphaDent · arXiv 2507.22512  
- Modality: RGB DSLR intraoral  
- Counts: 295 patients; ~1200+ images (HF viewer ~1455 rows)  
- Boxes + instance masks: **yes**, 9 **pathology** classes (Black 1–6, abrasion, filling, crown)  
- Whole-tooth class: **no**  
- FDI: **no**  
- Format: YOLO-style instance files (paper)  
- License: **Apache 2.0** (HF/GitHub)  
- Immediate: **yes**, no author email  
- API key: none required for HF/Zenodo; Kaggle mirror may need a free account  
- Academic: yes  
- Commercial: Apache 2.0 allows (attribution)  
- Train derivative: yes for **pathology** models  
- Usefulness: **do not use as tooth or FDI GT**. Optional later auxiliary only.

### 5. Zenodo intraoral caries (already on disk) — Rank X for this stage

- URL: https://zenodo.org/records/14827784 · paper https://doi.org/10.1038/s41597-025-05647-9  
- Local: `data_external/detection/` — 6265 RGB JPG (Stage 2C)  
- Views: maxillary/mandibular/frontal/laterals, with/without retractors  
- Boxes: **lesion `d`/`D`**, not whole teeth, not FDI  
- Formats: YOLO, VOC, COCO, LabelMe  
- Paper article license: **CC BY-NC-ND 4.0** (do not assume commercial/ND-ok training of derivatives from the article terms)  
- Usefulness: **must not** become class `tooth` or FDI

### 6. Mendeley “Teeth or Dental image dataset” — Rank D

- URL: https://data.mendeley.com/datasets/6zsnhrds9t/1  
- Paper: https://doi.org/10.1016/j.dib.2024.110772  
- 9562 pediatric RGB views (front/left/right/occlusal × arches)  
- Boxes/masks/FDI: **none**  
- License: Mendeley listing **CC BY 4.0**; Data in Brief HTML also mentions **CC BY-NC** — **do not guess**; read the file license on download  
- Immediate: public listing  
- Usefulness: unlabeled RGB for future human annotation only; mixed dentition ≠ adult camera domain

### 7. COde (zirak-ai) — Rank D / skip until license+DUA read

- URL: https://huggingface.co/datasets/zirak-ai/COde · DOI 10.57967/hf/6421  
- Claimed: 50k photographs, 8056 radiographs, 4800 patients, clinical text  
- Boxes: **no**  
- FDI: Palmer→FDI in **text**, not pixels  
- HF card: cc-by-4.0; also **Data Usage Agreement.pdf**; Scientific Data article **CC BY-NC-ND**  
- Do not download until a human reads DUA vs card vs article. Not a detector dataset.

### 8. Teeth3DS / 3DTeethSeg — auxiliary 3D only (Rank X for camera)

- OSF: https://osf.io/xctdy · GitHub: CC **BY-NC-ND 4.0** on data  
- 1800 3D scans, ~900 patients, per-vertex **FDI** JSON  
- Not RGB camera photos

### 9. DENTEX / panoramic Roboflow FDI — Rank X

- X-ray OPG. Wrong modality.

### 10. DigiLeap/TLNM, Yoon 24578, Peking 886 plaque — Rank X

- Restricted, unpublished, or author request.

### 11. Roboflow Universe “intraoral tooth” projects — Rank X until license+RGB verified

- Typical **API key**. Licenses often **UNVERIFIED** (Cloudflare). Sibling projects include caries/gingivitis. Do not assume “tooth” = whole-tooth RGB.

---

## Recommended path (still no training)

1. Annotate **your 420** RGB photos with class **`0 = tooth`** (Batch_01 seed already defined).  
2. Do **not** import Zenodo `d`/`D` or AlphaDent pathology as teeth/FDI/ICDAS.  
3. FDI only after whole-tooth boxes exist.  
4. Optional later (human license check): Mendeley unlabeled images; AlphaDent as pathology-only experiment.

ICDAS / backend / frontend / 420 files: **not modified** this pass.
