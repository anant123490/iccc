# Stage 2E — Existing RGB dataset FDI annotation feasibility audit

Date: 2026-08-26  
Scope: **read-only** inspection of RGB/tooth data already in this repository. No labeling, training, merging, or modification of images, annotations, ICDAS data, models, backend, or frontend.

**FDI** = which tooth. **ICDAS** = caries severity. These are separate tasks.  
**FDI NOT CURRENTLY AVAILABLE** in any on-disk annotation.

---

## 1. Executive Summary

This project already holds a large RGB intraoral photograph collection (Zenodo `10.5281/zenodo.14827784`: **6,265** JPG files). Those photographs are clinically usable as **full-mouth / arch / lateral** frames.

They do **not** contain whole-tooth localization or FDI identity.

Existing boxes are **decay lesions** (`d` = primary-tooth decay detection, `D` = permanent-tooth decay detection). Stage 2C counted **6,728** COCO boxes on **2,227** images. Those must not be counted as tooth instances and must not be converted into FDI labels.

The intended ICDAS crop tree (`dataset/train|val|test`) is empty. `dataset/annotations.csv` lists **643** ICDAS 0–4 crop filenames with **0** files on disk. `cropped_teeth/` and `labels/` contain only `.gitkeep`.

**Question 1 — tooth detector from existing annotations: NO.**  
**Question 2 — convert existing annotations into FDI: NO.** Full frames could be boxed and FDI-labelled later, but that is new work, not reuse of current boxes.  
**Recommended next action: pursue legitimate FDTooth (PhysioNet) access** (Strategy D / Question 5 option C). Do not start manual FDI labeling on lesion boxes.

| Metric | Result |
| --- | --- |
| RGB intraoral images | **6,265** (Zenodo; plus 16 local excluded files not recommended for FDI) |
| Whole-tooth annotated images | **0** |
| Whole-tooth instances | **0** |
| Bounding boxes | **6,728** lesion boxes (not whole-tooth) |
| Segmentation | **None** for teeth |
| Existing FDI labels | **NOT AVAILABLE** |
| FDI classes verified | **0** |
| Potential FDI classes | **NOT VERIFIED** (do not treat visual position as FDI) |
| Images requiring manual FDI | **6,265** if using Zenodo frames (after new tooth boxes) |
| Teeth requiring manual FDI | **0** existing; **ESTIMATE ~67,679** if boxing all views |
| Detection feasibility | **NO** (current boxes) |
| FDI feasibility | **NO** with current labels |
| Manual annotation burden | **High / not practical** as the primary path |
| Recommended strategy | **D** — legitimate FDTooth access |

---

## 2. Existing Dataset Inventory

Directories checked: `data_external/`, `dataset/`, `cropped_teeth/`, `labels/`, `reports/` (documentation only), plus a workspace image sweep excluding virtualenv packages.

| Resource | On disk? | Role for FDI / detection |
| --- | --- | --- |
| `data_external/detection/raw/` | Yes — 6,265 JPG | RGB intraoral photos; **no FDI** |
| `data_external/detection/annotations/{yolo,pascal-voc,ms_coco}` | Yes | **Lesion** boxes `d`/`D` only |
| `dataset/train`, `val`, `test` | **No image trees** | Cannot use |
| `dataset/raw` | Empty of images | — |
| `dataset/annotations.csv` | 643 rows | ICDAS crop index; **files missing** |
| `dataset/excluded/5/` | 11 RGB files | Tight/lesion-style crops; no boxes; no FDI |
| `dataset/excluded/6/` | 5 ChatGPT PNGs | Exclude from clinical training |
| `dataset/whatsapp_manifest.json` | Manifest only | Referenced `whatsapp_*.png` files not present |
| `cropped_teeth/images/` | `.gitkeep` only | **0** crops |
| `labels/` | `.gitkeep` only | **No** `labels.csv` |
| AlphaDent / FDTooth / COde / Teeth3DS pixels | **Not present** | Documented in Stage 2D; not in this repo |

Name-only folders were not treated as tooth datasets unless files were verified.

---

## 3. RGB Image Statistics

### Zenodo intraoral set (`data_external/detection/raw/`)

| Item | Count / value |
| --- | --- |
| Total images | **6,265** |
| Extension | `.jpg` only |
| Color | **RGB** (6,265); grayscale 0 |
| Corrupted (Stage 2C) | 0 |
| Width × height | 617–4080 × 347–3206 px |
| Intraoral photographs | **Yes** (sampled frontal, occlusal, lateral) |
| View type | Full-mouth / partial-mouth **arch or segment**, not single-tooth crops |
| Quality | Generally high; flash speculars; laterals often shallow depth of field |

**Protocol**

| Protocol | Images |
| --- | --- |
| no_retractors | 2,752 |
| retractors | 2,753 |
| pilot | 760 |

**View**

| View | Images |
| --- | --- |
| Frontal | 1,259 |
| Left_Lateral | 1,249 |
| Right_Lateral | 1,252 |
| Mandibular | 1,251 |
| Maxillary_Occlusal | 1,254 |

**Protocol × view**

| | Frontal | Left_Lateral | Mandibular | Maxillary_Occlusal | Right_Lateral |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_retractors | 550 | 550 | 551 | 551 | 550 |
| pilot | 156 | 149 | 151 | 152 | 152 |
| retractors | 553 | 550 | 549 | 551 | 550 |

### Other RGB on disk

- **11** files under `dataset/excluded/5/`: RGB; often tight occlusal/lesion crops; some look like photographs of a screen. Poor FDI context.
- **5** ChatGPT-generated PNGs under `dataset/excluded/6/`: not clinical photographs.
- Model `test_evaluation` PNGs under `models/` are plots, not intraoral data.

---

## 4. Tooth Annotation Statistics

**Whole-tooth instances in this repository: 0.**  
Lesion boxes are reported separately so they are not mistaken for teeth.

| Annotation statistic | Value | What it is |
| --- | --- | --- |
| YOLO `.txt` files | 2,245 | Darknet lesion labels |
| VOC `.xml` files | 2,245 | Same lesions |
| COCO JSON files | 3 (`train` / `valid` / `test`) | Same lesions |
| YOLO instance lines | 6,782 | Includes empty files as 0-line |
| Empty YOLO files | 65 | No objects |
| COCO bounding boxes | **6,728** | Primary Stage 2C count |
| Images with lesion annotations | **2,227** | Paper-scale subset |
| Images without annotations | **4,038** | Still have visible teeth; **no boxes** |
| Invalid boxes (Stage 2C) | 84 | Geometry issues |
| Annotation files missing image (Stage 2C) | 36 | Filename mismatch |
| Mean boxes per YOLO file | 3.02 (max 9) | Compatible with **few lesions per photo**, not a full dentition |
| Whole-tooth boxes | **0** | — |
| Tooth segmentation / masks | **0** | COCO `segmentation` empty |
| FDI labels | **NOT AVAILABLE** | — |

Sample VOC objects (e.g. mandibular XML open in the editor) use `<name>D</name>` with small `bndbox` regions — decay spots, not full crowns.

---

## 5. Annotation Format Analysis

| Format | Location | Classes | Geometry |
| --- | --- | --- | --- |
| Darknet YOLO | `annotations/yolo/` | id `0` = D (6,174), id `1` = d (554) | Normalized `cx cy w h` |
| Pascal VOC | `annotations/pascal-voc/` | names `D`, `d` | Pixel xmin/ymin/xmax/ymax |
| COCO | `annotations/ms_coco/` | see below | `bbox` only |

**COCO splits (image-level, lesion task)**

| Split | Images | Annotations | Category ids |
| --- | --- | --- | --- |
| train.json | 1,781 | 5,386 | 1=`d`, 2=`D` |
| valid.json | 214 | 652 | 1=`d`, 2=`D` |
| test.json | 232 | 690 | **inverted**: 1=`D`, 2=`d` (Stage 2C) |

There is no COCO/YOLO category named tooth, FDI 11–48, or ICDAS 0–4.

`dataset/annotations.csv` format: `filename,icdas_score,split` — ICDAS only; no boxes; no FDI; **0 matching files**.

---

## 6. Current Label Meaning

| Label | Meaning in this project | Not meaning |
| --- | --- | --- |
| YOLO `0` / VOC-COCO `D` | Permanent-tooth **decay lesion** (detection) | Not FDI. Not ICDAS 0–4. Not “whole tooth” |
| YOLO `1` / VOC-COCO `d` | Primary-tooth **decay lesion** (detection) | Not FDI. Not mixed dentition FDI map |
| `icdas_score` 0–4 in CSV | Claimed caries **severity** on missing crops | Not tooth identity |
| Folders `excluded/5` and `excluded/6` | Out-of-scope ICDAS grades / synthetic images | Not FDI classes |
| Numeric class `0` in YOLO | **D**, not FDI 11 | Do not map 0→11, 1→12, etc. |

**FDI ID: NOT AVAILABLE.**

---

## 7. Tooth Position / Ordering Analysis

This section is **feasibility only**. No FDI numbers were assigned.

Sampled frames:

- **Frontal (no retractor):** anterior upper and lower crowns are ordered and often separable; crowding and a shallow depth of field hide posteriors. A dentist could often label **anterior** teeth if whole-tooth boxes were drawn. Left/right depends on camera convention; **mirroring is NOT VERIFIED** in metadata.
- **Maxillary occlusal (retractors):** a single-jaw arch is visible; tooth sequence along the arch is the clearest of the three view types. Contacts still overlap. Third molars may be absent — absence is **not** proof of FDI 18/28 missing without a clinician.
- **Left lateral:** roughly a posterior segment; crowns overlap heavily. Exact FDI is unstable from this view alone. Pairing with the same `anonymous_*` occlusal/frontal (when the ID exists) would be required.

| Factor | Finding |
| --- | --- |
| Left/right orientation | Visible in frontal/occlusal **as photographed**; no documented mirror flag |
| Upper/lower | Occlusal folders are jaw-specific; frontal shows both |
| Tooth ordering | Clear enough **on many occlusal and anterior frontal** images for a human expert to *attempt* FDI after boxing |
| Complete arch | Common on retractor occlusal samples; not guaranteed |
| Quadrant-only | Laterals |
| Missing teeth | Would create numbering ambiguity; **not quantified** per image |
| Rotated / mirrored photos | Not audited exhaustively; treat as a labeling risk |
| Existing lesion boxes as order | **Unusable** — they do not enumerate teeth |

Existing annotations do **not** give enough **tooth-instance** spatial records to attach FDI. The **pixels** on occlusal/frontal views often would, **after new whole-tooth boxes**, for a subset of teeth.

---

## 8. FDI Coverage Feasibility

No annotation file contains FDI 11–48. Visual presence of an arch does **not** verify class coverage.

| FDI | Status | Evidence |
| --- | --- | --- |
| 11 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE. Frontal/occlusal photos may show a maxillary central; identity not proven. |
| 12 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 13 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 14 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 15 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 16 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 17 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 18 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 21 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 22 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 23 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 24 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 25 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 26 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 27 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 28 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 31 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 32 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 33 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 34 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 35 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 36 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 37 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 38 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 41 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 42 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 43 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 44 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 45 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 46 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 47 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |
| 48 | NOT VERIFIED | FDI NOT CURRENTLY AVAILABLE |

**Verified present: 0. Clearly absent: not claimed** (teeth may exist in pixels). **Possibly present:** only as unproven anatomy in photographs.

---

## 9. Tooth Detection Feasibility

**Do we have enough whole-tooth localization annotations to train a tooth detector?**

### NO

Reasons:

1. All instance labels are **lesions**, not whole teeth.
2. Mean ~3 boxes/image cannot represent a dentition.
3. 4,038 images have **no** boxes at all.
4. Using `d`/`D` boxes as tooth detectors would teach the model to find cavities, not teeth.

If **new** whole-tooth boxes were drawn on the 6,265 RGB frames, detection might become **POSSIBLY** feasible. That is a new annotation project, not this audit’s existing-label answer.

---

## 10. FDI Identification Feasibility

**Can the existing RGB data realistically support FDI identification after manual labeling?**

### NO with current labels. POSSIBLY only after large new work on a subset.

| Factor | Assessment |
| --- | --- |
| Number of images | 6,265 frames exist — enough photos, not enough **tooth** labels |
| Number of teeth | **0** labelled instances |
| Position clarity | Adequate on many occlusal/frontal samples; poor on laterals and tight excluded crops |
| Jaw visibility | Occlusal folders separate jaws; frontal mixes both |
| Class coverage | **0 verified** FDI classes |
| Left/right | Unverified mirroring policy |
| Image quality | Generally adequate for expert labeling of anteriors/occlusal |
| Missing teeth | Unquantified risk |
| Annotation consistency | Lesion formats are consistent with each other; they are the **wrong object** |

Strategy A (“existing tooth boxes + manual FDI”) is **not applicable**.

---

## 11. Manual Annotation Effort Estimate

| Item | Result |
| --- | --- |
| 1. RGB images potentially usable | **6,265** Zenodo intraoral JPGs |
| 2. Tooth instances potentially usable **now** | **0** |
| 3. Images requiring FDI labeling (if using those frames) | **6,265** |
| 4. Tooth instances requiring FDI labeling **now** | **0** |
| 5. Estimated FDI labels if boxing **all** views | **ESTIMATE 67,679** |
| 6. Images that appear easy | **ESTIMATE 1,100** (retractors occlusal: 551 maxillary + 549 mandibular) |
| 7. Images that appear ambiguous | **ESTIMATE 4,862** (laterals 2,501 + frontals 1,259 + no-retractor occlusal 1,102) |
| 8. Images that should probably be excluded | **5** ChatGPT; **11** tight `excluded/5` crops; laterals **if used without** a paired arch from the same ID (**2,501**) |
| 9. Workload | **High.** Relabeling lesion boxes as FDI is invalid. Even the 1,100 “easy” occlusal frames imply **ESTIMATE ~15,400** tooth boxes **and** FDI tags (14 teeth × 1,100). Full-set boxing is not a reasonable major-project primary path. |

**Estimate method for 67,679:**  
retractor occlusal 1,100 × 14 + no-retractor occlusal 1,102 × 12 + pilot occlusal 303 × 13 + frontal 1,259 × 12 + laterals 2,501 × 8.  
Assumed visible-tooth counts are **not** measured per image.

Pilot occlusal (**303** images) is a **medium** bucket (same view as the easy set, pilot protocol): 1,100 + 303 + 4,862 = **6,265**.

Ghost CSV crops (**643**) cannot be labelled: files are missing.

---

## 12. Data Leakage / Patient Split Analysis

| Signal | What exists |
| --- | --- |
| Patient IDs | **Not explicit.** Filename pattern `anonymous_(\d{3}-\d{3}-\d+)` yields **889** unique IDs on **4,356** images; **1,909** images lack that pattern |
| Photos per parsed ID | Mode **5** (852 IDs); max 5 — consistent with one photo per view |
| Image IDs | Filenames include timestamps and view names |
| Train/val/test | COCO **lesion** splits (1,781 / 214 / 232 images). **Not** verified as patient-level. CSV ICDAS splits exist for **missing** files (440 / 110 / 93) |
| Duplicate image stems (Stage 2C) | 0 duplicate stems; 0 content-hash groups reported |
| Multiple photos per person | **Likely** for the 889 IDs (typically 5 views). Using image-level COCO splits for a future tooth/FDI model would **risk leakage** across views of the same mouth |

**Patient-level splitting: POSSIBLY** for the 889 parsed IDs; **not possible** for the 1,909 unmatched names without another key. Splits were **not** changed.

---

## 13. License / Provenance

| Dataset | Source | URL (in repo) | License | Research / derivatives |
| --- | --- | --- | --- | --- |
| Zenodo intraoral caries detection | Zenodo + Scientific Data paper | https://zenodo.org/records/14827784 ; DOI `10.5281/zenodo.14827784` | **CC BY 4.0** (`data_external/detection/SOURCE.txt`, Stage 2C) | Research with attribution is the usual CC BY reading. **Not legal advice.** Manual derivative boxes would typically require attribution to the original. |
| `dataset/annotations.csv` / missing crops | Unclear local export | — | **LICENSE: NOT VERIFIED** | Cannot annotate missing files |
| `dataset/excluded/5` | Local / WhatsApp-style names | — | **LICENSE: NOT VERIFIED** | — |
| `dataset/excluded/6` | ChatGPT-generated | — | Synthetic; exclude from clinical FDI | — |
| FDTooth | PhysioNet (not downloaded) | Stage 2D-5 | Credentialed license + DUA | Research-only if access is granted; **not on disk** |

---

## 14. Problems / Risks

1. **Wrong object:** lesion boxes ≠ teeth ≠ FDI.
2. **No ICDAS pixels** in `dataset/train|val|test` despite a 643-row CSV.
3. **Scale:** tens of thousands of new boxes if all Zenodo views are used.
4. **32-class FDI** cannot be claimed from this audit.
5. **Laterals and crowding** make expert FDI error-prone.
6. **Unverified mirroring** can swap left/right FDI (1x vs 2x, 4x vs 3x).
7. **COCO test category-id inversion** (Stage 2C) if anyone reuses COCO for lesions.
8. **Patient leakage** if future splits ignore the 889 anonymous IDs.
9. **ChatGPT images** must not enter clinical training.
10. **d vs D** is dentition type for **decay detection**, not an FDI numbering system.

---

## 15. Recommended Strategy

### STRATEGY D — existing local data is insufficient for FDI; pursue legitimate FDTooth / PhysioNet credentialing

**Why this is the most realistic next step**

- **A** fails: there are **no** whole-tooth boxes to attach FDI to.
- **B** still requires drawing tooth boxes first; lesion boxes cannot be “partially” turned into FDI.
- **C** (new RGB collection) will likely be needed later for posterior / 32-class coverage, but Stage 2D already identified a **verified** RGB + whole-tooth + FDI source (FDTooth; restricted; anterior-only).
- **E** (combine later) is a plausible **long-term** stack (FDTooth + new posterior RGB + Zenodo **lesion-only**). It is not the next action: do not merge datasets now.

Question 5 lettering differs from Strategy A–E: the **single next action** is **C. pursue legitimate FDTooth access**, which is **Strategy D**.

---

## 16. Exact Next Step

1. Review this audit (no labeling in this stage).
2. Complete **legitimate** PhysioNet credentialing, required training, and DUA for **FDTooth 1.0.0** (Stage 2D-5: unauthenticated download was 403).
3. After access, run a **new** acquisition/inspection stage on FDTooth only. Keep it separate from ICDAS `d`/`D` and from `dataset/train|val|test`.
4. Do **not** manually FDI-label Zenodo lesion boxes. Do **not** train a tooth detector on `d`/`D`.

---

## Required final summary table

| Metric | Result |
| --- | --- |
| RGB intraoral images | 6,265 |
| Whole-tooth annotated images | 0 |
| Whole-tooth instances | 0 |
| Bounding boxes | 6,728 lesion boxes (not whole-tooth) |
| Segmentation | None (teeth); COCO segmentation empty |
| Existing FDI labels | NOT AVAILABLE |
| FDI classes verified | 0 |
| Potential FDI classes | NOT VERIFIED |
| Images requiring manual FDI | 6,265 (if using Zenodo frames after new boxes) |
| Teeth requiring manual FDI | 0 existing; ESTIMATE ~67,679 if boxing all views |
| Detection feasibility | NO |
| FDI feasibility | NO (current labels) |
| Manual annotation burden | High / not practical as primary path |
| Recommended strategy | D (legitimate FDTooth access) |

---

## Important final decision

### QUESTION 1

Can the existing RGB dataset be used to train a whole-tooth detector?

**NO** — not with current annotations. Boxes are lesions.

### QUESTION 2

Can the existing RGB dataset realistically be manually annotated with FDI labels?

**NO** as a conversion of existing boxes. **POSSIBLY** for a dentist on new whole-tooth boxes on occlusal/frontal subsets; **not realistic** as the sole path to 32-class FDI for this major project.

### QUESTION 3

Approximately how many teeth would need manual FDI labeling?

- **0** existing whole-tooth instances.
- **Not applicable** to the 6,728 lesion boxes.
- **ESTIMATE ~15,400** if only the 1,100 retractor-occlusal frames are boxed (~14 teeth/image).
- **ESTIMATE ~67,679** if all 6,265 views are boxed (method in §11).

### QUESTION 4

Would manual FDI annotation be practical for this major project?

**NO** as the primary path (wrong objects today; tens of thousands of new boxes for full coverage).

### QUESTION 5

Should we: A manually annotate now, B collect more RGB first, C pursue legitimate FDTooth access, D combine existing RGB detection data with another FDI resource, or E another strategy?

**C — pursue legitimate FDTooth access** (maps to Strategy D).

Explain: public search (Stage 2D-6) found no downloadable RGB + whole-tooth + FDI pack. Local data cannot be converted by tagging `d`/`D`. FDTooth remains the only previously verified RGB + whole-tooth + FDI source, behind PhysioNet. Combining Zenodo lesions with FDI later (option D/E) is a later architecture choice, not a substitute for obtaining real tooth-identity labels.

---

**STOP.** Stage 2E is inspection only. No FDI labeling, no training, no ICDAS/app/dataset merges.
