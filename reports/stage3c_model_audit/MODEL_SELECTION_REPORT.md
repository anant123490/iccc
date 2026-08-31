# Stage 3C model audit — pretrained whole-tooth detector

Date: 2026-08-26  
Scope: **documentation only**. No weights, datasets, packages, inference, annotations, or project-code changes outside this folder.

Target images: **420 RGB intraoral photographs** (class `0 = tooth`).  
ICDAS `.keras` models and Zenodo `d`/`D` lesion XML are **out of scope**.

---

## Decision

### OPTION A — SUITABLE PRETRAINED TOOTH DETECTOR FOUND

**Best candidate for later testing (not a claim that it will work on the 420 photos):**

**SegmentAnyTooth** (Nguyen et al., *Journal of Dental Sciences* 2025)  
https://github.com/thangngoc89/SegmentAnyTooth  
https://doi.org/10.1016/j.jds.2025.01.003

It is the only audited artifact that is **documented** as:

- RGB **intraoral** photographs (five standard views matching this project's Frontal / occlusal / laterals)
- **Whole-tooth** YOLO11 bounding boxes (then SAM masks)
- Pretrained **`.pt` weights exist** (released after a signed **non-commercial** agreement emailed to the authors)

It is **not** drop-in production GT. Treat any future boxes as **machine candidates requiring human QC**. Map all YOLO class IDs to project class **`tooth` / 0**. Do **not** treat FDI numbers from this model as the project's FDI stage.

---

## What was searched

| Source type | What was inspected |
| --- | --- |
| Hugging Face | Model cards: panoramic/pathology YOLO, OPG cavity, X-ray seg, DINOv3 disease detector, oral-lesion dataset card. HF search UI for `tooth detection yolo` returned **0** listed models (filter/UI), so named cards were opened separately. |
| GitHub | SegmentAnyTooth, AlphaDent, dental-detector wrapper, panoramic YOLO repos |
| Roboflow Universe | Intraoral tooth detection / teeth detection / intraoral-cam listings (partial: Cloudflare blocked some full pages) |
| Papers | SegmentAnyTooth JDS 2025; BMC occlusal detection 2025; RS intraoral YOLOv5 2024 — latter two **without** public weights |

No `wget`/`curl`/clone/`pip` of models. Dataset pages were read only as **training-data documentation for models**, not as downloads.

---

## Ranking summary

| Rank | Model | Why |
| --- | --- | --- |
| **A** | SegmentAnyTooth | RGB intraoral whole-tooth boxes + documented weights (NC, email-gated) |
| **B** | Roboflow `intraoral-tooth-detection-rohlq` | RGB intraoral instance-seg listing; **license UNKNOWN**; hosted weights |
| **B** | Roboflow `teeth-detection-0qd49` | OD listing, 403 images; modality/license unverified |
| **C** | Roboflow `tooth-1o3em`, Yang Dental intraoral-cam | Larger or intraoral-titled; mixed/unverified classes or license |
| **D** | AlphaDent YOLOv8x-seg | RGB but **pathology**, not whole tooth |
| **D** | HF panoramic/X-ray/cavity/disease YOLOS | Wrong modality and/or wrong object |
| **D** | COCO Ultralytics stock YOLO | Generic detector |
| **D** | Wrapper-only / papers without weights | No usable checkpoint |

---

## Final recommendation (required items)

1. **Best candidate:** SegmentAnyTooth YOLO11-nano (view-specific `.pt`) ± optional SAM.  
2. **Second-best:** Roboflow Intraoral Tooth Detection `intraoral-tooth-detection-rohlq/1` — **only after** license and class names are verified on the live Universe page.  
3. **Why best:** Only clearly documented **RGB intraoral + whole-tooth boxes + obtainable pretrained weights**.  
4. **License:** Code **MIT**; **weights Non-Commercial** (signed PDF + email). Paper **CC BY-NC-ND 4.0**. Ultralytics runtime is typically **AGPL-3.0**. **Not automatically “safe” for commercial ICCC product use.** Research/student use still needs the authors’ weight agreement.  
5. **RGB compatibility:** **HIGH** (trained on intraoral RGB; phones + DSLR). Domain shift still possible (cameras, lighting, retractors, mixed dentition excluded in training).  
6. **Whole-tooth compatibility:** **YES** for the YOLO stage (boxes around teeth). Output classes are **FDI numbers**, not a single `tooth` class. Masks can also yield boxes; conversion was **not** performed.  
7. **Integration effort:** Medium. Email agreement → receive weights → match each of 420 images to `upper` / `lower` / `front` / `left` / `right` → Ultralytics predict → remap classes to `0` → human QC. Extra SAM deps if masks wanted.  
8. **Risks:** NC/AGPL; view mis-routing; laterals omit posterior teeth in the paper protocol; crowding/overlap; leftover FDI IDs leaking into detection labels if not remapped; no proof on *this* 420-image set.  
9. **Weights obtainable:** **Yes, gated** — not a public anonymous URL.  
10. **Local run:** **Yes, after weights**, on the already-present Ultralytics/PyTorch stack for the YOLO stage (SAM extras not verified installed).  
11. **Before candidate boxes:** (a) complete SAT weight license email; (b) legal/academic OK for NC+AGPL; (c) view-routing spec; (d) remap to `tooth=0`; (e) separate Stage 3C-inference with QC — **not this audit**.

---

## Rejected close calls (do not confuse)

**AlphaDent** has public **Apache-2.0** RGB photo **YOLOv8x** weights — the **wrong objects** (caries/abrasion/crown/filling). Using it would repeat the lesion-XML mistake.

Stock **COCO YOLO** is installed-capable and **explicitly forbidden** as a tooth detector.

---

## Safety check

| Item | Result |
| --- | --- |
| Files downloaded | **0** |
| Datasets downloaded | **0** |
| Packages installed | **0** |
| Models trained | **0** |
| Tooth boxes generated | **0** |
| FDI labels generated | **0** |
| ICDAS modified | **NO** |
| RGB images modified | **NO** |
| Lesion XML modified | **NO** |

---

**STOP.** Do not download SAT weights or run inference in this stage.
