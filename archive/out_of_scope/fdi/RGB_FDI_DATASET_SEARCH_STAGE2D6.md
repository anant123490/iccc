# Stage 2D-6 — Exhaustive public RGB tooth / FDI dataset search

Date: 2026-08-26  
Scope: discovery and verification only. No PhysioNet bypass, no synthetic FDI, no ICDAS remaps, no application / `dataset/` / `ml/` / `models/` / backend / frontend changes, no training.

Target: **RGB intraoral photograph → whole-tooth detection → FDI identity → tooth crop** (then existing ICDAS 0–4). FDI ≠ ICDAS. Zenodo 14827784 remains **lesion `d`/`D` only**.

---

## 1. Executive summary

An internet-wide pass over Zenodo, Figshare, Mendeley, Kaggle, Hugging Face, Roboflow, GitHub, OSF, PhysioNet, journal data papers, and challenge pages **did not find a publicly downloadable pack** that jointly verifies:

1. real RGB intraoral photos,  
2. whole-tooth instance boxes or masks,  
3. explicit FDI (or documented equivalent) identity,  
4. open files in this environment.

**Final decision: OPTION 3.**

The only **technically documented** RGB + whole-tooth + FDI source remains **FDTooth**, which is **credentialed** and was **not** downloaded (Stage 2D-5, HTTP 403).

---

## 2. Search scope

Venues: Zenodo, Figshare, Mendeley Data, Kaggle, Hugging Face, Roboflow Universe, GitHub, OSF, PhysioNet, university/project pages, Scientific Data / JDS / J Dent / BMC / CVPR, arXiv, MICCAI Grand Challenge.

Not limited to Kaggle/HF/Roboflow. Titles were not trusted (Prime was panoramic). Files were not downloaded unless they were clearly public **and** on-label; none met that bar for FDI+RGB+boxes together, so **no new archives were fetched**.

---

## 3. Search terminology used

intraoral RGB tooth dataset; tooth bounding box FDI; tooth instance segmentation intraoral; 32-class tooth dataset; Universal numbering intraoral photographs; Palmer notation dental photos; IO150K RGB0.8K; AlphaDent; SegmentAnyTooth; Teeth3DS FDI; COde oro-dental; panoramic vs intraoral RGB.

Also: numeric class IDs vs FDI vs Universal vs Palmer.

---

## 4. Strong candidates

**None** that are publicly downloadable **and** file-verified.

FDTooth would be strong **after** legal PhysioNet access, but it is **not** an open dump and covers **12 anterior** teeth only.

---

## 5. Potential candidates

| Name | Why not A |
| --- | --- |
| **IO150K RGB0.8K** | Paper: ~800 RGB photos, orthodontists trained on **FDI**, instance labels. **github.com/zoubo9034/TeethSEG → 404**. Rest of IO150K is **renders/plaster**. **DOWNLOAD STATUS: NOT PUBLICLY DOWNLOADABLE** |
| **AlphaDent** | Public RGB + instance masks, Apache 2.0, Zenodo/HF. Classes are **pathology (Black 1–6, filling, crown, abrasion)** — **not** tooth FDI. Do not remap to FDI or ICDAS |

No other public RGB **whole-tooth** detector dump with **unclear-but-plausible** identity labels was file-verified.

---

## 6. Labeling / reference candidates

| Name | What it has | Gap |
| --- | --- | --- |
| **COde** (HF zirak-ai/COde) | Many photos + text; paper: Palmer **normalized to FDI in EMR text**; CC BY-NC-ND on the article | **No whole-tooth boxes** documented |
| **Teeth3DS / 3DTeethSeg** (OSF) | **3D scans** with **per-vertex FDI** | Not RGB camera photos |
| **COde / DigiLeap papers** | Protocol-level FDI | Images restricted or no boxes |

---

## 7. Rejected / not suitable

| Dataset | Reason |
| --- | --- |
| Roboflow Prime agi2i | **Panoramic X-ray** (Stage 2D-2) |
| DENTEX, PANDENT, panoramic Roboflow FDI | OPG / X-ray |
| Zenodo 14827784 | Lesion `d`/`D` |
| DentalMate6v | No paper; Cloudflare; sibling project is **caries/gingivitis** |
| DigiLeap / TLNM | Findata; not public |
| Yoon 24,578 | Not open source |
| SegmentAnyTooth **images** | Not released |
| Mendeley `6zsnhrds9t` | RGB views, **no** FDI boxes |
| Peking 886 plaque | Author request; single-tooth crops; plaque not FDI |
| MAP Trial Figshare | Clinical photos, **no** tooth annotations |
| DenPAR | Periapical **radiographs** |
| Odontify ICDAS folders | No FDI boxes |
| Kaggle cavity YOLO | Lesion class |

---

## 8. FDTooth status

Unchanged from 2D-5: **not downloaded**, **403** without credentials, **USER ACTION REQUIRED** (PhysioNet credentialing + CITI + DUA). Research use only; no redistribution; not commercial under documented license. Anterior-only.

---

## 9. Comparison table

| Dataset | RGB intraoral | Whole-tooth loc. | FDI | Public files | Role |
| --- | --- | --- | --- | --- | --- |
| FDTooth | Yes (paper) | Yes (paper) | Yes CSV | **No** | AUXILIARY after DUA |
| SegmentAnyTooth | Yes (paper) | Yes (paper) | Yes (paper) | **No** | NOT SUITABLE |
| IO150K RGB0.8K | Claimed | Claimed | Claimed protocol | **No (404)** | NOT SUITABLE |
| AlphaDent | Yes | Pathology masks | **No** | Yes | NOT SUITABLE |
| COde | Photos + X-ray | **No** | Text only | Listing | NOT SUITABLE |
| Teeth3DS | **No** (3D) | 3D | Yes | OSF | NOT SUITABLE |
| Zenodo 14827784 | Yes | Lesions | **No** | Yes | Lesion only |
| Prime Roboflow | **No** (OPG) | Yes | Listing | n/a | NOT SUITABLE |

---

## 10. Licensing / access

- **Open-ish RGB:** AlphaDent Apache 2.0; Zenodo lesions CC BY 4.0; Mendeley views CC BY 4.0 — none are FDI tooth instances.  
- **COde paper:** CC **BY-NC-ND 4.0**.  
- **FDTooth:** credentialed research-only.  
- **DentalMate / IO150K files:** license **UNVERIFIED**.

---

## 11. Downloadability

No candidate was downloaded this stage. Public zips that exist (AlphaDent, Zenodo lesions, Mendeley views) **fail** the FDI+whole-tooth requirement. Everything that **passes** the scientific requirement **fails** anonymous download.

---

## 12. Recommended next action

1. Treat **Option 3** as the data-availability fact for open 32-class camera FDI.  
2. If the project needs **any** RGB+FDI+boxes: user completes **PhysioNet FDTooth** for **anterior auxiliary research** only (not 32-class production).  
3. Otherwise **collect/annotate** ICCC camera photos with FDI — do not invent FDI, do not use ICDAS or `d`/`D` as tooth IDs.  
4. Optional later: human-browser check of DentalMate **samples** (still unverified).  
5. Do **not** implement a detector or change ICDAS/`keras` until a legal local dump exists.

---

### OPTION 3

No sufficiently suitable publicly downloadable RGB + whole-tooth + FDI dataset was found.

Do not claim Option 1. Architecture remains detect → **FDI** → crop → **ICDAS 0–4**.
