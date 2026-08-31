# Stage 2D-1 — Detection dataset role and architecture

Date: 2026-08-26  
Scope: inspection only. No search, download, training, application, FastAPI, Streamlit, database, `.keras`, or ICDAS-folder changes. No detection model.

Sources: `STAGE2C_ZENODO_DETECTION_REPORT.md`, `reports/stage2c_zenodo_detection_report.json`, `GLOBAL_DATASET_SEARCH.md`, `reports/global_dataset_search.json`, plus existing layout (`dataset/` vs `data_external/detection/`, `docs/ARCHITECTURE.md`).

---

## Current Zenodo role

Zenodo **10.5281/zenodo.14827784** is **lesion-detection data only**.

| Role | Decision |
| --- | --- |
| Lesion-detection data | **YES** |
| Tooth-detection data | **NO** |
| FDI data | **NO** |
| ICDAS classification data | **NO** |

On disk: **6,265** RGB JPGs; **2,227** with boxes; **6,728** boxes; classes **`d`** (primary decay) and **`D`** (permanent decay). Boxes are **caries lesions**, not whole-tooth instances.

**Warning: `d` and `D` must never be mapped to ICDAS 0–4.** They are dentition type (primary vs permanent decay), not severity grades.

---

## What Zenodo can provide

- Open RGB intraoral photographs (five clinical views; retractor / no-retractor / pilot).
- A later **lesion detector** (YOLO/VOC/COCO already present) that marks decay *patches* on a full intraoral frame.
- Optional **lesion crops** for a research overlay or a second-stage “where is the spot” map.
- Logical isolation under `data_external/detection/` (already gitignored).

It can sit **beside** the current ICDAS softmax path, not inside it.

---

## What Zenodo cannot provide

- Whole-tooth bounding boxes or tooth-instance masks.
- FDI numbering (11–48).
- Crops that are valid **tooth** inputs for ICDAS 0–4 (lesion boxes are smaller and location-biased; sound teeth have no boxes).
- ICDAS 0–4 (or 0–6) labels.
- A substitute for the existing MobileNetV3 ICDAS classifier in `dataset/` / `.keras`.

Lesion boxes **must not** replace tooth boxes in the camera pipeline.

---

## Intended ICCC pipeline

```
Camera image
  → whole-tooth detection
  → FDI numbering
  → tooth crop
  → ICDAS 0–4 classification
  → report
```

Current product (`docs/ARCHITECTURE.md`): camera/upload → **full-frame** resize 224 → softmax ICDAS 0–4. There is **no** tooth detector and **no** FDI stage.

### Where a Zenodo lesion detector could fit

| Stage | Zenodo lesion detector |
| --- | --- |
| Whole-tooth detection | **Does not fit.** Wrong object. |
| FDI numbering | **Does not fit.** No tooth IDs. |
| Tooth crop | **Does not fit.** Crops lesions, not teeth. Sound teeth are unlabeled. |
| ICDAS 0–4 | **Does not fit.** `d`/`D` ≠ ICDAS. |
| Report (optional extra) | **May fit later** as a **parallel overlay**: after (or beside) ICDAS, draw lesion boxes on the original intraoral image. Must not drive FDI or the 0–4 softmax. |

The **required** chain remains: **tooth instance → FDI → tooth crop → existing ICDAS classifier**. Zenodo is optional visual evidence, not a link in that chain.

---

## Data separation (keep)

| Tree | Role |
| --- | --- |
| `dataset/` | ICDAS 0–4 classification only |
| `data_external/detection/` | Zenodo lesion detection only |
| Future FDI / whole-tooth set (not on disk) | Whole-tooth boxes + FDI only |

Do not copy Zenodo into `dataset/train|val|test`. Do not train ICDAS on `d`/`D`. Do not train FDI on lesion classes.

---

## Missing capability

The camera pipeline is blocked on **RGB intraoral whole-tooth bounding boxes with FDI numbering**.

Stage 2B: no large easy-open RGB dump of **both** was file-verified. Listings exist (Roboflow Prime 32-class; PhysioNet FDTooth anterior + DUA; SegmentAnyTooth/Yoon not public). **None of that is acquired here.**

Also still missing for a complete clinical ICDAS camera path: clinician-verified ICDAS 0–4 (or D0–D6 with 5/6 held out) on **tooth crops**. That is separate from this detection-role decision. Zenodo does not fill either gap.

---

## Recommended next step

**B. Acquire/verify a genuine whole-tooth + FDI RGB dataset first.**

The intended architecture fails without tooth instances and FDI. Preparing Zenodo (A) would only ready an optional lesion overlay; it cannot produce tooth crops or tooth IDs. **B is more valuable.** Do not search or download in this stage.

A remains valid **after** a tooth+FDI source exists, as a non-ICDAS overlay experiment—never as a replacement for tooth detection.

---

## Coexistence with the ICDAS classifier

Keep the Keras ICDAS path on `dataset/` unchanged. Add detectors later as **separate** models and folders. Inference order, when built: tooth+FDI crop → current ICDAS engine; lesion model optional and last.
