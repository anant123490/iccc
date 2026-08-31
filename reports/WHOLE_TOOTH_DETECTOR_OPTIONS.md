# Whole-tooth detector options (ICDAS v2 bottleneck)

Date: 2026-08-26  
Scope: **investigation only**. No downloads, no `pip install`, no inference, no training, no changes to the 420 images, `dataset/`, or old ICDAS checkpoints.

Question: can we **automatically crop whole teeth** from the **6,265** registered public RGB photos (and later the 420 personal photos) using an **openly downloadable** pretrained detector?

Constraint: **no** API keys, credentials, author email, DUA, Roboflow/Ultralytics hosted inference, or SegmentAnyTooth **restricted weights**. Lesion `d`/`D` and the in-repo caries YOLO **must not** be used as tooth boxes.

---

## Verdict

**No Rank A model exists under these rules.**

There is **no** verified, anonymously downloadable pretrained checkpoint that is documented as:

1. RGB / visible-light **intraoral photographs** (not OPG/CBCT), and  
2. **whole-tooth** instances (not caries/pathology/mucosa), and  
3. **local** `.pt` / ONNX with a **clear public URL** and **no access request**.

Local repo check: **no** `.pt` / `.onnx` files in the workspace tree. `models/caries_detector/best.pt` (if present on disk later) is **D/d decay**, not whole teeth.

**Do not download anything for this stage.**

---

## Rank definitions

| Rank | Meaning |
| --- | --- |
| **A** | RGB intraoral whole-tooth detector; weights public; local run; no auth |
| **B** | Would work after adaptation **and** weights are already public without permission |
| **C** | Wrong object, wrong modality, no public weights, or blocked by your access rules |

---

## Candidates

### 1. SegmentAnyTooth — Rank **C** (under current rules)

- Repo: https://github.com/thangngoc89/SegmentAnyTooth  
- Paper: https://doi.org/10.1016/j.jds.2025.01.003  
- RGB intraoral, five views, **whole-tooth YOLO11 boxes** + SAM masks, FDI in output (paper).  
- **Weights:** not on GitHub. README: sign NC PDF, **email** `hi+segmentanytooth@khoanguyen.me`.  
- Code: MIT. Weights: **non-commercial agreement**.  
- **Why C:** you forbade author contact / restricted access. Technically the best *documented* match if access rules change.

Would integrate with `data_icdas/` only **after** legal weights: YOLO xyxy → crop → `data_icdas/crops/` + `crop_pool.csv`, remap **all** classes to unlabeled tooth crops (not FDI ground truth, not ICDAS).

---

### 2. AlphaDent YOLOv8x-seg — Rank **C** (wrong objects)

- Repo: https://github.com/ZFTurbo/AlphaDent  
- Weights (public GitHub Releases, Apache 2.0):  
  - https://github.com/ZFTurbo/AlphaDent/releases/download/v1.0/yolov8x_AlphaDent_9_classes_640px.pt  
  - same tag: `…_960px.pt`, `…_4_classes_960px.pt`  
- RGB **DSLR intraoral** photos; instance **pathology** (Black caries 1–6, abrasion, filling, crown).  
- Inference (from README, **not run**):  
  `python3 inference.py --weights './weights/yolov8x_AlphaDent_9_classes_640px.pt' --input_path './AlphaDent/images/test/' --output_path './output/'`  
- Output: YOLO-seg instances of **pathology**, not whole crowns.  
- **Why C:** using it would repeat the `d`/`D` mistake. Do not send boxes to ICDAS Labeling Studio.

---

### 3. In-repo / Zenodo lesion YOLO (`d`/`D`) — Rank **C**

- Local: `data_external/detection/` annotations; `backend/app/caries_detector.py` (`models/caries_detector/best.pt`).  
- **Why C:** decay regions, not whole teeth. Forbidden.

---

### 4. Roboflow Universe intraoral “tooth” projects — Rank **C**

- Examples previously listed: `dental-cdueb/intraoral-tooth-detection-rohlq`, `dentalmate6v/intraoral-tooth-detection`.  
- Hosted inference / dataset download typically needs **API key** and account. Licenses often **UNVERIFIED**.  
- **Why C:** access rules + unverified RGB/whole-tooth/license.

---

### 5. Ultralytics Platform “dental” (Darshan Modi) — Rank **C** (dataset, not a public `.pt`)

- https://platform.ultralytics.com/darshan-modi/datasets/dental  
- Listing: **724** intraoral images, **7 classes** (incisor / canine / premolar / molar types), **15,935** boxes — *if true*, that is whole-tooth **data**, not a drop-in model.  
- Page shows dataset clone / “New Model”; **no anonymous weights URL** in the fetched page. Platform docs use **API keys** for train/download workflows.  
- **Why C:** not a verified public pretrained file; likely authentication. Do not guess a hosted model is downloadable.

---

### 6. Hugging Face `nsitnov/8024-yolov8-model` — Rank **C**

- https://huggingface.co/nsitnov/8024-yolov8-model  
- Card: **dental X-ray** instance segmentation (caries, crown, filling, implant, …).  
- **Why C:** radiographs + pathology, not RGB whole-tooth photos.

---

### 7. `sach3v/oral-yolo-dataset` / oral lesion YOLO — Rank **C**

- Smartphone **lesion** boxes, not teeth.  
- **Why C.**

---

### 8. Eeman1113/dental_study `best.pt` — Rank **C**

- https://github.com/Eeman1113/dental_study (`best.pt` linked in README).  
- Documented as **caries** detection on **occlusal** photos, not whole-tooth class `tooth`.  
- **Why C.** File was **not** downloaded.

---

### 9. BMC Oral Health 2025 occlusal tooth detection + numbering — Rank **C**

- https://doi.org/10.1186/s12903-025-05803-y  
- YOLOv8n on **occlusal photographs** (paper).  
- Data availability: **from authors on request**. No public `.pt`.  
- **Why C:** author contact + weights unpublished. Occlusal-only would not cover 5-view public set anyway.

---

### 10. TLNM / DigiLeap Mask R-CNN (smartphone tooth + FDI) — Rank **C**

- arXiv 2608.06275. Findata / restricted images.  
- **Why C:** DUA / not a public weight dump.

---

### 11. Sandeep-4469/dental-detector — Rank **C**

- Wrapper only: **“You must supply your own trained YOLOv8 weights.”**  
- **Why C:** no checkpoint.

---

### 12. Panoramic / CBCT (YoloTeeth, Zephinax pano-yolo, MIC-DKFZ ToothSeg, DENTEX, etc.) — Rank **C**

- Wrong modality for an intraoral-camera ICDAS crop pipeline.

---

### 13. TeethDreamer (MICCAI 2024) — Rank **C**

- https://github.com/ShanghaiTech-IMPACT/TeethDreamer  
- 3D reconstruction from **five** photos; segmentation script is **interactive click**; checkpoints on **SharePoint**.  
- **Why C:** not automatic whole-tooth boxes for 6,265 images; extra access friction.

---

### 14. Stock Ultralytics `yolov8n.pt` / `yolo11n.pt` (COCO) — Rank **C**

- Public, local, no auth — **not a tooth detector**. Previously rejected `yolo11n.pt` from an unaudited GitHub repo as COCO.  
- **Why C.**

---

### 15. KOUSHIK-9 Tooth-Detection-Using-Yolo — Rank **C**

- Prior audit: committed `yolo11n.pt` was stock COCO; trained `best.pt` unpublished. License null.

---

## If a suitable model existed (template — none filled)

Would have listed: official repo, weights URL, license, install, inference command, output format, `data_icdas/` mapping.

**None qualify as A.** Closest *technical* fit remains SegmentAnyTooth, which is **C until/unless** you allow the authors’ weight request.

AlphaDent weights **are** public (Apache 2.0) but are **unsuitable** for this bottleneck.

---

## Can 6,265 images be processed?

Any local YOLO nano/small could **run** on 6,265 JPGs on CPU/GPU in hours. That is irrelevant until the **object** is whole teeth. Do not batch-run AlphaDent, caries YOLO, or COCO YOLO to fill `data_icdas/crops/`.

---

## Local integration (when you *do* have a tooth detector)

Intended path (already stubbed, not implemented here):

`source_manifest.csv` → detector xyxy → copies under `data_icdas/crops/` → `python tools/register_icdas_crops.py` → Labeling Studio → `icdas_labels.csv`.

Crops must stay **candidates** until a dentist assigns ICDAS 0–4. FDI from a detector must not become ICDAS or FDI ground truth.

---

## Single best next action

**Draw whole-tooth boxes yourself on a small seed (Stage 3B Batch_01, 60 images), class `tooth` only**, export YOLO into `fdi_detection_dataset/annotations/yolo/` (or drop crop JPGs into `data_icdas/crops/`), then train **your** lightweight YOLO on that seed. That is the only path that stays inside open local tools and does not misuse `d`/`D` or gated/API models.

Do **not** train ICDAS until those tooth crops exist and are dentist-labelled 0–4.
