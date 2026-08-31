# SegmentAnyTooth — technical and license readiness check

Date: 2026-08-26  
Repo inspected: https://github.com/thangngoc89/SegmentAnyTooth  
Sources: `README.md`, `LICENSE`, `pyproject.toml`, `segmentanytooth.py`, paper DOI `10.1016/j.jds.2025.01.003`.  
**The weight-agreement PDF was not downloaded and could not be text-extracted here.** The human must read `SegmentAnyTooth_license_agreement.pdf` before signing.

No weights downloaded. No packages installed. No inference. No ICDAS/FDI/annotation changes.

---

## FINAL DECISION

**READY_FOR_WEIGHT_REQUEST**

This means: it is reasonable for a **human** to complete the authors’ non-commercial agreement and **request** the weights. It does **not** mean the current Python environment can run SegmentAnyTooth today, and it does **not** authorize download/inference in this stage.

---

## 1. Exact pretrained weight names

From `get_model_path()` in `segmentanytooth.py`. Files live under `weight_dir` (README default `./weights`; code default `./weight`).

| Role | Filename on disk |
| --- | --- |
| upper occlusal YOLO | `segmentanytooth_yolo11_upper.pt` |
| lower occlusal YOLO | `segmentanytooth_yolo11_lower.pt` |
| right lateral YOLO | `segmentanytooth_yolo11_right.pt` |
| front YOLO | `segmentanytooth_yolo11_front.pt` |
| left lateral YOLO | **no separate file** — `view=="left"` **reuses** `segmentanytooth_yolo11_right.pt` after a horizontal flip |
| SAM | `segmentanytooth_vit_tiny.pt` |

Required unique files: **5** (upper, lower, right, front, SAM). “Left” is a routing/flip convention, not a sixth checkpoint.

---

## 2. Expected input views

`predict(..., view=)` accepts only:

`upper` | `lower` | `left` | `right` | `front`

Paper: five standard intraoral views — upper occlusal, lower occlusal, frontal, right lateral, left lateral. RGB photos (smartphone/DSLR). Training excluded primary/supernumerary teeth, dentures, most fixed appliances; laterals omitted second premolar through third molar.

---

## 3. Routing the 420 selected images

Stage 3B `mouth_view` is balanced **84 each**. Proposed map (filenames unchanged):

| Our `mouth_view` | SAT `view` | Count |
| --- | --- | ---: |
| Maxillary_Occlusal | `upper` | 84 |
| Mandibular | `lower` | 84 |
| Frontal | `front` | 84 |
| Left_Lateral | `left` | 84 |
| Right_Lateral | `right` | 84 |

**Reasonable to route.** Residual risk: Zenodo laterals/occlusals are not guaranteed to match SAT’s cropping, retractors, or “permanent-only” inclusion rules.

---

## 4. What YOLO detects

Ultralytics `YOLO.predict` on the view-specific YOLO11 checkpoint.

- Outputs **axis-aligned boxes** (`r.boxes.xyxy`) and a **class index** per box.
- Class **names** encode tooth identity (FDI). For `left`, names are replaced by `LEFT_CLASSES` (`le28`, `le11`, …). FDI is taken as `int(names[cls_id][-2:])`.
- **Object = whole tooth instance**, not caries/ICDAS.

---

## 5. What SAM produces

`sam_load(segmentanytooth_vit_tiny.pt)` then `sam_predict(sam, boxes_xyxy=boxes, image=RGB)`.

Per YOLO box: a binary mask. Masks are written into one `uint8` image.

---

## 6. How teeth are represented in the official output

`predict()` returns a **single semantic mask**: pixel value = **FDI number** (11–48 style), `0` = background. Overlapping masks: later FDI overwrites earlier pixels.

This project must **not** treat those integers as Stage 3E FDI labels.

---

## 7. Boxes from masks — technically valid?

**Yes, with QC**, for **instance** masks (one connected component per detection). Safer and aligned with the code: use **YOLO `xyxy` directly** (already whole-tooth boxes). Deriving boxes from the fused FDI mask can merge/split if two teeth share an FDI value or overlap.

Do **not** convert now.

---

## 8. YOLO / COCO / VOC

After human QC, boxes as `[x1,y1,x2,y2]` convert with existing `tools/coco_to_yolo.py`, `yolo_to_coco.py`, `voc_to_yolo.py` (reverse VOC write is straightforward). Normalized YOLO: `0 cx cy w h` with **class id 0 only**.

---

## 9–10. Detection class `0 = tooth`

**Confirmed: FDI must not be our detection class IDs.**

Intended mapping: every SAT instance → **one box, class `0` / `tooth`**. Drop or ignore `names` / mask pixel FDI for the detection dataset. Optional: store FDI only in a **non-training** debug column, never as YOLO class 11–48.

---

## 11. Licenses (several stacks)

| Component | What official docs say | Status |
| --- | --- | --- |
| SAT **code** | MIT (`LICENSE`, `pyproject.toml` `license = MIT`) | Documented |
| SAT **weights** | Non-commercial; `SegmentAnyTooth_license_agreement.pdf`; README: commercial use prohibited without permission | Documented NC; **PDF body not read here** |
| Code comment | “Refer to WEIGHTS_LICENSE.txt” | **File 404** on GitHub raw — PDF is the live agreement |
| **SAM** | Dependency `segment-anything-hq>=0.3` | **Package license not restated in SAT README** — treat as **UNKNOWN until the package LICENSE is read at install time** (do not install now) |
| **Ultralytics** | Required `ultralytics>=8.3.116` | Typical public license **AGPL-3.0** (Ultralytics docs); not restated in SAT README |
| **PyTorch / torchvision / numpy / opencv / timm** | Listed in `pyproject.toml` | Their own OSS licenses; not SAT-specific |
| Paper | CC BY-NC-ND 4.0 (publisher page) | Citation + NC-ND for the article text |

---

## 12. Weight license / agreement — summary of what *was* readable

From README + source header (not the PDF clauses):

- Weights are **separate** from MIT code.
- Released under **SegmentAnyTooth Non-Commercial License**.
- Obtain: **sign the PDF**, email **hi+segmentanytooth@khoanguyen.me**; authors send a download link on working days.
- **Commercial use of weights is prohibited** without explicit permission.
- Source header: no warranty language on weights beyond referring to the weight license.

**Human must open the PDF** for definitions of “non-commercial,” output rights, reverse-engineering, and redistribution. Those clauses are **not verified** in this file.

---

## 13. Student / academic / non-commercial research

**Appears consistent** with the README’s non-commercial weight terms for a university major project, **if** the signed PDF agrees and the course/product is not commercial deployment of those weights.

**Not legal advice.** Supervisor/TTO should confirm if the camera app could be treated as commercial.

---

## 14. What might prohibit uses

| Use | Appearance from docs |
| --- | --- |
| Academic research | Likely OK under NC weights after agreement |
| University project | Same, if non-commercial |
| Internal testing | Same |
| Candidate annotations on **our** Zenodo RGB copies | Likely OK as research use of the **model**; outputs still need human QC |
| Train **our** detector on **human-verified** boxes on **our** images | Generally the new labels are ours/Zenodo-derived; **do not** bake SAT weights into that training run; **do not** redistribute SAT `.pt` |
| Ship SAT weights in a product / GitHub | **Prohibited** without permission (README commercial ban + do not assume redistribution) |
| Using SAT FDI as clinical FDI | Out of project protocol (not a license issue) |

Gray area (not claimed prohibited): training on **uncorrected** SAT predictions as if they were original SAT IP. Prefer **human-edited** boxes.

---

## 15. Attribution / citation

**Yes.** README requests citation of Nguyen et al. 2025 (BibTeX in README). MIT also requires copyright notice if distributing **code**.

---

## 16. Redistribution of weights

**Do not assume yes.** README: NC; access is personal email link. **Do not commit `.pt` to git or public GitHub.**

---

## 17. Git / local storage

**Keep weights private/local. Gitignore them** (e.g. `**/segmentanytooth_*.pt`, `**/weight/`, `**/weights/`). Never copy into `models/` (ICDAS tree).

---

## 18. Documented technical dependencies

`pyproject.toml`:

- Python **`>=3.10,<3.11`**
- `numpy<2`
- `opencv-python>=4.11.0.86`
- `segment-anything-hq>=0.3`
- `timm>=0.9.2,<1`
- `torch>=2.7.0`, `torchvision>=0.22.0`
- `ultralytics>=8.3.116`

Plus repo modules `sam.py`, `utils.py` (not executed here).

---

## 19–20. Current environment (no installs)

Checked via import (default `python` on this machine):

| Package | Present | Version vs SAT pin |
| --- | --- | --- |
| ultralytics | yes | 8.4.45 (≥ 8.3.116) |
| torch | yes | **2.2.2+cpu** vs required **≥ 2.7.0** — **mismatch** |
| cv2 | yes | 4.11.0 (meets ≥ 4.11.0.86) |
| Python | **3.12.5** | SAT requires **3.10.x only** — **mismatch** |
| segment-anything-hq / timm | not checked beyond SAT extra deps | assume **missing** until a 3.10 venv is built later |

**Do not install now.** Inference needs a **separate Python 3.10** environment later.

---

## 22. Preprocessing of the 420 images

Official `predict()`: `cv2.imread` as-is; YOLO handles letterbox internally. Paper training size 1024. **No extra crop/normalize documented as mandatory.** Do **not** modify originals. Optional later: run from copies already in `fdi_detection_dataset/images/selected/`.

---

## 23–24. View routing

**Required.** Wrong `view` loads the wrong YOLO11 file (or skips the left-flip). Risks: missed teeth, boxes on gums/lips, left/right geometry errors, garbage FDI in the mask (even if we discard FDI). Laterals: SAT may not expect visible molars.

---

## 25. Candidate-annotation generator only

**Yes — that is the only appropriate use for Stage 3C.**  
Workflow: RGB → SAT candidates → boxes (prefer YOLO xyxy) → **class 0** → **human QC** → verified boxes → later FDI stage. Not GT. Not ICDAS. Not FDI finalization. Not training in this step.

---

## Technical readiness

| Item | Status |
| --- | --- |
| Weight filenames known | Yes |
| View map for 420 images | Yes |
| Box/mask → class 0 plan | Yes |
| Run SAT in **current** interpreter | **No** (Python 3.12; torch 2.2) |

## License readiness

| Item | Status |
| --- | --- |
| How to request weights | Yes (sign PDF, email) |
| Full PDF terms reviewed here | **No** — human reads PDF |
| Academic NC use | Appears permitted pending PDF |
| Redistribute weights | Assume **no** |

## Dependency readiness

**Not ready to run.** Ready only to **request** weights in parallel with a future 3.10 env.

---

## Mask/box and export plan (do not execute now)

1. Route image → SAT `view`.  
2. Run YOLO; take `xyxy` (primary). Optionally SAM masks for review overlays.  
3. Assign **class 0** to every box; **discard FDI** for detection files.  
4. Human QC (`TOOTH_ANNOTATION_GUIDELINES.md`, `ANNOTATION_QC_CHECKLIST.md`).  
5. Write YOLO/COCO/VOC only after QC (or clearly marked `candidate/`).

---

## Required human QC

Every image: missed/extra teeth, laterals/posteriors, overlaps, retractors, empty predictions, wrong-view failures. **Candidate ≠ verified.**

---

## Risks

Python/torch mismatch; NC + AGPL; unread PDF; view errors; training-set exclusions; semantic-mask overwrite; leaking FDI into YOLO classes; committing weights; treating SAT as clinical GT.

---

## Exact next action (human)

1. Open https://github.com/thangngoc89/SegmentAnyTooth/blob/main/SegmentAnyTooth_license_agreement.pdf and **read it**.  
2. If the university project fits the PDF: print/sign and email to **hi+segmentanytooth@khoanguyen.me**.  
3. Wait for the authors’ download link. **Do not** put weights in git.  
4. Later (new stage): Python **3.10** env matching `pyproject.toml`, then inference + QC — **not this check**.

**Do not download weights in this stage.**

---

**STOP.**
