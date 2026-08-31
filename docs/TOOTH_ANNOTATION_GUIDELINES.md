# Whole-tooth bounding box annotation guidelines

**Task:** draw one rectangle per **visible tooth** on RGB intraoral photographs.  
**Class:** `tooth` only (id `0`).  
**Not this task:** FDI numbers, ICDAS 0–4, caries/lesion boxes, gums, tongue, lips, retractors, instruments.

Zenodo `d` / `D` labels are **lesions**. Do not copy or convert them.

---

## What to annotate

- The **entire visible crown** of each tooth that can be seen in the photo.
- **One box per visible tooth** (including third molars if the crown is visible).
- **Partial teeth** (cut by the frame): box the visible crown only.
- **Occluded teeth**: box if any crown is still identifiable; do not guess hidden anatomy.
- **Missing teeth**: **no** box. Do not mark the gap.

**Do not annotate**

- Gingiva / mucosa  
- Tongue  
- Lips / cheeks  
- Retractors, mirrors, gloves, cotton rolls  
- Isolated stain or cavity without boxing the tooth (this is not a lesion task)

---

## Box rules

- Use a **tight** axis-aligned rectangle around the visible crown.
- Include the occlusal/incisal edge and the visible mesial/distal extent of the **crown**.
- Do **not** expand the box to include large areas of gum or background “for safety.”
- Neighboring boxes **may overlap** where crowns overlap in the photo. Overlap is expected on laterals and crowded anteriors. Do not merge two teeth into one box.
- Do not draw one box around a whole arch.
- If a tooth is so blurred that you cannot tell it is a separate tooth, skip it and flag the image in notes.

---

## Difficult cases

| Situation | What to do |
| --- | --- |
| **Overlapping teeth** | Separate boxes; allow overlap. Prefer the visible crown of each tooth, not a combined blob. |
| **Braces / wires / brackets** | Still box the **tooth**. Include brackets if they sit on the crown; do not box only the wire across several teeth. |
| **Restorations / crowns / fillings** | Box the tooth as a unit. Metal/ceramic color does not change the class (`tooth`). |
| **Mirrors** | Ignore the mirror hardware. If a mirrored reflection duplicates teeth, annotate **anatomical teeth in the real arch**, not a second copy in the mirror, unless the assigned protocol says the mirror view is the intended subject (default: real arch only). |
| **Saliva / bubbles / flash** | Ignore glare. Box the crown through highlights if the tooth outline is still clear. |
| **Blur / motion** | If the crown outline is still clear, box it. If not, skip that tooth and mark the image for review. |
| **Primary vs permanent** | Both are class `tooth`. Do **not** add FDI or dentition subclass. |
| **Rotated / unusual camera angle** | Box what is visible. Do not “correct” orientation in the filename. Flag orientation issues on the QC form. |
| **Possible mirrored photo** | Do **not** assign left/right FDI. Only box teeth. Note “possible mirror” if suspected. |

---

## Batch workflow

Work through `annotation_batches/Batch_01` … `Batch_07` (60 images each). Image files stay in `fdi_detection_dataset/images/selected/`.

Export COCO or YOLO when a batch is done. Keep empty files for images with **zero** visible teeth (rare).
