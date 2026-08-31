# Annotation QC checklist (whole-tooth detection)

Use after each batch. **Do not check FDI** in this stage.

Image / batch: _____________  Annotator: _____________  Reviewer: _____________

## Completeness

- [ ] Every **visible** tooth has exactly one box  
- [ ] Missing teeth have **no** box  
- [ ] Teeth clipped by the image border are boxed (visible crown only)  
- [ ] No arch-level “one box for all teeth”

## Box quality

- [ ] Boxes are tight on the crown (not large gum/tongue/lip padding)  
- [ ] No box on retractors, mirrors, gloves, or tongue alone  
- [ ] Overlapping crowns have overlapping boxes rather than a merged box  
- [ ] No duplicate boxes on the same tooth  
- [ ] No empty/tiny accidental clicks  
- [ ] No boxes on obvious non-teeth (burs, calculus-only specks without a crown)

## Image-level

- [ ] Image is the intended RGB photo (not a blank/wrong file)  
- [ ] Orientation looks plausible (frontal / occlusal / left / right); if wrong, note it — **do not relabel as FDI**  
- [ ] Severe blur/exposure: image flagged for `images/review` rather than forced labeling  
- [ ] Class name is `tooth` only (no extra classes)

## Export

- [ ] Saved in COCO and/or YOLO  
- [ ] Empty label file left empty if zero teeth  
- [ ] Lesion (`d`/`D`) files were **not** mixed into this export  

Notes:
________________________________________________________________
