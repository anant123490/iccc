# FDI mapping (template only)

**Status:** empty on purpose.

Stage 3A does **not** assign FDI numbers (11–18, 21–28, 31–38, 41–48).

FDI identification is a **separate** task from:

- whole-tooth detection (boxes)
- ICDAS 0–4 (caries severity)
- Zenodo `d` / `D` lesion detection

## Future mapping table (do not fill in Stage 3A)

| image_filename | annotation_id | tooth_box_id | fdi | annotator | date | notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

Rules when this table is used later:

1. Never infer FDI from lesion boxes.
2. Never map ICDAS scores to FDI.
3. Never invent left/right from an unverified mirrored photo.
4. Patient identifiers may be used only if already present in filenames; do not invent IDs.
