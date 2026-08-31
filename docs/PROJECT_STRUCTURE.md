# Project structure

CCC AI Dentist Camera 2.0. FDI numbering is out of scope.

```text
PROJECT_ROOT/
├── app/
│   ├── backend/              FastAPI (moved from backend/)
│   └── frontend/             Streamlit (moved from misspelled fronted/)
├── crm_backend/              Separate CRM; not wired to camera API (left in place)
├── ml/
│   ├── train.py              ICDAS trainer (Docker uses this path)
│   ├── configs/              default.yaml = 5-class softmax
│   ├── src/                  model, dataset, Grad-CAM, tooth_cropping
│   ├── tests/
│   ├── detection/            pointers to tools for YOLO train/infer/eval
│   └── icdas/                pointers to ICDAS train/infer/eval
├── data/
│   ├── detection/            FUTURE full-mouth images + boxes (Batch 01 not flattened here)
│   ├── tooth_crops/          generated detector crops (not ICDAS GT)
│   └── icdas/                clinician ICDAS 0–4 tooth images + CSV
├── fdi_detection_dataset/    CANONICAL Batch 01 RGB + YOLO split (DO NOT RENAME YET)
├── annotation_batches/       Human QC batches (code still references this path)
├── annotation_project/       CVAT / Label Studio configs
├── models/
│   ├── detection/            tooth_detector_batch01 + pretrained yolo11n
│   └── icdas/
│       ├── current/          empty until a softmax model is approved
│       └── historical/       including STALE ordinal deploy.keras
├── tools/                    CLIs (detect, crop, label, audit)
│   └── ingest/               former scripts/
├── docker/                   Dockerfiles
├── docs/
├── reports/
├── tests/                    index only; pytest lives under ml/ and app/backend/
├── archive/
│   ├── historical/
│   ├── experiments/
│   ├── obsolete/
│   ├── out_of_scope/fdi/
│   └── review_required/
├── docker-compose.yml
└── README.md
```

## Intentionally left in place (high risk)

| Path | Why |
|------|-----|
| `fdi_detection_dataset/` | Hardcoded in YOLO yaml, cropper, annotation JSON |
| `annotation_batches/`, `annotation_project/` | Tools and docs join this path |
| `crm_backend/` | Standalone app; product ownership unclear |
| `ml/.git/` | Nested git repo — do not move or delete |
| `assets/` | Directory junction; gitignored |
| `data_external/` | Gitignored Zenodo dump |
| `app/backend/app/caries_*.py` | Live FastAPI imports |

## Two datasets (never mix)

1. **Detection** — full RGB mouths + tooth rectangles → `fdi_detection_dataset/` (Batch 01) and `data/detection/` (new)
2. **ICDAS** — single-tooth images + grades 0–4 → `data/icdas/`

Generated crops sit in the middle: `data/tooth_crops/generated/`.
