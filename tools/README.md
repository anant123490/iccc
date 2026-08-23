# ICDAS dataset preparation toolkit

This toolkit prepares tooth crops for the **existing** ICDAS 0–4 trainer:

```text
python ml/train.py --config ml/configs/default.yaml
```

It does **not** replace MobileNetV3 + CBAM, FastAPI, or the Streamlit frontend.
It does **not** invent ICDAS grades from public-dataset class names.

## Workflow

```text
PUBLIC DATASET
      ↓
FULL MOUTH IMAGES
      ↓
EXISTING ANNOTATIONS  (boxes / regions only)
      ↓
python tools/crop_teeth.py
      ↓
cropped_teeth/images + cropped_teeth/crops.csv
      ↓
streamlit run tools/label_icdas.py   (human ICDAS 0–4)
      ↓
labels/labels.csv
      ↓
python tools/build_dataset.py
      ↓
dataset/train|val|test/0–4
      ↓
python tools/check_dataset.py
      ↓
python ml/train.py --config ml/configs/default.yaml
```

## Required folders

Scripts create these automatically (`mkdir(..., parents=True, exist_ok=True)`):

```text
dataset/train/{0,1,2,3,4}
dataset/val/{0,1,2,3,4}
dataset/test/{0,1,2,3,4}
cropped_teeth/images/
labels/
reports/
```

There is only **one** train/val/test tree: the existing `dataset/` directory used by `ml/train.py`.

## Raw dataset structure

Do not hardcode a drive letter. Pass the folder that contains the public images:

```powershell
python tools/crop_teeth.py --input "D:\ICDAS_DATASET" --output "cropped_teeth"
```

Typical layouts that work:

```text
RAW/
  images/*.jpg
  labels/*.txt          # YOLO
```

```text
RAW/
  JPEGImages/*.jpg
  Annotations/*.xml     # Pascal VOC
```

```text
RAW/
  *.jpg
  instances_default.json   # COCO
```

```text
RAW/
  *.jpg
  *.json                   # LabelMe (one JSON per image)
```

If images are present but annotations are not, the cropper reports the detected format and writes an empty `crops.csv`. It never downloads a dataset.

## Annotation format

`--format auto` inspects the folder and selects:

| Format | How it is recognized |
|--------|----------------------|
| YOLO | `.txt` labels (`class cx cy w h` normalized, or absolute xyxy) |
| COCO | JSON with `images` + `annotations` + `bbox` |
| Pascal VOC | `.xml` with `bndbox` |
| LabelMe | JSON with `shapes` |
| CSV | `--boxes-csv` with `filename,x1,y1,x2,y2[,annotation_class]` |

Override if needed:

```powershell
python tools/crop_teeth.py --input "D:\ICDAS_DATASET" --format coco --coco-json "annotations.json"
```

## Annotation meaning

Public labels are **region / detection classes** (tooth, caries box, etc.).

They are stored as `annotation_class` in `crops.csv`.

They are **not** automatically mapped to:

- ICDAS 0
- ICDAS 1
- ICDAS 2
- ICDAS 3
- ICDAS 4

ICDAS severity is assigned only by a person in `label_icdas.py`.

## Crop process

`tools/crop_teeth.py`:

- recursively finds images
- matches annotations
- pads and clamps boxes
- rejects invalid, tiny, blank, or extreme-aspect crops
- skips unreadable images (does not delete sources)
- writes unique `crop_id` files under `cropped_teeth/images/`
- writes `cropped_teeth/crops.csv`
- writes `reports/dataset_quality_report.csv`

`crops.csv` columns:

```text
crop_id,filename,source_image,annotation_id,annotation_class,x1,y1,x2,y2,width,height
```

Default crop size is 224×224 (training image size). Use `--keep-original-size` to keep the raw crop. Existing files are not overwritten unless you pass `--overwrite`.

## Labeling process

```powershell
streamlit run tools/label_icdas.py
```

- One crop at a time
- Buttons `[0] [1] [2] [3] [4]`
- Keys `0–4`, arrows, `S` skip, `U` undo
- Next / previous / skip / undo / resume
- Progress, labeled count, remaining count
- Edit existing labels (sidebar review mode)
- Saves immediately to `labels/labels.csv`

`labels.csv` columns:

```text
crop_id,filename,source_image,icdas_grade
```

Guidelines shown in the UI (labeling only):

- **ICDAS 0:** Sound tooth / no visible caries.
- **ICDAS 1:** First visual change in enamel.
- **ICDAS 2:** Distinct visual change in enamel.
- **ICDAS 3:** Localized enamel breakdown without visible dentin.
- **ICDAS 4:** Underlying dark shadow from dentin.

Disclaimer shown in the UI:

> ICDAS severity labels should be assigned or verified by a qualified dental professional. This tool does not provide a clinical diagnosis.

## Dataset generation

```powershell
python tools/build_dataset.py
```

- Reads `labels/labels.csv` only (human grades)
- Recreates `dataset/train|val|test/0–4`
- Split: **70% train / 15% val / 15% test**
- Fixed seed (`--seed 42`)
- Copies into the existing `dataset/` tree
- Default resize 224×224

## Train / val / test split and leakage

Crops from the **same source image** stay in the same split.

If a `patient_id` (or `patient` / `case_id`) column is present on the labels, the split is by patient instead.

That prevents the model from seeing another tooth from the same mouth in validation or test.

A copy of the assignment is written to `reports/split_manifest.csv`.

## Dataset validation

```powershell
python tools/check_dataset.py
```

Reports:

- total / train / val / test counts
- class 0–4 counts
- corrupt and missing images
- duplicate IDs and filenames
- source-image leakage
- class imbalance
- invalid labels
- missing class directories (creates them by default)

Source images are never deleted.

## Training command

After the checker looks reasonable:

```powershell
python ml/train.py --config ml/configs/default.yaml
```

Classes: `0 1 2 3 4`. Image size: `224`. Architecture stays MobileNetV3-Small + CBAM + ordinal regression.

## Install

```powershell
pip install -r tools/requirements.txt
```

These packages are already covered by the root `requirements.txt` (OpenCV, Pillow, numpy, pandas, Streamlit). The tools file is the minimal set for this pipeline.

## Git

Do not commit raw images, crops, or Keras weights. See the repository `.gitignore`.
