# Stage 2A dataset report

Inspection date: 2026-08-26  
Repository: ICCC local workspace  
No application code was changed. Existing `models/deploy.keras` and `models/best.keras` were not overwritten. No training was run. No ICDAS grades were invented or remapped from caries/disease labels.

## Datasets discovered

| Dataset | URL | License | Modality | Images (published) | Annotations | Boxes | FDI | Genuine ICDAS 0–4 | Appropriate for ICCC? |
|---------|-----|---------|----------|--------------------|-------------|-------|-----|-------------------|------------------------|
| Ghost index in this repo | `dataset/annotations.csv` | n/a (local) | Intraoral crop filenames | 643 listed, **0 on disk** | folder-style `icdas_score` 0–4 | No | No | **Claimed in CSV only; pixels missing** | Not usable until files are restored |
| Local excluded 5/6 | `dataset/excluded/` | n/a | Intraoral / mixed | **16 on disk** | folder 5 or 6 | No | No | No (grades 5–6, out of scope) | Do not train 0–4; do not remap to 4 |
| Odontify Clean ICDAS V2 | https://www.kaggle.com/datasets/leonardoaranguiz/odontify-clean-icdas-dataset-v2 | CC BY-NC-SA 4.0 (listing) | Tooth photos (claimed) | Listing: 3010 files; folder `0/` preview 982 | Folders `0/`–`6/` **parsed from filenames** | No | No | **Unverified.** ICDAS is claimed from filenames, not inspected here | Possible ICDAS source **after** Kaggle login + clinician review; **not downloaded** (no Kaggle CLI/credentials) |
| Roboflow “Caries Classification ICDAS II” | https://universe.roboflow.com/caries-detection-zmhsz/caries-classification-icdas-ii-ewijy | CC BY 4.0 (listing) | Object detection photos | ~1.9k | 4 classes: Healthy / Initial / Moderate / Extensive | Boxes | No | **No.** Name says ICDAS II; classes are **not** 0,1,2,3,4 | Do not map to ICDAS 0–4 |
| AlphaDent | https://github.com/ZFTurbo/AlphaDent ; zip https://zenodo.org/records/16582489 ; HF https://huggingface.co/datasets/ZFTurbo/AlphaDent | Apache 2.0 (repo LICENSE); dataset ~4.9 GB | Intraoral DSLR | ~1320 images, 295 patients | YOLO instance **masks**; 9 pathology classes | Yes (from masks/polygons) | No | **No.** Caries classes are **Black’s 1–6 (location)**, not ICDAS severity | **DETECTION / pathology only.** Not downloaded (4.9 GB; wrong label ontology) |
| Scientific Data / Zenodo intraoral caries | https://zenodo.org/records/14769743 ; paper https://doi.org/10.1038/s41597-025-05647-9 | Record **restricted**; paper text also cites CC BY-NC-ND | Intraoral mobile photos | 6313 | YOLO/COCO/VOC/LabelMe; decay `d`/`D` | Yes | No | **No** (binary primary/permanent decay) | DETECTION only; files not publicly downloadable without login |
| SegmentAnyTooth | https://github.com/thangngoc89/SegmentAnyTooth | Code MIT; **weights NC**; **images not public** | Intraoral 5 views | 5000 (paper) | FDI + surfaces | Yes (paper) | Yes (paper) | No | Cannot download images |
| Roboflow “teeth detection and numbering” | https://universe.roboflow.com/prime-snf1v/teeth-detection-and-numbering-agi2i | MIT (listing) | Object detection (dental photos) | ~1.4k | 32 FDI-style classes | Yes | Yes (listing) | No | DETECTION + FDI; Roboflow page blocked/API; **not downloaded** |
| Roboflow / other panoramic FDI | e.g. https://universe.roboflow.com/dentalxray-yjztn/panoramic-dental-xray-fdi | CC BY 4.0 (listing) | **Panoramic X-ray** | varies | FDI boxes | Yes | Yes | No | Wrong modality for camera app |
| BMC Oral Health HI Bogi ICDAS photos | https://doi.org/10.1186/s12903-025-07486-x | Paper open access; **data “from corresponding authors”** | Intraoral JPG | 3221 | ICDAS D0–D6 boxes, dentist+expert review | Yes | Not stated | **Yes in the paper**, not in a public dump | Request from authors; not downloaded |
| Mendeley (repo `download_datasets.py`) | https://data.mendeley.com/datasets/5vb5tvkjb5/1 | not re-verified this session | typically caries detection | unknown here | not ICDAS 0–4 in project docs | unknown | No | **Do not assume ICDAS** | Script only prints a URL; nothing downloaded |
| Kaggle oral-disease placeholder | https://www.kaggle.com/datasets/oral-disease | unknown | unknown | unknown | disease classes | unknown | No | No | Broken/placeholder URL in repo |

**Genuine public ICDAS 0–4 pixels were not obtained.** Filename-folder ICDAS (Odontify) and author-request ICDAS (BMC) were not accessed. Roboflow “ICDAS II” is a **4-bucket** scheme, not ICDAS 0–4.

## Datasets downloaded

**None.**  

Reasons: no unattended public dump of verified ICDAS 0–4; AlphaDent is the best **open intraoral box/mask** set but is **not ICDAS/FDI** and is **4.9 GB**; Roboflow/Kaggle/Zenodo restricted records need accounts; this stage does not invent labels.

Original downloads would have been stored under `dataset/raw/` (already gitignored). That folder has no new archives.

## Local inspection (this machine)

Script: `python tools/inspect_downloaded_dataset.py --root dataset --annotations-csv dataset/annotations.csv`

| Metric | Value |
|--------|--------|
| Images on disk | **16** |
| Corrupted | **0** |
| Duplicate MD5 groups | **0** |
| Folders | `excluded/5` = 11, `excluded/6` = 5 |
| `dataset/train\|val\|test` | **do not exist** |
| `cropped_teeth` images | **0** |
| `labels/labels.csv` | **absent** |
| `annotations.csv` rows | **643** |
| CSV files present on disk | **0** |
| CSV files missing | **643** |
| CSV class counts | 0:76, 1:145, 2:118, 3:121, 4:183 |
| CSV splits | train 440, val 110, test 93 |
| CSV `patient_id` | **column absent** |

CSV names look like **tooth crops** from occlusal/mandibular/maxillary views. That is filename evidence only.

## Exact labels (what exists vs what ICCC needs)

ICCC trainer (`ml/src/dataset.py`) requires **on-disk** `dataset/{train,val,test}/{0,1,2,3,4}/` and treats the **folder name** as ICDAS 0–4. It does not read `annotations.csv`.

Public sets above that have boxes are **DETECTION** (or Black’s caries location, or binary decay). Mapping those to ICDAS 0–4 was **not** done.

## Class counts for training 0–4

**Train/val/test 0–4 on disk: 0 / 0 / 0.**  
No leakage-safe split was built because there were no ICDAS 0–4 images to split.

## Preprocessing performed

None (no public ICDAS set extracted into `dataset/train|val|test`).

## Existing tooling vs recommended data

| Tool | Can use later |
|------|----------------|
| `tools/crop_teeth.py` | YOLO/COCO/VOC/LabelMe/**CSV boxes** → crops. Does **not** assign ICDAS. Fits Roboflow FDI or AlphaDent polygons-as-boxes **after** download. |
| `tools/label_icdas.py` | Human ICDAS 0–4 on crops. Required for any detection-only download. |
| `tools/build_dataset.py` | 70/15/15; split by `patient_id` if present, else `source_image`. |
| `tools/check_dataset.py` | Layout / leakage checks. |

## Limitations

- Cannot train a 5-class softmax ICDAS model from public pixels obtained in this stage.
- Current `.keras` files remain **4-output ordinal** and must not be used as production softmax.
- Training later would write `models/best.keras` and `models/deploy.keras`; copy those files first.
- Odontify ICDAS-from-filename still needs Kaggle access **and** clinical verification.
- AlphaDent “Caries 1–6” is **not** ICDAS 1–6.

## Suitability for ICDAS 0–4 training

**Not suitable.** No verified ICDAS 0–4 image set is on disk.

## Next action (not done here)

1. Restore the missing 643 crops if they exist elsewhere, **or** obtain Odontify via Kaggle and **review** grades, **or** request the BMC ICDAS photo set from the authors.  
2. Keep 5/6 out of class 4.  
3. Only then run `validate_dataset` / `train.py` with softmax yaml.  
4. Detection/FDI: download AlphaDent (Apache, large) or Roboflow FDI (MIT) into `dataset/raw/` in a later stage; run `crop_teeth.py` + human `label_icdas.py` if those crops should become ICDAS data.
