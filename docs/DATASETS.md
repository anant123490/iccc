# Dataset Integration

## Public Datasets

| Dataset | Labels | Access |
|---------|--------|--------|
| Mendeley Dental Caries | Binary/Multi | [Link](https://data.mendeley.com/datasets/5vb5tvkjb5/1) |
| Dental Caries Kaggle | Binary | Kaggle API |
| Tooth Segmentation | Segmentation | Various GitHub repos |
| Oral Disease Classification | Disease class | Academic request |

> **Note:** Few public datasets have full ICDAS 0-6 labels. Use weak supervision or expert annotation.

## Folder Structure

```
dataset/
├── train/0/ ... train/6/
├── val/0/   ... val/6/
├── test/0/  ... test/6/
├── raw/                    # Downloads
└── annotations.csv
```

## annotations.csv

```csv
filename,icdas_score,split,patient_id,notes
train/2/sample_001.jpg,2,train,P042,distinct visual change
val/4/sample_002.jpg,4,val,P042,dentin visible
```

**Important:** If `annotations.csv` exists but is older than your folders, training may ignore new images. After copying images into `train/val/test/<0-6>/`, always refresh the CSV:

```bash
python ml/scripts/sync_annotations.py
```

The loader uses folder paths automatically when folders contain more images than the CSV lists for that split.

## Download Scripts

```bash
python scripts/download_datasets.py --dataset dental_caries
python scripts/preprocess_dataset.py --input dataset/raw --output dataset
```

## Weak Supervision Pipeline

1. Train binary caries detector on public data
2. Run `weak_supervision_pseudo_labels()` on unlabeled images
3. Expert review high-confidence pseudo-labels
4. Retrain full ICDAS ordinal model

## ICDAS Labeling Guide

| Score | Criteria |
|-------|----------|
| 0 | Sound surface |
| 1 | First visual change |
| 2 | Distinct visual change in enamel |
| 3 | Localized enamel breakdown |
| 4 | Underlying dentin shadow |
| 5 | Distinct cavity with dentin |
| 6 | Extensive distinct cavity |
