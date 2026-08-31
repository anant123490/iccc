# Gold detector dataset

**Mode:** Dataset build + validation. **No training.** Existing datasets and models were not overwritten.

Location: `data/detection/gold_detector_dataset/`

## Contents

| Source | Role | Images | Boxes |
|--------|------|-------:|------:|
| Batch 01 `tooth_detector_batch01` | human GT | 60 | 767 |
| `batch02_manual_good` | Round 1 GOOD | 57 | 1415 |
| `batch02_manual_round2/good` | Round 2 GOOD | 59 | 1451 |
| Gold after dedup | copy | 176 | 3633 |

Excluded: QUESTIONABLE, BAD, unreviewed KEEP, detector predictions, ICDAS.

## Split (no source-stem leakage)

Seed 42. Batch 01 original test files stay in **test**; Batch 01 original val files stay in **valid**. Remaining unique stems were assigned to fill ~15% test, ~15% valid, rest train.

| Split | Images | % | Boxes | Batch 01 | R1 | R2 |
|-------|-------:|--:|------:|---------:|---:|---:|
| train | 124 | 70.5 | 2544 | 46 | 37 | 41 |
| valid | 26 | 14.8 | 577 | 6 | 14 | 6 |
| test | 26 | 14.8 | 512 | 8 | 6 | 12 |

Stem leakage across splits: **0** (must be 0).

## Validation

- Images that open + valid YOLO `0 xc yc w h` (class tooth, w/h>0, normalized): **176/176**
- Validation errors: **0**
- Tooth boxes in Gold: **3633**

## data.yaml

`nc: 1`, `names: [tooth]`, splits `train/images`, `valid/images`, `test/images`.

## Next

Do **not** train until you ask. Do **not** overwrite `models/detection/tooth_detector_batch01/`.

