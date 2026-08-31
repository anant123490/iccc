# Dataset Directory

Place intraoral images in class subfolders **0–4**.

ICDAS 5 and 6 images must **not** be copied into class 4. Keep them under `excluded/` if retained.

```
dataset/
├── train/
│   ├── 0/   # ICDAS 0 - sound
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/ ... 4/
├── test/
│   ├── 0/ ... 4/
├── excluded/     # out-of-scope grades (not remapped)
├── raw/
└── annotations.csv
```
