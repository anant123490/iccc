# Gold detector dataset — deduplication

**Mode:** Copy into a new folder only. Original Batch 01, Batch 02 CLEAN, GOOD folders, ICDAS, and models were not modified.

## Inputs (before unique filter)

| Source | Images | Boxes |
|--------|-------:|------:|
| Batch 01 human GT | 60 | 767 |
| Round 1 GOOD | 57 | 1415 |
| Round 2 GOOD | 59 | 1451 |
| **Total** | **176** | **3633** |

Expected: 60+57+59 = 176 images, 767+1415+1451 = 3633 boxes.

## Collision scan

| Check | Groups with >1 file |
|-------|--------------------:|
| Identical filename | 0 |
| Same source stem, different filenames | 0 |
| Identical image bytes (MD5) | 0 |

No identical filenames across sources.

No duplicate source stems (including Roboflow `.rf.` prefix) across the three GOOD/GT sets.

No identical image-byte duplicates.

## Removed from Gold (originals kept)

Dropped **0** candidate copies. Kept **176** unique images.

Nothing was dropped. All 176 candidates were unique by filename, source stem, and MD5.

## Final unique image count

**176** images, **3633** tooth boxes in `data/detection/gold_detector_dataset/`.

