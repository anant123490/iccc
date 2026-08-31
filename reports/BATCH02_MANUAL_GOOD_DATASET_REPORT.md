# Batch 02 manual QC analysis + GOOD training candidate

**Date:** 2026-08-27  
**Mode:** Analysis + copy of GOOD files only. **No training.** Batch 01, ICDAS, models, `batch02/yolo_detection/`, and `batch02_clean/` were **not** rewritten.

This folder is a **CANDIDATE** training subset, not a replacement for Batch 02 CLEAN.

---

## Part A — Manual review verification

Source: `reports/tooth_detection_batch02_qc/anterior_audit/manual_reviews.json`  
Catalog: `reports/tooth_detection_batch02_qc/anterior_audit/selection.json` (100 targeted images)

| Mark | Count | Matches claimed |
|------|------:|-----------------|
| GOOD | **57** | Yes |
| QUESTIONABLE | **26** | Yes |
| BAD | **17** | Yes |
| UNRATED | **0** | Yes |
| Ratings keys | 100 | Yes |
| Extra names not in catalog | 0 | |
| Catalog names missing a rating | 0 | |

**All 100 catalog images have a manual rating.**

Free-text notes: **0 of 100** had a non-empty `note`. “Recorded reason” below is therefore: sample bucket from the viewer + geometry measured on the **existing** YOLO labels (IoU, border, oversized vs median, filename cues). That does not change your GOOD/QUESTIONABLE/BAD marks.

---

## Part B — QUESTIONABLE (26) + BAD (17)

Problems are **not isolated**. On this targeted 100 (already biased toward overlap, crowding, borders, oversized boxes):

| Nature | Finding |
|--------|---------|
| Isolated | **No.** 43/100 failed (Q+BAD). Failures cluster. |
| Anterior-focused | **Yes, primary.** Overlapping AABB on crowded incisors/canines is the dominant pattern (24/43 tagged anterior crowding; 29/43 overlapping boxes). |
| Border-focused | **Secondary.** 9/43 have border-touching boxes; BAD includes close-up / flip / crop frames. |
| Conversion-related | **Minor.** Extreme-aspect AABBs from polygon→rectangle appear (2 Q). Mean polygon fill was already ~0.87 in the conversion QC. |
| Source/domain-related | **Minor.** 2 BAD files (`crop_open`, `flip_IMG_2188`) are cropped/flipped source variants. Roboflow 640×640 stretch remains a domain gap vs Batch 01 native photos. |
| Systematic | **Yes for anterior overlap.** True stacked duplicates (IoU ≥ 0.7) are **rare** (1 BAD). Neighbor-covering overlap (IoU 0.35–0.6) is common on anterior views. |

### Problem table

An image can have more than one tag. Counts are images, not boxes.

| Problem | Questionable | Bad | Total | Recommendation |
|---------|-------------:|----:|------:|----------------|
| Overlapping boxes | 17 | 12 | **29** | Do not treat as automatic discard. Tighten anterior AABBs or keep as known AABB-on-arch noise; do not use these boxes as ICDAS crop GT without review. |
| Anterior crowding | 16 | 8 | **24** | Same as overlap. Crowded incisors are the main crop-quality risk. |
| Border-truncated tooth | 5 | 4 | **9** | Keep for detector recall of edge teeth only if the crop would still be a single tooth; drop if the box is mostly off-frame. |
| Oversized box | 2 | 3 | **5** | Exclude from crop generation until split or shrunk. |
| Two teeth inside one box | 3 | 2 | **5** | Exclude from ICDAS crops; one box → two teeth would pollute the classifier. |
| Polygon-to-rectangle conversion issue | 2 | 0 | **2** | Expected AABB on rotated teeth; optional polygon-aware crop later. |
| Source-image issue | 0 | 2 | **2** | Do not mix `crop_` / `flip_` frames into the camera-domain train set without a domain decision. |
| Duplicate boxes | 0 | 1 | **1** | Drop or merge the IoU ≥ 0.7 pair (`IMG_6630`). |
| Wrong tooth association | 0 | 0 | **0** | Not evidenced (no notes; class is only `tooth`). |
| Excessive gum/palate/background | 0 | 0 | **0** | Not tagged as large+extreme together on Q/BAD; some oversized rows still include extra soft tissue. |
| Other | 0 | 0 | **0** | — |

### Every BAD image (17)

| Filename | Recorded reason |
|----------|-----------------|
| `OHI_datasetIMG_1771_JPG.rf.348aea523c987ae759539a3062ade4bf.jpg` | Manual **BAD**; no note; sample=`max_iou_fill`; overlapping boxes (max IoU 0.62) |
| `OHI_datasetIMG_2493_JPG.rf.129942024559016717f74909772bc8ef.jpg` | Manual **BAD**; sample=`overlap_pairs`; anterior crowding + overlapping boxes |
| `OHI_datasetIMG_3103_JPG.rf.5a75b8c86f3731c1e22d75ee55df2500.jpg` | Manual **BAD**; sample=`border_touching`; border-truncated |
| `OHI_datasetIMG_3146_JPG.rf.1f20cd371f0c4f3dafef9b61c1964e94.jpg` | Manual **BAD**; sample=`border_touching`; border + overlapping |
| `OHI_datasetIMG_3542_jpg.rf.171eeea429785b1497aba5eec2c9f3cf.jpg` | Manual **BAD**; sample=`overlap_pairs`; anterior crowding + overlapping |
| `OHI_datasetIMG_4455_JPG.rf.26c09664707f4daa885c56a7b6f13659.jpg` | Manual **BAD**; sample=`oversized_vs_neighbors`; oversized box |
| `OHI_datasetIMG_4697_JPG.rf.9616c0adb320cec4ae0b5b51b39bc64c.jpg` | Manual **BAD**; sample=`anterior_crowded`; anterior crowding + overlapping |
| `OHI_datasetIMG_4994_JPG.rf.7024d7ee59d6814505f09c8fad9ce09e.jpg` | Manual **BAD**; sample=`anterior_crowded`; anterior crowding + overlapping |
| `OHI_datasetIMG_5147_JPG.rf.6fec0ef1e4115e984dfcdb999b4b9858.jpg` | Manual **BAD**; sample=`oversized_vs_neighbors`; oversized + anterior crowding |
| `OHI_datasetIMG_5264_JPG.rf.bd06dc1b69bc857b62cece2cf72244d0.jpg` | Manual **BAD**; sample=`many_detections`; two-teeth-in-one-box candidate |
| `OHI_datasetIMG_5673_JPG.rf.4174895ce3952b2962d94e1eeb63dfc6.jpg` | Manual **BAD**; sample=`overlap_pairs`; two-teeth-in-one-box + overlapping |
| `OHI_datasetIMG_6064_JPG.rf.09a2a17f72d931a7d60838a9d033f523.jpg` | Manual **BAD**; sample=`max_iou_fill`; overlapping boxes |
| `OHI_datasetIMG_6630_JPG.rf.2d9a6068e169603c26d9e6b32575d940.jpg` | Manual **BAD**; sample=`high_iou_duplicate`; **duplicate** (IoU 0.71) + anterior overlap |
| `OHI_datasetIMG_6652_JPG.rf.363d6a2e583506ba1888c25c961529de.jpg` | Manual **BAD**; sample=`overlap_pairs`; overlapping boxes |
| `OHI_dataset_MG_3852_JPG.rf.2d94f42f28f1452dc8f4033c9ad58563.jpg` | Manual **BAD**; sample=`oversized_vs_neighbors`; oversized + anterior + overlapping |
| `OHI_datasetcrop_open_jpg.rf.2a01a46fb015a4d8f1626af2fad8b137.jpg` | Manual **BAD**; sample=`anterior_crowded`; border + crowding + **source crop_** |
| `OHI_datasetflip_IMG_2188_jpg.rf.6c9a71c0e9aaf424c68bb7addc2af3c8.jpg` | Manual **BAD**; sample=`border_touching`; border + **source flip_** |

### Every QUESTIONABLE image (26)

| Filename | Recorded reason |
|----------|-----------------|
| `OHI_datasetIMG_1845_JPG.rf.25750cf398d9b2588d8be216571aaaec.jpg` | Q; `anterior_crowded`; crowding + overlapping |
| `OHI_datasetIMG_23351_jpg.rf.778fd3654d07738fb96338d8f69d9230.jpg` | Q; `extreme_aspect`; border + conversion AABB |
| `OHI_datasetIMG_2335_JPG.rf.05072b41528cfa37c9fb8e61dafdf36e.jpg` | Q; `overlap_pairs`; overlapping |
| `OHI_datasetIMG_3048_JPG.rf.7b9f3c892b6240761127e844e35719ad.jpg` | Q; `anterior_crowded`; crowding + overlapping |
| `OHI_datasetIMG_3131_JPG.rf.125dc3548f629d163c07441d33da6c7a.jpg` | Q; `overlap_pairs`; crowding + overlapping |
| `OHI_datasetIMG_3400_JPG.rf.1e3930e2946f55768da1d48ed45822f6.jpg` | Q; `border_touching`; border-truncated |
| `OHI_datasetIMG_3448_jpg.rf.4d43e9bf5dbeaae842ed22062ce0a15c.jpg` | Q; `border_touching`; border-truncated |
| `OHI_datasetIMG_3942_JPG.rf.4e60403ee40365b985552a6211fe3751.jpg` | Q; `extreme_aspect`; conversion AABB |
| `OHI_datasetIMG_3993_JPG.rf.0a9b22bdc50782debaed048af8ae395d.jpg` | Q; `overlap_pairs`; crowding + overlapping |
| `OHI_datasetIMG_4233_JPG.rf.1765b214d5e4bf783c13f2eb615cd91e.jpg` | Q; `anterior_crowded`; crowding + overlapping |
| `OHI_datasetIMG_4648_JPG.rf.1d1708baf3978d5dd5d3a3ccb914ab60.jpg` | Q; `overlap_pairs`; crowding + overlapping |
| `OHI_datasetIMG_4753_JPG.rf.067ea770a01fad5a4a84dc00c04fc3ed.jpg` | Q; `overlap_pairs`; crowding + overlapping |
| `OHI_datasetIMG_4802_JPG.rf.2ee847e527da236cab8b718803678354.jpg` | Q; `many_detections`; overlapping |
| `OHI_datasetIMG_4809_JPG.rf.361917e6c7549e6f57afb66929d8640a.jpg` | Q; `anterior_crowded`; oversized + crowding + overlapping |
| `OHI_datasetIMG_4935_JPG.rf.9d8c2d158f77d03737fd136ca8fa549f.jpg` | Q; `anterior_crowded`; anterior crowding |
| `OHI_datasetIMG_4986_JPG.rf.d8ff496c3e5ee6b4e47efc6beb7a4c06.jpg` | Q; `overlap_pairs`; crowding + overlapping |
| `OHI_datasetIMG_5009_JPG.rf.6c18a412e43ac49f63f221ec0a0e5b51.jpg` | Q; `overlap_pairs`; crowding + overlapping |
| `OHI_datasetIMG_5055_JPG.rf.540dbe22c0d79b06d9c82e851b18def9.jpg` | Q; `anterior_crowded`; oversized + crowding + overlapping |
| `OHI_datasetIMG_5122_JPG.rf.25247f38a098e2bdbf5aa24a4125666a.jpg` | Q; `oversized_vs_neighbors`; two-teeth candidate + crowding |
| `OHI_datasetIMG_5188_JPG.rf.532920bdd622b9534c18834deafa5e19.jpg` | Q; `many_detections`; two-teeth candidate + overlapping |
| `OHI_datasetIMG_5192_JPG.rf.465a47a66317a2d29dad0d12aa891191.jpg` | Q; `many_detections`; anterior crowding |
| `OHI_datasetIMG_5201_JPG.rf.1dbeefdf555b186d9bc7764871df04cc.jpg` | Q; `tiny_packed`; crowding + overlapping |
| `OHI_datasetIMG_5296_JPG.rf.71a266c3369a94c72e6b7bf4cd3aca32.jpg` | Q; `anterior_crowded`; crowding + overlapping |
| `OHI_datasetIMG_5702_JPG.rf.34ac7be26abb30e7687ab484bf47fe0e.jpg` | Q; `overlap_pairs`; two-teeth candidate + overlapping |
| `OHI_datasetIMG_5834_JPG.rf.197c43f1a99ef1b241b1332b3a9bfdef.jpg` | Q; `border_touching`; border-truncated |
| `OHI_datasetIMG_6566_JPG.rf.0c32d811b9dd9233d4d286db7bce668f.jpg` | Q; `border_touching`; border-truncated |

---

## Part C — 57 GOOD images as training candidate

### Counts

| Item | Value |
|------|--------|
| GOOD images | **57** |
| Tooth boxes | **1415** |
| Splits (original CLEAN assignment) | train 44 / valid 9 / test 4 |
| Source | `data/detection/batches/batch02_clean/` (copy only) |
| Candidate location | `data/detection/batches/batch02_manual_good/` |
| Original filenames | **Preserved** |

### Annotation validation (all 57)

| Check | Result |
|-------|--------|
| Image file exists | **57/57** |
| Matching YOLO `.txt` exists | **57/57** |
| Class `0 = tooth` only | **Pass** |
| Normalized xc, yc, w, h in range | **Pass** |
| width > 0, height > 0 | **Pass** |
| Malformed rows | **0** |
| Invalid label files | **0** |

### Remaining suspicious geometry on GOOD (reviewer still marked GOOD)

These are **not** invalid labels. They are residual risk for ICDAS crops.

| Check | Images |
|-------|-------:|
| Duplicate boxes (IoU ≥ 0.7) | **0** |
| IoU ≥ 0.5 neighbor overlap (typical crowded AABB) | 20 |
| Box area > 2.5× image median (often molar vs incisor) | 18 |
| Extreme aspect > 5 | 13 |
| ≥3 border-touching boxes | **6** |
| Extreme oversized (area ≥ 0.08) | **1** (included in the 6) |

**Material remaining suspicious set (6):**

- `OHI_datasetIMG_2957_jpg.rf.1d202ba74d87a2db59d2d83b29dde43a.jpg` — largest box area 0.085, 7 border boxes  
- `OHI_datasetIMG_3497_JPG.rf.875ac9581b6f15d13ce22f4e454694d1.jpg`  
- `OHI_datasetIMG_5123_JPG.rf.4cd79561f80bacf2302f94ce8cfa5a09.jpg`  
- `OHI_datasetIMG_5284_JPG.rf.5513957e541e63585564bdd4f2c3d33b.jpg`  
- `OHI_datasetIMG_6094_JPG.rf.0eb3c9ba8a5e43ae512e2659d2c7c46f.jpg`  
- `OHI_datasetIMG_6161_JPG.rf.16769f0a6cd4bf53719e65670013acaa.jpg`  

**Suspicious labels (material): 6.** Invalid: **0.**

### Candidate dataset layout

```text
data/detection/batches/batch02_manual_good/
  images/{train,valid,test}/   # original names
  labels/{train,valid,test}/   # matching .txt, original names
  data.yaml                    # nc: 1, names: ['tooth']
  README.md
```

`data.yaml` also sets `train` / `val` / `test` so Ultralytics can load it later. It does **not** replace `batch02_clean/data.yaml`.

### All 57 GOOD filenames

1. `OHI_datasetIMG_01421_jpg.rf.5c7c7a8db8c7c504384541de0340a530.jpg`  
2. `OHI_datasetIMG_0281_JPG.rf.5246fd1b8883ddac184f832c9e67915d.jpg`  
3. `OHI_datasetIMG_0602_JPG.rf.13c9ea9fe1c17065d26a0896410c2d6f.jpg`  
4. `OHI_datasetIMG_1111_JPG.rf.c942055271ddc43953098ef41b84b5f5.jpg`  
5. `OHI_datasetIMG_1498_JPG.rf.3d609d133770e66f52d1d3f991f8fad1.jpg`  
6. `OHI_datasetIMG_1518_JPG.rf.16a3482cd27073c1cfec6eacf9f86a79.jpg`  
7. `OHI_datasetIMG_1553_JPG.rf.0109ee33adcf5decfc5cf27a8517419a.jpg`  
8. `OHI_datasetIMG_1746_JPG.rf.d600e314a6b016368ccf8dec0655e93f.jpg`  
9. `OHI_datasetIMG_1804_JPG.rf.a5431c958864445000dade59ac8534c2.jpg`  
10. `OHI_datasetIMG_2155_JPG.rf.3a7b797e515b3d1cec3d7596e0143e05.jpg`  
11. `OHI_datasetIMG_2207_JPG.rf.e1850a33e66a6267f7624b20534e3450.jpg`  
12. `OHI_datasetIMG_2347_jpg.rf.52fd84be01eaad771a7ac86e71b62902.jpg`  
13. `OHI_datasetIMG_2455_JPG.rf.07def7f4973124de390726d73e55b2c3.jpg`  
14. `OHI_datasetIMG_2481_JPG.rf.07214f38f17fc6384832d1a178baa259.jpg`  
15. `OHI_datasetIMG_2507_JPG.rf.373e796edec9827e66f8e581ecab71cf.jpg`  
16. `OHI_datasetIMG_2957_jpg.rf.1d202ba74d87a2db59d2d83b29dde43a.jpg`  
17. `OHI_datasetIMG_3098_JPG.rf.22421cbfb09f512558a9d502a01e6e70.jpg`  
18. `OHI_datasetIMG_3497_JPG.rf.875ac9581b6f15d13ce22f4e454694d1.jpg`  
19. `OHI_datasetIMG_3635_JPG.rf.4cb53362a045ca5ce999f68711c92ff3.jpg`  
20. `OHI_datasetIMG_3643_JPG.rf.b662b3d313159a24623fbbc3d1f79086.jpg`  
21. `OHI_datasetIMG_3690_JPG.rf.24a6e9c93e60e6d2ef0d6424aede2019.jpg`  
22. `OHI_datasetIMG_3773_jpg.rf.2d6bb88efe9142daecc81253e57392b7.jpg`  
23. `OHI_datasetIMG_3883_JPG.rf.3654a52a2a3d0768f68a46a55972752c.jpg`  
24. `OHI_datasetIMG_3937_JPG.rf.6e5e97f84ad85035fc407f442c187685.jpg`  
25. `OHI_datasetIMG_4015_JPG.rf.1e57c89894daae1f543c3ce451cd9d7a.jpg`  
26. `OHI_datasetIMG_4160_JPG.rf.33e47de79199dce8bbe81386f5bce9b6.jpg`  
27. `OHI_datasetIMG_4197_JPG.rf.ed04242005343e7beffd6cd73f2f5378.jpg`  
28. `OHI_datasetIMG_4276_JPG.rf.3f2c66e35bda0dcba3ea2c79b30f86b7.jpg`  
29. `OHI_datasetIMG_4517_JPG.rf.2327f0db52b62f6fdbb7641b8545fd7b.jpg`  
30. `OHI_datasetIMG_4569_JPG.rf.12624e4085e12bfcebdd5967d92eb129.jpg`  
31. `OHI_datasetIMG_4585_JPG.rf.174ce0a168bcf4cf7f42c6743aaaec5a.jpg`  
32. `OHI_datasetIMG_4682_JPG.rf.15590bc0ab6c50985b153f5b93553733.jpg`  
33. `OHI_datasetIMG_4685_JPG.rf.11c597fb7ef653f01a3a02a88e7ea886.jpg`  
34. `OHI_datasetIMG_4691_JPG.rf.6053b341018c66c9a1f79d99daf5989b.jpg`  
35. `OHI_datasetIMG_4732_JPG.rf.05c5680e08cf8f534f9f490e01a2de18.jpg`  
36. `OHI_datasetIMG_4741_JPG.rf.6080574cdf807e3aa4c836789fe89619.jpg`  
37. `OHI_datasetIMG_4892_JPG.rf.bd964b849dfcc4ed5c65ad461905c676.jpg`  
38. `OHI_datasetIMG_5002_JPG.rf.ec18fa71c12d2044eb636f1c791340d2.jpg`  
39. `OHI_datasetIMG_5111_JPG.rf.300eea1219317375d1caa437d7d7abb0.jpg`  
40. `OHI_datasetIMG_5123_JPG.rf.4cd79561f80bacf2302f94ce8cfa5a09.jpg`  
41. `OHI_datasetIMG_5150_JPG.rf.ecd0d45a0aa581f1d310e941438a87ee.jpg`  
42. `OHI_datasetIMG_5176_JPG.rf.5cca42d210ac66da38dbe7ab49adc470.jpg`  
43. `OHI_datasetIMG_5284_JPG.rf.5513957e541e63585564bdd4f2c3d33b.jpg`  
44. `OHI_datasetIMG_5285_JPG.rf.61b4841b7fb4b6fb3ae47fbd55eb2903.jpg`  
45. `OHI_datasetIMG_5831_JPG.rf.7b6876acf798d68093a0f38782a87863.jpg`  
46. `OHI_datasetIMG_5855_JPG.rf.11339478a9c00b3c44d07e1189cc9090.jpg`  
47. `OHI_datasetIMG_5884_JPG.rf.6095d76b8f785e4575c9db5bad87e9b8.jpg`  
48. `OHI_datasetIMG_6030_JPG.rf.1b3a38b20220bc664149f3e0039c5a92.jpg`  
49. `OHI_datasetIMG_6094_JPG.rf.0eb3c9ba8a5e43ae512e2659d2c7c46f.jpg`  
50. `OHI_datasetIMG_6158_JPG.rf.4b57c01bfae347ae38d6ae9b79438c4e.jpg`  
51. `OHI_datasetIMG_6161_JPG.rf.16769f0a6cd4bf53719e65670013acaa.jpg`  
52. `OHI_datasetIMG_6308_JPG.rf.9c15010b0f21cb9f5a244d771a46c0d1.jpg`  
53. `OHI_datasetIMG_6357_JPG.rf.079d014274331b8436adaee6ee9ef4bd.jpg`  
54. `OHI_datasetIMG_6637_JPG.rf.a046349ded325409318a4ac9b7016fb5.jpg`  
55. `OHI_dataset_MG_3889_JPG.rf.1e21792e6e4a29ef6d3f5596f5591a22.jpg`  
56. `OHI_dataset_MG_4765_JPG.rf.856662d380002ab196e4491f8488fe42.jpg`  
57. `OHI_datasetcrop_smile3_jpg.rf.042234afbffcb7c22456a94778d6e9a6.jpg`  

Machine-readable copy: `reports/tooth_detection_batch02_qc/anterior_audit/manual_qc_analysis.json`.

---

## Part D — Training

**No YOLO training was started.** Batch 01, Batch 02 original/clean, ICDAS, and existing models were not modified.

### How to use the 57 (A / B / C)

**B — keep as a separate supplemental training subset.**

- **Not A** (do not dump them into a combined Batch 01 + full Batch 02 CLEAN train set yet). These 57 are already inside CLEAN. Mixing the other 1,006 KEEP images would reintroduce the same anterior-overlap / oversized / border failures seen in the 43 Q+BAD. Batch 01 is also a different resolution/domain (native camera vs 640×640 Roboflow).
- **Not C.** The 57 passed human review, labels validate, and they are useful **gold** examples for a later detector whose boxes feed ICDAS 0–4 crops.
- **B** means: hold `batch02_manual_good` as a vetted supplement (fine-tune or high-weight gold), optionally drop the 6 border-heavy files before crop generation, and do **not** treat full `batch02_clean` as crop-safe until anterior boxes are cleaned or more images are reviewed.

### Final recommendation

**READY AFTER CORRECTION**

- The **57-image candidate** is **READY** as option **B** (small gold subset). It is **not** large enough by itself to train a strong whole-tooth detector for ICDAS crops.
- The **full Batch 02 KEEP set** is **not** ready for crop-quality training until anterior overlap / multi-tooth boxes are corrected or excluded. This 100 was a hard sample; 57% still passed, which is encouraging, but the 43% fail rate on crowded/overlap cases is systematic.

Next (when you ask): exclude the 17 BAD (and likely 26 Q) from any crop pipeline; consider a second visual pass on remaining KEEP; then train a **new** detector without overwriting Batch 01 weights.
