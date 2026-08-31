# Batch 02 selection (human QC)

Selected **60** images from the **360** remaining after Batch_01.
Selection used YOLO candidate metadata (counts, confidence, overlap) plus view and patient diversity.
Predictions are **not** accepted as labels.

## Pool

- Remaining images scored: 360
- Batch_01 excluded: 60
- Target: 60 (12 per view)

## Batch_02 composition

| View | Count |
| --- | ---: |
| Frontal | 12 |
| Left_Lateral | 12 |
| Mandibular | 12 |
| Maxillary_Occlusal | 12 |
| Right_Lateral | 12 |

- Distinct clinic patient IDs: 50
- Pilot-style images: 10
- >24 detections: 2
- ≤12 detections: 14
- Mean YOLO detections in batch: 16.55
- Mean YOLO confidence in batch: 0.5032

## Must-include (>24 detections)

```
anonymous_003-007-1215-01_1732863751319_Left_Lateral_View.jpg  n=27  mean_conf=0.547
anonymous_003-008-647-01_1729163338710_Mandibular_View.jpg  n=25  mean_conf=0.508
```

## Packaging (originals not modified)

- Upload copies: `annotation_batches/Batch_02/seed_60/`
- CVAT names: `annotation_batches/Batch_02/cvat_upload_filenames.txt`
- Overlay review only: `annotation_batches/Batch_02/yolo_overlays_for_review/`
- Candidate txt (do not treat as GT): `annotation_batches/Batch_02/yolo_candidate_labels/`
- Prior round-robin list: `annotation_batches/Batch_02/stage3b_round_robin_archive/`

## Annotate from scratch

Same protocol as Batch_01: empty CVAT task, class `tooth` only, no FDI, no ICDAS.
Use overlays as a second-screen QC aid.

## Selected files

| filename | view | n_det | mean_conf | overlap | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `anonymous-frontalView-1727418713868.jpg` | Frontal | 12 | 0.538 | 0.114 | low_count_possible_misses;many_low_conf_boxes;pilot_naming_style;view:Frontal |
| `anonymous-frontalView-1727424461503.jpg` | Frontal | 19 | 0.569 | 0.319 | many_low_conf_boxes;overlapping_boxes;pilot_naming_style;view:Frontal |
| `anonymous-frontalView-1727768018238.jpg` | Frontal | 16 | 0.508 | 0.084 | many_low_conf_boxes;pilot_naming_style;view:Frontal |
| `anonymous_003-007-1012-00_1731052265486_Frontal_View.jpg` | Frontal | 21 | 0.653 | 0.257 | overlapping_boxes;view:Frontal |
| `anonymous_003-007-634-00_1729146043051_Frontal_View.jpg` | Frontal | 20 | 0.621 | 0.145 | many_low_conf_boxes;view:Frontal |
| `anonymous_003-007-642-00_1729154783455_Frontal_View.jpg` | Frontal | 19 | 0.582 | 0.138 | many_low_conf_boxes;view:Frontal |
| `anonymous_003-007-750-01_1729655314264_Frontal_View.jpg` | Frontal | 14 | 0.507 | 0.102 | many_low_conf_boxes;view:Frontal |
| `anonymous_003-007-752-01_1729658514991_Frontal_View.jpg` | Frontal | 17 | 0.642 | 0.306 | overlapping_boxes;view:Frontal |
| `anonymous_003-007-770-01_1729752599826_Frontal_View.jpg` | Frontal | 16 | 0.531 | 0.071 | many_low_conf_boxes;view:Frontal |
| `anonymous_003-007-771-00_1729753377062_Frontal_View.jpg` | Frontal | 17 | 0.496 | 0.144 | many_low_conf_boxes;view:Frontal |
| `anonymous_003-008-572-01_1728917641191_Frontal_View.jpg` | Frontal | 21 | 0.598 | 0.120 | view:Frontal |
| `anonymous_004-007-1027-01_1731221247182_Frontal_View.jpg` | Frontal | 11 | 0.694 | 0.000 | low_count_possible_misses;view:Frontal |
| `anonymous-leftLateralView-1726892814255.jpg` | Left_Lateral | 22 | 0.599 | 0.107 | pilot_naming_style;view:Left_Lateral |
| `anonymous_003-007-1011-01_1731051758521_Left_Lateral_View.jpg` | Left_Lateral | 11 | 0.556 | 0.296 | low_count_possible_misses;many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-1083-01_1732175835410_Left_Lateral_View.jpg` | Left_Lateral | 19 | 0.579 | 0.433 | many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-1100-00_1732262920724_Left_Lateral_View.jpg` | Left_Lateral | 22 | 0.582 | 0.509 | many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-1123-01_1732512071673_Left_Lateral_View.jpg` | Left_Lateral | 24 | 0.511 | 0.514 | many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-1215-01_1732863751319_Left_Lateral_View.jpg` | Left_Lateral | 27 | 0.547 | 0.351 | over_24_detections;many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-690-01_1729312765724_Left_Lateral_View.jpg` | Left_Lateral | 14 | 0.516 | 0.287 | many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-691-00_1729313275146_Left_Lateral_View.jpg` | Left_Lateral | 18 | 0.412 | 0.223 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-700-01_1729325366305_Left_Lateral_View.jpg` | Left_Lateral | 14 | 0.509 | 0.187 | many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-721-00_1729498293665_Left_Lateral_View.jpg` | Left_Lateral | 12 | 0.458 | 0.113 | low_count_possible_misses;low_mean_confidence;many_low_conf_boxes;view:Left_Lateral |
| `anonymous_003-007-873-00_1730266781026_Left_Lateral_View.jpg` | Left_Lateral | 9 | 0.596 | 0.154 | low_count_possible_misses;overlapping_boxes;view:Left_Lateral |
| `anonymous_003-007-987-00_1730958766991_Left_Lateral_View.jpg` | Left_Lateral | 19 | 0.563 | 0.509 | many_low_conf_boxes;overlapping_boxes;view:Left_Lateral |
| `anonymous-mandibularView-1727434026636.jpg` | Mandibular | 16 | 0.419 | 0.166 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;pilot_naming_style;view:Mandibular |
| `anonymous-mandibularView-1727703597109.jpg` | Mandibular | 9 | 0.529 | 0.136 | low_count_possible_misses;many_low_conf_boxes;pilot_naming_style;view:Mandibular |
| `anonymous-mandibularView-1727775921173.jpg` | Mandibular | 10 | 0.542 | 0.000 | low_count_possible_misses;pilot_naming_style;view:Mandibular |
| `anonymous_003-008-1052-00_1731329608354_Mandibular_View.jpg` | Mandibular | 14 | 0.484 | 0.093 | many_low_conf_boxes;view:Mandibular |
| `anonymous_003-008-1134-00_1732531184012_Mandibular_View.jpg` | Mandibular | 20 | 0.493 | 0.296 | many_low_conf_boxes;overlapping_boxes;view:Mandibular |
| `anonymous_003-008-1203-01_1732790749679_Mandibular_View.jpg` | Mandibular | 14 | 0.410 | 0.000 | low_mean_confidence;many_low_conf_boxes;view:Mandibular |
| `anonymous_003-008-1204-00_1732791234357_Mandibular_View.jpg` | Mandibular | 16 | 0.460 | 0.259 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Mandibular |
| `anonymous_003-008-1244-01_1732962596526_Mandibular_View.jpg` | Mandibular | 12 | 0.439 | 0.114 | low_count_possible_misses;low_mean_confidence;many_low_conf_boxes;view:Mandibular |
| `anonymous_003-008-623-00_1729088109108_Mandibular_View.jpg` | Mandibular | 19 | 0.445 | 0.204 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Mandibular |
| `anonymous_003-008-647-01_1729163338710_Mandibular_View.jpg` | Mandibular | 25 | 0.508 | 0.220 | over_24_detections;many_low_conf_boxes;overlapping_boxes;view:Mandibular |
| `anonymous_003-008-997-00_1730975206077_Mandibular_View.jpg` | Mandibular | 20 | 0.449 | 0.067 | low_mean_confidence;many_low_conf_boxes;view:Mandibular |
| `anonymous_003-008-999-00_1730981358006_Mandibular_View.jpg` | Mandibular | 18 | 0.461 | 0.253 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Mandibular |
| `anonymous-maxillaryView-1727682092188.jpg` | Maxillary_Occlusal | 19 | 0.402 | 0.202 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;pilot_naming_style;view:Maxillary_Occlusal |
| `anonymous-maxillaryView-1727768058789.jpg` | Maxillary_Occlusal | 9 | 0.425 | 0.155 | low_count_possible_misses;low_mean_confidence;many_low_conf_boxes;overlapping_boxes;pilot_naming_style;view:Maxillary_Occlusal |
| `anonymous_003-008-1183-01_1732710009967_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 19 | 0.471 | 0.356 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-1273-00_1733140554614_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 19 | 0.416 | 0.284 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-1288-01_1733308646255_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 21 | 0.511 | 0.066 | many_low_conf_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-1311-00_1733404330359_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 18 | 0.420 | 0.297 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-1345-01_1733567991677_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 15 | 0.374 | 0.079 | low_mean_confidence;many_low_conf_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-619-00_1729080900725_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 12 | 0.411 | 0.105 | low_count_possible_misses;low_mean_confidence;many_low_conf_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-620-01_1729083811620_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 21 | 0.490 | 0.148 | many_low_conf_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-621-00_1729084830092_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 14 | 0.357 | 0.183 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Maxillary_Occlusal |
| `anonymous_003-008-931-00_1730717738893_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 16 | 0.397 | 0.135 | low_mean_confidence;many_low_conf_boxes;view:Maxillary_Occlusal |
| `anonymous_003-009-704-01_1729339214008_Maxillary_Occlusal_View.jpg` | Maxillary_Occlusal | 16 | 0.384 | 0.168 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Maxillary_Occlusal |
| `anonymous-rightLateralView-1727761954744.jpg` | Right_Lateral | 10 | 0.462 | 0.000 | low_count_possible_misses;low_mean_confidence;many_low_conf_boxes;pilot_naming_style;view:Right_Lateral |
| `anonymous_003-007-1055-01_1731386400029_Right_Lateral_View.jpg` | Right_Lateral | 10 | 0.582 | 0.339 | low_count_possible_misses;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-007-1099-01_1732262197016_Right_Lateral_View.jpg` | Right_Lateral | 19 | 0.523 | 0.415 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-007-1150-01_1732607471437_Right_Lateral_View.jpg` | Right_Lateral | 15 | 0.490 | 0.447 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-007-670-01_1729238261312_Right_Lateral_View.jpg` | Right_Lateral | 10 | 0.453 | 0.000 | low_count_possible_misses;low_mean_confidence;many_low_conf_boxes;view:Right_Lateral |
| `anonymous_003-007-717-00_1729494912902_Right_Lateral_View.jpg` | Right_Lateral | 22 | 0.529 | 0.413 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-007-753-00_1729659446300_Right_Lateral_View.jpg` | Right_Lateral | 20 | 0.530 | 0.379 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-008-1253-00_1732971070859_Right_Lateral_View.jpg` | Right_Lateral | 17 | 0.383 | 0.282 | low_mean_confidence;many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-008-593-01_1728999384697_Right_Lateral_View.jpg` | Right_Lateral | 11 | 0.554 | 0.126 | low_count_possible_misses;many_low_conf_boxes;view:Right_Lateral |
| `anonymous_003-009-1194-00_1732775915152_Right_Lateral_View.jpg` | Right_Lateral | 16 | 0.591 | 0.514 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_003-009-707-00_1729343513251_Right_Lateral_View.jpg` | Right_Lateral | 16 | 0.537 | 0.551 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
| `anonymous_004-007-1321-00_1733473332768_Right_Lateral_View.jpg` | Right_Lateral | 21 | 0.490 | 0.460 | many_low_conf_boxes;overlapping_boxes;view:Right_Lateral |
