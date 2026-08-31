# Batch 02 (preparation)

External Roboflow *Intraoral Tooth Detection* v1, converted for YOLO **detection** (not Batch 01).

| Folder | Contents |
|--------|----------|
| `source_polygons/SOURCE.md` | Pointers to ZIP + extract. **No** polygon copies (originals preserved off-repo). |
| `yolo_detection/` | Copied 640×640 JPGs + **converted** 5-value YOLO boxes + `data.yaml` |

Do **not** merge into `fdi_detection_dataset/` or `models/detection/tooth_detector_batch01/`.

Do **not** train until `reports/BATCH02_TOOTH_DETECTION_QC_REPORT.md` cleanup items are addressed.

Convert again if needed:

```text
python tools/convert_tooth_polygons_to_yolo_boxes.py --src "%TEMP%\icdas_inspect_intraoral_tooth_detection_v1i" --dst "data/detection/batches/batch02/yolo_detection" --qc-dir "reports/tooth_detection_batch02_qc"
```

Omit `--copy-images` to write labels only.
