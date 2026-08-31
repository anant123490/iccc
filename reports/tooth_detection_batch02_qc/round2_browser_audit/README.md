# Batch 02 Round 2 browser audit

100 **new** KEEP images (none from Round 1). Overlays existing YOLO boxes only.

```text
.venv\Scripts\streamlit.exe run reports\tooth_detection_batch02_qc\round2_browser_audit\app.py --server.port 8503
```

URL: http://localhost:8503

Ratings: `manual_reviews_round2.json` (does not touch Round 1).  
Catalog: `../round2_catalog.json`  
After 100/100 ratings, copies (not moves) into `data/detection/batches/batch02_manual_round2/{good,questionable,bad}/{images,labels}/`.
