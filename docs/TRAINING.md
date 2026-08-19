# Training Instructions

## Configuration

Edit `ml/configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_classes` | 5 | ICDAS 0–4 (4 ordinal thresholds) |
| `use_attention` | cbam | cbam / se / none |
| `ordinal_regression` | true | Cumulative link model |
| `k_folds` | 5 | Stratified K-Fold CV |
| `loss` | focal | focal / ordinal / weighted_ce |

## Train

```bash
cd ml
python train.py --config configs/default.yaml
python train.py --config configs/default.yaml --fold 0  # Single fold
```

## Export for Edge Deployment

```bash
python export.py --checkpoint ../models/best.keras --quantize
```

Outputs:
- `models/deploy.keras` — deployment model
- `models/model.tflite` — mobile TFLite
- `models/tfjs_model/` — PWA TensorFlow.js
- `models/export_report.json` — latency & size benchmarks

## Evaluation Metrics

After training, check `models/icdas_mobilenet_cbam/test_evaluation/`:
- `metrics.json` — accuracy, QWK, F1
- `confusion_matrix.png`
- `roc_curves.png`

## Weak Supervision

When ICDAS labels are unavailable:

```python
from src.advanced import weak_supervision_pseudo_labels
pseudo = weak_supervision_pseudo_labels(model, unlabeled_paths, confidence_threshold=0.9)
```

## Advanced Modules

See `ml/src/advanced.py`:
- Active learning sample selection
- Federated learning simulation (FedAvg)
- Temporal progression tracking
