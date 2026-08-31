# Training Instructions

## ICDAS (blocked until pixels exist)

Edit `ml/configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_classes` | 5 | ICDAS 0–4 softmax |
| `use_attention` | cbam | cbam / se / none |
| `ordinal_regression` | false | Production uses softmax |
| `dataset_root` | data/icdas | Class folders 0–4 |
| `overwrite_root_checkpoints` | false | Do not auto-write current/deploy.keras |

```bash
python ml/train.py --config ml/configs/default.yaml
```

Outputs: `models/icdas/current/<experiment_name>/`

Do not train on auto-labeled YOLO crops. Do not overwrite `models/icdas/historical/stale_ordinal_4output/`.

## Tooth detector

Batch 01 is frozen by default:

```bash
python tools/train_tooth_detector_new_batch.py --batch 02
```

Retrain Batch 01 only with `--force-retrain-batch01` (archives the previous run first).
