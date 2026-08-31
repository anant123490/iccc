#!/usr/bin/env python3
"""Audit on-disk ICDAS keras heads vs labeled dataset. Does not train or overwrite weights."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from src.attention import CBAM, ChannelAttention, SEBlock, SpatialAttention  # noqa: E402
from src.losses import ordinal_to_class_probabilities  # noqa: E402

CO = {
    "CBAM": CBAM,
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
    "SEBlock": SEBlock,
}


def count_disk_labels() -> dict:
    csv_path = ROOT / "data" / "icdas" / "annotations" / "annotations.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    by_split = Counter(r["split"] for r in rows)
    by_class = Counter(r["icdas_score"] for r in rows)
    disk = {}
    for split in ("train", "val", "test"):
        disk[split] = {}
        for c in range(5):
            d = ROOT / "data" / "icdas" / split / str(c)
            n = 0
            if d.exists():
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            disk[split][str(c)] = n
    v2 = {}
    for split in ("train", "val", "test"):
        v2[split] = {}
        for c in range(5):
            d = ROOT / "data" / "icdas" / "labeling_v2" / "final" / split / str(c)
            n = 0
            if d.exists():
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            v2[split][str(c)] = n
    return {
        "annotations_csv_rows": len(rows),
        "annotations_by_split": dict(by_split),
        "annotations_by_class": dict(by_class),
        "dataset_pixels_by_split_class": disk,
        "data_icdas_final_pixels": v2,
        "labeled_pixels_total": sum(sum(v.values()) for v in disk.values()),
    }


def inspect_keras(path: Path) -> dict:
    import tensorflow as tf

    if not path.exists():
        return {"path": str(path), "exists": False}
    m = tf.keras.models.load_model(str(path), compile=False, custom_objects=CO)
    units = int(m.output_shape[-1])
    names = list(getattr(m, "output_names", []) or [])
    if units == 5:
        kind = "softmax_5"
    elif units == 4:
        kind = "ordinal_4_thresholds_icdas_0_4"
    elif units == 6:
        kind = "ordinal_6_thresholds_or_7class_incomplete"
    elif units == 7:
        kind = "softmax_7_or_unsupported"
    else:
        kind = f"unknown_{units}"
    return {
        "path": str(path),
        "exists": True,
        "input_shape": list(m.input_shape),
        "output_shape": list(m.output_shape) if not isinstance(m.output_shape, list) else str(m.output_shape),
        "output_names": names,
        "output_units": units,
        "head_kind": kind,
        "n_layers": len(m.layers),
    }


def sample_ordinal_raw(n: int = 64) -> dict | None:
    import tensorflow as tf
    from ml.src.tooth_cropping import read_bgr
    from src.preprocessing import preprocess_image

    weights = (
        ROOT
        / "models"
        / "icdas"
        / "historical"
        / "stale_ordinal_4output"
        / "deploy.keras"
    )
    crops = sorted(
        (ROOT / "data" / "tooth_crops" / "generated" / "images").glob("*.jpg")
    )[:n]
    if not weights.exists() or not crops:
        return None
    m = tf.keras.models.load_model(str(weights), compile=False, custom_objects=CO)
    xs = []
    for p in crops:
        im = read_bgr(p)
        if im is None:
            continue
        xs.append(preprocess_image(im, target_size=224, use_roi=False, use_clahe=False, use_specular=False, color_norm=False))
    batch = np.stack(xs, axis=0)
    raw = np.asarray(m.predict(batch, verbose=0), dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    decoded = ordinal_to_class_probabilities(raw) if raw.shape[-1] == 4 else raw
    pred = decoded.argmax(axis=1)
    return {
        "n": int(raw.shape[0]),
        "raw_shape": list(raw.shape),
        "raw_mean_per_output": [float(x) for x in raw.mean(axis=0)],
        "raw_std_per_output": [float(x) for x in raw.std(axis=0)],
        "raw_min_per_output": [float(x) for x in raw.min(axis=0)],
        "raw_max_per_output": [float(x) for x in raw.max(axis=0)],
        "decoded_mean_per_class": [float(x) for x in decoded.mean(axis=0)],
        "decoded_pred_counts": {str(i): int((pred == i).sum()) for i in range(decoded.shape[-1])},
    }


def main() -> None:
    sys.path.insert(0, str(ROOT))
    out = {
        "labeled_dataset": count_disk_labels(),
        "models": {
            "deploy": inspect_keras(
                ROOT
                / "models"
                / "icdas"
                / "historical"
                / "stale_ordinal_4output"
                / "deploy.keras"
            ),
            "best": inspect_keras(
                ROOT
                / "models"
                / "icdas"
                / "historical"
                / "stale_ordinal_4output"
                / "best.keras"
            ),
        },
        "fold_config": None,
        "ordinal_sample_on_yolo_crops": None,
        "intended_by_default_yaml": {
            "ordinal_regression": False,
            "loss": "sparse_categorical_crossentropy",
            "num_classes": 5,
            "output": "softmax 5",
        },
        "historical_fold_config": json.loads(
            (
                ROOT
                / "models"
                / "icdas"
                / "historical"
                / "icdas_mobilenet_cbam"
                / "config.json"
            ).read_text(encoding="utf-8")
        )
        if (
            ROOT
            / "models"
            / "icdas"
            / "historical"
            / "icdas_mobilenet_cbam"
            / "config.json"
        ).exists()
        else None,
        "test_evaluation": {
            "status": "NOT_RUN",
            "reason": "No ICDAS 0–4 labeled pixels on disk under data/icdas/{train,val,test}/0–4.",
            "metrics": None,
        },
    }
    try:
        out["ordinal_sample_on_yolo_crops"] = sample_ordinal_raw(64)
    except Exception as exc:
        out["ordinal_sample_error"] = str(exc)
    dest = ROOT / "reports" / "icdas_classifier_audit.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("labeled_dataset", "models", "test_evaluation")}, indent=2)[:4000])
    print("wrote", dest)


if __name__ == "__main__":
    main()
