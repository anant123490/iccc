#!/usr/bin/env python3
"""Build an active-learning review queue. Never saves predictions as ICDAS labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    CROP_POOL_COLUMNS,
    LABEL_COLUMNS,
    PROJECT_ROOT,
    crop_pool_path,
    labels_path,
    load_csv,
)


def main():
    pool = load_csv(crop_pool_path(), CROP_POOL_COLUMNS)
    labels = load_csv(labels_path(), LABEL_COLUMNS)
    labelled_ids = set(labels["sample_id"].astype(str)) if not labels.empty else set()
    unlabeled = pool[~pool["sample_id"].astype(str).isin(labelled_ids)] if not pool.empty else pool
    out = PROJECT_ROOT / "reports" / "icdas_active_learning_queue.csv"
    model = PROJECT_ROOT / "models" / "icdas_mobilenet_cbam_v2" / "best.keras"
    if unlabeled.empty:
        pd.DataFrame(columns=["sample_id", "crop_path", "confidence", "note"]).to_csv(out, index=False)
        print("EMPTY_POOL")
        return
    if not model.exists():
        unlabeled.assign(
            confidence="",
            note="NO_V2_MODEL — show unlabeled crops in Labeling Studio; do not auto-label",
        )[["sample_id", "crop_path", "confidence", "note"]].to_csv(out, index=False)
        print(f"wrote {out} without model scores")
        return
    print("MODEL_PRESENT: inference-on-unlabeled is not auto-run (predictions must not become labels).")
    print(f"Unlabeled crops: {len(unlabeled)}. Label them in the studio.")
    unlabeled.assign(confidence="", note="awaiting_human").to_csv(out, index=False)


if __name__ == "__main__":
    main()
