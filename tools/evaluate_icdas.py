#!/usr/bin/env python3
"""Evaluate models/icdas_mobilenet_cbam_v2 if it exists. Does not train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "models" / "icdas" / "current" / "icdas_mobilenet_cbam_v2"
DATA = PROJECT / "data" / "icdas" / "labeling_v2" / "final"


def main():
    best = MODEL_DIR / "best.keras"
    test = DATA / "test"
    report = {
        "status": "NOT_RUN",
        "reason": None,
        "model": str(best),
        "test_dir": str(test),
        "metrics": None,
    }
    if not best.exists():
        report["reason"] = "NO_V2_CHECKPOINT — train only after dentist labels exist in data/icdas/labeling_v2/final"
    else:
        n = sum(1 for _ in test.rglob("*.jpg")) + sum(1 for _ in test.rglob("*.png"))
        if n == 0:
            report["reason"] = "EMPTY_TEST_SPLIT"
        else:
            report["reason"] = "Model exists; run python ml/train.py --config ml/configs/icdas_v2.yaml to refresh metrics.json"
            metrics = MODEL_DIR / "metrics.json"
            if metrics.exists():
                report["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
                report["status"] = "LOADED_EXISTING_METRICS"
    out = PROJECT / "reports" / "ICDAS_V2_RESULTS.md"
    lines = [
        "# ICDAS v2 evaluation",
        "",
        f"Status: **{report['status']}**",
        "",
        str(report.get("reason")),
        "",
        "Do not claim improvement over the existing ICDAS model unless metrics exist and are better.",
        "",
        "Existing `models/icdas/historical/` checkpoints were not overwritten.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
