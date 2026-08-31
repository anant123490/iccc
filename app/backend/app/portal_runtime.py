"""Load Tooth Detector V2 and an approved 5-class ICDAS model once.

Historical stale ordinal checkpoints are never used for production inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, get_settings
from .storage_paths import TOOTH_V2_WEIGHTS

logger = logging.getLogger("icdas.runtime")

ICDAS_NOT_DEPLOYED = (
    "ICDAS model has not yet been trained/deployed. "
    "Tooth Detector V2 can still run. No ICDAS grades, Grad-CAM, or Groq "
    "tooth findings were generated."
)

CURRENT_DEPLOY = PROJECT_ROOT / "models" / "icdas" / "current" / "deploy.keras"
CURRENT_BEST = PROJECT_ROOT / "models" / "icdas" / "current" / "best.keras"


def is_blocked_icdas_checkpoint(path: Path | str) -> bool:
    text = str(path).replace("\\", "/").lower()
    return "stale_ordinal" in text or "stale_ordinal_4output" in text


def resolve_icdas_checkpoint() -> Path | None:
    """Return an approved 5-class softmax checkpoint, or None.

    Never returns historical stale ordinal files.
    """
    settings = get_settings()
    ordered = [
        CURRENT_DEPLOY,
        CURRENT_BEST,
        Path(settings.deploy_model_path),
        Path(settings.model_path),
    ]
    seen: set[str] = set()
    for p in ordered:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if is_blocked_icdas_checkpoint(p):
            logger.warning("Ignoring blocked stale ICDAS checkpoint: %s", p)
            continue
        if p.exists() and p.is_file() and p.stat().st_size > 1000:
            return p
    return None


class PortalRuntime:
    def __init__(self) -> None:
        self.detector = None
        self.detector_error: str | None = None
        self.engine = None
        self.icdas_error: str | None = ICDAS_NOT_DEPLOYED
        self.icdas_kind: str = "NOT_TRAINED / NOT_DEPLOYED"
        self.icdas_path: str | None = None
        self.groq_ready: bool = False

    def detector_ok(self) -> bool:
        return self.detector is not None

    def icdas_ok(self) -> bool:
        return self.engine is not None

    def status(self) -> dict:
        from .groq_service import groq_configured
        from .tooth_detector import detector_error

        return {
            "detector_v2": self.detector_ok(),
            "detector_status": "AVAILABLE" if self.detector_ok() else "UNAVAILABLE",
            "detector_path": str(TOOTH_V2_WEIGHTS) if TOOTH_V2_WEIGHTS.exists() else None,
            "detector_error": self.detector_error or detector_error(),
            "icdas_loaded": self.icdas_ok(),
            "icdas_status": "DEPLOYED" if self.icdas_ok() else "NOT_TRAINED / NOT_DEPLOYED",
            "icdas_kind": self.icdas_kind,
            "icdas_path": self.icdas_path,
            "icdas_error": None if self.icdas_ok() else (self.icdas_error or ICDAS_NOT_DEPLOYED),
            "groq_configured": groq_configured(),
            "disclaimer": (
                "AI-assisted screening. Not a clinical diagnosis. ICDAS 5–6 out of scope."
            ),
        }

    def predict_crop_rgb(self, image_rgb: np.ndarray) -> dict:
        if self.engine is None:
            raise RuntimeError(self.icdas_error or ICDAS_NOT_DEPLOYED)
        original, processed = self.engine.preprocess_image(image_rgb)
        pred = self.engine.predict(processed)
        grade = int(pred["icdas_grade"])
        if grade < 0 or grade > 4:
            raise RuntimeError("Classifier returned an out-of-scope ICDAS grade.")
        return {
            "icdas_grade": grade,
            "confidence": float(pred["confidence"]),
            "probabilities": pred["probabilities"],
            "low_confidence": pred.get("low_confidence", False),
            "processed": processed,
            "original": original,
        }

    def explain_crop(self, pred: dict) -> dict:
        if self.engine is None:
            raise RuntimeError(self.icdas_error or ICDAS_NOT_DEPLOYED)
        grade = int(pred["icdas_grade"])
        return self.engine.explain(
            pred["processed"], pred["original"], predicted_grade=grade
        )


_RUNTIME: PortalRuntime | None = None


def reset_portal_runtime() -> None:
    """Drop cached engines so SET ACTIVE can pick up a new ICDAS keras file."""
    global _RUNTIME
    _RUNTIME = None


def load_portal_runtime() -> PortalRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    rt = PortalRuntime()
    from .groq_service import groq_configured
    from .tooth_detector import get_pipeline, detector_error

    try:
        rt.detector = get_pipeline()
        rt.detector_error = None if rt.detector else detector_error()
    except Exception as exc:
        rt.detector_error = str(exc)
        logger.exception("Tooth Detector V2 failed to load")

    ckpt = resolve_icdas_checkpoint()
    if ckpt is None:
        rt.engine = None
        rt.icdas_kind = "NOT_TRAINED / NOT_DEPLOYED"
        rt.icdas_error = ICDAS_NOT_DEPLOYED
        rt.icdas_path = None
        logger.warning(ICDAS_NOT_DEPLOYED)
    else:
        rt.icdas_path = str(ckpt)
        try:
            from .inference import InferenceEngine

            settings = get_settings()
            rt.engine = InferenceEngine(
                model_path=str(ckpt),
                num_classes=settings.num_classes,
                image_size=settings.image_size,
                ordinal_regression=False,
                confidence_threshold=settings.confidence_threshold,
            )
            rt.icdas_kind = "softmax_5class"
            rt.icdas_error = None
        except Exception as exc:
            rt.engine = None
            rt.icdas_kind = "NOT_TRAINED / NOT_DEPLOYED"
            rt.icdas_error = ICDAS_NOT_DEPLOYED + f" ({exc})"
            logger.exception("Approved ICDAS softmax model failed to load")

    rt.groq_ready = groq_configured()
    _RUNTIME = rt
    logger.info(
        "Portal runtime: detector=%s icdas=%s kind=%s",
        rt.detector_ok(),
        rt.icdas_ok(),
        rt.icdas_kind,
    )
    return rt


def get_runtime() -> PortalRuntime:
    return load_portal_runtime()
