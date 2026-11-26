"""Load Tooth Detector V2 and an approved 5-class ICDAS model once.

Historical stale ordinal checkpoints are never used for production inference.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
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


@lru_cache(maxsize=16)
def _has_five_class_softmax(path_text: str, modified_ns: int, file_size: int) -> bool:
    """Load a candidate once and verify it has a five-output softmax head."""
    del modified_ns, file_size  # cache-key inputs invalidate the result on file changes
    try:
        ml_root = str(PROJECT_ROOT / "ml")
        if ml_root not in sys.path:
            sys.path.insert(0, ml_root)
        from src.model import get_custom_objects
        import tensorflow as tf

        model = tf.keras.models.load_model(
            path_text,
            compile=False,
            custom_objects=get_custom_objects(),
        )
        output_shape = model.output_shape
        if isinstance(output_shape, list) or not output_shape or output_shape[-1] != 5:
            return False
        activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", None)
        return activation == "softmax"
    except Exception:
        logger.exception("ICDAS checkpoint is not a loadable five-class softmax model: %s", path_text)
        return False


def is_valid_icdas_checkpoint(path: Path | str) -> bool:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size <= 1000 or is_blocked_icdas_checkpoint(candidate):
        return False
    resolved = str(candidate.resolve())
    stat = candidate.stat()
    return _has_five_class_softmax(resolved, stat.st_mtime_ns, stat.st_size)


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
        if p.exists() and is_valid_icdas_checkpoint(p):
            return p
    return None


def resolve_demo_icdas_checkpoint() -> Path | None:
    """Resolve an explicitly configured local demo model; never a promotion."""
    settings = get_settings()
    if not settings.local_demo_mode or not settings.demo_model_path.strip():
        return None
    path = Path(settings.demo_model_path.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if is_blocked_icdas_checkpoint(path) or not is_valid_icdas_checkpoint(path):
        logger.error("Configured ICDAS demo checkpoint is unavailable or invalid.")
        return None
    return path.resolve()


class PortalRuntime:
    def __init__(self) -> None:
        self.detector = None
        self.detector_error: str | None = None
        self.engine = None
        self.icdas_error: str | None = ICDAS_NOT_DEPLOYED
        self.icdas_kind: str = "NOT_TRAINED / NOT_DEPLOYED"
        self.icdas_path: str | None = None
        self.icdas_demo: bool = False
        self.prediction_mode: str = "NOT_DEPLOYED"
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
            "detector_error": self.detector_error or detector_error(),
            "icdas_loaded": self.icdas_ok(),
            "icdas_status": (
                "DEMO MODE — not production active"
                if self.icdas_ok() and self.icdas_demo
                else ("DEPLOYED" if self.icdas_ok() else "NOT_TRAINED / NOT_DEPLOYED")
            ),
            "icdas_kind": self.icdas_kind,
            "prediction_mode": self.prediction_mode,
            "icdas_demo_mode": self.icdas_demo,
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
    demo_mode = False
    if ckpt is None:
        ckpt = resolve_demo_icdas_checkpoint()
        demo_mode = ckpt is not None
    if ckpt is None:
        rt.engine = None
        rt.icdas_kind = "NOT_TRAINED / NOT_DEPLOYED"
        rt.icdas_error = ICDAS_NOT_DEPLOYED
        rt.icdas_path = None
        logger.warning(ICDAS_NOT_DEPLOYED)
    else:
        rt.icdas_path = str(ckpt)
        rt.icdas_demo = demo_mode
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
            rt.prediction_mode = "DEMO" if demo_mode else "PRODUCTION"
            rt.icdas_error = None
        except Exception as exc:
            rt.engine = None
            rt.icdas_kind = "NOT_TRAINED / NOT_DEPLOYED"
            rt.icdas_demo = False
            rt.prediction_mode = "NOT_DEPLOYED"
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
