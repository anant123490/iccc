"""Portal business logic. ML stays here / in detectors — not in Streamlit."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from .config import PROJECT_ROOT, get_settings
from .groq_service import generate_screening_report
from .image_quality import assess_image_quality
from .portal_db import (
    ClinicalReport,
    DatasetVersion,
    ICDASPrediction,
    ModelVersion,
    Patient,
    ToothCrop,
    ToothDetection,
    TrainingImage,
    TrainingJob,
    TrainingLabel,
    UploadedImage,
    Visit,
)
from .storage_paths import (
    HEATMAPS,
    PATIENT_ROOT,
    REPORTS_PATIENT,
    TOOTH_V1_WEIGHTS,
    TOOTH_V2_WEIGHTS,
    TRAIN_DETECTED,
    TRAIN_LABELED,
    TRAIN_UPLOADS,
    TRAIN_VERSIONS,
    ensure_dirs,
)
from .tooth_detector import detect_rgb, detector_available, detector_error

logger = logging.getLogger("icdas.portal")

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_BYTES = 12 * 1024 * 1024
DISCLAIMER = (
    "AI-assisted screening tool. This is not a clinical diagnosis. "
    "A licensed dentist should examine the patient. ICDAS 5 and 6 are out of scope."
)


def _b64_rgb(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


def _decode_upload(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise ValueError("Only JPG, JPEG, and PNG are accepted.")
    if len(data) > MAX_BYTES:
        raise ValueError("File exceeds size limit (12 MB).")
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Image could not be decoded.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    enc.tofile(str(path))


def _read_rgb(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _public_id() -> str:
    return "P" + uuid.uuid4().hex[:10].upper()


def seed_model_registry(db: Session) -> None:
    if db.query(ModelVersion).count() > 0:
        return
    v2_metrics = TOOTH_V2_WEIGHTS.parent.parent / "eval_metrics.json"
    metrics = v2_metrics.read_text(encoding="utf-8") if v2_metrics.exists() else None
    rows = [
        ModelVersion(
            name="Tooth Detector V1 (Batch 01)",
            kind="detection",
            path=str(TOOTH_V1_WEIGHTS),
            dataset_version="batch01",
            metrics_json=None,
            is_active=False,
        ),
        ModelVersion(
            name="Tooth Detector V2",
            kind="detection",
            path=str(TOOTH_V2_WEIGHTS),
            dataset_version="gold_detector_dataset",
            metrics_json=metrics,
            is_active=True,
        ),
        ModelVersion(
            name="ICDAS historical 5-class (not production deploy)",
            kind="icdas",
            path=str(PROJECT_ROOT / "models" / "icdas" / "historical"),
            dataset_version="historical",
            metrics_json=None,
            is_active=False,
        ),
    ]
    db.add_all(rows)
    db.commit()


def create_patient(db: Session, payload: dict) -> Patient:
    ensure_dirs()
    public_id = (payload.get("public_id") or "").strip() or _public_id()
    existing = db.query(Patient).filter(Patient.public_id == public_id).first()
    if existing:
        return existing
    patient = Patient(
        public_id=public_id,
        name=str(payload.get("name") or "Unknown").strip()[:256],
        age=payload.get("age"),
        gender=payload.get("gender"),
        phone=payload.get("phone"),
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def create_visit(
    db: Session,
    patient: Patient,
    visit_date: str,
    notes: str | None,
    test_only: bool = False,
) -> Visit:
    visit = Visit(
        patient_id=patient.id,
        visit_date=visit_date or datetime.utcnow().strftime("%Y-%m-%d"),
        notes=notes,
        test_only=test_only,
    )
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit


def patient_to_dict(p: Patient) -> dict:
    return {
        "id": p.id,
        "public_id": p.public_id,
        "name": p.name,
        "age": p.age,
        "gender": p.gender,
        "phone": p.phone,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def _visit_image_dirs(visit_id: int, image_id: int) -> dict[str, Path]:
    root = PATIENT_ROOT / f"visit_{visit_id}" / f"image_{image_id}"
    dirs = {
        "original": root / "original",
        "overlay": root / "overlay",
        "crops": root / "crops",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def patient_stage(grade: int, lang: str = "en") -> dict:
    from .groq_dentist_prompt import ICDAS_PLAIN

    pack = ICDAS_PLAIN.get(lang) or ICDAS_PLAIN["en"]
    info = pack[int(np.clip(grade, 0, 4))]
    return {
        "icdas_grade": int(grade),
        "current_stage": info["stage"],
        "explanation": info["explanation"],
        "priority": info["priority"],
        "next_step": info["next_step"],
    }


def store_visit_image(
    db: Session,
    visit: Visit,
    data: bytes,
    filename: str,
) -> tuple[UploadedImage, dict]:
    ensure_dirs()
    rgb = _decode_upload(data, filename)
    digest = _hash(data)
    dup = (
        db.query(UploadedImage)
        .filter(UploadedImage.visit_id == visit.id, UploadedImage.content_hash == digest)
        .first()
    )
    quality = assess_image_quality(rgb)
    if not dup:
        rec = UploadedImage(
            visit_id=visit.id,
            filename=Path(filename).name,
            path="pending",
            content_hash=digest,
            quality_json=json.dumps(quality),
        )
        db.add(rec)
        db.flush()
        dirs = _visit_image_dirs(visit.id, rec.id)
        dest = dirs["original"] / Path(filename).name
        _write_rgb(dest, rgb)
        rec.path = str(dest)
        db.commit()
        db.refresh(rec)
    else:
        rec = dup
        quality = json.loads(rec.quality_json) if rec.quality_json else quality
        rgb = _read_rgb(rec.path) or rgb
    return rec, {
        "image_id": rec.id,
        "duplicate": bool(dup),
        "filename": rec.filename,
        "quality": quality,
        "preview_base64": _b64_rgb(rgb),
        "disclaimer": DISCLAIMER,
    }


def attach_detection_warnings(quality: dict, n_kept: int, items=None, image_shape=None) -> dict:
    q = dict(quality)
    warnings = list(q.get("warnings") or [])
    if n_kept == 0:
        warnings.append("Teeth partially visible")
        if q.get("verdict") == "PASS":
            q["verdict"] = "WARNING"
            q["status"] = "low_quality"
            q["ok"] = False
            q["message"] = (q.get("message") or "") + " Detector found no usable tooth crops."
    elif n_kept < 6:
        warnings.append("Teeth partially visible")
        if q.get("verdict") == "PASS":
            q["verdict"] = "WARNING"
            q["status"] = "low_quality"
            q["message"] = (q.get("message") or "") + " Few teeth detected."
    if items and image_shape is not None and n_kept > 0:
        h, w = image_shape[:2]
        if w > 0:
            xs = [((it.x1 + it.x2) / 2.0) / w for it in items]
            mean_x = sum(xs) / len(xs)
            if mean_x < 0.22 or mean_x > 0.78:
                warnings.append("Mouth not centered")
                if q.get("verdict") == "PASS":
                    q["verdict"] = "WARNING"
                    q["status"] = "low_quality"
    q["warnings"] = warnings
    return q


def run_detection_for_image(db: Session, image: UploadedImage) -> dict:
    if not detector_available():
        raise RuntimeError(detector_error() or "Tooth Detector V2 unavailable.")
    rgb = _read_rgb(image.path)
    if rgb is None:
        raise RuntimeError("Stored image could not be read.")
    det = detect_rgb(rgb, source_name=image.filename)
    quality = json.loads(image.quality_json) if image.quality_json else assess_image_quality(rgb)
    quality = attach_detection_warnings(quality, det["n_kept"], det["items"], rgb.shape)
    overlay = det["overlay_rgb"] if det["overlay_rgb"] is not None else rgb
    dirs = _visit_image_dirs(image.visit_id, image.id)
    ov_path = dirs["overlay"] / "overlay.jpg"
    _write_rgb(ov_path, overlay)
    image.overlay_path = str(ov_path)
    image.quality_json = json.dumps(quality)
    det_ids = [
        d.id
        for d in db.query(ToothDetection).filter(ToothDetection.image_id == image.id).all()
    ]
    if det_ids:
        crop_ids = [
            c.id for c in db.query(ToothCrop).filter(ToothCrop.detection_id.in_(det_ids)).all()
        ]
        if crop_ids:
            db.query(ICDASPrediction).filter(ICDASPrediction.crop_id.in_(crop_ids)).delete(
                synchronize_session=False
            )
            db.query(ToothCrop).filter(ToothCrop.id.in_(crop_ids)).delete(synchronize_session=False)
        db.query(ToothDetection).filter(ToothDetection.id.in_(det_ids)).delete(
            synchronize_session=False
        )
    db.commit()
    crop_payload = []
    for idx, item in enumerate(det["items"]):
        crop_rgb = det["crops"][idx][1]
        crop_path = dirs["crops"] / f"tooth_{idx:03d}.jpg"
        _write_rgb(crop_path, crop_rgb)
        td = ToothDetection(
            image_id=image.id,
            index_in_image=idx,
            confidence=float(item.confidence),
            x1=item.x1,
            y1=item.y1,
            x2=item.x2,
            y2=item.y2,
        )
        db.add(td)
        db.flush()
        db.add(ToothCrop(detection_id=td.id, path=str(crop_path)))
        crop_payload.append(
            {
                "index": idx,
                "confidence": item.confidence,
                "box": {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2},
                "crop_base64": _b64_rgb(crop_rgb),
            }
        )
    db.commit()
    return {
        "image_id": image.id,
        "teeth_detected": det["n_kept"],
        "raw_boxes": det["n_raw"],
        "mean_confidence": det["mean_confidence"],
        "quality": quality,
        "original_base64": _b64_rgb(rgb),
        "overlay_base64": _b64_rgb(overlay),
        "crops": crop_payload,
        "disclaimer": DISCLAIMER,
    }


def classify_visit_crops(
    db: Session,
    visit: Visit,
    engine=None,
    include_heatmaps: bool = True,
    persist: bool = True,
    language: str = "en",
) -> dict:
    from .portal_runtime import get_runtime

    runtime = get_runtime()
    if engine is None and not runtime.icdas_ok():
        from .portal_runtime import ICDAS_NOT_DEPLOYED

        raise RuntimeError(runtime.icdas_error or ICDAS_NOT_DEPLOYED)
    teeth = []
    for image in visit.images:
        for det in image.detections:
            crop = det.crop
            if crop is None:
                continue
            rgb = _read_rgb(crop.path)
            if rgb is None:
                continue
            pred = runtime.predict_crop_rgb(rgb)
            grade = int(pred["icdas_grade"])
            heatmap_b64 = overlay_b64 = None
            hm_path = ov_path = None
            if include_heatmaps:
                try:
                    exp = runtime.explain_crop(pred)
                    heatmap_b64 = exp.get("heatmap")
                    overlay_b64 = exp.get("overlay")
                    if persist and heatmap_b64:
                        hm_path = HEATMAPS / f"crop_{crop.id}_heatmap.png"
                        hm_path.parent.mkdir(parents=True, exist_ok=True)
                        hm_path.write_bytes(base64.b64decode(heatmap_b64))
                    if persist and overlay_b64:
                        ov_path = HEATMAPS / f"crop_{crop.id}_overlay.png"
                        ov_path.write_bytes(base64.b64decode(overlay_b64))
                except Exception as exc:
                    logger.exception("Grad-CAM failed for crop %s: %s", crop.id, exc)
            if persist:
                row = (
                    db.query(ICDASPrediction).filter(ICDASPrediction.crop_id == crop.id).first()
                )
                if row is None:
                    row = ICDASPrediction(
                        crop_id=crop.id, grade=grade, confidence=0.0, probabilities_json="{}"
                    )
                    db.add(row)
                row.grade = grade
                row.confidence = float(pred["confidence"])
                row.probabilities_json = json.dumps(pred["probabilities"])
                row.heatmap_path = str(hm_path) if hm_path else None
                row.overlay_path = str(ov_path) if ov_path else None
            stage = patient_stage(grade, language)
            teeth.append(
                {
                    "crop_id": crop.id,
                    "image_id": image.id,
                    "index": det.index_in_image,
                    "tooth_id": f"T{int(det.index_in_image) + 1:02d}",
                    "detection_confidence": det.confidence,
                    "icdas_grade": grade,
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "label": stage["current_stage"],
                    **stage,
                    "crop_base64": _b64_rgb(rgb),
                    "heatmap_base64": heatmap_b64,
                    "overlay_base64": overlay_b64,
                    "low_confidence": pred.get("low_confidence", False),
                }
            )
    if persist:
        db.commit()
    counts = Counter(int(t["icdas_grade"]) for t in teeth)
    dist = {str(i): int(counts.get(i, 0)) for i in range(5)}
    high = [t["index"] for t in teeth if t["icdas_grade"] >= 3]
    grades = [int(t["icdas_grade"]) for t in teeth]
    summary = {
        "teeth_detected": sum(len(im.detections) for im in visit.images),
        "teeth_analyzed": len(teeth),
        "healthy_teeth": dist["0"],
        "icdas_distribution": dist,
        "highest_grade": max(grades) if grades else None,
        "average_confidence": (
            round(sum(t["confidence"] for t in teeth) / len(teeth), 2) if teeth else 0.0
        ),
    }
    return {
        "visit_id": visit.id,
        "test_only": bool(getattr(visit, "test_only", False)),
        **summary,
        "icdas_distribution": dist,
        "high_severity_indices": high,
        "mean_confidence": summary["average_confidence"],
        "summary": summary,
        "teeth": teeth,
        "disclaimer": DISCLAIMER,
    }


def analysis_from_stored(visit: Visit, language: str = "en") -> dict | None:
    teeth = []
    for image in visit.images:
        for det in image.detections:
            crop = det.crop
            if crop is None or crop.prediction is None:
                continue
            row = crop.prediction
            rgb = _read_rgb(crop.path) if Path(crop.path).exists() else None
            hm = None
            ov = None
            if row.heatmap_path and Path(row.heatmap_path).exists():
                hm = base64.b64encode(Path(row.heatmap_path).read_bytes()).decode("ascii")
            if row.overlay_path and Path(row.overlay_path).exists():
                ov = base64.b64encode(Path(row.overlay_path).read_bytes()).decode("ascii")
            stage = patient_stage(int(row.grade), language)
            teeth.append(
                {
                    "crop_id": crop.id,
                    "image_id": image.id,
                    "index": det.index_in_image,
                    "tooth_id": f"T{int(det.index_in_image) + 1:02d}",
                    "detection_confidence": det.confidence,
                    "icdas_grade": int(row.grade),
                    "confidence": float(row.confidence),
                    "probabilities": json.loads(row.probabilities_json or "{}"),
                    **stage,
                    "label": stage["current_stage"],
                    "crop_base64": _b64_rgb(rgb),
                    "heatmap_base64": hm,
                    "overlay_base64": ov,
                    "low_confidence": float(row.confidence) < 55,
                }
            )
    if not teeth:
        return None
    counts = Counter(int(t["icdas_grade"]) for t in teeth)
    dist = {str(i): int(counts.get(i, 0)) for i in range(5)}
    grades = [int(t["icdas_grade"]) for t in teeth]
    summary = {
        "teeth_detected": sum(len(im.detections) for im in visit.images),
        "teeth_analyzed": len(teeth),
        "healthy_teeth": dist["0"],
        "icdas_distribution": dist,
        "highest_grade": max(grades) if grades else None,
        "average_confidence": round(sum(t["confidence"] for t in teeth) / len(teeth), 2),
    }
    return {
        "visit_id": visit.id,
        "test_only": bool(getattr(visit, "test_only", False)),
        **summary,
        "high_severity_indices": [t["index"] for t in teeth if t["icdas_grade"] >= 3],
        "mean_confidence": summary["average_confidence"],
        "summary": summary,
        "teeth": teeth,
        "disclaimer": DISCLAIMER,
    }


def visit_quality_summary(visit: Visit) -> list[dict]:
    out = []
    for im in visit.images:
        q = json.loads(im.quality_json) if im.quality_json else {}
        out.append({"image_id": im.id, "filename": im.filename, "quality": q})
    return out


def structured_for_groq(patient: Patient, visit: Visit, analysis: dict) -> dict:
    teeth = []
    for t in analysis.get("teeth") or []:
        idx = int(t.get("index") or 0) + 1
        teeth.append(
            {
                "tooth_id": f"T{idx:02d}",
                "crop_index": t.get("index"),
                "icdas_grade": t["icdas_grade"],
                "confidence": t["confidence"],
                "current_stage": t.get("current_stage") or patient_stage(int(t["icdas_grade"]))["current_stage"],
            }
        )
    return {
        "patient": {
            "patient_id": patient.public_id,
            "public_id": patient.public_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "visit_date": visit.visit_date,
        },
        "image_quality": visit_quality_summary(visit),
        "teeth_detected": analysis.get("teeth_detected"),
        "teeth_analyzed": analysis.get("teeth_analyzed"),
        "icdas_counts": analysis.get("icdas_distribution"),
        "high_severity_teeth": analysis.get("high_severity_indices"),
        "mean_confidence": analysis.get("mean_confidence"),
        "teeth": teeth,
        "rules": {
            "icdas_source": "MobileNetV3+CBAM classifier",
            "groq_must_not_change_grades": True,
            "icdas_5_6_out_of_scope": True,
            "no_fdi": True,
        },
    }


def save_clinical_report(
    db: Session,
    visit: Visit,
    language: str,
    structured: dict,
    groq_out: dict,
) -> ClinicalReport:
    ensure_dirs()
    html = _report_html(visit, structured, groq_out, language)
    html_path = REPORTS_PATIENT / f"visit_{visit.id}_{language}.html"
    html_path.write_text(html, encoding="utf-8")
    existing = db.query(ClinicalReport).filter(ClinicalReport.visit_id == visit.id).first()
    if existing is None:
        existing = ClinicalReport(visit_id=visit.id, language=language, structured_json="{}")
        db.add(existing)
    existing.language = language
    existing.structured_json = json.dumps(structured, ensure_ascii=False)
    existing.screening_summary = groq_out["screening_summary"]
    existing.tooth_findings = groq_out["tooth_findings"]
    existing.oral_health = groq_out["oral_health_summary"]
    existing.recommendations = groq_out["recommendations"]
    existing.follow_up = groq_out["follow_up"]
    existing.html_path = str(html_path)
    db.commit()
    db.refresh(existing)
    return existing


def _report_html(visit: Visit, structured: dict, groq_out: dict, language: str) -> str:
    p = structured.get("patient") or {}
    md = groq_out.get("markdown") or groq_out.get("screening_summary") or ""
    safe = (
        md.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    imgs = ""
    for im in visit.images[:4]:
        orig = _b64_rgb(_read_rgb(im.path)) if Path(im.path).exists() else None
        ov = (
            _b64_rgb(_read_rgb(im.overlay_path))
            if im.overlay_path and Path(im.overlay_path).exists()
            else None
        )
        if orig:
            imgs += f'<p>Original</p><img src="data:image/png;base64,{orig}" style="max-width:100%"/>'
        if ov:
            imgs += f'<p>Detection overlay</p><img src="data:image/png;base64,{ov}" style="max-width:100%"/>'
    return f"""<!doctype html>
<html lang="{language}"><head><meta charset="utf-8"/>
<title>CCC AI Dentist Camera 2.0 — Visit {visit.id}</title>
<style>
body {{ font-family: Segoe UI, sans-serif; max-width: 880px; margin: 24px auto; color: #0f172a; }}
.banner {{ background: #0ea5e9; color: white; padding: 16px 20px; border-radius: 12px; }}
.card {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 16px 0; }}
.warn {{ background: #fef3c7; padding: 12px; border-radius: 8px; }}
pre.md {{ white-space: pre-wrap; font-family: Segoe UI, sans-serif; line-height: 1.45; }}
img {{ border-radius: 8px; margin: 8px 0; }}
</style></head><body>
<div class="banner"><h1>CCC AI Dentist Camera 2.0</h1>
<p>AI-Assisted Dental Screening and Tooth Condition Analysis</p></div>
<div class="warn">{DISCLAIMER}</div>
<div class="card"><h2>Patient</h2>
<p>ID: {p.get("patient_id") or p.get("public_id")} · Name: {p.get("name")} · Age: {p.get("age")} ·
Visit: {p.get("visit_date")}</p></div>
<div class="card"><h2>Photographs</h2>{imgs}</div>
<div class="card"><pre class="md">{safe}</pre></div>
</body></html>"""


def visit_history(db: Session, public_id: str) -> dict:
    patient = db.query(Patient).filter(Patient.public_id == public_id).first()
    if patient is None:
        raise KeyError(public_id)
    visits = (
        db.query(Visit)
        .filter(Visit.patient_id == patient.id, Visit.deleted_at.is_(None))
        .order_by(Visit.created_at.desc())
        .all()
    )
    items = []
    for v in visits:
        n_teeth = 0
        highest = None
        thumb = None
        for im in v.images:
            n_teeth += len(im.detections)
            if thumb is None and Path(im.path).exists():
                thumb = _b64_rgb(_read_rgb(im.path))
            for det in im.detections:
                if det.crop and det.crop.prediction:
                    g = int(det.crop.prediction.grade)
                    highest = g if highest is None else max(highest, g)
        items.append(
            {
                "visit_id": v.id,
                "visit_date": v.visit_date,
                "notes": v.notes,
                "test_only": bool(getattr(v, "test_only", False)),
                "n_images": len(v.images),
                "n_teeth_analyzed": n_teeth,
                "highest_icdas": highest,
                "thumbnail_base64": thumb,
                "has_report": v.report is not None,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
        )
    return {"patient": patient_to_dict(patient), "visits": items, "disclaimer": DISCLAIMER}


def visit_detail(db: Session, visit_id: int) -> dict:
    visit = db.query(Visit).filter(Visit.id == visit_id, Visit.deleted_at.is_(None)).first()
    if visit is None:
        raise KeyError(visit_id)
    images = []
    for im in visit.images:
        rgb = _read_rgb(im.path) if Path(im.path).exists() else None
        ov = (
            _read_rgb(im.overlay_path)
            if im.overlay_path and Path(im.overlay_path).exists()
            else None
        )
        images.append(
            {
                "image_id": im.id,
                "filename": im.filename,
                "quality": json.loads(im.quality_json) if im.quality_json else {},
                "original_base64": _b64_rgb(rgb),
                "overlay_base64": _b64_rgb(ov),
                "n_teeth": len(im.detections),
            }
        )
    report = None
    if visit.report:
        r = visit.report
        report = {
            "language": r.language,
            "markdown": r.screening_summary,
            "screening_summary": r.screening_summary,
            "tooth_findings": r.tooth_findings,
            "oral_health": r.oral_health,
            "recommendations": r.recommendations,
            "follow_up": r.follow_up,
            "structured": json.loads(r.structured_json) if r.structured_json else {},
            "html_path": r.html_path,
        }
    return {
        "patient": patient_to_dict(visit.patient),
        "visit_id": visit.id,
        "visit_date": visit.visit_date,
        "notes": visit.notes,
        "images": images,
        "report": report,
        "disclaimer": DISCLAIMER,
    }


def admin_kpis(db: Session) -> dict:
    from .training_workflow import dataset_inventory

    n_labeled = db.query(TrainingLabel).filter(TrainingLabel.grade.isnot(None)).count()
    n_crops = db.query(TrainingLabel).count()
    latest = db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()
    inv = dataset_inventory(db)
    return {
        "patients": db.query(Patient).count(),
        "visits": db.query(Visit).filter(Visit.deleted_at.is_(None)).count(),
        "images": db.query(UploadedImage).count(),
        "training_images": db.query(TrainingImage).count(),
        "icdas_crops": n_crops,
        "labeled_crops": n_labeled,
        "labeled_verified_crops": inv["labeled_verified_crops"],
        "unique_training_images": inv["unique_images"],
        "dataset_versions": db.query(DatasetVersion).count(),
        "current_models": db.query(ModelVersion).count(),
        "latest_model": latest.name if latest else None,
        "icdas_train_enabled": inv["icdas_train_enabled"],
        "dataset_status": inv["status"],
        "training_progress": inv,
        "disclaimer": DISCLAIMER,
    }


def add_training_images(db: Session, files: list[tuple[str, bytes]]) -> list[dict]:
    from .training_workflow import ingest_training_image

    ensure_dirs()
    out = []
    duplicates = []
    for filename, data in files:
        item = ingest_training_image(db, filename, data, _decode_upload)
        out.append(item)
        if item.get("duplicate_status") in {"EXACT_DUPLICATE", "LIKELY_DUPLICATE", "INVALID"}:
            duplicates.append(item)
    return out


def unlabeled_queue(
    db: Session,
    image_id: int | None = None,
    label_id: int | None = None,
    resume: bool = False,
) -> dict:
    from .training_workflow import labeling_queue

    return labeling_queue(db, image_id=image_id, label_id=label_id, resume=resume)


def skip_label(db: Session, label_id: int) -> dict:
    from .training_workflow import skip_icdas_label

    return skip_icdas_label(db, label_id)


def save_label(db: Session, label_id: int, grade: int) -> dict:
    from .training_workflow import save_icdas_label

    return save_icdas_label(db, label_id, grade)


def dataset_summary(db: Session) -> dict:
    from .training_workflow import dataset_inventory

    return dataset_inventory(db)


def build_dataset_version(db: Session) -> dict:
    from .training_workflow import build_dataset_version as _build

    return _build(db)


def queue_training_job(db: Session, dataset_name: str | None) -> dict:
    from .training_workflow import launch_icdas_training

    return launch_icdas_training(db, dataset_name)


def training_status(db: Session, job_id: int | None) -> dict:
    q = db.query(TrainingJob).order_by(TrainingJob.id.desc())
    job = q.filter(TrainingJob.id == job_id).first() if job_id else q.first()
    if job is None:
        return {"status": "idle", "message": "No training jobs.", "log": ""}
    log = job.log_text or ""
    if "log=" in log:
        p = log.split("log=", 1)[-1].strip()
        if Path(p).exists():
            log = Path(p).read_text(encoding="utf-8", errors="replace")[-8000:]
    return {
        "job_id": job.id,
        "status": job.status,
        "message": job.message,
        "log": log,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def list_models(db: Session) -> list[dict]:
    seed_model_registry(db)
    rows = db.query(ModelVersion).order_by(ModelVersion.id.asc()).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "kind": m.kind,
            "path": m.path,
            "dataset_version": m.dataset_version,
            "metrics": json.loads(m.metrics_json) if m.metrics_json else None,
            "is_active": m.is_active,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in rows
    ]


def set_active_model(db: Session, model_id: int, active: bool) -> dict:
    from .training_workflow import apply_active_icdas_model

    return apply_active_icdas_model(db, model_id, active)


def evaluation_bundle() -> dict:
    v2 = TOOTH_V2_WEIGHTS.parent.parent / "eval_metrics.json"
    icdas_hist = (
        PROJECT_ROOT
        / "models"
        / "icdas"
        / "historical"
        / "icdas_mobilenet_cbam_5class_weighted"
        / "test_evaluation"
        / "metrics.json"
    )
    det = json.loads(v2.read_text(encoding="utf-8")) if v2.exists() else {}
    icd = json.loads(icdas_hist.read_text(encoding="utf-8")) if icdas_hist.exists() else {}
    return {
        "tooth_detector_v2": det,
        "icdas_historical": icd,
        "note": "Metrics are read from existing evaluation files. No new training was run.",
    }


def real_world_test(data: bytes, filename: str, engine=None, db: Session | None = None) -> dict:
    from .portal_runtime import get_runtime

    runtime = get_runtime()
    rgb = _decode_upload(data, filename)
    quality = assess_image_quality(rgb)
    if not detector_available():
        raise RuntimeError(detector_error() or "Tooth Detector V2 unavailable.")
    det = detect_rgb(rgb, filename)
    quality = attach_detection_warnings(quality, det["n_kept"], det["items"], rgb.shape)
    teeth = []
    if runtime.icdas_ok():
        for idx, item in enumerate(det["items"]):
            crop_rgb = det["crops"][idx][1]
            pred = runtime.predict_crop_rgb(crop_rgb)
            heat = overlay = None
            try:
                exp = runtime.explain_crop(pred)
                heat = exp.get("heatmap")
                overlay = exp.get("overlay")
            except Exception as exc:
                logger.exception("Grad-CAM failed in real-world test: %s", exc)
            grade = int(pred["icdas_grade"])
            stage = patient_stage(grade)
            teeth.append(
                {
                    "index": idx,
                    "tooth_id": f"T{idx + 1:02d}",
                    "detection_confidence": item.confidence,
                    "icdas_grade": grade,
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    **stage,
                    "crop_base64": _b64_rgb(crop_rgb),
                    "heatmap_base64": heat,
                    "overlay_base64": overlay,
                }
            )
    visit_id = None
    if db is not None:
        patient = create_patient(db, {"public_id": "TEST-ONLY", "name": "Real-world test"})
        visit = create_visit(
            db,
            patient,
            datetime.utcnow().strftime("%Y-%m-%d"),
            notes="TEST ONLY — not a training sample",
            test_only=True,
        )
        rec, _ = store_visit_image(db, visit, data, filename)
        run_detection_for_image(db, rec)
        db.expire_all()
        visit = db.query(Visit).filter(Visit.id == visit.id).first()
        classify_visit_crops(db, visit, include_heatmaps=True, persist=True)
        visit_id = visit.id
    return {
        "quality": quality,
        "teeth_detected": det["n_kept"],
        "mean_detection_confidence": det["mean_confidence"],
        "original_base64": _b64_rgb(rgb),
        "overlay_base64": _b64_rgb(det["overlay_rgb"]),
        "teeth": teeth,
        "labels_saved": False,
        "training_dataset_updated": False,
        "test_only": True,
        "visit_id": visit_id,
        "disclaimer": DISCLAIMER,
    }


def generate_report_for_visit(
    db: Session,
    visit: Visit,
    analysis: dict,
    language: str,
) -> dict:
    if not (analysis.get("teeth") or []):
        from .portal_runtime import ICDAS_NOT_DEPLOYED

        raise RuntimeError(ICDAS_NOT_DEPLOYED)
    structured = structured_for_groq(visit.patient, visit, analysis)
    groq_out = generate_screening_report(structured, language=language)
    rec = save_clinical_report(db, visit, language, structured, groq_out)
    html = Path(rec.html_path).read_text(encoding="utf-8") if rec.html_path else ""
    return {
        "visit_id": visit.id,
        "language": language,
        "structured": structured,
        **groq_out,
        "html": html,
        "disclaimer": DISCLAIMER,
    }
