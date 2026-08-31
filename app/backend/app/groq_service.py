"""
Groq-backed AI explanation.

The CNN/ordinal model is the only source of the ICDAS grade.
Groq never reclassifies the image.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from .icdas_actions import get_clinical_action

logger = logging.getLogger("icdas.groq")

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
_client = None


def _get_client():
    global _client
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.warning("GROQ_API_KEY is not configured; using local report fallback.")
        return None
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=key)
    return _client


def groq_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _local_report(icdas_grade: int, confidence: float, finding: str, urgency: str) -> dict:
    action = get_clinical_action(icdas_grade)
    recommendation = action["recommendation"]
    report = (
        f"AI-assisted assessment (not a definitive clinical diagnosis).\n\n"
        f"Classification: {action['name']} ({action['label']}).\n"
        f"Confidence: {confidence:.1f}%.\n"
        f"Urgency: {urgency}.\n\n"
        f"Finding: {finding}\n\n"
        f"Recommendation: {recommendation}\n\n"
        "This is an AI-generated research prototype output and is not a "
        "definitive medical diagnosis. A licensed dentist should examine the patient."
    )
    return {
        "finding": finding,
        "recommendation": recommendation,
        "urgency": urgency,
        "report": report,
    }


def generate_report(
    icdas_grade: int,
    confidence: float,
    finding: Optional[str] = None,
    urgency: Optional[str] = None,
    model_name: str = "MobileNetV3-Small + CBAM + 5-class softmax",
    detected_region_count: Optional[int] = None,
    quality_status: Optional[str] = None,
    gradcam_available: Optional[bool] = None,
    localization_mode: Optional[str] = None,
) -> dict:
    action = get_clinical_action(icdas_grade)
    finding = finding or action["finding"]
    urgency = (urgency or action["urgency"]).upper()

    extra = []
    if detected_region_count is not None:
        extra.append(f"- Localized decay regions: {detected_region_count}")
    if quality_status:
        extra.append(f"- Image quality status: {quality_status}")
    if gradcam_available is not None:
        extra.append(f"- Grad-CAM available: {gradcam_available}")
    if localization_mode:
        extra.append(f"- Localization mode: {localization_mode}")
    extra_block = "\n".join(extra)

    prompt = f"""
You are an AI dental assistant writing a concise decision-support note.

A convolutional neural network already classified the image. You must NOT
change, second-guess, or re-grade the ICDAS result.
Do not invent FDI tooth numbers.

Machine-learning prediction (authoritative):
- ICDAS grade: {icdas_grade} (supported range: 0–4 only)
- Model: {model_name}
- Confidence: {confidence:.1f}%
- Finding: {finding}
- Urgency: {urgency}
{extra_block}

Write a short AI-assisted assessment for a student dental-AI prototype.
The report must state this is an AI-assisted assessment, not a definitive clinical diagnosis.

Return ONLY valid JSON with this shape:
{{
  "finding": "one or two sentences, consistent with ICDAS {icdas_grade}",
  "recommendation": "practical next step, not a definitive diagnosis",
  "urgency": "{urgency}",
  "report": "120-180 word plain-text assessment. Include classification, confidence, finding, recommendation, and a clear disclaimer that this is AI-assisted assessment, not a medical diagnosis."
}}

ICDAS 0–4 reference used by this project:
- 0: Sound tooth surface
- 1: First visual change in enamel
- 2: Distinct visual change in enamel
- 3: Localized enamel breakdown without visible dentin
- 4: Underlying dark shadow from dentin

Do not mention ICDAS 5 or 6.
Do not invent a different grade.
Do not claim the model is medically definitive.
"""

    client = _get_client()
    if client is None:
        return _local_report(icdas_grade, confidence, finding, urgency)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dental AI assistant. Return only valid JSON. "
                        "Never override the provided ICDAS grade."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)

        out_finding = str(data.get("finding") or finding)
        recommendation = str(data.get("recommendation") or action["recommendation"])
        out_urgency = str(data.get("urgency") or urgency).upper()
        if out_urgency not in {"LOW", "MODERATE", "HIGH"}:
            out_urgency = urgency if urgency in {"LOW", "MODERATE", "HIGH"} else "MODERATE"
        report = str(data.get("report") or "").strip()
        if not report:
            report = _local_report(icdas_grade, confidence, out_finding, out_urgency)["report"]

        return {
            "finding": out_finding,
            "recommendation": recommendation,
            "urgency": out_urgency,
            "report": report,
        }
    except Exception as exc:
        logger.exception("Groq report generation failed: %s", exc)
        return _local_report(icdas_grade, confidence, finding, urgency)


DISCLAIMER_EN = (
    "This report is generated by an AI-assisted dental screening system. "
    "It is intended for educational and preliminary screening purposes only "
    "and does not replace a clinical examination by a qualified dental professional."
)


def _conf01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def groq_payload_from_structured(structured: dict, language: str = "en") -> dict:
    """JSON Groq is allowed to see. ICDAS grades are copied, never computed."""
    patient = structured.get("patient") or {}
    quality = structured.get("image_quality")
    if isinstance(quality, list):
        verdicts = [
            (item.get("quality") or {}).get("verdict")
            for item in quality
            if isinstance(item, dict)
        ]
        if "FAIL" in verdicts:
            qstatus = "FAIL"
        elif "WARNING" in verdicts:
            qstatus = "WARNING"
        elif "PASS" in verdicts:
            qstatus = "PASS"
        else:
            qstatus = "UNKNOWN"
        quality_out = {"status": qstatus, "images": quality}
    elif isinstance(quality, dict):
        quality_out = {
            "status": quality.get("status") or quality.get("verdict") or "UNKNOWN"
        }
    else:
        quality_out = {"status": "UNKNOWN"}

    teeth_in = structured.get("teeth") or []
    teeth = []
    for i, t in enumerate(teeth_in, start=1):
        grade = int(t.get("icdas_grade"))
        if grade < 0 or grade > 4:
            continue
        tooth_id = str(t.get("tooth_id") or f"T{i:02d}")
        teeth.append(
            {
                "tooth_id": tooth_id,
                "icdas_grade": grade,
                "confidence": round(_conf01(t.get("confidence")), 4),
            }
        )
    counts = structured.get("icdas_counts") or {}
    if not counts:
        from collections import Counter

        c = Counter(int(t["icdas_grade"]) for t in teeth)
        counts = {str(g): int(c.get(g, 0)) for g in range(5)}
    lang = {"en": "English", "hi": "Hindi", "kn": "Kannada"}.get(language, "English")
    return {
        "language": lang,
        "patient": {
            "patient_id": patient.get("patient_id") or patient.get("public_id"),
            "name": patient.get("name"),
            "age": patient.get("age"),
            "visit_date": patient.get("visit_date"),
        },
        "image_quality": quality_out,
        "summary": {
            "teeth_detected": structured.get("teeth_detected"),
            "teeth_analyzed": structured.get("teeth_analyzed") or len(teeth),
            "icdas_counts": counts,
        },
        "teeth": teeth,
        "rules": {
            "icdas_source": "classifier",
            "groq_must_not_change_grades": True,
            "icdas_5_6_out_of_scope": True,
            "no_fdi": True,
            "no_diagnosis": True,
        },
    }


def _plain(lang: str, grade: int) -> dict:
    from .groq_dentist_prompt import ICDAS_PLAIN

    pack = ICDAS_PLAIN.get(lang) or ICDAS_PLAIN["en"]
    return pack[int(grade)]


def _split_markdown_sections(md: str) -> dict:
    """Keep DB columns populated from the markdown report."""
    return {
        "screening_summary": md,
        "tooth_findings": md,
        "oral_health_summary": md,
        "recommendations": md,
        "follow_up": md,
        "markdown": md,
        "disclaimer": DISCLAIMER_EN,
    }


def _local_screening_report(payload: dict, language: str = "en") -> dict:
    from .groq_dentist_prompt import DISCLAIMER, DISCLAIMER_HI, DISCLAIMER_KN, ICDAS_PLAIN

    data = groq_payload_from_structured(payload, language)
    lang = language if language in ICDAS_PLAIN else "en"
    p = data["patient"]
    q = (data.get("image_quality") or {}).get("status", "UNKNOWN")
    summary = data.get("summary") or {}
    teeth = data.get("teeth") or []
    counts = summary.get("icdas_counts") or {}
    n0 = int(counts.get("0") or 0)
    n12 = int(counts.get("1") or 0) + int(counts.get("2") or 0)
    n3 = int(counts.get("3") or 0)
    n4 = int(counts.get("4") or 0)
    analyzed = int(summary.get("teeth_analyzed") or len(teeth))
    detected = summary.get("teeth_detected")

    if lang == "hi":
        title = "# 🦷 CCC AI Dentist Camera 2.0 — एआई दंत स्क्रीनिंग रिपोर्ट"
        disc = DISCLAIMER_HI
    elif lang == "kn":
        title = "# 🦷 CCC AI Dentist Camera 2.0 — ಎಐ ದಂತ ಸ್ಕ್ರೀನಿಂಗ್ ವರದಿ"
        disc = DISCLAIMER_KN
    else:
        title = "# 🦷 CCC AI Dentist Camera 2.0 — AI Dental Screening Report"
        disc = DISCLAIMER

    lines = [
        title,
        "",
        "## Patient Information" if lang == "en" else ("## रोगी जानकारी" if lang == "hi" else "## ರೋಗಿ ಮಾಹಿತಿ"),
        f"- Patient ID: {p.get('patient_id')}",
        f"- Name: {p.get('name')}",
        f"- Age: {p.get('age')}",
        f"- Visit Date: {p.get('visit_date')}",
        "",
        "## AI Screening Summary" if lang == "en" else ("## एआई स्क्रीनिंग सारांश" if lang == "hi" else "## ಎಐ ಸ್ಕ್ರೀನಿಂಗ್ ಸಾರಾಂಶ"),
        (
            f"This AI-assisted screening used uploaded photographs with image quality **{q}**. "
            f"The detector marked **{detected}** teeth and the classifier analyzed **{analyzed}** tooth crop(s). "
            f"ICDAS counts from the model (not from this text): 0={counts.get('0', 0)}, "
            f"1={counts.get('1', 0)}, 2={counts.get('2', 0)}, 3={counts.get('3', 0)}, 4={counts.get('4', 0)}."
        ),
    ]
    if n3 or n4:
        lines.append(
            "Most findings are summarized below. One or more teeth may need a dentist to take a closer look."
        )
    elif n12:
        lines.append("Most analyzed teeth appear largely healthy, with some early surface changes to watch.")
    else:
        lines.append("Based on the uploaded photographs, most analyzed teeth appear healthy on this screening.")
    lines += [
        "",
        "## Tooth-by-Tooth Findings",
        "",
        "| Tooth | ICDAS | Confidence | Explanation |",
        "| ----- | ----- | ---------- | ----------- |",
    ]
    for t in teeth:
        g = int(t["icdas_grade"])
        info = _plain(lang, g)
        conf = float(t["confidence"])
        lines.append(
            f"| {t['tooth_id']} | {g} | {conf:.2f} | {info['table']} |"
        )
    lines.append("")
    for t in teeth:
        g = int(t["icdas_grade"])
        info = _plain(lang, g)
        conf = float(t["confidence"])
        lines += [
            f"### Tooth {t['tooth_id']}",
            "",
            f"**ICDAS Grade:** {g}",
            "",
            f"**Current Tooth Stage (Easy to Understand):** {info['stage']}",
            "",
            f"**What This Means:** {info['explanation']} {info['meaning']}",
            "",
            f"**Confidence:** {conf:.0%}",
            "",
        ]
        if conf < 0.55:
            lines.append(
                "The prediction confidence for this tooth is relatively low, so the finding "
                "should be interpreted cautiously and confirmed through professional examination."
            )
            lines.append("")
        lines += [f"**Suggested Next Step:** {info['next_step']}", ""]
    if not teeth:
        lines.append("No tooth crops were classified. No extra teeth were added.")
        lines.append("")

    lines += [
        "## Overall Oral Health Summary",
        "",
        f"- Healthy teeth (ICDAS 0): {n0}",
        f"- Early enamel changes (ICDAS 1–2): {n12}",
        f"- Closer examination suggested (ICDAS 3): {n3}",
        f"- Prompt professional assessment suggested (ICDAS 4): {n4}",
        "",
        "## Understanding Your Overall Dental Health",
        "",
    ]
    if analyzed and n0 == analyzed:
        lines.append("Most of your teeth appear healthy on this AI-assisted screening.")
    elif n4:
        lines.append(
            "One or more teeth show more advanced visible changes and should be examined by a dentist."
        )
    elif n3:
        lines.append("Some teeth show more noticeable surface damage and deserve a professional look.")
    elif n12:
        lines.append("A few teeth show very early changes that should be monitored.")
    else:
        lines.append("The analysis suggests a mixed picture; please read the tooth list above.")
    lines += [
        "",
        "## Visual Severity Summary",
        "",
        "| Tooth | ICDAS | Current Stage | Priority |",
        "| ----- | ----- | ------------- | -------- |",
    ]
    for t in teeth:
        g = int(t["icdas_grade"])
        info = _plain(lang, g)
        lines.append(f"| {t['tooth_id']} | {g} | {info['stage']} | {info['priority']} |")
    lines += ["", "## Personalized Recommendations", ""]
    if n4:
        lines.append(
            "Please arrange a prompt consultation with a dentist. Further clinical examination "
            "and imaging may be appropriate. This is not a prescription and not a treatment plan."
        )
    elif n3:
        lines.append(
            "Schedule a dental examination and avoid delaying evaluation. Keep cleaning carefully "
            "around the flagged teeth."
        )
    elif n12:
        lines.append(
            "Monitor the affected teeth, keep up oral hygiene, and consider fluoride-based "
            "preventive care after consulting a dentist. Do not start medicines from this report."
        )
    else:
        lines.append(
            "Continue regular brushing, flossing, and routine dental check-ups."
        )
    lines += ["", "## What Should I Do Next?", ""]
    if n0:
        lines.append("- **Healthy teeth:** Continue regular brushing and flossing.")
    if n12:
        lines.append("- **Teeth with ICDAS 1–2:** Monitor these teeth and maintain preventive oral care.")
    if n3 or n4:
        lines.append(
            "- **Teeth with ICDAS 3–4:** Schedule a dental examination within a reasonable time."
        )
    lines += [
        "",
        "## Preventive Oral Care Tips",
        "",
        "- Brush twice daily using fluoride toothpaste.",
        "- Clean between teeth regularly.",
        "- Cut down on frequent sugary snacks and drinks.",
        "- Drink water after meals when you can.",
        "- Visit a dentist periodically even if you feel well.",
        "",
        "## Important AI Disclaimer",
        "",
        f"> {disc}",
        "",
        f"> {DISCLAIMER}",
    ]
    md = "\n".join(lines)
    out = _split_markdown_sections(md)
    out["disclaimer"] = disc
    return out


def _grades_preserved(payload: dict, markdown: str) -> bool:
    text = markdown or ""
    if "ICDAS 5" in text or "ICDAS 6" in text:
        return False
    for t in payload.get("teeth") or []:
        tid = t["tooth_id"]
        grade = int(t["icdas_grade"])
        if tid not in text:
            return False
        if f"**ICDAS Grade:** {grade}" not in text and f"| {tid} | {grade} |" not in text:
            if str(grade) not in text:
                return False
    return True


def generate_screening_report(structured: dict, language: str = "en") -> dict:
    """Explain classifier JSON. Never assign or modify ICDAS grades."""
    from .groq_dentist_prompt import SCREENING_SYSTEM_PROMPT

    payload = groq_payload_from_structured(structured, language)
    if not (payload.get("teeth") or []):
        raise RuntimeError(
            "ICDAS model has not yet been trained/deployed. "
            "Groq will not invent ICDAS grades or tooth findings."
        )
    fallback = _local_screening_report(structured, language=language)
    client = _get_client()
    if client is None:
        return fallback
    user = (
        "Treat this JSON as the source of truth. Explain it. Do not add teeth or change ICDAS.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=4000,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        markdown = str(data.get("markdown") or "").strip()
        if not markdown or not _grades_preserved(payload, markdown):
            logger.warning("Groq markdown missing or grades not preserved; using local report.")
            return fallback
        out = _split_markdown_sections(markdown)
        return out
    except Exception as exc:
        logger.exception("Groq screening report failed: %s", exc)
        return fallback


def translate_ui_bundle(labels: dict, language: str) -> dict:
    """Translate UI strings. Never translate ICDAS numeric grades."""
    if language == "en":
        return labels
    client = _get_client()
    if client is None:
        return labels
    prompt = f"""
Translate JSON string values to language code "{language}".
Do not translate numbers, ICDAS, CCC, or proper nouns that are product names.
Return ONLY JSON with the same keys.
Input:
{json.dumps(labels, ensure_ascii=False)}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        if isinstance(data, dict):
            return {k: str(data.get(k, v)) for k, v in labels.items()}
    except Exception as exc:
        logger.exception("Groq translation failed: %s", exc)
    return labels

