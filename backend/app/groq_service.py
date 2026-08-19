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
        f"AI dental assessment (decision support only).\n\n"
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
    model_name: str = "MobileNetV3-Small + CBAM + ordinal regression",
) -> dict:
    action = get_clinical_action(icdas_grade)
    finding = finding or action["finding"]
    urgency = (urgency or action["urgency"]).upper()

    prompt = f"""
You are an AI dental assistant writing a concise decision-support note.

A convolutional neural network already classified the image. You must NOT
change, second-guess, or re-grade the ICDAS result.

Machine-learning prediction (authoritative):
- ICDAS grade: {icdas_grade} (supported range: 0–4 only)
- Model: {model_name}
- Confidence: {confidence:.1f}%
- Finding: {finding}
- Urgency: {urgency}

Write a short assessment for a student dental-AI prototype.

Return ONLY valid JSON with this shape:
{{
  "finding": "one or two sentences, consistent with ICDAS {icdas_grade}",
  "recommendation": "practical next step, not a definitive diagnosis",
  "urgency": "{urgency}",
  "report": "120-180 word plain-text assessment. Include classification, confidence, finding, recommendation, and a clear disclaimer that this is AI decision support, not a medical diagnosis."
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
