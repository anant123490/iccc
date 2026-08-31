"""ICDAS grade to clinical action mapping (ICDAS 0–4)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from .config import PROJECT_ROOT

ML_SRC = PROJECT_ROOT / "ml"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from src.icdas import ICDAS_INFO, get_icdas_info  # noqa: E402


class ClinicalAction(TypedDict):
    grade: int
    name: str
    label: str
    action: str
    description: str
    finding: str
    recommendation: str
    urgency: str


ICDAS_ACTIONS: dict[int, ClinicalAction] = {
    grade: {
        "grade": info["grade"],
        "name": info["name"],
        "label": info["label"],
        "action": info["action"],
        "description": info["description"],
        "finding": info["finding"],
        "recommendation": info["recommendation"],
        "urgency": info["urgency"],
    }
    for grade, info in ICDAS_INFO.items()
}


def get_clinical_action(grade: int) -> ClinicalAction:
    return get_icdas_info(grade)
