"""
Central ICDAS 0–4 metadata for the ML pipeline and API.

This project classifies only ICDAS grades 0 through 4.
ICDAS 5 and 6 are out of scope and must not be remapped to 4.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


NUM_CLASSES = 5
MAX_ICDAS_GRADE = 4
MIN_ICDAS_GRADE = 0
ORDINAL_THRESHOLDS = NUM_CLASSES - 1  # 4 thresholds for 5 classes
VALID_CLASS_NAMES = ("0", "1", "2", "3", "4")
VALID_CLASSES = (0, 1, 2, 3, 4)
SPLITS = ("train", "val", "test")


class IcdasInfo(TypedDict):
    grade: int
    name: str
    label: str
    action: str
    description: str
    finding: str
    recommendation: str
    urgency: str


ICDAS_INFO: Dict[int, IcdasInfo] = {
    0: {
        "grade": 0,
        "name": "ICDAS 0",
        "label": "Sound",
        "action": "Monitor",
        "description": (
            "Sound tooth surface. No evidence of caries after visual inspection."
        ),
        "finding": "No visible evidence of dental caries",
        "recommendation": "Continue routine oral hygiene and preventive dental care",
        "urgency": "LOW",
    },
    1: {
        "grade": 1,
        "name": "ICDAS 1",
        "label": "Initial enamel change",
        "action": "Monitor",
        "description": (
            "First visual change in enamel. Opacity or discoloration is typically "
            "visible after air drying."
        ),
        "finding": "First visual change in enamel",
        "recommendation": "Preventive care, oral-hygiene reinforcement, and monitoring",
        "urgency": "LOW",
    },
    2: {
        "grade": 2,
        "name": "ICDAS 2",
        "label": "Distinct visual change",
        "action": "Preventive treatment",
        "description": (
            "Distinct visual change in enamel when wet. Demineralization is more "
            "established but remains non-cavitated."
        ),
        "finding": "Distinct visual change in enamel",
        "recommendation": "Dental review and preventive fluoride / remineralization care",
        "urgency": "MODERATE",
    },
    3: {
        "grade": 3,
        "name": "ICDAS 3",
        "label": "Localized enamel breakdown",
        "action": "Restorative assessment",
        "description": (
            "Localized enamel breakdown due to caries, without visible dentin."
        ),
        "finding": "Localized enamel breakdown without visible dentin",
        "recommendation": "Dentist review for restorative assessment",
        "urgency": "HIGH",
    },
    4: {
        "grade": 4,
        "name": "ICDAS 4",
        "label": "Underlying dentin shadow",
        "action": "Restorative assessment",
        "description": (
            "Underlying dark shadow from dentin, with or without localized enamel "
            "breakdown. Indicates dentin involvement."
        ),
        "finding": "Underlying dark shadow indicating possible dentin involvement",
        "recommendation": "Dentist review to assess dentin involvement and restoration need",
        "urgency": "HIGH",
    },
}


def get_icdas_info(grade: int) -> IcdasInfo:
    if grade not in ICDAS_INFO:
        raise ValueError(
            f"Unsupported ICDAS grade {grade}. This system supports ICDAS 0–4 only."
        )
    return ICDAS_INFO[grade]


def class_names(num_classes: int = NUM_CLASSES) -> List[str]:
    return [f"ICDAS {i}" for i in range(num_classes)]
