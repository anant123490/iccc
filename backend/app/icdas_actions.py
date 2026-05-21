"""ICDAS grade to clinical action mapping."""

from typing import TypedDict


class ClinicalAction(TypedDict):
    grade: int
    label: str
    action: str
    description: str
    finding: str
    recommendation: str
    urgency: str


ICDAS_ACTIONS: dict[int, ClinicalAction] = {
    0: {
        "grade": 0,
        "label": "Sound",
        "action": "Monitor",
        "description": "No evidence of caries. Continue routine preventive care.",
        "finding": "Sound tooth surface",
        "recommendation": "Continue routine monitoring and preventive care",
        "urgency": "low",
    },
    1: {
        "grade": 1,
        "label": "Initial lesion",
        "action": "Monitor",
        "description": "First visual change in enamel. Monitor and reinforce oral hygiene.",
        "finding": "First visual change in enamel",
        "recommendation": "Monitor + reinforce oral hygiene",
        "urgency": "low",
    },
    2: {
        "grade": 2,
        "label": "Distinct visual change",
        "action": "Fluoride treatment",
        "description": "Distinct visual change in enamel. Consider fluoride varnish or remineralization.",
        "finding": "Distinct visual change in enamel",
        "recommendation": "Dentist review + preventive fluoride treatment",
        "urgency": "medium",
    },
    3: {
        "grade": 3,
        "label": "Localized breakdown",
        "action": "Restoration needed",
        "description": "Localized enamel breakdown. Restorative treatment may be indicated.",
        "finding": "Localized enamel breakdown",
        "recommendation": "Dentist review + restorative assessment",
        "urgency": "high",
    },
    4: {
        "grade": 4,
        "label": "Underlying dentin",
        "action": "Restoration needed",
        "description": "Underlying dentin shadow. Restoration recommended.",
        "finding": "Underlying dentin shadow",
        "recommendation": "Dentist review + restoration needed",
        "urgency": "high",
    },
    5: {
        "grade": 5,
        "label": "Distinct cavity",
        "action": "Restoration needed",
        "description": "Distinct cavity with visible dentin. Prompt restoration required.",
        "finding": "Distinct cavity with visible dentin",
        "recommendation": "Prompt restorative treatment by dentist",
        "urgency": "critical",
    },
    6: {
        "grade": 6,
        "label": "Extensive cavity",
        "action": "Restoration needed",
        "description": "Extensive distinct cavity with dentin. Urgent restorative care.",
        "finding": "Extensive distinct cavity with dentin involvement",
        "recommendation": "Urgent dentist review + restoration",
        "urgency": "critical",
    },
}


def get_clinical_action(grade: int) -> ClinicalAction:
    return ICDAS_ACTIONS.get(grade, ICDAS_ACTIONS[0])
