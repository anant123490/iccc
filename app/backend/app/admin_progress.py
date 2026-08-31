"""Training progress rules. No ICDAS auto-assignment. No per-photo class forcing."""

from __future__ import annotations

MIN_DATASET_CROPS = 5


def validate_icdas_grade(grade: object) -> int:
    """Accept only integer ICDAS 0–4. Never remap 5/6."""
    if isinstance(grade, bool) or grade is None:
        raise ValueError("ICDAS labels must be 0–4.")
    if isinstance(grade, str):
        raise ValueError("ICDAS labels must be 0–4.")
    try:
        value = int(grade)
    except (TypeError, ValueError) as exc:
        raise ValueError("ICDAS labels must be 0–4.") from exc
    if value != grade:
        raise ValueError("ICDAS labels must be 0–4.")
    if value < 0 or value > 4:
        raise ValueError("ICDAS labels must be 0–4.")
    return value


def dataset_min_crops_message(labeled_n: int) -> str | None:
    n = int(labeled_n or 0)
    if n >= MIN_DATASET_CROPS:
        return None
    return (
        f"Need at least {MIN_DATASET_CROPS} verified labeled UNIQUE crops. Current: {n}."
    )


def dataset_build_ready(labeled_n: int, class_counts: dict | None = None) -> bool:
    """BUILD DATASET gate: enough verified labeled UNIQUE crops. Missing classes do not block."""
    return int(labeled_n or 0) >= MIN_DATASET_CROPS


def crop_is_resolved(grade: int | None, skipped: bool | None) -> bool:
    """Labeled 0–4 or dentist chose Leave/Skip."""
    if skipped:
        return True
    return grade is not None


def dataset_classes_ready(class_counts: dict) -> bool:
    """TRAIN gate: genuine labeled examples exist for every ICDAS 0–4 class."""
    return all(int(class_counts.get(str(i) or i, 0) or 0) > 0 for i in range(5))


def missing_icdas_classes(class_counts: dict) -> list[int]:
    return [i for i in range(5) if int(class_counts.get(str(i), class_counts.get(i, 0)) or 0) == 0]


def imbalance_warning_text(class_counts: dict) -> str | None:
    vals = [(i, int(class_counts.get(str(i), 0) or 0)) for i in range(5)]
    present = [(i, n) for i, n in vals if n > 0]
    if len(present) < 2:
        return None
    lo_i, lo_n = min(present, key=lambda t: t[1])
    hi_i, hi_n = max(present, key=lambda t: t[1])
    if hi_n > 0 and lo_n * 4 < hi_n:
        return (
            f"Dataset imbalance: ICDAS {lo_i} has significantly fewer samples "
            f"({lo_n}) than ICDAS {hi_i} ({hi_n}). Keep labeling what you see — "
            "do not assign a grade just to fill a class."
        )
    return None


def missing_classes_message(missing: list[int]) -> str | None:
    if not missing:
        return None
    names = ", ".join(str(i) for i in missing)
    return (
        f"ICDAS classes {names} are currently missing. The dataset can be built, "
        "but model training is not recommended until sufficient genuine examples "
        "of all ICDAS 0–4 classes are collected."
    )
