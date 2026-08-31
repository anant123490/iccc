"""Frontend copy of the backend reset confirmation phrase. Keep in sync."""

RESET_CONFIRMATION_TEXT = (
    "I understand that this permanently deletes the selected training data."
)


def reset_confirmation_ok(checked: bool, typed: str) -> bool:
    return bool(checked) and (typed or "").strip() == RESET_CONFIRMATION_TEXT
