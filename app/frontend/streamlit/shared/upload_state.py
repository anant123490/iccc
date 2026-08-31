"""Streamlit file-uploader nonce. Does not touch disk or the database."""

from __future__ import annotations


TRAINING_UI_KEYS = (
    "train_up",
    "last_upload_names",
    "label_id",
    "label_resume",
    "job",
    "job_status",
)


def next_training_uploader_nonce(current: int | None) -> int:
    """Return a new widget key so the uploader can accept a fresh batch."""
    try:
        n = int(current or 0)
    except (TypeError, ValueError):
        n = 0
    return n + 1


def clear_admin_training_ui_state(state: dict) -> dict:
    """Drop stale Admin workflow widget/session keys. Does not change navigation keys."""
    removed = []
    for key in TRAINING_UI_KEYS:
        if key in state:
            state.pop(key, None)
            removed.append(key)
    state["training_uploader_nonce"] = next_training_uploader_nonce(
        state.get("training_uploader_nonce")
    )
    return {"removed": removed, "training_uploader_nonce": state["training_uploader_nonce"]}
