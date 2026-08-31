"""Admin training file-uploader state (no disk, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

STREAMLIT_ROOT = Path(__file__).resolve().parents[2] / "frontend" / "streamlit"
sys.path.insert(0, str(STREAMLIT_ROOT))

from shared.admin_workflow import merge_pending_nav  # noqa: E402
from shared.upload_state import next_training_uploader_nonce  # noqa: E402


def test_uploader_nonce_starts_at_one_from_empty():
    assert next_training_uploader_nonce(None) == 1
    assert next_training_uploader_nonce(0) == 1


def test_uploader_nonce_increments_for_next_batch():
    first = next_training_uploader_nonce(0)
    second = next_training_uploader_nonce(first)
    assert first != second
    assert second == first + 1
    assert f"training_files_{first}" != f"training_files_{second}"


def test_merge_pending_nav_overrides_radio_before_widget():
    assert merge_pending_nav("Training upload", "Box review") == "Box review"
    assert merge_pending_nav("Dashboard", None) == "Dashboard"
    assert merge_pending_nav(None, None) == "Dashboard"
