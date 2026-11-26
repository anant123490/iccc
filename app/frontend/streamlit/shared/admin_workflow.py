"""Guided Admin training steps. Navigation only."""

from __future__ import annotations

from html import escape

WORKFLOW_PAGES = [
    ("Training upload", "Upload Photos", "upload"),
    ("Box review", "Review Teeth", "review"),
    ("Labeling", "Label ICDAS", "label"),
    ("Dataset", "Dataset", "dataset"),
    ("Training", "Train Model", "train"),
]

NAV_PENDING_KEY = "admin_nav_pending"
NAV_PAGE_KEY = "admin_page"
NAV_GEN_KEY = "admin_nav_gen"


def merge_pending_nav(current: str | None, pending: str | None) -> str:
    """Resolve sidebar page before the radio widget is created."""
    if pending:
        return pending
    return current or "Dashboard"


def apply_pending_nav() -> str:
    """Apply queued Continue/Back without writing a live radio widget key."""
    import streamlit as st

    pending = st.session_state.pop(NAV_PENDING_KEY, None)
    current = st.session_state.get(NAV_PAGE_KEY)
    page = merge_pending_nav(current, pending)
    st.session_state[NAV_PAGE_KEY] = page
    if pending:
        st.session_state[NAV_GEN_KEY] = int(st.session_state.get(NAV_GEN_KEY) or 0) + 1
    if NAV_GEN_KEY not in st.session_state:
        st.session_state[NAV_GEN_KEY] = 0
    return page


def set_admin_nav(page: str) -> None:
    """Button callback: queue a page. Do not touch the radio widget key."""
    import streamlit as st

    st.session_state[NAV_PENDING_KEY] = page


def goto_page(page: str) -> None:
    """Queue a page change and rerun (only call before a radio with key admin_nav exists)."""
    import streamlit as st

    st.session_state[NAV_PENDING_KEY] = page
    st.rerun()


def render_stepper(current_page: str, inventory: dict) -> None:
    import streamlit as st

    wf = inventory.get("workflow") or {}
    parts = []
    for page, title, key in WORKFLOW_PAGES:
        step = wf.get(key) or {}
        done = bool(step.get("done"))
        here = page == current_page
        mark = "CURRENT" if here else ("✓" if done else "")
        state_class = "ccc-workflow-current" if here else ("ccc-workflow-done" if done else "ccc-workflow-pending")
        parts.append(
            f'<div class="ccc-workflow-step {state_class}">'
            f'<span class="ccc-workflow-title">{escape(title)}</span>'
            f'<span class="ccc-workflow-state">{escape(str(mark or step.get("status", "Pending")))}</span>'
            "</div>"
        )
    st.markdown(
        "<div class='ccc-workflow-strip'>"
        + "".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )
    cur = next((k for p, _, k in WORKFLOW_PAGES if p == current_page), None)
    info = wf.get(cur) or {}
    if info.get("detail"):
        st.caption(info["detail"])
