"""Guided Admin training steps. Navigation only."""

from __future__ import annotations

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
        bg = "#0ea5e9" if here else ("#16a34a" if done else "#e2e8f0")
        color = "white" if here or done else "#334155"
        parts.append(
            f'<div style="flex:1;min-width:110px;text-align:center;padding:10px 6px;'
            f'border-radius:12px;background:{bg};color:{color};font-weight:600;font-size:0.85rem;">'
            f"{title}<br><span style='font-size:0.75rem'>{mark or step.get('status', '')}</span></div>"
        )
    st.markdown(
        "<div style='display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 16px 0'>"
        + "".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )
    cur = next((k for p, _, k in WORKFLOW_PAGES if p == current_page), None)
    info = wf.get(cur) or {}
    if info.get("detail"):
        st.caption(info["detail"])
