"""Reusable Streamlit cards and helpers."""

from __future__ import annotations

import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from .theme import ICDAS_COLORS


def show_disclaimer(text: str) -> None:
    st.markdown(f'<div class="ccc-disclaimer">{text}</div>', unsafe_allow_html=True)


def b64_image(data: str | None, caption: str | None = None, width: int | None = 220):
    if not data:
        st.caption("No image")
        return
    raw = base64.b64decode(data)
    img = Image.open(BytesIO(raw))
    st.image(img, caption=caption, width=width)


def icdas_badge(grade: int) -> str:
    color = ICDAS_COLORS.get(int(grade), "#64748b")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 10px;'
        f'border-radius:999px;font-weight:700;">ICDAS {int(grade)}</span>'
    )


def kpi(label: str, value) -> None:
    st.markdown(
        f'<div class="ccc-kpi"><div style="font-size:0.85rem;color:#0369a1;">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;">{value}</div></div>',
        unsafe_allow_html=True,
    )
