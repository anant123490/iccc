"""Reusable Streamlit cards and helpers."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO

import streamlit as st
from PIL import Image

from .theme import ICDAS_SOFT_COLORS


def show_disclaimer(text: str) -> None:
    st.markdown(
        f'<div class="ccc-disclaimer">{escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand(audience: str = "Patient workspace") -> None:
    st.sidebar.markdown(
        '<div class="ccc-sidebar-brand">'
        '<div class="ccc-brand-mark">IC</div>'
        '<div class="ccc-brand-copy">'
        '<span class="ccc-brand-name">ICDAS CARE</span>'
        f'<span class="ccc-brand-caption">{escape(audience)}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def page_heading(title: str, subtitle: str = "", eyebrow: str = "Clinical workspace") -> None:
    subtitle_html = f'<p class="ccc-page-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        '<div class="ccc-page-head">'
        f'<div class="ccc-eyebrow">{escape(eyebrow)}</div>'
        f'<h1 class="ccc-page-title">{escape(title)}</h1>'
        f'{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def b64_image(data: str | None, caption: str | None = None, width: int | None = 220):
    if not data:
        st.caption("No image")
        return
    raw = base64.b64decode(data)
    img = Image.open(BytesIO(raw))
    st.image(img, caption=caption, width=width)


def icdas_badge(grade: int) -> str:
    grade = int(grade)
    background, foreground = ICDAS_SOFT_COLORS.get(grade, ("#eef0ed", "#536477"))
    return (
        f'<span class="icdas-badge icdas-badge-{grade}" '
        f'style="--grade-bg:{background};--grade-ink:{foreground};">'
        f'ICDAS {grade}</span>'
    )


def kpi(label: str, value) -> None:
    value_text = "—" if value is None else str(value)
    st.markdown(
        '<div class="ccc-kpi">'
        f'<div class="ccc-kpi-label">{escape(str(label))}</div>'
        f'<div class="ccc-kpi-value">{escape(value_text)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
