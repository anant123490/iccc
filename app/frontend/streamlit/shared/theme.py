"""Dental healthcare theme for Streamlit portals."""

from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; font-size: 17px; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.ccc-hero {
  background: linear-gradient(120deg, #0369a1 0%, #0ea5e9 55%, #22c55e 100%);
  color: white; padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;
}
.ccc-hero h1 { margin: 0 0 6px 0; font-size: 2rem; }
.ccc-card {
  background: var(--secondary-background-color, #fff);
  border: 1px solid #dbeafe; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
}
.ccc-kpi { background: #f0f9ff; border-radius: 14px; padding: 16px; border: 1px solid #bae6fd; }
.ccc-disclaimer {
  background: #ecfdf5; border-left: 5px solid #16a34a; padding: 12px 14px; border-radius: 8px;
  font-size: 0.95rem;
}
.icdas-0 { background:#16a34a; color:white; padding:4px 10px; border-radius:999px; }
.icdas-1 { background:#84cc16; color:#14532d; padding:4px 10px; border-radius:999px; }
.icdas-2 { background:#eab308; color:#422006; padding:4px 10px; border-radius:999px; }
.icdas-3 { background:#f97316; color:white; padding:4px 10px; border-radius:999px; }
.icdas-4 { background:#dc2626; color:white; padding:4px 10px; border-radius:999px; }
</style>
"""


def apply() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


ICDAS_COLORS = {0: "#16a34a", 1: "#84cc16", 2: "#eab308", 3: "#f97316", 4: "#dc2626"}
