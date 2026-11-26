"""Shared premium styling tokens for the ICDAS Streamlit portals."""

from __future__ import annotations


CSS = """
<style>
:root {
  --ink: #17283b;
  --ink-soft: #536477;
  --ink-muted: #778596;
  --navy: #14283d;
  --navy-deep: #102236;
  --teal: #397e7b;
  --teal-dark: #286967;
  --teal-soft: #e7f0ed;
  --ivory: #f6f5f1;
  --paper: #fffefa;
  --line: #e5e5de;
  --champagne: #efe7d8;
  --shadow: 0 10px 30px rgba(20, 40, 61, .055);
}

html, body, [class*="css"], button, input, textarea, select {
  font-family: Inter, "Aptos", "Segoe UI", Arial, sans-serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--ivory) !important;
  color: var(--ink) !important;
}
[data-testid="stHeader"] {
  background: rgba(246, 245, 241, .94) !important;
  border-bottom: 1px solid rgba(229, 229, 222, .65);
}
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1460px;
  padding: 2.1rem clamp(1.1rem, 4vw, 3.6rem) 3.2rem;
}
h1, h2, h3, h4, p, label, span, div { color: inherit; }
h1, h2, h3 { color: var(--ink) !important; letter-spacing: -.025em; }
h1 { font-size: clamp(1.7rem, 2.4vw, 2.2rem) !important; font-weight: 680 !important; }
h2 { font-size: 1.35rem !important; font-weight: 650 !important; }
h3 { font-size: 1.1rem !important; font-weight: 650 !important; }
p, [data-testid="stCaptionContainer"] { color: var(--ink-soft); }

[data-testid="stSidebar"] {
  background: var(--paper) !important;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
  background: var(--paper) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: 1.35rem 1rem 1.5rem;
}
[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--ink-muted) !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p { color: var(--ink-soft) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #fffefa !important;
  color: var(--ink) !important;
  border-color: rgba(255, 255, 255, .16) !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: .22rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border: 1px solid transparent;
  border-radius: 10px;
  padding: .48rem .62rem;
  transition: background .18s ease, border-color .18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255, 255, 255, .07);
  border-color: rgba(255, 255, 255, .08);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  color: var(--teal-dark) !important;
  background: var(--teal-soft);
  border-color: #d5e4de;
  box-shadow: inset 3px 0 0 #74a99f;
}

.ccc-sidebar-brand { display:flex; align-items:center; gap:.72rem; margin:.1rem 0 1.35rem; }
.ccc-brand-mark {
  display:grid; place-items:center; width:2.55rem; height:2.55rem; flex:none;
  border:1px solid #d2e2db; border-radius:12px;
  color:#286967; font-size:.82rem; font-weight:750; letter-spacing:.04em;
  background:#e7f0ed;
}
.ccc-brand-copy { display:flex; flex-direction:column; gap:.12rem; }
.ccc-brand-name { color:var(--ink); font-size:.84rem; font-weight:720; letter-spacing:.055em; }
.ccc-brand-caption { color:var(--ink-muted); font-size:.66rem; letter-spacing:.095em; text-transform:uppercase; }

.ccc-page-head { margin:0 0 1.45rem; }
.ccc-eyebrow { color:var(--teal-dark); font-size:.68rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase; margin:0 0 .48rem; }
.ccc-page-title { color:var(--ink); font-size:clamp(1.7rem, 2.6vw, 2.25rem); line-height:1.12; font-weight:690; letter-spacing:-.035em; margin:0; }
.ccc-page-subtitle { color:var(--ink-soft); max-width:760px; margin:.62rem 0 0; line-height:1.6; font-size:.98rem; }
.ccc-hero {
  position:relative; overflow:hidden; background:var(--paper); color:var(--ink);
  border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:20px;
  padding:clamp(1.5rem, 3.7vw, 2.7rem); margin:.4rem 0 1.45rem; box-shadow:var(--shadow);
}
.ccc-hero h1 { color:var(--ink) !important; max-width:850px; margin:.25rem 0 .7rem; font-size:clamp(1.8rem, 3vw, 2.65rem) !important; line-height:1.12; }
.ccc-hero p { color:var(--ink-soft) !important; max-width:710px; margin:0; font-size:1.04rem; line-height:1.65; }
.ccc-hero-meta { display:flex; align-items:center; flex-wrap:wrap; gap:.55rem; margin-top:1.2rem; }
.ccc-pill { display:inline-flex; align-items:center; border:1px solid #dfe6df; border-radius:999px; padding:.32rem .72rem; background:#f8f8f3; color:#435568; font-size:.76rem; font-weight:650; }
.ccc-pill-teal { background:var(--teal-soft); border-color:#d2e4de; color:#28645f; }
.ccc-disclaimer {
  background:#f1f4ef; border:1px solid #e1e7dc; border-left:3px solid #93a995;
  color:#4d5e59 !important; padding:.68rem .9rem; border-radius:10px; margin:.1rem 0 1.05rem;
  font-size:.83rem; line-height:1.5;
}
.ccc-disclaimer * { color:inherit !important; }
.ccc-panel { background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 5px 18px rgba(20,40,61,.035); }
.ccc-kpi {
  min-height:105px; background:var(--paper); border:1px solid var(--line); border-radius:15px;
  padding:1rem 1.05rem; box-shadow:0 5px 17px rgba(20,40,61,.035); border-top:2px solid #bbd0c7;
}
.ccc-kpi-label { color:var(--ink-muted) !important; font-size:.73rem; font-weight:720; letter-spacing:.08em; text-transform:uppercase; }
.ccc-kpi-value { color:var(--ink) !important; font-size:1.7rem; font-weight:710; letter-spacing:-.035em; margin-top:.4rem; line-height:1.2; overflow-wrap:anywhere; }
.ccc-kpi-note { color:var(--ink-soft) !important; font-size:.75rem; margin-top:.25rem; }
.ccc-step { color:#5f716e; background:#f1f3ed; border:1px solid #e3e7df; border-radius:10px; padding:.68rem .78rem; font-size:.8rem; font-weight:650; }
.ccc-step-active { color:#255f5b; background:#e5efeb; border-color:#cdded7; }
.ccc-workflow-strip { display:flex; gap:.55rem; flex-wrap:wrap; margin:.75rem 0 1.15rem; }
.ccc-workflow-step { flex:1 1 125px; display:flex; flex-direction:column; gap:.28rem; min-width:110px; padding:.68rem .76rem; border:1px solid #e2e5de; border-radius:11px; background:#fafaf6; }
.ccc-workflow-current { background:#e5efeb; border-color:#cbded6; box-shadow:inset 0 2px 0 #5e9187; }
.ccc-workflow-done { background:#f1f4ee; border-color:#dce4da; }
.ccc-workflow-title { color:var(--ink) !important; font-size:.78rem; font-weight:680; }
.ccc-workflow-state { color:var(--ink-muted) !important; font-size:.67rem; letter-spacing:.05em; text-transform:uppercase; }
.ccc-status { display:flex; flex-direction:column; gap:.2rem; color:#315f58; background:#e7f0ed; border:1px solid #d2e4de; border-radius:10px; padding:.58rem .68rem; font-size:.73rem; line-height:1.45; }
.ccc-demo-banner { color:#684c32; background:#f2ece0; border:1px solid #e6dac4; border-left:3px solid #b79563; border-radius:10px; padding:.7rem .9rem; margin:.65rem 0 1rem; font-size:.84rem; line-height:1.5; }
.ccc-demo-banner strong { color:#5b412c; }
.ccc-step-number { color:var(--teal-dark); font-size:.68rem; font-weight:760; letter-spacing:.13em; margin-bottom:.45rem; }
.ccc-result-stage { color:var(--ink); margin-left:.55rem; font-size:1.06rem; font-weight:650; }
.ccc-image-label { color:var(--ink-muted); font-size:.67rem; font-weight:760; letter-spacing:.12em; margin:.25rem 0 .5rem; }
.ccc-confidence-label { color:var(--ink-muted); font-size:.7rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.ccc-confidence-value { color:var(--teal-dark); font-size:1.25rem; font-weight:720; margin:.2rem 0 .35rem; }
.ccc-quality { display:flex; justify-content:space-between; align-items:center; gap:1rem; border-radius:10px; padding:.7rem .9rem; margin:.45rem 0 .7rem; border:1px solid #e5e5de; color:var(--ink); }
.ccc-quality span { color:var(--ink-soft); font-size:.8rem; }
.ccc-quality-pass { background:#eaf1eb; border-color:#d4e2d7; }
.ccc-quality-review { background:#f3efe5; border-color:#e8dfca; }
.ccc-quality-retry { background:#f3e9e5; border-color:#ead8d0; }
.ccc-grade-strip { display:flex; gap:.35rem; flex-wrap:wrap; margin:.45rem 0 .15rem; }
.ccc-grade-count { display:inline-flex; align-items:center; gap:.32rem; border:1px solid #e4e6df; border-radius:8px; background:#fafaf6; padding:.22rem .4rem; color:#607082; font-size:.72rem; }
.ccc-grade-count b { color:var(--ink); font-weight:700; }
.icdas-badge { display:inline-flex; align-items:center; color:var(--grade-ink) !important; background:var(--grade-bg); border:1px solid color-mix(in srgb, var(--grade-ink) 15%, white); border-radius:999px; padding:.38rem .74rem; font-size:.76rem; font-weight:760; letter-spacing:.045em; white-space:nowrap; }
.icdas-badge * { color:inherit !important; }

.stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] button {
  border-radius:10px; border:1px solid #d7ded8; background:#fffefa; color:var(--ink);
  font-weight:650; min-height:2.55rem; transition:transform .16s ease, box-shadow .16s ease, background .16s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  color:var(--teal-dark); border-color:#8fb3a8; background:#fbfcf8; transform:translateY(-1px); box-shadow:0 5px 14px rgba(20,40,61,.07);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:var(--teal-dark); border-color:var(--teal-dark); color:#fffefa !important;
}
.stButton > button[kind="primary"] *, .stFormSubmitButton > button[kind="primary"] * { color:#fffefa !important; opacity:1 !important; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { background:#225b58; border-color:#225b58; color:#fff; }
.stButton > button:disabled, .stFormSubmitButton > button:disabled {
  background:#e8ece7 !important; border-color:#dfe4dc !important; color:#62706c !important; opacity:1 !important;
}
[data-testid="stFileUploader"] section { border:1px dashed #aabbb2; border-radius:16px; background:#fbfcf8; padding:1.15rem; }
[data-testid="stFileUploader"] section:hover { border-color:#629188; background:#f4f8f4; }
[data-testid="stFileUploader"] button { border-radius:9px; }
[data-testid="stForm"] { background:var(--paper); border:1px solid var(--line); border-radius:15px; padding:1.05rem 1.1rem; }
[data-testid="stImage"] img { border:1px solid #e4e6df; border-radius:12px; background:#f3f4ef; }
input[type="radio"], input[type="checkbox"] { accent-color:var(--teal); }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea, [data-testid="stDateInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background:var(--paper); border-color:#dfe2db; border-radius:9px; color:var(--ink);
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#65978e; box-shadow:0 0 0 1px #65978e; }
[data-testid="stProgressBar"] > div > div { background-color:var(--teal) !important; border-radius:999px; }
[data-testid="stProgressBar"] > div { background:#e4e9e3; border-radius:999px; }
[data-testid="stMetric"] { background:var(--paper); border:1px solid var(--line); border-radius:13px; padding:.9rem 1rem; box-shadow:0 4px 13px rgba(20,40,61,.025); }
[data-testid="stMetricLabel"] { color:var(--ink-muted) !important; }
[data-testid="stMetricValue"] { color:var(--ink) !important; font-weight:700; }
[data-testid="stAlert"] { border-radius:12px; border:1px solid #e8e3d8; background:#f8f6f0; color:#34475a; }
[data-testid="stAlert"] p, [data-testid="stAlert"] span { color:#34475a !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line) !important; border-radius:16px !important; background:var(--paper); box-shadow:0 5px 16px rgba(20,40,61,.03); }
[data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom-color:#e3e5de; }
[data-testid="stTabs"] button[role="tab"] { color:#647486; border-radius:9px 9px 0 0; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--teal-dark); border-bottom-color:var(--teal); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(255,254,250,.8); }
div[data-testid="stExpander"] summary { color:var(--ink); font-weight:620; }
hr { border-color:#e5e5de; }
.stPlotlyChart, [data-testid="stPyplotGlobalUseContainer"] { background:transparent; }

@media (max-width: 850px) {
  [data-testid="stMainBlockContainer"], .block-container { padding:1.3rem 1rem 2.2rem; }
  .ccc-hero { border-radius:15px; padding:1.45rem; }
  .ccc-kpi { min-height:88px; }
  .ccc-quality { align-items:flex-start; flex-direction:column; gap:.25rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior:auto !important; transition:none !important; }
}
</style>
"""


def apply() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


ICDAS_COLORS = {
    0: "#4b8a79",
    1: "#91a979",
    2: "#bc9b62",
    3: "#bd7a5f",
    4: "#a95558",
}
ICDAS_SOFT_COLORS = {
    0: ("#e3efe8", "#2d6658"),
    1: ("#edf1e3", "#566a42"),
    2: ("#f3eddf", "#7a6031"),
    3: ("#f4e8e0", "#87513d"),
    4: ("#f2e3e2", "#813f46"),
}
