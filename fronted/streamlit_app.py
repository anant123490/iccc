"""
ICDAS Dental AI — Streamlit frontend.

Classification is performed by the existing FastAPI + TensorFlow backend
(MobileNetV3-Small + CBAM + ordinal regression, ICDAS 0–4).
This file redesigns the UI only; prediction, Grad-CAM, history, and stats
continue to use the same API endpoints.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
LOW_CONFIDENCE_COPY = (
    "Low confidence prediction. Professional examination recommended."
)
DISCLAIMER = (
    "AI-assisted screening tool. This application is intended for "
    "educational, research, and screening purposes and should not replace "
    "examination or diagnosis by a qualified dental professional."
)
ACCEPTED_UPLOAD_TYPES = ["jpg", "jpeg", "png"]

PAGE_HOME = "Home"
PAGE_DETECTION = "Detection"
PAGE_ANALYSIS = "Analysis"
PAGE_HISTORY = "Scan History"
PAGE_GUIDE = "ICDAS Guide"
PAGE_ABOUT = "About"
TOP_NAV = [PAGE_HOME, PAGE_DETECTION, PAGE_GUIDE, PAGE_ABOUT]
SIDEBAR_NAV = [PAGE_HOME, PAGE_DETECTION, PAGE_ANALYSIS, PAGE_HISTORY, PAGE_GUIDE, PAGE_ABOUT]

# Official ICDAS 0–6 reference. Deployed model predicts ICDAS 0–4 only.
ICDAS_GUIDE = {
    0: {
        "title": "Sound tooth",
        "short": "Sound",
        "icon": "0",
        "description": "Sound tooth surface. No evidence of caries after visual inspection.",
        "in_model": True,
    },
    1: {
        "title": "First visual change",
        "short": "First visual change",
        "icon": "1",
        "description": "First visual change in enamel. Opacity or discoloration is typically visible after air drying.",
        "in_model": True,
    },
    2: {
        "title": "Distinct visual change",
        "short": "Distinct visual change",
        "icon": "2",
        "description": "Distinct visual change in enamel when wet. Demineralization is more established but remains non-cavitated.",
        "in_model": True,
    },
    3: {
        "title": "Localized enamel breakdown",
        "short": "Localized enamel breakdown",
        "icon": "3",
        "description": "Localized enamel breakdown due to caries, without visible dentin.",
        "in_model": True,
    },
    4: {
        "title": "Underlying dark shadow",
        "short": "Underlying dark shadow",
        "icon": "4",
        "description": "Underlying dark shadow from dentin, with or without localized enamel breakdown.",
        "in_model": True,
    },
    5: {
        "title": "Distinct cavity with visible dentin",
        "short": "Distinct cavity with visible dentin",
        "icon": "5",
        "description": "Distinct cavity with visible dentin. Reference only — not predicted by the current model.",
        "in_model": False,
    },
    6: {
        "title": "Extensive distinct cavity with visible dentin",
        "short": "Extensive distinct cavity",
        "icon": "6",
        "description": "Extensive distinct cavity with visible dentin. Reference only — not predicted by the current model.",
        "in_model": False,
    },
}
ICDAS_MODEL_GRADES = (0, 1, 2, 3, 4)
URGENCY_COLORS = {
    "LOW": "#19A7A8",
    "MODERATE": "#d97706",
    "HIGH": "#ef4444",
}

st.set_page_config(
    page_title="Dental AI — ICDAS Assessment",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


def backend_url() -> str:
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL).rstrip("/")


def set_page(name: str) -> None:
    st.session_state.nav_page = name
    st.rerun()


def _http_error_message(exc: requests.exceptions.HTTPError) -> str:
    response = exc.response
    if response is None:
        return "The AI backend returned an error."
    try:
        payload = response.json()
        detail = payload.get("detail", payload)
        if isinstance(detail, str):
            return detail
        return "The AI backend returned an error. Please try again."
    except Exception:
        text = (response.text or "").strip()
        if text and len(text) < 280 and "Traceback" not in text:
            return text
        return "The AI backend returned an error. Please try again."


def api_get(path: str, timeout: int = 8):
    try:
        response = requests.get(f"{backend_url()}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to the AI backend. Please ensure the FastAPI server is running."
    except requests.exceptions.Timeout:
        return None, "The AI backend timed out. Please try again."
    except requests.exceptions.HTTPError as exc:
        return None, _http_error_message(exc)
    except Exception:
        return None, "An unexpected error occurred while contacting the AI backend."


def api_post(path: str, **kwargs):
    try:
        response = requests.post(
            f"{backend_url()}{path}",
            timeout=kwargs.pop("timeout", 120),
            **kwargs,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to the AI backend. Please ensure the FastAPI server is running."
    except requests.exceptions.Timeout:
        return None, "The request timed out while contacting the AI backend."
    except requests.exceptions.HTTPError as exc:
        return None, _http_error_message(exc)
    except Exception:
        return None, "An unexpected error occurred while contacting the AI backend."


def b64_to_image(b64_string: str | None) -> Image.Image | None:
    if not b64_string:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_string)))
    except Exception:
        return None


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def icdas_severity_copy(grade: Any, result: dict | None = None) -> tuple[str, str]:
    try:
        grade_int = int(grade)
    except (TypeError, ValueError):
        return "Unknown", "The model did not return a valid ICDAS grade."
    if result:
        label = result.get("label") or ICDAS_GUIDE.get(grade_int, {}).get("title", "")
        description = result.get("description") or ICDAS_GUIDE.get(grade_int, {}).get("description", "")
        return str(label), str(description)
    info = ICDAS_GUIDE.get(grade_int)
    if info:
        return info["title"], info["description"]
    return f"ICDAS {grade_int}", "No description is available for this grade."


def probability_percent(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 1.0:
        return max(0.0, min(100.0, number * 100.0))
    return max(0.0, min(100.0, number))


def render_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@500;600;700&display=swap');

        :root {
            --bg: #121c26;
            --bg-alt: #17232e;
            --card: #1c2a36;
            --primary: #2f80ed;
            --secondary: #19a7a8;
            --text: #f5f7fa;
            --muted: #aab4be;
            --border: #2a3c4c;
            --shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
            --radius: 12px;
        }

        html, body, [class*="css"], .stApp, .stMarkdown, p, label, span {
            font-family: 'Source Sans 3', sans-serif;
        }
        .stApp { background: var(--bg); color: var(--text); }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer, .stDeployButton { visibility: hidden; }
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2.2rem;
            max-width: 1120px;
        }

        [data-testid="stSidebar"] {
            background: #0e1720;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        [data-testid="stSidebar"] hr { border-color: var(--border); }

        .brand-mark {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 1.28rem;
            font-weight: 700;
            margin: 0;
            color: var(--text);
        }
        .brand-sub {
            margin: 2px 0 0 0;
            color: var(--muted);
            font-size: 0.82rem;
        }
        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 0.82rem;
        }
        .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .dot-ok { background: #34d399; }
        .dot-off { background: #f87171; }

        .hero-card, .panel, .feature-card, .result-card, .guide-card,
        .privacy-card, .input-card, .step-card, .site-footer {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }
        .hero-card {
            padding: 32px 30px 28px 30px;
            margin-bottom: 18px;
            background: var(--bg-alt);
        }
        .hero-card h1 {
            font-family: 'IBM Plex Sans', sans-serif;
            margin: 0 0 8px 0;
            font-size: 2rem;
            letter-spacing: -0.03em;
            color: var(--text);
        }
        .hero-card .lede { color: var(--text); font-size: 1.05rem; margin: 0 0 10px 0; }
        .hero-card p { margin: 0; color: var(--muted); line-height: 1.55; }

        .feature-card, .guide-card, .input-card, .step-card {
            padding: 18px;
            height: 100%;
        }
        .feature-card h4, .guide-card h4, .input-card h3, .step-card h4 {
            margin: 0 0 8px 0;
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--text);
        }
        .feature-card p, .guide-card p, .input-card p, .step-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .guide-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: #243544;
            color: var(--secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .page-title {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 4px 0;
            color: var(--text);
        }
        .page-sub { color: var(--muted); margin: 0 0 18px 0; }
        .section-label {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 22px 0 12px 0;
            color: var(--text);
        }

        .icdas-big {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #7db4ff;
            margin: 6px 0;
            text-align: center;
        }
        .severity-label {
            text-align: center;
            color: var(--muted);
            font-weight: 600;
            margin-bottom: 12px;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #fff;
        }
        .chip {
            display: inline-block;
            background: #1a3344;
            color: #9fd8d9;
            border: 1px solid #2a5560;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 700;
        }
        .disclaimer {
            border: 1px solid #4a3d1f;
            background: #2a2416;
            color: #e7d7a8;
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 13px;
            margin-top: 12px;
        }
        .privacy-card {
            padding: 14px 16px;
            background: #173038;
            border-color: #2a5560;
        }
        .privacy-card p { color: #9fd8d9 !important; }
        .workflow {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px;
            text-align: center;
            font-weight: 600;
            color: var(--text);
        }
        .workflow span { color: var(--secondary); display: block; margin: 6px 0; }
        .or-divider {
            text-align: center;
            color: var(--muted);
            font-weight: 700;
            letter-spacing: 0.08em;
            padding-top: 72px;
        }
        .site-footer {
            margin-top: 28px;
            padding: 18px 20px;
            background: var(--bg-alt);
        }
        .site-footer h4 {
            margin: 0 0 4px 0;
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--text);
        }
        .site-footer p { color: var(--muted); margin: 0 0 8px 0; font-size: 0.88rem; }
        .site-footer .links { color: var(--muted); font-size: 0.85rem; }

        .stButton > button {
            border-radius: 8px;
            font-weight: 650;
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            background: #243544;
            color: var(--text);
        }
        .stButton > button[kind="primary"] {
            background: var(--primary);
            border-color: var(--primary);
            color: #fff;
        }
        [data-testid="stFileUploader"] {
            background: #16222c;
            border: 1px dashed #3b5568;
            border-radius: var(--radius);
            padding: 8px 12px 16px 12px;
        }
        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
        }
        [data-testid="stCaption"], .stCaption, small { color: var(--muted) !important; }
        img { border-radius: 8px; max-width: 100%; }
        .stProgress > div > div { background: var(--secondary); }
        [data-testid="stCameraInput"] {
            background: #16222c;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 8px;
        }
        div[data-testid="stAlert"] {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--text);
        }
        @media (max-width: 900px) {
            .hero-card h1 { font-size: 1.55rem; }
            .or-divider { padding-top: 8px; padding-bottom: 8px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    health, _ = api_get("/api/v1/health")
    model_loaded = bool(health and health.get("model_loaded"))

    with st.sidebar:
        st.markdown('<p class="brand-mark">Dental AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="brand-sub">Offline AI-Based Dental Caries Detection</p>',
            unsafe_allow_html=True,
        )
        st.radio("Pages", SIDEBAR_NAV, key="nav_page", label_visibility="collapsed")

        st.markdown("---")
        st.caption("AI Model Status")
        if model_loaded:
            st.markdown(
                '<div class="status-pill"><span class="dot dot-ok"></span>'
                "<strong>Model Loaded</strong></div>",
                unsafe_allow_html=True,
            )
        elif health is None:
            st.markdown(
                '<div class="status-pill"><span class="dot dot-off"></span>'
                "Backend offline</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-pill"><span class="dot dot-off"></span>'
                "Model not loaded</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Backend connection"):
            st.session_state.backend_url = st.text_input(
                "Backend URL",
                value=backend_url(),
                help="Local FastAPI inference server",
            )
            if health:
                st.caption(f"API status: {health.get('status', 'unknown')}")
                st.caption(f"Database: {'ok' if health.get('database_ok') else 'unavailable'}")
                groq = health.get("groq_configured")
                st.caption(
                    "Narrative reports: Groq configured"
                    if groq
                    else "Narrative reports: local fallback"
                )


def render_top_header() -> None:
    brand, nav = st.columns([1.35, 2])
    with brand:
        st.markdown(
            """
            <div>
                <p class="brand-mark">🦷 Dental AI</p>
                <p class="brand-sub">AI-Powered Dental Caries Screening</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav:
        cols = st.columns(len(TOP_NAV))
        current = st.session_state.get("nav_page", PAGE_HOME)
        for col, name in zip(cols, TOP_NAV):
            with col:
                btn_type = "primary" if current == name else "secondary"
                if st.button(name, type=btn_type, use_container_width=True, key=f"topnav_{name}"):
                    if current != name:
                        set_page(name)
    st.markdown(
        "<hr style='border-color:#2a3c4c;margin:4px 0 18px 0;'>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="site-footer">
            <h4>Dental AI</h4>
            <p>Offline AI-Based Dental Caries Detection using ICDAS Classification</p>
            <p class="links">Home · Detection · ICDAS Guide · About</p>
            <p>AI-assisted screening tool. This application is intended for educational,
            research, and screening purposes and should not replace examination or diagnosis
            by a qualified dental professional.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_privacy_card() -> None:
    st.markdown(
        """
        <div class="privacy-card">
            <strong>Privacy First</strong>
            <p style="margin:6px 0 0 0;font-size:0.88rem;">
            Images can be processed locally without sending them to an external server
            when using the offline inference mode (local FastAPI + TensorFlow).
            Optional narrative text may use Groq only if that service is configured.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def render_how_it_works() -> None:
    st.markdown('<p class="section-label">How It Works</p>', unsafe_allow_html=True)
    steps = [
        ("1", "Upload or capture", "Upload or capture an intraoral photograph."),
        ("2", "AI processing", "The image is processed by the AI model."),
        ("3", "ICDAS prediction", "The system predicts the ICDAS classification."),
        ("4", "Explainable result", "The result is shown with confidence and explainability."),
    ]
    cols = st.columns(4)
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="step-card"><div class="guide-icon">{num}</div>'
                f"<h4>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )


def render_dashboard() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>AI-Powered Dental Caries Detection</h1>
            <p class="lede">Intelligent ICDAS classification from intraoral images using artificial intelligence.</p>
            <p>Upload an intraoral dental image or capture an image using your camera to receive an
            AI-assisted ICDAS classification with confidence and explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Detection", type="primary"):
        set_page(PAGE_DETECTION)

    features = [
        ("ICDAS Classification", "Grades tooth images using ICDAS 0–4 from the trained ordinal model."),
        ("Edge AI Detection", "MobileNetV3-Small with CBAM attention for compact inference."),
        ("Offline & Private", "Inference runs on the local backend; images need not leave your machine."),
        ("Explainable AI", "Grad-CAM highlights regions that influenced the prediction."),
        ("Lesion Localization", "Contour overlays mark high-activation Grad-CAM regions."),
        ("Camera Compatible", "Upload a photograph or capture one with the device camera."),
    ]
    rows = [features[:3], features[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (title, body) in zip(cols, row):
            with col:
                st.markdown(
                    f'<div class="feature-card"><h4>{title}</h4><p>{body}</p></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<p class="section-label">Screening overview</p>', unsafe_allow_html=True)
    stats, error = api_get("/api/v1/stats")
    if error:
        st.info("Dashboard statistics will appear when the backend is available.")
        st.caption(error)
    elif not stats or stats.get("total_analyses", 0) == 0:
        st.info("No scan data available yet.")
    else:
        dist = stats.get("grade_distribution") or {}
        healthy = int(dist.get("0", 0) or 0)
        total = int(stats["total_analyses"])
        caries = max(0, total - healthy)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans", total)
        c2.metric("Healthy", healthy)
        c3.metric("Caries Detected", caries)
        c4.metric("Average Confidence", f"{stats['average_confidence']:.1f}%")

    st.markdown("")
    render_privacy_card()
    render_disclaimer()


def render_probability_bars(probabilities: Any) -> None:
    if not probabilities:
        st.caption("Class probabilities were not returned for this prediction.")
        return
    if isinstance(probabilities, dict):
        items = sorted(
            probabilities.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0,
        )
    else:
        items = [(str(i), p) for i, p in enumerate(probabilities)]
    st.markdown("**Prediction probability**")
    for key, value in items:
        pct = probability_percent(value)
        st.caption(f"ICDAS {key}  ·  {pct:.1f}%")
        st.progress(min(1.0, pct / 100.0))


def render_results(result: dict, original: Image.Image) -> None:
    grade = result.get("icdas_grade")
    confidence = float(result.get("confidence") or 0)
    urgency = str(result.get("urgency") or "").upper()
    label, description = icdas_severity_copy(grade, result)
    color = URGENCY_COLORS.get(urgency, "#334155")

    st.markdown('<p class="section-label">AI Analysis Result</p>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1.35])
    with left:
        st.markdown('<div class="result-card" style="padding:22px 18px;">', unsafe_allow_html=True)
        st.markdown("**ICDAS Classification**")
        st.markdown(f'<div class="icdas-big">ICDAS {grade}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="severity-label">Severity: {label}</div>', unsafe_allow_html=True)
        if urgency:
            st.markdown(
                f'<div style="text-align:center"><span class="badge" style="background:{color}">{urgency}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown("**Confidence**")
        st.progress(min(1.0, max(0.0, confidence / 100.0)))
        st.markdown(f"**{confidence:.1f}%**")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(description)
        if result.get("low_confidence"):
            st.warning(result.get("low_confidence_message") or LOW_CONFIDENCE_COPY)

    with right:
        render_probability_bars(result.get("probabilities"))
        if result.get("finding"):
            st.markdown("**Finding**")
            st.write(result["finding"])
        if result.get("recommendation"):
            st.markdown("**Recommendation**")
            st.write(result["recommendation"])
        if result.get("action"):
            st.caption(f"Suggested action: {result['action']}")

    heatmap = b64_to_image(result.get("heatmap_base64"))
    overlay = b64_to_image(result.get("overlay_base64"))
    contour = b64_to_image(result.get("contour_base64"))

    st.markdown('<p class="section-label">AI Explainability</p>', unsafe_allow_html=True)
    st.caption(
        "Highlighted regions represent areas that contributed to the model's prediction. "
        "Grad-CAM is an attention visualization, not an exact clinical lesion boundary."
    )
    if heatmap or overlay:
        g1, g2, g3 = st.columns(3)
        g1.image(original, caption="Original Image", use_container_width=True)
        g2.image(overlay or heatmap, caption="Grad-CAM Overlay", use_container_width=True)
        if overlay and heatmap and overlay.tobytes() != heatmap.tobytes():
            g3.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
        elif contour:
            g3.image(contour, caption="Attention / Detected Region", use_container_width=True)
        else:
            g3.info("A separate raw heatmap is not provided; the overlay is shown.")
    else:
        st.info("Grad-CAM is unavailable for this prediction.")

    st.markdown('<p class="section-label">Lesion Localization</p>', unsafe_allow_html=True)
    if contour:
        l1, l2, l3 = st.columns(3)
        l1.image(original, caption="Original Image", use_container_width=True)
        l2.image(contour, caption="Localization Overlay", use_container_width=True)
        with l3:
            st.caption("Detected region")
            st.write(
                "Contours are derived from high-activation Grad-CAM areas. "
                "They indicate model attention, not a confirmed clinical lesion outline."
            )
    else:
        st.info("Lesion localization is not available for this prediction.")

    if result.get("report"):
        with st.expander("Narrative assessment"):
            st.write(result["report"])

    render_disclaimer()


def _open_image(image_bytes: bytes) -> Image.Image | None:
    if not image_bytes:
        st.error("No image selected. Please upload a JPG, JPEG, or PNG file.")
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid image. Please upload a JPG, JPEG, or PNG file.")
    except OSError:
        st.error("The image appears to be corrupt or incomplete. Please try another file.")
    except Exception:
        st.error("Unable to read the selected image. Please try a different JPG or PNG file.")
    return None


def clear_analysis_state() -> None:
    for key in ("last_result", "last_image", "result_image_hash", "active_source"):
        st.session_state.pop(key, None)


def run_prediction(
    image_file: Any,
    image_bytes: bytes,
    original: Image.Image,
    include_explainability: bool,
    source: str,
) -> None:
    current_hash = image_hash(image_bytes)
    with st.status("Analyzing image...", expanded=True) as status:
        st.write("Preparing image...")
        st.write("Running AI model...")
        result, error = api_post(
            "/api/v1/predict",
            files={
                "file": (
                    getattr(image_file, "name", f"{source}.jpg"),
                    image_bytes,
                    "image/jpeg",
                )
            },
            params={"include_explainability": str(include_explainability).lower()},
        )
        if error:
            status.update(label="Analysis failed", state="error")
            lowered = error.lower()
            if "model" in lowered and ("load" in lowered or "not found" in lowered):
                st.error("The AI model could not be loaded. Please check the backend model path.")
            elif "unavailable" in lowered or "503" in lowered:
                st.error("The prediction service is currently unavailable. Please try again.")
            else:
                st.error(error)
            return
        st.write("Analyzing dental features...")
        st.write("Generating ICDAS prediction...")
        st.write("Preparing explanation...")
        status.update(label="Prediction complete", state="complete")

    st.session_state["last_result"] = result
    st.session_state["last_image"] = original
    st.session_state["result_image_hash"] = current_hash
    st.session_state["active_source"] = source


def render_image_source_panel(
    *,
    source: str,
    image_file: Any,
    include_explainability: bool,
    analyze_label: str,
    remove_label: str,
) -> None:
    if image_file is None:
        return

    image_bytes = image_file.getvalue()
    original = _open_image(image_bytes)
    if original is None:
        return

    current_hash = image_hash(image_bytes)
    st.markdown("**Image preview**")
    st.image(
        original,
        caption=getattr(image_file, "name", "Captured image"),
        use_container_width=True,
    )

    a1, a2 = st.columns(2)
    analyze = a1.button(analyze_label, type="primary", key=f"analyze_{source}")
    remove = a2.button(remove_label, key=f"remove_{source}")

    if remove:
        clear_analysis_state()
        st.rerun()

    if analyze:
        run_prediction(image_file, image_bytes, original, include_explainability, source)

    if (
        st.session_state.get("last_result")
        and st.session_state.get("result_image_hash") == current_hash
        and st.session_state.get("active_source") == source
    ):
        render_results(
            st.session_state["last_result"],
            st.session_state.get("last_image", original),
        )


def render_detection() -> None:
    st.markdown('<p class="page-title">Dental Caries Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Choose one method: upload an existing photograph, or capture a new image with your camera.</p>',
        unsafe_allow_html=True,
    )

    include_cam = st.checkbox("Generate Grad-CAM and lesion localization", value=True)

    left, mid, right = st.columns([1, 0.12, 1])

    with left:
        st.markdown(
            """
            <div class="input-card">
                <h3>Upload Dental Photograph</h3>
                <p>Select an existing intraoral photograph from your device.</p>
                <p style="margin-top:10px;"><strong>Upload Photo</strong></p>
                <p>Drag and drop your dental photograph here or browse your device. Accepted formats: JPG, JPEG, PNG.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Browse device",
            type=ACCEPTED_UPLOAD_TYPES,
            help="Accepted formats: JPG, JPEG, PNG",
            key="upload_photo_input",
        )
        if uploaded is None:
            st.caption("No image selected yet.")
        render_image_source_panel(
            source="upload",
            image_file=uploaded,
            include_explainability=include_cam,
            analyze_label="Analyze Image",
            remove_label="Remove Image",
        )

    with mid:
        st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="input-card">
                <h3>Capture Using Camera</h3>
                <p>Use your device camera to capture a new intraoral photograph.</p>
                <p style="margin-top:10px;"><strong>Live Camera Capture</strong></p>
                <p>Open the camera, capture a photograph, preview it, then analyze or retake.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            camera = st.camera_input(
                "Open camera and capture",
                key="live_camera_input",
            )
        except Exception:
            camera = None
            st.error(
                "Camera unavailable. Please allow camera permission in your browser, "
                "or use Upload Photo instead."
            )
        if camera is None:
            st.caption("No photograph captured yet.")
        render_image_source_panel(
            source="camera",
            image_file=camera,
            include_explainability=include_cam,
            analyze_label="Analyze Image",
            remove_label="Retake Photograph",
        )

    render_how_it_works()


def render_analysis() -> None:
    st.markdown('<p class="page-title">Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Stored screening statistics from previous scans.</p>',
        unsafe_allow_html=True,
    )
    stats, error = api_get("/api/v1/stats")
    if error:
        st.warning(error)
        return
    if not stats or stats.get("total_analyses", 0) == 0:
        st.info("No scan data available yet.")
        return

    dist = stats.get("grade_distribution") or {}
    healthy = int(dist.get("0", 0) or 0)
    total = int(stats["total_analyses"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", total)
    c2.metric("Healthy", healthy)
    c3.metric("Caries Detected", max(0, total - healthy))
    c4.metric("Average Confidence", f"{stats['average_confidence']:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**ICDAS distribution**")
        st.bar_chart(pd.Series({f"ICDAS {k}": v for k, v in dist.items()}))
    with right:
        st.markdown("**Confidence distribution**")
        st.bar_chart(pd.Series(stats.get("confidence_buckets") or {}))

    common = stats.get("most_common_grade")
    st.caption(
        f"Most common grade: ICDAS {common}" if common is not None else "Most common grade: —"
    )
    render_disclaimer()


def render_history() -> None:
    st.markdown('<p class="page-title">Scan History</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Previous ICDAS assessments stored by the local backend.</p>',
        unsafe_allow_html=True,
    )
    rows, error = api_get("/api/v1/history")
    if error:
        st.warning(error)
        return
    if not rows:
        st.info("No scan data available yet.")
        return

    filter_labels = ["All"] + [f"ICDAS {i}" for i in ICDAS_MODEL_GRADES]
    selected_filter = st.radio("Filter", filter_labels, horizontal=True)
    if selected_filter != "All":
        grade = int(selected_filter.split()[-1])
        rows = [row for row in rows if row.get("icdas_grade") == grade]
        if not rows:
            st.info(f"No stored scans for {selected_filter}.")
            return

    table = pd.DataFrame(
        [
            {
                "Date": datetime.fromisoformat(r["created_at"].replace("Z", "")).strftime(
                    "%d %b %Y %H:%M"
                )
                if r.get("created_at")
                else "",
                "ICDAS Grade": f"ICDAS {r['icdas_grade']}",
                "Confidence": f"{r['confidence']:.1f}%",
                "Status": r.get("urgency") or "",
                "id": r["id"],
            }
            for r in rows
        ]
    )
    st.dataframe(table.drop(columns=["id"]), use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Open record",
        options=table["id"].tolist(),
        format_func=lambda i: f"Scan #{i}",
    )
    if not selected:
        return

    detail, err = api_get(f"/api/v1/history/{selected}")
    if err:
        st.error(err)
        return

    original = b64_to_image(detail.get("image_base64"))
    if original is None:
        st.caption("Original image is not available for this record.")
        original = Image.new("RGB", (224, 224), color=(28, 42, 54))

    payload = {
        **detail,
        "label": f"ICDAS {detail['icdas_grade']}",
        "description": detail.get("finding"),
        "heatmap_base64": detail.get("heatmap_base64"),
        "overlay_base64": detail.get("heatmap_base64"),
        "contour_base64": detail.get("contour_base64"),
        "probabilities": detail.get("probabilities"),
    }
    if isinstance(payload.get("probabilities"), str):
        try:
            payload["probabilities"] = json.loads(payload["probabilities"])
        except json.JSONDecodeError:
            payload["probabilities"] = None
    render_results(payload, original)


def render_icdas_guide() -> None:
    st.markdown('<p class="page-title">ICDAS Guide</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">International Caries Detection and Assessment System reference. '
        "This application classifies ICDAS 0–4.</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for grade, info in ICDAS_GUIDE.items():
        with cols[grade % 2]:
            scope = (
                "Classified by this model"
                if info["in_model"]
                else "Reference only (not predicted)"
            )
            st.markdown(
                f'<div class="guide-card"><div class="guide-icon">{info["icon"]}</div>'
                f"<h4>ICDAS {grade} → {info['short']}</h4>"
                f'<span class="chip">{scope}</span>'
                f'<p style="margin-top:8px">{info["description"]}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("")


def render_about() -> None:
    st.markdown('<p class="page-title">About Dental AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Offline AI-Based Dental Caries Detection using ICDAS Classification.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Dental caries problem")
    st.write(
        "Dental caries is a progressive disease that benefits from earlier visual detection. "
        "ICDAS provides a standardized scale for describing lesion severity from sound enamel "
        "through cavitation."
    )
    st.markdown("#### Purpose of the application")
    st.write(
        "This project supports educational and research screening of intraoral photographs. "
        "It classifies images into ICDAS grades 0–4 using a trained convolutional model and "
        "returns confidence, class probabilities, and optional explainability maps."
    )
    st.markdown("#### ICDAS classification")
    st.write(
        "The deployed model predicts ICDAS 0 (sound) through ICDAS 4 (underlying dentin shadow). "
        "ICDAS 5 and 6 are shown in the guide for completeness and are not model outputs."
    )
    st.markdown("#### AI-based detection")
    st.write(
        "A photograph is preprocessed, passed through the neural network, and converted into "
        "an ICDAS grade with class probabilities reconstructed from ordinal thresholds."
    )
    st.markdown("#### Offline processing")
    st.write(
        "When the FastAPI backend and TensorFlow model run on this machine, inference does not "
        "require an external vision API."
    )
    st.markdown("#### Explainable AI")
    st.write(
        "Grad-CAM highlights image regions that contributed to the predicted class. "
        "It is an explanation aid, not a clinical segmentation of the lesion."
    )
    st.markdown("#### Privacy")
    render_privacy_card()

    st.markdown("#### Workflow")
    st.markdown(
        """
        <div class="workflow">
            Image
            <span>↓</span>
            Preprocessing
            <span>↓</span>
            AI Model
            <span>↓</span>
            ICDAS Classification
            <span>↓</span>
            Confidence
            <span>↓</span>
            Explainability
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Model information")
    info, err = api_get("/api/v1/model/info")
    if err:
        st.info("Connect the backend to load live model details from `/api/v1/model/info`.")
        st.caption(err)
        st.write(
            "From this repository: MobileNetV3-Small, CBAM attention, ordinal regression, "
            "TensorFlow/Keras. ICDAS mode 0–4."
        )
    else:
        st.write(f"**Name:** {info.get('name')}")
        st.write(f"**Architecture:** {info.get('architecture')}")
        st.write(f"**ICDAS mode:** {info.get('icdas_mode')}")
        st.write(f"**Classes:** {info.get('num_classes')}")
        st.write(f"**Input size:** {info.get('image_size')}×{info.get('image_size')}")
        st.write(f"**Ordinal regression:** {'yes' if info.get('ordinal_regression') else 'no'}")
        st.caption("These values are reported by the running backend, not guessed by the UI.")

    render_disclaimer()


def main() -> None:
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND_URL
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = PAGE_HOME

    render_css()
    render_sidebar()
    render_top_header()

    page = st.session_state.nav_page
    if page == PAGE_HOME:
        render_dashboard()
    elif page == PAGE_DETECTION:
        render_detection()
    elif page == PAGE_ANALYSIS:
        render_analysis()
    elif page == PAGE_HISTORY:
        render_history()
    elif page == PAGE_GUIDE:
        render_icdas_guide()
    else:
        render_about()

    render_footer()


main()
