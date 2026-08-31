
"""
ICDAS Dental AI — Streamlit Frontend

Architecture
------------
Streamlit
    ↓
FastAPI /api/v1/predict
    ↓
InferenceEngine
    ↓
MobileNetV3-Small + CBAM
    ↓
ICDAS 0-4
    ↓
Streamlit displays the exact backend result

IMPORTANT
---------
The frontend NEVER calculates the ICDAS grade.

It always uses:

    result["icdas_grade"]

returned by FastAPI.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

ACCEPTED_UPLOAD_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp",
]

LOW_CONFIDENCE_COPY = (
    "Low confidence prediction. Professional examination recommended."
)

DISCLAIMER = (
    "AI-assisted screening tool. This application is intended for "
    "educational, research, and screening purposes and should not replace "
    "examination or diagnosis by a qualified dental professional."
)

PAGE_HOME = "Home"
PAGE_DETECTION = "Detection"
PAGE_ANALYSIS = "Analysis"
PAGE_HISTORY = "Scan History"
PAGE_LABEL = "Dataset Labeling"
PAGE_GUIDE = "ICDAS Guide"
PAGE_ABOUT = "About"

TOP_NAV = [
    PAGE_HOME,
    PAGE_DETECTION,
    PAGE_GUIDE,
    PAGE_ABOUT,
]

SIDEBAR_NAV = [
    PAGE_HOME,
    PAGE_DETECTION,
    PAGE_ANALYSIS,
    PAGE_LABEL,
    PAGE_HISTORY,
    PAGE_GUIDE,
    PAGE_ABOUT,
]

ICDAS_MODEL_GRADES = (0, 1, 2, 3, 4)

ICDAS_GUIDE = {
    0: {
        "title": "Sound tooth",
        "short": "Sound",
        "icon": "0",
        "description": (
            "Sound tooth surface. No evidence of caries after visual inspection."
        ),
        "in_model": True,
    },
    1: {
        "title": "First visual change",
        "short": "First visual change",
        "icon": "1",
        "description": (
            "First visual change in enamel. Opacity or discoloration "
            "is typically visible after air drying."
        ),
        "in_model": True,
    },
    2: {
        "title": "Distinct visual change",
        "short": "Distinct visual change",
        "icon": "2",
        "description": (
            "Distinct visual change in enamel. Demineralization is more "
            "established but remains non-cavitated."
        ),
        "in_model": True,
    },
    3: {
        "title": "Localized enamel breakdown",
        "short": "Localized enamel breakdown",
        "icon": "3",
        "description": (
            "Localized enamel breakdown due to caries, without visible dentin."
        ),
        "in_model": True,
    },
    4: {
        "title": "Underlying dark shadow",
        "short": "Underlying dark shadow",
        "icon": "4",
        "description": (
            "Underlying dark shadow from dentin, with or without localized "
            "enamel breakdown."
        ),
        "in_model": True,
    },
    5: {
        "title": "Distinct cavity with visible dentin",
        "short": "Distinct cavity with visible dentin",
        "icon": "5",
        "description": (
            "Reference only — not predicted by the current model."
        ),
        "in_model": False,
    },
    6: {
        "title": "Extensive distinct cavity with visible dentin",
        "short": "Extensive distinct cavity",
        "icon": "6",
        "description": (
            "Reference only — not predicted by the current model."
        ),
        "in_model": False,
    },
}

URGENCY_COLORS = {
    "LOW": "#19A7A8",
    "MODERATE": "#D97706",
    "HIGH": "#EF4444",
    "CRITICAL": "#DC2626",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dental AI — ICDAS Assessment",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    defaults = {
        "backend_url": DEFAULT_BACKEND_URL,
        "nav_page": PAGE_HOME,
        "_pending_nav_page": None,
        "last_result": None,
        "last_image": None,
        "result_image_hash": None,
        "active_source": None,
        "debug_backend": False,
        "label_index": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# BACKEND
# ============================================================

def backend_url() -> str:
    return str(
        st.session_state.get(
            "backend_url",
            DEFAULT_BACKEND_URL,
        )
    ).rstrip("/")


def api_get(path: str, timeout: int = 15):
    try:
        response = requests.get(
            f"{backend_url()}{path}",
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return (
            None,
            "Unable to connect to the AI backend. "
            "Please start FastAPI first.",
        )

    except requests.exceptions.Timeout:
        return (
            None,
            "The AI backend timed out. Please try again.",
        )

    except requests.exceptions.HTTPError as exc:
        response = exc.response
        if response is not None:
            try:
                payload = response.json()
                detail = payload.get(
                    "detail",
                    payload,
                )
                if isinstance(detail, str):
                    return None, detail
            except Exception:
                pass

        return (
            None,
            "The AI backend returned an HTTP error.",
        )

    except Exception as exc:
        return (
            None,
            f"Unexpected backend error: {exc}",
        )


def api_post(
    path: str,
    *,
    files=None,
    params=None,
    timeout: int = 120,
):
    try:
        response = requests.post(
            f"{backend_url()}{path}",
            files=files,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return (
            None,
            "Unable to connect to the AI backend. "
            "Please start FastAPI first.",
        )

    except requests.exceptions.Timeout:
        return (
            None,
            "The prediction request timed out.",
        )

    except requests.exceptions.HTTPError as exc:
        response = exc.response
        if response is not None:
            try:
                payload = response.json()
                detail = payload.get(
                    "detail",
                    payload,
                )
                if isinstance(detail, str):
                    return None, detail
            except Exception:
                pass

        return (
            None,
            "The prediction request failed.",
        )

    except Exception as exc:
        return (
            None,
            f"Unexpected backend error: {exc}",
        )


# ============================================================
# NAVIGATION
# ============================================================

def set_page(name: str) -> None:
    st.session_state["_pending_nav_page"] = name
    st.rerun()


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_image(data: bytes) -> Image.Image | None:
    if not data:
        st.error("No image was selected.")
        return None

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")

    except UnidentifiedImageError:
        st.error(
            "Invalid image. Please use JPG, JPEG, PNG, BMP or WEBP."
        )

    except OSError:
        st.error(
            "The image appears to be corrupt or incomplete."
        )

    except Exception:
        st.error(
            "Unable to read the selected image."
        )

    return None


def b64_to_image(
    b64_string: str | None,
) -> Image.Image | None:
    if not b64_string:
        return None

    try:
        raw = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")

    except Exception:
        return None


# ============================================================
# RESPONSE VALIDATION
# ============================================================

def normalize_prediction(result: Any) -> dict:
    """
    Validate the exact FastAPI prediction.

    The frontend NEVER calculates argmax.

    The backend has already calculated:

        icdas_grade

    and this function only validates/displays it.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Backend returned an invalid response."
        )

    if not isinstance(result, dict):
        raise ValueError(
            "Backend returned an invalid response."
        )

    mode = str(result.get("mode") or "")
    if result.get("icdas_grade") is None and mode in {
        "no_detection",
        "detector_unavailable",
    }:
        cleaned = dict(result)
        cleaned["icdas_grade"] = None
        cleaned["confidence"] = float(result.get("confidence") or 0.0)
        cleaned["probabilities"] = result.get("probabilities") or {}
        cleaned["mode"] = mode
        return cleaned

    if "icdas_grade" not in result:
        raise ValueError(
            "Backend response does not contain icdas_grade."
        )

    try:
        grade = int(result["icdas_grade"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Backend returned an invalid ICDAS grade."
        ) from exc

    if grade not in ICDAS_MODEL_GRADES:
        raise ValueError(
            f"Backend returned unsupported ICDAS grade {grade}."
        )

    try:
        confidence = float(
            result.get("confidence", 0.0)
        )
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(
        0.0,
        min(100.0, confidence),
    )

    probabilities = result.get(
        "probabilities",
        {},
    )

    if isinstance(probabilities, str):
        try:
            probabilities = json.loads(probabilities)
        except Exception:
            probabilities = {}

    normalized_probabilities = {}

    if isinstance(probabilities, dict):
        for class_id in ICDAS_MODEL_GRADES:
            value = probabilities.get(
                str(class_id),
                probabilities.get(class_id, 0.0),
            )

            try:
                normalized_probabilities[
                    str(class_id)
                ] = float(value)
            except (TypeError, ValueError):
                normalized_probabilities[
                    str(class_id)
                ] = 0.0

    cleaned = dict(result)

    cleaned["icdas_grade"] = grade
    cleaned["confidence"] = confidence
    cleaned["probabilities"] = normalized_probabilities

    return cleaned


# ============================================================
# SEVERITY
# ============================================================

def icdas_severity_copy(
    grade: Any,
    result: dict | None = None,
) -> tuple[str, str]:

    try:
        grade_int = int(grade)
    except (TypeError, ValueError):
        return (
            "Unknown",
            "The model did not return a valid ICDAS grade.",
        )

    info = ICDAS_GUIDE.get(
        grade_int,
        {},
    )

    label = (
        result.get("label")
        if result
        else None
    ) or info.get(
        "title",
        f"ICDAS {grade_int}",
    )

    description = (
        result.get("description")
        if result
        else None
    ) or info.get(
        "description",
        "No description is available.",
    )

    return (
        str(label),
        str(description),
    )


def probability_percent(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if number <= 1.0:
        number *= 100.0

    return max(
        0.0,
        min(100.0, number),
    )


# ============================================================
# CSS
# ============================================================

def render_css() -> None:
    st.markdown(
        """
        <style>

        :root {
            --bg: #121c26;
            --bg-alt: #17232e;
            --card: #1c2a36;
            --primary: #2f80ed;
            --secondary: #19a7a8;
            --text: #f5f7fa;
            --muted: #aab4be;
            --border: #2a3c4c;
            --shadow: 0 8px 24px rgba(0,0,0,0.22);
            --radius: 12px;
        }

        html, body, [class*="css"],
        .stApp, .stMarkdown, p, label, span {
            font-family: "Source Sans 3", sans-serif;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu,
        footer,
        .stDeployButton {
            visibility: hidden;
        }

        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2.2rem;
            max-width: 1120px;
        }

        [data-testid="stSidebar"] {
            background: #0e1720;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        .hero-card,
        .panel,
        .feature-card,
        .result-card,
        .guide-card,
        .privacy-card,
        .input-card,
        .step-card,
        .site-footer {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .hero-card {
            padding: 32px 30px 28px;
            margin-bottom: 18px;
            background: var(--bg-alt);
        }

        .hero-card h1 {
            font-size: 2rem;
            margin: 0 0 8px;
            color: var(--text);
            letter-spacing: -0.03em;
        }

        .hero-card .lede {
            color: var(--text);
            font-size: 1.05rem;
            margin: 0 0 10px;
        }

        .hero-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .feature-card,
        .guide-card,
        .input-card,
        .step-card {
            padding: 18px;
            height: 100%;
        }

        .feature-card h4,
        .guide-card h4,
        .input-card h3,
        .step-card h4 {
            margin: 0 0 8px;
            color: var(--text);
        }

        .feature-card p,
        .guide-card p,
        .input-card p,
        .step-card p {
            color: var(--muted);
            line-height: 1.5;
            font-size: 0.9rem;
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
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 4px;
            color: var(--text);
        }

        .page-sub {
            color: var(--muted);
            margin-bottom: 18px;
        }

        .section-label {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 22px 0 12px;
            color: var(--text);
        }

        .icdas-big {
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
            color: white;
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

        .privacy-card {
            padding: 14px 16px;
            background: #173038;
            border-color: #2a5560;
        }

        .privacy-card p {
            color: #9fd8d9 !important;
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

        .workflow {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px;
            text-align: center;
            font-weight: 600;
            color: var(--text);
        }

        .workflow span {
            color: var(--secondary);
            display: block;
            margin: 6px 0;
        }

        .or-divider {
            text-align: center;
            color: var(--muted);
            font-weight: 700;
            padding-top: 72px;
        }

        .site-footer {
            margin-top: 28px;
            padding: 18px 20px;
            background: var(--bg-alt);
        }

        .site-footer h4 {
            margin: 0 0 4px;
            color: var(--text);
        }

        .site-footer p {
            color: var(--muted);
            margin-bottom: 8px;
            font-size: 0.88rem;
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

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .dot-ok {
            background: #34d399;
        }

        .dot-off {
            background: #f87171;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 650;
            border: 1px solid var(--border);
            background: #243544;
            color: var(--text);
        }

        [data-testid="stFileUploader"] {
            background: #16222c;
            border: 1px dashed #3b5568;
            border-radius: var(--radius);
        }

        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        .stProgress > div > div {
            background: var(--secondary);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    health, error = api_get("/api/v1/health")

    model_loaded = bool(
        health and health.get("model_loaded")
    )

    with st.sidebar:

        st.markdown(
            '<p class="brand-mark">Dental AI</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="brand-sub">'
            "Offline AI-Based Dental Caries Detection"
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # IMPORTANT:
        # This widget itself controls nav_page.
        # We do not mutate nav_page after creating it.
        st.radio(
            "Pages",
            SIDEBAR_NAV,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.caption("AI Model Status")

        if model_loaded:

            st.markdown(
                '<div class="status-pill">'
                '<span class="dot dot-ok"></span>'
                "<strong>Model Loaded</strong>"
                "</div>",
                unsafe_allow_html=True,
            )

        elif health is None:

            st.markdown(
                '<div class="status-pill">'
                '<span class="dot dot-off"></span>'
                "Backend offline"
                "</div>",
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="status-pill">'
                '<span class="dot dot-off"></span>'
                "Model not loaded"
                "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Backend connection"):

            st.session_state["backend_url"] = (
                st.text_input(
                    "Backend URL",
                    value=backend_url(),
                    help="Local FastAPI inference server",
                )
            )

            if health:

                st.caption(
                    f"API status: {health.get('status', 'unknown')}"
                )

                st.caption(
                    "Database: "
                    + (
                        "ok"
                        if health.get("database_ok")
                        else "unavailable"
                    )
                )

                st.caption(
                    "Narrative reports: "
                    + (
                        "Groq configured"
                        if health.get("groq_configured")
                        else "local fallback"
                    )
                )

            elif error:

                st.caption(error)

        st.checkbox(
            "Backend debug",
            key="debug_backend",
            help="Show the exact JSON returned by FastAPI.",
        )


# ============================================================
# TOP HEADER
# ============================================================

def render_top_header() -> None:
    brand, nav = st.columns(
        [1.35, 2]
    )

    with brand:

        st.markdown(
            """
            <div>
                <p class="brand-mark">🦷 Dental AI</p>
                <p class="brand-sub">
                    AI-Powered Dental Caries Screening
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav:

        cols = st.columns(
            len(TOP_NAV)
        )

        current = st.session_state.get(
            "nav_page",
            PAGE_HOME,
        )

        for col, name in zip(
            cols,
            TOP_NAV,
        ):

            with col:

                button_type = (
                    "primary"
                    if current == name
                    else "secondary"
                )

                if st.button(
                    name,
                    type=button_type,
                    use_container_width=True,
                    key=f"topnav_{name}",
                ):

                    if current != name:
                        set_page(name)

    st.markdown(
        "<hr style='border-color:#2a3c4c;"
        "margin:4px 0 18px 0;'>",
        unsafe_allow_html=True,
    )


# ============================================================
# PRIVACY / DISCLAIMER / FOOTER
# ============================================================

def render_privacy_card() -> None:
    st.markdown(
        """
        <div class="privacy-card">
            <strong>Privacy First</strong>
            <p style="margin:6px 0 0 0;font-size:0.88rem;">
                Images are processed by the local FastAPI backend.
                Optional narrative text may use Groq only when configured.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        f"""
        <div class="disclaimer">
            {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="site-footer">
            <h4>Dental AI</h4>
            <p>
                Offline AI-Based Dental Caries Detection using ICDAS Classification.
            </p>
            <p>
                Home · Detection · ICDAS Guide · About
            </p>
            <p>
                AI-assisted screening tool for educational,
                research and screening purposes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOW IT WORKS
# ============================================================

def render_how_it_works() -> None:
    st.markdown(
        '<p class="section-label">How It Works</p>',
        unsafe_allow_html=True,
    )

    steps = [
        (
            "1",
            "Upload or capture",
            "Upload or capture an intraoral photograph.",
        ),
        (
            "2",
            "AI processing",
            "Image is sent to the local FastAPI backend.",
        ),
        (
            "3",
            "ICDAS prediction",
            "MobileNetV3-Small + CBAM predicts ICDAS 0-4.",
        ),
        (
            "4",
            "Explainable result",
            "Confidence, probabilities and Grad-CAM are displayed.",
        ),
    ]

    cols = st.columns(4)

    for col, (
        number,
        title,
        body,
    ) in zip(cols, steps):

        with col:

            st.markdown(
                f"""
                <div class="step-card">
                    <div class="guide-icon">{number}</div>
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# HOME
# ============================================================

def render_dashboard() -> None:

    # This MUST remain st.markdown with unsafe_allow_html.
    st.markdown(
        """
        <div class="hero-card">

            <h1>
                AI-Powered Dental Caries Detection
            </h1>

            <p class="lede">
                Intelligent ICDAS classification from
                intraoral images using artificial intelligence.
            </p>

            <p>
                Upload an intraoral dental image or capture an image
                using your camera to receive an AI-assisted ICDAS
                classification with confidence and explainability.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Start Detection",
        type="primary",
    ):
        set_page(PAGE_DETECTION)

    features = [
        (
            "ICDAS Classification",
            "Predicts ICDAS 0-4 using the trained classification model.",
        ),
        (
            "Edge AI Detection",
            "MobileNetV3-Small with CBAM attention for compact inference.",
        ),
        (
            "Offline & Private",
            "Inference runs through the local FastAPI backend.",
        ),
        (
            "Explainable AI",
            "Grad-CAM highlights regions influencing the prediction.",
        ),
        (
            "Lesion Localization",
            "Attention contours are derived from Grad-CAM.",
        ),
        (
            "Camera Compatible",
            "Upload a photograph or capture one with your device camera.",
        ),
    ]

    for row in (
        features[:3],
        features[3:],
    ):

        cols = st.columns(3)

        for col, (
            title,
            body,
        ) in zip(cols, row):

            with col:

                st.markdown(
                    f"""
                    <div class="feature-card">
                        <h4>{title}</h4>
                        <p>{body}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<p class="section-label">Screening overview</p>',
        unsafe_allow_html=True,
    )

    stats, error = api_get("/api/v1/stats")

    if error:

        st.info(
            "Dashboard statistics will appear when the backend is available."
        )
        st.caption(error)

    elif not stats or stats.get(
        "total_analyses",
        0,
    ) == 0:

        st.info(
            "No scan data available yet."
        )

    else:

        dist = stats.get(
            "grade_distribution",
            {},
        ) or {}

        healthy = int(
            dist.get("0", 0) or 0
        )

        total = int(
            stats.get(
                "total_analyses",
                0,
            )
        )

        caries = max(
            0,
            total - healthy,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Scans",
            total,
        )

        c2.metric(
            "Healthy",
            healthy,
        )

        c3.metric(
            "Caries Detected",
            caries,
        )

        c4.metric(
            "Average Confidence",
            f"{stats.get('average_confidence', 0):.1f}%",
        )

    st.markdown("")

    render_privacy_card()
    render_disclaimer()


# ============================================================
# PROBABILITY BARS
# ============================================================

def render_probability_bars(
    probabilities: dict,
) -> None:

    if not probabilities:

        st.caption(
            "Class probabilities were not returned."
        )

        return

    st.markdown("**Prediction probability**")

    for grade in ICDAS_MODEL_GRADES:

        pct = probability_percent(
            probabilities.get(
                str(grade),
                0.0,
            )
        )

        st.caption(
            f"ICDAS {grade} · {pct:.1f}%"
        )

        st.progress(
            min(
                1.0,
                pct / 100.0,
            )
        )


# ============================================================
# RESULT
# ============================================================

def render_results(
    result: dict,
    original: Image.Image,
) -> None:

    try:
        result = normalize_prediction(result)

    except ValueError as exc:

        st.error(
            f"Invalid backend prediction: {exc}"
        )

        return

    grade = result.get("icdas_grade")
    confidence = result["confidence"]

    st.caption(
        result.get("ai_assisted_note")
        or "AI-assisted assessment — not a definitive clinical diagnosis."
    )
    quality = result.get("quality") or {}
    if quality.get("message"):
        st.info("Image quality: " + str(quality["message"]))
    if result.get("message"):
        st.warning(result["message"])
    if result.get("low_confidence") and result.get("low_confidence_message"):
        st.warning(result["low_confidence_message"])

    annotated = result.get("annotated_image_base64")
    regions = result.get("regions") or []

    label, description = (
        icdas_severity_copy(
            grade,
            result,
        )
        if grade is not None
        else (
            result.get("label") or "No localized region",
            result.get("description") or "",
        )
    )

    urgency = str(
        result.get("urgency", "")
        or ""
    ).upper()

    urgency_color = URGENCY_COLORS.get(
        urgency,
        "#334155",
    )

    st.markdown(
        '<p class="section-label">AI Analysis Result</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.get(
        "debug_backend",
        False,
    ):

        with st.expander(
            "Exact FastAPI response",
        ):

            st.json(result)

    left, right = st.columns(
        [1.15, 1.35]
    )

    with left:

        st.markdown(
            '<div class="result-card" style="padding:22px 18px;">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**ICDAS Classification**"
        )

        st.markdown(
            f"""
            <div class="icdas-big">
                {"No ROI" if grade is None else f"ICDAS {grade}"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="severity-label">
                {label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if urgency:

            st.markdown(
                f"""
                <div style="text-align:center">
                    <span
                        class="badge"
                        style="background:{urgency_color}"
                    >
                        {urgency}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        st.markdown("**Confidence**")

        st.progress(
            min(
                1.0,
                max(
                    0.0,
                    confidence / 100.0,
                ),
            )
        )

        st.markdown(
            f"**{confidence:.2f}%**"
        )

        if regions:
            st.markdown("**Localized decay regions**")
            st.caption(
                "Detector classes D/d are decay regions, not ICDAS grades."
            )
            for region in regions:
                st.markdown(
                    f"Region {region.get('region_id')}: "
                    f"caries detected ({region.get('detection_class')}) · "
                    f"ICDAS {region.get('icdas_grade')} · "
                    f"confidence {region.get('confidence')}"
                )
                roi_b64 = region.get("roi_base64")
                cam_b64 = region.get("heatmap_base64")
                cols = st.columns(2)
                if roi_b64:
                    roi_im = b64_to_image(roi_b64)
                    if roi_im is not None:
                        cols[0].image(
                            roi_im,
                            caption="Original ROI",
                            use_container_width=True,
                        )
                if cam_b64:
                    cam_im = b64_to_image(cam_b64)
                    if cam_im is not None:
                        cols[1].image(
                            cam_im,
                            caption="Grad-CAM",
                            use_container_width=True,
                        )

        if annotated:
            ann_im = b64_to_image(annotated)
            if ann_im is not None:
                st.image(
                    ann_im,
                    caption="Detector boxes (not FDI)",
                    use_container_width=True,
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.caption(description)

        if result.get(
            "low_confidence",
            False,
        ):

            st.warning(
                result.get(
                    "low_confidence_message",
                    LOW_CONFIDENCE_COPY,
                )
            )

    with right:

        render_probability_bars(
            result.get(
                "probabilities",
                {},
            )
        )

        if result.get("finding"):

            st.markdown("**Finding**")

            st.write(
                result["finding"]
            )

        if result.get("recommendation"):

            st.markdown("**Recommendation**")

            st.write(
                result["recommendation"]
            )

        if result.get("action"):

            st.caption(
                f"Suggested action: {result['action']}"
            )

    heatmap = b64_to_image(
        result.get(
            "heatmap_base64"
        )
    )

    overlay = b64_to_image(
        result.get(
            "overlay_base64"
        )
    )

    contour = b64_to_image(
        result.get(
            "contour_base64"
        )
    )

    st.markdown(
        '<p class="section-label">AI Explainability</p>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Highlighted regions represent areas that contributed "
        "to the model prediction. Grad-CAM is an attention "
        "visualization, not an exact clinical lesion boundary."
    )

    if heatmap or overlay or contour:

        g1, g2, g3 = st.columns(3)

        with g1:

            st.image(
                original,
                caption="Original Image",
                use_container_width=True,
            )

        with g2:

            st.image(
                overlay or heatmap,
                caption="Grad-CAM Overlay",
                use_container_width=True,
            )

        with g3:

            if heatmap:

                st.image(
                    heatmap,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True,
                )

            elif contour:

                st.image(
                    contour,
                    caption="Attention Region",
                    use_container_width=True,
                )

            else:

                st.info(
                    "No additional visualization."
                )

    else:

        st.info(
            "Grad-CAM is unavailable for this prediction."
        )

    st.markdown(
        '<p class="section-label">Lesion Localization</p>',
        unsafe_allow_html=True,
    )

    if contour:

        l1, l2, l3 = st.columns(3)

        with l1:

            st.image(
                original,
                caption="Original Image",
                use_container_width=True,
            )

        with l2:

            st.image(
                contour,
                caption="Localization Overlay",
                use_container_width=True,
            )

        with l3:

            st.caption("Detected region")

            st.write(
                "Contours are derived from high-activation "
                "Grad-CAM regions. They represent model attention, "
                "not confirmed clinical lesion boundaries."
            )

    else:

        st.info(
            "Lesion localization is unavailable."
        )

    if result.get("report"):

        with st.expander(
            "Narrative assessment"
        ):

            st.write(
                result["report"]
            )

    render_disclaimer()


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(
    image_file: Any,
    image_bytes: bytes,
    original: Image.Image,
    include_explainability: bool,
    source: str,
) -> None:

    current_hash = image_hash(
        image_bytes
    )

    filename = getattr(
        image_file,
        "name",
        f"{source}.jpg",
    )

    mime_type = getattr(
        image_file,
        "type",
        None,
    ) or "image/jpeg"

    with st.status(
        "Analyzing image...",
        expanded=True,
    ) as status:

        st.write(
            "Uploading image to local FastAPI backend..."
        )

        result, error = api_post(
            "/api/v1/analyze",
            files={
                "file": (
                    filename,
                    image_bytes,
                    mime_type,
                )
            },
            params={
                "include_explainability":
                    str(
                        include_explainability
                    ).lower(),
                "allow_whole_image_fallback": "false",
            },
            timeout=180,
        )

        if error:

            status.update(
                label="Analysis failed",
                state="error",
            )

            st.error(error)

            return

        try:

            result = normalize_prediction(
                result
            )

        except ValueError as exc:

            status.update(
                label="Invalid backend result",
                state="error",
            )

            st.error(
                str(exc)
            )

            if st.session_state.get(
                "debug_backend",
                False,
            ):

                st.json(result)

            return

        st.session_state["last_result"] = result
        st.session_state["last_image"] = original
        st.session_state["result_image_hash"] = current_hash
        st.session_state["active_source"] = source

        grade_label = result.get("icdas_grade")
        if grade_label is None:
            done = result.get("message") or "No localized region"
        else:
            done = f"Prediction complete — ICDAS {grade_label}"
        status.update(
            label=done,
            state="complete",
        )

    if st.session_state.get(
        "debug_backend",
        False,
    ):

        with st.expander(
            "Exact backend response",
        ):

            st.json(result)


# ============================================================
# IMAGE PANEL
# ============================================================

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

    original = open_image(
        image_bytes
    )

    if original is None:
        return

    current_hash = image_hash(
        image_bytes
    )

    st.markdown(
        "**Image preview**"
    )

    st.image(
        original,
        caption=getattr(
            image_file,
            "name",
            "Captured image",
        ),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    analyze = c1.button(
        analyze_label,
        type="primary",
        use_container_width=True,
        key=f"analyze_{source}",
    )

    remove = c2.button(
        remove_label,
        use_container_width=True,
        key=f"remove_{source}",
    )

    if remove:

        for key in (
            "last_result",
            "last_image",
            "result_image_hash",
            "active_source",
        ):
            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

    if analyze:
        now = time.time()
        if source == "camera":
            last_ts = float(st.session_state.get("camera_analyze_ts") or 0)
            last_h = st.session_state.get("camera_analyze_hash")
            if last_h == current_hash and (now - last_ts) < 8:
                st.caption("Camera capture unchanged — skipped duplicate analysis.")
                return
            st.session_state["camera_analyze_ts"] = now
            st.session_state["camera_analyze_hash"] = current_hash

        run_prediction(
            image_file=image_file,
            image_bytes=image_bytes,
            original=original,
            include_explainability=include_explainability,
            source=source,
        )

    stored_result = st.session_state.get(
        "last_result"
    )

    if (
        stored_result
        and st.session_state.get(
            "result_image_hash"
        ) == current_hash
        and st.session_state.get(
            "active_source"
        ) == source
    ):

        render_results(
            stored_result,
            st.session_state.get(
                "last_image",
                original,
            ),
        )


# ============================================================
# DETECTION
# ============================================================

def render_detection() -> None:

    st.markdown(
        '<p class="page-title">Dental Caries Detection</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Choose one method: upload an existing photograph "
        "or capture a new image with your camera."
        "</p>",
        unsafe_allow_html=True,
    )

    include_cam = st.checkbox(
        "Generate Grad-CAM and lesion localization",
        value=True,
    )

    left, mid, right = st.columns(
        [1, 0.12, 1]
    )

    with left:

        st.markdown(
            """
            <div class="input-card">
                <h3>Upload Dental Photograph</h3>
                <p>Select an existing intraoral photograph.</p>
                <p style="margin-top:10px;">
                    <strong>Accepted formats:</strong>
                    JPG, JPEG, PNG, BMP, WEBP
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Browse device",
            type=ACCEPTED_UPLOAD_TYPES,
            key="upload_photo_input",
        )

        if uploaded is None:

            st.caption(
                "No image selected yet."
            )

        render_image_source_panel(
            source="upload",
            image_file=uploaded,
            include_explainability=include_cam,
            analyze_label="Analyze Image",
            remove_label="Remove Image",
        )

    with mid:

        st.markdown(
            '<div class="or-divider">OR</div>',
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="input-card">
                <h3>Capture Using Camera</h3>
                <p>Use your device camera to capture an intraoral photograph.</p>
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
                "Camera unavailable. Please allow browser camera permission "
                "or use Upload Photo."
            )

        if camera is None:

            st.caption(
                "No photograph captured yet."
            )

        render_image_source_panel(
            source="camera",
            image_file=camera,
            include_explainability=include_cam,
            analyze_label="Analyze Image",
            remove_label="Retake Photograph",
        )

    render_how_it_works()


# ============================================================
# DATASET LABELING
# ============================================================

def render_dataset_labeling() -> None:

    st.markdown(
        '<p class="page-title">Dataset Labeling</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Upload cropped tooth images and assign ICDAS grades 0-4."
        "</p>",
        unsafe_allow_html=True,
    )

    base = Path("data/icdas")

    uploaded_files = st.file_uploader(
        "Upload Tooth Images",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
        key="dataset_label_uploader",
    )

    if not uploaded_files:

        st.info(
            "Upload one or more cropped tooth images to begin labeling."
        )

        return

    if "label_index" not in st.session_state:

        st.session_state[
            "label_index"
        ] = 0

    index = int(
        st.session_state[
            "label_index"
        ]
    )

    total = len(
        uploaded_files
    )

    if index >= total:

        st.success(
            "All uploaded images have been labeled."
        )

        if st.button(
            "Start Again"
        ):

            st.session_state[
                "label_index"
            ] = 0

            st.rerun()

        return

    image_file = uploaded_files[
        index
    ]

    image = Image.open(
        image_file
    ).convert(
        "RGB"
    )

    st.progress(
        index / total
    )

    st.markdown(
        f"### Image {index + 1} / {total}"
    )

    st.image(
        image,
        use_container_width=True,
    )

    st.caption(
        image_file.name
    )

    grade = st.radio(
        "Select ICDAS Grade",
        [
            "0",
            "1",
            "2",
            "3",
            "4",
        ],
        horizontal=True,
        key=f"grade_{index}",
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "💾 Save & Next",
            type="primary",
            use_container_width=True,
            key=f"save_{index}",
        ):

            # Random 70/15/15 split.
            random_value = random.random()

            if random_value < 0.70:
                split = "train"

            elif random_value < 0.85:
                split = "val"

            else:
                split = "test"

            folder = (
                base
                / split
                / grade
            )

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (
                folder
                / image_file.name
            )

            with open(
                destination,
                "wb",
            ) as output:

                output.write(
                    image_file.getbuffer()
                )

            st.success(
                f"Saved → {split}/{grade}/{image_file.name}"
            )

            st.session_state[
                "label_index"
            ] = index + 1

            st.rerun()

    with c2:

        if st.button(
            "⏭ Skip Image",
            use_container_width=True,
            key=f"skip_{index}",
        ):

            st.session_state[
                "label_index"
            ] = index + 1

            st.rerun()

    st.markdown("---")

    st.subheader(
        "Dataset Summary"
    )

    columns = st.columns(3)

    for column, split in zip(
        columns,
        [
            "train",
            "val",
            "test",
        ],
    ):

        with column:

            st.markdown(
                f"#### {split.upper()}"
            )

            for grade_id in range(5):

                folder = (
                    base
                    / split
                    / str(grade_id)
                )

                count = (
                    len(
                        [
                            p
                            for p in folder.glob("*")
                            if p.is_file()
                        ]
                    )
                    if folder.exists()
                    else 0
                )

                st.write(
                    f"ICDAS {grade_id}: {count}"
                )


# ============================================================
# ANALYSIS
# ============================================================

def render_analysis() -> None:

    st.markdown(
        '<p class="page-title">Analysis</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Stored screening statistics from previous scans."
        "</p>",
        unsafe_allow_html=True,
    )

    stats, error = api_get(
        "/api/v1/stats"
    )

    if error:

        st.warning(error)

        return

    if not stats or stats.get(
        "total_analyses",
        0,
    ) == 0:

        st.info(
            "No scan data available yet."
        )

        return

    dist = (
        stats.get(
            "grade_distribution",
            {},
        )
        or {}
    )

    total = int(
        stats.get(
            "total_analyses",
            0,
        )
    )

    healthy = int(
        dist.get(
            "0",
            0,
        )
        or 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Scans",
        total,
    )

    c2.metric(
        "Healthy",
        healthy,
    )

    c3.metric(
        "Caries Detected",
        max(
            0,
            total - healthy,
        ),
    )

    c4.metric(
        "Average Confidence",
        f"{float(stats.get('average_confidence', 0)):.1f}%",
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "**ICDAS distribution**"
        )

        chart_data = pd.Series(
            {
                f"ICDAS {grade}":
                    int(
                        dist.get(
                            str(grade),
                            0,
                        )
                    )
                for grade in ICDAS_MODEL_GRADES
            }
        )

        st.bar_chart(
            chart_data
        )

    with right:

        st.markdown(
            "**Confidence distribution**"
        )

        st.bar_chart(
            pd.Series(
                stats.get(
                    "confidence_buckets",
                    {},
                )
            )
        )

    most_common = stats.get(
        "most_common_grade"
    )

    st.caption(
        (
            f"Most common grade: ICDAS {most_common}"
            if most_common is not None
            else "Most common grade: —"
        )
    )

    render_disclaimer()


# ============================================================
# HISTORY
# ============================================================

def render_history() -> None:

    st.markdown(
        '<p class="page-title">Scan History</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Previous ICDAS assessments stored by the local backend."
        "</p>",
        unsafe_allow_html=True,
    )

    rows, error = api_get(
        "/api/v1/history"
    )

    if error:

        st.warning(error)

        return

    if not rows:

        st.info(
            "No scan data available yet."
        )

        return

    filter_labels = [
        "All"
    ] + [
        f"ICDAS {grade}"
        for grade in ICDAS_MODEL_GRADES
    ]

    selected_filter = st.radio(
        "Filter",
        filter_labels,
        horizontal=True,
    )

    filtered_rows = list(rows)

    if selected_filter != "All":

        selected_grade = int(
            selected_filter.split()[-1]
        )

        filtered_rows = [
            row
            for row in rows
            if int(
                row.get(
                    "icdas_grade",
                    -1,
                )
            ) == selected_grade
        ]

        if not filtered_rows:

            st.info(
                f"No stored scans for {selected_filter}."
            )

            return

    table_rows = []

    for row in filtered_rows:

        created = row.get(
            "created_at"
        )

        date_text = ""

        if created:

            try:

                date_text = datetime.fromisoformat(
                    created.replace(
                        "Z",
                        "+00:00",
                    )
                ).strftime(
                    "%d %b %Y %H:%M"
                )

            except Exception:

                date_text = str(
                    created
                )

        table_rows.append(
            {
                "Date":
                    date_text,
                "ICDAS Grade":
                    f"ICDAS {row.get('icdas_grade')}",
                "Confidence":
                    f"{float(row.get('confidence', 0)):.1f}%",
                "Status":
                    row.get(
                        "urgency",
                        "",
                    ),
                "id":
                    row.get(
                        "id"
                    ),
            }
        )

    table = pd.DataFrame(
        table_rows
    )

    st.dataframe(
        table.drop(
            columns=["id"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    ids = [
        row["id"]
        for row in table_rows
        if row["id"] is not None
    ]

    if not ids:
        return

    selected_id = st.selectbox(
        "Open record",
        ids,
        format_func=lambda value:
            f"Scan #{value}",
    )

    detail, error = api_get(
        f"/api/v1/history/{selected_id}"
    )

    if error:

        st.error(error)

        return

    original = b64_to_image(
        detail.get(
            "image_base64"
        )
    )

    if original is None:

        original = Image.new(
            "RGB",
            (
                224,
                224,
            ),
            color=(28, 42, 54),
        )

    payload = {
        **detail,
        "icdas_grade":
            detail.get(
                "icdas_grade"
            ),
        "label":
            f"ICDAS {detail.get('icdas_grade')}",
        "description":
            detail.get(
                "finding"
            ),
        "probabilities":
            detail.get(
                "probabilities"
            ),
        "heatmap_base64":
            detail.get(
                "heatmap_base64"
            ),
        "overlay_base64":
            detail.get(
                "overlay_base64"
            ),
        "contour_base64":
            detail.get(
                "contour_base64"
            ),
    }

    try:

        payload = normalize_prediction(
            payload
        )

    except ValueError as exc:

        st.error(
            str(exc)
        )

        return

    render_results(
        payload,
        original,
    )


# ============================================================
# ICDAS GUIDE
# ============================================================

def render_icdas_guide() -> None:

    st.markdown(
        '<p class="page-title">ICDAS Guide</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "The current AI model predicts ICDAS 0-4. "
        "ICDAS 5-6 are shown as reference only."
        "</p>",
        unsafe_allow_html=True,
    )

    columns = st.columns(2)

    for grade, info in ICDAS_GUIDE.items():

        with columns[grade % 2]:

            scope = (
                "Classified by this model"
                if info["in_model"]
                else "Reference only"
            )

            st.markdown(
                f"""
                <div class="guide-card">

                    <div class="guide-icon">
                        {info["icon"]}
                    </div>

                    <h4>
                        ICDAS {grade} → {info["short"]}
                    </h4>

                    <span class="chip">
                        {scope}
                    </span>

                    <p style="margin-top:8px">
                        {info["description"]}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")


# ============================================================
# ABOUT
# ============================================================

def render_about() -> None:

    st.markdown(
        '<p class="page-title">About Dental AI</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Offline AI-Based Dental Caries Detection using ICDAS."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "#### Dental caries problem"
    )

    st.write(
        "Dental caries is a progressive disease that benefits "
        "from earlier visual detection. ICDAS provides a standardized "
        "scale for describing lesion severity."
    )

    st.markdown(
        "#### Purpose of the application"
    )

    st.write(
        "This application supports educational and research screening "
        "of intraoral photographs. The current model predicts ICDAS 0-4 "
        "and returns confidence, class probabilities and explainability."
    )

    st.markdown(
        "#### Current AI model"
    )

    st.write(
        "MobileNetV3-Small with CBAM attention and a five-class "
        "softmax output."
    )

    st.markdown(
        "#### ICDAS classification"
    )

    st.write(
        "The deployed model predicts ICDAS 0, 1, 2, 3 and 4. "
        "ICDAS 5 and 6 are shown in the guide for reference only."
    )

    st.markdown(
        "#### Offline processing"
    )

    st.write(
        "When FastAPI and TensorFlow run locally, image inference "
        "is performed by the local backend."
    )

    st.markdown(
        "#### Explainable AI"
    )

    st.write(
        "Grad-CAM highlights image regions that contributed to "
        "the prediction. It is an attention visualization, not "
        "clinical lesion segmentation."
    )

    st.markdown(
        "#### Workflow"
    )

    st.markdown(
        """
        <div class="workflow">
            Image
            <span>↓</span>
            Streamlit
            <span>↓</span>
            FastAPI
            <span>↓</span>
            MobileNetV3-Small + CBAM
            <span>↓</span>
            ICDAS 0-4
            <span>↓</span>
            Confidence + Explainability
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "#### Live model information"
    )

    info, error = api_get(
        "/api/v1/model/info"
    )

    if error:

        st.warning(error)

    elif info:

        st.write(
            f"**Name:** {info.get('name', '—')}"
        )

        st.write(
            f"**ICDAS mode:** {info.get('icdas_mode', '—')}"
        )

        st.write(
            f"**Classes:** {info.get('num_classes', '—')}"
        )

        st.write(
            f"**Input size:** "
            f"{info.get('image_size', '—')} × "
            f"{info.get('image_size', '—')}"
        )

        st.write(
            "**Ordinal regression:** "
            + (
                "yes"
                if info.get("ordinal_regression")
                else "no"
            )
        )

        st.caption(
            "These values come directly from the running backend."
        )

    render_privacy_card()
    render_disclaimer()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    initialize_session_state()

    # Apply pending navigation BEFORE sidebar radio widget exists.
    if st.session_state.get(
        "_pending_nav_page"
    ):

        st.session_state["nav_page"] = (
            st.session_state.pop(
                "_pending_nav_page"
            )
        )

    render_css()
    render_sidebar()
    render_top_header()

    page = st.session_state.get(
        "nav_page",
        PAGE_HOME,
    )

    if page == PAGE_HOME:

        render_dashboard()

    elif page == PAGE_DETECTION:

        render_detection()

    elif page == PAGE_ANALYSIS:

        render_analysis()

    elif page == PAGE_LABEL:

        render_dataset_labeling()

    elif page == PAGE_HISTORY:

        render_history()

    elif page == PAGE_GUIDE:

        render_icdas_guide()

    elif page == PAGE_ABOUT:

        render_about()

    else:

        render_dashboard()

    render_footer()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

