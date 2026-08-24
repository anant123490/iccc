"""
ICDAS Dental AI — Streamlit frontend.

The frontend is UI-only.

ALL classification is performed by the FastAPI backend:

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

IMPORTANT:
    The frontend NEVER calculates the ICDAS grade itself.
    It always uses:

        result["icdas_grade"]

returned by FastAPI.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
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

DEFAULT_BACKEND_URL = (
    "http://127.0.0.1:8000"
)

ACCEPTED_UPLOAD_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp",
]

LOW_CONFIDENCE_COPY = (
    "Low confidence prediction. "
    "Professional dental examination recommended."
)

DISCLAIMER = (
    "AI-assisted screening tool. This application is intended "
    "for educational, research, and screening purposes and should "
    "not replace examination or diagnosis by a qualified dental professional."
)


# ============================================================
# PAGES
# ============================================================

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


# ============================================================
# ICDAS GUIDE
# ============================================================

ICDAS_GUIDE = {
    0: {
        "title": "Sound tooth",
        "short": "Sound",
        "icon": "0",
        "description": (
            "Sound tooth surface. No evidence of caries "
            "after visual inspection."
        ),
        "in_model": True,
    },
    1: {
        "title": "First visual change",
        "short": "First visual change",
        "icon": "1",
        "description": (
            "First visual change in enamel. "
            "Opacity or discoloration is typically visible "
            "after air drying."
        ),
        "in_model": True,
    },
    2: {
        "title": "Distinct visual change",
        "short": "Distinct visual change",
        "icon": "2",
        "description": (
            "Distinct visual change in enamel when wet. "
            "Demineralization is more established but "
            "remains non-cavitated."
        ),
        "in_model": True,
    },
    3: {
        "title": "Localized enamel breakdown",
        "short": "Localized enamel breakdown",
        "icon": "3",
        "description": (
            "Localized enamel breakdown due to caries, "
            "without visible dentin."
        ),
        "in_model": True,
    },
    4: {
        "title": "Underlying dark shadow",
        "short": "Underlying dark shadow",
        "icon": "4",
        "description": (
            "Underlying dark shadow from dentin, "
            "with or without localized enamel breakdown."
        ),
        "in_model": True,
    },
    5: {
        "title": "Distinct cavity with visible dentin",
        "short": "Distinct cavity with visible dentin",
        "icon": "5",
        "description": (
            "Reference only. ICDAS 5 is not predicted "
            "by the current model."
        ),
        "in_model": False,
    },
    6: {
        "title": "Extensive distinct cavity with visible dentin",
        "short": "Extensive distinct cavity",
        "icon": "6",
        "description": (
            "Reference only. ICDAS 6 is not predicted "
            "by the current model."
        ),
        "in_model": False,
    },
}

ICDAS_MODEL_GRADES = (
    0,
    1,
    2,
    3,
    4,
)

URGENCY_COLORS = {
    "LOW": "#19A7A8",
    "MODERATE": "#d97706",
    "HIGH": "#ef4444",
    "CRITICAL": "#dc2626",
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
# BACKEND URL
# ============================================================

def backend_url() -> str:

    value = (
        st.session_state.get(
            "backend_url",
            DEFAULT_BACKEND_URL,
        )
    )

    return str(value).rstrip("/")


# ============================================================
# NAVIGATION
# ============================================================

def set_page(
    page_name: str,
) -> None:

    st.session_state[
        "_pending_nav_page"
    ] = page_name

    st.rerun()


# ============================================================
# HTTP ERROR
# ============================================================

def http_error_message(
    exc: requests.exceptions.HTTPError,
) -> str:

    response = exc.response

    if response is None:

        return (
            "The AI backend returned an error."
        )

    try:

        payload = response.json()

        detail = payload.get(
            "detail",
            payload,
        )

        if isinstance(
            detail,
            str,
        ):

            return detail

        return (
            "The AI backend returned an error."
        )

    except Exception:

        text = (
            response.text
            or ""
        ).strip()

        if (
            text
            and len(text) < 280
            and "Traceback" not in text
        ):

            return text

        return (
            "The AI backend returned an error."
        )


# ============================================================
# API GET
# ============================================================

def api_get(
    path: str,
    timeout: int = 15,
):

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
            "Start FastAPI first.",
        )

    except requests.exceptions.Timeout:

        return (
            None,
            "The AI backend timed out.",
        )

    except requests.exceptions.HTTPError as exc:

        return (
            None,
            http_error_message(exc),
        )

    except Exception as exc:

        return (
            None,
            f"Unexpected backend error: {exc}",
        )


# ============================================================
# API POST
# ============================================================

def api_post(
    path: str,
    timeout: int = 120,
    **kwargs,
):

    try:

        response = requests.post(
            f"{backend_url()}{path}",
            timeout=timeout,
            **kwargs,
        )

        response.raise_for_status()

        return response.json(), None

    except requests.exceptions.ConnectionError:

        return (
            None,
            "Unable to connect to the AI backend. "
            "Start FastAPI first.",
        )

    except requests.exceptions.Timeout:

        return (
            None,
            "The request timed out while contacting the AI backend.",
        )

    except requests.exceptions.HTTPError as exc:

        return (
            None,
            http_error_message(exc),
        )

    except Exception as exc:

        return (
            None,
            f"Unexpected backend error: {exc}",
        )


# ============================================================
# RESPONSE VALIDATION
# ============================================================

def validate_prediction_response(
    result: Any,
) -> dict:
    """
    Validate and normalize exactly what FastAPI returned.

    IMPORTANT:
        The frontend does NOT calculate the ICDAS grade.

        It trusts only:
            result["icdas_grade"]

    """

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Backend returned an invalid response."
        )

    if "icdas_grade" not in result:

        raise ValueError(
            "Backend response does not contain "
            "'icdas_grade'."
        )

    # --------------------------------------------------------
    # EXACT BACKEND GRADE
    # --------------------------------------------------------

    try:

        grade = int(
            result["icdas_grade"]
        )

    except (
        TypeError,
        ValueError,
    ):

        raise ValueError(
            "Backend returned an invalid ICDAS grade."
        )

    if grade not in ICDAS_MODEL_GRADES:

        raise ValueError(
            f"Backend returned ICDAS {grade}. "
            "Only ICDAS 0-4 are supported."
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    try:

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = 0.0

    confidence = max(
        0.0,
        min(
            100.0,
            confidence,
        ),
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        result.get(
            "probabilities",
            {},
        )
    )

    if isinstance(
        probabilities,
        str,
    ):

        try:

            probabilities = json.loads(
                probabilities
            )

        except Exception:

            probabilities = {}

    normalized_probabilities = {}

    if isinstance(
        probabilities,
        dict,
    ):

        for key, value in (
            probabilities.items()
        ):

            try:

                class_id = int(
                    key
                )

                probability = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if class_id in ICDAS_MODEL_GRADES:

                normalized_probabilities[
                    str(class_id)
                ] = probability

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    normalized = dict(
        result
    )

    normalized[
        "icdas_grade"
    ] = grade

    normalized[
        "confidence"
    ] = confidence

    normalized[
        "probabilities"
    ] = normalized_probabilities

    return normalized


# ============================================================
# BASE64 IMAGE
# ============================================================

def b64_to_image(
    b64_string: str | None,
) -> Image.Image | None:

    if not b64_string:

        return None

    try:

        data = base64.b64decode(
            b64_string
        )

        image = Image.open(
            io.BytesIO(data)
        )

        image.load()

        return image.convert(
            "RGB"
        )

    except Exception:

        return None


# ============================================================
# HASH
# ============================================================

def image_hash(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


# ============================================================
# SEVERITY
# ============================================================

def icdas_severity_copy(
    grade: Any,
    result: dict | None = None,
) -> tuple[str, str]:

    try:

        grade_int = int(
            grade
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            "Unknown",
            "The model did not return a valid ICDAS grade.",
        )

    if result:

        label = (
            result.get(
                "label"
            )
            or ICDAS_GUIDE.get(
                grade_int,
                {},
            ).get(
                "title",
                "",
            )
        )

        description = (
            result.get(
                "description"
            )
            or ICDAS_GUIDE.get(
                grade_int,
                {},
            ).get(
                "description",
                "",
            )
        )

        return (
            str(label),
            str(description),
        )

    info = ICDAS_GUIDE.get(
        grade_int
    )

    if info:

        return (
            info["title"],
            info["description"],
        )

    return (
        f"ICDAS {grade_int}",
        "No description is available.",
    )


# ============================================================
# PROBABILITY
# ============================================================

def probability_percent(
    value: Any,
) -> float:

    try:

        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    if number <= 1.0:

        number *= 100.0

    return max(
        0.0,
        min(
            100.0,
            number,
        ),
    )


# ============================================================
# CSS
# ============================================================

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

        html, body, [class*="css"], .stApp,
        .stMarkdown, p, label, span {
            font-family: 'Source Sans 3', sans-serif;
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

        [data-testid="stSidebar"] hr {
            border-color: var(--border);
        }

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

        .hero-card .lede {
            color: var(--text);
            font-size: 1.05rem;
            margin: 0 0 10px 0;
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
            margin: 0 0 8px 0;
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--text);
        }

        .feature-card p,
        .guide-card p,
        .input-card p,
        .step-card p {
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

        .page-sub {
            color: var(--muted);
            margin: 0 0 18px 0;
        }

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

        .privacy-card p {
            color: #9fd8d9 !important;
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

        .site-footer p {
            color: var(--muted);
            margin: 0 0 8px 0;
            font-size: 0.88rem;
        }

        .site-footer .links {
            color: var(--muted);
            font-size: 0.85rem;
        }

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

        [data-testid="stCaption"],
        .stCaption,
        small {
            color: var(--muted) !important;
        }

        img {
            border-radius: 8px;
            max-width: 100%;
        }

        .stProgress > div > div {
            background: var(--secondary);
        }

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
            .hero-card h1 {
                font-size: 1.55rem;
            }

            .or-divider {
                padding-top: 8px;
                padding-bottom: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:

    health, error = api_get(
        "/api/v1/health"
    )

    model_loaded = bool(
        health
        and health.get(
            "model_loaded"
        )
    )

    with st.sidebar:

        st.markdown(
            '<p class="brand-mark">Dental AI</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="brand-sub">'
            'Offline AI-Based Dental Caries Detection'
            '</p>',
            unsafe_allow_html=True,
        )

        st.radio(
            "Pages",
            SIDEBAR_NAV,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.caption(
            "AI Model Status"
        )

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

        with st.expander(
            "Backend connection"
        ):

            st.session_state[
                "backend_url"
            ] = st.text_input(
                "Backend URL",
                value=backend_url(),
                help=(
                    "Local FastAPI inference server"
                ),
            )

            if error:

                st.caption(
                    error
                )

            if health:

                st.caption(
                    "API status: "
                    f"{health.get('status', 'unknown')}"
                )

                st.caption(
                    "Database: "
                    + (
                        "ok"
                        if health.get(
                            "database_ok"
                        )
                        else "unavailable"
                    )
                )

                groq = health.get(
                    "groq_configured"
                )

                st.caption(
                    (
                        "Narrative reports: "
                        "Groq configured"
                        if groq
                        else
                        "Narrative reports: "
                        "local fallback"
                    )
                )

        st.checkbox(
            "Backend debug",
            key="debug_backend",
            help=(
                "Show the exact JSON returned by FastAPI."
            ),
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

                btn_type = (
                    "primary"
                    if current == name
                    else "secondary"
                )

                if st.button(
                    name,
                    type=btn_type,
                    use_container_width=True,
                    key=f"topnav_{name}",
                ):

                    if current != name:

                        set_page(
                            name
                        )

    st.markdown(
        "<hr style='border-color:#2a3c4c;"
        "margin:4px 0 18px 0;'>",
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer() -> None:

    st.markdown(
        """
        <div class="site-footer">
            <h4>Dental AI</h4>
            <p>
                Offline AI-Based Dental Caries Detection
                using ICDAS Classification
            </p>
            <p class="links">
                Home · Detection · ICDAS Guide · About
            </p>
            <p>
                AI-assisted screening tool. This application is
                intended for educational, research, and screening
                purposes and should not replace examination or
                diagnosis by a qualified dental professional.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PRIVACY
# ============================================================

def render_privacy_card() -> None:

    st.markdown(
        """
        <div class="privacy-card">
            <strong>Privacy First</strong>
            <p style="margin:6px 0 0 0;font-size:0.88rem;">
            Images are sent only to the local FastAPI backend
            when using this local application.
            Optional narrative text may use Groq only when configured.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISCLAIMER
# ============================================================

def render_disclaimer() -> None:

    st.markdown(
        f"""
        <div class="disclaimer">
            {DISCLAIMER}
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
            "The image is sent to the local FastAPI backend.",
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

    cols = st.columns(
        4
    )

    for col, (
        number,
        title,
        body,
    ) in zip(
        cols,
        steps,
    ):

        with col:

            st.markdown(
                f"""
                <div class="step-card">
                    <div class="guide-icon">
                        {number}
                    </div>
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

        set_page(
            PAGE_DETECTION
        )

    features = [
        (
            "ICDAS Classification",
            "Predicts ICDAS 0-4 using the trained classification model.",
        ),
        (
            "Edge AI Detection",
            "MobileNetV3-Small with CBAM attention for compact local inference.",
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
            "Upload a photograph or capture one with the device camera.",
        ),
    ]

    rows = [
        features[:3],
        features[3:],
    ]

    for row in rows:

        cols = st.columns(
            3
        )

        for col, (
            title,
            body,
        ) in zip(
            cols,
            row,
        ):

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

    stats, error = api_get(
        "/api/v1/stats"
    )

    if error:

        st.info(
            "Dashboard statistics will appear "
            "when the backend is available."
        )

        st.caption(
            error
        )

    elif not stats or (
        stats.get(
            "total_analyses",
            0,
        )
        == 0
    ):

        st.info(
            "No scan data available yet."
        )

    else:

        distribution = (
            stats.get(
                "grade_distribution"
            )
            or {}
        )

        healthy = int(
            distribution.get(
                "0",
                0,
            )
            or 0
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

        c1, c2, c3, c4 = (
            st.columns(4)
        )

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

    st.markdown(
        "**Prediction probability**"
    )

    for grade in ICDAS_MODEL_GRADES:

        value = probabilities.get(
            str(grade),
            0.0,
        )

        percentage = (
            probability_percent(
                value
            )
        )

        st.caption(
            f"ICDAS {grade} · "
            f"{percentage:.1f}%"
        )

        st.progress(
            min(
                1.0,
                percentage / 100.0,
            )
        )


# ============================================================
# RESULT DISPLAY
# ============================================================

def render_results(
    result: dict,
    original: Image.Image,
) -> None:
    """
    Display the EXACT result returned by FastAPI.

    No grade calculation happens here.
    """

    try:

        result = (
            validate_prediction_response(
                result
            )
        )

    except ValueError as exc:

        st.error(
            f"Invalid backend prediction: {exc}"
        )

        return

    # ========================================================
    # EXACT BACKEND RESULT
    # ========================================================

    grade = result[
        "icdas_grade"
    ]

    confidence = result[
        "confidence"
    ]

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    if st.session_state.get(
        "debug_backend",
        False,
    ):

        with st.expander(
            "Backend response (debug)"
        ):

            st.json(
                result
            )

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    label, description = (
        icdas_severity_copy(
            grade,
            result,
        )
    )

    urgency = str(
        result.get(
            "urgency",
            "",
        )
        or ""
    ).upper()

    urgency_color = (
        URGENCY_COLORS.get(
            urgency,
            "#334155",
        )
    )

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<p class="section-label">'
        "AI Analysis Result"
        "</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.15, 1.35]
    )

    # ========================================================
    # LEFT
    # ========================================================

    with left:

        st.markdown(
            '<div class="result-card" '
            'style="padding:22px 18px;">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**ICDAS Classification**"
        )

        st.markdown(
            f"""
            <div class="icdas-big">
                ICDAS {grade}
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

        st.markdown(
            "**Confidence**"
        )

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
            f"**{confidence:.1f}%**"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.caption(
            description
        )

        if result.get(
            "low_confidence",
            False,
        ):

            st.warning(
                result.get(
                    "low_confidence_message"
                )
                or LOW_CONFIDENCE_COPY
            )

    # ========================================================
    # RIGHT
    # ========================================================

    with right:

        render_probability_bars(
            result.get(
                "probabilities",
                {},
            )
        )

        if result.get(
            "finding"
        ):

            st.markdown(
                "**Finding**"
            )

            st.write(
                result[
                    "finding"
                ]
            )

        if result.get(
            "recommendation"
        ):

            st.markdown(
                "**Recommendation**"
            )

            st.write(
                result[
                    "recommendation"
                ]
            )

        if result.get(
            "action"
        ):

            st.caption(
                "Suggested action: "
                f"{result['action']}"
            )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

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
        '<p class="section-label">'
        "AI Explainability"
        "</p>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Highlighted regions represent areas that "
        "contributed to the model prediction. "
        "Grad-CAM is an attention visualization, "
        "not an exact clinical lesion boundary."
    )

    if (
        heatmap is not None
        or overlay is not None
    ):

        g1, g2, g3 = st.columns(
            3
        )

        with g1:

            st.image(
                original,
                caption="Original Image",
                use_container_width=True,
            )

        with g2:

            st.image(
                overlay
                or heatmap,
                caption="Grad-CAM Overlay",
                use_container_width=True,
            )

        with g3:

            if heatmap is not None:

                st.image(
                    heatmap,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True,
                )

            elif contour is not None:

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
            "Grad-CAM is unavailable."
        )

    # ========================================================
    # LOCALIZATION
    # ========================================================

    st.markdown(
        '<p class="section-label">'
        "Lesion Localization"
        "</p>",
        unsafe_allow_html=True,
    )

    if contour is not None:

        l1, l2, l3 = st.columns(
            3
        )

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

            st.caption(
                "Detected region"
            )

            st.write(
                "Contours are derived from high-activation "
                "Grad-CAM regions. They represent model "
                "attention, not confirmed clinical boundaries."
            )

    else:

        st.info(
            "Lesion localization is unavailable."
        )

    # ========================================================
    # REPORT
    # ========================================================

    if result.get(
        "report"
    ):

        with st.expander(
            "Narrative assessment"
        ):

            st.write(
                result[
                    "report"
                ]
            )

    render_disclaimer()


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(
    image_bytes: bytes,
) -> Image.Image | None:

    if not image_bytes:

        st.error(
            "No image selected."
        )

        return None

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        image.load()

        return image.convert(
            "RGB"
        )

    except UnidentifiedImageError:

        st.error(
            "Invalid image. "
            "Please upload a JPG, JPEG, PNG, BMP or WEBP image."
        )

    except OSError:

        st.error(
            "The image appears to be corrupt."
        )

    except Exception:

        st.error(
            "Unable to read the selected image."
        )

    return None


# ============================================================
# CLEAR ANALYSIS
# ============================================================

def clear_analysis_state() -> None:

    for key in [
        "last_result",
        "last_image",
        "result_image_hash",
        "active_source",
    ]:

        st.session_state.pop(
            key,
            None,
        )


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(
    image_file: Any,
    image_bytes: bytes,
    original: Image.Image,
    include_explainability: bool,
    source: str,
) -> None:
    """
    Send image to FastAPI.

    IMPORTANT:
        We do not calculate ICDAS here.
        The backend owns the prediction.
    """

    current_hash = (
        image_hash(
            image_bytes
        )
    )

    filename = getattr(
        image_file,
        "name",
        f"{source}.jpg",
    )

    mime_type = (
        getattr(
            image_file,
            "type",
            None,
        )
        or "image/jpeg"
    )

    with st.status(
        "Analyzing image...",
        expanded=True,
    ) as status:

        st.write(
            "Uploading image to local AI backend..."
        )

        result, error = api_post(
            "/api/v1/predict",
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
                    ).lower()
            },
            timeout=120,
        )

        if error:

            status.update(
                label="Analysis failed",
                state="error",
            )

            st.error(
                error
            )

            return

        st.write(
            "FastAPI prediction received."
        )

        # ----------------------------------------------------
        # VERY IMPORTANT:
        # Validate backend response.
        # ----------------------------------------------------

        try:

            result = (
                validate_prediction_response(
                    result
                )
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
                "debug_backend"
            ):

                st.json(
                    result
                )

            return

        # ----------------------------------------------------
        # Store EXACT backend response.
        # ----------------------------------------------------

        st.session_state[
            "last_result"
        ] = result

        st.session_state[
            "last_image"
        ] = original

        st.session_state[
            "result_image_hash"
        ] = current_hash

        st.session_state[
            "active_source"
        ] = source

        status.update(
            label=(
                "Prediction complete — "
                f"ICDAS {result['icdas_grade']}"
            ),
            state="complete",
        )

    # --------------------------------------------------------
    # DEBUG RESULT
    # --------------------------------------------------------

    if st.session_state.get(
        "debug_backend",
        False,
    ):

        with st.expander(
            "Exact backend response"
        ):

            st.json(
                result
            )


# ============================================================
# IMAGE SOURCE PANEL
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

    image_bytes = (
        image_file.getvalue()
    )

    original = open_image(
        image_bytes
    )

    if original is None:

        return

    current_hash = (
        image_hash(
            image_bytes
        )
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

    col1, col2 = st.columns(
        2
    )

    analyze = col1.button(
        analyze_label,
        type="primary",
        key=f"analyze_{source}",
        use_container_width=True,
    )

    remove = col2.button(
        remove_label,
        key=f"remove_{source}",
        use_container_width=True,
    )

    if remove:

        clear_analysis_state()

        st.rerun()

    if analyze:

        run_prediction(
            image_file,
            image_bytes,
            original,
            include_explainability,
            source,
        )

    # --------------------------------------------------------
    # Show stored result only for this exact image.
    # --------------------------------------------------------

    stored_result = (
        st.session_state.get(
            "last_result"
        )
    )

    stored_hash = (
        st.session_state.get(
            "result_image_hash"
        )
    )

    stored_source = (
        st.session_state.get(
            "active_source"
        )
    )

    if (
        stored_result
        and stored_hash == current_hash
        and stored_source == source
    ):

        render_results(
            stored_result,
            st.session_state.get(
                "last_image",
                original,
            ),
        )


# ============================================================
# DETECTION PAGE
# ============================================================

def render_detection() -> None:

    st.markdown(
        '<p class="page-title">'
        "Dental Caries Detection"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Upload an existing intraoral photograph "
        "or capture one using your camera."
        "</p>",
        unsafe_allow_html=True,
    )

    include_cam = st.checkbox(
        "Generate Grad-CAM and lesion localization",
        value=True,
    )

    left, middle, right = st.columns(
        [1, 0.12, 1]
    )

    with left:

        st.markdown(
            """
            <div class="input-card">
                <h3>Upload Dental Photograph</h3>
                <p>
                    Select an existing intraoral photograph.
                </p>
                <p style="margin-top:10px;">
                    <strong>Accepted formats</strong>
                </p>
                <p>
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
            help=(
                "Upload an intraoral dental photograph."
            ),
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

    with middle:

        st.markdown(
            '<div class="or-divider">OR</div>',
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="input-card">
                <h3>Capture Using Camera</h3>
                <p>
                    Capture a new intraoral photograph.
                </p>
                <p style="margin-top:10px;">
                    <strong>Live Camera Capture</strong>
                </p>
                <p>
                    Preview it, then analyze or retake.
                </p>
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
                "Camera unavailable. "
                "Please allow browser camera permission "
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
        '<p class="page-title">'
        "Dataset Labeling"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "Upload tooth photographs and assign ICDAS grades 0-4."
        "</p>",
        unsafe_allow_html=True,
    )

    base = Path(
        "dataset"
    )

    uploaded_files = st.file_uploader(
        "Upload Tooth Images",
        type=ACCEPTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="dataset_label_uploader",
    )

    if not uploaded_files:

        st.info(
            "Upload one or more images to begin labeling."
        )

        return

    total = len(
        uploaded_files
    )

    index = int(
        st.session_state.get(
            "label_index",
            0,
        )
    )

    if index >= total:

        st.success(
            "All uploaded images have been labeled."
        )

        if st.button(
            "Start Again",
        ):

            st.session_state[
                "label_index"
            ] = 0

            st.rerun()

        return

    image_file = (
        uploaded_files[index]
    )

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

    col1, col2 = st.columns(
        2
    )

    with col1:

        if st.button(
            "Save & Next",
            type="primary",
            key=f"save_{index}",
            use_container_width=True,
        ):

            # ------------------------------------------------
            # Deterministic split instead of random split.
            #
            # The labeler can change this later.
            # ------------------------------------------------

            total_existing = (
                sum(
                    1
                    for _ in base.rglob(
                        "*.*"
                    )
                )
            )

            remainder = (
                total_existing % 10
            )

            if remainder < 7:

                split = "train"

            elif remainder < 9:

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
            ) as file:

                file.write(
                    image_file.getbuffer()
                )

            st.success(
                f"Saved → "
                f"{split}/{grade}/{image_file.name}"
            )

            st.session_state[
                "label_index"
            ] = index + 1

            st.rerun()

    with col2:

        if st.button(
            "Skip Image",
            key=f"skip_{index}",
            use_container_width=True,
        ):

            st.session_state[
                "label_index"
            ] = index + 1

            st.rerun()

    st.markdown(
        "---"
    )

    st.subheader(
        "Dataset Summary"
    )

    c1, c2, c3 = st.columns(
        3
    )

    for column, split in zip(
        [
            c1,
            c2,
            c3,
        ],
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

            for grade_id in range(
                5
            ):

                folder = (
                    base
                    / split
                    / str(grade_id)
                )

                count = (
                    len(
                        list(
                            folder.glob(
                                "*"
                            )
                        )
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
        '<p class="page-title">'
        "Analysis"
        "</p>",
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

        st.warning(
            error
        )

        return

    if not stats or (
        stats.get(
            "total_analyses",
            0,
        )
        == 0
    ):

        st.info(
            "No scan data available yet."
        )

        return

    distribution = (
        stats.get(
            "grade_distribution"
        )
        or {}
    )

    healthy = int(
        distribution.get(
            "0",
            0,
        )
        or 0
    )

    total = int(
        stats.get(
            "total_analyses",
            0,
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

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
        f"{stats.get('average_confidence', 0):.1f}%",
    )

    left, right = st.columns(
        2
    )

    with left:

        st.markdown(
            "**ICDAS distribution**"
        )

        chart_data = pd.Series(
            {
                f"ICDAS {grade}":
                    distribution.get(
                        str(grade),
                        0,
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

    common = stats.get(
        "most_common_grade"
    )

    st.caption(
        (
            f"Most common grade: ICDAS {common}"
            if common is not None
            else "Most common grade: —"
        )
    )

    render_disclaimer()


# ============================================================
# HISTORY
# ============================================================

def render_history() -> None:

    st.markdown(
        '<p class="page-title">'
        "Scan History"
        "</p>",
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

        st.warning(
            error
        )

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

    if selected_filter != "All":

        selected_grade = int(
            selected_filter.split()[-1]
        )

        rows = [
            row
            for row in rows
            if int(
                row.get(
                    "icdas_grade",
                    -1,
                )
            )
            == selected_grade
        ]

        if not rows:

            st.info(
                f"No stored scans for {selected_filter}."
            )

            return

    records = []

    for row in rows:

        created = row.get(
            "created_at"
        )

        formatted_date = ""

        if created:

            try:

                formatted_date = (
                    datetime.fromisoformat(
                        created.replace(
                            "Z",
                            "+00:00",
                        )
                    ).strftime(
                        "%d %b %Y %H:%M"
                    )
                )

            except Exception:

                formatted_date = str(
                    created
                )

        records.append(
            {
                "Date":
                    formatted_date,

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
        records
    )

    st.dataframe(
        table.drop(
            columns=["id"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    ids = [
        value
        for value in table[
            "id"
        ].tolist()
        if value is not None
    ]

    if not ids:

        return

    selected = st.selectbox(
        "Open record",
        options=ids,
        format_func=lambda value:
            f"Scan #{value}",
    )

    if selected is None:

        return

    detail, error = api_get(
        f"/api/v1/history/{selected}"
    )

    if error:

        st.error(
            error
        )

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
            color=(
                28,
                42,
                54,
            ),
        )

    # IMPORTANT:
    # Preserve the actual grade from backend history.
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
        "probabilities":
            detail.get(
                "probabilities"
            ),
    }

    try:

        payload = (
            validate_prediction_response(
                payload
            )
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
        '<p class="page-title">'
        "ICDAS Guide"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub">'
        "ICDAS reference. The deployed model predicts ICDAS 0-4."
        "</p>",
        unsafe_allow_html=True,
    )

    columns = st.columns(
        2
    )

    for grade, info in (
        ICDAS_GUIDE.items()
    ):

        with columns[
            grade % 2
        ]:

            scope = (
                "Classified by this model"
                if info[
                    "in_model"
                ]
                else
                "Reference only"
            )

            st.markdown(
                f"""
                <div class="guide-card">
                    <div class="guide-icon">
                        {info["icon"]}
                    </div>

                    <h4>
                        ICDAS {grade}
                        → {info["short"]}
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
        '<p class="page-title">'
        "About Dental AI"
        "</p>",
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
        "Dental caries is a progressive disease that "
        "benefits from earlier visual detection. "
        "ICDAS provides a standardized scale for "
        "describing lesion severity."
    )

    st.markdown(
        "#### Purpose"
    )

    st.write(
        "This project supports educational and research "
        "screening of intraoral photographs. The current "
        "model predicts ICDAS 0-4 and returns confidence, "
        "probabilities, and optional Grad-CAM visualizations."
    )

    st.markdown(
        "#### Current AI model"
    )

    st.write(
        "MobileNetV3-Small with CBAM attention and a "
        "five-class softmax output."
    )

    st.markdown(
        "#### ICDAS scope"
    )

    st.write(
        "The deployed model predicts ICDAS 0, 1, 2, 3, and 4. "
        "ICDAS 5 and 6 are shown in the guide for reference "
        "only and are not model outputs."
    )

    st.markdown(
        "#### Local processing"
    )

    st.write(
        "The Streamlit frontend communicates with the "
        "local FastAPI backend. Image inference is performed "
        "through the local TensorFlow model."
    )

    st.markdown(
        "#### Explainable AI"
    )

    st.write(
        "Grad-CAM highlights regions that contributed to "
        "the model's prediction. It should not be interpreted "
        "as exact clinical segmentation."
    )

    st.markdown(
        "#### Workflow"
    )

    st.markdown(
        """
        <div class="workflow">
            Dental Image
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

        st.warning(
            error
        )

    elif info:

        st.write(
            f"**Name:** "
            f"{info.get('name', '—')}"
        )

        st.write(
            f"**ICDAS mode:** "
            f"{info.get('icdas_mode', '—')}"
        )

        st.write(
            f"**Classes:** "
            f"{info.get('num_classes', '—')}"
        )

        st.write(
            f"**Input size:** "
            f"{info.get('image_size', '—')}×"
            f"{info.get('image_size', '—')}"
        )

        st.write(
            "**Ordinal regression:** "
            f"{'yes' if info.get('ordinal_regression') else 'no'}"
        )

        st.caption(
            "These values come directly from the running FastAPI backend."
        )

    render_privacy_card()

    render_disclaimer()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    initialize_session_state()

    # --------------------------------------------------------
    # Apply pending navigation BEFORE radio widget creation.
    # --------------------------------------------------------

    pending_page = (
        st.session_state.get(
            "_pending_nav_page"
        )
    )

    if pending_page:

        st.session_state[
            "nav_page"
        ] = pending_page

        st.session_state[
            "_pending_nav_page"
        ] = None

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