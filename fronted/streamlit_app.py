"""
ICDAS Dental Caries Detection — Streamlit Frontend
====================================================

Flow:

    Dental Image
          |
    FastAPI /api/v1/predict
          |
    ICDAS Model
          |
    Grade + Confidence + Clinical Finding
          |
    FastAPI /api/v1/report
          |
    Groq (LLM)
          |
    AI Clinical-Support Report
          |
    Streamlit

--------------------------------------------------------------------
WHY THE OLD VERSION SHOWED RAW HTML TAGS AS TEXT
--------------------------------------------------------------------
Markdown treats any line indented by 4+ spaces as a fenced code
block. The previous file built its HTML inside heavily-indented
f-strings and passed them straight to st.markdown(...). Streamlit's
Markdown parser saw the indentation, decided "this is code", and
rendered the literal tags instead of the HTML — which is exactly
what your screenshots show.

Fix: every HTML string that goes into `render_html()` is passed
through `textwrap.dedent()` first, which strips the common leading
whitespace so Markdown parses it as real HTML.
"""

import base64
import io
import textwrap
from datetime import datetime

import requests
import streamlit as st
from PIL import Image


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ICDAS Dental AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONFIG / CONSTANTS
# ============================================================================

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# Grade 0-6 ICDAS scale -> (color, human label)
URGENCY_STYLE = {
    "low":      {"color": "#22c55e", "glow": "rgba(34,197,94,0.35)",  "label": "LOW"},
    "medium":   {"color": "#f59e0b", "glow": "rgba(245,158,11,0.35)", "label": "MEDIUM"},
    "high":     {"color": "#f97316", "glow": "rgba(249,115,22,0.35)", "label": "HIGH"},
    "critical": {"color": "#ef4444", "glow": "rgba(239,68,68,0.40)",  "label": "CRITICAL"},
}
DEFAULT_URGENCY_STYLE = {"color": "#94a3b8", "glow": "rgba(148,163,184,0.30)", "label": "UNKNOWN"}


def render_html(html: str) -> None:
    """
    Safely render an HTML string in Streamlit.

    IMPORTANT: this uses st.html(), NOT st.markdown(..., unsafe_allow_html=True).

    st.markdown() runs its input through a Markdown parser first, and Markdown's
    spec treats ANY line indented 4+ spaces as a fenced code block. textwrap.dedent()
    only removes the *common* leading whitespace across all lines — nested tags
    still have relative indentation deeper than their parent, so inner lines of a
    nested HTML block can still trip the "4-space = code block" rule and get
    printed as literal text (this was the root cause of the still-broken card).

    st.html() bypasses Markdown parsing entirely and renders the string as raw
    HTML, so indentation/nesting can never be misinterpreted as a code block.
    """
    st.html(textwrap.dedent(html))


# ============================================================================
# GLOBAL CSS — "professional clinical-tech" theme
# ============================================================================

render_html(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---------- Hero header ---------- */
    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 45%, #312e81 100%);
        border-radius: 20px;
        padding: 34px 38px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(49,46,129,0.25);
    }
    .hero::after {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, rgba(129,140,248,0.35), transparent 70%);
    }
    .hero-eyebrow {
        display: inline-block;
        font-size: 11px; font-weight: 700; letter-spacing: 2px;
        color: #a5b4fc; background: rgba(129,140,248,0.12);
        border: 1px solid rgba(129,140,248,0.35);
        padding: 4px 12px; border-radius: 20px; margin-bottom: 14px;
    }
    .hero-title {
        font-size: 34px; font-weight: 800; color: #f8fafc; margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle { font-size: 15px; color: #cbd5e1; max-width: 620px; line-height: 1.5; }

    /* ---------- Section labels ---------- */
    .section-eyebrow {
        font-size: 11px; font-weight: 700; letter-spacing: 2px;
        color: #818cf8; margin-bottom: 4px; text-transform: uppercase;
    }
    .section-title { font-size: 22px; font-weight: 700; margin-bottom: 4px; color: inherit; }
    .section-caption { font-size: 13px; opacity: 0.6; margin-bottom: 18px; }

    /* ---------- AI Suggestion Card ---------- */
    .ai-card {
        border-radius: 22px;
        padding: 2px;
        background: linear-gradient(135deg, var(--glow), transparent 60%);
        margin-bottom: 18px;
    }
    .ai-card-inner {
        border-radius: 20px;
        padding: 26px 28px;
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(148,163,184,0.15);
    }
    .ai-card-top {
        display: flex; justify-content: space-between; align-items: flex-start;
        margin-bottom: 22px;
    }
    .grade-eyebrow {
        font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
        opacity: 0.55; margin-bottom: 6px;
    }
    .grade-number {
        font-size: 40px; font-weight: 800; line-height: 1;
        background: linear-gradient(135deg, #fff, #cbd5e1);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .urgency-pill {
        display: flex; align-items: center; gap: 6px;
        padding: 8px 16px; border-radius: 30px;
        font-size: 12px; font-weight: 800; letter-spacing: 0.6px;
        color: #0b1120;
        background: var(--pill-color);
        box-shadow: 0 0 18px var(--glow);
    }
    .urgency-dot { width: 7px; height: 7px; border-radius: 50%; background: #0b1120; opacity: 0.55; }

    /* Confidence ring */
    .ring-wrap { display: flex; align-items: center; gap: 16px; }
    .ring-value { font-size: 24px; font-weight: 800; }
    .ring-label { font-size: 11px; font-weight: 600; letter-spacing: 1px; opacity: 0.55; }

    .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 20px 0; }
    .info-tile {
        border-radius: 14px; padding: 16px 18px;
        background: rgba(148,163,184,0.06);
        border: 1px solid rgba(148,163,184,0.12);
    }
    .info-tile-label {
        font-size: 10.5px; font-weight: 700; letter-spacing: 1.2px;
        opacity: 0.5; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
    }
    .info-tile-value { font-size: 15px; font-weight: 500; line-height: 1.5; }

    .note-line {
        font-size: 12.5px; opacity: 0.55; margin-top: 4px;
        display: flex; gap: 6px; align-items: flex-start;
    }

    /* ---------- Report ---------- */
    .report-shell {
        border-radius: 20px;
        border: 1px solid rgba(148,163,184,0.15);
        background: linear-gradient(180deg, rgba(99,102,241,0.05), rgba(15,23,42,0) 30%);
        padding: 0;
        overflow: hidden;
        margin-top: 6px;
    }
    .report-header {
        padding: 22px 28px;
        border-bottom: 1px solid rgba(148,163,184,0.14);
        display: flex; justify-content: space-between; align-items: center;
        background: rgba(99,102,241,0.06);
    }
    .report-title { font-size: 19px; font-weight: 700; display:flex; align-items:center; gap:8px;}
    .report-subtitle { font-size: 12.5px; opacity: 0.55; margin-top: 3px; }
    .report-badge {
        font-size: 10.5px; font-weight: 700; letter-spacing: 1px;
        color: #a5b4fc; background: rgba(129,140,248,0.12);
        border: 1px solid rgba(129,140,248,0.3);
        padding: 5px 12px; border-radius: 20px;
    }
    .report-body { padding: 24px 28px; font-size: 14.5px; line-height: 1.7; }
    .report-body p { margin-bottom: 12px; }

    .report-fail {
        padding: 26px 28px; text-align: center;
    }
    .report-fail-icon { font-size: 30px; margin-bottom: 8px; }
    .report-fail-title { font-weight: 700; font-size: 15px; margin-bottom: 4px; }
    .report-fail-sub { font-size: 13px; opacity: 0.6; max-width: 420px; margin: 0 auto; }

    /* ---------- Disclaimer ---------- */
    .disclaimer-box {
        border: 1px solid rgba(245,158,11,0.35);
        background: rgba(245,158,11,0.07);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 16px;
        font-size: 12.5px;
        display: flex; gap: 10px; align-items: flex-start;
    }

    /* ---------- Upload zone ---------- */
    .upload-title { font-size: 19px; font-weight: 700; margin-bottom: 2px; }
    .upload-caption { font-size: 13px; opacity: 0.6; margin-bottom: 14px; }

    /* ---------- History chips ---------- */
    .history-chip {
        display: flex; justify-content: space-between; align-items: center;
        padding: 9px 12px; border-radius: 10px;
        background: rgba(148,163,184,0.06);
        border: 1px solid rgba(148,163,184,0.1);
        margin-bottom: 6px; font-size: 12.5px;
    }
    .history-chip-grade { font-weight: 700; }
    .history-chip-time { opacity: 0.5; font-family: 'JetBrains Mono', monospace; font-size: 11px; }

    </style>
    """
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🦷 ICDAS Dental AI")
    st.caption("AI-assisted dental caries screening & clinical decision support")
    st.divider()

    backend_url = st.text_input("FastAPI Backend URL", value=DEFAULT_BACKEND_URL)
    include_explainability = st.checkbox("Show Grad-CAM explainability", value=True)

    st.divider()
    st.markdown("**Session History**")

    if not st.session_state.history:
        st.caption("No scans yet this session.")
    else:
        for item in reversed(st.session_state.history):
            render_html(
                f"""
                <div class="history-chip">
                    <span class="history-chip-grade">Grade {item['grade']} · {item['confidence']:.1f}%</span>
                    <span class="history-chip-time">{item['time']}</span>
                </div>
                """
            )

    st.divider()
    st.caption("⚠️ AI output is not a final diagnosis. Use this system as clinical decision support only.")


# ============================================================================
# FASTAPI — PREDICTION
# ============================================================================

def call_predict_api(image_bytes: bytes, filename: str) -> dict | None:
    """Calls POST /api/v1/predict — sends the image to the ICDAS model."""
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        params = {"include_explainability": str(include_explainability).lower()}

        response = requests.post(
            f"{backend_url}/api/v1/predict",
            files=files,
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to FastAPI at `{backend_url}`.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Prediction request timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        error_text = e.response.text if e.response is not None else str(e)
        st.error(f"❌ Prediction failed: {status_code}")
        st.code(error_text)
        return None
    except Exception as e:
        st.error(f"❌ Unexpected prediction error: {e}")
        return None


# ============================================================================
# FASTAPI — GROQ REPORT
# ============================================================================

def call_report_api(icdas_grade, confidence) -> str | None:
    """
    Calls POST /api/v1/report.

    Prediction API may return confidence as 32.0 (meaning 32%), but the
    report API expects a 0-1 float, so we normalize before sending.
    """
    try:
        try:
            grade = int(icdas_grade)
        except (ValueError, TypeError):
            st.error(f"Invalid ICDAS grade: {icdas_grade}")
            return None

        try:
            confidence_value = float(confidence)
        except (ValueError, TypeError):
            confidence_value = 0.0

        if confidence_value > 1:
            confidence_value = confidence_value / 100.0
        confidence_value = max(0.0, min(confidence_value, 1.0))

        payload = {"icdas_grade": grade, "confidence": confidence_value}

        response = requests.post(
            f"{backend_url}/api/v1/report",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        report = data.get("report")

        if not report:
            st.session_state["_last_report_error_payload"] = data
            return None

        return report

    except requests.exceptions.ConnectionError:
        st.session_state["_last_report_error_payload"] = {"error": "connection_error"}
        return None
    except requests.exceptions.Timeout:
        st.session_state["_last_report_error_payload"] = {"error": "timeout"}
        return None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        error_text = e.response.text if e.response is not None else str(e)
        st.session_state["_last_report_error_payload"] = {"error": f"http_{status_code}", "detail": error_text}
        return None
    except Exception as e:
        st.session_state["_last_report_error_payload"] = {"error": str(e)}
        return None


# ============================================================================
# BASE64 IMAGE CONVERSION
# ============================================================================

def b64_to_image(b64_string: str | None) -> Image.Image | None:
    if not b64_string:
        return None
    try:
        image_bytes = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None


# ============================================================================
# PAGE HEADER
# ============================================================================

render_html(
    """
    <div class="hero">
        <div class="hero-eyebrow">CLINICAL DECISION SUPPORT · ICDAS</div>
        <div class="hero-title">🦷 Dental Caries Detection</div>
        <div class="hero-subtitle">
            Upload an intraoral photograph and get an ICDAS-grade classification,
            Grad-CAM explainability, and an AI-generated clinical-support report —
            powered by a computer-vision model and Groq LLM.
        </div>
    </div>
    """
)


# ============================================================================
# IMAGE UPLOAD
# ============================================================================

render_html(
    """
    <div class="upload-title">📤 Dental Image</div>
    <div class="upload-caption">Upload an intraoral photograph or use your camera.</div>
    """
)

col_upload, col_camera = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

with col_camera:
    camera_file = st.camera_input("Take a photo")

image_file = uploaded_file or camera_file


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

if image_file is not None:

    image_bytes = image_file.getvalue()

    try:
        original_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Uploaded Image")
        st.image(original_image, use_container_width=True)

    with st.spinner("Analyzing dental image..."):
        result = call_predict_api(image_bytes, getattr(image_file, "name", "upload.jpg"))

    if result:

        grade = result.get("icdas_grade", "?")
        confidence = result.get("confidence", 0)
        urgency = result.get("urgency", "medium")
        finding = result.get("finding", "Not available")
        recommendation = result.get("recommendation", "Not available")
        ai_result_note = result.get("ai_result_note", "AI result is not a final diagnosis.")
        disclaimer = result.get("disclaimer", "")

        try:
            confidence_float = float(confidence)
        except (ValueError, TypeError):
            confidence_float = 0.0

        confidence_percentage = confidence_float * 100 if 0 <= confidence_float <= 1 else confidence_float

        urgency_key = str(urgency).lower()
        style = URGENCY_STYLE.get(urgency_key, DEFAULT_URGENCY_STYLE)

        # =====================================================================
        # AI SUGGESTION CARD
        # =====================================================================

        with right:
            st.subheader("AI Suggestion")

            render_html(
                f"""
                <div class="ai-card" style="--glow:{style['glow']};">
                    <div class="ai-card-inner">

                        <div class="ai-card-top">
                            <div>
                                <div class="grade-eyebrow">ICDAS CLASSIFICATION</div>
                                <div class="grade-number">Grade {grade}</div>
                            </div>
                            <div class="urgency-pill" style="--pill-color:{style['color']}; --glow:{style['glow']};">
                                <span class="urgency-dot"></span>{style['label']}
                            </div>
                        </div>

                        <div class="info-grid">
                            <div class="info-tile">
                                <div class="info-tile-label">📊 CONFIDENCE</div>
                                <div class="ring-value">{confidence_percentage:.1f}%</div>
                            </div>
                            <div class="info-tile">
                                <div class="info-tile-label">⚡ URGENCY</div>
                                <div class="ring-value" style="color:{style['color']};">{style['label']}</div>
                            </div>
                        </div>

                        <div class="info-tile" style="margin-bottom:14px;">
                            <div class="info-tile-label">🔬 CLINICAL FINDING</div>
                            <div class="info-tile-value">{finding}</div>
                        </div>

                        <div class="info-tile">
                            <div class="info-tile-label">✅ RECOMMENDATION</div>
                            <div class="info-tile-value">{recommendation}</div>
                        </div>

                    </div>
                </div>
                """
            )

            if ai_result_note:
                render_html(f'<div class="note-line">ℹ️ <span>{ai_result_note}</span></div>')
            if disclaimer:
                render_html(f'<div class="disclaimer-box">⚠️ <span>{disclaimer}</span></div>')

        # =========================================================================
        # GROQ REPORT
        # =========================================================================

        st.divider()

        with st.spinner("Generating clinical-support report..."):
            ai_report = call_report_api(grade, confidence_float)

        if ai_report:
            render_html(
                f"""
                <div class="report-shell">
                    <div class="report-header">
                        <div>
                            <div class="report-title">📝 AI Dental Clinical-Support Report</div>
                            <div class="report-subtitle">Generated using the ICDAS prediction and Groq AI</div>
                        </div>
                        <div class="report-badge">GROQ LLM</div>
                    </div>
                    <div class="report-body">
                """
            )
            st.markdown(ai_report)
            render_html("</div></div>")
        else:
            error_payload = st.session_state.get("_last_report_error_payload", {})
            render_html(
                f"""
                <div class="report-shell">
                    <div class="report-header">
                        <div>
                            <div class="report-title">📝 AI Dental Clinical-Support Report</div>
                            <div class="report-subtitle">Generated using the ICDAS prediction and Groq AI</div>
                        </div>
                        <div class="report-badge" style="color:#f87171; border-color:rgba(248,113,113,0.35); background:rgba(248,113,113,0.1);">
                            UNAVAILABLE
                        </div>
                    </div>
                    <div class="report-fail">
                        <div class="report-fail-icon">🩺</div>
                        <div class="report-fail-title">The ICDAS prediction succeeded, but the AI report could not be generated.</div>
                        <div class="report-fail-sub">
                            The backend's /api/v1/report call returned no report content. This usually means
                            the Groq API key is missing/invalid on the FastAPI server, the request timed out,
                            or the report field name doesn't match what the frontend expects.
                        </div>
                    </div>
                </div>
                """
            )
            if error_payload:
                with st.expander("🔧 Debug details"):
                    st.json(error_payload)

        # =========================================================================
        # CLASS PROBABILITIES
        # =========================================================================

        probabilities = result.get("probabilities", [])

        if probabilities:
            st.divider()
            with st.expander("📊 Class Probabilities"):
                st.bar_chart({f"Grade {i}": p for i, p in enumerate(probabilities)})

        # =========================================================================
        # GRAD-CAM
        # =========================================================================

        if include_explainability:
            st.divider()
            st.subheader("🔍 Explainability — Grad-CAM")
            st.caption("Highlighted regions show areas that contributed to the model prediction.")

            heatmap_img = b64_to_image(result.get("heatmap_base64"))
            overlay_img = b64_to_image(result.get("overlay_base64"))
            contour_img = b64_to_image(result.get("contour_base64"))

            hcol1, hcol2, hcol3 = st.columns(3)

            with hcol1:
                st.markdown("**Heatmap**")
                if heatmap_img:
                    st.image(heatmap_img, use_container_width=True)
                else:
                    st.info("Heatmap unavailable.")

            with hcol2:
                st.markdown("**Overlay**")
                if overlay_img:
                    st.image(overlay_img, use_container_width=True)
                else:
                    st.info("Overlay unavailable.")

            with hcol3:
                st.markdown("**Lesion Contour**")
                if contour_img:
                    st.image(contour_img, use_container_width=True)
                else:
                    st.info("Contour unavailable.")

        # =========================================================================
        # SESSION HISTORY
        # =========================================================================

        st.session_state.history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "grade": grade,
                "confidence": confidence_percentage,
            }
        )

else:
    st.info("👆 Upload or capture a dental image to begin ICDAS analysis.")