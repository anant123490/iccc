"""
ICDAS Dental AI — Streamlit frontend (ICDAS 0–4).

The CNN performs classification. Groq only writes an explanation.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
LOW_CONFIDENCE_COPY = (
    "Low confidence prediction. Professional examination recommended."
)

URGENCY_COLORS = {
    "LOW": "#0f766e",
    "MODERATE": "#b45309",
    "HIGH": "#b91c1c",
}


st.set_page_config(
    page_title="Dental AI — ICDAS Assessment",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


def backend_url() -> str:
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL).rstrip("/")


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
        detail = exc.response.text if exc.response is not None else str(exc)
        return None, f"Backend error: {detail}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def api_post(path: str, **kwargs):
    try:
        response = requests.post(f"{backend_url()}{path}", timeout=kwargs.pop("timeout", 120), **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to the AI backend. Please ensure the FastAPI server is running."
    except requests.exceptions.Timeout:
        return None, "The request timed out while contacting the AI backend."
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        return None, f"Backend error: {detail}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def b64_to_image(b64_string: str | None) -> Image.Image | None:
    if not b64_string:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_string)))
    except Exception:
        return None


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Sans:wght@500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
        .block-container { padding-top: 1.4rem; max-width: 1280px; }
        .app-header {
            background: #0b1f2a;
            color: #f8fafc;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 18px;
            border: 1px solid #163445;
        }
        .app-header h1 { font-family: 'IBM Plex Sans', sans-serif; font-size: 1.7rem; margin: 0 0 4px 0; }
        .app-header p { margin: 0; color: #cbd5e1; font-size: 0.95rem; }
        .metric-card, .panel {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #fff;
        }
        .upload-box {
            border: 1.5px dashed #94a3b8;
            border-radius: 16px;
            padding: 36px 20px;
            text-align: center;
            background: #f8fafc;
            color: #334155;
        }
        .disclaimer {
            border: 1px solid #f6d98b;
            background: #fffbeb;
            color: #78350f;
            border-radius: 12px;
            padding: 12px 14px;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>Dental AI</h1>
            <p>AI-powered caries assessment using ICDAS 0–4 · MobileNetV3-Small + CBAM + ordinal regression</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str) -> None:
    st.info(message)


def page_dashboard() -> None:
    header()
    st.subheader("Overview")
    stats, error = api_get("/api/v1/stats")
    if error:
        st.warning(error)
        return
    if not stats or stats.get("total_analyses", 0) == 0:
        empty_state("No analyses yet. Run a new analysis to populate this dashboard.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analyses", stats["total_analyses"])
    c2.metric("Average confidence", f"{stats['average_confidence']:.1f}%")
    common = stats.get("most_common_grade")
    c3.metric("Most common grade", f"ICDAS {common}" if common is not None else "—")
    c4.metric("High severity cases", stats["high_severity_cases"])

    left, right = st.columns(2)
    with left:
        st.markdown("**ICDAS distribution**")
        dist = pd.Series({f"ICDAS {k}": v for k, v in stats["grade_distribution"].items()})
        st.bar_chart(dist)
    with right:
        st.markdown("**Confidence distribution**")
        buckets = pd.Series(stats["confidence_buckets"])
        st.bar_chart(buckets)
    st.markdown(
        '<div class="disclaimer">These statistics come from stored predictions only. They are not clinical performance claims.</div>',
        unsafe_allow_html=True,
    )


def render_prediction(result: dict, original: Image.Image) -> None:
    grade = result.get("icdas_grade")
    confidence = float(result.get("confidence") or 0)
    urgency = str(result.get("urgency") or "MODERATE").upper()
    color = URGENCY_COLORS.get(urgency, "#334155")

    st.markdown("### AI classification")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"## ICDAS {grade}")
        st.caption(result.get("description") or result.get("finding") or "")
    with col_b:
        st.markdown(f"### {confidence:.1f}%")
        st.markdown(
            f'<span class="badge" style="background:{color}">{urgency} SEVERITY</span>',
            unsafe_allow_html=True,
        )

    if result.get("low_confidence"):
        st.warning(result.get("low_confidence_message") or LOW_CONFIDENCE_COPY)

    g1, g2, g3 = st.columns(3)
    heatmap = b64_to_image(result.get("heatmap_base64") or result.get("overlay_base64"))
    contour = b64_to_image(result.get("contour_base64"))
    g1.image(original, caption="Original image", use_container_width=True)
    if heatmap:
        g2.image(heatmap, caption="Grad-CAM", use_container_width=True)
    else:
        g2.info("Grad-CAM unavailable for this prediction.")
    if contour:
        g3.image(contour, caption="Highlighted area", use_container_width=True)
    else:
        g3.info("Highlighted contour unavailable.")

    st.markdown("### AI dental assessment")
    st.write("**Classification**")
    st.write(f"ICDAS {grade} — {result.get('label', '')}")
    st.write("**Confidence**")
    st.write(f"{confidence:.1f}%")
    st.write("**Finding**")
    st.write(result.get("finding") or "Not available")
    st.write("**Recommendation**")
    st.write(result.get("recommendation") or "Not available")
    if result.get("report"):
        st.write("**Narrative**")
        st.write(result["report"])

    probabilities = result.get("probabilities") or {}
    if probabilities:
        if isinstance(probabilities, dict):
            chart = {f"ICDAS {k}": float(v) for k, v in probabilities.items()}
        else:
            chart = {f"ICDAS {i}": float(p) for i, p in enumerate(probabilities)}
        st.bar_chart(pd.Series(chart))

    st.markdown(
        f'<div class="disclaimer">{result.get("disclaimer") or "This is AI decision support, not a definitive diagnosis."}</div>',
        unsafe_allow_html=True,
    )


def page_analysis() -> None:
    header()
    st.subheader("New analysis")
    st.markdown(
        '<div class="upload-box"><div style="font-size:32px">🦷</div>'
        "<h3>Upload dental image</h3>"
        "<p>Drag and drop or select an intraoral photograph (JPG, PNG).</p></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    camera = st.camera_input("Or capture from camera")
    image_file = uploaded or camera
    include_cam = st.checkbox("Generate Grad-CAM", value=True)

    if image_file is None:
        empty_state("Upload a dental image to begin ICDAS analysis.")
        return

    image_bytes = image_file.getvalue()
    try:
        original = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError):
        st.error("The selected file is not a valid image. Please upload a JPG or PNG file.")
        return

    if st.button("Run AI assessment", type="primary"):
        with st.spinner("Running MobileNetV3 + CBAM ordinal classifier..."):
            result, error = api_post(
                "/api/v1/predict",
                files={"file": (getattr(image_file, "name", "upload.jpg"), image_bytes, "image/jpeg")},
                params={"include_explainability": str(include_cam).lower()},
            )
        if error:
            st.error(error)
            return
        st.session_state["last_result"] = result
        st.session_state["last_image"] = original

    if "last_result" in st.session_state:
        render_prediction(st.session_state["last_result"], st.session_state.get("last_image", original))


def page_history() -> None:
    header()
    st.subheader("Prediction history")
    rows, error = api_get("/api/v1/history")
    if error:
        st.warning(error)
        return
    if not rows:
        empty_state("No stored predictions yet.")
        return

    table = pd.DataFrame(
        [
            {
                "Date": datetime.fromisoformat(r["created_at"].replace("Z", "")).strftime("%d %b %Y %H:%M")
                if r.get("created_at")
                else "",
                "ICDAS": f"Grade {r['icdas_grade']}",
                "Confidence": f"{r['confidence']:.0f}%",
                "Status": r.get("urgency") or "",
                "id": r["id"],
            }
            for r in rows
        ]
    )
    st.dataframe(table.drop(columns=["id"]), use_container_width=True, hide_index=True)
    selected = st.selectbox("Open record", options=table["id"].tolist(), format_func=lambda i: f"#{i}")
    if selected:
        detail, err = api_get(f"/api/v1/history/{selected}")
        if err:
            st.error(err)
            return
        original = b64_to_image(detail.get("image_base64"))
        payload = {
            **detail,
            "label": f"ICDAS {detail['icdas_grade']}",
            "description": detail.get("finding"),
            "heatmap_base64": detail.get("heatmap_base64"),
            "overlay_base64": detail.get("heatmap_base64"),
        }
        if original is None:
            original = Image.new("RGB", (224, 224), color=(241, 245, 249))
        render_prediction(payload, original)


def page_analytics() -> None:
    header()
    st.subheader("Analytics")
    stats, error = api_get("/api/v1/stats")
    if error:
        st.warning(error)
        return
    if not stats or stats.get("total_analyses", 0) == 0:
        empty_state("Analytics will appear after the first stored prediction.")
        return
    st.metric("Prediction count", stats["total_analyses"])
    st.bar_chart(pd.Series({f"ICDAS {k}": v for k, v in stats["grade_distribution"].items()}))
    st.bar_chart(pd.Series(stats["confidence_buckets"]))


def page_settings() -> None:
    header()
    st.subheader("Settings")
    st.session_state.backend_url = st.text_input("Backend URL", value=backend_url())
    health, err = api_get("/api/v1/health")
    info, _ = api_get("/api/v1/model/info")
    if err:
        st.error(err)
        st.caption("Backend connection: unavailable")
    else:
        st.success("Backend connection: online")
        st.write(f"API status: {health.get('status')}")
        st.write(f"Model loaded: {'yes' if health.get('model_loaded') else 'no'}")
        st.write(f"Database: {'ok' if health.get('database_ok') else 'unavailable'}")
        st.write(f"Groq configured: {'yes' if health.get('groq_configured') else 'no (local fallback reports)'}")
        if info:
            st.write(f"Model: {info.get('name')} · {info.get('architecture')}")
            st.write(f"Classes: {info.get('num_classes')} ({info.get('icdas_mode')})")
    st.caption("API keys and database passwords are never shown in this interface.")


inject_css()
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

with st.sidebar:
    st.markdown("**🦷 DENTAL AI**")
    st.caption("AI-powered caries assessment")
    page = st.radio(
        "Navigation",
        ["Dashboard", "New Analysis", "History", "Analytics", "Settings"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Research prototype. Not a replacement for professional dental diagnosis.")

if page == "Dashboard":
    page_dashboard()
elif page == "New Analysis":
    page_analysis()
elif page == "History":
    page_history()
elif page == "Analytics":
    page_analytics()
else:
    page_settings()
