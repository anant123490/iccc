"""
ICDAS Dental Caries Detection — Streamlit Frontend
Replaces the React/Node.js frontend. Calls the existing FastAPI backend
(backend/app/main.py) for inference, Grad-CAM, and clinical recommendations.

Run:
    pip install streamlit requests pillow
    streamlit run streamlit_app.py

Make sure the backend is running first:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

import base64
import io
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ICDAS Dental Caries Detection",
    page_icon="🦷",
    layout="wide",
)

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

URGENCY_COLORS = {
    "low": "#2e7d32",       # green
    "medium": "#f9a825",    # yellow
    "high": "#ef6c00",      # orange
    "critical": "#c62828",  # red
}

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🦷 ICDAS Detector")
    st.caption("Offline-capable AI dental caries screening")

    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    include_explainability = st.checkbox("Show Grad-CAM heatmap", value=True)

    st.divider()
    st.subheader("Session History")
    if not st.session_state.history:
        st.caption("No scans yet this session.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.write(f"**{item['time']}** — Grade {item['grade']} ({item['confidence']:.1f}%)")

    st.divider()
    st.caption(
        "⚠️ AI result is not a final diagnosis. "
        "This tool is for clinical decision support only."
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def call_predict_api(image_bytes: bytes, filename: str) -> dict | None:
    """POST the image to the FastAPI backend and return the parsed JSON response."""
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        params = {"include_explainability": str(include_explainability).lower()}
        resp = requests.post(
            f"{backend_url}/api/v1/predict",
            files=files,
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the backend at `{backend_url}`. "
            "Is it running? (`uvicorn app.main:app --reload --port 8000`)"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend returned an error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error calling backend: {e}")
    return None


def b64_to_image(b64_string: str) -> Image.Image | None:
    if not b64_string:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_string)))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Dental Caries Detection (ICDAS Grading)")
st.write(
    "Upload an intraoral photo to get an AI-assisted ICDAS grade (0–6), "
    "confidence score, and a suggested clinical action."
)

col_upload, col_camera = st.columns(2)
with col_upload:
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
with col_camera:
    camera_file = st.camera_input("Or take a photo")

image_file = uploaded_file or camera_file

if image_file is not None:
    image_bytes = image_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes))

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with st.spinner("Running AI inference..."):
        result = call_predict_api(image_bytes, getattr(image_file, "name", "upload.jpg"))

    if result:
        grade = result.get("icdas_grade", "?")
        confidence = result.get("confidence", 0)
        urgency = result.get("urgency", "medium")
        color = URGENCY_COLORS.get(urgency, "#999999")

        with right:
            st.subheader("AI Suggestion")
            st.markdown(
                f"""
                <div style="border-left: 6px solid {color}; padding: 12px 16px;
                            background-color: rgba(128,128,128,0.08); border-radius: 6px;">
                    <h3 style="margin-top:0;">ICDAS Grade {grade}</h3>
                    <p><b>Confidence:</b> {confidence:.1f}%</p>
                    <p><b>Finding:</b> {result.get('finding', '—')}</p>
                    <p><b>Recommendation:</b> {result.get('recommendation', '—')}</p>
                    <p><b>Urgency:</b> <span style="color:{color}; font-weight:bold;">
                        {urgency.upper()}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(result.get("ai_result_note", "AI result is not final diagnosis."))
            st.caption(result.get("disclaimer", ""))

            with st.expander("Class probabilities"):
                probs = result.get("probabilities", [])
                if probs:
                    st.bar_chart({f"Grade {i}": p for i, p in enumerate(probs)})

        # Grad-CAM visuals
        if include_explainability:
            st.divider()
            st.subheader("Explainability (Grad-CAM)")
            hcol1, hcol2, hcol3 = st.columns(3)

            heatmap_img = b64_to_image(result.get("heatmap_base64"))
            overlay_img = b64_to_image(result.get("overlay_base64"))
            contour_img = b64_to_image(result.get("contour_base64"))

            with hcol1:
                if heatmap_img:
                    st.image(heatmap_img, caption="Heatmap", use_container_width=True)
            with hcol2:
                if overlay_img:
                    st.image(overlay_img, caption="Overlay", use_container_width=True)
            with hcol3:
                if contour_img:
                    st.image(contour_img, caption="Lesion Contour", use_container_width=True)

        # Save to session history
        st.session_state.history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "grade": grade,
                "confidence": confidence,
            }
        )
else:
    st.info("Upload or capture a photo above to get started.")