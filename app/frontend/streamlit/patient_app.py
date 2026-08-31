"""CCC AI Dentist Camera 2.0 — Patient portal. ML runs only on the FastAPI backend."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.auth import DEFAULT_BACKEND, PortalClient
from shared.charts import icdas_charts
from shared.components import b64_image, icdas_badge, kpi, show_disclaimer
from shared.language import t
from shared.report import html_bytes
from shared.theme import apply

st.set_page_config(page_title="CCC AI Dentist Camera 2.0", page_icon="🦷", layout="wide")
apply()

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "page" not in st.session_state:
    st.session_state.page = "landing"

lang = st.session_state.lang
backend = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND)
client = PortalClient(backend)

try:
    health = requests.get(f"{backend.rstrip('/')}/api/v1/portal/health", timeout=8).json()
except Exception:
    health = {}
if health:
    det_ok = health.get("detector_v2")
    icd_ok = health.get("icdas_loaded")
    st.sidebar.caption(
        f"Tooth Detector V2: {health.get('detector_status', 'AVAILABLE' if det_ok else 'UNAVAILABLE')} · "
        f"ICDAS: {health.get('icdas_status', 'NOT_TRAINED / NOT_DEPLOYED')}"
    )
    if not det_ok:
        st.sidebar.error(health.get("detector_error") or "Tooth Detector V2 is unavailable.")
    if not icd_ok:
        st.sidebar.warning("ICDAS model not yet deployed. Detection and crops still work.")

st.sidebar.selectbox(
    t(lang, "language"),
    ["en", "hi", "kn"],
    key="lang",
    format_func=lambda x: {"en": "English", "hi": "हिन्दी", "kn": "ಕನ್ನಡ"}[x],
)

NAV = [
    ("landing", "Home"),
    ("register", t(lang, "register")),
    ("upload", t(lang, "upload")),
    ("detect", t(lang, "detect")),
    ("icdas", t(lang, "icdas")),
    ("dashboard", t(lang, "dashboard")),
    ("report", t(lang, "report")),
    ("history", t(lang, "history")),
    ("about", t(lang, "about_icdas")),
    ("help", t(lang, "help")),
]
NAV_BY_LABEL = {label: key for key, label in NAV}
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = "Home"


def _sync_nav() -> None:
    st.session_state.page = NAV_BY_LABEL.get(st.session_state.nav_radio, "landing")


st.sidebar.radio("Menu", [label for _, label in NAV], key="nav_radio", on_change=_sync_nav)
st.session_state.page = NAV_BY_LABEL.get(st.session_state.nav_radio, st.session_state.page)
show_disclaimer(t(lang, "disclaimer"))

page = st.session_state.page

if page == "landing":
    st.markdown(
        f'<div class="ccc-hero"><h1>{t(lang, "brand")}</h1><p>{t(lang, "subtitle")}</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    if c1.button(t(lang, "start"), use_container_width=True):
        st.session_state.nav_radio = t(lang, "register")
        st.session_state.page = "register"
        st.rerun()
    if c2.button(t(lang, "previous"), use_container_width=True):
        st.session_state.nav_radio = t(lang, "history")
        st.session_state.page = "history"
        st.rerun()
    if c3.button(t(lang, "about_icdas"), use_container_width=True):
        st.session_state.nav_radio = t(lang, "about_icdas")
        st.session_state.page = "about"
        st.rerun()
    if c4.button(t(lang, "help"), use_container_width=True):
        st.session_state.nav_radio = t(lang, "help")
        st.session_state.page = "help"
        st.rerun()
    st.subheader(t(lang, "about"))
    st.write(
        "This portal screens intraoral photographs. YOLO Tooth Detector V2 finds teeth. "
        "A MobileNetV3+CBAM model predicts ICDAS 0–4 on each crop. Grad-CAM shows attention. "
        "Groq only explains those structured scores."
    )
    st.subheader(t(lang, "how"))
    st.write("Photo → quality check → tooth boxes → crops → ICDAS 0–4 → heatmap → Groq summary.")
    st.caption("Languages: English, हिन्दी, ಕನ್ನಡ. ICDAS numbers stay 0–4.")

elif page == "register":
    st.header(t(lang, "register"))
    with st.form("reg"):
        pid = st.text_input("Patient ID (optional, auto if blank)")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["", "Female", "Male", "Other", "Prefer not to say"])
        phone = st.text_input("Phone (optional)")
        notes = st.text_area("Visit notes")
        visit_date = st.date_input("Visit date", value=date.today())
        submitted = st.form_submit_button("Save and continue")
    if submitted:
        if not name.strip():
            st.error("Name is required.")
        else:
            try:
                with st.spinner("Saving…"):
                    out = client.post(
                        "/api/v1/patients",
                        json={
                            "public_id": pid or None,
                            "name": name.strip(),
                            "age": int(age),
                            "gender": gender or None,
                            "phone": phone or None,
                            "notes": notes or None,
                            "visit_date": str(visit_date),
                        },
                    )
                st.session_state.patient = out["patient"]
                st.session_state.visit_id = out["visit_id"]
                st.success(f"Patient {out['patient']['public_id']} · visit {out['visit_id']}")
                st.session_state.nav_radio = t(lang, "upload")
                st.session_state.page = "upload"
                st.rerun()
            except requests.HTTPError as exc:
                st.error(exc.response.text if exc.response is not None else str(exc))

elif page == "upload":
    st.header(t(lang, "upload"))
    if "visit_id" not in st.session_state:
        st.warning("Register a patient first.")
    else:
        files = st.file_uploader(
            "JPG / JPEG / PNG — multiple intraoral photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        cam = st.camera_input("Camera (if available)")
        blobs = list(files or [])
        if cam is not None:
            blobs.append(cam)
        if st.button("Upload") and blobs:
            body = [("files", (getattr(f, "name", "camera.jpg"), f.getvalue(), "image/jpeg")) for f in blobs]
            try:
                with st.spinner("Uploading and checking quality…"):
                    out = requests.post(
                        f"{backend.rstrip('/')}/api/v1/visits/{st.session_state.visit_id}/images",
                        files=body,
                        timeout=120,
                    )
                    out.raise_for_status()
                    st.session_state.uploads = out.json()
                st.success("Upload complete.")
            except requests.HTTPError as exc:
                st.error(exc.response.text if exc.response is not None else str(exc))
        uploads = st.session_state.get("uploads") or {}
        for im in uploads.get("images") or []:
            q = im.get("quality") or {}
            verdict = q.get("verdict", "?")
            color = {"PASS": "#16a34a", "WARNING": "#ca8a04", "FAIL": "#dc2626"}.get(verdict, "#64748b")
            st.markdown(
                f'<div style="background:{color};color:white;padding:10px 14px;border-radius:10px;'
                f'font-weight:700;margin-bottom:8px;">Quality: {verdict}</div>',
                unsafe_allow_html=True,
            )
            if im.get("duplicate"):
                st.info("Duplicate of an image already on this visit.")
            cols = st.columns([1, 2])
            with cols[0]:
                b64_image(im.get("preview_base64"), width=240)
            with cols[1]:
                st.write(q.get("message"))
                st.write("Warnings:", ", ".join(q.get("warnings") or []) or "None")
                st.caption(f"Sharpness {q.get('sharpness')} · Brightness {q.get('brightness')}")
        if uploads.get("images") and st.button(t(lang, "analyze")):
            st.session_state.nav_radio = t(lang, "detect")
            st.session_state.page = "detect"
            st.rerun()

elif page == "detect":
    st.header(t(lang, "detect"))
    vid = st.session_state.get("visit_id")
    if not vid:
        st.warning("No active visit.")
    else:
        if st.button("Run Tooth Detector V2"):
            try:
                bar = st.progress(0, text="Starting analysis…")
                bar.progress(20, text="Running Tooth Detector V2…")
                with st.spinner("Running Tooth Detector V2…"):
                    st.session_state.analysis = client.post(f"/api/v1/visits/{vid}/analyze")
                bar.progress(100, text="Detection complete.")
                st.success("Tooth detection finished.")
            except requests.HTTPError as exc:
                st.error(exc.response.text if exc.response is not None else str(exc))
        analysis = st.session_state.get("analysis") or {}
        if analysis.get("icdas_status") == "NOT_TRAINED / NOT_DEPLOYED" or analysis.get("message"):
            st.warning(analysis.get("message") or "ICDAS model not yet deployed.")
        for det in analysis.get("detections") or []:
            st.subheader(f"Image {det.get('image_id')} · {det.get('teeth_detected')} teeth")
            a, b = st.columns(2)
            with a:
                b64_image(det.get("original_base64"), "Original", width=420)
            with b:
                b64_image(det.get("overlay_base64"), "Detector overlay", width=420)
            st.caption(f"Mean detection confidence: {det.get('mean_confidence')}")
            with st.expander("Detection list"):
                for crop in det.get("crops") or []:
                    st.write(
                        f"Tooth {crop.get('index')} · conf {crop.get('confidence')} · box {crop.get('box')}"
                    )
        st.subheader(t(lang, "crops"))
        crops = []
        for det in analysis.get("detections") or []:
            crops.extend(det.get("crops") or [])
        n = max(1, min(4, len(crops) or 1))
        for i in range(0, len(crops), n):
            row = crops[i : i + n]
            cols = st.columns(len(row))
            for col, crop in zip(cols, row):
                with col:
                    b64_image(crop.get("crop_base64"), width=180)
                    st.caption(f"Det conf {crop.get('confidence')}")

elif page == "icdas":
    st.header(t(lang, "icdas"))
    analysis = st.session_state.get("analysis") or {}
    teeth = analysis.get("teeth") or []
    if analysis.get("icdas_status") == "NOT_TRAINED / NOT_DEPLOYED" or not teeth:
        st.warning("ICDAS model not yet deployed. No ICDAS grades, heatmaps, or Groq tooth findings were generated.")
    if not teeth:
        st.info("Run tooth detection first. ICDAS classification is unavailable until a 5-class model is deployed.")
    for tooth in teeth:
        grade = int(tooth["icdas_grade"])
        st.markdown(icdas_badge(grade) + f" · {tooth.get('current_stage', '')} · {tooth['confidence']}%", unsafe_allow_html=True)
        st.caption(f"Priority: {tooth.get('priority', '')}")
        with st.expander("What this means"):
            st.write(tooth.get("explanation") or "")
            st.write("Suggested next step:", tooth.get("next_step") or "")
        tabs = st.tabs(["Original crop", "Heatmap", "Overlay", "Confidence"])
        with tabs[0]:
            b64_image(tooth.get("crop_base64"), width=280)
        with tabs[1]:
            b64_image(tooth.get("heatmap_base64"), width=280)
            st.caption(t(lang, "heatmap_note"))
        with tabs[2]:
            b64_image(tooth.get("overlay_base64"), width=280)
        with tabs[3]:
            st.write(tooth.get("probabilities") or {})
            st.caption("Blue = low AI attention. Red = high AI attention. Heatmap is attention, not lesion segmentation.")
        if grade >= 3:
            st.warning("Highlighted: ICDAS ≥ 3")

elif page == "dashboard":
    st.header(t(lang, "dashboard"))
    analysis = st.session_state.get("analysis") or {}
    if not analysis:
        st.info("Run analysis first.")
    else:
        if (analysis.get("teeth_analyzed") or 0) == 0:
            st.warning("ICDAS model not yet deployed. Summary cards below are detection-only.")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Teeth detected", analysis.get("teeth_detected"))
        with k2:
            kpi("Teeth analyzed", analysis.get("teeth_analyzed"))
        with k3:
            kpi("Mean confidence %", analysis.get("mean_confidence"))
        with k4:
            kpi("Highest ICDAS", (analysis.get("summary") or {}).get("highest_grade"))
        icdas_charts(analysis.get("icdas_distribution") or {})
        high = analysis.get("high_severity_indices") or []
        if analysis.get("icdas_status") == "NOT_TRAINED / NOT_DEPLOYED" or (
            analysis.get("teeth_analyzed") or 0
        ) == 0:
            st.info("ICDAS model not yet deployed — no ICDAS severity summary.")
        elif high:
            st.error(f"Crops with ICDAS ≥ 3: {high}")
        else:
            st.success("No crops with ICDAS ≥ 3 in this visit.")

elif page == "report":
    st.header(t(lang, "report"))
    vid = st.session_state.get("visit_id")
    if not vid:
        st.warning("No active visit.")
    elif st.button("Generate / refresh report (does not re-run detector or ICDAS)"):
        try:
            with st.spinner("Summarizing structured ICDAS JSON…"):
                st.session_state.report = client.post(
                    f"/api/v1/visits/{vid}/report",
                    params={"language": lang},
                )
            st.success("Report ready.")
        except requests.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
            if exc.response is not None and exc.response.status_code == 503:
                st.warning("ICDAS model not yet deployed. Groq will not invent ICDAS grades.")
    report = st.session_state.get("report")
    if report:
        st.markdown(report.get("markdown") or report.get("screening_summary") or "")
        st.caption("Groq did not assign ICDAS grades. Classifier JSON is authoritative.")
        with st.expander("Structured JSON sent to Groq"):
            st.json(report.get("structured"))
        html = report.get("html") or ""
        st.download_button(
            "Download HTML report (print to PDF)",
            data=html_bytes(html),
            file_name=f"visit_{vid}_{lang}.html",
            mime="text/html",
        )

elif page == "history":
    st.header(t(lang, "history"))
    pid = st.text_input("Patient ID", value=(st.session_state.get("patient") or {}).get("public_id", ""))
    if st.button("Load visits") and pid:
        try:
            st.session_state.history = client.get(f"/api/v1/patients/{pid}/history")
        except requests.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
    hist = st.session_state.get("history")
    if hist:
        for v in hist.get("visits") or []:
            with st.expander(f"Visit {v['visit_id']} · {v['visit_date']} · highest ICDAS {v.get('highest_icdas')}"):
                b64_image(v.get("thumbnail_base64"), width=160)
                st.write(
                    f"Teeth analyzed: {v.get('n_teeth_analyzed')} · test only: {v.get('test_only')}"
                )
                st.write(v.get("notes"))
                if st.button("Open", key=f"open{v['visit_id']}"):
                    detail = client.get(f"/api/v1/visits/{v['visit_id']}")
                    st.write(detail.get("patient"))
                    st.write(detail.get("report"))
                    for im in detail.get("images") or []:
                        b64_image(im.get("original_base64"), im.get("filename"), width=320)

elif page == "about":
    st.header(t(lang, "about_icdas"))
    st.write(
        {
            0: "Sound tooth surface",
            1: "First visual change in enamel",
            2: "Distinct visual change in enamel",
            3: "Localized enamel breakdown without visible dentin",
            4: "Underlying dark shadow from dentin",
        }
    )
    st.warning("ICDAS 5 and 6 are out of scope and never displayed.")

elif page == "help":
    st.header(t(lang, "help"))
    st.write(
        "1. Register. 2. Upload clear intraoral photos. 3. Review quality. "
        "4. Analyze. 5. Read ICDAS cards and heatmaps. 6. Download the report. "
        "This is screening support, not a diagnosis."
    )
    st.caption("Backend: uvicorn app.main:app --app-dir app/backend --port 8000")
