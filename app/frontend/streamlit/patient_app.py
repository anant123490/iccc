"""CCC AI Dentist Camera 2.0 — Patient portal. ML runs only on the FastAPI backend."""

from __future__ import annotations

import sys
import importlib
from io import BytesIO
from datetime import date
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import charts as _shared_charts
from shared import components as _shared_components
from shared import theme as _shared_theme

for _shared_module in (_shared_theme, _shared_components, _shared_charts):
    importlib.reload(_shared_module)

from shared.auth import DEFAULT_BACKEND, PortalClient
from shared.charts import icdas_charts
from shared.components import b64_image, icdas_badge, kpi, page_heading, show_disclaimer, sidebar_brand
from shared.language import t
from shared.report import html_bytes
from shared.theme import apply

st.set_page_config(page_title="CCC AI Dentist Camera 2.0", page_icon="🦷", layout="wide")
apply()
st.markdown(
    """<style>
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button[data-testid^="stBaseButton"] {
      background:#fffefa !important; color:#17283b !important;
      border:1px solid #b9ccc3 !important; box-shadow:0 3px 10px rgba(20,40,61,.06) !important;
    }
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] button[data-testid^="stBaseButton"]:hover {
      background:#e7f0ed !important; color:#286967 !important; border-color:#8fb3a8 !important;
    }
    [data-testid="stSidebar"] button * { color:inherit !important; }
    </style>""",
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "page" not in st.session_state:
    st.session_state["page"] = "landing"
RESET_GENERATION_KEY = "patient_reset_generation"
st.session_state.setdefault(RESET_GENERATION_KEY, 0)

lang = st.session_state.lang
sidebar_brand("Patient portal")
backend = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND)
client = PortalClient(backend)

try:
    health = requests.get(f"{backend.rstrip('/')}/api/v1/portal/health", timeout=8).json()
except Exception:
    health = {}
if health:
    det_ok = health.get("detector_v2")
    icd_ok = health.get("icdas_loaded")
    st.sidebar.markdown(
        '<div class="ccc-status">'
        f"Detector · {health.get('detector_status', 'AVAILABLE' if det_ok else 'UNAVAILABLE')}<br>"
        f"ICDAS · {health.get('icdas_status', 'NOT_TRAINED / NOT_DEPLOYED')}"
        "</div>",
        unsafe_allow_html=True,
    )
    if not det_ok:
        st.sidebar.error("The tooth detector is unavailable. Please try again later.")
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
NAV_PENDING_KEY = "patient_nav_pending"
st.session_state.setdefault("nav_radio", "Home")
pending_nav = st.session_state.pop(NAV_PENDING_KEY, None)
if pending_nav in NAV_BY_LABEL:
    st.session_state["nav_radio"] = pending_nav
    st.session_state["page"] = NAV_BY_LABEL[pending_nav]


def _clear_patient_session() -> None:
    """Clear only in-memory visit state; saved server records remain untouched."""
    for key in ("patient", "visit_id", "uploads", "analysis", "report", "history", NAV_PENDING_KEY):
        st.session_state.pop(key, None)
    st.session_state["page"] = "landing"
    st.session_state["nav_radio"] = "Home"
    st.session_state[RESET_GENERATION_KEY] = int(st.session_state.get(RESET_GENERATION_KEY, 0)) + 1


st.sidebar.button(
    "Clear / Reset",
    key="patient_clear_reset",
    help="Clear the current visit from this screen. Saved records are not deleted.",
    use_container_width=True,
    on_click=_clear_patient_session,
)


def _sync_nav() -> None:
    selected_label = st.session_state.get("nav_radio", "Home")
    st.session_state["page"] = NAV_BY_LABEL.get(selected_label, "landing")


st.sidebar.radio("Menu", [label for _, label in NAV], key="nav_radio", on_change=_sync_nav)
st.session_state["page"] = NAV_BY_LABEL.get(
    st.session_state.get("nav_radio", "Home"),
    st.session_state.get("page", "landing"),
)
show_disclaimer(t(lang, "disclaimer"))

if health.get("icdas_demo_mode"):
    st.markdown(
        '<div class="ccc-demo-banner"><strong>Demonstration mode</strong> · '
        'Predictions use an inactive research candidate. They are not production-active '
        'and must not be used for clinical decisions.</div>',
        unsafe_allow_html=True,
    )

page = st.session_state.page
reset_generation = int(st.session_state.get(RESET_GENERATION_KEY, 0))

def _friendly_error(exc: BaseException) -> str:
    if isinstance(exc, (requests.ConnectionError, requests.ConnectTimeout)):
        return "The analysis service is unavailable. Please try again in a moment."
    if isinstance(exc, requests.Timeout):
        return "This request took too long. Please try again."
    response = exc.response if isinstance(exc, requests.HTTPError) else None
    status = response.status_code if response is not None else 0
    if status == 400:
        return "This photo could not be processed. Check that it is a clear JPG or PNG image."
    if status == 401:
        return "Your session needs to be refreshed. Please sign in again."
    if status == 404:
        return "The selected visit could not be found. Please start a new visit."
    if status == 413:
        return "This photo is too large. Choose an image under 10 MB."
    if status == 503:
        return "Tooth detection or ICDAS classification is temporarily unavailable. Please try later."
    return "Something went wrong. No new result was shown; please try again."

if page == "landing":
    st.markdown(
        f'<div class="ccc-hero"><div class="ccc-eyebrow">A MORE THOUGHTFUL DENTAL SCREENING EXPERIENCE</div>'
        f'<h1>{t(lang, "brand")}</h1><p>{t(lang, "subtitle")}</p>'
        '<div class="ccc-hero-meta"><span class="ccc-pill ccc-pill-teal">ICDAS grades 0–4</span>'
        '<span class="ccc-pill">AI-assisted screening</span>'
        '<span class="ccc-pill">Dentist review recommended</span></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1])
    if c1.button(t(lang, "start"), type="primary", use_container_width=True):
        st.session_state[NAV_PENDING_KEY] = t(lang, "register")
        st.rerun()
    if c2.button(t(lang, "previous"), use_container_width=True):
        st.session_state[NAV_PENDING_KEY] = t(lang, "history")
        st.rerun()
    st.markdown("### A clear, guided process")
    st.caption("Your photograph is checked and organized into a simple screening summary for discussion with your dentist.")
    how_cols = st.columns(3)
    steps = [
        ("01", "Add a photo", "Choose a clear JPG or PNG image of the teeth."),
        ("02", "Review the analysis", "See detected tooth crops, model grades, and attention heatmaps."),
        ("03", "Discuss next steps", "Use the report as a conversation aid with a licensed dentist."),
    ]
    for col, (number, title, detail) in zip(how_cols, steps):
        with col, st.container(border=True):
            st.markdown(f'<div class="ccc-step-number">{number}</div><h3>{title}</h3><p>{detail}</p>', unsafe_allow_html=True)
    st.caption("Languages: English, हिन्दी, ಕನ್ನಡ. ICDAS numbers stay 0–4.")

elif page == "register":
    page_heading(t(lang, "register"), "Create a visit record before adding dental photographs.", "Patient intake")
    with st.form("reg"):
        pid = st.text_input("Patient ID (optional, auto if blank)", key=f"patient_form_id_{reset_generation}")
        name = st.text_input("Name", key=f"patient_form_name_{reset_generation}")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, key=f"patient_form_age_{reset_generation}")
        gender = st.selectbox("Gender", ["", "Female", "Male", "Other", "Prefer not to say"], key=f"patient_form_gender_{reset_generation}")
        phone = st.text_input("Phone (optional)", key=f"patient_form_phone_{reset_generation}")
        notes = st.text_area("Visit notes", key=f"patient_form_notes_{reset_generation}")
        visit_date = st.date_input("Visit date", value=date.today(), key=f"patient_form_date_{reset_generation}")
        submitted = st.form_submit_button("Save and continue", type="primary", use_container_width=True)
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
                st.session_state[NAV_PENDING_KEY] = t(lang, "upload")
                st.rerun()
            except requests.HTTPError as exc:
                st.error(_friendly_error(exc))
            except requests.RequestException as exc:
                st.error(_friendly_error(exc))

elif page == "upload":
    page_heading(t(lang, "upload"), "Add clear JPG or PNG dental photographs. Review each image before sending it for screening.", "Secure photo upload")
    if "visit_id" not in st.session_state:
        st.warning("Register a patient first.")
    else:
        files = st.file_uploader(
            "JPG / JPEG / PNG — multiple intraoral photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"patient_upload_files_{reset_generation}",
        )
        cam = st.camera_input("Camera (if available)", key=f"patient_camera_photo_{reset_generation}")
        blobs = list(files or [])
        if cam is not None:
            blobs.append(cam)
        valid_blobs = []
        if blobs:
            st.markdown("### Review photos before upload")
            for idx, uploaded in enumerate(blobs):
                raw = uploaded.getvalue()
                if len(raw) > 10 * 1024 * 1024:
                    st.error(f"{getattr(uploaded, 'name', 'Camera photo')} is over the 10 MB limit.")
                    continue
                try:
                    with Image.open(BytesIO(raw)) as image:
                        image.verify()
                    with Image.open(BytesIO(raw)) as image:
                        st.image(image.convert("RGB"), caption=f"{uploaded.name} · {len(raw) / 1024:.0f} KB", width=260)
                    valid_blobs.append(uploaded)
                except (UnidentifiedImageError, OSError, ValueError):
                    st.error(f"{getattr(uploaded, 'name', 'Photo')} is not a readable image. Choose a JPG or PNG.")
        if st.button("Upload photos", type="primary", disabled=not valid_blobs) and valid_blobs:
            body = [
                ("files", (getattr(f, "name", "camera.jpg"), f.getvalue(), "image/png" if Path(getattr(f, "name", "")).suffix.lower() == ".png" else "image/jpeg"))
                for f in valid_blobs
            ]
            try:
                with st.spinner("Uploading and checking quality…"):
                    out = requests.post(
                        f"{backend.rstrip('/')}/api/v1/visits/{st.session_state.visit_id}/images",
                        files=body,
                        timeout=120,
                    )
                    out.raise_for_status()
                    st.session_state.uploads = out.json()
                st.success("Your photos are ready for analysis.")
            except requests.HTTPError as exc:
                st.error(_friendly_error(exc))
            except requests.RequestException as exc:
                st.error(_friendly_error(exc))
        uploads = st.session_state.get("uploads") or {}
        for im in uploads.get("images") or []:
            q = im.get("quality") or {}
            verdict = q.get("verdict", "?")
            quality_class = {"PASS": "pass", "WARNING": "review", "FAIL": "retry"}.get(verdict, "review")
            quality_label = {"PASS": "Ready for analysis", "WARNING": "Review suggested", "FAIL": "Choose a clearer photo"}.get(verdict, "Photo quality status")
            st.markdown(
                f'<div class="ccc-quality ccc-quality-{quality_class}"><strong>{quality_label}</strong>'
                f'<span>Image quality · {verdict}</span></div>',
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
            st.session_state[NAV_PENDING_KEY] = t(lang, "detect")
            st.rerun()

elif page == "detect":
    page_heading(t(lang, "detect"), "Tooth Detector V2 identifies visible teeth and prepares individual crops for classification.", "Photo analysis")
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
                st.success("Photo analysis finished.")
            except requests.HTTPError as exc:
                st.error(_friendly_error(exc))
            except requests.RequestException as exc:
                st.error(_friendly_error(exc))
        analysis = st.session_state.get("analysis") or {}
        if analysis.get("prediction_mode") == "DEMO":
            st.markdown('<div class="ccc-demo-banner"><strong>Demonstration result</strong> · Have a qualified dental professional review this AI output.</div>', unsafe_allow_html=True)
        if analysis.get("icdas_status") == "NOT_TRAINED / NOT_DEPLOYED" or analysis.get("message"):
            st.warning(analysis.get("message") or "ICDAS model not yet deployed.")
        if analysis and not (analysis.get("detections") or []):
            st.info("No tooth crops were detected in this photo. Try a clearer, well-lit intraoral photo.")
        for det in analysis.get("detections") or []:
            with st.container(border=True):
                st.markdown(f"### Photo analysis · {det.get('teeth_detected')} teeth found")
                a, b = st.columns(2)
                with a:
                    b64_image(det.get("original_base64"), "Uploaded photo", width=520)
                with b:
                    b64_image(det.get("overlay_base64"), "Detected tooth areas", width=520)
                st.caption(f"Mean tooth-detection confidence · {float(det.get('mean_confidence') or 0) * 100:.0f}%")
                with st.expander("Detection details"):
                    for crop in det.get("crops") or []:
                        st.write(f"Tooth {int(crop.get('index') or 0) + 1} · {float(crop.get('confidence') or 0) * 100:.0f}% detection confidence")
        st.markdown("### Individual tooth crops")
        crops = []
        for det in analysis.get("detections") or []:
            crops.extend(det.get("crops") or [])
        n = max(1, min(4, len(crops) or 1))
        for i in range(0, len(crops), n):
            row = crops[i : i + n]
            cols = st.columns(len(row))
            for col, crop in zip(cols, row):
                with col:
                    with st.container(border=True):
                        b64_image(crop.get("crop_base64"), width=220)
                        st.caption(f"Tooth {int(crop.get('index') or 0) + 1} · {float(crop.get('confidence') or 0) * 100:.0f}% detected")

elif page == "icdas":
    page_heading(t(lang, "icdas"), "Each grade is a model estimate for one detected crop. Review the crop, attention heatmap, and confidence together.", "Tooth-level results")
    analysis = st.session_state.get("analysis") or {}
    teeth = analysis.get("teeth") or []
    if analysis.get("icdas_status") == "NOT_TRAINED / NOT_DEPLOYED":
        st.warning("ICDAS model not yet deployed. No ICDAS grades, heatmaps, or Groq tooth findings were generated.")
    if not teeth:
        st.info("Run photo analysis first. Tooth-level grades appear here when classification is available.")
    for index, tooth in enumerate(teeth):
        grade = int(tooth["icdas_grade"])
        with st.container(border=True):
            head, score = st.columns([3, 1])
            with head:
                st.markdown(f'{icdas_badge(grade)} <span class="ccc-result-stage">{tooth.get("current_stage", "")}</span>', unsafe_allow_html=True)
                st.caption(f"Tooth {int(tooth.get('index') or index) + 1} · {tooth.get('priority', 'Discuss with your dentist')} priority")
            with score:
                confidence = max(0.0, min(100.0, float(tooth.get("confidence") or 0)))
                st.markdown(f'<div class="ccc-confidence-label">Model confidence</div><div class="ccc-confidence-value">{confidence:.1f}%</div>', unsafe_allow_html=True)
                st.progress(confidence / 100)
            original, heatmap = st.columns(2)
            with original:
                st.markdown('<div class="ccc-image-label">TOOTH CROP</div>', unsafe_allow_html=True)
                b64_image(tooth.get("crop_base64"), width=440)
            with heatmap:
                st.markdown('<div class="ccc-image-label">MODEL ATTENTION · GRAD-CAM</div>', unsafe_allow_html=True)
                b64_image(tooth.get("heatmap_base64"), width=440)
                st.caption(t(lang, "heatmap_note"))
            with st.expander("Grade explanation and confidence by class"):
                st.write(tooth.get("explanation") or "")
                st.write("Suggested next step:", tooth.get("next_step") or "")
                probabilities = tooth.get("probabilities") or {}
                prob_cols = st.columns(5)
                for class_id, col in enumerate(prob_cols):
                    value = float(probabilities.get(str(class_id), 0) or 0)
                    with col:
                        st.caption(f"Grade {class_id} · {value * 100:.1f}%")
                        st.progress(max(0.0, min(1.0, value)))
                b64_image(tooth.get("overlay_base64"), "Detection and attention overlay", width=360)
            if grade >= 3:
                st.caption("Higher grade estimate · Discuss this result with a dental professional.")

elif page == "dashboard":
    page_heading(t(lang, "dashboard"), "A concise overview of this visit and its tooth-level screening results.", "Visit overview")
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
            st.warning(f"{len(high)} crop(s) have a model-estimated ICDAS grade of 3 or 4. Review with a dentist.")
        else:
            st.success("No crops with ICDAS ≥ 3 in this visit.")

elif page == "report":
    page_heading(t(lang, "report"), "A plain-language summary based on the saved classifier results.", "Screening report")
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
        except requests.RequestException as exc:
            st.error(_friendly_error(exc))
            if exc.response is not None and exc.response.status_code == 503:
                st.warning("ICDAS model not yet deployed. Groq will not invent ICDAS grades.")
    report = st.session_state.get("report")
    if report:
        if report.get("prediction_mode") == "DEMO":
            st.warning("This report summarizes DEMO predictions from an inactive research candidate.")
        st.caption(f"Report text: {report.get('report_provider', 'Local fallback')}")
        with st.container(border=True):
            st.markdown(report.get("markdown") or report.get("screening_summary") or "")
        st.caption("Groq did not assign ICDAS grades. Classifier JSON is authoritative.")
        html = report.get("html") or ""
        st.download_button(
            "Download HTML report (print to PDF)",
            data=html_bytes(html),
            file_name=f"visit_{vid}_{lang}.html",
            mime="text/html",
        )

elif page == "history":
    page_heading(t(lang, "history"), "Review saved visits and reports using the patient record ID.", "Saved visits")
    pid = st.text_input(
        "Patient ID",
        value=(st.session_state.get("patient") or {}).get("public_id", ""),
        key=f"patient_history_id_{reset_generation}",
    )
    if st.button("Load visits") and pid:
        try:
            st.session_state.history = client.get(f"/api/v1/patients/{pid}/history")
        except requests.RequestException as exc:
            st.error(_friendly_error(exc))
    hist = st.session_state.get("history")
    if hist:
        for v in hist.get("visits") or []:
            with st.expander(f"Visit · {v['visit_date']} · highest ICDAS {v.get('highest_icdas')}"):
                summary_cols = st.columns([1, 2])
                with summary_cols[0]:
                    b64_image(v.get("thumbnail_base64"), width=240)
                with summary_cols[1]:
                    st.metric("Teeth analyzed", v.get("n_teeth_analyzed", 0))
                if st.button("Open", key=f"open{v['visit_id']}"):
                    try:
                        detail = client.get(f"/api/v1/visits/{v['visit_id']}")
                        report_detail = detail.get("report") or {}
                        st.markdown(report_detail.get("screening_summary") or "No saved report for this visit.")
                        for im in detail.get("images") or []:
                            b64_image(im.get("original_base64"), im.get("filename"), width=320)
                    except requests.RequestException as exc:
                        st.error(_friendly_error(exc))

elif page == "about":
    page_heading(t(lang, "about_icdas"), "A short guide to the five grades supported by this research prototype.", "About ICDAS")
    st.table(
        [
            {"Grade": 0, "Description": "Sound tooth surface"},
            {"Grade": 1, "Description": "First visual change in enamel"},
            {"Grade": 2, "Description": "Distinct visual change in enamel"},
            {"Grade": 3, "Description": "Localized enamel breakdown without visible dentin"},
            {"Grade": 4, "Description": "Underlying dark shadow from dentin"},
        ]
    )
    st.warning("ICDAS 5 and 6 are out of scope and never displayed.")

elif page == "help":
    page_heading(t(lang, "help"), "Use this guide to move through a screening without losing saved results.", "Help and guidance")
    st.write(
        "1. Register. 2. Upload clear intraoral photos. 3. Review quality. "
        "4. Analyze. 5. Read ICDAS cards and heatmaps. 6. Download the report. "
        "This is screening support, not a diagnosis."
    )
    st.caption("If the screening service is unavailable, ask your administrator to restart the service and try again.")
