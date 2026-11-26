"""CCC AI Dentist Camera 2.0 — Admin portal. Requires backend admin login."""

from __future__ import annotations

import sys
import hashlib
import importlib
import re
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import admin_workflow as _shared_workflow
from shared import charts as _shared_charts
from shared import components as _shared_components
from shared import theme as _shared_theme

for _shared_module in (_shared_theme, _shared_components, _shared_charts, _shared_workflow):
    importlib.reload(_shared_module)

from shared.admin_reset import RESET_CONFIRMATION_TEXT, reset_confirmation_ok
from shared.admin_workflow import NAV_GEN_KEY, NAV_PAGE_KEY, apply_pending_nav, render_stepper, set_admin_nav
from shared.auth import DEFAULT_BACKEND, PortalClient
from shared.charts import icdas_charts
from shared.components import b64_image, kpi, page_heading, show_disclaimer, sidebar_brand
from shared.theme import apply
from shared.upload_state import clear_admin_training_ui_state, next_training_uploader_nonce

st.set_page_config(page_title="CCC Admin — Dentist Camera 2.0", page_icon="🛠️", layout="wide")
apply()

DISCLAIMER = (
    "Admin tools for dentists and dataset work. ICDAS labels are 0–4 only. "
    "Patient images and training images are stored separately. AI is not a diagnosis."
)

sidebar_brand("Administrator workspace")
backend = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND)
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None

show_disclaimer(DISCLAIMER)

if not st.session_state.admin_token:
    page_heading("Admin sign in", "Access patient records, photo review, labeling, dataset preparation, and model controls.", "Secure administrator access")
    pw = st.text_input("Password", type="password", help="Enter your administrator password.")
    if st.button("Sign in"):
        try:
            r = requests.post(f"{backend.rstrip('/')}/api/v1/admin/login", json={"password": pw}, timeout=30)
            r.raise_for_status()
            st.session_state.admin_token = r.json()["token"]
            st.rerun()
        except requests.ConnectionError:
            st.error("Backend unavailable. Make sure FastAPI is running on port 8000.")
        except requests.Timeout:
            st.error("The backend took too long to respond.")
        except requests.HTTPError:
            st.error("Login failed.")
    st.stop()

client = PortalClient(backend, token=st.session_state.admin_token)
if st.sidebar.button("Sign out"):
    st.session_state.admin_token = None
    st.rerun()

st.sidebar.markdown("### System status")
try:
    runtime_status = requests.get(f"{backend.rstrip('/')}/api/v1/portal/health", timeout=8).json()
    st.sidebar.markdown(
        '<div class="ccc-status">'
        f"Detector · {runtime_status.get('detector_status', 'unknown')}<br>"
        f"ICDAS · {runtime_status.get('icdas_status', 'unknown')}"
        "</div>",
        unsafe_allow_html=True,
    )
    if runtime_status.get("icdas_demo_mode"):
        st.sidebar.warning("ICDAS model · DEMO ONLY")
    else:
        st.sidebar.caption(f"ICDAS model · {runtime_status.get('icdas_status', 'unknown')}")
except requests.RequestException:
    st.sidebar.error("Backend unavailable")

NAV_PAGES = [
    "Dashboard",
    "Patients",
    "Training upload",
    "Box review",
    "Labeling",
    "Dataset",
    "Training",
    "Model registry",
    "Evaluation",
    "Real-world test",
]
page = apply_pending_nav()
_nav_gen = int(st.session_state.get(NAV_GEN_KEY) or 0)
_idx = NAV_PAGES.index(page) if page in NAV_PAGES else 0
choice = st.sidebar.radio("Admin", NAV_PAGES, index=_idx, key=f"admin_nav_{_nav_gen}")
if choice != page:
    st.session_state[NAV_PAGE_KEY] = choice
    page = choice


def _progress_card(inv: dict) -> None:
    photos = inv.get("photos") or {}
    crops = inv.get("crops") or {}
    labels = inv.get("icdas_labels") or inv.get("class_counts") or {}
    st.markdown("### Current training dataset progress")
    a, b, c = st.columns(3)
    with a:
        st.metric("Unique source photos", photos.get("unique", 0))
        st.caption(
            f"{photos.get('completed', 0)} completed · "
            f"{photos.get('remaining', 0)} still processing"
        )
    with b:
        st.metric("Verified tooth crops", crops.get("verified", 0))
        st.caption(
            f"{crops.get('labeled', 0)} labeled · {crops.get('unlabeled', 0)} remaining · "
            f"{crops.get('eligible', crops.get('labeled', 0))} eligible"
        )
    with c:
        st.metric("Labeled crops", crops.get("labeled", 0))
        grade_items = "".join(
            f'<span class="ccc-grade-count">{g}<b>{int(labels.get(str(g), 0) or 0)}</b></span>'
            for g in range(5)
        )
        st.markdown(f'<div class="ccc-grade-strip">{grade_items}</div>', unsafe_allow_html=True)
        st.caption("Saved ICDAS grade distribution")
    if inv.get("imbalance_message"):
        st.warning(_safe_notice(inv["imbalance_message"], "The ICDAS grade distribution is imbalanced."))


def _dentist_error_message(exc: BaseException) -> str:
    if isinstance(exc, (requests.ConnectionError, requests.ConnectTimeout)):
        return "Backend unavailable. Make sure FastAPI is running on port 8000."
    if isinstance(exc, requests.Timeout):
        return "The backend took too long to respond."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status == 401:
            return "Your admin session expired. Sign in again."
        if status == 404:
            return "The requested item could not be found. Refresh and try again."
        if status == 409:
            return "The requested change conflicts with current data. Refresh and review the item."
        if status == 400:
            return "The request could not be completed. Review the selected values and try again."
        if status >= 500:
            return "The backend could not complete this request. Try again or contact support."
    return "Backend unavailable. Make sure FastAPI is running on port 8000."


def _show_backend_error(exc: BaseException) -> None:
    st.error(_dentist_error_message(exc))
    if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 401:
        st.session_state.admin_token = None


def _friendly_upload_issue(value) -> str:
    """Turn backend processing details into an operator-safe, useful status."""
    if not value:
        return ""
    message = str(value).casefold()
    if "size" in message or "large" in message:
        return "Photo exceeds the upload size limit."
    if "format" in message or "extension" in message or "unsupported" in message:
        return "Photo format is not supported. Use JPG or PNG."
    if "image" in message or "decode" in message or "read" in message:
        return "Photo could not be read. Choose a clear JPG or PNG and retry."
    return "Detection did not finish. Review the photo and retry processing."


def _safe_notice(value, fallback: str = "A system issue needs review. Try again or contact your administrator.") -> str:
    text = str(value or "").strip()
    lowered = text.casefold()
    if any(token in lowered for token in ("traceback", "filenotfounderror", "stack trace", "site-packages", "line ")):
        return fallback
    path_pattern = r"(?i)(?:[A-Z]:\\[^\r\n,;]+|\\\\[^\s,;]+|\b(?:models|data|reports|storage|app)[\\/][^\s,;]+)"
    text = re.sub(path_pattern, "a local file", text)
    return text[:320] or fallback


def _upload_rejection_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "Filename": str(item.get("filename") or item.get("name") or f"Photo {index + 1}"),
            "Issue": _friendly_upload_issue(item.get("reason") or item.get("error") or item.get("message"))
            or "Photo was not added. Check the file and try again.",
        }
        for index, raw_item in enumerate(items)
        for item in [raw_item if isinstance(raw_item, dict) else {"error": raw_item}]
    ]


def _grade_distribution_text(counts: dict | None) -> str:
    counts = counts or {}
    return " · ".join(f"{grade}: {int(counts.get(str(grade), 0) or 0)}" for grade in range(5))


def _split_summary_rows(summary: dict | None) -> list[dict]:
    rows = []
    for split, details in (summary or {}).items():
        rows.append(
            {
                "Split": "Validation" if str(split).lower() in {"val", "valid", "validation"} else str(split).title(),
                "Patients": details.get("patient_count", 0),
                "Original photos": details.get("image_count", details.get("photo_count", 0)),
                "Tooth crops": details.get("tooth_crop_count", details.get("crop_count", 0)),
                "Labeled crops": details.get("labeled_crop_count", 0),
                "Verified crops": details.get("verified_crop_count", 0),
                **{
                    f"Grade {grade}": (details.get("icdas_grade_distribution") or details.get("grade_counts") or {}).get(str(grade), 0)
                    for grade in range(5)
                },
                "Actual patients %": details.get("actual_patient_percent", 0),
                "Target patients %": details.get("target_patient_percent", 0),
                "Deviation (pp)": details.get("deviation_percentage_points", 0),
            }
        )
    return rows


def _audit_display_rows(events: list[dict]) -> list[dict]:
    return [
        {
            "When": str(event.get("created_at") or "")[:16].replace("T", " "),
            "Action": str(event.get("event_type") or "Activity").replace("_", " ").title(),
            "Record type": str(event.get("entity_type") or "").replace("_", " ").title(),
            "Record ID": str(event.get("entity_id") or "")[:14],
            "Actor": event.get("actor") or "Admin",
        }
        for event in events
        if isinstance(event, dict)
    ]


def _label_history_display_rows(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        if not isinstance(event, dict):
            continue
        details = event.get("details") or {}
        old = details.get("old") or {}
        new = details.get("new") or {}
        previous = old.get("grade")
        updated = new.get("grade")
        rows.append(
            {
                "When": str(event.get("created_at") or "")[:16].replace("T", " "),
                "Action": str(event.get("event_type") or "Label update").replace("_", " ").title(),
                "Previous grade": previous if previous is not None else "—",
                "Saved grade": updated if updated is not None else "—",
                "Status": str(new.get("annotation_status") or old.get("annotation_status") or "Saved").replace("_", " ").title(),
                "Actor": event.get("actor") or "Admin",
            }
        )
    return rows


def _dataset_version_rows(versions: list[dict]) -> list[dict]:
    return [
        {
            "Dataset version": row.get("name") or "—",
            "Status": row.get("status") or "—",
            "Train patients": row.get("n_train", 0),
            "Validation patients": row.get("n_valid", 0),
            "Test patients": row.get("n_test", 0),
            "Created": str(row.get("created_at") or "")[:16].replace("T", " "),
        }
        for row in versions
        if isinstance(row, dict)
    ]


def _workflow_nav(page_key: str, back: str | None, nxt: str | None, nxt_label: str) -> None:
    cols = st.columns(2)
    if back:
        with cols[0]:
            st.button(
                "BACK",
                key=f"nav_back_{page_key}",
                on_click=set_admin_nav,
                args=(back,),
            )
    if nxt:
        with cols[1]:
            st.button(
                nxt_label,
                type="primary",
                key=f"nav_next_{page_key}",
                on_click=set_admin_nav,
                args=(nxt,),
            )


def _show_reset_summary(out: dict) -> None:
    st.success("Training workflow data was cleared. Training was not started.")
    a, b, c, d = st.columns(4)
    a.metric("uploads", out.get("uploads", 0))
    b.metric("duplicates", out.get("duplicates", 0))
    c.metric("crops", out.get("crops", 0))
    d.metric("labeled", out.get("labeled", 0))
    e, f, g = st.columns(3)
    e.metric("unlabeled", out.get("unlabeled", 0))
    f.metric("dataset", out.get("dataset", "NOT READY"))
    g.metric("training", out.get("training", "DISABLED"))
    if out.get("recovery_archive"):
        st.info("A recovery archive was created.")
    if out.get("audit_event_id"):
        st.caption(f"Reset audit event: {out['audit_event_id']}")


def _dataset_management() -> None:
    if not st.session_state.get("admin_reset_controls_open"):
        st.markdown("### Advanced data controls")
        st.caption("Reset actions permanently remove selected training images, crops, labels, and dataset versions. Patient records and model weights are preserved.")
        if st.button("Review training data reset options", key="open_admin_reset_controls"):
            st.session_state.admin_reset_controls_open = True
            st.rerun()
        return
    if st.button("Hide reset options", key="hide_admin_reset_controls"):
        st.session_state.admin_reset_controls_open = False
        st.rerun()
    st.markdown("---")
    st.subheader("Training data reset")
    if st.session_state.get("reset_summary"):
        _show_reset_summary(st.session_state.reset_summary)
    st.warning(
        "Advanced destructive controls. Clearing training data permanently deletes Admin training photographs, crops, "
        "ICDAS labels, and built dataset versions. Model weights, source code, and patient records are preserved."
    )
    plan = {}
    full_plan = {}
    try:
        plan = client.get("/api/v1/admin/training/reset/plan", params={"scope": "dataset"})
        full_plan = client.get("/api/v1/admin/training/reset/plan", params={"scope": "full"})
        with st.expander("Reset plan (what will be deleted)"):
            st.markdown("**Clear Training Dataset**")
            st.table([
                {"Record type": str(key).replace("_", " ").title(), "Rows / files": value}
                for key, value in (plan.get("affected_counts") or {}).items()
            ])
            st.markdown("**Full Training Project Reset**")
            st.table([
                {"Record type": str(key).replace("_", " ").title(), "Rows / files": value}
                for key, value in (full_plan.get("affected_counts") or {}).items()
            ])
            st.write("A recovery archive will be created before removal.")
            st.caption("Preserved: " + ", ".join((plan.get("will_not_delete") or [])[:8]) + " …")
    except requests.RequestException as exc:
        _show_backend_error(exc)
    running_jobs = int((full_plan.get("affected_counts") or {}).get("running_training_jobs") or 0)
    if running_jobs:
        st.warning("Reset is unavailable while an ICDAS training job is running. Wait for it to finish.")

    phrase = RESET_CONFIRMATION_TEXT
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clear Training Dataset**")
        st.caption("Uploads, duplicates, boxes/crops, labels, and dataset versions.")
        ok_ds = st.checkbox(
            "I understand that this permanently deletes the selected training data.",
            key="reset_ds_check",
        )
        typed_ds = st.text_input("Type the confirmation sentence", key="reset_ds_text")
        if st.button(
            "Clear Training Dataset",
            type="primary",
            disabled=running_jobs > 0 or not reset_confirmation_ok(ok_ds, typed_ds),
            key="reset_ds_btn",
        ):
            try:
                out = client.post(
                    "/api/v1/admin/training/reset",
                    json={
                        "scope": "dataset",
                        "confirm": True,
                        "confirmation_text": phrase,
                    },
                )
                clear_admin_training_ui_state(st.session_state)
                st.session_state.reset_summary = out
                st.rerun()
            except requests.RequestException as exc:
                _show_backend_error(exc)
    with c2:
        st.markdown("**Full Training Project Reset**")
        st.caption("Same as dataset clear, plus Admin training job records. Weights stay.")
        ok_full = st.checkbox(
            "I understand that this permanently deletes the selected training data.",
            key="reset_full_check",
        )
        typed_full = st.text_input("Type the confirmation sentence", key="reset_full_text")
        if st.button(
            "Full Training Project Reset",
            type="primary",
            disabled=running_jobs > 0 or not reset_confirmation_ok(ok_full, typed_full),
            key="reset_full_btn",
        ):
            try:
                out = client.post(
                    "/api/v1/admin/training/reset",
                    json={
                        "scope": "full",
                        "confirm": True,
                        "confirmation_text": phrase,
                    },
                )
                clear_admin_training_ui_state(st.session_state)
                st.session_state.reset_summary = out
                st.rerun()
            except requests.RequestException as exc:
                _show_backend_error(exc)
    st.caption(f'Type exactly: "{phrase}"')

try:
    if page == "Dashboard":
        page_heading("Admin dashboard", "Live overview of patients, uploaded photographs, labeled crops, and dataset readiness.", "Clinical operations")
        kpis = client.get("/api/v1/admin/kpis")
        cols = st.columns(4)
        keys = [
            ("patients", "Patients"),
            ("training_images", "Training photos"),
            ("labeled_crops", "Labeled crops"),
            ("dataset_versions", "Dataset versions"),
        ]
        for i, (k, label) in enumerate(keys):
            with cols[i % 4]:
                kpi(label, kpis.get(k))
        st.caption(f"Latest model: {kpis.get('latest_model')}")
        inv = kpis.get("training_progress") or client.get("/api/v1/admin/dataset")
        render_stepper("Dashboard", inv)
        _progress_card(inv)
        grade_counts = inv.get("icdas_labels") or inv.get("class_counts") or {}
        icdas_charts(grade_counts)
        st.button(
            "Go to training workflow",
            type="primary",
            key="nav_dash_workflow",
            on_click=set_admin_nav,
            args=("Training upload",),
        )
        _dataset_management()

    elif page == "Patients":
        page_heading("Patient management", "Find registered patients and review their saved visit summaries.", "Patient records")
        q = st.text_input("Search name or Patient ID")
        rows = client.get("/api/v1/admin/patients", params={"q": q} if q else None)
        for p in rows.get("patients") or []:
            with st.expander(f"{p['public_id']} · {p['name']}"):
                patient_cols = st.columns(3)
                patient_cols[0].markdown(f"**Patient ID**  \n{p.get('public_id') or '—'}")
                patient_cols[1].markdown(f"**Age**  \n{p.get('age') if p.get('age') is not None else 'Not provided'}")
                patient_cols[2].markdown(f"**Created**  \n{str(p.get('created_at') or '—')[:10]}")
                hist = client.get(f"/api/v1/patients/{p['public_id']}/history")
                for v in hist.get("visits") or []:
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(
                        f"**Visit {v.get('visit_id')}** · {v.get('visit_date') or 'Date unavailable'}  \n"
                        f"{int(v.get('n_teeth_analyzed') or 0)} tooth crops analyzed · "
                        f"Highest saved grade: {v.get('highest_icdas') if v.get('highest_icdas') is not None else '—'}"
                    )
                    if c2.button("Soft-delete visit", key=f"del{v['visit_id']}"):
                        client.delete(f"/api/v1/visits/{v['visit_id']}")
                        st.success("Visit marked deleted.")
                        st.rerun()
        with st.form("newp"):
            st.subheader("Create patient")
            name = st.text_input("Name")
            age = st.number_input("Age", 0, 120, 30)
            if st.form_submit_button("Create") and name:
                client.post("/api/v1/admin/patients", json={"name": name, "age": int(age)})
                st.success("Created.")

    elif page == "Training upload":
        page_heading("Bulk photo upload", "Add source photographs, track detector processing, and group photos under the correct patient.", "Training photo library")
        st.info(
            "Originals are kept in the training photo store and SHA-256 duplicates are flagged. "
            "Detector V2 runs in the background for unique photos. You can keep uploads Unknown until detection "
            "finishes, then explicitly group source photos under an anonymous Patient ID below."
        )

        with st.expander("Register a new patient before upload"):
            with st.form("training_upload_register_patient"):
                new_patient_name = st.text_input("Patient name")
                new_patient_age = st.number_input("Age", min_value=0, max_value=120, value=0)
                new_patient_gender = st.selectbox("Gender (optional)", ["", "F", "M", "Other"])
                register_submit = st.form_submit_button("Register patient")
            if register_submit:
                if not new_patient_name.strip():
                    st.error("Enter a patient name to register the record.")
                else:
                    try:
                        created = client.post(
                            "/api/v1/admin/patients",
                            json={
                                "name": new_patient_name.strip(),
                                "age": int(new_patient_age) if new_patient_age else None,
                                "gender": new_patient_gender or None,
                            },
                        )
                        registered_id = (created.get("patient") or {}).get("public_id")
                        if not registered_id:
                            raise ValueError("The patient record was not returned by the backend.")
                        st.session_state["training_upload_patient_id"] = registered_id
                        st.success(f"Patient registered. Assigned database ID: {registered_id}")
                        st.rerun()
                    except (requests.RequestException, ValueError) as exc:
                        _show_backend_error(exc)

        patient_search = st.text_input("Find a registered patient", key="training_upload_patient_search")
        patient_query = {"q": patient_search} if patient_search.strip() else None
        patient_records = client.get(
            "/api/v1/admin/patients",
            params={**(patient_query or {}), "limit": 1000},
        ).get("patients") or []
        patient_by_id = {str(row["public_id"]): row for row in patient_records if row.get("public_id")}
        patient_ids = list(patient_by_id)
        preferred_id = st.session_state.get("training_upload_patient_id")
        if preferred_id and preferred_id not in patient_ids:
            preferred = client.get(
                "/api/v1/admin/patients",
                params={"q": preferred_id, "limit": 1000},
            ).get("patients") or []
            if any(str(row.get("public_id")) == str(preferred_id) for row in preferred):
                patient_by_id[str(preferred_id)] = next(row for row in preferred if str(row.get("public_id")) == str(preferred_id))
                patient_ids.insert(0, str(preferred_id))

        assignment_mode = st.radio(
            "Patient linking",
            [
                "Leave Unassigned and group after detection",
                "One registered patient for every photo",
                "CSV patient mapping per photo",
            ],
            horizontal=True,
            key="training_upload_link_mode",
        )
        selected_patient_id = None
        if assignment_mode == "One registered patient for every photo":
            if patient_ids:
                patient_options = [""] + patient_ids
                default_patient = patient_options.index(preferred_id) if preferred_id in patient_options else 0
                selected_patient_id = st.selectbox(
                    "Patient for this batch",
                    patient_options,
                    index=default_patient,
                    format_func=lambda value: "Select a patient" if not value else (
                        f"{value} · {patient_by_id.get(value, {}).get('name', '')}"
                    ),
                    key="training_upload_selected_patient",
                )
            else:
                st.warning("No registered patients matched. Register a patient or search again before uploading.")

        if "training_uploader_nonce" not in st.session_state:
            st.session_state.training_uploader_nonce = 0
        files = st.file_uploader(
            "Select up to 250 JPG/PNG photographs (maximum 12 MB each, 512 MB per batch)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"training_files_{st.session_state.training_uploader_nonce}",
        )
        manifest_file = None
        if assignment_mode == "CSV patient mapping per photo":
            manifest_file = st.file_uploader(
                "CSV mapping: filename, patient_id, image_id (image_id optional)",
                type=["csv"],
                key=f"training_manifest_{st.session_state.training_uploader_nonce}",
                help="Any nonblank patient_id must be a registered database Patient ID. Blank patient_id values stay Unknown / Unassigned. The CSV must cover every selected filename exactly once.",
            )
        pending = list(files or [])
        if pending:
            st.caption(f"{len(pending)} file(s) selected for this batch (not uploaded yet).")
        can_upload = bool(pending) and len(pending) <= 250 and (
            True
            if assignment_mode == "Leave Unassigned and group after detection"
            else bool(selected_patient_id)
            if assignment_mode == "One registered patient for every photo"
            else manifest_file is not None
        )
        if st.button("Upload photos and queue detector", type="primary", disabled=not can_upload):
            body = [
                (
                    "files",
                    (
                        f.name,
                        f.getvalue(),
                        "image/png" if f.name.lower().endswith(".png") else "image/jpeg",
                    ),
                )
                for f in pending
            ]
            if manifest_file is not None:
                body.append(("patient_manifest", (manifest_file.name, manifest_file.getvalue(), "text/csv")))
            form_data = {"patient_id": selected_patient_id} if selected_patient_id else None
            try:
                with st.spinner(f"Uploading and saving {len(pending)} photo(s)…"):
                    r = requests.post(
                        f"{backend.rstrip('/')}/api/v1/admin/training/images",
                        files=body,
                        data=form_data,
                        headers={"Authorization": f"Bearer {st.session_state.admin_token}"},
                        timeout=600,
                    )
                    r.raise_for_status()
                    result = r.json()
                st.session_state.training_upload_last_batch = result.get("upload_batch_id")
                st.session_state.training_upload_last_response = result
                st.session_state.training_uploader_nonce = next_training_uploader_nonce(
                    st.session_state.training_uploader_nonce
                )
                st.success(
                    f"Saved {result.get('count', 0)} photo(s). "
                    f"{result.get('queued_count', 0)} unique photo(s) queued for Tooth Detector V2."
                )
                if result.get("rejected"):
                    st.warning(f"{len(result['rejected'])} photo(s) could not be staged; see the upload response below.")
                    st.dataframe(_upload_rejection_rows(result["rejected"]), use_container_width=True, hide_index=True)
                st.caption(f"Upload batch: {result.get('upload_batch_id')}")
                st.rerun()
            except requests.RequestException as exc:
                _show_backend_error(exc)
        last_response = st.session_state.get("training_upload_last_response") or {}
        if last_response:
            st.success(
                f"Batch {last_response.get('upload_batch_id')}: "
                f"{last_response.get('count', 0)} photo(s) saved, "
                f"{last_response.get('queued_count', 0)} queued for detection, "
                f"{last_response.get('duplicate_count', 0)} SHA/perceptual duplicate(s)."
            )
            if last_response.get("rejected"):
                st.dataframe(_upload_rejection_rows(last_response["rejected"]), use_container_width=True, hide_index=True)
        listing = client.get("/api/v1/admin/training/images")
        st.markdown("### Photo processing status")
        status_rows = [
            {"Check": check, "Status": status, "Photos": count}
            for check, values in (
                ("Duplicate review", listing.get("status_counts") or {}),
                ("Tooth detection", listing.get("processing_status_counts") or {}),
            )
            for status, count in values.items()
        ]
        if status_rows:
            st.dataframe(status_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No upload statuses to show yet.")

        @st.fragment(run_every="3s")
        def upload_history_panel():
            history = client.get("/api/v1/admin/training/uploads")
            batches = history.get("batches") or []
            history_images = history.get("images") or []
            if not batches:
                st.info("Upload history will appear here after your first batch.")
                return
            st.dataframe(
                [
                    {
                        "Upload date": str(batch.get("uploaded_at") or "")[:16].replace("T", " "),
                        "Batch": str(batch.get("upload_batch_id") or "")[:12],
                        "Photos": batch.get("total", 0),
                        "Detected": batch.get("completed", 0),
                        "Queued": batch.get("queued", 0),
                        "Failed": batch.get("failed", 0),
                        "Tooth crops": batch.get("tooth_crops", 0),
                        "Status": batch.get("status", "Unknown"),
                    }
                    for batch in batches[:30]
                ],
                use_container_width=True,
                hide_index=True,
            )
            selectable_batches = [str(batch.get("upload_batch_id")) for batch in batches if batch.get("upload_batch_id")]
            selected_batch = st.selectbox(
                "Batch details",
                selectable_batches,
                index=0,
                key="training_upload_history_batch",
            )
            batch_rows = [row for row in history_images if str(row.get("upload_batch_id")) == selected_batch]
            st.dataframe(
                [
                    {
                        "Photo ID": f"PHOTO-{int(row.get('training_image_id') or 0):03d}",
                        "Original filename": row.get("filename"),
                        "Patient ID": row.get("patient_id") or "Unassigned",
                        "Detection": row.get("processing_status"),
                        "Attempts": row.get("processing_attempts"),
                        "Tooth crops": row.get("n_crops"),
                        "Duplicate check": row.get("duplicate_status"),
                        "Next step": _friendly_upload_issue(row.get("processing_error")),
                    }
                    for row in batch_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
            selected_summary = next(
                (batch for batch in batches if str(batch.get("upload_batch_id")) == selected_batch),
                {},
            )
            st.progress(
                min(
                    1.0,
                    (selected_summary.get("completed", 0) + selected_summary.get("duplicates", 0) + selected_summary.get("invalid", 0))
                    / max(1, selected_summary.get("total", 0)),
                ),
                text=(
                    f"{selected_summary.get('completed', 0) + selected_summary.get('duplicates', 0) + selected_summary.get('invalid', 0) + selected_summary.get('failed', 0)}"
                    f"/{selected_summary.get('total', 0)} terminal · "
                    f"{selected_summary.get('processing', 0)} processing · {selected_summary.get('queued', 0)} queued · "
                    f"{selected_summary.get('failed', 0)} failed"
                ),
            )
            if selected_summary.get("failed") or selected_summary.get("queued"):
                if st.button("Resume failed / queued photos", key=f"resume_upload_{selected_batch}"):
                    try:
                        resumed = client.post(f"/api/v1/admin/training/uploads/{selected_batch}/resume")
                        st.success(f"Resumed {resumed.get('resumed_count', 0)} photo(s).")
                        st.rerun()
                    except requests.RequestException as exc:
                        _show_backend_error(exc)

        upload_history_panel()

        st.divider()
        st.subheader("Anonymous Patient ID assignment")
        st.caption(
            "Group photos only when you know they belong to the same patient. Crops inherit the source photo's "
            "patient ID. Unknown photos remain Unassigned and cannot enter a confirmed dataset version."
        )
        try:
            group_inventory = client.get("/api/v1/admin/training/patient-groups")
            group_photos = group_inventory.get("photos") or []
            group_patients = group_inventory.get("anonymous_patients") or []
            for row in group_photos:
                row.setdefault(
                    "photo_id",
                    f"PHOTO-{int(row.get('training_image_id') or 0):03d}",
                )
            st.caption(
                "Photo IDs are stable labels for the photo database records (for example, PHOTO-001). "
                "Only detector-complete photos can be assigned."
            )
            photo_search = st.text_input(
                "Search photos by Photo ID or filename",
                key="anonymous_group_photo_search",
            ).strip().casefold()
            visible_group_photos = [
                row for row in group_photos
                if not photo_search
                or photo_search in str(row.get("photo_id") or "").casefold()
                or photo_search in str(row.get("filename") or "").casefold()
            ]
            table_rows = [
                {
                    "Photo ID": row.get("photo_id"),
                    "Original filename": row.get("filename"),
                    "Current Patient ID": row.get("patient_id") or "Unassigned",
                    "Detection status": row.get("processing_status"),
                    "Detected tooth crops": row.get("crop_count", 0),
                }
                for row in visible_group_photos
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
            if visible_group_photos:
                with st.expander("Copy Photo IDs from the filtered table"):
                    st.caption("Select and copy the Photo IDs below.")
                    st.text_area(
                        "Photo IDs",
                        value="\n".join(str(row["photo_id"]) for row in visible_group_photos),
                        height=120,
                        disabled=True,
                        key="copy_filtered_photo_ids",
                    )
            if not group_photos:
                st.info("No uploaded photos are available for patient grouping yet.")
            elif not any(row.get("assignment_ready") for row in group_photos):
                st.info("No detector-complete photos are ready for patient grouping yet.")

            st.subheader("Patient Photo Grouping Review")
            st.caption(
                "A group is eligible for automatic anonymous-ID assignment only when the database photo's exact SHA-256 "
                "matches the source manifest and the linked image manifest has one explicit patient reference. "
                "Filename-only clues and ambiguous or unmatched records stay unassigned until you use manual assignment."
            )
            review_inventory = client.get("/api/v1/admin/training/patient-groups/review")
            review_suggestions = review_inventory.get("suggestions") or []
            review_summary = review_inventory.get("summary") or {}
            summary_cols = st.columns(4)
            summary_cols[0].metric("Confident source groups", review_summary.get("confident_groups", 0))
            summary_cols[1].metric("Photos with verified references", review_summary.get("confident_photos", 0))
            summary_cols[2].metric("Ambiguous references", review_summary.get("ambiguous_photos", 0))
            summary_cols[3].metric("Unmatched photos", review_summary.get("unmatched_photos", 0))
            missing_refs = int(review_summary.get("missing_reference_photos", 0) or 0)
            if missing_refs:
                st.info(f"{missing_refs} exact-hash source match(es) have no verified patient reference and require manual review.")
            if not review_inventory.get("source_metadata_available"):
                st.warning("One or both source manifests are unavailable. No automatic patient grouping will be applied.")
            if review_suggestions:
                st.dataframe(
                    [
                        {
                            "Photo ID": row.get("photo_id"),
                            "Filename": row.get("filename"),
                            "Source patient reference": row.get("source_patient_reference") or "Not available",
                            "Possible references": ", ".join(row.get("possible_patient_references") or []) or "None",
                            "Suggested grouping": row.get("suggested_group"),
                            "Evidence / reason": row.get("evidence"),
                            "Confidence": row.get("confidence"),
                            "Match status": row.get("source_metadata_status"),
                            "Proposed Patient ID": row.get("proposed_patient_id") or "Review required",
                            "Current Patient ID": row.get("current_patient_id") or "Unassigned",
                            "Review": str(row.get("review_decision") or "pending").title(),
                        }
                        for row in review_suggestions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                option_labels = {
                    row["photo_id"]: f"{row['photo_id']} — {row.get('filename') or 'filename unavailable'}"
                    for row in review_suggestions
                }
                selected_review_ids = st.multiselect(
                    "Select suggested photos to assign",
                    list(option_labels),
                    format_func=lambda value: option_labels[value],
                    key="patient_group_review_selected_photos",
                )
                if st.button(
                    "Use selected Photo IDs in assignment form",
                    disabled=not selected_review_ids,
                    key="patient_group_review_prefill_assignment",
                ):
                    st.session_state["anonymous_group_photo_ids_text"] = ", ".join(selected_review_ids)
                    st.rerun()

                review_groups = {}
                for row in review_suggestions:
                    if row.get("suggestion_key"):
                        review_groups.setdefault(row.get("suggestion_key"), []).append(row)
                for suggestion_key, rows in review_groups.items():
                    group_photo_ids = [row["photo_id"] for row in rows]
                    group_label = " + ".join(group_photo_ids)
                    decision = str(rows[0].get("review_decision") or "pending").title()
                    review_token = hashlib.sha256(suggestion_key.encode("utf-8")).hexdigest()[:12]
                    with st.expander(f"Review suggested group: {group_label} — {decision}"):
                        st.write(rows[0].get("evidence"))
                        is_confident = all(row.get("auto_group_eligible") for row in rows)
                        has_conflict = any(row.get("assignment_conflict") for row in rows)
                        not_ready = any(not row.get("assignment_ready") for row in rows)
                        if is_confident:
                            proposed_id = rows[0].get("proposed_patient_id")
                            if has_conflict:
                                st.warning("Patient assignment conflict found. Resolve it in the manual assignment form; this group will not overwrite existing Patient IDs.")
                            elif proposed_id:
                                st.info(f"Confirming this verified source group will assign all listed photos to **{proposed_id}**.")
                        else:
                            st.warning("This is a filename clue only. Confirmation records the review and pre-fills manual assignment; it will not create or assign a Patient ID.")
                        confirm_col, reject_col = st.columns(2)
                        if confirm_col.button(
                            "Confirm source group and assign ID" if is_confident else "Keep clue for manual review",
                            disabled=(has_conflict or not_ready) if is_confident else False,
                            key=f"confirm_photo_suggestion_{review_token}",
                        ):
                            try:
                                result = client.post(
                                    "/api/v1/admin/training/patient-groups/review",
                                    json={
                                        "suggestion_key": suggestion_key,
                                        "photo_ids": group_photo_ids,
                                        "decision": "confirm",
                                    },
                                )
                                if result.get("assigned_patient_id"):
                                    st.success(
                                        f"Assigned {len(group_photo_ids)} photo(s) to {result['assigned_patient_id']} "
                                        "using the verified source reference. Evidence and assignment were audited."
                                    )
                                else:
                                    st.session_state["anonymous_group_photo_ids_text"] = ", ".join(group_photo_ids)
                                    st.info("Review saved. This clue did not assign a patient ID; verify the grouping and use the manual assignment form below if appropriate.")
                                st.rerun()
                            except requests.RequestException as exc:
                                _show_backend_error(exc)
                        if reject_col.button(
                            "Reject this suggested group",
                            key=f"reject_photo_suggestion_{review_token}",
                        ):
                            try:
                                client.post(
                                    "/api/v1/admin/training/patient-groups/review",
                                    json={
                                        "suggestion_key": suggestion_key,
                                        "photo_ids": group_photo_ids,
                                        "decision": "reject",
                                    },
                                )
                                st.rerun()
                            except requests.RequestException as exc:
                                _show_backend_error(exc)
            else:
                st.info("No active training photos are available for source-manifest review.")

            photo_ids_text = st.text_area(
                "Photo IDs to assign",
                placeholder="PHOTO-001, PHOTO-002\nPHOTO-003",
                help="Enter Photo IDs separated by commas, spaces, or new lines. The preview reports invalid or missing IDs.",
                key="anonymous_group_photo_ids_text",
            )
            group_action = st.radio(
                "Assignment action",
                ["Create a new anonymous ID", "Use an existing anonymous ID", "Leave Unassigned"],
                horizontal=True,
                key="anonymous_group_action",
            )
            target_patient_id = None
            if group_action == "Use an existing anonymous ID":
                group_ids = [str(row["public_id"]) for row in group_patients]
                if group_ids:
                    target_patient_id = st.selectbox(
                        "Existing anonymous Patient ID",
                        group_ids,
                        format_func=lambda value: (
                            f"{value} · "
                            f"{next((row.get('photo_count', 0) for row in group_patients if row.get('public_id') == value), 0)} photos"
                        ),
                        key="anonymous_group_target_patient",
                    )
                else:
                    st.info("No anonymous groups exist yet. Choose Create a new anonymous ID first.")

            action_value = {
                "Create a new anonymous ID": "create",
                "Use an existing anonymous ID": "existing",
                "Leave Unassigned": "unassigned",
            }[group_action]
            assignment_plan = {
                "photo_ids": photo_ids_text,
                "action": action_value,
                "target_patient_id": target_patient_id if action_value == "existing" else None,
            }
            if st.button(
                "Preview bulk assignment",
                type="primary",
                disabled=not photo_ids_text.strip(),
                key="anonymous_group_preview_button",
            ):
                try:
                    st.session_state.anonymous_group_preview = client.post(
                        "/api/v1/admin/training/patient-groups/preview",
                        json=assignment_plan,
                    )
                    st.session_state.anonymous_group_preview_plan = assignment_plan
                except requests.RequestException as exc:
                    st.session_state.pop("anonymous_group_preview", None)
                    st.session_state.pop("anonymous_group_preview_plan", None)
                    _show_backend_error(exc)

            preview = st.session_state.get("anonymous_group_preview")
            if preview and st.session_state.get("anonymous_group_preview_plan") == assignment_plan:
                st.subheader("Preview before saving")
                st.metric("Photos selected", preview.get("photo_count", 0))
                if preview.get("requested_photo_ids"):
                    st.caption("Photo IDs entered: " + ", ".join(preview["requested_photo_ids"]))
                assignable_photo_ids = [
                    row.get("photo_id") for row in preview.get("photos") or []
                    if row.get("photo_id") and row.get("status") != "CONFLICT"
                ]
                if preview.get("can_assign") and assignable_photo_ids:
                    st.caption("Photo IDs in this assignment: " + ", ".join(assignable_photo_ids))
                if preview.get("photos"):
                    st.dataframe(
                        [
                            {
                                "Photo ID": row.get("photo_id"),
                                "Original filename": row.get("filename"),
                                "Current Patient ID": row.get("current_patient_id") or "Unassigned",
                                "New Patient ID": row.get("new_patient_id") or "Unassigned",
                                "Detection status": row.get("processing_status"),
                                "Detected tooth crops": row.get("crop_count", 0),
                                "Preview status": row.get("status"),
                            }
                            for row in preview["photos"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                if preview.get("target_patient_id"):
                    st.info(f"Target Patient ID: {preview['target_patient_id']}")
                for title, key in (
                    ("Invalid Photo IDs", "invalid_photo_ids"),
                    ("Photo IDs entered more than once", "duplicate_photo_ids"),
                    ("Photo IDs not found in the database", "nonexistent_photo_ids"),
                ):
                    values = preview.get(key) or []
                    if values:
                        st.error(f"{title}: " + ", ".join(values))
                if preview.get("reassignment_required"):
                    st.warning(
                        "Reassignment is required for: "
                        + ", ".join(preview["reassignment_required"])
                    )
                for conflict in preview.get("assignment_conflicts") or []:
                    st.error(
                        f"{conflict.get('photo_id')}: "
                        + "; ".join(conflict.get("reasons") or [])
                    )
                for blocker in preview.get("blockers") or []:
                    st.error(blocker)

                confirmed_reassignment = False
                if preview.get("reassignment_required"):
                    reassignment_key = hashlib.sha256(
                        f"{assignment_plan}|{preview.get('target_patient_id')}".encode("utf-8")
                    ).hexdigest()[:16]
                    confirmed_reassignment = st.checkbox(
                        "Confirm reassignment. The selected photos and their crops move together; ICDAS labels and verification history stay attached.",
                        key=f"anonymous_group_confirm_reassignment_{reassignment_key}",
                    )
                apply_disabled = (
                    not preview.get("can_assign")
                    or (bool(preview.get("reassignment_required")) and not confirmed_reassignment)
                )
                if st.button(
                    "Apply patient grouping",
                    type="primary",
                    disabled=apply_disabled,
                    key="anonymous_group_apply",
                ):
                    try:
                        result = client.post(
                            "/api/v1/admin/training/patient-groups/assign",
                            json={
                                "photo_ids": photo_ids_text,
                                "action": action_value,
                                "target_patient_id": target_patient_id if action_value == "existing" else None,
                                "expected_patient_id": preview.get("target_patient_id") if action_value == "create" else None,
                                "confirm_reassignment": confirmed_reassignment,
                            },
                        )
                        st.session_state.pop("anonymous_group_preview", None)
                        st.session_state.pop("anonymous_group_preview_plan", None)
                        st.success(
                            f"Processed {result.get('assigned_count', 0)} selected photo(s) for "
                            f"{result.get('patient_id') or 'Unknown / Unassigned'}; "
                            f"{result.get('changed_count', 0)} relationship(s) updated."
                        )
                        st.rerun()
                    except requests.RequestException as exc:
                        _show_backend_error(exc)
            with st.expander("Patient grouping audit history"):
                audit = client.get("/api/v1/admin/training/patient-groups/audit").get("events") or []
                st.dataframe(
                    [
                        {
                            "When (UTC)": row.get("created_at"),
                            "Who": row.get("actor"),
                            "Action": (row.get("details") or {}).get("action"),
                            "Target Patient ID": (row.get("details") or {}).get("target_patient_id") or "Unassigned",
                            "Changed photos": len((row.get("details") or {}).get("changes") or []),
                        }
                        for row in audit
                    ],
                    use_container_width=True,
                )
        except requests.RequestException as exc:
            _show_backend_error(exc)

        inv = client.get("/api/v1/admin/dataset")
        photos = inv.get("photos") or {}
        st.subheader("Library totals (not deleted when you upload a new batch)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Uploaded", photos.get("uploaded", 0))
        m2.metric("Unique", photos.get("unique", 0))
        m3.metric("Exact dupes", photos.get("exact_duplicates", 0))
        m4.metric("Likely dupes", photos.get("likely_duplicates", 0))
        m5.metric("Invalid", photos.get("invalid", 0))
        render_stepper(page, inv)
        _workflow_nav("upload", None, "Box review", "CONTINUE TO BOX REVIEW")

    elif page == "Box review":
        page_heading("Review detected teeth", "Inspect each unique photograph and verify detector boxes before ICDAS labeling.", "Box review")
        inv = client.get("/api/v1/admin/dataset")
        render_stepper(page, inv)
        verification_progress = inv.get("verification_progress") or {}
        st.progress(
            float(verification_progress.get("percent") or 0) / 100,
            text=(
                f"Box verification: {verification_progress.get('reviewed_photos', 0)} / "
                f"{verification_progress.get('total_photos', 0)} unique photos reviewed"
            ),
        )
        st.info(
            "Check the overlay, then verify. A photo does not need every ICDAS grade. "
            "You assign 0–4 later, one tooth at a time."
        )
        listing = client.get("/api/v1/admin/training/images")
        uniques = [
            im
            for im in listing.get("images") or []
            if im.get("duplicate_status") == "UNIQUE"
            and im.get("processing_status", "COMPLETED") == "COMPLETED"
        ]
        pending = [im for im in uniques if not im.get("boxes_verified")]
        st.caption(
            f"{len(uniques)} unique photos · {len(pending)} still need box review · "
            f"{len(uniques) - len(pending)} verified"
        )
        if not uniques:
            processing = listing.get("processing_status_counts") or {}
            if processing.get("QUEUED") or processing.get("PROCESSING"):
                st.info("Tooth Detector V2 is still processing uploads. Refresh this page when the upload history shows completion.")
            elif processing.get("FAILED"):
                st.warning("Some uploads need retry. Open Training upload, review the error, and resume the failed batch.")
            else:
                st.info("Upload a photo and wait for Detector V2 to finish before reviewing tooth boxes.")
            _workflow_nav("review_empty", "Training upload", None, "")
        else:
            filter_cols = st.columns(3)
            patient_options = sorted({im.get("patient_id") for im in uniques if im.get("patient_id")})
            selected_patient = filter_cols[0].selectbox(
                "Filter by pseudonymous patient ID", ["All patients", *patient_options]
            )
            review_filter = filter_cols[1].selectbox(
                "Review status", ["All", "Needs review", "Verified", "Deactivated"]
            )
            image_search = filter_cols[2].text_input("Search photo or image ID").strip().casefold()
            filtered_uniques = [
                im
                for im in uniques
                if (selected_patient == "All patients" or im.get("patient_id") == selected_patient)
                and (
                    review_filter == "All"
                    or (review_filter == "Needs review" and im.get("is_active", True) and not im.get("boxes_verified"))
                    or (review_filter == "Verified" and im.get("is_active", True) and im.get("boxes_verified"))
                    or (review_filter == "Deactivated" and not im.get("is_active", True))
                )
                and (
                    not image_search
                    or image_search in str(im.get("filename") or "").casefold()
                    or image_search in str(im.get("image_id") or "").casefold()
                    or image_search in str(im.get("patient_id") or "").casefold()
                )
            ]
            if not filtered_uniques:
                st.info("No photos match the current search and filters.")
                _workflow_nav("review_filtered_empty", "Training upload", None, "")
                st.stop()
            labels = [
                f"{im['image_id']}: {im['filename']} · {im.get('patient_id') or 'patient ID missing'} "
                f"({'Deactivated' if not im.get('is_active', True) else ('verified' if im.get('boxes_verified') else 'needs review')})"
                for im in filtered_uniques
            ]
            default_i = 0
            for i, im in enumerate(filtered_uniques):
                if im.get("is_active", True) and not im.get("boxes_verified"):
                    default_i = i
                    break
            choice = st.selectbox(
                "UNIQUE image",
                range(len(filtered_uniques)),
                index=min(default_i, len(filtered_uniques) - 1),
                format_func=lambda i: labels[i],
            )
            image_id = filtered_uniques[int(choice)]["training_image_id"]
            detail = client.get(f"/api/v1/admin/training/images/{image_id}")
            is_act = bool(detail.get("is_active", True))
            status_str = "Active" if is_act else "Deactivated"

            hdr_col1, hdr_col2 = st.columns([3, 1])
            with hdr_col1:
                st.markdown(
                    f"**Image {detail.get('image_id')}** · Patient `{detail.get('patient_id') or 'unassigned'}` · "
                    f"Filename: `{detail.get('filename')}` · Status: **{status_str}**"
                )
            with hdr_col2:
                if is_act:
                    confirm_key = f"confirm_deactivate_{image_id}"
                    if not st.session_state.get(confirm_key):
                        if st.button("Delete this photo", key=f"btn_deactivate_{image_id}"):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        st.warning(
                            "Are you sure you want to deactivate this photo?\n\n"
                            "This will remove this photo and its associated tooth crops/labels from future dataset builds and training, but the records will be preserved for safety."
                        )
                        col_c1, col_c2 = st.columns(2)
                        if col_c1.button("Confirm Deactivation", type="primary", key=f"btn_confirm_deactivate_{image_id}"):
                            try:
                                client.post(f"/api/v1/admin/training/photos/{image_id}/deactivate")
                                st.session_state[confirm_key] = False
                                st.success(f"Photo ID #{image_id} deactivated successfully.")
                                st.rerun()
                            except requests.RequestException as exc:
                                _show_backend_error(exc)
                        if col_c2.button("Cancel", key=f"btn_cancel_deactivate_{image_id}"):
                            st.session_state[confirm_key] = False
                            st.rerun()
                else:
                    st.button("Deactivated", disabled=True, key=f"btn_deactivated_{image_id}")

            known_patients = client.get("/api/v1/admin/patients", params={"limit": 1000}).get("patients") or []
            patient_options = [str(patient.get("public_id")) for patient in known_patients if patient.get("public_id")]
            patient_rows = {str(patient.get("public_id")): patient for patient in known_patients}
            with st.form(f"training_identity_{image_id}"):
                if patient_options:
                    current_patient_id = str(detail.get("patient_id") or "")
                    select_options = [""] + patient_options
                    patient_index = select_options.index(current_patient_id) if current_patient_id in select_options else 0
                    patient_value = st.selectbox(
                        "Registered patient",
                        select_options,
                        index=patient_index,
                        format_func=lambda value: (
                            "Select a patient" if not value else f"{value} · {patient_rows.get(value, {}).get('name', '')}"
                        ),
                        key=f"training_patient_id_{image_id}",
                    )
                else:
                    patient_value = ""
                    st.warning("Register a patient before saving a photo identity.")
                image_value = st.text_input(
                    "Image ID",
                    value=detail.get("image_id") or "",
                    key=f"training_image_id_{image_id}",
                )
                identity_saved = st.form_submit_button("Save image identity", disabled=not patient_value)
                if identity_saved:
                    try:
                        client.put(
                            f"/api/v1/admin/training/images/{image_id}/identity",
                            json={"patient_id": patient_value, "image_id": image_value},
                        )
                        st.success("Pseudonymous image identity saved.")
                        st.rerun()
                    except requests.RequestException as exc:
                        _show_backend_error(exc)

            c1, c2 = st.columns(2)
            with c1:
                b64_image(detail.get("original_base64"), "Original", width=420)
            with c2:
                b64_image(
                    detail.get("overlay_base64") or detail.get("original_base64"),
                    "Detection overlay",
                    width=420,
                )
            boxes = detail.get("boxes") or []
            st.caption(f"Detected tooth count: {len(boxes)} · verified={detail.get('boxes_verified')}")
            edited = []
            deleted = []
            for box in boxes:
                lid = box["label_id"]
                cols = st.columns([1, 1, 1, 1, 1])
                x1 = cols[0].number_input("x1", value=int(box["x1"]), key=f"x1{lid}")
                y1 = cols[1].number_input("y1", value=int(box["y1"]), key=f"y1{lid}")
                x2 = cols[2].number_input("x2", value=int(box["x2"]), key=f"x2{lid}")
                y2 = cols[3].number_input("y2", value=int(box["y2"]), key=f"y2{lid}")
                if cols[4].checkbox("Delete", key=f"del{lid}"):
                    deleted.append(lid)
                else:
                    edited.append(
                        {"label_id": lid, "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                    )
                b64_image(box.get("crop_base64"), width=120)
                st.caption(
                    f"Detector confidence: {float(box.get('confidence') or 0.0):.3f} · "
                    f"{box.get('crop_status_caption') or ''}"
                )
            st.subheader("Add missing tooth box")
            a1, a2, a3, a4 = st.columns(4)
            nx1 = a1.number_input("new x1", value=0, key="nx1")
            ny1 = a2.number_input("new y1", value=0, key="ny1")
            nx2 = a3.number_input("new x2", value=0, key="nx2")
            ny2 = a4.number_input("new y2", value=0, key="ny2")
            if st.button("Save boxes and regenerate crops"):
                payload_boxes = list(edited)
                if nx2 > nx1 and ny2 > ny1:
                    payload_boxes.append(
                        {"x1": int(nx1), "y1": int(ny1), "x2": int(nx2), "y2": int(ny2)}
                    )
                try:
                    client.put(
                        f"/api/v1/admin/training/images/{image_id}/boxes",
                        json={"boxes": payload_boxes, "deleted_ids": deleted},
                    )
                    st.success("Crops regenerated from the original photo.")
                    st.rerun()
                except requests.RequestException as exc:
                    _show_backend_error(exc)
            if st.button("Verify boxes for ICDAS labeling"):
                try:
                    client.post(f"/api/v1/admin/training/images/{image_id}/verify-boxes", json={})
                    st.success("Verified. Continue to labeling when you are ready.")
                    st.rerun()
                except requests.RequestException as exc:
                    _show_backend_error(exc)
            if pending:
                st.warning(
                    f"{len(pending)} unique photo(s) still need box review. "
                    "Unverified crops are not added to the ICDAS labeling queue."
                )
            n_verified = len(uniques) - len(pending)
            nxt = "Labeling" if n_verified else None
            if not n_verified:
                st.caption("Verify at least one UNIQUE photo before ICDAS labeling.")
            _workflow_nav("review", "Training upload", nxt, "CONTINUE TO ICDAS LABELING")

    elif page == "Labeling":
        page_heading("ICDAS labeling", "Search by photo, resume at the first unlabeled crop, or review a saved grade.", "Annotation workspace")
        try:
            photo_inventory = client.get("/api/v1/admin/training/labeling/photos")
        except requests.RequestException as exc:
            st.error("Could not load photo labeling progress. Refresh and try again.")
            _show_backend_error(exc)
            photo_inventory = {"photos": []}

        resume_position = photo_inventory.get("saved_resume_position")
        if resume_position:
            st.info(
                "Saved resume position: "
                f"{resume_position['photo_id']} · Crop ID {resume_position['crop_id']} "
                f"(updated {resume_position.get('updated_at') or 'time unavailable'})."
            )
        last_saved = photo_inventory.get("last_saved_label")
        if last_saved:
            st.caption(
                "Most recent saved grade in the audit history: "
                f"{last_saved['photo_id']} · Crop ID {last_saved['crop_id']} · "
                f"ICDAS {last_saved['grade']} · {last_saved.get('saved_at') or 'time unavailable'}."
            )

        all_photo_rows = photo_inventory.get("photos") or []
        patients = sorted({row["patient_id"] for row in all_photo_rows if row.get("patient_id")})
        patient_options = ["All", "Unassigned", *patients]
        status_options = [
            "All",
            "FULLY LABELED",
            "PARTIALLY LABELED",
            "NOT LABELED",
            "NEEDS BOX VERIFICATION",
            "NO ELIGIBLE CROPS",
            "EXCLUDED DUPLICATE",
            "PROCESSING",
            "INACTIVE",
        ]
        if st.session_state.get("label_patient_filter") not in patient_options:
            st.session_state["label_patient_filter"] = "All"
        if st.session_state.get("label_split_filter") not in ["All", "UNASSIGNED", "train", "val", "test"]:
            st.session_state["label_split_filter"] = "All"
        if st.session_state.get("label_status_filter") not in status_options:
            st.session_state["label_status_filter"] = "All"
        if st.session_state.get("label_grade_filter") not in ["All", 0, 1, 2, 3, 4]:
            st.session_state["label_grade_filter"] = "All"

        fcols = st.columns(4)
        patient_filter = fcols[0].selectbox("Patient", patient_options, key="label_patient_filter")
        split_filter = fcols[1].selectbox(
            "Split", ["All", "UNASSIGNED", "train", "val", "test"], key="label_split_filter"
        )
        status_filter = fcols[2].selectbox("Photo labeling status", status_options, key="label_status_filter")
        grade_filter = fcols[3].selectbox("ICDAS grade", ["All", 0, 1, 2, 3, 4], key="label_grade_filter")
        search = st.text_input(
            "Search Photo ID, photo/reference number, or original filename",
            key="label_crop_search",
        ).strip().casefold()

        filtered_photos = []
        for row in all_photo_rows:
            if patient_filter == "Unassigned" and row.get("patient_id"):
                continue
            if patient_filter not in {"All", "Unassigned"} and row.get("patient_id") != patient_filter:
                continue
            if split_filter != "All" and row.get("split_name") != split_filter:
                continue
            if status_filter != "All" and row.get("labeling_status") != status_filter:
                continue
            if grade_filter != "All" and not int((row.get("grade_counts") or {}).get(str(grade_filter), 0)):
                continue
            searchable = " ".join(
                str(row.get(key) or "")
                for key in ("photo_id", "photo_number", "reference_number", "original_filename", "patient_id")
            ).casefold()
            if search and search not in searchable:
                continue
            filtered_photos.append(row)

        if filtered_photos:
            display_rows = []
            for row in filtered_photos:
                filename_display = (
                    row.get("original_filename")
                    if row.get("original_filename_available")
                    else f"(legacy name unavailable) {row.get('filename') or ''}"
                )
                display_rows.append(
                    {
                        "Photo ID": row["photo_id"],
                        "Photo number": row.get("photo_number"),
                        "Reference number": row.get("reference_number"),
                        "Original filename / stored filename": filename_display,
                        "Patient ID": row.get("patient_id") or "Unassigned",
                        "Status": row.get("labeling_status"),
                        "Eligible crops": row.get("eligible_crops", 0),
                        "Labeled": row.get("labeled_crops", 0),
                        "Remaining unlabeled": row.get("remaining_unlabeled_crops", 0),
                        "Uncertain": row.get("uncertain_crops", 0),
                        "Ungradable": row.get("ungradable_crops", 0),
                        **{f"ICDAS {grade}": (row.get("grade_counts") or {}).get(str(grade), 0) for grade in range(5)},
                    }
                )
            st.dataframe(display_rows, use_container_width=True, hide_index=True)
            st.caption("Older uploads may not have retained their original filename; their stored safe filename is shown in that case.")
        else:
            st.info("No photos match the current search and filters.")

        option_ids = [int(row["training_image_id"]) for row in filtered_photos]
        if st.session_state.get("label_selected_photo") not in [None, *option_ids]:
            st.session_state["label_selected_photo"] = None
        if not st.session_state.get("label_resume_photo_initialized"):
            saved_photo_id = int(resume_position["training_image_id"]) if resume_position else None
            if saved_photo_id in option_ids:
                st.session_state["label_selected_photo"] = saved_photo_id
            st.session_state["label_resume_photo_initialized"] = True
        selected_photo = st.selectbox(
            "Choose Photo",
            [None, *option_ids],
            format_func=lambda value: "Choose a photo"
            if value is None
            else next(
                f"{row['photo_id']} · Ref {row.get('reference_number') or row.get('photo_number')} · "
                f"{row.get('original_filename')} · {row.get('patient_id') or 'Unassigned'} · "
                f"{row.get('labeled_crops', 0)}/{row.get('eligible_crops', 0)} labeled · {row.get('labeling_status')}"
                for row in filtered_photos
                if int(row["training_image_id"]) == int(value)
            ),
            key="label_selected_photo",
        )

        selection_signature = selected_photo
        if st.session_state.get("label_selected_photo_signature") != selection_signature:
            st.session_state.label_selected_photo_signature = selection_signature
            st.session_state.label_id = None
            st.session_state.label_review_crop_id = None
            st.session_state.label_work_mode = "resume"
            st.session_state.pop("pending_label_overwrite", None)
        if selected_photo is None:
            st.info("Choose a photo from the table or search by Photo ID, reference number, or filename.")
        else:
            selected_summary = next(
                row for row in all_photo_rows if int(row["training_image_id"]) == int(selected_photo)
            )
            selected_filename = selected_summary.get("original_filename") or selected_summary.get("filename") or "Filename unavailable"
            if not selected_summary.get("original_filename_available"):
                selected_filename += " (stored filename; original name unavailable)"
            st.markdown(
                f"**{selected_summary['photo_id']}** · Reference `{selected_summary.get('reference_number') or selected_summary.get('photo_number')}` · "
                f"{selected_filename} · Patient: {selected_summary.get('patient_id') or 'Unassigned'}"
            )
            mcols = st.columns(5)
            mcols[0].metric("Eligible crops", selected_summary.get("eligible_crops", 0))
            mcols[1].metric("Already labeled", selected_summary.get("labeled_crops", 0))
            mcols[2].metric("Remaining unlabeled", selected_summary.get("remaining_unlabeled_crops", 0))
            mcols[3].metric("Uncertain", selected_summary.get("uncertain_crops", 0))
            mcols[4].metric("Ungradable", selected_summary.get("ungradable_crops", 0))
            gcols = st.columns(5)
            for grade in range(5):
                gcols[grade].metric(
                    f"ICDAS {grade}", int((selected_summary.get("grade_counts") or {}).get(str(grade), 0))
                )
            eligible_total = int(selected_summary.get("eligible_crops", 0) or 0)
            labeled_total = int(selected_summary.get("labeled_crops", 0) or 0)
            st.progress(
                (labeled_total / eligible_total) if eligible_total else 0.0,
                text=f"Selected photo progress: {labeled_total} / {eligible_total} eligible crops labeled",
            )
            if selected_summary.get("labeling_status") == "FULLY LABELED":
                st.success("All crops labeled. You can review existing labels below.")
            elif not selected_summary.get("remaining_unlabeled_crops") and selected_summary.get("eligible_crops"):
                st.info("No unlabeled crops remain. Review uncertain or ungradable crops below if needed.")
            elif selected_summary.get("labeling_status") in {"NEEDS BOX VERIFICATION", "NO ELIGIBLE CROPS", "EXCLUDED DUPLICATE"}:
                st.warning(
                    f"This photo has no eligible labeling crops ({selected_summary.get('labeling_status')}). "
                    "Complete box verification on a unique photo first."
                )

            try:
                photo_crop_response = client.get(
                    "/api/v1/admin/training/crops", params={"training_image_id": selected_photo}
                )
                crops_for_photo = photo_crop_response.get("crops") or []
            except requests.RequestException as exc:
                st.error(f"Could not load crops for {selected_summary['photo_id']}. Refresh and try again.")
                _show_backend_error(exc)
                crops_for_photo = []
            eligible_crops = [
                row
                for row in crops_for_photo
                if row.get("active") and row.get("box_verified") and row.get("label_id") is not None
            ]
            if crops_for_photo:
                st.dataframe(
                    [
                        {
                            "Crop ID": row.get("crop_id"),
                            "Crop number": index + 1,
                            "ICDAS": row.get("grade") if row.get("grade") is not None else row.get("annotation_status"),
                            "Verified": bool(row.get("box_verified")),
                            "Label record": row.get("label_id"),
                        }
                        for index, row in enumerate(crops_for_photo)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            mode = st.session_state.get("label_work_mode", "resume")
            action_cols = st.columns(2)
            if action_cols[0].button("Resume from first unlabeled crop", key="label_resume_first"):
                st.session_state.label_work_mode = "resume"
                st.session_state.label_id = None
                st.session_state.pop("pending_label_overwrite", None)
                st.rerun()
            if action_cols[1].button("Review all crops", key="label_review_all"):
                st.session_state.label_work_mode = "review"
                if eligible_crops:
                    st.session_state.label_review_crop_id = eligible_crops[0].get("crop_id")
                st.session_state.pop("pending_label_overwrite", None)
                st.rerun()

            jump_cols = st.columns([3, 1])
            crop_id_text = jump_cols[0].text_input("Jump to Crop ID", key="label_jump_crop_input").strip()
            if jump_cols[1].button("Jump to crop", key="label_jump_crop_button"):
                match = next(
                    (row for row in crops_for_photo if str(row.get("crop_id") or "").casefold() == crop_id_text.casefold()),
                    None,
                )
                if match is None:
                    st.error("That Crop ID does not belong to the selected photo.")
                elif not match.get("active") or not match.get("box_verified"):
                    st.error("That crop is inactive or unverified and cannot be labeled.")
                else:
                    st.session_state.label_work_mode = "review"
                    st.session_state.label_review_crop_id = match.get("crop_id")
                    st.session_state.pop("pending_label_overwrite", None)
                    st.rerun()

            crop_options = [row.get("crop_id") for row in eligible_crops]
            if st.session_state.get("label_review_crop_id") not in crop_options:
                st.session_state.label_review_crop_id = crop_options[0] if mode == "review" and crop_options else None
            crop_by_public_id = {str(row.get("crop_id")): row for row in eligible_crops}
            if st.session_state.get("label_review_crop_selector") != st.session_state.get("label_review_crop_id"):
                st.session_state["label_review_crop_selector"] = st.session_state.get("label_review_crop_id")
            selected_crop_public_id = st.selectbox(
                "Review crop",
                [None, *crop_options],
                format_func=lambda value: "Choose a crop"
                if value is None
                else next(
                    f"{value} · ICDAS {crop_by_public_id[str(value)].get('grade') if crop_by_public_id[str(value)].get('grade') is not None else crop_by_public_id[str(value)].get('annotation_status')}"
                    for _ in [0]
                ),
                key="label_review_crop_selector",
            )
            st.session_state.label_review_crop_id = selected_crop_public_id
            if selected_crop_public_id:
                st.session_state.label_work_mode = "review"
            selected_crop_row = crop_by_public_id.get(str(selected_crop_public_id)) if selected_crop_public_id else None
            queue_params = {
                "image_id": int(selected_photo),
                "mode": "first_unlabeled" if st.session_state.get("label_work_mode") == "resume" else "photo",
                "track": "initial" if st.session_state.get("label_work_mode") == "resume" else "review",
            }
            if queue_params["mode"] == "photo" and selected_crop_row:
                queue_params["label_id"] = int(selected_crop_row["label_id"])
            queue = client.get("/api/v1/admin/training/queue", params=queue_params)
            render_stepper(page, queue)
            cur = queue.get("current")
            if st.session_state.get("labeling_flash"):
                st.success(st.session_state.pop("labeling_flash"))
            if cur:
                st.caption(
                    f"Crop {cur.get('tooth_on_photo')}/{cur.get('teeth_on_photo')} · "
                    f"Crop ID {cur.get('crop_id')} · Label record {cur.get('label_id')} · "
                    f"{selected_filename}"
                )

                def _select_label(label_id):
                    row = next((item for item in eligible_crops if int(item.get("label_id") or -1) == int(label_id or -2)), None)
                    if row:
                        st.session_state.label_review_crop_id = row.get("crop_id")
                    st.session_state.label_work_mode = "review"
                    st.session_state.label_id = int(label_id) if label_id else None
                    st.session_state.pop("pending_label_overwrite", None)
                    st.rerun()

                def _advance_after_annotation():
                    if st.session_state.get("label_work_mode") == "review":
                        next_id = queue.get("next_label_id")
                        next_row = next(
                            (item for item in eligible_crops if int(item.get("label_id") or -1) == int(next_id or -2)),
                            None,
                        )
                        if next_row:
                            st.session_state.label_review_crop_id = next_row.get("crop_id")
                        else:
                            st.session_state.label_work_mode = "resume"
                            st.session_state.label_review_crop_id = None
                    st.session_state.label_id = None
                    st.session_state.pop("pending_label_overwrite", None)
                    st.rerun()

                nav = st.columns(3)
                previous_id = queue.get("prev_label_id")
                if nav[0].button("Previous crop", disabled=previous_id is None, key="label_previous_crop"):
                    _select_label(previous_id)
                ordered_crops = sorted(
                    eligible_crops,
                    key=lambda row: (int(row.get("index_in_image") or 0), int(row.get("label_id") or 0)),
                )
                current_index = next(
                    (index for index, row in enumerate(ordered_crops) if int(row.get("label_id") or -1) == int(cur["label_id"])),
                    -1,
                )
                next_unlabeled = next(
                    (
                        row for row in ordered_crops[current_index + 1 :]
                        if row.get("grade") is None and (row.get("annotation_status") or "UNLABELED").upper() == "UNLABELED"
                    ),
                    None,
                )
                if nav[1].button("Next unlabeled crop", disabled=next_unlabeled is None, key="label_next_unlabeled"):
                    _select_label(next_unlabeled["label_id"])
                current_annotation_status = (cur.get("annotation_status") or "UNLABELED").upper()
                skip_disabled = cur.get("grade") is not None or current_annotation_status == "UNCERTAIN"
                if nav[2].button("Skip as uncertain", disabled=skip_disabled, key="label_skip_uncertain"):
                    try:
                        client.post("/api/v1/admin/training/labels/skip", json={"label_id": cur["label_id"]})
                        st.session_state.labeling_flash = f"Crop {cur.get('crop_id')} saved as uncertain; no grade was assigned."
                        _advance_after_annotation()
                    except requests.RequestException as exc:
                        st.error("The uncertainty status was not saved. Try again.")
                        _show_backend_error(exc)
                if skip_disabled:
                    if cur.get("grade") is not None:
                        st.caption("This crop already has a saved grade. It is protected from being cleared by Skip as uncertain.")
                    else:
                        st.caption("This crop is already marked uncertain.")

                grade_text = (
                    f"ICDAS {cur['grade']}"
                    if cur.get("grade") is not None
                    else (cur.get("annotation_status") or "UNLABELED")
                )
                st.write("Current:", grade_text)
                cap = cur.get("crop_status_caption") or ""
                if (cur.get("crop_duplicate_status") or "UNIQUE") == "CONFLICT":
                    st.error(cap or "Duplicate crop has conflicting ICDAS labels. Manual review required.")
                elif (cur.get("crop_duplicate_status") or "UNIQUE") != "UNIQUE":
                    st.warning(cap)
                else:
                    st.caption(cap)
                b64_image(cur.get("crop_base64"), width=420)
                history = client.get(f"/api/v1/admin/training/labels/{cur['label_id']}/history")
                history_events = history.get("events") or []
                if history_events:
                    with st.expander("Label history and previous values"):
                        st.dataframe(_label_history_display_rows(history_events), use_container_width=True, hide_index=True)
                        can_restore = any(
                            (event.get("details") or {}).get("old", {}).get("grade") is not None
                            for event in history_events
                        )
                        if st.button("Restore previous grade", disabled=not can_restore, key=f"undo_label_{cur['label_id']}"):
                            client.post(f"/api/v1/admin/training/labels/{cur['label_id']}/undo")
                            st.session_state.pop("pending_label_overwrite", None)
                            st.session_state.labeling_flash = f"Restored the previous label for crop {cur.get('crop_id')}."
                            st.rerun()

                status_cols = st.columns(2)
                if status_cols[0].button(
                    "Mark uncertain",
                    disabled=cur.get("grade") is not None or current_annotation_status == "UNCERTAIN",
                    key=f"mark_uncertain_{cur['label_id']}",
                ):
                    try:
                        client.post(
                            "/api/v1/admin/training/labels/status",
                            json={"label_id": cur["label_id"], "status": "UNCERTAIN"},
                        )
                        st.session_state.labeling_flash = f"Crop {cur.get('crop_id')} marked uncertain without a grade."
                        _advance_after_annotation()
                    except requests.RequestException as exc:
                        st.error("The uncertainty status was not saved. Try again.")
                        _show_backend_error(exc)
                if status_cols[1].button(
                    "Mark ungradable",
                    disabled=cur.get("grade") is not None or current_annotation_status == "UNGRADABLE",
                    key=f"mark_ungradable_{cur['label_id']}",
                ):
                    try:
                        client.post(
                            "/api/v1/admin/training/labels/status",
                            json={"label_id": cur["label_id"], "status": "UNGRADABLE"},
                        )
                        st.session_state.labeling_flash = f"Crop {cur.get('crop_id')} marked ungradable without a grade."
                        _advance_after_annotation()
                    except requests.RequestException as exc:
                        st.error("The ungradable status was not saved. Try again.")
                        _show_backend_error(exc)

                st.caption("Click an ICDAS grade to save it immediately. A different saved grade requires confirmation.")
                grade_cols = st.columns(5)
                for grade in range(5):
                    if grade_cols[grade].button(
                        str(grade),
                        key=f"gradebtn{grade}_{cur['label_id']}",
                        disabled=cur.get("grade") is not None and int(cur["grade"]) == grade,
                    ):
                        try:
                            out = client.post(
                                "/api/v1/admin/training/labels",
                                json={
                                    "label_id": cur["label_id"],
                                    "grade": grade,
                                    "track": queue_params["track"],
                                },
                            )
                            if out.get("needs_confirmation"):
                                st.session_state.pending_label_overwrite = {
                                    "label_id": cur["label_id"],
                                    "previous_grade": out.get("previous_grade"),
                                    "grade": grade,
                                    "track": queue_params["track"],
                                }
                                st.rerun()
                            st.session_state.labeling_flash = f"Saved ICDAS {grade} for crop {cur.get('crop_id')}. Progress refreshed."
                            _advance_after_annotation()
                        except requests.RequestException as exc:
                            st.error("The label was not saved. Review the crop and try again.")
                            _show_backend_error(exc)

                pending_overwrite = st.session_state.get("pending_label_overwrite")
                if pending_overwrite and pending_overwrite.get("label_id") == cur["label_id"]:
                    st.warning(
                        f"This crop is already labeled ICDAS {pending_overwrite['previous_grade']}. "
                        f"Confirm replacing it with ICDAS {pending_overwrite['grade']}."
                    )
                    confirm_cols = st.columns(2)
                    if confirm_cols[0].button("Confirm relabel", type="primary", key=f"confirm_label_{cur['label_id']}"):
                        try:
                            client.post(
                                "/api/v1/admin/training/labels",
                                json={
                                    "label_id": cur["label_id"],
                                    "grade": pending_overwrite["grade"],
                                    "confirm_overwrite": True,
                                    "track": pending_overwrite["track"],
                                },
                            )
                            st.session_state.pop("pending_label_overwrite", None)
                            st.session_state.labeling_flash = f"Updated label for crop {cur.get('crop_id')}; history was retained."
                            _advance_after_annotation()
                        except requests.RequestException as exc:
                            st.error("The revised label was not saved. Try again.")
                            _show_backend_error(exc)
                    if confirm_cols[1].button("Keep current label", key=f"cancel_label_{cur['label_id']}"):
                        st.session_state.pop("pending_label_overwrite", None)
                        st.rerun()
            else:
                if st.session_state.get("label_work_mode") == "resume":
                    if selected_summary.get("eligible_crops") and selected_summary.get("labeled_crops") == selected_summary.get("eligible_crops"):
                        st.success("All crops labeled. Use Review all crops or Jump to Crop ID to inspect existing labels.")
                    elif not selected_summary.get("remaining_unlabeled_crops"):
                        st.info("No unlabeled crops remain. Use Review all crops to inspect uncertain or ungradable crops.")
                    else:
                        st.warning("No eligible crop is currently available. Verify tooth boxes on this photo first.")
                else:
                    st.info("This photo has no verified crops available for review.")
        _workflow_nav("label", "Box review", "Dataset", "CONTINUE TO DATASET")

    elif page == "Dataset":
        page_heading("Dataset management", "Review patient-level assignments, validate readiness, and create immutable dataset versions.", "Training data")
        summary = client.get("/api/v1/admin/dataset")
        render_stepper(page, summary)
        photos = summary.get("photos") or {}
        crops = summary.get("crops") or {}
        labels = summary.get("icdas_labels") or summary.get("class_counts") or {}
        workflow_status = summary.get("workflow_status") or {}
        st.subheader("ICDAS workflow status")
        status_cols = st.columns(6)
        for col, title, key in zip(
            status_cols,
            ["Labeling enabled", "Dataset available", "Labels completed", "Dataset ready", "Classifier trained", "Classifier deployed"],
            ["labeling_enabled", "dataset_available", "labeling_completed", "dataset_ready_for_training", "model_trained", "model_deployed"],
        ):
            col.metric(title, "Yes" if workflow_status.get(key) else "No")
        dup_n = int(photos.get("exact_duplicates", 0) or 0) + int(
            photos.get("likely_duplicates", 0) or 0
        )
        st.subheader("Dataset statistics (database)")
        r1 = st.columns(5)
        r1[0].metric("Uploaded photos", photos.get("uploaded", 0))
        r1[1].metric("Unique photos", photos.get("unique", 0))
        r1[2].metric("Duplicate photos", dup_n)
        r1[3].metric("Verified crops", crops.get("verified", 0))
        r1[4].metric("Labeled crops", crops.get("labeled", 0))
        st.metric(
            "Distinct pseudonymous patients",
            f"{photos.get('patients', 0)} / {photos.get('target_patients', 100)} target",
        )
        if summary.get("patient_count_warning"):
            st.warning(_safe_notice(summary["patient_count_warning"], "More patient groups are needed."))
        if summary.get("identity_warning"):
            st.warning(_safe_notice(summary["identity_warning"], "Some photos need patient identity review."))
        if summary.get("review_warning"):
            st.warning(_safe_notice(summary["review_warning"], "Some tooth detections need review."))
        if summary.get("annotation_warning"):
            st.warning(_safe_notice(summary["annotation_warning"], "Some eligible crops still need labels or verification."))
        if summary.get("patient_split_warning"):
            st.warning(_safe_notice(summary["patient_split_warning"], "Patient-level split integrity needs review."))
        r2 = st.columns(6)
        r2[0].metric("Unlabeled crops", crops.get("unlabeled", 0))
        for g in range(5):
            r2[g + 1].metric(f"ICDAS {g}", int(labels.get(str(g), 0) or 0))
        r3 = st.columns(5)
        r3[0].metric("Unique crops", crops.get("unique", 0))
        r3[1].metric("Exact duplicate crops", crops.get("exact_duplicates", 0))
        r3[2].metric("Possible duplicate crops", crops.get("possible_duplicates", 0))
        r3[3].metric("Conflicting crops", crops.get("conflicts", 0))
        r3[4].metric("Eligible training crops", crops.get("eligible", 0))
        _progress_card(summary)
        if summary.get("crop_conflict_message"):
            st.error(_safe_notice(summary["crop_conflict_message"], "Crop integrity conflicts need review."))
        ready = bool(summary.get("dataset_ready"))
        st.metric("Dataset readiness", summary.get("status") or ("READY TO BUILD" if ready else "NOT READY"))
        if summary.get("min_crops_message"):
            st.info(_safe_notice(summary["min_crops_message"], "More eligible crops are needed before training."))
        if summary.get("missing_classes_message"):
            st.warning(_safe_notice(summary["missing_classes_message"], "One or more ICDAS grades are missing."))
        latest_dataset = summary.get("latest_dataset") or {}
        latest_stats = latest_dataset.get("statistics") or {}
        if latest_stats.get("split_summary"):
            st.subheader("Latest persisted patient-level split")
            st.dataframe(_split_summary_rows(latest_stats["split_summary"]), use_container_width=True, hide_index=True)
            validation_result = latest_stats.get("validation") or {}
            if validation_result:
                st.caption(
                    "Integrity validation: "
                    + ("passed" if validation_result.get("ok") else "needs review")
                    + f" · patient leakage: {validation_result.get('patient_leakage_count', 0)}"
                )
            for note in latest_stats.get("split_limitations") or []:
                st.warning(_safe_notice(note, "Review the documented split limitations."))
        st.caption(
            "Patient assignments come from the database. All of a patient's photos and crops stay in one split; only verified ICDAS tooth crops enter classifier folders. Exporting does not start training."
        )
        with st.expander("Patient-wise assignment and export", expanded=True):
            inventory = client.get("/api/v1/admin/dataset/patients")
            patient_rows = inventory.get("patients") or []
            split_filter_cols = st.columns(3)
            patient_search = split_filter_cols[0].text_input("Search patients", key="split_patient_search").strip().casefold()
            existing_assignment_filter = split_filter_cols[1].selectbox(
                "Current assignment", ["All", "UNASSIGNED", "train", "val", "test"], key="split_current_filter"
            )
            strategy_name = split_filter_cols[2].selectbox(
                "Assignment strategy",
                ["Automatic percentages", "Manual patient assignment", "Hybrid manual + automatic", "Bulk assign selected patients"],
                key="patient_split_strategy",
            )
            strategy_key = {
                "Automatic percentages": "percentage",
                "Manual patient assignment": "manual",
                "Hybrid manual + automatic": "hybrid",
                "Bulk assign selected patients": "bulk",
            }[strategy_name]
            visible_patients = [
                row for row in patient_rows
                if (existing_assignment_filter == "All" or row.get("current_assignment") == existing_assignment_filter)
                and (
                    not patient_search
                    or patient_search in " ".join(
                        str(row.get(key) or "")
                        for key in ("patient_id", "patient_key", "assignment_status")
                    ).casefold()
                )
            ]
            editor_rows = []
            for row in visible_patients:
                current = row.get("current_assignment")
                initial_assignment = current if strategy_key == "manual" and current in {"train", "val", "test"} else "Unassigned"
                editor_rows.append(
                    {
                        "Select": False,
                        "Patient ID": row.get("patient_id") or f"Missing ID · photo group {row['patient_key']}",
                        "Original photos": int(row.get("photo_count") or 0),
                        "Detected crops": int(row.get("crop_count") or 0),
                        "Labelled crops": int(row.get("labeled_crop_count") or 0),
                        "Verified crops": int(row.get("verified_crop_count") or 0),
                        "ICDAS 0": int((row.get("grade_counts") or {}).get("0", 0)),
                        "ICDAS 1": int((row.get("grade_counts") or {}).get("1", 0)),
                        "ICDAS 2": int((row.get("grade_counts") or {}).get("2", 0)),
                        "ICDAS 3": int((row.get("grade_counts") or {}).get("3", 0)),
                        "ICDAS 4": int((row.get("grade_counts") or {}).get("4", 0)),
                        "Current assignment": current or "UNASSIGNED",
                        "Assignment": initial_assignment,
                        "Assignment status": row.get("assignment_status") or "",
                        "Patient key": row["patient_key"],
                    }
                )
            visible_patient_keys = [row["patient_key"] for row in visible_patients]
            editor_key_hash = hashlib.sha1(",".join(visible_patient_keys).encode("utf-8")).hexdigest()[:12]
            edited_rows = st.data_editor(
                editor_rows,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key=f"patient_assignment_editor_{strategy_key}_{editor_key_hash}",
                disabled=[
                    "Patient ID", "Original photos", "Detected crops", "Labelled crops", "Verified crops",
                    "ICDAS 0", "ICDAS 1", "ICDAS 2", "ICDAS 3", "ICDAS 4", "Current assignment",
                    "Assignment status", "Patient key",
                ],
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select"),
                    "Patient ID": st.column_config.TextColumn("Patient ID"),
                    "Assignment": st.column_config.SelectboxColumn(
                        "Assignment", options=["Unassigned", "train", "val", "test"], required=True
                    ),
                },
            )
            if hasattr(edited_rows, "to_dict"):
                edited_rows = edited_rows.to_dict(orient="records")
            records_by_key = {row["patient_key"]: row for row in visible_patients}
            selected_photo_ids = sorted(
                {
                    int(image_id)
                    for row in edited_rows
                    if row.get("Select") and row.get("Patient key") in records_by_key
                    for image_id in records_by_key[row["Patient key"]].get("training_image_ids") or []
                }
            )
            manual_assignments = {}
            for row in edited_rows:
                key = row.get("Patient key")
                assignment = row.get("Assignment")
                if key not in records_by_key or assignment not in {"train", "val", "test"}:
                    continue
                for image_id in records_by_key[key].get("training_image_ids") or []:
                    manual_assignments[str(int(image_id))] = assignment

            plan = {
                "strategy": strategy_key,
                "seed": int(st.number_input("Random seed", min_value=0, value=42, step=1, key="split_seed")),
                "allow_reassign_locked": True,
            }
            if strategy_key == "percentage":
                ratio_cols = st.columns(3)
                plan["train_pct"] = float(ratio_cols[0].number_input("Train target %", min_value=0.0, max_value=100.0, value=70.0, key="split_train_pct"))
                plan["val_pct"] = float(ratio_cols[1].number_input("Validation target %", min_value=0.0, max_value=100.0, value=15.0, key="split_val_pct"))
                plan["test_pct"] = float(ratio_cols[2].number_input("Test target %", min_value=0.0, max_value=100.0, value=15.0, key="split_test_pct"))
            elif strategy_key in {"manual", "hybrid"}:
                if strategy_key == "manual":
                    edited_by_key = {row.get("Patient key"): row for row in edited_rows}
                    for patient in patient_rows:
                        edited = edited_by_key.get(patient["patient_key"])
                        assignment = (
                            edited.get("Assignment")
                            if edited is not None
                            else patient.get("current_assignment")
                        )
                        if assignment in {"train", "val", "test"}:
                            for image_id in patient.get("training_image_ids") or []:
                                manual_assignments[str(int(image_id))] = assignment
                plan["assignments"] = manual_assignments
                st.caption("Manual mode uses each patient's selected split. Hybrid mode keeps those explicit assignments and allocates the rest automatically.")
            elif strategy_key == "bulk":
                plan["selected_image_ids"] = selected_photo_ids
                plan["bulk_split"] = st.selectbox("Bulk assign selected patients to", ["train", "val", "test"], key="bulk_split_target")
                st.caption("Selecting one patient assigns all of that patient's source photos and crops together.")

            if strategy_key in {"hybrid", "bulk"}:
                ratio_cols = st.columns(3)
                plan["train_pct"] = float(ratio_cols[0].number_input("Train target %", min_value=0.0, max_value=100.0, value=70.0, key="split_train_pct_hybrid"))
                plan["val_pct"] = float(ratio_cols[1].number_input("Validation target %", min_value=0.0, max_value=100.0, value=15.0, key="split_val_pct_hybrid"))
                plan["test_pct"] = float(ratio_cols[2].number_input("Test target %", min_value=0.0, max_value=100.0, value=15.0, key="split_test_pct_hybrid"))

            plan["allow_exclusions"] = st.checkbox(
                "I understand only unique, box-verified ICDAS 0–4 crops are exported; missing-ID, unassigned, duplicate, unverified, unlabeled, and ungradable records are excluded.",
                value=False,
                key="acknowledge_dataset_exclusions",
            )
            action_cols = st.columns(2)
            refresh_preview = action_cols[0].button("Refresh preview", type="primary", key="preview_patient_splits")
            if (
                refresh_preview
                or st.session_state.get("split_preview_plan") != plan
                or st.session_state.get("split_preview") is None
            ):
                try:
                    preview = client.post("/api/v1/admin/dataset/preview-split", json={"plan": plan})
                    st.session_state.split_preview = preview
                    st.session_state.split_preview_plan = plan
                except requests.RequestException as exc:
                    _show_backend_error(exc)

            preview = st.session_state.get("split_preview")
            preview_plan = st.session_state.get("split_preview_plan")
            if preview and preview_plan == plan:
                st.subheader("Preview before confirmation")
                split_rows = []
                for split, details in (preview.get("split_summary") or {}).items():
                    split_rows.append(
                        {
                            "Split": "Validation" if split == "val" else split.title(),
                            "Patients": details.get("patient_count", 0),
                            "Original photos": details.get("image_count", 0),
                            "Detected crops": details.get("tooth_crop_count", 0),
                            "Labelled crops": details.get("labeled_crop_count", 0),
                            "Verified crops": details.get("verified_crop_count", 0),
                            **{f"ICDAS {grade}": (details.get("icdas_grade_distribution") or {}).get(str(grade), 0) for grade in range(5)},
                            "Actual patients %": details.get("actual_patient_percent", 0),
                            "Target %": details.get("target_patient_percent", 0),
                            "Deviation (pp)": details.get("deviation_percentage_points", 0),
                        }
                    )
                st.dataframe(split_rows, use_container_width=True, hide_index=True)
                st.dataframe(
                    [
                        {
                            "Patient ID": row.get("patient_id") or f"Missing ID · {row.get('patient_key')}",
                            "Original photos": row.get("photo_count", 0),
                            "Detected crops": row.get("crop_count", 0),
                            "Labelled crops": row.get("labeled_crop_count", 0),
                            "Verified crops": row.get("verified_crop_count", 0),
                            "ICDAS grade distribution": _grade_distribution_text(row.get("grade_counts")),
                            "Current assignment": row.get("current_assignment"),
                            "Preview assignment": row.get("planned_assignment"),
                            "Status": row.get("assignment_status"),
                        }
                        for row in preview.get("patient_assignments") or []
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                preview_cols = st.columns(5)
                preview_cols[0].metric("Unassigned patients", len(preview.get("unassigned_patients") or []))
                preview_cols[1].metric("Missing-ID photos", len(preview.get("missing_patient_ids") or []))
                preview_cols[2].metric("Unlabelled crops", preview.get("unlabeled_crop_count", 0))
                preview_cols[3].metric("Unverified crops", preview.get("unverified_crop_count", 0))
                preview_cols[4].metric("Duplicate photos", preview.get("duplicate_photo_count", 0))
                if preview.get("patient_conflicts"):
                    st.error("Saved patient assignment conflicts: " + ", ".join(preview["patient_conflicts"][:10]))
                for warning in preview.get("warnings") or []:
                    st.warning(_safe_notice(warning, "Review the dataset preview warning."))
                for issue in preview.get("build_blockers") or []:
                    st.error(_safe_notice(issue, "A dataset integrity blocker needs review."))
                if preview.get("can_build") and st.button(
                    "Confirm and export this dataset version", type="primary", key="confirm_patient_split_build"
                ):
                    try:
                        out = client.post(
                            "/api/v1/admin/dataset/build",
                            json={"plan": preview_plan, "expected_fingerprint": preview.get("fingerprint")},
                        )
                        st.session_state.pop("split_preview", None)
                        st.session_state.pop("split_preview_plan", None)
                        st.success(f"Created dataset version {out.get('name')}.")
                        split_summary = (out.get("statistics") or {}).get("split_summary") or {}
                        if split_summary:
                            st.dataframe(
                                _split_summary_rows(split_summary),
                                use_container_width=True,
                                hide_index=True,
                            )
                        st.rerun()
                    except requests.RequestException as exc:
                        st.session_state.pop("split_preview", None)
                        _show_backend_error(exc)
                elif not preview.get("can_build"):
                    st.caption("Resolve the hard blockers or acknowledge the listed exclusions, then preview again.")
        st.subheader("Versions (never overwritten)")
        st.dataframe(_dataset_version_rows(summary.get("versions") or []), use_container_width=True, hide_index=True)
        with st.expander("Recent annotation and training audit history"):
            audit = client.get("/api/v1/admin/training/audit", params={"limit": 200})
            st.dataframe(_audit_display_rows(audit.get("events") or []), use_container_width=True, hide_index=True)
        nxt = "Training" if summary.get("latest_dataset") else None
        _workflow_nav("dataset", "Labeling", nxt, "CONTINUE TO TRAINING")

    elif page == "Training":
        page_heading("Train ICDAS model", "Select a confirmed dataset version and review the training safeguards before starting a run.", "Model training")
        summary = client.get("/api/v1/admin/dataset")
        render_stepper(page, summary)
        enabled = bool(summary.get("icdas_train_enabled"))
        small_data_warning = (summary.get("workflow_status") or {}).get("small_dataset_warning")
        if small_data_warning:
            st.warning(_safe_notice(small_data_warning, "This dataset has a small number of patient groups."))
        versions = summary.get("versions") or []
        version_names = [str(version.get("name")) for version in versions if version.get("name")]
        if version_names:
            selected_name = st.selectbox(
                "Confirmed dataset version",
                version_names,
                index=0,
                key="training_dataset_version",
            )
            latest = next((version for version in versions if version.get("name") == selected_name), {})
        else:
            latest = summary.get("latest_dataset") or {}
        stats = latest.get("statistics") or {}
        patient_count = int(stats.get("total_patients") or 0)
        validation = stats.get("validation") or {}
        validation_issues = validation.get("issues") or []
        missing_train_classes = validation.get("missing_classes_in_train") or []
        missing_validation_classes = validation.get("missing_classes_in_validation") or []
        missing_test_classes = validation.get("missing_classes_in_test") or []
        validation_ok = bool(validation.get("ok")) and not validation_issues
        cc = stats.get("class_counts_overall") or summary.get("class_counts") or {}
        st.subheader("Training readiness")
        st.write(f"Dataset: **{latest.get('name') or '(none built yet)'}**")
        photos = summary.get("photos") or {}
        crops = summary.get("crops") or {}
        min_patients = int(summary.get("min_patients_for_training", 3) or 3)
        st.write(f"Total source photos (unique): {photos.get('unique', 0)}")
        st.write(f"Distinct patients in this dataset: {patient_count} / {min_patients} needed to populate three splits")
        st.write(f"Total verified tooth crops: {crops.get('verified', 0)}")
        st.write(f"Total labeled crops: {crops.get('labeled', 0)}")
        st.write(f"Eligible unique crops: {crops.get('eligible', 0)}")
        st.write(
            f"Crop exact duplicates: {crops.get('exact_duplicates', 0)} · "
            f"Possible: {crops.get('possible_duplicates', 0)} · "
            f"Conflicts: {crops.get('conflicts', 0)}"
        )
        for g in range(5):
            st.write(f"ICDAS {g}: {int(cc.get(str(g), 0) or 0)}")
        st.write(
            f"Train: {latest.get('n_train', stats.get('n_train', 0))} · "
            f"Validation: {latest.get('n_valid', stats.get('n_valid', 0))} · "
            f"Test: {latest.get('n_test', stats.get('n_test', 0))}"
        )
        st.write(f"Patient/image/crop leakage and manifest issues: {len(validation_issues)}")
        split_summary = stats.get("split_summary") or {}
        if split_summary:
            st.dataframe(
                _split_summary_rows(split_summary),
                use_container_width=True,
                hide_index=True,
            )
        if validation_issues:
            st.error(f"Dataset validation found {len(validation_issues)} issue(s). Resolve them in Dataset Management.")
        st.write(f"Dataset status: **{summary.get('status')}**")
        classes_ok = not (missing_train_classes or missing_validation_classes or missing_test_classes)
        if not stats:
            classes_ok = bool(summary.get("classes_ready"))
        crop_conflicts = int((summary.get("crops") or {}).get("conflicts") or 0)
        if summary.get("crop_conflict_message") or crop_conflicts:
            st.error(
                summary.get("crop_conflict_message")
                or "Duplicate crop has conflicting ICDAS labels. Manual review required."
            )
        if not classes_ok:
            st.warning(
                summary.get("missing_classes_message")
                or "Training requires genuine labeled examples of ICDAS 0, 1, 2, 3, and 4. "
                "Do not assign a missing class to a tooth that does not show it."
            )
        if patient_count < min_patients:
            st.warning(
                f"Three non-empty patient-level splits require at least {min_patients} distinct patients; "
                f"it currently has {patient_count}."
            )
        if not validation_ok:
            st.warning("Training is blocked until patient-level split and leakage checks pass.")
        if missing_train_classes:
            st.warning("Train split is missing ICDAS class(es): " + ", ".join(map(str, missing_train_classes)))
        if missing_validation_classes:
            st.warning("Validation split is missing ICDAS class(es): " + ", ".join(map(str, missing_validation_classes)))
        if missing_test_classes:
            st.warning("Held-out test split is missing ICDAS class(es): " + ", ".join(map(str, missing_test_classes)))
        st.write(f"ICDAS training switch: **{'ENABLED' if enabled else 'DISABLED'}**")
        if not enabled:
            st.error("ICDAS training is disabled. Set ICDAS_ALLOW_TRAIN=1 on the backend to enable training.")
        st.warning("Uploading, labeling, and BUILD DATASET never start training by themselves.")
        if st.button(
            "TRAIN ICDAS MODEL",
            disabled=(
                not enabled
                or not latest
                or not classes_ok
                or crop_conflicts > 0
                or patient_count < min_patients
                or not validation_ok
            ),
        ):
            out = client.post("/api/v1/admin/train", json={"dataset_name": latest.get("name")})
            st.session_state.job = out
            st.session_state.job_status = out
            st.success("Training request recorded.")
        if st.button("Refresh status"):
            st.session_state.job_status = client.get("/api/v1/admin/train/status")
        status = st.session_state.get("job_status") or st.session_state.get("job")
        if status:
            job_enabled = status.get("icdas_train_enabled")
            if job_enabled is None:
                job_enabled = enabled
            st.write(f"Status: **{str(status.get('status', 'unknown')).upper()}**")
            if status.get("status") in {"failed", "blocked"}:
                st.warning("The training job did not complete. Review dataset readiness and backend configuration.")
            if status.get("status") == "blocked" and enabled:
                st.info(
                    "This is a previous job from when training was disabled. "
                    "The backend currently has ICDAS training enabled; use TRAIN ICDAS MODEL to start a new run."
                )
            running = status.get("status") == "running"
            st.progress(
                0.4
                if running
                else (1.0 if status.get("status") in {"blocked", "completed", "failed"} else 0.0)
            )
            st.caption("Detailed technical logs remain on the backend.")
        _workflow_nav("train", "Dataset", None, "")

    elif page == "Model registry":
        page_heading("Model registry", "Review saved candidates, active status, validation evidence, and rollback options.", "Model governance")
        st.caption(
            "SET ACTIVE only changes which local ICDAS keras the backend loads. "
            "It is not cloud deployment. Previously saved model versions remain available for review."
        )
        models = client.get("/api/v1/admin/models").get("models") or []
        for m in models:
            with st.expander(f"{m['name']} · {'ACTIVE' if m['is_active'] else 'stored'}"):
                metrics = m.get("metrics") or {}
                st.write(f"Type: **{m.get('kind', 'model')}** · Dataset: **{m.get('dataset_version') or 'not recorded'}**")
                validation_only = metrics.get("evaluation_scope") == "validation_only"
                if validation_only:
                    st.warning("Validation-only DEMO candidate. It has not passed the production promotion gate.")
                    vm = metrics.get("validation_metrics") or {}
                    cols = st.columns(4)
                    cols[0].metric("Validation accuracy", f"{100 * float(vm.get('accuracy', 0)):.1f}%")
                    cols[1].metric("Macro F1", f"{float(vm.get('macro_f1', 0)):.3f}")
                    cols[2].metric("Balanced accuracy", f"{100 * float(vm.get('balanced_accuracy', 0)):.1f}%")
                    cols[3].metric("QWK", f"{float(vm.get('quadratic_kappa', 0)):.3f}")
                elif metrics:
                    cols = st.columns(4)
                    for col, key in zip(cols, ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")):
                        if metrics.get(key) is not None:
                            col.metric(key.replace("_", " ").title(), f"{float(metrics[key]):.3f}")
                b1, b2 = st.columns(2)
                if b1.button("SET ACTIVE (local inference)", key=f"d{m['id']}", disabled=validation_only):
                    try:
                        client.post(f"/api/v1/admin/models/{m['id']}/set-active", json={})
                        st.success("Model is active for local inference.")
                        st.rerun()
                    except requests.RequestException as exc:
                        _show_backend_error(exc)
                if b2.button("Clear active flag", key=f"r{m['id']}"):
                    try:
                        client.post(f"/api/v1/admin/models/{m['id']}/rollback", json={})
                        st.success("Model deactivated.")
                        st.rerun()
                    except requests.RequestException as exc:
                        _show_backend_error(exc)

    elif page == "Evaluation":
        page_heading("Evaluation", "Inspect recorded detector and classifier evaluation results with their provenance notes.", "Model quality")
        data = client.get("/api/v1/admin/evaluation")
        st.subheader("Tooth Detector V2")
        detector_metrics = data.get("tooth_detector_v2") or {}
        st.write("Status: **Available**" if detector_metrics else "No detector metrics are available.")
        for key, label in (("map50", "mAP@50"), ("precision", "Precision"), ("recall", "Recall")):
            if detector_metrics.get(key) is not None:
                st.metric(label, f"{float(detector_metrics[key]):.3f}")
        st.subheader("ICDAS historical metrics")
        icdas_metrics = data.get("icdas_historical") or {}
        if icdas_metrics:
            st.caption("Previously saved model evaluation metrics; confirm their dataset provenance before use.")
            for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"), ("quadratic_kappa", "QWK")):
                if icdas_metrics.get(key) is not None:
                    st.metric(label, f"{float(icdas_metrics[key]):.3f}")
        else:
            st.info("No historical ICDAS metrics are available.")
        st.caption(
            (data.get("note") or "")
            + " ICDAS metrics after a future train live on the model registry. "
            "They are research/screening metrics, not a clinical accuracy claim."
        )

    elif page == "Real-world test":
        page_heading("Real-world test", "Run an image through the current inference path and inspect the resulting crops.", "Inference check")
        st.info("Runs detector + ICDAS. Labels are not saved.")
        f = st.file_uploader("Unseen image", type=["jpg", "jpeg", "png"])
        if st.button("Run test") and f:
            try:
                r = requests.post(
                    f"{backend.rstrip('/')}/api/v1/admin/real-world-test",
                    files={"file": (f.name, f.getvalue(), "image/jpeg")},
                    headers={"Authorization": f"Bearer {st.session_state.admin_token}"},
                    timeout=180,
                )
                r.raise_for_status()
                out = r.json()
            except requests.RequestException as exc:
                _show_backend_error(exc)
                out = None
            if out:
                quality = out.get("quality") or {}
                st.info(f"Image quality: {quality.get('verdict', 'review')} · {quality.get('message', '')}")
                a, b = st.columns(2)
                with a:
                    b64_image(out.get("original_base64"), "Original", width=400)
                with b:
                    b64_image(out.get("overlay_base64"), "Overlay", width=400)
                for tth in out.get("teeth") or []:
                    st.markdown(
                        f"**{tth.get('tooth_id')} · ICDAS {tth['icdas_grade']} · {tth.get('current_stage')}**"
                    )
                    tabs = st.tabs(["Crop", "Heatmap", "Overlay"])
                    with tabs[0]:
                        b64_image(tth.get("crop_base64"), width=160)
                    with tabs[1]:
                        b64_image(tth.get("heatmap_base64"), width=160)
                    with tabs[2]:
                        b64_image(tth.get("overlay_base64"), width=160)
                st.caption(
                    f"TEST ONLY={out.get('test_only')} · labels_saved={out.get('labels_saved')} · "
                    f"training_dataset_updated={out.get('training_dataset_updated')}"
                )

except requests.RequestException as exc:
    _show_backend_error(exc)
