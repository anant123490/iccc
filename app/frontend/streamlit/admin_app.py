"""CCC AI Dentist Camera 2.0 — Admin portal. Requires backend admin login."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.admin_reset import RESET_CONFIRMATION_TEXT, reset_confirmation_ok
from shared.admin_workflow import NAV_GEN_KEY, NAV_PAGE_KEY, apply_pending_nav, render_stepper, set_admin_nav
from shared.auth import DEFAULT_BACKEND, PortalClient
from shared.charts import icdas_charts
from shared.components import b64_image, kpi, show_disclaimer
from shared.theme import apply
from shared.upload_state import clear_admin_training_ui_state, next_training_uploader_nonce

st.set_page_config(page_title="CCC Admin — Dentist Camera 2.0", page_icon="🛠️", layout="wide")
apply()

DISCLAIMER = (
    "Admin tools for dentists and dataset work. ICDAS labels are 0–4 only. "
    "Patient images and training images are stored separately. AI is not a diagnosis."
)

backend = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND)
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None

show_disclaimer(DISCLAIMER)

if not st.session_state.admin_token:
    st.header("Admin login")
    pw = st.text_input("Password", type="password", help="ICDAS_ADMIN_PASSWORD (default changeme)")
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
    st.subheader("Current Training Dataset Progress")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Photos**")
        st.write(
            f"Uploaded: {photos.get('uploaded', 0)}  \n"
            f"Unique: {photos.get('unique', 0)}  \n"
            f"Completed: {photos.get('completed', 0)}  \n"
            f"Remaining: {photos.get('remaining', 0)}"
        )
    with b:
        st.markdown("**Tooth crops**")
        st.write(
            f"Detected: {crops.get('detected', 0)}  \n"
            f"Verified: {crops.get('verified', 0)}  \n"
            f"Labeled: {crops.get('labeled', 0)}  \n"
            f"Eligible: {crops.get('eligible', crops.get('labeled', 0))}  \n"
            f"Unlabeled: {crops.get('unlabeled', 0)}"
        )
        st.caption(
            f"Skipped (Leave): {crops.get('skipped', 0)} · "
            f"Unique crops: {crops.get('unique', 0)} · "
            f"Exact dup: {crops.get('exact_duplicates', 0)} · "
            f"Possible dup: {crops.get('possible_duplicates', 0)} · "
            f"Conflicts: {crops.get('conflicts', 0)}"
        )
    with c:
        st.markdown("**ICDAS labels (verified manual)**")
        for g in range(5):
            st.write(f"{g} → {int(labels.get(str(g), 0) or 0)}")
        st.write(f"**Total labeled: {crops.get('labeled', 0)}**")
    if inv.get("imbalance_message"):
        st.warning(inv["imbalance_message"])


def _dentist_error_message(exc: BaseException) -> str:
    if isinstance(exc, (requests.ConnectionError, requests.ConnectTimeout)):
        return "Backend unavailable. Make sure FastAPI is running on port 8000."
    if isinstance(exc, requests.Timeout):
        return "The backend took too long to respond."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        resp = exc.response
        detail = None
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail")
            elif isinstance(body, str):
                detail = body
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):
            detail = "; ".join(str(x) for x in detail[:8])
        text = str(detail or resp.text or exc)
        if "Traceback" in text or 'File "' in text:
            return f"Backend error ({resp.status_code}). Make sure FastAPI is running on port 8000."
        if len(text) > 600:
            text = text[:600] + "…"
        return text
    return "Backend unavailable. Make sure FastAPI is running on port 8000."


def _show_backend_error(exc: BaseException) -> None:
    st.error(_dentist_error_message(exc))
    if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 401:
        st.session_state.admin_token = None


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


def _dataset_management() -> None:
    st.markdown("---")
    st.subheader("Dataset Management")
    if st.session_state.get("reset_summary"):
        _show_reset_summary(st.session_state.reset_summary)
    st.error(
        "Destructive. This permanently deletes Admin training photographs, crops, "
        "ICDAS labels, and built dataset versions. Model weights, source code, and "
        "patient records are not deleted. Training is not started."
    )
    try:
        plan = client.get("/api/v1/admin/training/reset/plan", params={"scope": "dataset"})
        with st.expander("Reset plan (what will be deleted)"):
            st.write("Paths:", [p.get("path") for p in plan.get("paths") or []])
            st.write("Database tables:", plan.get("database_tables"))
            st.caption("Preserved: " + ", ".join((plan.get("will_not_delete") or [])[:8]) + " …")
    except requests.RequestException as exc:
        _show_backend_error(exc)

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
            disabled=not reset_confirmation_ok(ok_ds, typed_ds),
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
            disabled=not reset_confirmation_ok(ok_full, typed_full),
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
        st.header("Admin dashboard")
        kpis = client.get("/api/v1/admin/kpis")
        cols = st.columns(4)
        keys = [
            ("patients", "Patients"),
            ("visits", "Visits"),
            ("images", "Images"),
            ("training_images", "Training images"),
            ("icdas_crops", "ICDAS crops"),
            ("labeled_crops", "Labeled crops"),
            ("dataset_versions", "Dataset versions"),
            ("current_models", "Models"),
        ]
        for i, (k, label) in enumerate(keys):
            with cols[i % 4]:
                kpi(label, kpis.get(k))
        st.caption(f"Latest model: {kpis.get('latest_model')}")
        inv = kpis.get("training_progress") or client.get("/api/v1/admin/dataset")
        render_stepper("Dashboard", inv)
        _progress_card(inv)
        st.button(
            "Go to training workflow",
            type="primary",
            key="nav_dash_workflow",
            on_click=set_admin_nav,
            args=("Training upload",),
        )
        _dataset_management()

    elif page == "Patients":
        st.header("Patient management")
        q = st.text_input("Search name or Patient ID")
        rows = client.get("/api/v1/admin/patients", params={"q": q} if q else None)
        for p in rows.get("patients") or []:
            with st.expander(f"{p['public_id']} · {p['name']}"):
                st.write(p)
                hist = client.get(f"/api/v1/patients/{p['public_id']}/history")
                for v in hist.get("visits") or []:
                    c1, c2 = st.columns([3, 1])
                    c1.write(v)
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
        st.header("Training image upload")
        st.info(
            "These files never mix with patient visit photos. Stored under data/training/uploads/. "
            "Duplicates are flagged and kept, but excluded from the ICDAS dataset. "
            "Tooth Detector V2 runs only on UNIQUE images. "
            "The file picker clears after a successful upload so you can choose a new batch; "
            "already-stored photos are not deleted or re-sent."
        )
        if "training_uploader_nonce" not in st.session_state:
            st.session_state.training_uploader_nonce = 0
        files = st.file_uploader(
            "1–100+ JPG/PNG (new batch only)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"training_files_{st.session_state.training_uploader_nonce}",
        )
        pending = list(files or [])
        if pending:
            st.caption(f"{len(pending)} file(s) selected for this batch (not uploaded yet).")
        if st.button("Upload and run Tooth Detector V2", disabled=not pending):
            # Copy bytes now so the request does not depend on the widget after reset.
            body = [("files", (f.name, f.getvalue(), "image/jpeg")) for f in pending]
            names = [f.name for f in pending]
            try:
                with st.spinner("Hashing, duplicate check, Tooth Detector V2…"):
                    r = requests.post(
                        f"{backend.rstrip('/')}/api/v1/admin/training/images",
                        files=body,
                        headers={"Authorization": f"Bearer {st.session_state.admin_token}"},
                        timeout=600,
                    )
                    r.raise_for_status()
                    st.session_state.train_up = r.json()
                st.session_state.last_upload_names = names
                st.session_state.training_uploader_nonce = next_training_uploader_nonce(
                    st.session_state.training_uploader_nonce
                )
                st.rerun()
            except requests.RequestException as exc:
                _show_backend_error(exc)
        last_names = st.session_state.get("last_upload_names") or []
        if last_names:
            st.success(
                "Last batch sent: "
                + ", ".join(last_names[:12])
                + ("…" if len(last_names) > 12 else "")
                + ". File picker is empty — choose the next photos when ready."
            )
        if st.session_state.get("train_up"):
            result = st.session_state.train_up
            st.caption(
                f"Stored {result.get('count')} files · "
                f"{result.get('unique_count')} UNIQUE · "
                f"{len(result.get('duplicates') or [])} flagged"
            )
            st.success(
                f"{result.get('count')} photos uploaded · "
                f"{result.get('unique_count')} unique · "
                f"{len(result.get('duplicates') or [])} duplicates/invalid"
            )
        listing = client.get("/api/v1/admin/training/images")
        st.subheader("Duplicate status (all uploads)")
        st.write(listing.get("status_counts") or {})
        flagged = [
            im
            for im in listing.get("images") or []
            if im.get("duplicate_status") != "UNIQUE"
        ]
        if flagged:
            st.warning("Flagged images are not deleted. They are excluded from dataset build.")
            st.dataframe(flagged, use_container_width=True)
        for im in (st.session_state.get("train_up") or {}).get("images") or []:
            st.subheader(f"{im.get('filename')} · {im.get('duplicate_status')}")
            st.caption(im.get("note"))
            b64_image(im.get("overlay_base64") or im.get("original_base64"), "Preview", width=480)
            if im.get("duplicate_status") == "UNIQUE":
                st.caption(f"{im.get('n_crops')} detector crops — review boxes next.")
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
        st.header("Review teeth")
        inv = client.get("/api/v1/admin/dataset")
        render_stepper(page, inv)
        st.info(
            "Check the overlay, then verify. A photo does not need every ICDAS grade. "
            "You assign 0–4 later, one tooth at a time."
        )
        listing = client.get("/api/v1/admin/training/images")
        uniques = [
            im
            for im in listing.get("images") or []
            if im.get("duplicate_status") == "UNIQUE"
        ]
        pending = [im for im in uniques if not im.get("boxes_verified")]
        st.caption(
            f"{len(uniques)} unique photos · {len(pending)} still need box review · "
            f"{len(uniques) - len(pending)} verified"
        )
        if not uniques:
            st.info("Upload UNIQUE training images first.")
            _workflow_nav("review_empty", "Training upload", None, "")
        else:
            labels = [
                f"{im['training_image_id']}: {im['filename']} "
                f"({'Deactivated' if not im.get('is_active', True) else ('verified' if im.get('boxes_verified') else 'needs review')})"
                for im in uniques
            ]
            default_i = 0
            for i, im in enumerate(uniques):
                if im.get("is_active", True) and not im.get("boxes_verified"):
                    default_i = i
                    break
            choice = st.selectbox(
                "UNIQUE image",
                range(len(uniques)),
                index=min(default_i, len(uniques) - 1),
                format_func=lambda i: labels[i],
            )
            image_id = uniques[int(choice)]["training_image_id"]
            detail = client.get(f"/api/v1/admin/training/images/{image_id}")
            is_act = bool(detail.get("is_active", True))
            status_str = "Active" if is_act else "Deactivated"

            hdr_col1, hdr_col2 = st.columns([3, 1])
            with hdr_col1:
                st.markdown(
                    f"**Photo ID #{image_id}** · Filename: `{detail.get('filename')}` · Status: **{status_str}**"
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
                st.caption(box.get("crop_status_caption") or "")
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
        st.header("ICDAS labeling")
        params = {}
        if st.session_state.get("label_resume"):
            params["resume"] = True
            st.session_state.label_resume = False
        elif st.session_state.get("label_id"):
            params["label_id"] = st.session_state.label_id
        queue = client.get("/api/v1/admin/training/queue", params=params or None)
        render_stepper(page, queue)
        n_lab = int(queue.get("n_labeled") or 0)
        n_crop = int(queue.get("n_crops") or 0)
        pct = (n_lab / n_crop) if n_crop else 0.0
        st.progress(pct, text=f"Overall: {n_lab:,} / {n_crop:,} teeth labeled ({pct:.1%})")
        photos = queue.get("photos") or {}
        st.caption(
            f"Photos completed {photos.get('completed', 0)} / {photos.get('box_verified', 0)} "
            f"(remaining {photos.get('remaining', 0)}) · "
            f"Unlabeled crops {queue.get('n_unlabeled', 0)} · Skipped {queue.get('n_skipped', 0)}"
        )
        counts = queue.get("icdas_labels") or queue.get("class_counts") or {}
        ccols = st.columns(6)
        for g in range(5):
            ccols[g].metric(f"ICDAS {g}", int(counts.get(str(g), 0) or 0))
        ccols[5].metric("Total labeled", n_lab)
        if queue.get("imbalance_message"):
            st.warning(queue["imbalance_message"])
        st.caption(
            "Click 0–4 to autosave. Leave/Skip if you are not sure. "
            "A photo may have only 0, 1 and 4 — Next stays enabled."
        )
        cur = queue.get("current")
        nav = st.columns(4)
        if nav[0].button("Previous", disabled=queue.get("prev_label_id") is None):
            st.session_state.label_id = queue.get("prev_label_id")
            st.rerun()
        if nav[1].button("Next", disabled=not queue.get("next_enabled")):
            st.session_state.label_id = queue.get("next_label_id")
            st.rerun()
        if nav[2].button("Leave / Skip", disabled=not cur):
            if cur:
                client.post("/api/v1/admin/training/labels/skip", json={"label_id": cur["label_id"]})
                st.session_state.label_id = queue.get("next_label_id") or cur["label_id"]
                st.rerun()
        if nav[3].button("Resume unlabeled"):
            st.session_state.label_id = None
            st.session_state.label_resume = True
            st.rerun()
        if not cur:
            st.info("Verify tooth boxes on UNIQUE images before labeling.")
        else:
            st.write(
                f"**Tooth {cur.get('global_index')} / {cur.get('global_total')}** · "
                f"Photo {cur.get('photo_index')} / {cur.get('photo_total')} · "
                f"{cur.get('filename')} · crop {cur.get('tooth_on_photo')}/{cur.get('teeth_on_photo')}"
            )
            status = (
                f"ICDAS {cur.get('grade')}"
                if cur.get("grade") is not None
                else ("Skipped" if cur.get("skipped") else "Unlabeled")
            )
            st.write("Current:", status)
            cap = cur.get("crop_status_caption") or ""
            if (cur.get("crop_duplicate_status") or "UNIQUE") == "CONFLICT":
                st.error(cap or "Duplicate crop has conflicting ICDAS labels. Manual review required.")
            elif (cur.get("crop_duplicate_status") or "UNIQUE") != "UNIQUE":
                st.warning(cap)
            else:
                st.caption(cap)
            b64_image(cur.get("crop_base64"), width=420)
            gcols = st.columns(5)
            for g in range(5):
                if gcols[g].button(str(g), key=f"gradebtn{g}"):
                    client.post(
                        "/api/v1/admin/training/labels",
                        json={"label_id": cur["label_id"], "grade": g},
                    )
                    st.session_state.label_id = queue.get("next_label_id") or cur["label_id"]
                    st.rerun()
        _workflow_nav("label", "Box review", "Dataset", "CONTINUE TO DATASET")

    elif page == "Dataset":
        st.header("Dataset")
        summary = client.get("/api/v1/admin/dataset")
        render_stepper(page, summary)
        photos = summary.get("photos") or {}
        crops = summary.get("crops") or {}
        labels = summary.get("icdas_labels") or summary.get("class_counts") or {}
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
            st.error(summary["crop_conflict_message"])
        ready = bool(summary.get("dataset_ready"))
        st.metric("Dataset readiness", summary.get("status") or ("READY TO BUILD" if ready else "NOT READY"))
        if summary.get("min_crops_message"):
            st.info(summary["min_crops_message"])
        if summary.get("missing_classes_message"):
            st.warning(summary["missing_classes_message"])
        st.caption(
            "BUILD DATASET uses all previous plus new verified labels, excludes duplicates, "
            "splits by original photo (seed 42, 70/15/15). Never overwrites v1, v2, … "
            "This does not start training."
        )
        if st.button("BUILD DATASET", disabled=not ready):
            try:
                out = client.post("/api/v1/admin/dataset/build", json={})
                st.success(out.get("name"))
                st.json(out.get("statistics") or out)
            except requests.RequestException as exc:
                _show_backend_error(exc)
        if not ready:
            st.caption(
                summary.get("min_crops_message")
                or f"Build stays off until at least {summary.get('min_dataset_crops', 5)} "
                "verified labeled UNIQUE crops exist."
            )
        st.subheader("Versions (never overwritten)")
        st.dataframe(summary.get("versions") or [], use_container_width=True)
        nxt = "Training" if summary.get("latest_dataset") else None
        _workflow_nav("dataset", "Labeling", nxt, "CONTINUE TO TRAINING")

    elif page == "Training":
        st.header("Train ICDAS model")
        summary = client.get("/api/v1/admin/dataset")
        render_stepper(page, summary)
        enabled = bool(summary.get("icdas_train_enabled"))
        latest = summary.get("latest_dataset") or {}
        stats = latest.get("statistics") or {}
        cc = stats.get("class_counts_overall") or summary.get("class_counts") or {}
        st.subheader("Training readiness")
        st.write(f"Dataset: **{latest.get('name') or '(none built yet)'}**")
        photos = summary.get("photos") or {}
        crops = summary.get("crops") or {}
        st.write(f"Total source photos (unique): {photos.get('unique', 0)}")
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
        leak = (stats.get("validation") or {}).get("issues") or []
        st.write(f"Duplicate leakage issues: {len(leak)}")
        st.write(f"Dataset status: **{summary.get('status')}**")
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
        if not enabled:
            st.error("ICDAS training is disabled. Set ICDAS_ALLOW_TRAIN=1 on the backend to enable training.")
        st.warning("Uploading, labeling, and BUILD DATASET never start training by themselves.")
        if st.button(
            "TRAIN ICDAS MODEL",
            disabled=not enabled or not latest or not classes_ok or crop_conflicts > 0,
        ):
            out = client.post("/api/v1/admin/train", json={})
            st.session_state.job = out
            st.json(out)
        if st.button("Refresh status"):
            st.session_state.job_status = client.get("/api/v1/admin/train/status")
        status = st.session_state.get("job_status") or st.session_state.get("job")
        if status:
            st.write(status.get("status"), status.get("message"))
            running = status.get("status") == "running"
            st.progress(
                0.4
                if running
                else (1.0 if status.get("status") in {"blocked", "completed", "failed"} else 0.0)
            )
            st.text_area("Logs", status.get("log") or "", height=240)
        _workflow_nav("train", "Dataset", None, "")

    elif page == "Model registry":
        st.header("Model registry")
        st.caption(
            "SET ACTIVE only changes which local ICDAS keras the backend loads. "
            "It is not cloud deployment. Versioned folders under models/icdas/vN/ stay intact."
        )
        models = client.get("/api/v1/admin/models").get("models") or []
        for m in models:
            with st.expander(f"{m['name']} · {'ACTIVE' if m['is_active'] else 'stored'}"):
                st.write(m)
                if m.get("metrics"):
                    st.json(m["metrics"])
                b1, b2 = st.columns(2)
                if b1.button("SET ACTIVE (local inference)", key=f"d{m['id']}"):
                    st.write(client.post(f"/api/v1/admin/models/{m['id']}/set-active", json={}))
                if b2.button("Clear active flag", key=f"r{m['id']}"):
                    st.write(client.post(f"/api/v1/admin/models/{m['id']}/rollback", json={}))

    elif page == "Evaluation":
        st.header("Evaluation")
        data = client.get("/api/v1/admin/evaluation")
        st.subheader("Tooth Detector V2")
        st.json(data.get("tooth_detector_v2") or {})
        st.subheader("ICDAS historical metrics")
        st.json(data.get("icdas_historical") or {})
        st.caption(
            (data.get("note") or "")
            + " ICDAS metrics after a future train live on the model registry. "
            "They are research/screening metrics, not a clinical accuracy claim."
        )

    elif page == "Real-world test":
        st.header("Real-world test")
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
                st.write("Quality", out.get("quality"))
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
