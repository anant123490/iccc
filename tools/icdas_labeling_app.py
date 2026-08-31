#!/usr/bin/env python3
"""ICDAS Labeling Studio — dentist-assigned 0–4 or SKIP only.

Predictions are never saved as labels. d/D is never shown as an ICDAS grade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from icdas_v2_lib import (  # noqa: E402
    CROP_POOL_COLUMNS,
    LABEL_COLUMNS,
    PROJECT_ROOT,
    crop_pool_path,
    ensure_icdas_dirs,
    labels_path,
    load_csv,
    persist_label,
)

from common import read_image  # noqa: E402

DISCLAIMER = (
    "ICDAS Labeling Studio. Grades must be assigned by a qualified dental "
    "professional. This is not a clinical diagnosis. Model predictions are never "
    "saved as ground truth."
)
GUIDE = {
    0: "Sound / no visible caries",
    1: "First visual change in enamel",
    2: "Distinct visual change in enamel",
    3: "Localized enamel breakdown, no visible dentin",
    4: "Underlying dark shadow from dentin",
}


def inject_shortcuts():
    st.components.v1.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          if (doc.body.dataset.icdasV2Keys === "1") return;
          doc.body.dataset.icdasV2Keys = "1";
          const clickByPrefix = (prefix) => {
            const buttons = Array.from(doc.querySelectorAll("button"));
            const target = buttons.find((b) => (b.innerText || "").trim().startsWith(prefix));
            if (target) target.click();
          };
          doc.addEventListener("keydown", (e) => {
            const tag = (e.target && e.target.tagName) || "";
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            const map = {
              "0": "[ 0 ]", "1": "[ 1 ]", "2": "[ 2 ]", "3": "[ 3 ]", "4": "[ 4 ]",
              "s": "[ SKIP ]", "S": "[ SKIP ]",
              "ArrowRight": "[ Next ]", "ArrowLeft": "[ Previous ]"
            };
            const prefix = map[e.key];
            if (!prefix) return;
            e.preventDefault();
            clickByPrefix(prefix);
          }, true);
        })();
        </script>
        """,
        height=0,
    )


def merged_queue() -> pd.DataFrame:
    ensure_icdas_dirs()
    pool = load_csv(crop_pool_path(), CROP_POOL_COLUMNS)
    labels = load_csv(labels_path(), LABEL_COLUMNS)
    if pool.empty:
        return pool
    if not labels.empty:
        pool = pool.merge(
            labels[["sample_id", "icdas_grade", "status"]].rename(
                columns={"icdas_grade": "saved_grade", "status": "saved_status"}
            ),
            on="sample_id",
            how="left",
        )
    else:
        pool["saved_grade"] = ""
        pool["saved_status"] = ""
    return pool


def counts(catalog: pd.DataFrame) -> dict:
    labels = load_csv(labels_path(), LABEL_COLUMNS)
    out = {
        "pool": int(len(catalog)),
        "labelled": 0,
        "skipped": 0,
        "unlabelled": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    if labels.empty:
        out["unlabelled"] = out["pool"]
        return out
    out["labelled"] = int((labels["status"] == "labelled").sum())
    out["skipped"] = int((labels["status"] == "skipped").sum())
    out["unlabelled"] = max(0, out["pool"] - out["labelled"] - out["skipped"])
    for g in "01234":
        out[g] = int(((labels["status"] == "labelled") & (labels["icdas_grade"] == g)).sum())
    return out


def save_and_advance(row, grade: str, n: int):
    persist_label(
        {
            "sample_id": row["sample_id"],
            "source_type": row.get("source_type", ""),
            "source_image": row.get("source_image", ""),
            "crop_path": row.get("crop_path", ""),
            "bbox_x1": row.get("bbox_x1", ""),
            "bbox_y1": row.get("bbox_y1", ""),
            "bbox_x2": row.get("bbox_x2", ""),
            "bbox_y2": row.get("bbox_y2", ""),
            "icdas_grade": grade,
        }
    )
    st.session_state.index = min(st.session_state.index + 1, max(n - 1, 0))
    st.rerun()


def main():
    st.set_page_config(page_title="ICDAS Labeling Studio", layout="wide")
    inject_shortcuts()
    st.title("ICDAS LABELING STUDIO")
    st.warning(DISCLAIMER)

    catalog = merged_queue()
    c = counts(catalog)
    st.sidebar.header("ICDAS LABELING PROGRESS")
    st.sidebar.metric("Total tooth crops", c["pool"])
    st.sidebar.metric("Labelled", c["labelled"])
    st.sidebar.metric("Remaining", c["unlabelled"])
    st.sidebar.metric("Skipped", c["skipped"])
    st.sidebar.caption("Class distribution (labelled only; not balanced)")
    total_l = max(c["labelled"], 1)
    for g in "01234":
        st.sidebar.write(f"ICDAS {g}: {c[g]}")
        st.sidebar.progress(c[g] / total_l if c["labelled"] else 0.0)

    if catalog.empty:
        st.error(
            "No tooth crops in the labeling pool.\n\n"
            "There is no reliable whole-tooth detector in this repo. "
            "Do not crop lesion `d`/`D` boxes as ICDAS samples.\n\n"
            "When crops exist, put them in `data/icdas/labeling_v2/crops/` and run:\n"
            "`python tools/register_icdas_crops.py`"
        )
        st.stop()

    review = st.sidebar.checkbox("Review already labelled / skipped", value=False)
    saved = catalog.get("saved_status", pd.Series([""] * len(catalog)))
    if review:
        queue = catalog
    else:
        queue = catalog.loc[~saved.isin(["labelled", "skipped"])].reset_index(drop=True)
    if queue.empty:
        st.success("Nothing left in this view. Enable review to edit.")
        st.stop()

    if "index" not in st.session_state:
        st.session_state.index = 0
    st.session_state.index = max(0, min(st.session_state.index, len(queue) - 1))
    row = queue.iloc[st.session_state.index]
    crop_path = Path(str(row["crop_path"]))
    pred = str(row.get("saved_grade") or "Not labelled")
    if str(row.get("saved_status") or "") == "skipped":
        pred = "SKIP"

    left, right = st.columns([3, 2])
    with left:
        st.caption(f"Image: {st.session_state.index + 1} / {len(queue)}")
        st.write(f"**Source:** `{row.get('source_type') or 'unknown'}`")
        st.write(f"**sample_id:** `{row['sample_id']}`")
        st.write(f"**Prediction:** {pred} (saved label, not a model grade)")
        img = read_image(crop_path) if crop_path.exists() else None
        if img is None:
            st.error("Missing or unreadable crop. Use SKIP. Source files are not deleted.")
        else:
            st.image(img[:, :, ::-1], caption=crop_path.name, use_container_width=True)
    with right:
        for g, t in GUIDE.items():
            st.markdown(f"**ICDAS {g}:** {t}")
        cols = st.columns(6)
        labels_btn = ["[ 0 ]", "[ 1 ]", "[ 2 ]", "[ 3 ]", "[ 4 ]", "[ SKIP ]"]
        grades = ["0", "1", "2", "3", "4", "SKIP"]
        for i, (lab, g) in enumerate(zip(labels_btn, grades)):
            if cols[i].button(lab, use_container_width=True, key=f"g{g}"):
                save_and_advance(row, g, len(queue))
        n1, n2 = st.columns(2)
        if n1.button("[ Previous ]", use_container_width=True):
            st.session_state.index = max(0, st.session_state.index - 1)
            st.rerun()
        if n2.button("[ Next ]", use_container_width=True):
            st.session_state.index = min(len(queue) - 1, st.session_state.index + 1)
            st.rerun()
        st.caption("Keys: 0–4, S skip, arrows previous/next. Auto-saves to data/icdas/labeling_v2/manifest/icdas_labels.csv")
        st.caption(str(labels_path().relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
