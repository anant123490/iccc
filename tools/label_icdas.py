#!/usr/bin/env python3
"""Human ICDAS 0–4 labeling UI for tooth crops.

ICDAS severity labels should be assigned or verified by a qualified dental
professional. This tool does not provide a clinical diagnosis.

Never auto-assigns ICDAS from a model or from public-dataset class names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import (  # noqa: E402
    LABEL_CSV_COLUMNS,
    PROJECT_ROOT,
    ensure_pipeline_dirs,
    load_csv,
    project_path,
    read_image,
    save_csv,
)

ICDAS_GUIDE = {
    0: "Sound tooth / no visible caries.",
    1: "First visual change in enamel.",
    2: "Distinct visual change in enamel.",
    3: "Localized enamel breakdown without visible dentin.",
    4: "Underlying dark shadow from dentin.",
}

DISCLAIMER = (
    "ICDAS severity labels should be assigned or verified by a qualified "
    "dental professional. This tool does not provide a clinical diagnosis."
)


def crops_dir() -> Path:
    return project_path("cropped_teeth", "images")


def crops_csv_path() -> Path:
    return project_path("cropped_teeth", "crops.csv")


def labels_csv_path() -> Path:
    return project_path("labels", "labels.csv")


def load_catalog() -> pd.DataFrame:
    ensure_pipeline_dirs()
    crops = load_csv(crops_csv_path(), ["crop_id", "filename", "source_image", "annotation_id", "annotation_class"])
    image_files = sorted(p for p in crops_dir().iterdir() if p.is_file())
    if crops.empty and image_files:
        crops = pd.DataFrame(
            {
                "crop_id": [p.stem for p in image_files],
                "filename": [p.name for p in image_files],
                "source_image": [""] * len(image_files),
            }
        )
    if "filename" not in crops.columns:
        crops["filename"] = crops.get("crop_id", "") + ".jpg"
    crops = crops.drop_duplicates(subset=["crop_id"], keep="last").reset_index(drop=True)
    labels = load_csv(labels_csv_path(), LABEL_CSV_COLUMNS)
    if labels.empty:
        labels = pd.DataFrame(columns=LABEL_CSV_COLUMNS)
    merged = crops.merge(labels[["crop_id", "icdas_grade"]], on="crop_id", how="left")
    return merged


def save_label(crop_id: str, filename: str, source_image: str, grade: int) -> None:
    path = labels_csv_path()
    df = load_csv(path, LABEL_CSV_COLUMNS)
    if df.empty:
        df = pd.DataFrame(columns=LABEL_CSV_COLUMNS)
    mask = df["crop_id"].astype(str) == str(crop_id)
    row = {
        "crop_id": crop_id,
        "filename": filename,
        "source_image": source_image,
        "icdas_grade": int(grade),
    }
    if mask.any():
        df.loc[mask, list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(path, df, LABEL_CSV_COLUMNS)


def delete_label(crop_id: str) -> dict | None:
    path = labels_csv_path()
    df = load_csv(path, LABEL_CSV_COLUMNS)
    if df.empty:
        return None
    mask = df["crop_id"].astype(str) == str(crop_id)
    if not mask.any():
        return None
    removed = df.loc[mask].iloc[0].to_dict()
    df = df.loc[~mask]
    save_csv(path, df, LABEL_CSV_COLUMNS)
    return removed


def inject_shortcuts():
    st.components.v1.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          if (doc.body.dataset.icdasKeys === "1") return;
          doc.body.dataset.icdasKeys = "1";
          const clickByPrefix = (prefix) => {
            const buttons = Array.from(doc.querySelectorAll("button"));
            const target = buttons.find((b) => (b.innerText || "").trim().startsWith(prefix));
            if (target) target.click();
          };
          doc.addEventListener("keydown", (e) => {
            const tag = (e.target && e.target.tagName) || "";
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            const map = {
              "0": "[0]",
              "1": "[1]",
              "2": "[2]",
              "3": "[3]",
              "4": "[4]",
              "ArrowRight": "[Next]",
              "ArrowLeft": "[Previous]",
              "s": "[Skip]",
              "S": "[Skip]",
              "u": "[Undo]",
              "U": "[Undo]",
              "z": "[Undo]"
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


def init_state(n: int):
    if "index" not in st.session_state:
        st.session_state.index = 0
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    if "show_labeled" not in st.session_state:
        st.session_state.show_labeled = False
    st.session_state.index = max(0, min(st.session_state.index, max(n - 1, 0)))


def goto(delta: int, n: int):
    if n <= 0:
        return
    st.session_state.index = (st.session_state.index + delta) % n


def main():
    st.set_page_config(page_title="ICDAS labeling", layout="wide")
    inject_shortcuts()
    ensure_pipeline_dirs()

    st.title("ICDAS 0–4 tooth labeling")
    st.warning(DISCLAIMER)
    st.caption("These are labeling guidelines only. They are not a clinical diagnosis.")

    catalog = load_catalog()
    if catalog.empty:
        st.error(
            f"No crops found. Run crop_teeth.py first.\n\nExpected images in `{crops_dir()}` "
            f"and metadata in `{crops_csv_path()}`."
        )
        st.stop()

    labeled_mask = catalog["icdas_grade"].notna()
    unlabeled = catalog.loc[~labeled_mask].reset_index(drop=True)
    labeled = catalog.loc[labeled_mask].reset_index(drop=True)

    st.sidebar.header("Progress")
    st.sidebar.metric("Total crops", len(catalog))
    st.sidebar.metric("Labeled", int(labeled_mask.sum()))
    st.sidebar.metric("Remaining", int((~labeled_mask).sum()))
    if len(catalog):
        st.sidebar.progress(float(labeled_mask.sum()) / len(catalog))

    review = st.sidebar.checkbox("Review / edit already labeled crops", value=st.session_state.get("show_labeled", False))
    st.session_state.show_labeled = review
    queue = labeled if review else unlabeled
    if queue.empty:
        queue = catalog if review else unlabeled
    if queue.empty:
        st.success("All crops in this view are labeled. Enable review mode to edit.")
        st.stop()

    init_state(len(queue))
    row = queue.iloc[st.session_state.index]
    crop_id = str(row["crop_id"])
    filename = str(row.get("filename") or f"{crop_id}.jpg")
    source_image = str(row.get("source_image") or "")
    image_path = crops_dir() / filename
    if not image_path.exists():
        matches = list(crops_dir().glob(f"{crop_id}.*"))
        image_path = matches[0] if matches else image_path

    col_img, col_ctrl = st.columns([3, 2])
    with col_img:
        st.subheader(f"Crop {st.session_state.index + 1} / {len(queue)}")
        st.write(f"**crop_id:** `{crop_id}`")
        st.write(f"**filename:** `{filename}`")
        st.write(f"**source_image:** `{source_image}`")
        if "annotation_class" in row and pd.notna(row["annotation_class"]):
            st.info(
                f"Public-dataset region class: `{row['annotation_class']}`. "
                "This is not an ICDAS grade."
            )
        image = read_image(image_path) if image_path.exists() else None
        if image is None:
            st.error("Corrupted or missing crop image. Use Skip to continue. Source files were not deleted.")
        else:
            rgb = image[:, :, ::-1]
            st.image(rgb, caption=filename, use_container_width=True)

    with col_ctrl:
        st.markdown("### ICDAS guidelines (labeling only)")
        for grade, text in ICDAS_GUIDE.items():
            st.markdown(f"**ICDAS {grade}:** {text}")

        current = row.get("icdas_grade")
        if pd.notna(current):
            st.success(f"Current saved label: ICDAS {int(current)}")

        bcols = st.columns(5)
        for grade in range(5):
            if bcols[grade].button(f"[{grade}] ICDAS {grade}", use_container_width=True, key=f"g{grade}"):
                previous = None
                if pd.notna(current):
                    previous = {
                        "crop_id": crop_id,
                        "filename": filename,
                        "source_image": source_image,
                        "icdas_grade": int(current),
                    }
                save_label(crop_id, filename, source_image, grade)
                st.session_state.undo_stack.append(
                    {"action": "set", "crop_id": crop_id, "previous": previous, "new": grade}
                )
                goto(1, len(queue))
                st.rerun()

        nav1, nav2, nav3, nav4 = st.columns(4)
        if nav1.button("[Previous]", use_container_width=True):
            goto(-1, len(queue))
            st.rerun()
        if nav2.button("[Next]", use_container_width=True):
            goto(1, len(queue))
            st.rerun()
        if nav3.button("[Skip]", use_container_width=True):
            goto(1, len(queue))
            st.rerun()
        if nav4.button("[Undo]", use_container_width=True):
            if not st.session_state.undo_stack:
                st.warning("Nothing to undo.")
            else:
                item = st.session_state.undo_stack.pop()
                if item.get("previous") is None:
                    delete_label(item["crop_id"])
                else:
                    prev = item["previous"]
                    save_label(prev["crop_id"], prev["filename"], prev["source_image"], int(prev["icdas_grade"]))
                st.rerun()

        st.caption("Shortcuts: 0–4 label, ← previous, → next, S skip, U undo.")
        st.caption(f"Labels save immediately to `{labels_csv_path().relative_to(PROJECT_ROOT)}`.")


if __name__ == "__main__":
    main()
