"""One-shot repository layout moves. Does not delete sources; shutil.move only."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG: list[str] = []


def ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        LOG.append(f"SKIP missing {src.relative_to(ROOT)}")
        return
    ensure(dst.parent)
    if dst.exists():
        LOG.append(f"SKIP dest exists {dst.relative_to(ROOT)}")
        return
    shutil.move(str(src), str(dst))
    LOG.append(f"MOVED {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> None:
    for p in [
        ROOT / "archive" / "historical",
        ROOT / "archive" / "experiments",
        ROOT / "archive" / "obsolete",
        ROOT / "archive" / "out_of_scope" / "fdi",
        ROOT / "archive" / "review_required",
        ROOT / "data" / "detection" / "raw_images",
        ROOT / "data" / "detection" / "annotations",
        ROOT / "data" / "detection" / "train",
        ROOT / "data" / "detection" / "val",
        ROOT / "data" / "detection" / "test",
        ROOT / "data" / "detection" / "batches" / "batch01",
        ROOT / "data" / "tooth_crops" / "reviewed",
        ROOT / "data" / "icdas" / "raw",
        ROOT / "data" / "icdas" / "images",
        ROOT / "data" / "icdas" / "annotations",
        ROOT / "models" / "detection" / "pretrained",
        ROOT / "models" / "icdas" / "current",
        ROOT / "models" / "icdas" / "historical" / "stale_ordinal_4output",
        ROOT / "ml" / "detection" / "train",
        ROOT / "ml" / "detection" / "inference",
        ROOT / "ml" / "detection" / "evaluation",
        ROOT / "ml" / "icdas" / "train",
        ROOT / "ml" / "icdas" / "inference",
        ROOT / "ml" / "icdas" / "evaluation",
        ROOT / "tests",
        ROOT / "reports" / "historical",
    ]:
        ensure(p)

    for split in ("train", "val", "test"):
        for g in "01234":
            ensure(ROOT / "data" / "icdas" / split / g)

    # Detection model
    move(
        ROOT / "models" / "tooth_detector_batch01",
        ROOT / "models" / "detection" / "tooth_detector_batch01",
    )
    move(ROOT / "yolo11n.pt", ROOT / "models" / "detection" / "pretrained" / "yolo11n.pt")
    move(
        ROOT / "models" / "archives",
        ROOT / "archive" / "experiments" / "yolo_run_archives",
    )
    move(
        ROOT / "models" / "tooth_detector_batch01_run1_adamw_lr0.01_collapsed",
        ROOT / "archive" / "experiments" / "tooth_detector_batch01_run1_adamw_lr0.01_collapsed",
    )

    # ICDAS keras (do not replace contents)
    stale = ROOT / "models" / "icdas" / "historical" / "stale_ordinal_4output"
    move(ROOT / "models" / "deploy.keras", stale / "deploy.keras")
    move(ROOT / "models" / "best.keras", stale / "best.keras")

    for name in (
        "icdas_mobilenet_cbam",
        "icdas_mobilenet_cbam_ordinal",
        "icdas_mobilenet_cbam_5class_v2",
        "icdas_mobilenet_cbam_5class_v3",
        "icdas_mobilenet_cbam_5class_weighted",
        "tfjs_model",
    ):
        move(ROOT / "models" / name, ROOT / "models" / "icdas" / "historical" / name)
    move(
        ROOT / "models" / "export_report.json",
        ROOT / "models" / "icdas" / "historical" / "export_report.json",
    )

    # Crops / predictions
    move(ROOT / "cropped_teeth", ROOT / "data" / "tooth_crops" / "generated")
    if (ROOT / "predictions").exists():
        move(ROOT / "predictions", ROOT / "data" / "tooth_crops" / "detector_predictions")

    # ICDAS dataset
    ds = ROOT / "dataset"
    icdas = ROOT / "data" / "icdas"
    move(ds / "annotations.csv", icdas / "annotations" / "annotations.csv")
    move(ds / "whatsapp_manifest.json", icdas / "annotations" / "whatsapp_manifest.json")
    move(ds / "README.md", icdas / "annotations" / "LEGACY_dataset_README.md")
    for sub in ("train", "val", "test", "excluded", "raw"):
        if (ds / sub).exists():
            # train/val/test may already exist as empty class dirs we created
            src = ds / sub
            dst = icdas / sub
            if sub in ("train", "val", "test") and dst.exists():
                # merge: move children then remove empty src
                for child in src.iterdir():
                    target = dst / child.name
                    if target.exists():
                        if child.is_dir():
                            for gchild in child.iterdir():
                                move(gchild, target / gchild.name)
                        else:
                            LOG.append(f"SKIP merge dest exists {target}")
                    else:
                        move(child, target)
                try:
                    src.rmdir()
                    LOG.append(f"REMOVED empty dir {src.relative_to(ROOT)}")
                except OSError:
                    LOG.append(f"LEFT non-empty {src.relative_to(ROOT)}")
            else:
                move(src, dst)

    move(ROOT / "data_icdas", icdas / "labeling_v2")
    move(ROOT / "labels", icdas / "annotations" / "labeling_studio")

    # Apps
    move(ROOT / "backend", ROOT / "app" / "backend")
    move(ROOT / "fronted", ROOT / "app" / "frontend")

    # Ultralytics runs
    move(ROOT / "runs", ROOT / "archive" / "experiments" / "ultralytics_runs")

    # bak
    bak_dir = ROOT / "archive" / "historical" / "bak_pre_caries_pipeline"
    ensure(bak_dir)
    for src, name in [
        (
            ROOT / "app" / "backend" / "app" / "main.py.bak_pre_caries_pipeline",
            "main.py.bak_pre_caries_pipeline",
        ),
        (
            ROOT / "app" / "backend" / "app" / "schemas.py.bak_pre_caries_pipeline",
            "schemas.py.bak_pre_caries_pipeline",
        ),
        (
            ROOT / "app" / "backend" / "app" / "groq_service.py.bak_pre_caries_pipeline",
            "groq_service.py.bak_pre_caries_pipeline",
        ),
        (
            ROOT / "app" / "frontend" / "streamlit_app.py.bak_pre_caries_pipeline",
            "streamlit_app.py.bak_pre_caries_pipeline",
        ),
    ]:
        move(src, bak_dir / name)

    # FDI docs
    fdi = ROOT / "archive" / "out_of_scope" / "fdi"
    for name in (
        "FDI_RGB_DATASET_SEARCH.md",
        "FDI_RGB_FINAL_VERIFICATION.md",
        "FDTooth_ACQUISITION_REPORT.md",
        "RGB_FDI_DATASET_SEARCH_STAGE2D6.md",
        "STAGE2E_RGB_FDI_FEASIBILITY_REPORT.md",
    ):
        move(ROOT / name, fdi / name)
    move(
        ROOT / "reports" / "RGB_TOOTH_FDI_PUBLIC_DATASET_RANKING.md",
        fdi / "RGB_TOOTH_FDI_PUBLIC_DATASET_RANKING.md",
    )
    for name in (
        "stage2e_rgb_fdi_feasibility.json",
        "stage2d6_rgb_fdi_dataset_search.json",
        "stage2d5_fdtooth_acquisition.json",
        "stage2d4_fdi_final_verification.json",
        "stage2d3_fdi_rgb_search.json",
        "stage2d2_fdi_candidate_verification.md",
        "stage2d1_architecture_decision.md",
    ):
        move(ROOT / "reports" / name, fdi / name)

    mapping = ROOT / "fdi_detection_dataset" / "annotations" / "fdi_mapping"
    move(mapping, fdi / "fdi_mapping")

    # Root historical reports
    hist = ROOT / "reports" / "historical"
    for name in (
        "STAGE2C_ZENODO_DETECTION_REPORT.md",
        "STAGE3A_DETECTION_DATASET_REPORT.md",
        "STAGE3B_ANNOTATION_PROJECT_REPORT.md",
        "STAGE3C_ANNOTATION_REPORT.md",
        "STAGE3C_SEED_QC_REPORT.md",
        "STAGE3C_SEED_TRAINING_REPORT.md",
        "README_STAGE3B.md",
        "DATASET_REPORT.md",
        "GLOBAL_DATASET_SEARCH.md",
    ):
        move(ROOT / name, hist / name)

    docs_keep = ROOT / "docs"
    for name in (
        "STAGE3C_MANUAL_ANNOTATION.md",
        "TOOTH_ANNOTATION_GUIDELINES.md",
        "ANNOTATION_QC_CHECKLIST.md",
    ):
        move(ROOT / name, docs_keep / name)

    # empty labels.csv directory
    lc = ROOT / "labels.csv"
    if lc.exists() and lc.is_dir():
        move(lc, ROOT / "archive" / "review_required" / "labels.csv_empty_directory")

    leftover = ROOT / "dataset"
    if leftover.exists():
        remaining = list(leftover.rglob("*"))
        if not remaining:
            leftover.rmdir()
            LOG.append("REMOVED empty dataset/")
        else:
            move(leftover, ROOT / "archive" / "review_required" / "dataset_leftover")

    log_path = ROOT / "archive" / "review_required" / "phase4_move_log.txt"
    log_path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
