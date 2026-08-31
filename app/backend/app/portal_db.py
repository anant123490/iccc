"""Portal SQLAlchemy models. Does not replace PredictionRecord."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    visits: Mapped[list["Visit"]] = relationship(back_populates="patient")


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    visit_date: Mapped[str] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_only: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    patient: Mapped[Patient] = relationship(back_populates="visits")
    images: Mapped[list["UploadedImage"]] = relationship(back_populates="visit")
    report: Mapped["ClinicalReport | None"] = relationship(back_populates="visit", uselist=False)


class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"), index=True)
    filename: Mapped[str] = mapped_column(String(256))
    path: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    quality_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    visit: Mapped[Visit] = relationship(back_populates="images")
    detections: Mapped[list["ToothDetection"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ToothDetection(Base):
    __tablename__ = "tooth_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("uploaded_images.id"), index=True)
    index_in_image: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[int] = mapped_column(Integer)
    y1: Mapped[int] = mapped_column(Integer)
    x2: Mapped[int] = mapped_column(Integer)
    y2: Mapped[int] = mapped_column(Integer)
    image: Mapped[UploadedImage] = relationship(back_populates="detections")
    crop: Mapped["ToothCrop | None"] = relationship(
        back_populates="detection", uselist=False, cascade="all, delete-orphan"
    )


class ToothCrop(Base):
    __tablename__ = "tooth_crops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("tooth_detections.id"), index=True)
    path: Mapped[str] = mapped_column(String(512))
    detection: Mapped[ToothDetection] = relationship(back_populates="crop")
    prediction: Mapped["ICDASPrediction | None"] = relationship(
        back_populates="crop", uselist=False, cascade="all, delete-orphan"
    )


class ICDASPrediction(Base):
    __tablename__ = "icdas_predictions"
    __table_args__ = (
        CheckConstraint("grade >= 0 AND grade <= 4", name="ck_portal_icdas_0_4"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    crop_id: Mapped[int] = mapped_column(ForeignKey("tooth_crops.id"), index=True)
    grade: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    probabilities_json: Mapped[str] = mapped_column(Text)
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    crop: Mapped[ToothCrop] = relationship(back_populates="prediction")


class ClinicalReport(Base):
    __tablename__ = "clinical_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"), unique=True, index=True)
    language: Mapped[str] = mapped_column(String(16), default="en")
    structured_json: Mapped[str] = mapped_column(Text)
    screening_summary: Mapped[str] = mapped_column(Text)
    tooth_findings: Mapped[str] = mapped_column(Text)
    oral_health: Mapped[str] = mapped_column(Text)
    recommendations: Mapped[str] = mapped_column(Text)
    follow_up: Mapped[str] = mapped_column(Text)
    html_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    visit: Mapped[Visit] = relationship(back_populates="report")


class TrainingImage(Base):
    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(256))
    path: Mapped[str] = mapped_column(String(512))
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    n_crops: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    phash: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    duplicate_status: Mapped[str] = mapped_column(String(32), default="UNIQUE", index=True)
    canonical_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    boxes_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    exclude_from_dataset: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    labels: Mapped[list["TrainingLabel"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class TrainingLabel(Base):
    __tablename__ = "training_labels"
    __table_args__ = (
        CheckConstraint(
            "grade IS NULL OR (grade >= 0 AND grade <= 4)",
            name="ck_train_label_icdas_0_4",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("training_images.id"), index=True)
    crop_path: Mapped[str] = mapped_column(String(512))
    grade: Mapped[int | None] = mapped_column(Integer, nullable=True)
    x1: Mapped[int] = mapped_column(Integer, default=0)
    y1: Mapped[int] = mapped_column(Integer, default=0)
    x2: Mapped[int] = mapped_column(Integer, default=0)
    y2: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    index_in_image: Mapped[int] = mapped_column(Integer, default=0)
    crop_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    crop_phash: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    crop_duplicate_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    duplicate_of_label_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    box_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    image: Mapped[TrainingImage] = relationship(back_populates="labels")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="IN_PROGRESS")
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_valid: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    version_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    statistics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    split_seed: Mapped[int] = mapped_column(Integer, default=42)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(String(512))
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    message: Mapped[str] = mapped_column(Text, default="")
    log_text: Mapped[str] = mapped_column(Text, default="")
    dataset_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
