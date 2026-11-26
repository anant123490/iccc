"""Explicit admin assignment of detector photos to anonymous patient groups."""

from __future__ import annotations

import json
import re
import time
import csv
import hashlib
import ntpath
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from .portal_db import Patient, TrainingAuditEvent, TrainingImage

_PATIENT_ID = re.compile(r"^PAT-(\d+)$")
_PHOTO_ID = re.compile(r"^(?:PHOTO-)?(\d+)$", re.IGNORECASE)
_TERMINAL_UPLOAD_STATUSES = {"COMPLETED", "DUPLICATE"}


def photo_id_for(training_image_id: int) -> str:
    """Return the copyable, stable Photo ID shown in the Admin Portal."""
    return f"PHOTO-{int(training_image_id):03d}"


def parse_photo_id_text(value: str | None) -> dict:
    """Parse comma-, whitespace-, or newline-separated Photo IDs."""
    tokens = [token for token in re.split(r"[\s,]+", str(value or "").strip()) if token]
    photo_ids: list[int] = []
    invalid: list[str] = []
    repeated: list[str] = []
    seen: set[int] = set()
    for token in tokens:
        match = _PHOTO_ID.fullmatch(token)
        if match is None or int(match.group(1)) < 1:
            invalid.append(token)
            continue
        photo_id = int(match.group(1))
        if photo_id in seen:
            repeated.append(token)
            continue
        seen.add(photo_id)
        photo_ids.append(photo_id)
    return {
        "tokens": tokens,
        "training_image_ids": photo_ids,
        "invalid_photo_ids": invalid,
        "duplicate_photo_ids": repeated,
    }


def resolve_photo_ids(db: Session, value: str | None) -> dict:
    """Resolve pasted Photo IDs without ignoring malformed or missing IDs."""
    parsed = parse_photo_id_text(value)
    if not parsed["tokens"]:
        raise ValueError("Enter at least one Photo ID.")
    if parsed["invalid_photo_ids"]:
        raise ValueError("Invalid Photo ID(s): " + ", ".join(parsed["invalid_photo_ids"][:20]))
    if parsed["duplicate_photo_ids"]:
        raise ValueError("Photo ID(s) were entered more than once: " + ", ".join(parsed["duplicate_photo_ids"][:20]))
    if len(parsed["training_image_ids"]) > 1000:
        raise ValueError("Assign at most 1,000 photos per action.")
    images = (
        db.query(TrainingImage)
        .filter(TrainingImage.id.in_(parsed["training_image_ids"]))
        .all()
    )
    by_id = {int(image.id): image for image in images}
    missing_ids = [photo_id for photo_id in parsed["training_image_ids"] if photo_id not in by_id]
    if missing_ids:
        raise ValueError("Photo ID(s) do not exist: " + ", ".join(photo_id_for(value) for value in missing_ids[:20]))
    return {**parsed, "images_by_id": by_id}


def next_anonymous_patient_id(db: Session) -> str:
    """Read the next sequential PAT ID for preview; does not create a row."""
    highest = int(
        db.query(Patient.anonymous_sequence)
        .filter(Patient.anonymous_sequence.is_not(None))
        .order_by(Patient.anonymous_sequence.desc())
        .limit(1)
        .scalar()
        or 0
    )
    for public_id, in db.query(Patient.public_id).all():
        match = _PATIENT_ID.fullmatch(str(public_id or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"PAT-{highest + 1:03d}"


def _next_anonymous_patient(db: Session) -> Patient:
    """Add the next PAT-NNN row; the unique sequence index arbitrates races."""
    public_id = next_anonymous_patient_id(db)
    sequence = int(public_id.removeprefix("PAT-"))
    patient = Patient(
        public_id=public_id,
        name="Anonymous Patient",
        age=None,
        gender=None,
        phone=None,
        is_anonymous=True,
        anonymous_sequence=sequence,
    )
    db.add(patient)
    db.flush()
    return patient


def create_anonymous_patient(db: Session) -> dict:
    """Create an anonymous group with collision-safe sequential public IDs."""
    for attempt in range(5):
        try:
            patient = _next_anonymous_patient(db)
            db.commit()
            db.refresh(patient)
            return {"id": patient.id, "public_id": patient.public_id, "is_anonymous": True}
        except IntegrityError:
            db.rollback()
        except OperationalError as exc:
            db.rollback()
            if "locked" not in str(exc).casefold() and "busy" not in str(exc).casefold():
                raise
        if attempt < 4:
            time.sleep(0.02 * (attempt + 1))
    raise RuntimeError("Could not allocate an anonymous patient ID after concurrent updates. Retry the operation.")


def anonymous_patient_inventory(db: Session) -> dict:
    """Return processed upload rows and anonymous groups for the admin UI."""
    images = (
        db.query(TrainingImage)
        .filter(TrainingImage.is_active.is_not(False))
        .order_by(TrainingImage.upload_timestamp.desc(), TrainingImage.id.desc())
        .all()
    )
    patients = (
        db.query(Patient)
        .filter(Patient.is_anonymous.is_(True))
        .order_by(Patient.anonymous_sequence.asc())
        .all()
    )
    photos = []
    photo_counts: dict[int, int] = {}
    for image in images:
        patient = image.patient_record
        if patient and patient.is_anonymous:
            photo_counts[patient.id] = photo_counts.get(patient.id, 0) + 1
        labels = [label for label in image.labels if label.active is not False]
        photos.append(
            {
                "training_image_id": int(image.id),
                "photo_id": photo_id_for(image.id),
                "image_id": image.image_id,
                "filename": image.original_filename or image.filename,
                "content_hash": image.content_hash,
                "processing_status": image.processing_status,
                "duplicate_status": image.duplicate_status,
                "patient_id": image.patient_id or (patient.public_id if patient else None),
                "patient_link_conflict": bool(
                    image.patient_record_id is not None
                    and (patient is None or patient.public_id != image.patient_id)
                ),
                "patient_is_anonymous": bool(patient and patient.is_anonymous),
                "crop_count": len(labels),
                "labeled_crop_count": sum(
                    1 for label in labels if label.grade is not None and label.annotation_status == "LABELED"
                ),
                "verified_crop_count": sum(
                    1 for label in labels if image.boxes_verified and label.box_verified
                ),
                "assignment_ready": image.processing_status in _TERMINAL_UPLOAD_STATUSES,
                "split_name": image.split_name or "UNASSIGNED",
            }
        )
    groups = [
        {
            "public_id": patient.public_id,
            "sequence": patient.anonymous_sequence,
            "photo_count": photo_counts.get(patient.id, 0),
        }
        for patient in patients
    ]
    return {
        "photos": photos,
        "anonymous_patients": groups,
        "ready_count": sum(1 for row in photos if row["assignment_ready"]),
        "unassigned_count": sum(1 for row in photos if not row["patient_id"]),
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _manifest_path_key(value: str | None) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    return ntpath.normcase(ntpath.normpath(value.replace("/", "\\")))


def _manifest_filename_key(value: str | None) -> str:
    return str(value or "").strip().casefold()


def _valid_sha256(value: str | None) -> str | None:
    value = str(value or "").strip().casefold()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def _suggestion_key(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:40]
    return f"{kind}:{digest}"


def _read_manifest_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def patient_photo_review_candidates(db: Session) -> list[dict]:
    """Build review candidates; only exact-hash, unambiguous references are actionable."""
    root = _project_root()
    source_path = root / "data" / "icdas" / "labeling_v2" / "source_manifest.csv"
    metadata_path = root / "fdi_detection_dataset" / "metadata" / "image_manifest.csv"
    source_rows = _read_manifest_rows(source_path)
    metadata_rows = _read_manifest_rows(metadata_path)
    source_by_hash: dict[str, list[dict]] = {}
    for row in source_rows:
        digest = _valid_sha256(row.get("sha256"))
        if digest:
            source_by_hash.setdefault(digest, []).append(row)

    metadata_by_path: dict[str, list[dict]] = {}
    metadata_by_name: dict[str, list[dict]] = {}
    for row in metadata_rows:
        path_key = _manifest_path_key(row.get("source_path"))
        if path_key:
            metadata_by_path.setdefault(path_key, []).append(row)
        for name in (row.get("original_filename"), row.get("current_filename")):
            name_key = _manifest_filename_key(name)
            if name_key:
                metadata_by_name.setdefault(name_key, []).append(row)

    images = (
        db.query(TrainingImage)
        .filter(TrainingImage.is_active.is_not(False))
        .order_by(TrainingImage.id.asc())
        .all()
    )
    matched: list[dict] = []
    for image in images:
        digest = _valid_sha256(image.content_hash)
        source_matches = source_by_hash.get(digest or "", [])
        source = source_matches[0] if source_matches else {}
        matched_filename = str(source.get("filename") or "").strip()
        references: set[str] = set()
        metadata_match_methods: set[str] = set()
        metadata_match_count = 0
        for source_record in source_matches:
            path_key = _manifest_path_key(source_record.get("original_path"))
            metadata_matches = metadata_by_path.get(path_key, []) if path_key else []
            if metadata_matches:
                metadata_match_methods.add("exact_source_path")
            else:
                name_key = _manifest_filename_key(source_record.get("filename"))
                metadata_matches = metadata_by_name.get(name_key, []) if name_key else []
                if metadata_matches:
                    metadata_match_methods.add("exact_filename_crosswalk")
            metadata_match_count += len(metadata_matches)
            for metadata in metadata_matches:
                reference = str(metadata.get("patient_identifier_if_available") or "").strip()
                if reference:
                    references.add(reference)

        reference = next(iter(references)) if len(references) == 1 else ""
        if not digest:
            source_status = "hash_missing_or_invalid"
        elif not source_matches:
            source_status = "hash_not_found_in_source_manifest"
        elif len(references) > 1:
            source_status = "ambiguous_patient_reference"
        elif reference:
            source_status = "verified_patient_reference"
        elif metadata_match_count:
            source_status = "source_match_without_patient_reference"
        else:
            source_status = "source_match_without_metadata_crosswalk"

        filename = matched_filename or image.original_filename or image.filename
        stem = Path(filename).stem
        prefix_match = re.split(r"_(?=\d{10,}_)", stem, maxsplit=1)
        filename_prefix = prefix_match[0] if len(prefix_match) > 1 else ""
        patient = image.patient_record
        matched.append(
            {
                "training_image_id": int(image.id),
                "photo_id": photo_id_for(image.id),
                "filename": filename or image.original_filename or image.filename,
                "source_patient_reference": reference or None,
                "filename_prefix": filename_prefix or None,
                "current_patient_id": image.patient_id or (patient.public_id if patient else None),
                "source_type": str(source.get("source_type") or "unknown"),
                "source_metadata_status": source_status,
                "hash_matched": bool(source_matches),
                "reference_match_method": ", ".join(sorted(metadata_match_methods)) or None,
                "possible_patient_references": sorted(references),
                "assignment_ready": image.processing_status in _TERMINAL_UPLOAD_STATUSES,
                "assignment_conflict": False,
                "auto_group_eligible": source_status == "verified_patient_reference",
            }
        )

    group_counts: dict[str, int] = {}
    for row in matched:
        reference = row["source_patient_reference"]
        if row["auto_group_eligible"] and reference:
            key = _suggestion_key("source-ref", reference)
            row["suggestion_key"] = key
            group_counts[key] = group_counts.get(key, 0) + 1
        elif row["filename_prefix"]:
            # Filename repetition is useful for manual review, never an assignment key.
            key = _suggestion_key("filename-clue", row["filename_prefix"])
            row["suggestion_key"] = key
            row["auto_group_eligible"] = False
            group_counts[key] = group_counts.get(key, 0) + 1
        else:
            row["suggestion_key"] = None

    next_number = int(next_anonymous_patient_id(db).removeprefix("PAT-"))
    allocated = 0
    group_rows: dict[str, list[dict]] = {}
    for row in matched:
        key = row["suggestion_key"]
        if key:
            group_rows.setdefault(key, []).append(row)
    for key, rows in group_rows.items():
        count = group_counts[key]
        references = {str(item["source_patient_reference"]) for item in rows if item["source_patient_reference"]}
        high_confidence = all(item["auto_group_eligible"] for item in rows) and len(references) == 1
        current_ids = {str(item["current_patient_id"]) for item in rows if item["current_patient_id"]}
        current_patient = None
        if len(current_ids) == 1:
            current_patient = (
                db.query(Patient)
                .filter(Patient.public_id == next(iter(current_ids)))
                .first()
            )
        conflict = len(current_ids) > 1 or bool(current_ids and (current_patient is None or not current_patient.is_anonymous))
        if high_confidence and conflict:
            for item in rows:
                item["assignment_conflict"] = True

        if high_confidence:
            reference = next(iter(references))
            group_label = f"Verified source reference {reference} ({count} photo(s))"
            evidence = (
                "Database SHA-256 exactly matches source_manifest.csv; the source record "
                "crosswalks by exact source path or filename to image_manifest.csv, which "
                "contains one unambiguous patient_identifier_if_available value."
            )
            for item in rows:
                item["confidence"] = "High"
                item["evidence"] = evidence
                if conflict:
                    item["proposed_patient_id"] = None
                    item["suggested_group"] = group_label + " — existing patient assignment conflict"
                elif current_patient is not None:
                    item["proposed_patient_id"] = current_patient.public_id
                    item["suggested_group"] = group_label + f" → {current_patient.public_id}"
                else:
                    item["proposed_patient_id"] = f"PAT-{next_number + allocated:03d}"
                    item["suggested_group"] = group_label + f" → {item['proposed_patient_id']}"
            if not conflict and current_patient is None:
                allocated += 1
        else:
            prefix = rows[0]["filename_prefix"]
            possible_references = sorted(
                {
                    str(reference)
                    for item in rows
                    for reference in item.get("possible_patient_references", [])
                }
            )
            if count > 1:
                group_label = (
                    f"Conflicting source references for filename clue {prefix} ({count} photo(s))"
                    if possible_references
                    else f"Repeated filename clue {prefix} ({count} possible match(es))"
                )
                confidence = "Medium"
            else:
                group_label = (
                    f"Conflicting source references for {prefix}"
                    if possible_references
                    else f"Filename clue only: {prefix or 'no patient reference'}"
                )
                confidence = "Low"
            if possible_references:
                evidence = (
                    "Exact SHA-256 match found, but source metadata contains conflicting explicit "
                    "patient references: " + ", ".join(possible_references) + ". No automatic assignment is allowed."
                )
            else:
                evidence = (
                    "Filename similarity is supporting evidence only. No unambiguous explicit "
                    "patient reference was verified, so this suggestion cannot assign a Patient ID."
                )
            for item in rows:
                item["confidence"] = confidence
                item["proposed_patient_id"] = None
                item["suggested_group"] = group_label
                item["evidence"] = evidence

    for row in matched:
        if row.get("suggestion_key"):
            continue
        if row["source_metadata_status"] == "ambiguous_patient_reference":
            evidence = (
                "The exact-hash source match crosswalks to conflicting explicit patient references: "
                + ", ".join(row.get("possible_patient_references") or [])
                + ". Human review is required; no ID was assigned."
            )
            suggested_group = "Ambiguous source patient reference"
        elif row["hash_matched"]:
            evidence = (
                "Exact SHA-256 match found, but no explicit patient reference could be verified. "
                "Leave unassigned or use manual assignment after review."
            )
            suggested_group = "Source matched; patient reference missing"
        else:
            evidence = (
                "No exact SHA-256 source-manifest match or reliable patient reference was found. "
                "Leave unassigned or assign manually after human review."
            )
            suggested_group = "No source grouping suggestion"
        row.update(
            {
                "confidence": "Low",
                "proposed_patient_id": None,
                "suggested_group": suggested_group,
                "evidence": evidence,
            }
        )
    return matched


def patient_photo_group_review_inventory(db: Session) -> dict:
    """Return source-backed suggestions and each suggestion's latest review."""
    suggestions = patient_photo_review_candidates(db)
    root = _project_root()
    source_manifest_available = (root / "data" / "icdas" / "labeling_v2" / "source_manifest.csv").is_file()
    image_manifest_available = (root / "fdi_detection_dataset" / "metadata" / "image_manifest.csv").is_file()
    events = (
        db.query(TrainingAuditEvent)
        .filter(TrainingAuditEvent.event_type == "patient_photo_group_review")
        .order_by(TrainingAuditEvent.id.desc())
        .limit(1000)
        .all()
    )
    reviews = {}
    for event in events:
        try:
            details = json.loads(event.details_json or "{}")
        except (TypeError, ValueError):
            continue
        key = details.get("suggestion_key")
        if key and key not in reviews:
            reviews[key] = {
                "decision": details.get("decision"),
                "photo_ids": details.get("photo_ids") or [],
                "assigned_patient_id": details.get("assigned_patient_id"),
                "evidence": details.get("evidence"),
                "method": details.get("method"),
                "reviewed_at_utc": details.get("reviewed_at_utc"),
                "actor": event.actor,
            }
    for row in suggestions:
        row["review_decision"] = (reviews.get(row["suggestion_key"]) or {}).get("decision") or "pending"
    groups = {
        key: [row for row in suggestions if row.get("suggestion_key") == key]
        for key in {row.get("suggestion_key") for row in suggestions if row.get("suggestion_key")}
    }
    summary = {
        "active_photos": len(suggestions),
        "confident_groups": sum(1 for rows in groups.values() if rows and all(row["auto_group_eligible"] for row in rows)),
        "confident_photos": sum(len(rows) for rows in groups.values() if rows and all(row["auto_group_eligible"] for row in rows)),
        "ambiguous_photos": sum(1 for row in suggestions if row["source_metadata_status"] == "ambiguous_patient_reference"),
        "unmatched_photos": sum(1 for row in suggestions if not row["hash_matched"]),
        "missing_reference_photos": sum(
            1 for row in suggestions
            if row["hash_matched"] and row["source_metadata_status"] != "verified_patient_reference"
        ),
    }
    return {
        "suggestions": suggestions,
        "reviews": reviews,
        "summary": summary,
        "source_metadata_available": source_manifest_available and image_manifest_available,
        "source_manifest_available": source_manifest_available,
        "image_manifest_available": image_manifest_available,
    }


def record_patient_photo_group_review(
    db: Session,
    *,
    suggestion_key: str,
    photo_ids: list[str],
    decision: str,
) -> dict:
    """Audit a proposal; only exact-hash groups with one verified reference can assign."""
    key = str(suggestion_key or "").strip()
    decision = str(decision or "").strip().lower()
    if not key or len(key) > 160:
        raise ValueError("Select a valid suggested group.")
    if decision not in {"confirm", "reject"}:
        raise ValueError("Decision must be confirm or reject.")
    resolved = resolve_photo_ids(db, " ".join(str(value) for value in photo_ids or []))
    normalized = [photo_id_for(value) for value in resolved["training_image_ids"]]
    candidates = [
        row for row in patient_photo_review_candidates(db)
        if row.get("suggestion_key") == key
    ]
    candidate_ids = {row["photo_id"] for row in candidates}
    if not candidate_ids or set(normalized) != candidate_ids:
        raise ValueError("The selected photos no longer match this source-backed suggestion.")

    details = {
        "suggestion_key": key,
        "decision": decision,
        "photo_ids": normalized,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if decision == "confirm" and candidates and all(row.get("auto_group_eligible") for row in candidates):
        if any(row.get("assignment_conflict") for row in candidates):
            raise ValueError(
                "These source-matched photos already have conflicting patient assignments. "
                "Review and correct them with the manual assignment workflow first."
            )
        current_ids = {str(row["current_patient_id"]) for row in candidates if row.get("current_patient_id")}
        if len(current_ids) > 1:
            raise ValueError("The suggested group spans multiple Patient IDs; manual correction is required.")
        if current_ids:
            target_id = next(iter(current_ids))
            target_patient = (
                db.query(Patient)
                .filter(Patient.public_id == target_id, Patient.is_anonymous.is_(True))
                .first()
            )
            if target_patient is None:
                raise ValueError("The current assignment is not an anonymous patient group; manual review is required.")
            action = "existing"
            expected_patient_id = None
        else:
            target_id = next_anonymous_patient_id(db)
            action = "create"
            expected_patient_id = target_id
        details.update(
            {
                "assigned_patient_id": target_id,
                "method": "exact_sha256_and_verified_source_reference",
                "source_patient_reference": candidates[0].get("source_patient_reference"),
                "evidence": candidates[0].get("evidence"),
            }
        )
        result = assign_training_photos(
            db,
            [row["training_image_id"] for row in candidates],
            action=action,
            target_patient_id=target_id if action == "existing" else None,
            expected_patient_id=expected_patient_id,
            review_event_details=details,
        )
        return {**details, "assigned_patient_id": result["patient_id"], "assignment": result}

    if decision == "confirm":
        details.update(
            {
                "assigned_patient_id": None,
                "method": "manual_review_only",
                "evidence": candidates[0].get("evidence") if candidates else None,
                "manual_assignment_required": True,
            }
        )
    db.add(
        TrainingAuditEvent(
            event_type="patient_photo_group_review",
            entity_type="training_images",
            entity_id=",".join(str(value) for value in resolved["training_image_ids"]),
            actor="admin",
            details_json=json.dumps(details, sort_keys=True),
        )
    )
    db.commit()
    return details


def preview_training_photo_assignment(
    db: Session,
    photo_id_text: str | None,
    *,
    action: str,
    target_patient_id: str | None = None,
) -> dict:
    """Return a database-backed, non-mutating preview of a bulk assignment."""
    parsed = parse_photo_id_text(photo_id_text)
    blockers: list[str] = []
    if not parsed["tokens"]:
        blockers.append("Enter at least one Photo ID.")
    if parsed["invalid_photo_ids"]:
        blockers.append("Invalid Photo ID(s): " + ", ".join(parsed["invalid_photo_ids"][:20]))
    if parsed["duplicate_photo_ids"]:
        blockers.append("Photo ID(s) were entered more than once: " + ", ".join(parsed["duplicate_photo_ids"][:20]))
    if len(parsed["training_image_ids"]) > 1000:
        blockers.append("Assign at most 1,000 photos per action.")

    action = str(action or "").strip().lower()
    if action not in {"create", "existing", "unassigned"}:
        blockers.append("Choose create, existing, or unassigned.")
        action = ""
    target_patient_id = str(target_patient_id or "").strip() or None
    target_patient = None
    if action == "create":
        if target_patient_id:
            blockers.append("A target Patient ID is not allowed when creating a new ID.")
        target_patient_id = next_anonymous_patient_id(db)
    elif action == "existing":
        if not target_patient_id:
            blockers.append("Select an existing anonymous Patient ID.")
        else:
            target_patient = (
                db.query(Patient)
                .filter(Patient.public_id == target_patient_id, Patient.is_anonymous.is_(True))
                .first()
            )
            if target_patient is None:
                blockers.append(f"{target_patient_id} is not an existing anonymous Patient ID.")
    elif action == "unassigned":
        if target_patient_id:
            blockers.append("A target Patient ID is not allowed when leaving photos unassigned.")
        target_patient_id = None

    parsed_ids = parsed["training_image_ids"]
    images = (
        db.query(TrainingImage)
        .filter(TrainingImage.id.in_(parsed_ids))
        .all()
        if parsed_ids
        else []
    )
    by_id = {int(image.id): image for image in images}
    missing_ids = [photo_id for photo_id in parsed_ids if photo_id not in by_id]
    if missing_ids:
        blockers.append("Photo ID(s) do not exist: " + ", ".join(photo_id_for(value) for value in missing_ids[:20]))

    photo_rows = []
    reassignments = []
    conflicts = []
    for photo_id in parsed_ids:
        image = by_id.get(photo_id)
        if image is None:
            continue
        patient = image.patient_record
        current_patient_id = image.patient_id or (patient.public_id if patient else None)
        row_blockers = []
        if image.is_active is False:
            row_blockers.append("Photo is deactivated.")
        if image.processing_status not in _TERMINAL_UPLOAD_STATUSES:
            row_blockers.append(f"Detection is not complete ({image.processing_status}).")
        identity_conflict = image.patient_record_id is not None and (
            patient is None or patient.public_id != image.patient_id
        )
        if row_blockers:
            blockers.extend(f"{photo_id_for(photo_id)}: {message}" for message in row_blockers)
            conflicts.append({"photo_id": photo_id_for(photo_id), "reasons": row_blockers})
        if identity_conflict:
            conflicts.append(
                {
                    "photo_id": photo_id_for(photo_id),
                    "reasons": ["Stored Patient ID and Patient table relationship disagree; applying will repair the link."],
                }
            )
        needs_reassignment = bool(
            (current_patient_id and current_patient_id != target_patient_id)
            or identity_conflict
        )
        if needs_reassignment:
            reassignments.append(photo_id_for(photo_id))
        photo_rows.append(
            {
                "training_image_id": photo_id,
                "photo_id": photo_id_for(photo_id),
                "filename": image.original_filename or image.filename,
                "current_patient_id": current_patient_id,
                "new_patient_id": target_patient_id,
                "processing_status": image.processing_status,
                "crop_count": int(image.n_crops or image.crop_count or 0),
                "status": (
                    "CONFLICT" if row_blockers
                    else "IDENTITY LINK CONFLICT" if identity_conflict
                    else "REASSIGNMENT REQUIRED" if needs_reassignment
                    else "ALREADY ASSIGNED" if current_patient_id == target_patient_id and target_patient_id
                    else "WILL ASSIGN"
                ),
            }
        )

    return {
        "action": action,
        "target_patient_id": target_patient_id,
        "photo_count": len(photo_rows),
        "requested_photo_ids": parsed["tokens"],
        "photos": photo_rows,
        "selected_image_ids": [row["training_image_id"] for row in photo_rows],
        "invalid_photo_ids": parsed["invalid_photo_ids"],
        "duplicate_photo_ids": parsed["duplicate_photo_ids"],
        "nonexistent_photo_ids": [photo_id_for(value) for value in missing_ids],
        "reassignment_required": reassignments,
        "assignment_conflicts": conflicts,
        "blockers": list(dict.fromkeys(blockers)),
        "can_assign": bool(photo_rows) and not blockers,
        "note": "Preview is read-only. Apply revalidates the submitted Photo IDs against the database.",
    }


def assign_training_photos(
    db: Session,
    selected_image_ids: list[int],
    *,
    action: str,
    target_patient_id: str | None = None,
    confirm_reassignment: bool = False,
    expected_patient_id: str | None = None,
    review_event_details: dict | None = None,
) -> dict:
    """Assign photos atomically, leaving crops and their review records untouched."""
    selected = sorted({int(value) for value in selected_image_ids or []})
    if not selected:
        raise ValueError("Select at least one source photo.")
    if len(selected) > 1000:
        raise ValueError("Assign at most 1,000 source photos per action.")
    action = str(action or "").strip().lower()
    if action not in {"create", "existing", "unassigned"}:
        raise ValueError("Action must be create, existing, or unassigned.")
    target_patient_id = str(target_patient_id or "").strip() or None
    if action == "existing" and not target_patient_id:
        raise ValueError("Select an existing anonymous patient ID.")
    if action != "existing" and target_patient_id:
        raise ValueError("An existing patient ID is only valid for the existing action.")
    expected_patient_id = str(expected_patient_id or "").strip() or None
    if action != "create" and expected_patient_id:
        raise ValueError("An expected Patient ID is only valid when creating a new ID.")

    for attempt in range(5):
        try:
            images = (
                db.query(TrainingImage)
                .filter(TrainingImage.id.in_(selected))
                .order_by(TrainingImage.id.asc())
                .all()
            )
            by_id = {int(image.id): image for image in images}
            missing = sorted(set(selected) - set(by_id))
            if missing:
                raise ValueError(f"Source photo ID(s) not found: {missing[:10]}.")
            not_ready = [
                image.id for image in images
                if image.processing_status not in _TERMINAL_UPLOAD_STATUSES
            ]
            inactive = [image.id for image in images if image.is_active is False]
            if inactive:
                raise ValueError(
                    "Deactivated photo ID(s) cannot be assigned: "
                    + ", ".join(photo_id_for(value) for value in inactive[:10])
                )
            if not_ready:
                raise ValueError(
                    "Wait for Tooth Detector processing to finish before assigning these photo(s): "
                    + ", ".join(str(value) for value in not_ready[:10])
                )

            if action == "create":
                patient = _next_anonymous_patient(db)
                if expected_patient_id and patient.public_id != expected_patient_id:
                    raise ValueError(
                        f"The next anonymous ID is now {patient.public_id}; refresh the preview before applying."
                    )
            elif action == "existing":
                patient = (
                    db.query(Patient)
                    .filter(
                        Patient.public_id == target_patient_id,
                        Patient.is_anonymous.is_(True),
                    )
                    .first()
                )
                if patient is None:
                    raise ValueError("That ID is not an existing anonymous patient group.")
            else:
                patient = None

            target_id = patient.public_id if patient else None
            target_record_id = patient.id if patient else None
            changes = []
            for image in images:
                old_patient_id = image.patient_id or (
                    image.patient_record.public_id if image.patient_record else None
                )
                if image.patient_id != target_id or image.patient_record_id != target_record_id:
                    changes.append(
                        {
                            "training_image_id": int(image.id),
                            "image_id": image.image_id,
                            "from_patient_id": old_patient_id,
                            "from_patient_record_id": image.patient_record_id,
                            "to_patient_id": target_id,
                            "to_patient_record_id": target_record_id,
                        }
                    )
            if (
                changes
                and any(item["from_patient_id"] and item["from_patient_id"] != target_id for item in changes)
                and not confirm_reassignment
            ):
                raise ValueError(
                    "Some selected photos already have a patient assignment. Confirm reassignment to continue."
                )

            if changes:
                for image in images:
                    image.patient_record = patient
                    image.patient_id = target_id
                    # Patient reassignment invalidates old per-photo split locks.
                    image.split_name = "UNASSIGNED"
                    image.split_locked = False
                    image.dataset_version = None
            if changes:
                assignment_details = {
                    "action": action,
                    "target_patient_id": target_id,
                    "changes": changes,
                    "changed_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
                }
                if review_event_details:
                    assignment_details["source_group_review"] = review_event_details
                db.add(
                    TrainingAuditEvent(
                        event_type="patient_group_assignment",
                        entity_type="training_images",
                        entity_id=",".join(str(value) for value in selected),
                        actor="admin",
                        details_json=json.dumps(assignment_details, sort_keys=True),
                    )
                )
            if review_event_details:
                review_details = {**review_event_details, "assigned_patient_id": target_id}
                db.add(
                    TrainingAuditEvent(
                        event_type="patient_photo_group_review",
                        entity_type="training_images",
                        entity_id=",".join(str(value) for value in selected),
                        actor="admin",
                        details_json=json.dumps(review_details, sort_keys=True),
                    )
                )
            db.commit()
            return {
                "assigned_count": len(images),
                "changed_count": len(changes),
                "patient_id": target_id,
                "action": action,
                "training_image_ids": selected,
            "photo_ids": [photo_id_for(value) for value in selected],
            }
        except IntegrityError:
            db.rollback()
            if action != "create":
                raise
        except OperationalError as exc:
            db.rollback()
            if "locked" not in str(exc).casefold() and "busy" not in str(exc).casefold():
                raise
        except Exception:
            db.rollback()
            raise
        if attempt < 4:
            time.sleep(0.02 * (attempt + 1))
    raise RuntimeError("Could not assign the selected photos after concurrent updates. Retry the operation.")
