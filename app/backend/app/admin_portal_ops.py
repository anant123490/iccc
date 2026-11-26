"""Admin labelling cursors and dataset-split planning.

Used by the existing training workflow. Does not train a model.
"""

from __future__ import annotations

import math
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .portal_db import LabelingProgress

SPLIT_NAMES = ("train", "val", "test")


def content_digest_for_image(image) -> str:
    """Return the stored SHA-256 or derive it from an accessible source file."""
    digest = (getattr(image, "content_hash", None) or "").strip().lower()
    if digest:
        return digest
    source_path = getattr(image, "path", None)
    if not source_path:
        return ""
    path = Path(source_path)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fractions(plan: dict) -> tuple[float, float, float]:
    train = float(plan.get("train_pct", 70))
    val = float(plan.get("val_pct", 15))
    test = float(plan.get("test_pct", 15))
    total = train + val + test
    if not all(math.isfinite(value) for value in (train, val, test)) or min(train, val, test) < 0 or total <= 0:
        raise ValueError("Split percentages must be non-negative and add up to more than zero.")
    return (train / total, val / total, test / total)


def _patient_of(image) -> str:
    return str(image.patient_id or "").strip()


def _subset_of_groups(group_ids: list[str], by_group: dict[str, list], target: int) -> set[str] | None:
    """Find a whole-patient subset with exactly ``target`` source photos."""
    if target < 0:
        return None
    if target == 0:
        return set()
    parents: dict[int, tuple[int, str] | None] = {0: None}
    for group_id in group_ids:
        size = len(by_group[group_id])
        for total in sorted(tuple(parents), reverse=True):
            candidate = total + size
            if candidate <= target and candidate not in parents:
                parents[candidate] = (total, group_id)
        if target in parents:
            break
    if target not in parents:
        return None
    chosen: set[str] = set()
    cursor = target
    while cursor:
        previous = parents[cursor]
        if previous is None:
            return None
        cursor, group_id = previous
        chosen.add(group_id)
    return chosen


def _candidate_subsets_of_groups(
    group_ids: list[str],
    by_group: dict[str, list],
    target: int,
    max_candidates_per_total: int = 512,
) -> list[set[str]]:
    """Return deterministic whole-group subsets for an exact photo count."""
    if target < 0:
        return []
    states: dict[int, list[frozenset[str]]] = {0: [frozenset()]}
    for group_id in group_ids:
        size = len(by_group[group_id])
        for total in sorted(tuple(states), reverse=True):
            candidate_total = total + size
            if candidate_total > target:
                continue
            bucket = states.setdefault(candidate_total, [])
            for existing in states[total]:
                if len(bucket) >= max_candidates_per_total:
                    break
                bucket.append(existing | {group_id})
    return [set(candidate) for candidate in states.get(target, [])]


def resolve_image_splits(images: list, patient_class_counts: dict, plan: dict | None) -> dict:
    """Assign photos as indivisible patient/hash groups; crops inherit their photo split."""
    from .training_workflow import split_patients

    plan = dict(plan or {})
    strategy = str(plan.get("strategy") or "percentage").lower()
    allow_reassign_locked = bool(plan.get("allow_reassign_locked"))
    if strategy not in {"percentage", "counts", "manual", "hybrid", "bulk"}:
        raise ValueError("Split strategy must be percentage, counts, manual, hybrid, or bulk.")
    seed_value = plan.get("seed")
    seed = 42 if seed_value is None else int(seed_value)
    fractions = _fractions(plan)

    # Patient IDs keep all known photos together. Exact SHA-256 duplicates link
    # patient groups for split purposes without changing or inventing identity.
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            low, high = sorted((a, b))
            parent[high] = low

    image_group: dict[int, str] = {}
    missing_patient_ids: list[int] = []
    hash_owner: dict[str, str] = {}
    image_by_id = {int(image.id): image for image in images}
    for image in images:
        pid = _patient_of(image)
        if not pid:
            missing_patient_ids.append(int(image.id))
            pid = f"MISSING-{image.id}"
        token = f"PATIENT:{pid}" if not pid.startswith("MISSING-") else pid
        find(token)
        image_group[int(image.id)] = token
        digest = content_digest_for_image(image)
        if digest:
            previous = hash_owner.get(digest)
            if previous is not None:
                union(token, previous)
            else:
                hash_owner[digest] = token
    for image in images:
        canonical_id = getattr(image, "canonical_id", None)
        if canonical_id is not None and int(canonical_id) in image_group:
            union(image_group[int(image.id)], image_group[int(canonical_id)])

    by_group: dict[str, list] = defaultdict(list)
    group_patients: dict[str, set[str]] = defaultdict(set)
    for image in images:
        token = image_group[int(image.id)]
        group_id = find(token)
        by_group[group_id].append(image)
        pid = _patient_of(image)
        if pid:
            group_patients[group_id].add(pid)

    issues: list[str] = []
    warnings: list[str] = []
    deviations: list[str] = []
    if missing_patient_ids:
        warnings.append(
            f"{len(missing_patient_ids)} photo(s) have no patient ID. Missing-ID photos remain unassigned and excluded unless an exact-duplicate group safely links them to a known patient."
        )

    # Locked photos reserve their entire patient/hash component.
    locked: dict[str, str] = {}
    lock_conflicts: set[str] = set()
    for image in images:
        group_id = find(image_group[int(image.id)])
        name = str(getattr(image, "split_name", "") or "")
        if getattr(image, "split_locked", False) and name in SPLIT_NAMES:
            previous = locked.get(group_id)
            if previous and previous != name:
                lock_conflicts.add(group_id)
                locked.pop(group_id, None)
                if not allow_reassign_locked:
                    issues.append(f"Patient/hash group {group_id} has locked photos in more than one split.")
            else:
                if not allow_reassign_locked:
                    locked[group_id] = name

    requested: dict[str, set[str]] = defaultdict(set)
    requested_images: set[int] = set()
    selected = {int(value) for value in (plan.get("selected_image_ids") or [])}
    valid_ids = {int(image.id) for image in images}
    unknown = sorted(selected - valid_ids)
    if unknown:
        issues.append(f"Selected photo IDs are not in the active dataset: {unknown[:10]}.")
    assignments = {str(key): str(value) for key, value in (plan.get("assignments") or {}).items()}
    if strategy == "bulk":
        target = str(plan.get("bulk_split") or "")
        if target not in SPLIT_NAMES:
            issues.append("Bulk assignment needs a split of train, val, or test.")
        elif not selected:
            issues.append("Select at least one photo for bulk assignment.")
        else:
            for image in images:
                if int(image.id) in selected:
                    requested[find(image_group[int(image.id)])].add(target)
                    requested_images.add(int(image.id))
    elif strategy in {"manual", "hybrid"}:
        for image in images:
            image_id = int(image.id)
            chosen = assignments.get(str(image_id))
            if chosen in SPLIT_NAMES and (not selected or image_id in selected):
                requested[find(image_group[image_id])].add(chosen)
                requested_images.add(image_id)

    manual_groups: set[str] = set()
    for group_id, choices in requested.items():
        if not group_patients[group_id]:
            issues.append(f"Patient/hash group {group_id} has no known patient ID and cannot be assigned.")
            continue
        if len(choices) > 1:
            issues.append(f"Photos for one patient or duplicate image were manually assigned to {sorted(choices)}.")
            continue
        choice = next(iter(choices))
        if group_id in lock_conflicts and not allow_reassign_locked:
            continue
        if group_id in locked and locked[group_id] != choice and not allow_reassign_locked:
            issues.append(f"Patient/hash group {group_id} is locked to {locked[group_id]} and cannot move to {choice}.")
            continue
        locked[group_id] = choice
        manual_groups.add(group_id)

    missing_only_groups = [group_id for group_id in by_group if not group_patients[group_id]]
    remaining = [
        group_id for group_id in by_group
        if group_id not in locked and group_id not in missing_only_groups
    ]

    # Aggregate known grade support when duplicate hashes connect patients.
    grouped_classes: dict[str, dict[str, int]] = {}
    for group_id, patient_ids in group_patients.items():
        values: Counter = Counter()
        for pid in patient_ids:
            values.update({int(k): int(v) for k, v in (patient_class_counts.get(pid) or {}).items()})
        grouped_classes[group_id] = {str(grade): int(values.get(grade, 0)) for grade in range(5)}

    if strategy in {"percentage", "hybrid", "bulk"} and remaining:
        from .training_workflow import patient_split_sizes

        target_sizes = patient_split_sizes(sum(bool(group_patients[group_id]) for group_id in by_group), fractions)
        target_counts = {split: float(target_sizes[split]) for split in SPLIT_NAMES}
        for split in SPLIT_NAMES:
            target_counts[split] = max(
                0.0,
                target_counts[split]
                - sum(
                    1
                    for group_id, value in locked.items()
                    if value == split and group_patients[group_id]
                ),
            )
        denominator = sum(target_counts.values())
        remaining_fractions = (
            tuple(target_counts[split] / denominator for split in SPLIT_NAMES)
            if denominator
            else fractions
        )
        group_order = sorted(remaining)
        parts = split_patients(
            group_order,
            seed=seed,
            patient_class_counts=grouped_classes,
            fractions=remaining_fractions,
        )
        for split, group_ids in parts.items():
            for group_id in group_ids:
                locked[group_id] = split
    elif strategy == "counts":
        assignable_images = sum(
            len(by_group[group_id]) for group_id in by_group if group_patients[group_id]
        )
        targets = {
            "train": int(plan.get("train_images") or 0),
            "val": int(plan.get("val_images") or 0),
            "test": int(plan.get("test_images") or 0),
        }
        if any(value < 0 for value in targets.values()) or sum(targets.values()) != assignable_images:
            issues.append(
                f"Exact photo targets must be non-negative and sum to {assignable_images} photos with a patient ID."
            )
        else:
            residual = {
                split: targets[split] - sum(
                    len(by_group[group_id])
                    for group_id, assigned in locked.items()
                    if assigned == split and group_patients[group_id]
                )
                for split in SPLIT_NAMES
            }
            if any(value < 0 for value in residual.values()):
                issues.append("Locked or manually assigned photos exceed one or more exact split counts.")
            elif sum(residual.values()) != sum(len(by_group[group_id]) for group_id in remaining):
                issues.append("Exact split counts do not leave a valid whole-patient assignment for remaining photos.")
            else:
                rng = np.random.default_rng(seed)
                shuffled = list(remaining)
                rng.shuffle(shuffled)
                shuffled.sort(key=lambda group_id: -len(by_group[group_id]))
                selected_partition = None
                for train_groups in _candidate_subsets_of_groups(shuffled, by_group, residual["train"]):
                    left = [group_id for group_id in remaining if group_id not in train_groups]
                    val_groups = _subset_of_groups(left, by_group, residual["val"])
                    if val_groups is not None:
                        selected_partition = (train_groups, val_groups, set(left) - val_groups)
                        break
                if selected_partition is None:
                    issues.append("Requested exact photo counts cannot be met without splitting a patient group.")
                else:
                    train_groups, val_groups, test_groups = selected_partition
                    for group_id in train_groups:
                        locked[group_id] = "train"
                    for group_id in val_groups:
                        locked[group_id] = "val"
                    for group_id in test_groups:
                        locked[group_id] = "test"

    image_to_split: dict[int, str] = {}
    unassigned: list[int] = []
    for group_id, group in by_group.items():
        split = locked.get(group_id, "") if group_patients[group_id] else ""
        for image in group:
            if split in SPLIT_NAMES:
                image_to_split[int(image.id)] = split
            else:
                unassigned.append(int(image.id))

    # All known patient IDs in a duplicate-linked component report the same split.
    patient_to_split: dict[str, str] = {}
    for group_id, patient_ids in group_patients.items():
        split = locked.get(group_id)
        if split in SPLIT_NAMES:
            patient_to_split.update({pid: split for pid in patient_ids})

    manual_image_ids = sorted(
        int(image.id)
        for group_id in manual_groups
        for image in by_group[group_id]
    )

    return {
        "strategy": strategy,
        "seed": seed,
        "fractions": {"train": fractions[0], "val": fractions[1], "test": fractions[2]},
        "image_to_split": image_to_split,
        "patient_to_split": patient_to_split,
        "unassigned_image_ids": sorted(unassigned),
        "manual_groups": sorted(manual_groups),
        "manual_image_ids": manual_image_ids,
        "issues": issues,
        "warnings": warnings,
        "deviations": deviations,
        "missing_patient_ids": missing_patient_ids,
        "patient_conflicts": [note for note in issues if note.startswith("Patient")],
    }


def summarize_split(images: list, labels: list, resolved: dict) -> dict:
    image_to_split = resolved["image_to_split"]
    per = {
        name: {
            "patients": set(),
            "images": 0,
            "crops": 0,
            "labeled_crops": 0,
            "verified_crops": 0,
            "classes": Counter(),
            "all_labeled_classes": Counter(),
        }
        for name in SPLIT_NAMES
    }
    labels_by_image: dict[int, list] = defaultdict(list)
    for label in labels:
        image_id = getattr(label, "image_id", None)
        if image_id is not None:
            labels_by_image[int(image_id)].append(label)
        elif getattr(label, "image", None) is not None:
            labels_by_image[int(label.image.id)].append(label)
    for image in images:
        split = image_to_split.get(int(image.id))
        if split not in per:
            continue
        per[split]["images"] += 1
        if _patient_of(image):
            per[split]["patients"].add(_patient_of(image))
        image_verified = bool(getattr(image, "boxes_verified", False))
        for lab in labels_by_image.get(int(image.id), getattr(image, "labels", []) or []):
            if getattr(lab, "active", True) is False:
                continue
            per[split]["crops"] += 1
            verified = image_verified and bool(getattr(lab, "box_verified", False))
            if verified:
                per[split]["verified_crops"] += 1
            if getattr(lab, "grade", None) is not None:
                per[split]["labeled_crops"] += 1
                if 0 <= int(lab.grade) <= 4:
                    per[split]["all_labeled_classes"][int(lab.grade)] += 1
                if verified and 0 <= int(lab.grade) <= 4:
                    per[split]["classes"][int(lab.grade)] += 1
    total_patients = len(set().union(*(row["patients"] for row in per.values())))
    target_fractions = resolved.get("fractions") or {"train": 0.7, "val": 0.15, "test": 0.15}
    summary = {}
    for name, row in per.items():
        actual_pct = 100.0 * len(row["patients"]) / total_patients if total_patients else 0.0
        target_pct = 100.0 * float(target_fractions.get(name, 0.0))
        summary[name] = {
            "patient_count": len(row["patients"]),
            "total_patients": len(row["patients"]),
            "image_count": row["images"],
            "original_photo_count": row["images"],
            "crop_count": row["crops"],
            "tooth_crop_count": row["crops"],
            "detected_crop_count": row["crops"],
            "labeled_crop_count": row["labeled_crops"],
            "labelled_crop_count": row["labeled_crops"],
            "verified_crop_count": row["verified_crops"],
            "class_counts": {str(grade): int(row["classes"].get(grade, 0)) for grade in range(5)},
            "eligible_grade_counts": {str(grade): int(row["classes"].get(grade, 0)) for grade in range(5)},
            "icdas_grade_distribution": {
                str(grade): int(row["all_labeled_classes"].get(grade, 0)) for grade in range(5)
            },
            "actual_patient_percent": round(actual_pct, 2),
            "target_patient_percent": round(target_pct, 2),
            "deviation_percentage_points": round(actual_pct - target_pct, 2),
        }
    return {
        **resolved,
        "image_to_split": {str(key): value for key, value in image_to_split.items()},
        "split_summary": summary,
        "ok": not resolved["issues"],
    }


def remember_progress(db, track: str, label_id: int | None, image_id: int | None) -> None:
    row = db.query(LabelingProgress).filter(LabelingProgress.track == track).first()
    if row is None:
        row = LabelingProgress(track=track)
        db.add(row)
    row.label_id = label_id
    row.image_id = image_id
    from datetime import datetime

    row.updated_at = datetime.utcnow()


def progress_for(db, track: str) -> LabelingProgress | None:
    return db.query(LabelingProgress).filter(LabelingProgress.track == track).first()
