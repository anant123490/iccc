"""Portal API smoke tests (no model training)."""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.groq_service import _local_screening_report
from app.image_quality import assess_image_quality
from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_portal_health(client):
    response = await client.get("/api/v1/portal/health")
    assert response.status_code == 200
    body = response.json()
    assert "detector_v2" in body
    assert "disclaimer" in body
    assert body.get("icdas_loaded") is False
    assert "NOT_TRAINED" in str(body.get("icdas_status", ""))


@pytest.mark.asyncio
async def test_create_patient_and_history(client):
    response = await client.post(
        "/api/v1/patients",
        json={"name": "Portal Test", "age": 22, "gender": "F", "visit_date": "2026-08-28"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["patient"]["public_id"].startswith("P")
    pid = body["patient"]["public_id"]
    hist = await client.get(f"/api/v1/patients/{pid}/history")
    assert hist.status_code == 200
    assert hist.json()["visits"]


@pytest.mark.asyncio
async def test_admin_login_rejects_wrong_password(client):
    response = await client.post("/api/v1/admin/login", json={"password": "not-the-password-xxx"})
    assert response.status_code == 401


def test_patient_stage_mapping():
    from app.portal_service import patient_stage

    s0 = patient_stage(0)
    s4 = patient_stage(4)
    assert s0["current_stage"]
    assert s4["priority"] == "High"
    assert s0["icdas_grade"] == 0
    img = np.full((120, 120, 3), 30, dtype=np.uint8)
    q = assess_image_quality(img)
    assert q["verdict"] in {"PASS", "WARNING", "FAIL"}
    assert isinstance(q["warnings"], list)


def test_local_screening_does_not_change_grades():
    payload = {
        "patient": {"patient_id": "P1024", "name": "Test", "age": 22, "visit_date": "2026-08-28"},
        "teeth_analyzed": 2,
        "teeth_detected": 2,
        "icdas_counts": {"0": 1, "3": 1},
        "high_severity_teeth": [1],
        "teeth": [
            {"crop_index": 0, "icdas_grade": 0, "confidence": 90.0},
            {"crop_index": 1, "icdas_grade": 3, "confidence": 70.0},
        ],
    }
    out = _local_screening_report(payload)
    md = out["markdown"]
    assert "ICDAS Grade:** 0" in md or "**ICDAS Grade:** 0" in md
    assert "**ICDAS Grade:** 3" in md
    assert "T01" in md and "T02" in md
    assert "ICDAS 5" not in md and "ICDAS 6" not in md
    assert "filling" not in md.lower()
    assert "root canal" not in md.lower()
    assert "AI-assisted" in md or "एआई" in md


def test_groq_payload_keeps_grades_and_normalizes_confidence():
    from app.groq_service import groq_payload_from_structured

    payload = groq_payload_from_structured(
        {
            "patient": {"public_id": "P1", "name": "A", "age": 22, "visit_date": "2026-08-28"},
            "teeth": [{"icdas_grade": 2, "confidence": 91.0}],
        }
    )
    assert payload["teeth"][0]["icdas_grade"] == 2
    assert payload["teeth"][0]["tooth_id"] == "T01"
    assert abs(payload["teeth"][0]["confidence"] - 0.91) < 1e-6


def test_stale_ordinal_is_blocked():
    from app.portal_runtime import is_blocked_icdas_checkpoint

    assert is_blocked_icdas_checkpoint("models/icdas/historical/stale_ordinal_4output/deploy.keras")
    assert not is_blocked_icdas_checkpoint("models/icdas/current/deploy.keras")


@pytest.mark.asyncio
async def test_icdas_train_is_gated_off(client):
    login = await client.post("/api/v1/admin/login", json={"password": "changeme"})
    assert login.status_code == 200
    token = login.json()["token"]
    response = await client.post(
        "/api/v1/admin/train",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("launched") is False
    assert body.get("status") == "blocked"
    assert "disabled" in (body.get("message") or "").lower()


@pytest.mark.asyncio
async def test_admin_training_list_does_not_wipe_library(client):
    login = await client.post("/api/v1/admin/login", json={"password": "changeme"})
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    first = await client.get("/api/v1/admin/training/images", headers=headers)
    assert first.status_code == 200
    n_before = first.json().get("count")
    second = await client.get("/api/v1/admin/training/images", headers=headers)
    assert second.status_code == 200
    assert second.json().get("count") == n_before
    assert "images" in second.json()
@pytest.mark.asyncio
async def test_admin_dataset_not_ready_without_all_classes(client):
    login = await client.post("/api/v1/admin/login", json={"password": "changeme"})
    token = login.json()["token"]
    response = await client.get(
        "/api/v1/admin/dataset",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "photos" in body and "crops" in body and "icdas_labels" in body
    assert "workflow" in body
    counts = body.get("class_counts") or {}
    labeled = int((body.get("crops") or {}).get("labeled") or 0)
    if labeled < int(body.get("min_dataset_crops") or 5):
        assert body.get("dataset_ready") is False
    if any(int(counts.get(str(i), 0) or 0) == 0 for i in range(5)):
        assert body.get("classes_ready") is False
        assert body.get("missing_classes_message")
    train = await client.post(
        "/api/v1/admin/train",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert train.json().get("launched") is False
