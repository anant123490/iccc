"""Backend API tests for the ICDAS 0-4 softmax production contract."""

import inspect

import pytest
from httpx import ASGITransport, AsyncClient

from app.inference import InferenceEngine, NUM_CLASSES
from app.main import app
from app.schemas import ModelInfoResponse


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "disclaimer" in data
    assert data.get("icdas_classifier") == "NOT_TRAINED / NOT_DEPLOYED"
    assert data.get("model_loaded") is False
    assert data.get("tooth_detector_v2") in {"AVAILABLE", "UNAVAILABLE"}


@pytest.mark.asyncio
async def test_model_info_softmax_contract(client):
    response = await client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == 5
    assert body["icdas_mode"] == "0-4"
    assert body["ordinal_regression"] is False
    assert "softmax" in body["architecture"].lower()
    parsed = ModelInfoResponse.model_validate(body)
    assert parsed.num_classes == NUM_CLASSES


@pytest.mark.asyncio
async def test_report_rejects_grade_5(client):
    response = await client.post(
        "/api/v1/report",
        json={"icdas_grade": 5, "confidence": 90},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_report_rejects_grade_6(client):
    response = await client.post(
        "/api/v1/report",
        json={"icdas_grade": 6, "confidence": 90},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_history_empty_or_list(client):
    response = await client.get("/api/v1/history")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert isinstance(response.json(), list)


def test_explain_signature_uses_predicted_grade():
    params = inspect.signature(InferenceEngine.explain).parameters
    assert "predicted_grade" in params
    assert "grade" not in params


def test_production_engine_requires_five_classes():
    assert NUM_CLASSES == 5
