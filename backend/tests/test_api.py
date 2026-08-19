"""Backend API tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


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
    assert data["num_classes"] if "num_classes" in data else True


@pytest.mark.asyncio
async def test_model_info(client):
    response = await client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == 5
    assert body["icdas_mode"] == "0-4"


@pytest.mark.asyncio
async def test_report_rejects_grade_5(client):
    response = await client.post(
        "/api/v1/report",
        json={"icdas_grade": 5, "confidence": 90},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_history_empty_or_list(client):
    response = await client.get("/api/v1/history")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert isinstance(response.json(), list)
