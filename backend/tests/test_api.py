"""Backend API tests."""

import pytest
from httpx import AsyncClient, ASGITransport
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
    assert data["status"] == "healthy"
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_model_info(client):
    response = await client.get("/api/v1/model/info")
    assert response.status_code == 200
    assert response.json()["num_classes"] == 7
