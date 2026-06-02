"""pytest suite for the Grokadile Stealth API.

Run with:
    pytest tests/test_api.py -v

Requires:
    pip install fastapi httpx pytest pytest-asyncio
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Patch heavy deps before importing the app ──────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def mock_brain():
    """Replace MegatronBrain with a lightweight mock for all tests."""
    mock = MagicMock()
    mock.think.return_value = "MOCKED RESPONSE"
    mock.initialize.return_value = None
    with patch("api.main.get_brain", return_value=mock):
        yield mock


@pytest.fixture(scope="session")
def client(mock_brain):
    from api.main import app
    return TestClient(app)


# ── Health ──────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── /infer ──────────────────────────────────────────────────────────────────
class TestInfer:
    def test_basic_infer(self, client):
        resp = client.post("/infer", json={"prompt": "Hello Stealth"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "STEALTH"
        assert isinstance(data["response"], str)
        assert data["prompt_len"] > 0

    def test_infer_with_agent(self, client):
        resp = client.post(
            "/infer",
            json={
                "prompt": "Analyze case 1FDV-23-0001009",
                "agent_name": "LEGAL",
                "mission": "Constitutional analysis",
                "max_len": 256,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["agent"] == "LEGAL"

    def test_infer_empty_prompt_rejected(self, client):
        resp = client.post("/infer", json={"prompt": ""})
        assert resp.status_code == 422  # Pydantic validation

    def test_infer_temperature_bounds(self, client):
        # temperature > 2.0 should be rejected
        resp = client.post("/infer", json={"prompt": "test", "temperature": 9.9})
        assert resp.status_code == 422


# ── /broadcast ──────────────────────────────────────────────────────────────
class TestBroadcast:
    def test_broadcast_all_agents(self, client):
        resp = client.post("/broadcast", json={"prompt": "Status report"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] >= 1
        assert isinstance(data["results"], dict)

    def test_broadcast_subset(self, client):
        resp = client.post(
            "/broadcast",
            json={"prompt": "Legal status", "agents": ["legal", "recon"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] == 2
        assert "legal" in data["results"]
        assert "recon" in data["results"]

    def test_broadcast_bad_agent_ignored(self, client):
        # A mix of valid + invalid agent names — invalid are ignored
        resp = client.post(
            "/broadcast",
            json={"prompt": "test", "agents": ["legal", "nonexistent_agent"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] == 1

    def test_broadcast_all_invalid_agents(self, client):
        resp = client.post(
            "/broadcast",
            json={"prompt": "test", "agents": ["ghost", "phantom"]},
        )
        assert resp.status_code == 400


# ── /agents ─────────────────────────────────────────────────────────────────
class TestAgents:
    def test_list_agents(self, client):
        resp = client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "legal" in data
        assert "code" in data
        assert "memory" in data
