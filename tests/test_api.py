"""Basic smoke tests for the FastAPI inference API."""
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mock heavy deps before importing api.main
for mod in ["jax", "jax.numpy", "flax", "dm_haiku", "sentencepiece", "sentry_sdk"]:
    sys.modules.setdefault(mod, MagicMock())

# Stub runners.generate so we don't need GPU in CI
runners_mock = MagicMock()
runners_mock.generate.return_value = "hello world test output"
sys.modules["runners"] = runners_mock

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_infer_basic():
    resp = client.post(
        "/infer",
        json={"prompt": "Hello grokadile", "max_tokens": 32, "temperature": 0.7},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "output" in data
    assert "tokens_used" in data
    assert "model_version" in data


def test_infer_empty_prompt():
    resp = client.post("/infer", json={"prompt": "", "max_tokens": 32})
    assert resp.status_code == 422  # Pydantic min_length=1 validation


def test_infer_invalid_temperature():
    resp = client.post("/infer", json={"prompt": "test", "temperature": 99.0})
    assert resp.status_code == 422  # temperature max=2.0
