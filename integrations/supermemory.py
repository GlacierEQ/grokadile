"""Supermemory.ai — stores and retrieves inference context across sessions."""
import os
from typing import Any

import httpx

_BASE = "https://api.supermemory.ai/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['SUPERMEMORY_API_KEY']}"}


def store_inference_memory(
    prompt: str,
    output: str,
    meta: dict[str, Any] | None = None,
) -> dict:
    """Persist a prompt/output pair to Supermemory for long-term context."""
    payload = {
        "content": f"PROMPT: {prompt}\nOUTPUT: {output}",
        "metadata": meta or {},
        "tags": ["grokadile", "inference"],
    }
    with httpx.Client(timeout=10) as client:
        r = client.post(f"{_BASE}/memories", json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()


def search_past_inferences(query: str, limit: int = 5) -> list[dict]:
    """Semantic search over stored inference history."""
    with httpx.Client(timeout=10) as client:
        r = client.get(
            f"{_BASE}/search",
            params={"q": query, "limit": limit},
            headers=_headers(),
        )
        r.raise_for_status()
        return r.json().get("results", [])


def delete_memory(memory_id: str) -> bool:
    with httpx.Client(timeout=10) as client:
        r = client.delete(f"{_BASE}/memories/{memory_id}", headers=_headers())
        return r.status_code == 200
