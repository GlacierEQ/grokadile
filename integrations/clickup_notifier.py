"""ClickUp notifier — creates tasks on deploy and on training failures."""
import os
from typing import Any

import httpx

_BASE = "https://api.clickup.com/api/v2"


def _headers() -> dict:
    return {"Authorization": os.environ["CLICKUP_API_TOKEN"], "Content-Type": "application/json"}


def create_task(
    name: str,
    description: str = "",
    status: str = "open",
    tags: list[str] | None = None,
    priority: int = 3,
) -> dict[str, Any]:
    """Create a task in the configured ClickUp list."""
    list_id = os.environ["CLICKUP_LIST_ID"]
    payload = {
        "name": name,
        "description": description,
        "status": status,
        "priority": priority,
        "tags": tags or [],
    }
    with httpx.Client(timeout=10) as client:
        r = client.post(f"{_BASE}/list/{list_id}/task", json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()


def notify_training_failure(epoch: int, loss: float, error: str) -> dict:
    return create_task(
        name=f"[ALERT] Training failure @ epoch {epoch}",
        description=f"Loss: {loss}\nError: {error}\nRun ID: {os.getenv('RUN_ID', 'unknown')}",
        status="open",
        tags=["grokadile", "training", "alert"],
        priority=1,  # Urgent
    )


def notify_deploy(sha: str, env: str = "production") -> dict:
    return create_task(
        name=f"Deployed grokadile @ {sha[:8]} → {env}",
        description=f"Full SHA: {sha}\nEnvironment: {env}",
        status="complete",
        tags=["grokadile", "deployment"],
        priority=4,  # Low
    )
