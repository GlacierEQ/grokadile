"""Supabase run logger — writes epoch/loss/config to training_runs table."""
import os
from datetime import datetime, timezone
from typing import Any

from supabase import create_client, Client


def _client() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def log_training_run(
    epoch: int,
    loss: float,
    config: dict[str, Any],
    checkpoint_path: str,
) -> dict:
    data = {
        "epoch": epoch,
        "loss": float(loss),
        "config": config,
        "checkpoint_path": checkpoint_path,
        "model_version": os.getenv("MODEL_VERSION", "dev"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = _client().table("training_runs").insert(data).execute()
    return result.data


def get_best_run() -> dict:
    result = (
        _client()
        .table("training_runs")
        .select("*")
        .order("loss", desc=False)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else {}


def get_recent_runs(limit: int = 20) -> list[dict]:
    result = (
        _client()
        .table("training_runs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data
