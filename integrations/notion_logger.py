"""Notion run tracker — creates a page per training run in a Notion database."""
import os
from typing import Literal

from notion_client import Client  # type: ignore

StatusT = Literal["Running", "Completed", "Failed"]

_DB_ID = os.getenv("NOTION_TRAINING_DB_ID", "")


def _notion() -> Client:
    return Client(auth=os.environ["NOTION_TOKEN"])


def log_run(
    epoch: int,
    loss: float,
    checkpoint: str,
    status: StatusT = "Completed",
    notes: str = "",
) -> dict:
    """
    Creates a row in the Notion training_runs database.

    Required database columns:
      Run (Title), Loss (Number), Checkpoint (Text),
      Status (Select: Running | Completed | Failed), Notes (Text)
    """
    return _notion().pages.create(
        parent={"database_id": _DB_ID},
        properties={
            "Run": {"title": [{"text": {"content": f"Epoch {epoch} — loss {loss:.4f}"}}]},
            "Loss": {"number": float(loss)},
            "Checkpoint": {"rich_text": [{"text": {"content": checkpoint}}]},
            "Status": {"select": {"name": status}},
            "Notes": {"rich_text": [{"text": {"content": notes}}]},
        },
    )


def update_run_status(page_id: str, status: StatusT, notes: str = "") -> dict:
    return _notion().pages.update(
        page_id=page_id,
        properties={
            "Status": {"select": {"name": status}},
            "Notes": {"rich_text": [{"text": {"content": notes}}]},
        },
    )
