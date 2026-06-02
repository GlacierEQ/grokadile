"""Google Sheets live dashboard — appends per-epoch metrics to a spreadsheet."""
import json
import os
from datetime import datetime, timezone

import gspread  # type: ignore
from google.oauth2.service_account import Credentials  # type: ignore

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_HEADERS = ["epoch", "loss", "step", "lr", "model_version", "recorded_at"]


def _sheet():
    creds_raw = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    creds_info = json.loads(creds_raw)
    creds = Credentials.from_service_account_info(creds_info, scopes=_SCOPES)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(os.environ["GSHEET_ID"]).sheet1
    # Write headers if sheet is empty
    if not sheet.row_values(1):
        sheet.append_row(_HEADERS)
    return sheet


def append_epoch_row(epoch: int, loss: float, step: int, lr: float) -> None:
    _sheet().append_row(
        [
            epoch,
            round(float(loss), 6),
            step,
            lr,
            os.getenv("MODEL_VERSION", "dev"),
            datetime.now(timezone.utc).isoformat(),
        ]
    )
