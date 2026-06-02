"""MotherDuck analytics — streams per-step training metrics to DuckDB cloud."""
import os

import duckdb


def _conn() -> duckdb.DuckDBPyConnection:
    token = os.environ["MOTHERDUCK_TOKEN"]
    return duckdb.connect(f"md:grokadile?motherduck_token={token}")


def ensure_table() -> None:
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id     VARCHAR,
            epoch      INTEGER,
            step       INTEGER,
            loss       FLOAT,
            lr         FLOAT,
            recorded_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.close()


def push_step_metrics(epoch: int, step: int, loss: float, lr: float) -> None:
    ensure_table()
    conn = _conn()
    conn.execute(
        "INSERT INTO run_metrics (run_id, epoch, step, loss, lr) VALUES (?, ?, ?, ?, ?)",
        [os.getenv("RUN_ID", "default"), epoch, step, float(loss), lr],
    )
    conn.close()


def get_loss_trend(last_n_epochs: int = 10):
    conn = _conn()
    df = conn.execute(
        f"""
        SELECT epoch,
               AVG(loss)  AS avg_loss,
               MIN(loss)  AS min_loss,
               MAX(loss)  AS max_loss,
               COUNT(*)   AS steps
        FROM run_metrics
        WHERE run_id = ?
        GROUP BY epoch
        ORDER BY epoch DESC
        LIMIT {last_n_epochs}
        """,
        [os.getenv("RUN_ID", "default")],
    ).df()
    conn.close()
    return df
