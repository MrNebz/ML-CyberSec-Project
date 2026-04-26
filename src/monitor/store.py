"""SQLite-backed prediction log for real-time drift monitoring.

All writes are batched through a module-level lock so the FastAPI background
task and any direct callers can coexist safely in the same process.

Schema
------
predictions(
    id INTEGER, ts REAL, model_key TEXT, predicted_class_id INT,
    confidence REAL, feature_json TEXT
)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..serve.config import DATA_DIR

DB_PATH: Path = DATA_DIR / "monitor" / "predictions.db"
_LOCK = threading.Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL    NOT NULL,
    model_key        TEXT    NOT NULL,
    predicted_class_id INTEGER NOT NULL,
    confidence       REAL    NOT NULL,
    feature_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_id ON predictions (model_key, id);
"""


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_CREATE_SQL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "feature_json" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN feature_json TEXT")
    conn.commit()
    conn.close()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_predictions(
    model_key: str,
    predicted_class_ids: list[int],
    confidences: list[float],
    feature_rows: list[dict] | None = None,
) -> None:
    ts = time.time()
    if feature_rows is None:
        feature_rows = [None] * len(predicted_class_ids)
    with _LOCK:
        _ensure_db()
        with _conn() as c:
            c.executemany(
                "INSERT INTO predictions (ts, model_key, predicted_class_id, confidence, feature_json) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        ts,
                        model_key,
                        int(cid),
                        float(conf),
                        json.dumps(row) if row is not None else None,
                    )
                    for cid, conf, row in zip(predicted_class_ids, confidences, feature_rows)
                ],
            )


def get_window(model_key: str, n: int = 500) -> list[dict]:
    """Return the last *n* logged rows for *model_key*, newest-first."""
    with _LOCK:
        _ensure_db()
        with _conn() as c:
            rows = c.execute(
                "SELECT ts, predicted_class_id, confidence "
                "FROM predictions WHERE model_key=? ORDER BY id DESC LIMIT ?",
                (model_key, n),
            ).fetchall()
    return [
        {"ts": r[0], "predicted_class_id": r[1], "confidence": r[2]}
        for r in rows
    ]


def get_total_logged(model_key: str) -> int:
    with _LOCK:
        _ensure_db()
        with _conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM predictions WHERE model_key=?", (model_key,)
            ).fetchone()[0]


def clear_log(model_key: str) -> int:
    """Delete all logged rows for *model_key*. Returns number of rows deleted."""
    with _LOCK:
        _ensure_db()
        with _conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE model_key=?", (model_key,)
            ).fetchone()[0]
            c.execute("DELETE FROM predictions WHERE model_key=?", (model_key,))
    return n


def get_feature_window(model_key: str, feature_names: list[str], n: int = 500) -> list[list[float]]:
    """Return feature vectors for the last *n* logged predictions, oldest-first."""
    with _LOCK:
        _ensure_db()
        with _conn() as c:
            rows = c.execute(
                "SELECT feature_json FROM predictions "
                "WHERE model_key=? AND feature_json IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (model_key, n),
            ).fetchall()

    vectors: list[list[float]] = []
    for (payload,) in reversed(rows):
        try:
            item = json.loads(payload)
            vectors.append([float(item[name]) for name in feature_names])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return vectors


def get_confidence_history(model_key: str, bucket_size: int = 100) -> list[dict]:
    """Return mean confidence per *bucket_size* predictions, oldest-first.

    Useful for plotting confidence degradation over a traffic stream.
    """
    with _LOCK:
        _ensure_db()
        with _conn() as c:
            rows = c.execute(
                "SELECT id, confidence FROM predictions WHERE model_key=? ORDER BY id",
                (model_key,),
            ).fetchall()

    if not rows:
        return []

    buckets: list[dict] = []
    for i in range(0, len(rows), bucket_size):
        chunk = rows[i : i + bucket_size]
        confs = [r[1] for r in chunk]
        buckets.append(
            {
                "bucket_index": i // bucket_size,
                "start_row": chunk[0][0],
                "n": len(confs),
                "mean_confidence": sum(confs) / len(confs),
                "min_confidence": min(confs),
                "max_confidence": max(confs),
            }
        )
    return buckets
