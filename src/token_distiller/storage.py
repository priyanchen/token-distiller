import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from token_distiller.config import DB_PATH, ensure_home

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                       TEXT NOT NULL,
    source_path              TEXT NOT NULL,
    source_type              TEXT NOT NULL,
    trigger                  TEXT NOT NULL,
    page_count               INTEGER,
    pages_native_text        INTEGER DEFAULT 0,
    pages_ocr                INTEGER DEFAULT 0,
    pages_vision_fallback    INTEGER DEFAULT 0,
    ocr_mean_confidence      REAL,
    raw_tokens_est           INTEGER,
    distilled_tokens_est     INTEGER,
    token_estimation_method  TEXT,
    compression_ratio        REAL,
    vision_api_calls         INTEGER DEFAULT 0,
    vision_model             TEXT,
    duration_ms              INTEGER,
    status                   TEXT NOT NULL,
    error_message            TEXT,
    output_path               TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_source_path ON runs(source_path);
"""


@contextmanager
def connect():
    ensure_home()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_run(**fields) -> int:
    fields.setdefault("ts", datetime.now(timezone.utc).isoformat())
    columns = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO runs ({columns}) VALUES ({placeholders})",
            tuple(fields.values()),
        )
        return cur.lastrowid


def query_runs(since: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        if since:
            cur = conn.execute(
                "SELECT * FROM runs WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (since, limit),
            )
        else:
            cur = conn.execute("SELECT * FROM runs ORDER BY ts DESC LIMIT ?", (limit,))
        return cur.fetchall()


def savings_summary(since: str | None = None) -> dict:
    with connect() as conn:
        if since:
            cur = conn.execute(
                "SELECT COUNT(*) as n, "
                "SUM(raw_tokens_est) as raw, SUM(distilled_tokens_est) as distilled "
                "FROM runs WHERE ts >= ? AND status = 'ok'",
                (since,),
            )
        else:
            cur = conn.execute(
                "SELECT COUNT(*) as n, "
                "SUM(raw_tokens_est) as raw, SUM(distilled_tokens_est) as distilled "
                "FROM runs WHERE status = 'ok'"
            )
        row = cur.fetchone()
        raw = row["raw"] or 0
        distilled = row["distilled"] or 0
        return {
            "run_count": row["n"] or 0,
            "raw_tokens_est": raw,
            "distilled_tokens_est": distilled,
            "tokens_saved_est": max(0, raw - distilled),
            "compression_ratio": (raw / distilled) if distilled else 0.0,
        }
