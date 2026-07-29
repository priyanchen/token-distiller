"""Classifies the session's recent tool-call pattern into an activity mode
(code/debug/review/infra/general) with plain rule-based bucketing — no LLM call,
so it's instant and free. Feeds session_audit's mode-aware ordering."""

from collections import Counter
from datetime import datetime, timezone

from context_distill.config import ACTIVITY_WINDOW_SIZE
from context_distill.storage import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS activity_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    ts          TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    bucket      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity_log(ts);
"""

TEST_MARKERS = ("pytest", "npm test", "npm run test", "go test", "jest", "cargo test", "mvn test")


def _ensure_schema() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def _bucket(tool_name: str, tool_input: dict) -> str:
    if tool_name in ("Edit", "Write", "NotebookEdit"):
        return "edit"
    if tool_name in ("Read", "Glob", "Grep"):
        return "read"
    if tool_name == "Bash":
        command = (tool_input.get("command") or "").lower()
        if any(marker in command for marker in TEST_MARKERS):
            return "bash_test"
        if command.startswith("git ") or " git " in command:
            return "bash_git"
        return "bash_infra"
    if tool_name in ("WebFetch", "WebSearch"):
        return "web"
    return "other"


def record(tool_name: str, tool_input: dict, session_id: str | None = None) -> None:
    _ensure_schema()
    bucket = _bucket(tool_name, tool_input or {})
    with connect() as conn:
        conn.execute(
            "INSERT INTO activity_log (session_id, ts, tool_name, bucket) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now(timezone.utc).isoformat(), tool_name, bucket),
        )
        # keep the table small — only the rolling window is ever read
        conn.execute(
            "DELETE FROM activity_log WHERE id NOT IN "
            "(SELECT id FROM activity_log ORDER BY id DESC LIMIT 500)"
        )


def recent_buckets(session_id: str | None = None, window: int = ACTIVITY_WINDOW_SIZE) -> list[str]:
    _ensure_schema()
    with connect() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT bucket FROM activity_log WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, window),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT bucket FROM activity_log ORDER BY id DESC LIMIT ?", (window,)
            ).fetchall()
        return [r["bucket"] for r in rows]


def classify(buckets: list[str]) -> str:
    if not buckets:
        return "general"
    counts = Counter(buckets)
    n = len(buckets)
    edit = counts.get("edit", 0)
    read = counts.get("read", 0)
    test = counts.get("bash_test", 0)
    git = counts.get("bash_git", 0)
    infra = counts.get("bash_infra", 0)

    if test >= 2 and edit >= 1:
        return "debug"
    if edit >= max(1, n // 3):
        return "code"
    if read >= max(1, int(n * 0.6)) and edit == 0:
        return "review"
    if (infra + git) >= max(1, int(n * 0.5)):
        return "infra"
    return "general"


def current_mode(session_id: str | None = None, window: int = ACTIVITY_WINDOW_SIZE) -> dict:
    buckets = recent_buckets(session_id=session_id, window=window)
    return {
        "mode": classify(buckets),
        "window_size": len(buckets),
        "bucket_counts": dict(Counter(buckets)),
    }
