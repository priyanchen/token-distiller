"""Content-addressed distillation cache. Keyed on file bytes, not path or mtime, so a
renamed or copied file is a cache hit and an edited one never is.

This is also what makes every downstream compaction lossless: the complete distilled
text is persisted here before anything is collapsed or deferred, so `distill expand`
can always return it in full.
"""

import hashlib
import json
from datetime import datetime, timezone

from token_distiller.models import DistillMethod, DistillResult, PageResult
from token_distiller.storage import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS distillations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash  TEXT NOT NULL UNIQUE,
    source_path   TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    pages_json    TEXT NOT NULL,
    boilerplate_json TEXT,
    duration_ms   INTEGER,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_distillations_hash ON distillations(content_hash);

CREATE TABLE IF NOT EXISTS session_reads (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT,
    content_hash  TEXT NOT NULL,
    seen_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_reads ON session_reads(session_id, content_hash);
"""


def _ensure_schema() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def content_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pages_to_json(result: DistillResult) -> str:
    return json.dumps(
        [
            {
                "page_index": p.page_index,
                "method": p.method.value,
                "text": p.text,
                "ocr_confidence": p.ocr_confidence,
                "ocr_word_count": p.ocr_word_count,
                "raw_tokens_est": p.raw_tokens_est,
                "distilled_tokens_est": p.distilled_tokens_est,
                "warnings": p.warnings,
                "image_count": p.image_count,
                "figures": p.figures,
            }
            for p in result.pages
        ]
    )


def _pages_from_json(blob: str) -> list[PageResult]:
    return [
        PageResult(
            page_index=d["page_index"],
            method=DistillMethod(d["method"]),
            text=d["text"],
            ocr_confidence=d["ocr_confidence"],
            ocr_word_count=d["ocr_word_count"],
            raw_tokens_est=d["raw_tokens_est"],
            distilled_tokens_est=d["distilled_tokens_est"],
            warnings=d["warnings"],
            # .get() with a 0 default: cache rows written before this field existed
            # don't have it, and a missing image count should read as "none seen",
            # not raise on load.
            image_count=d.get("image_count", 0),
            figures=d.get("figures", []),
        )
        for d in json.loads(blob)
    ]


def get(hash_value: str) -> tuple[int, DistillResult] | None:
    _ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, source_path, source_type, pages_json, boilerplate_json, duration_ms "
            "FROM distillations WHERE content_hash = ?",
            (hash_value,),
        ).fetchone()
    if row is None:
        return None
    result = DistillResult(
        source_path=row["source_path"],
        source_type=row["source_type"],
        pages=_pages_from_json(row["pages_json"]),
        duration_ms=row["duration_ms"] or 0,
        boilerplate=json.loads(row["boilerplate_json"] or "[]"),
    )
    return row["id"], result


def put(hash_value: str, result: DistillResult) -> int:
    _ensure_schema()
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO distillations "
            "(content_hash, source_path, source_type, pages_json, boilerplate_json, "
            "duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                hash_value,
                result.source_path,
                result.source_type,
                _pages_to_json(result),
                json.dumps(result.boilerplate),
                result.duration_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        row = conn.execute(
            "SELECT id FROM distillations WHERE content_hash = ?", (hash_value,)
        ).fetchone()
        return row["id"]


def get_by_id(handle: int) -> DistillResult | None:
    _ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT source_path, source_type, pages_json, boilerplate_json, duration_ms "
            "FROM distillations WHERE id = ?",
            (handle,),
        ).fetchone()
    if row is None:
        return None
    return DistillResult(
        source_path=row["source_path"],
        source_type=row["source_type"],
        pages=_pages_from_json(row["pages_json"]),
        duration_ms=row["duration_ms"] or 0,
        boilerplate=json.loads(row["boilerplate_json"] or "[]"),
    )


def list_entries(limit: int = 50) -> list[dict]:
    _ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, source_path, source_type, created_at FROM distillations "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def seen_in_session(session_id: str | None, hash_value: str) -> bool:
    if session_id is None:
        return False
    _ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM session_reads WHERE session_id = ? AND content_hash = ? LIMIT 1",
            (session_id, hash_value),
        ).fetchone()
        return row is not None


def mark_seen(session_id: str | None, hash_value: str) -> None:
    if session_id is None:
        return
    _ensure_schema()
    with connect() as conn:
        conn.execute(
            "INSERT INTO session_reads (session_id, content_hash, seen_at) VALUES (?, ?, ?)",
            (session_id, hash_value, datetime.now(timezone.utc).isoformat()),
        )
