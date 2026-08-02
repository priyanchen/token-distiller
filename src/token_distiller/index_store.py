import json
import re
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi

from token_distiller.storage import connect
from token_distiller.tokens import estimate_text_tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path   TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    token_count   INTEGER,
    indexed_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id   INTEGER PRIMARY KEY REFERENCES chunks(id),
    model      TEXT NOT NULL,
    vector     TEXT NOT NULL
);
"""


def _ensure_schema() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def clear_source(source_path: str) -> None:
    _ensure_schema()
    with connect() as conn:
        conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE source_path = ?)",
            (source_path,),
        )
        conn.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))


def add_chunks(source_path: str, texts: list[str]) -> list[int]:
    _ensure_schema()
    ts = datetime.now(timezone.utc).isoformat()
    ids = []
    with connect() as conn:
        for i, text in enumerate(texts):
            cur = conn.execute(
                "INSERT INTO chunks (source_path, chunk_index, text, token_count, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (source_path, i, text, estimate_text_tokens(text), ts),
            )
            ids.append(cur.lastrowid)
    return ids


def add_embeddings(chunk_ids: list[int], vectors: list[list[float]], model: str) -> None:
    _ensure_schema()
    with connect() as conn:
        for chunk_id, vector in zip(chunk_ids, vectors):
            conn.execute(
                "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model, vector) VALUES (?, ?, ?)",
                (chunk_id, model, json.dumps(vector)),
            )


def all_chunks() -> list[dict]:
    _ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, source_path, chunk_index, text, token_count FROM chunks"
        ).fetchall()
        return [dict(r) for r in rows]


def embeddings_for(chunk_ids: list[int]) -> dict[int, list[float]]:
    _ensure_schema()
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    with connect() as conn:
        rows = conn.execute(
            f"SELECT chunk_id, vector FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {row["chunk_id"]: json.loads(row["vector"]) for row in rows}


def tokenize(text: str) -> list[str]:
    """Splitting on whitespace alone leaves punctuation attached, so a document ending
    "...locking readers." never matches the query term "readers". Queries and documents
    must run through this same function or scoring silently misses obvious hits."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(chunks: list[dict]):
    if not chunks:
        return None
    return BM25Okapi([tokenize(c["text"]) for c in chunks])
