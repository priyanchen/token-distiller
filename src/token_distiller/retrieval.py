from token_distiller import index_store
from token_distiller.config import DEFAULT_TOP_K


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def query(question: str, top_k: int = DEFAULT_TOP_K, use_semantic: bool = True) -> list[dict]:
    chunks = index_store.all_chunks()
    if not chunks:
        return []

    bm25 = index_store.build_bm25(chunks)
    bm25_scores = list(bm25.get_scores(index_store.tokenize(question)))
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0

    semantic_scores: dict[int, float] | None = None
    if use_semantic:
        cached = index_store.embeddings_for([c["id"] for c in chunks])
        if cached:
            try:
                from token_distiller import embeddings

                query_vec = embeddings.embed_texts([question], input_type="query")[0]
                semantic_scores = {
                    chunk_id: _cosine(query_vec, vec) for chunk_id, vec in cached.items()
                }
            except Exception:
                semantic_scores = None

    blended = []
    for chunk, bm25_score in zip(chunks, bm25_scores):
        score = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0
        if semantic_scores is not None and chunk["id"] in semantic_scores:
            score = 0.5 * score + 0.5 * semantic_scores[chunk["id"]]
        blended.append((chunk, score))

    blended.sort(key=lambda pair: pair[1], reverse=True)
    return [{**chunk, "score": round(float(score), 4)} for chunk, score in blended[:top_k]]
