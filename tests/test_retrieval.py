from token_distiller import index_store, retrieval
from token_distiller.chunker import chunk_text

APPLES = "The Gravenstein apple is an heirloom variety prized for applesauce and baking."
POSTGRES = "PostgreSQL uses multi-version concurrency control to avoid locking readers."
ORANGES = "Valencia oranges are the main variety used for commercial juice production."


def _index():
    for name, text in [("apples.txt", APPLES), ("db.txt", POSTGRES), ("oranges.txt", ORANGES)]:
        index_store.add_chunks(name, [text])


def test_tokenizer_strips_trailing_punctuation():
    """Whitespace-only splitting leaves "readers." attached, so the query term
    "readers" scores zero against a sentence that plainly contains it."""
    assert index_store.tokenize("locking readers.") == ["locking", "readers"]


def test_tokenizer_is_case_insensitive():
    assert index_store.tokenize("PostgreSQL") == index_store.tokenize("postgresql")


def test_query_without_an_index_returns_nothing():
    assert retrieval.query("anything at all") == []


def test_query_ranks_the_relevant_document_first():
    _index()
    top = retrieval.query("which apple variety for applesauce", use_semantic=False)[0]
    assert top["source_path"] == "apples.txt"


def test_query_discriminates_between_topics():
    _index()
    top = retrieval.query("does postgres lock readers", use_semantic=False)[0]
    assert top["source_path"] == "db.txt"


def test_top_k_limits_results():
    _index()
    assert len(retrieval.query("variety", top_k=2, use_semantic=False)) == 2


def test_results_carry_scores_in_descending_order():
    _index()
    scores = [r["score"] for r in retrieval.query("apple variety", use_semantic=False)]
    assert scores == sorted(scores, reverse=True)


def test_missing_embeddings_falls_back_to_keyword_search():
    """No VOYAGE_API_KEY is the normal case; retrieval must still work."""
    _index()
    assert retrieval.query("applesauce", use_semantic=True)[0]["source_path"] == "apples.txt"


def test_reindexing_a_source_replaces_its_chunks():
    index_store.add_chunks("notes.txt", ["original content"])
    index_store.clear_source("notes.txt")
    index_store.add_chunks("notes.txt", ["replacement content"])
    texts = [c["text"] for c in index_store.all_chunks() if c["source_path"] == "notes.txt"]
    assert texts == ["replacement content"]


def test_chunker_returns_whole_short_text_intact():
    assert chunk_text("short paragraph") == ["short paragraph"]


def test_chunker_splits_oversized_input():
    chunks = chunk_text("word " * 500, chunk_tokens=20, overlap_tokens=0)
    assert len(chunks) > 1


def test_chunker_preserves_all_words():
    text = " ".join(f"w{i}" for i in range(200))
    joined = " ".join(chunk_text(text, chunk_tokens=20, overlap_tokens=0))
    assert all(f"w{i}" in joined for i in range(200))


def test_chunker_handles_empty_input():
    assert chunk_text("") == []


def test_chunker_never_splits_mid_word():
    """A term cut across a boundary is unmatchable from either side."""
    words = [f"token{i}" for i in range(120)]
    for chunk in chunk_text(" ".join(words), chunk_tokens=15, overlap_tokens=0):
        for piece in chunk.split():
            assert piece in words
