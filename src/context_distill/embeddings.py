import os

from context_distill.config import VOYAGE_API_KEY_ENV

VOYAGE_MODEL = "voyage-3"


class EmbeddingsUnavailable(Exception):
    pass


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    api_key = os.environ.get(VOYAGE_API_KEY_ENV)
    if not api_key:
        raise EmbeddingsUnavailable(f"{VOYAGE_API_KEY_ENV} not set")

    try:
        import voyageai
    except ImportError as exc:
        raise EmbeddingsUnavailable(
            "voyageai not installed; run `pip install -e '.[rag-semantic]'`"
        ) from exc

    client = voyageai.Client(api_key=api_key)
    result = client.embed(texts, model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings
