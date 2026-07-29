"""Splits text into overlapping chunks at paragraph boundaries, falling back to a
hard split for any single paragraph that's larger than the chunk size on its own."""

from context_distill.config import CHARS_PER_TOKEN, CHUNK_TOKEN_OVERLAP, CHUNK_TOKEN_SIZE


def chunk_text(
    text: str,
    chunk_tokens: int = CHUNK_TOKEN_SIZE,
    overlap_tokens: int = CHUNK_TOKEN_OVERLAP,
) -> list[str]:
    chunk_chars = int(chunk_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > chunk_chars:
            flush()
            step = max(1, chunk_chars - overlap_chars)
            for i in range(0, len(para), step):
                piece = para[i : i + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            flush()
            current = para

    flush()

    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap_chars:]
        overlapped.append(f"{tail}\n\n{chunks[i]}")
    return overlapped
