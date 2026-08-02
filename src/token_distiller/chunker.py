"""Splits text into overlapping chunks at paragraph boundaries, falling back to a
hard split for any single paragraph that's larger than the chunk size on its own."""

from token_distiller.config import CHARS_PER_TOKEN, CHUNK_TOKEN_OVERLAP, CHUNK_TOKEN_SIZE


def _split_long_paragraph(para: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Break on whitespace rather than at a fixed character offset. A blind slice cuts
    words in half, and a term severed across a boundary is unmatchable by either chunk."""
    words = para.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        if current and length + len(word) + 1 > chunk_chars:
            chunks.append(" ".join(current))
            if overlap_chars > 0:
                carried: list[str] = []
                carried_len = 0
                for prev in reversed(current):
                    if carried_len + len(prev) + 1 > overlap_chars:
                        break
                    carried.insert(0, prev)
                    carried_len += len(prev) + 1
                current = carried
                length = carried_len
            else:
                current = []
                length = 0
        current.append(word)
        length += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


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
            chunks.extend(_split_long_paragraph(para, chunk_chars, overlap_chars))
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
