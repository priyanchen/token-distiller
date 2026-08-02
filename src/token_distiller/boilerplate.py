"""Collapses per-page boilerplate (running headers, footers, copyright lines) into a
single recorded note.

The page-fraction threshold is deliberately conservative. On a real 25-page lecture deck
the copyright footer appears on 25/25 pages and collapses; "Example:" appears on 15/25
and does not, because a structural marker carries position-specific meaning that a
running footer does not. Every collapsed line is reported back in the manifest, so the
transformation is auditable and nothing becomes unknowable.
"""

from collections import Counter

from token_distiller.config import (
    BOILERPLATE_MAX_LINE_CHARS,
    BOILERPLATE_MIN_PAGES,
    BOILERPLATE_PAGE_FRACTION,
)


def find_boilerplate(page_texts: list[str]) -> list[str]:
    page_count = len(page_texts)
    if page_count < BOILERPLATE_MIN_PAGES:
        return []

    counts: Counter[str] = Counter()
    for text in page_texts:
        for line in {ln.strip() for ln in text.split("\n") if ln.strip()}:
            if len(line) <= BOILERPLATE_MAX_LINE_CHARS:
                counts[line] += 1

    threshold = max(BOILERPLATE_MIN_PAGES, page_count * BOILERPLATE_PAGE_FRACTION)
    return sorted(line for line, n in counts.items() if n >= threshold)


def strip_boilerplate(page_texts: list[str]) -> tuple[list[str], list[dict]]:
    lines_to_strip = find_boilerplate(page_texts)
    if not lines_to_strip:
        return page_texts, []

    strip_set = set(lines_to_strip)
    occurrences: Counter[str] = Counter()
    stripped_pages = []

    for text in page_texts:
        kept = []
        for line in text.split("\n"):
            if line.strip() in strip_set:
                occurrences[line.strip()] += 1
                continue
            kept.append(line)
        stripped_pages.append("\n".join(kept).strip())

    manifest = [
        {"line": line, "occurrences": occurrences[line]}
        for line in lines_to_strip
        if occurrences[line] > 0
    ]
    return stripped_pages, manifest


def render_manifest(manifest: list[dict]) -> str:
    if not manifest:
        return ""
    parts = ["[token-distiller] repeated on every page, listed once instead of inline:"]
    for entry in manifest:
        parts.append(f"  ({entry['occurrences']}x) {entry['line']}")
    return "\n".join(parts)
