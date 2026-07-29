---
name: context-distill
description: Distill PDFs and photos into token-efficient text, pack repos Repomix-style, index content for retrieval, and check session activity mode. Use when the user wants to load a large PDF/photo/repo into context efficiently, or asks about token savings for documents.
---

# context-distill

A local, open-source pipeline (Python, MIT license) that keeps large PDFs, photos, and
codebases from bloating context. All commands run through the `distill` CLI, installed
in this plugin's bundled virtualenv.

## When Read intercepts a PDF/photo automatically

Once the hook is installed (`distill install-hook`, or bundled automatically with this
plugin), Claude Code's `Read` tool is intercepted for `.pdf`, `.jpg`, `.jpeg`, `.png`,
`.heic`, `.heif`, `.tiff`, `.tif`, `.bmp`, `.webp` files: instead of the raw file, the
result is distilled text plus a token-savings note. No user action needed.

## Manual commands

- `distill file <path> [--json]` — distill one PDF/photo, print the result and a
  before/after token estimate.
- `distill scan <dir> [--recursive]` — batch distill every PDF/photo in a directory.
- `distill repo <dir>` — pack a codebase into one token-counted file (Repomix-style),
  distilling any embedded PDFs/images along the way instead of dumping them raw.
- `distill index <dir>` then `distill query "<question>"` — build a local retrieval
  index (BM25, optionally semantic via Voyage AI) over distilled/packed content and
  pull only the relevant chunks instead of loading everything.
- `distill mode` — current session activity classification (code/debug/review/infra/general).
- `distill audit [path]` — structural CLAUDE.md/MEMORY.md audit (size, orphaned files,
  duplicate content), mode-aware.
- `distill report` — cumulative token/savings report across all past runs.

## Notes

- OCR runs locally via Tesseract, no API key required. Vision-model fallback (for
  low-confidence OCR or non-text content like charts/diagrams) requires
  `ANTHROPIC_API_KEY`; without it, the pipeline degrades gracefully to the OCR result
  with a warning rather than failing.
- Semantic retrieval requires `VOYAGE_API_KEY`; without it, retrieval falls back to
  BM25 keyword search only.
