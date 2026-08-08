---
name: token-distiller
description: Manual token-distiller commands -- pack a repo Repomix-style, build/query a local retrieval index over distilled content, check session activity mode, or audit CLAUDE.md/MEMORY.md files. Not for single PDF/photo reads -- once this plugin is installed, its hook already intercepts Read on those files automatically; call Read directly and skip this skill.
---

# Token Distiller

A local, open-source pipeline (Python) that keeps large PDFs, photos, and codebases from
bloating context. All commands run through the `distill` CLI, installed in this plugin's
bundled virtualenv.

## Don't use this skill for a single PDF/photo

Once the hook is installed (`distill install-hook`, or bundled automatically with this
plugin), Claude Code's `Read` tool is intercepted for `.pdf`, `.jpg`, `.jpeg`, `.png`,
`.heic`, `.heif`, `.tiff`, `.tif`, `.bmp`, `.webp` files: instead of the raw file, the
result is distilled text plus a token-savings note, with no shell command and no approval
prompt. Call `Read` on the file directly for that case — invoking this skill or running
`distill file` manually only adds an extra shell step (and an approval prompt) to
something the hook already does silently.

Re-reading the same unchanged file later in a session returns a short pointer rather than
repeating the text. That is a context optimization, not a loss — if the full text is
actually needed again (for example after a compaction dropped it), run
`distill expand <handle>` using the handle named in the pointer. Edited files are always
re-distilled in full, so a pointer never refers to stale content.

## Manual commands

Use this skill for the commands below — they have no hook equivalent.

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
- `distill expand <handle>` — the full distilled text behind any handle. Whenever a
  hook result says a document was collapsed, deferred, or truncated, this returns the
  complete version. `--list` shows every handle available.

## Notes

- OCR runs locally via Tesseract, no API key required. Vision-model fallback (for
  low-confidence OCR or non-text content like charts/diagrams) requires
  `ANTHROPIC_API_KEY`; without it, the pipeline degrades gracefully to the OCR result
  with a warning rather than failing.
- Semantic retrieval requires `VOYAGE_API_KEY`; without it, retrieval falls back to
  BM25 keyword search only.
