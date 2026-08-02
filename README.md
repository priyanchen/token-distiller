# Token Distiller

Distills PDFs and photos into token-efficient text before they enter an LLM's context, packs repos Repomix-style, indexes ingested content for retrieval instead of raw dumping, and tracks session activity mode to bias what a context audit flags.

Pure Python. No shell scripts, no Node/TypeScript.

## Setup

```bash
brew install poppler   # required by pdf2image for PDF page rasterization
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Use a regular install, not `pip install -e .`. On this machine the editable install's
`_editable_impl_token_distiller.pth` was silently ignored by `site.py` (the file was
readable and its contents correct, and a byte-identical copy under a different filename
*was* honored — root cause unresolved), leaving `token_distiller` unimportable. A regular
install copies the package into `site-packages` and avoids the `.pth` indirection
entirely. Re-run `pip install .` after editing source.

Optional: set `ANTHROPIC_API_KEY` to enable vision-model fallback for low-confidence OCR pages. Set `VOYAGE_API_KEY` and `pip install ".[rag-semantic]"` to enable semantic (embedding) retrieval on top of the default BM25 keyword index.

## CLI

```bash
distill file <path>              # distill one PDF/photo
distill scan <dir>               # batch distill a directory
distill repo <dir>                # Repomix-style repo pack
distill index <dir>               # build a retrieval index
distill query "<question>"        # query the index
distill mode                      # current session activity mode
distill audit [path]              # CLAUDE.md/MEMORY.md structural audit
distill report                    # cumulative token/savings report
distill expand <handle>           # full distilled text for any handle (--list to browse)
distill install-hook              # wire the PreToolUse Read-interception hook into a project
```

## How it avoids losing anything

Every distillation is stored whole, keyed by a SHA-256 of the file's bytes, before any
shortening happens. Anything the hook shortens carries a handle, and `distill expand
<handle>` returns the complete text. Concretely:

- **Re-reads collapse.** Reading the same unchanged file twice in a session returns a
  one-line pointer the second time instead of the whole document again (7,634 → 287
  chars measured). Edit the file and the hash changes, so it is re-distilled in full —
  a stale cache can never be served.
- **Repeated page boilerplate is restated once.** A line must appear on ≥80% of pages to
  qualify, so a running copyright footer (25/25 pages) collapses while a structural
  marker like `Example:` (15/25) is left alone. Collapsed lines are listed at the top of
  the output.
- **Large documents defer rather than truncate.** Past `TOKEN_DISTILLER_LARGE_DOC_TOKENS`
  (default 8000) the hook returns a head plus retrieval instructions. Nothing is
  discarded — `distill expand` or `distill index` + `distill query` reach the rest.

Toggle any of it off: `TOKEN_DISTILLER_CACHE=0`, `TOKEN_DISTILLER_REREAD_COLLAPSE=0`,
`TOKEN_DISTILLER_BOILERPLATE=0`.

## A note on the numbers

`raw_tokens_est` models what the **host** pays to ingest the file, not what its text
alone would cost. Reading a PDF natively renders each page to an image and bills those
pixels on top of the text, so a 25-page text PDF costs ~60,000 tokens to read raw but
~1,800 distilled (33x). Scoring it as text-only would have reported a meaningless 1.0x.

## License

**PolyForm Noncommercial 1.0.0.** Free for personal, private, and noncommercial use —
no license purchase needed. Commercial use requires contacting Sri PriYa N. Chen
(p.chen@NeoclassicalPopArt.com) for a commercial license.
