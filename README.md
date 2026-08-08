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
- **Pages with embedded images are flagged, not silently skipped.** Native-text
  extraction reads a page's text layer only — a diagram, chart, or illustration sitting
  next to that text isn't described anywhere in the output. Every native-text page's
  embedded-image count is recorded, and `pages_with_uncaptured_images()` surfaces it as
  one compact note (`"N page(s) contain embedded image(s) ... (pages 3, 12, 19, ...)"`)
  in `distill file`'s output, `--json`, and the hook-read response Claude actually sees
  — never one line per page. This is the one case the guarantee above doesn't fully
  cover: text that gets shortened always expands back in full via `distill expand`;
  image content on an otherwise-native-text page is flagged rather than distilled —
  there's no OCR/vision path that reaches it automatically today.

Toggle any of it off: `TOKEN_DISTILLER_CACHE=0`, `TOKEN_DISTILLER_REREAD_COLLAPSE=0`,
`TOKEN_DISTILLER_BOILERPLATE=0`.

## A note on the numbers

`raw_tokens_est` models what the **host** pays to ingest the file, not what its text
alone would cost. Reading a PDF natively renders each page to an image and bills those
pixels on top of the text, so a 25-page text PDF costs ~60,000 tokens to read raw but
~1,800 distilled (33x). Scoring it as text-only would have reported a meaningless 1.0x.

That 33x describes a *sparse* page, where a fixed per-page rendering cost dominates a
small amount of actual text — it is not a document-size-independent multiplier. A
densely-written page compresses by far less through this mechanism alone, because the
distilled side scales with real content: measured on a 765-page, prose-dense book, the
whole-document ratio was 4.77x (1,567,889 raw → 328,589 distilled), not 33x. That is
expected, not a regression — a page's text cannot be compressed below its own token
count by an extraction step that isn't lossy.

For a document that size, the number that actually matters is not the whole-document
ratio anyway. 328,589 distilled tokens is well past `TOKEN_DISTILLER_LARGE_DOC_TOKENS`
(default 8000), so the hook's large-document deferral fires: a live session reading that
765-page file through the hook receives a head plus a `distill index` / `distill query`
pointer, measured at ~1,675 tokens — roughly **940x** against the 1,567,889 raw cost.
(The payload embeds the file's absolute path, so the exact token count shifts a little
with where the file lives.)
For large documents, the deferral-and-retrieval path is where the real savings come
from; the raw-vs-native-text mechanism the 33x figure describes matters most for
documents small enough to be read in full.

## License

**PolyForm Noncommercial 1.0.0.** Free for personal, private, and noncommercial use —
no license purchase needed. Commercial use requires contacting Sri PriYa N. Chen
(priya.n.chen@gmail.com) for a commercial license.
