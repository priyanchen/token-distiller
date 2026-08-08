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

Optional: set **`TOKEN_DISTILLER_ANTHROPIC_API_KEY`** to enable vision-model fallback for
figures and pages OCR can't read. Prefer that name over plain `ANTHROPIC_API_KEY` — a host
agent (Claude Code included) may also read `ANTHROPIC_API_KEY` and switch from subscription
authentication to per-token API billing if it finds one. The scoped name is checked first,
so this tool gets a key without changing anyone else's auth.

Set `VOYAGE_API_KEY` and `pip install ".[rag-semantic]"` to enable semantic (embedding)
retrieval on top of the default BM25 keyword index.

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
distill compress                  # compress verbose command output read from stdin
distill install-hook              # wire the PreToolUse Read-interception hook into a project
```

## Compressing command output

Verbose CLI output is the one context cost the document pipeline doesn't touch. Pipe it in:

```bash
pytest -q | distill compress --stats
git status | distill compress
npm install | distill compress
```

Measured on this repo's own output: a 169-test `pytest` run goes **289 → 8 tokens (97%)**,
keeping the failure list, the first assertion detail, and the summary line while dropping
the wall of dots. Plain `git status` goes **153 → 58 tokens (62%)**, grouped by state.

Two properties worth knowing:

- **It never inflates.** `git status --porcelain` is already denser than any per-state
  summary of it, so when compression would produce more text than it consumed, the original
  is returned unchanged.
- **It never executes anything.** `distill compress` reads stdin and writes stdout. The
  alternative — a hook that rewrites your Bash command to route it through a wrapper —
  means building shell strings out of model-supplied input, which is exactly where command
  injection lives. Piping output that you already ran has no such surface. Automatic
  interception is deliberately not implemented for that reason.

## Verified in a live session

The `Read` interception was checked against a real Claude Code session, not only against
synthetic hook payloads. A one-page PDF holding a canary string was read through the hook:
the session reported `2,572 → 38 tokens` and the model quoted the canary correctly,
confirming the substituted text is what actually reaches it. A second run asking for one
specific page of a four-page PDF passed straight through to the native ranged read with no
hook note, confirming a page range is never answered with whole-document content.

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
- **Embedded figures are read, not skipped.** Native-text extraction sees a page's text
  layer only, so a diagram sitting beside that text would otherwise go unread. Each
  embedded figure is cropped out by its bounding box and put through the same OCR →
  vision chain used for scanned pages, then written into the output labelled
  `[figure N on page M]`. Cropping to the figure matters: the surrounding prose is
  already captured losslessly, so including it would pay vision tokens to re-read text
  we already have. Hairline rules and background strips are skipped via
  `TOKEN_DISTILLER_FIGURE_MIN_SIDE_PT` (default 48pt). `--no-figures` turns it off.
- **A weak OCR pass is retried on a preprocessed copy.** Figures cropped from a PDF are
  often below the ~300 DPI Tesseract expects and sit on a tinted panel, which is exactly
  when raw OCR returns nothing. A second attempt greyscales, stretches contrast, upscales
  small crops, and binarizes with an Otsu threshold. It is a retry rather than the default
  because binarizing can destroy anti-aliased text that read fine raw, so the preprocessed
  pass has to win on word count and confidence to be used. Measured on a real 765-page
  book: of 13 figures that raw OCR could not read at all, **10 were recovered** — one at
  confidence 96, transcribing `Market Research / Competitive Analysis / SWOT Analysis /
  Goal Setting / Resource Allocation`. That book ends at 60 of 63 figures read, with no
  API key.
- **Figures that still can't be read are flagged, never dropped silently.** A purely
  graphical diagram with no legible labels yields nothing from OCR, and without
  `ANTHROPIC_API_KEY` there's no vision fallback to describe it. Those pages stay in
  `pages_with_uncaptured_images()` and surface as one compact note — never one line per
  page — in `distill file`, `--json`, and the hook-read response. Set `ANTHROPIC_API_KEY`
  to close the remainder.

## What leaves your machine

With no API keys set, nothing does. OCR runs locally through Tesseract and retrieval runs
locally through BM25, so the default configuration makes **zero network calls**.

Two features are opt-in, and they send different things to different companies:

| Enabled by | Goes to | What is sent |
|---|---|---|
| `TOKEN_DISTILLER_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`) | Anthropic | A PNG of a **single cropped figure**, plus a fixed prompt. No source code, no file paths, no surrounding page text. |
| `VOYAGE_API_KEY` + `pip install ".[rag-semantic]"` | Voyage AI | **Chunk text** from whatever you indexed. If you indexed a repo pack, that includes your source code. |

The Voyage path is the one to think hardest about — it is a separate company under separate
terms, and it is the only path that can transmit code. It is off unless you both install the
extra and set the key; BM25 retrieval works without it.

On the Anthropic path, [their commercial terms](https://www.anthropic.com/legal/commercial-terms)
state that the customer "retains all rights to its Inputs", "owns its Outputs", that
Anthropic "disclaims any rights it receives to the Customer Content", and that Anthropic
"may not train models on Customer Content from Services". Read them yourself rather than
relying on this summary; this is not legal advice.

Turn figure reading off entirely with `--no-figures` or
`TOKEN_DISTILLER_DESCRIBE_FIGURES=0`, and it will never reach for the vision model.

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
