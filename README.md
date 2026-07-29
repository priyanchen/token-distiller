# context-distill

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
`_editable_impl_context_distill.pth` was silently ignored by `site.py` (the file was
readable and its contents correct, and a byte-identical copy under a different filename
*was* honored — root cause unresolved), leaving `context_distill` unimportable. A regular
install copies the package into `site-packages` and avoids the `.pth` indirection
entirely. Re-run `pip install .` after editing source.

Optional: set `ANTHROPIC_API_KEY` to enable vision-model fallback for low-confidence OCR pages. Set `VOYAGE_API_KEY` and `pip install -e ".[rag-semantic]"` to enable semantic (embedding) retrieval on top of the default BM25 keyword index.

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
distill install-hook              # wire the PreToolUse Read-interception hook into a project
```

## License

MIT. Copyright Sri PriYa N. Chen.
