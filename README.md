# context-distill

Distills PDFs and photos into token-efficient text before they enter an LLM's context, packs repos Repomix-style, indexes ingested content for retrieval instead of raw dumping, and tracks session activity mode to bias what a context audit flags.

Pure Python. No shell scripts, no Node/TypeScript.

## Setup

```bash
brew install poppler   # required by pdf2image for PDF page rasterization
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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
