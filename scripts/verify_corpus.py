#!/usr/bin/env python3
"""Runs distillation over every file in corpus/ (gitignored, never committed) and prints
aggregate stats -- page counts, timing, token ratios, methods used -- never document
content. Exists so a code change can be checked against real, messy documents in one
command, since every real bug found this session (OCR language, RTL reversal, a
form-feed edge case, an uncaught API error) was only visible on real files, never on
synthetic fixtures.

Usage:
    python3 scripts/verify_corpus.py            # local Tesseract only
    python3 scripts/verify_corpus.py --vision    # also allow vision fallback
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from token_distiller import pipeline
from token_distiller.config import DISTILLABLE_EXTENSIONS

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def main() -> int:
    allow_vision = "--vision" in sys.argv

    if not CORPUS_DIR.exists():
        print(f"no corpus/ directory at {CORPUS_DIR}")
        print("create it and drop PDFs/photos in -- it's gitignored, never committed")
        return 1

    files = sorted(p for p in CORPUS_DIR.rglob("*") if p.suffix.lower() in DISTILLABLE_EXTENSIONS)
    if not files:
        print(f"corpus/ exists but has no distillable files")
        return 1

    header = f"{'file':42} {'pages':>6} {'time':>7} {'raw':>10} {'distilled':>10} {'ratio':>7}  methods"
    print(header)
    print("-" * len(header))

    total_raw = total_distilled = 0
    failures: list[tuple[Path, Exception]] = []

    for path in files:
        try:
            t0 = time.perf_counter()
            result, _, _ = pipeline.distill(
                str(path), allow_vision=allow_vision, use_cache=False
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop the run
            failures.append((path, exc))
            print(f"{path.name[:42]:42}  FAILED: {type(exc).__name__}: {exc}")
            continue

        total_raw += result.raw_tokens_est
        total_distilled += result.distilled_tokens_est
        methods = ",".join(f"{k}:{v}" for k, v in result.method_counts().items())
        print(
            f"{path.name[:42]:42} {len(result.pages):6} {elapsed:6.1f}s "
            f"{result.raw_tokens_est:10,} {result.distilled_tokens_est:10,} "
            f"{result.compression_ratio:6.2f}x  {methods}"
        )

    print("-" * len(header))
    if total_distilled:
        print(
            f"{'TOTAL':42} {'':6} {'':7} {total_raw:10,} {total_distilled:10,} "
            f"{total_raw / max(1, total_distilled):6.2f}x"
        )

    if failures:
        print(f"\n{len(failures)} failure(s) -- see FAILED lines above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
