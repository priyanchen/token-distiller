"""CLI dispatch. `pipeline`/`storage` are imported lazily inside each command
function, not at module top-level: `hook-read` runs on every Claude Code Read
call, and the extension check must happen before pdfplumber/pytesseract/anthropic
ever get imported, or every Read in every session pays that cost."""

import argparse
import json
import sys
from pathlib import Path

from token_distiller.config import DISTILLABLE_EXTENSIONS


def _result_to_dict(result) -> dict:
    return {
        "source_path": result.source_path,
        "source_type": result.source_type,
        "page_count": len(result.pages),
        "method_counts": result.method_counts(),
        "raw_tokens_est": result.raw_tokens_est,
        "distilled_tokens_est": result.distilled_tokens_est,
        "compression_ratio": round(result.compression_ratio, 2),
        "duration_ms": result.duration_ms,
        "warnings": result.warnings,
        "pages_with_uncaptured_images": result.pages_with_uncaptured_images(),
        "pages_with_described_figures": result.pages_with_described_figures(),
        "figure_count": result.figure_count,
        "text": result.text,
    }


def _page_list(pages: list[int], one_indexed: bool = True) -> str:
    offset = 1 if one_indexed else 0
    shown = ", ".join(str(p + offset) for p in pages[:8])
    return shown + (f", +{len(pages) - 8} more" if len(pages) > 8 else "")


def _figures_note(result) -> str | None:
    """One compact line, never one per page -- enumerating pages bloats exactly the thing
    this tool exists to shrink."""
    described = result.pages_with_described_figures()
    if not described:
        return None
    return (
        f"{result.figure_count} embedded figure(s) on {len(described)} page(s) were read "
        f"and transcribed into the text below, labelled [figure N on page M] "
        f"(pages {_page_list(described)})"
    )


def _uncaptured_images_note(result, one_indexed: bool = True) -> str | None:
    pages = result.pages_with_uncaptured_images()
    if not pages:
        return None
    return (
        f"{len(pages)} page(s) contain embedded image(s) (diagrams/figures/"
        f"illustrations) that could not be read -- their content isn't "
        f"in the distilled text (pages {_page_list(pages, one_indexed)})"
    )


def _mean_ocr_confidence(result):
    confs = [p.ocr_confidence for p in result.pages if p.ocr_confidence is not None]
    return sum(confs) / len(confs) if confs else None


def _log_run(result, trigger: str) -> None:
    from token_distiller import storage

    method_counts = result.method_counts()
    storage.insert_run(
        source_path=result.source_path,
        source_type=result.source_type,
        trigger=trigger,
        page_count=len(result.pages),
        pages_native_text=method_counts.get("native_text", 0),
        pages_ocr=method_counts.get("ocr", 0) + method_counts.get("ocr_degraded", 0),
        pages_vision_fallback=method_counts.get("vision", 0),
        ocr_mean_confidence=_mean_ocr_confidence(result),
        raw_tokens_est=result.raw_tokens_est,
        distilled_tokens_est=result.distilled_tokens_est,
        token_estimation_method="chars4+image_formula",
        compression_ratio=result.compression_ratio,
        vision_api_calls=method_counts.get("vision", 0),
        vision_model=None,
        duration_ms=result.duration_ms,
        status="ok",
        error_message=None,
        output_path=None,
    )


def cmd_file(args) -> int:
    from token_distiller import pipeline

    result, handle, cached = pipeline.distill(
        args.path,
        allow_vision=not args.no_vision,
        use_cache=not args.no_cache,
        describe_figures=not args.no_figures,
    )
    if not args.no_save_log:
        _log_run(result, trigger="cli")

    exact_tokens = exact_tokens_error = None
    if args.accurate_tokens:
        from token_distiller.exact_tokens import ExactCountUnavailable, count_tokens_exact

        try:
            exact_tokens = count_tokens_exact(result.rendered_text)
        except ExactCountUnavailable as exc:
            exact_tokens_error = str(exc)

    if args.json:
        payload = _result_to_dict(result)
        payload["cache_handle"] = handle
        payload["cache_hit"] = cached
        if args.accurate_tokens:
            payload["distilled_tokens_exact"] = exact_tokens
            payload["accurate_tokens_error"] = exact_tokens_error
        print(json.dumps(payload, indent=2))
    else:
        suffix = "  [cached]" if cached else ""
        print(
            f"{result.source_path}  ({result.source_type}, {len(result.pages)} page(s)){suffix}"
        )
        print(f"  methods: {result.method_counts()}")
        print(
            f"  tokens: {result.raw_tokens_est} -> {result.distilled_tokens_est}  "
            f"({result.compression_ratio:.1f}x compression)"
        )
        if args.accurate_tokens:
            if exact_tokens is not None:
                print(f"  exact distilled tokens: {exact_tokens}  (estimate was {result.distilled_tokens_est})")
            else:
                print(f"  exact tokens unavailable: {exact_tokens_error}")
        if result.boilerplate:
            collapsed = sum(e["occurrences"] for e in result.boilerplate)
            print(f"  boilerplate: {len(result.boilerplate)} line(s) collapsed, {collapsed} occurrence(s)")
        if handle is not None:
            print(f"  handle: {handle}  (distill expand {handle})")
        for w in result.warnings:
            print(f"  warning: {w}")
        figures = _figures_note(result)
        if figures:
            print(f"  figures: {figures}")
        note = _uncaptured_images_note(result)
        if note:
            print(f"  note: {note}")

    if args.out:
        Path(args.out).write_text(result.rendered_text)
    return 0


def cmd_scan(args) -> int:
    from token_distiller import pipeline

    root = Path(args.dir)
    glob = root.rglob("*") if args.recursive else root.glob("*")
    paths = [p for p in glob if p.suffix.lower() in DISTILLABLE_EXTENSIONS]

    total_raw = total_distilled = 0
    ok = failed = 0
    for p in paths:
        try:
            result, _, cached = pipeline.distill(
                str(p),
                allow_vision=not args.no_vision,
                describe_figures=not args.no_figures,
            )
            _log_run(result, trigger="cli")
            total_raw += result.raw_tokens_est
            total_distilled += result.distilled_tokens_est
            ok += 1
            tag = " [cached]" if cached else ""
            print(f"ok    {p}  {result.raw_tokens_est} -> {result.distilled_tokens_est}{tag}")
        except Exception as exc:
            failed += 1
            print(f"error {p}  {exc}", file=sys.stderr)

    print(f"\n{ok} distilled, {failed} failed. tokens: {total_raw} -> {total_distilled}")
    return 0 if failed == 0 else 1


def cmd_report(args) -> int:
    from token_distiller import storage

    summary = storage.savings_summary(since=args.since)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"runs: {summary['run_count']}")
        print(f"raw tokens:       {summary['raw_tokens_est']}")
        print(f"distilled tokens: {summary['distilled_tokens_est']}")
        print(f"tokens saved:     {summary['tokens_saved_est']}")
        print(f"compression:      {summary['compression_ratio']:.1f}x")
    return 0


def cmd_hook_read(args) -> int:
    """Invoked by Claude Code's PreToolUse hook on every Read call. The extension
    check below must stay dependency-free so the overwhelming majority of calls
    (.py, .md, etc.) pass through with negligible added latency."""
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id")
    suffix = Path(file_path).suffix.lower()

    if suffix not in DISTILLABLE_EXTENSIONS:
        return 0  # pass through, no output, no heavy imports ever touched

    # A ranged read asks for a specific slice; distillation is whole-file, so answering
    # with the whole document (or its head) would silently return the wrong content.
    # Pass through and let the native ranged read do its job.
    if any(tool_input.get(k) is not None for k in ("pages", "offset", "limit")):
        return 0

    from token_distiller import cache, pipeline
    from token_distiller.config import (
        LARGE_DOC_HEAD_TOKENS,
        LARGE_DOC_TOKEN_THRESHOLD,
        REREAD_COLLAPSE_ENABLED,
    )
    from token_distiller.tokens import estimate_text_tokens

    try:
        hash_value = cache.content_hash(file_path)
        already_seen = REREAD_COLLAPSE_ENABLED and cache.seen_in_session(session_id, hash_value)
        result, handle, was_cached = pipeline.distill(
            file_path, allow_vision=not args.no_vision
        )
    except Exception as exc:
        print(json.dumps({"warning": f"token-distiller failed on {file_path}: {exc}"}))
        return 0  # fail open: let Claude Code's normal Read proceed

    if not was_cached:
        _log_run(result, trigger="hook")

    header = (
        f"[token-distiller] {file_path}: {result.raw_tokens_est} -> "
        f"{result.distilled_tokens_est} tokens ({result.compression_ratio:.1f}x). "
        f"Methods: {result.method_counts()}."
    )
    figures_note = _figures_note(result)
    if figures_note:
        header += f" Figures: {figures_note}."
    images_note = _uncaptured_images_note(result)
    if images_note:
        header += f" Note: {images_note}."
    expand_hint = f"Full text: run `distill expand {handle}`." if handle is not None else ""

    if already_seen:
        body = (
            f"{header}\n\nUnchanged since this session already read it, so the text is not "
            f"repeated here. {expand_hint}"
        )
    else:
        full = result.rendered_text
        if estimate_text_tokens(full) > LARGE_DOC_TOKEN_THRESHOLD:
            head = full[: int(LARGE_DOC_HEAD_TOKENS * 4)]
            body = (
                f"{header}\n\nLarge document — showing the first ~{LARGE_DOC_HEAD_TOKENS} "
                f"tokens. Nothing was discarded: {expand_hint} "
                f"Or search it with `distill index` + `distill query`.\n\n{head}\n\n"
                f"[truncated here — {expand_hint}]"
            )
        else:
            body = f"{header}\n\n{full}"

    cache.mark_seen(session_id, hash_value)

    # Verified against Claude Code's current hook docs: hookSpecificOutput /
    # permissionDecision="deny" / permissionDecisionReason is the supported
    # substitution schema for PreToolUse.
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": body,
        }
    }
    print(json.dumps(output))
    return 0


def cmd_expand(args) -> int:
    from token_distiller import cache

    if args.list:
        entries = cache.list_entries(limit=args.limit)
        if args.json:
            print(json.dumps(entries, indent=2))
        elif not entries:
            print("nothing distilled yet")
        else:
            for e in entries:
                print(f"{e['id']:>5}  {e['source_type']:<5}  {e['created_at'][:19]}  {e['source_path']}")
        return 0

    if args.handle is None:
        print("give a handle, or --list", file=sys.stderr)
        return 1

    result = cache.get_by_id(args.handle)
    if result is None:
        print(f"no distillation with handle {args.handle}", file=sys.stderr)
        return 1

    print(result.rendered_text)
    return 0


def cmd_repo(args) -> int:
    from token_distiller import repo_pack

    result = repo_pack.pack(
        args.dir,
        include=args.include,
        exclude=args.exclude,
        allow_vision=not args.no_vision,
        describe_figures=not args.no_figures,
    )
    output = repo_pack.render(result, style=args.style)

    if args.output:
        Path(args.output).write_text(output)
        print(f"wrote {args.output}  ({len(result.files)} files, ~{result.total_tokens_est} tokens)")
    else:
        print(output)

    if result.skipped:
        print(f"\nskipped {len(result.skipped)} file(s):", file=sys.stderr)
        for s in result.skipped[:20]:
            print(f"  {s}", file=sys.stderr)
    return 0


def cmd_index(args) -> int:
    from token_distiller import chunker, embeddings, index_store, repo_pack

    result = repo_pack.pack(
        args.dir,
        allow_vision=not args.no_vision,
        describe_figures=not args.no_figures,
    )
    total_chunks = 0
    embed_model = None
    embed_warned = False

    for f in result.files:
        index_store.clear_source(f.path)
        pieces = chunker.chunk_text(f.text)
        if not pieces:
            continue
        chunk_ids = index_store.add_chunks(f.path, pieces)
        total_chunks += len(chunk_ids)

        if not args.no_semantic:
            try:
                vectors = embeddings.embed_texts(pieces)
                index_store.add_embeddings(chunk_ids, vectors, embeddings.VOYAGE_MODEL)
                embed_model = embeddings.VOYAGE_MODEL
            except embeddings.EmbeddingsUnavailable as exc:
                if not embed_warned:
                    print(f"semantic embeddings skipped: {exc}", file=sys.stderr)
                    embed_warned = True

    mode = f"embeddings via {embed_model}" if embed_model else "BM25 keyword index only"
    print(f"indexed {len(result.files)} file(s), {total_chunks} chunk(s) ({mode})")
    return 0


def cmd_query(args) -> int:
    from token_distiller import retrieval

    results = retrieval.query(args.question, top_k=args.top_k, use_semantic=not args.no_semantic)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if results else 1

    if not results:
        print("no indexed content found — run `distill index <dir>` first")
        return 1

    for r in results:
        preview = r["text"][:300].strip()
        if len(r["text"]) > 300:
            preview += "..."
        print(f"[{r['score']:.3f}] {r['source_path']} (chunk {r['chunk_index']})")
        print(f"  {preview}\n")
    return 0


def cmd_hook_activity(args) -> int:
    """Invoked by Claude Code's PostToolUse hook on every tool call. Records the
    call for activity-mode classification; never blocks or alters anything."""
    payload = json.load(sys.stdin)
    from token_distiller import activity

    activity.record(
        tool_name=payload.get("tool_name", ""),
        tool_input=payload.get("tool_input") or {},
        session_id=payload.get("session_id"),
    )
    return 0


def cmd_mode(args) -> int:
    from token_distiller import activity

    result = activity.current_mode(session_id=args.session_id)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"mode: {result['mode']}  "
            f"(window={result['window_size']}, counts={result['bucket_counts']})"
        )
    return 0


def cmd_audit(args) -> int:
    from token_distiller import activity, session_audit

    mode = activity.current_mode()["mode"]
    findings = session_audit.audit(args.path, memory_dir=args.memory_dir, mode=mode)

    if args.json:
        print(json.dumps(findings, indent=2))
        return 0

    print(f"session mode: {mode}")
    for f in findings["files"]:
        print(f"  {f['path']}: {f['bytes']} bytes, ~{f['tokens_est']} tokens, {f['lines']} lines")

    sections = [
        ("duplicate content", findings["duplicates"], "occurrences"),
        ("orphaned memory files", findings["orphaned_memory_files"], "path"),
    ]
    if mode in ("review", "infra"):
        sections.reverse()

    for label, items, kind in sections:
        print(f"\n{label}: {len(items)} finding(s)")
        for item in items[:10]:
            if kind == "occurrences":
                print(f"  {item['occurrences']}x: {item['snippet']}")
            else:
                print(f"  {item}")

    if not args.memory_dir:
        print("\n(pass --memory-dir to also check for orphaned per-topic memory files)")
    return 0


def cmd_compress(args) -> int:
    """Reads command output on stdin. Never runs a command: the caller pipes into this, so
    no shell string is ever constructed from untrusted input."""
    from token_distiller import bash_compress
    from token_distiller.tokens import estimate_text_tokens

    raw = sys.stdin.read()
    compressed = bash_compress.compress(raw, kind=args.kind)

    if args.stats:
        before = estimate_text_tokens(raw)
        after = estimate_text_tokens(compressed)
        saved = (1 - after / before) * 100 if before else 0.0
        print(
            f"[token-distiller] {before} -> {after} tokens ({saved:.0f}% saved)",
            file=sys.stderr,
        )
    print(compressed)
    return 0


def cmd_install_hook(args) -> int:
    from token_distiller import hook_installer

    target_path = hook_installer.resolve_target(args.target, args.project_dir)
    print(
        hook_installer.install(
            target_path, dry_run=args.dry_run, include_images=args.images
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distill")
    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser("file", help="distill one PDF/photo")
    p_file.add_argument("path")
    p_file.add_argument("--out")
    p_file.add_argument("--json", action="store_true")
    p_file.add_argument("--no-vision", action="store_true")
    p_file.add_argument("--no-cache", action="store_true")
    p_file.add_argument(
        "--no-figures",
        action="store_true",
        help="skip reading embedded figures (faster; leaves diagram content uncaptured)",
    )
    p_file.add_argument("--no-save-log", action="store_true")
    p_file.add_argument(
        "--accurate-tokens",
        action="store_true",
        help="get an exact distilled-token count via the Anthropic count_tokens API "
        "(needs an API key; adds a network call; free to call)",
    )
    p_file.set_defaults(func=cmd_file)

    p_expand = sub.add_parser("expand", help="retrieve the full distilled text for a handle")
    p_expand.add_argument("handle", nargs="?", type=int)
    p_expand.add_argument("--list", action="store_true")
    p_expand.add_argument("--limit", type=int, default=50)
    p_expand.add_argument("--json", action="store_true")
    p_expand.set_defaults(func=cmd_expand)

    p_scan = sub.add_parser("scan", help="batch distill a directory")
    p_scan.add_argument("dir")
    p_scan.add_argument("--recursive", action="store_true")
    p_scan.add_argument("--no-vision", action="store_true")
    p_scan.add_argument(
        "--no-figures",
        action="store_true",
        help="skip reading embedded figures (faster; leaves diagram content uncaptured)",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_report = sub.add_parser("report", help="cumulative token/savings report")
    p_report.add_argument("--since")
    p_report.add_argument("--json", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_hook = sub.add_parser("hook-read", help="internal: PreToolUse Read hook entry point")
    p_hook.add_argument("--no-vision", action="store_true")
    p_hook.set_defaults(func=cmd_hook_read)

    p_repo = sub.add_parser("repo", help="Repomix-style repo pack")
    p_repo.add_argument("dir")
    p_repo.add_argument("--include")
    p_repo.add_argument("--exclude")
    p_repo.add_argument("--style", choices=["xml", "markdown"], default="markdown")
    p_repo.add_argument("--output")
    p_repo.add_argument("--no-vision", action="store_true")
    p_repo.add_argument(
        "--no-figures",
        action="store_true",
        help="skip reading embedded figures (faster; leaves diagram content uncaptured)",
    )
    p_repo.set_defaults(func=cmd_repo)

    p_index = sub.add_parser("index", help="build a retrieval index over a directory")
    p_index.add_argument("dir")
    p_index.add_argument("--no-vision", action="store_true")
    p_index.add_argument(
        "--no-figures",
        action="store_true",
        help="skip reading embedded figures (faster; leaves diagram content uncaptured)",
    )
    p_index.add_argument("--no-semantic", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="query the retrieval index")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--json", action="store_true")
    p_query.add_argument("--no-semantic", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_hook_activity = sub.add_parser(
        "hook-activity", help="internal: PostToolUse activity-tracking hook entry point"
    )
    p_hook_activity.set_defaults(func=cmd_hook_activity)

    p_mode = sub.add_parser("mode", help="current session activity-mode classification")
    p_mode.add_argument("--session-id")
    p_mode.add_argument("--json", action="store_true")
    p_mode.set_defaults(func=cmd_mode)

    p_audit = sub.add_parser("audit", help="CLAUDE.md/MEMORY.md structural audit")
    p_audit.add_argument("path", nargs="?", default=".")
    p_audit.add_argument("--memory-dir")
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    p_compress = sub.add_parser(
        "compress", help="compress verbose command output read from stdin"
    )
    p_compress.add_argument(
        "--kind",
        choices=["auto", "git-status", "pytest", "install", "generic"],
        default="auto",
    )
    p_compress.add_argument("--stats", action="store_true", help="report savings on stderr")
    p_compress.set_defaults(func=cmd_compress)

    p_install = sub.add_parser("install-hook", help="wire the PreToolUse Read hook into settings.json")
    p_install.add_argument("--target", choices=["project", "global"], default="project")
    p_install.add_argument("--project-dir")
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument(
        "--images",
        action="store_true",
        help="also intercept image reads (loses the model's own vision on them)",
    )
    p_install.set_defaults(func=cmd_install_hook)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
