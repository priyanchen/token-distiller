"""CLI dispatch. `pipeline`/`storage` are imported lazily inside each command
function, not at module top-level: `hook-read` runs on every Claude Code Read
call, and the extension check must happen before pdfplumber/pytesseract/anthropic
ever get imported, or every Read in every session pays that cost."""

import argparse
import json
import sys
from pathlib import Path

from context_distill.config import DISTILLABLE_EXTENSIONS


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
        "text": result.text,
    }


def _mean_ocr_confidence(result):
    confs = [p.ocr_confidence for p in result.pages if p.ocr_confidence is not None]
    return sum(confs) / len(confs) if confs else None


def _log_run(result, trigger: str) -> None:
    from context_distill import storage

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
    from context_distill import pipeline

    result = pipeline.distill(args.path, allow_vision=not args.no_vision)
    if not args.no_save_log:
        _log_run(result, trigger="cli")

    if args.json:
        print(json.dumps(_result_to_dict(result), indent=2))
    else:
        print(f"{result.source_path}  ({result.source_type}, {len(result.pages)} page(s))")
        print(f"  methods: {result.method_counts()}")
        print(
            f"  tokens: {result.raw_tokens_est} -> {result.distilled_tokens_est}  "
            f"({result.compression_ratio:.1f}x compression)"
        )
        for w in result.warnings:
            print(f"  warning: {w}")

    if args.out:
        Path(args.out).write_text(result.text)
    return 0


def cmd_scan(args) -> int:
    from context_distill import pipeline

    root = Path(args.dir)
    glob = root.rglob("*") if args.recursive else root.glob("*")
    paths = [p for p in glob if p.suffix.lower() in DISTILLABLE_EXTENSIONS]

    total_raw = total_distilled = 0
    ok = failed = 0
    for p in paths:
        try:
            result = pipeline.distill(str(p), allow_vision=not args.no_vision)
            _log_run(result, trigger="cli")
            total_raw += result.raw_tokens_est
            total_distilled += result.distilled_tokens_est
            ok += 1
            print(f"ok    {p}  {result.raw_tokens_est} -> {result.distilled_tokens_est}")
        except Exception as exc:
            failed += 1
            print(f"error {p}  {exc}", file=sys.stderr)

    print(f"\n{ok} distilled, {failed} failed. tokens: {total_raw} -> {total_distilled}")
    return 0 if failed == 0 else 1


def cmd_report(args) -> int:
    from context_distill import storage

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
    file_path = payload.get("tool_input", {}).get("file_path", "")
    suffix = Path(file_path).suffix.lower()

    if suffix not in DISTILLABLE_EXTENSIONS:
        return 0  # pass through, no output, no heavy imports ever touched

    from context_distill import pipeline

    try:
        result = pipeline.distill(file_path, allow_vision=not args.no_vision)
    except Exception as exc:
        print(json.dumps({"warning": f"context-distill failed on {file_path}: {exc}"}))
        return 0  # fail open: let Claude Code's normal Read proceed

    _log_run(result, trigger="hook")

    note = (
        f"[context-distill] {file_path}: {result.raw_tokens_est} -> "
        f"{result.distilled_tokens_est} tokens ({result.compression_ratio:.1f}x). "
        f"Methods: {result.method_counts()}."
    )
    body = f"{note}\n\n{result.text}"

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


def cmd_repo(args) -> int:
    from context_distill import repo_pack

    result = repo_pack.pack(
        args.dir, include=args.include, exclude=args.exclude, allow_vision=not args.no_vision
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


def cmd_install_hook(args) -> int:
    from context_distill import hook_installer

    target_path = hook_installer.resolve_target(args.target, args.project_dir)
    print(hook_installer.install(target_path, dry_run=args.dry_run))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distill")
    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser("file", help="distill one PDF/photo")
    p_file.add_argument("path")
    p_file.add_argument("--out")
    p_file.add_argument("--json", action="store_true")
    p_file.add_argument("--no-vision", action="store_true")
    p_file.add_argument("--no-save-log", action="store_true")
    p_file.set_defaults(func=cmd_file)

    p_scan = sub.add_parser("scan", help="batch distill a directory")
    p_scan.add_argument("dir")
    p_scan.add_argument("--recursive", action="store_true")
    p_scan.add_argument("--no-vision", action="store_true")
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
    p_repo.set_defaults(func=cmd_repo)

    p_install = sub.add_parser("install-hook", help="wire the PreToolUse Read hook into settings.json")
    p_install.add_argument("--target", choices=["project", "global"], default="project")
    p_install.add_argument("--project-dir")
    p_install.add_argument("--dry-run", action="store_true")
    p_install.set_defaults(func=cmd_install_hook)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
