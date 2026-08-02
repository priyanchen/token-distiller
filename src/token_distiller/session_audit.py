"""Structural CLAUDE.md/MEMORY.md audit: size, duplicate content, orphaned memory
files. `--memory-dir` has no default — there's no established Claude Code convention
for where per-topic memory files live, so guessing one would be worse than requiring it."""

import hashlib
import re
from pathlib import Path

from token_distiller.tokens import estimate_text_tokens


def _normalize(paragraph: str) -> str:
    return re.sub(r"\s+", " ", paragraph).strip().lower()


def audit_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "bytes": len(text.encode("utf-8")),
        "tokens_est": estimate_text_tokens(text),
        "lines": text.count("\n") + 1,
    }


def find_duplicates(paths: list[Path], min_chars: int = 40) -> list[dict]:
    seen: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for para in text.split("\n\n"):
            norm = _normalize(para)
            if len(norm) < min_chars:
                continue
            key = hashlib.sha1(norm.encode()).hexdigest()
            seen.setdefault(key, []).append(f"{path}: {norm[:60]}...")

    return [
        {"snippet": locations[0], "occurrences": len(locations), "locations": locations}
        for locations in seen.values()
        if len(locations) > 1
    ]


def find_orphaned_memory_files(memory_dir: Path, reference_files: list[Path]) -> list[str]:
    if not memory_dir.exists():
        return []

    memory_files = sorted(memory_dir.glob("*.md"))
    all_text_by_file = {}
    for f in [*reference_files, *memory_files]:
        if f.exists():
            all_text_by_file[f] = f.read_text(encoding="utf-8", errors="replace")

    orphans = []
    for mf in memory_files:
        stem = mf.stem
        referenced_elsewhere = any(
            (stem in text or mf.name in text)
            for other, text in all_text_by_file.items()
            if other != mf
        )
        if not referenced_elsewhere:
            orphans.append(str(mf))
    return orphans


def audit(target_dir: str, memory_dir: str | None = None, mode: str | None = None) -> dict:
    root = Path(target_dir)
    claude_md = root / "CLAUDE.md"
    memory_md = root / "MEMORY.md"

    findings: dict = {"mode": mode, "files": [], "duplicates": [], "orphaned_memory_files": []}

    dup_targets = []
    for f in (claude_md, memory_md):
        if f.exists():
            findings["files"].append(audit_file(f))
            dup_targets.append(f)

    if memory_dir:
        dup_targets += sorted(Path(memory_dir).glob("*.md"))
        findings["orphaned_memory_files"] = find_orphaned_memory_files(
            Path(memory_dir), [claude_md, memory_md]
        )

    findings["duplicates"] = find_duplicates(dup_targets)
    return findings
