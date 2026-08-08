"""Compresses verbose command output.

This module reads text and returns text. It never executes a command, builds a shell
string, or interpolates anything into one — the caller pipes output in. That is a
deliberate boundary: the obvious alternative (a hook that rewrites a Bash command to route
it through a wrapper) means constructing shell strings from model-supplied input, which is
where command injection lives. Compressing a stream we are simply handed has no such
surface.

Every handler is lossy by design but keeps the parts a reader acts on: what failed, what
changed, and the totals. Detection is by content, so it works regardless of how the
command was invoked.
"""

import re

from token_distiller.config import (
    BASH_MAX_LINES,
    BASH_TAIL_LINES,
)

_GIT_STATUS_CODES = {
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "U": "unmerged",
    "?": "untracked",
    "!": "ignored",
}


def _dedupe_adjacent(lines: list[str]) -> list[str]:
    """Collapse runs of identical lines. Log tails and progress output repeat the same line
    hundreds of times; the count carries the same information."""
    out: list[str] = []
    previous: str | None = None
    repeats = 0
    for line in lines:
        if line == previous:
            repeats += 1
            continue
        if repeats:
            out.append(f"    ... previous line repeated {repeats} more time(s)")
            repeats = 0
        out.append(line)
        previous = line
    if repeats:
        out.append(f"    ... previous line repeated {repeats} more time(s)")
    return out


def _truncate(lines: list[str], max_lines: int, tail: int) -> list[str]:
    if len(lines) <= max_lines:
        return lines
    hidden = len(lines) - max_lines - tail
    if hidden <= 0:
        return lines
    return [
        *lines[:max_lines],
        f"    ... {hidden} line(s) omitted ...",
        *lines[-tail:],
    ]


def looks_like_git_status(text: str) -> bool:
    """`git status` with no flags prints a human format with no porcelain codes at all, and
    that is the form people actually run — detecting only `--porcelain` output meant real
    git status passed through uncompressed."""
    if re.search(r"^\?\?\s|^\s?[MADRCU]\s+\S", text, re.MULTILINE):
        return True
    return "On branch " in text and bool(
        re.search(r"(modified:|new file:|deleted:|Untracked files:|nothing to commit)", text)
    )


_HUMAN_STATUS = re.compile(
    r"^(modified|new file|deleted|renamed|copied|both modified):\s+(.+)$"
)
# Matched against the *stripped* line. Allowing a space inside the code class here would
# swallow git's own hint lines ("  (use \"git add <file>...\"") as untracked paths.
_PORCELAIN_STATUS = re.compile(r"^([MADRCU?!]{1,2})\s+(.+)$")

_HUMAN_LABELS = {"new file": "added", "both modified": "unmerged"}


def compress_git_status(text: str) -> str:
    """Handles both `--porcelain` codes and the default human format. The human format
    lists untracked files as bare indented paths under a section header, so the current
    section has to be tracked to label them."""
    entries: list[tuple[str, str]] = []
    section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("("):  # blank, or one of git's "(use ...)" hints
            continue

        human = _HUMAN_STATUS.match(line)
        if human:
            entries.append(
                (_HUMAN_LABELS.get(human.group(1), human.group(1)), human.group(2).strip())
            )
            continue

        if line.endswith(":"):
            low = line.lower()
            if "untracked" in low:
                section = "untracked"
            elif "unmerged" in low:
                section = "unmerged"
            else:
                section = None
            continue

        porcelain = _PORCELAIN_STATUS.match(line)
        if porcelain:
            entries.append(
                (
                    _GIT_STATUS_CODES.get(porcelain.group(1)[0], porcelain.group(1)),
                    porcelain.group(2).strip(),
                )
            )
            continue

        # bare indented path inside a section that doesn't prefix its entries
        if section and raw_line[:1] in ("\t", " "):
            entries.append((section, line))

    if not entries:
        return text

    grouped: dict[str, list[str]] = {}
    for label, path in entries:
        grouped.setdefault(label, []).append(path)

    branch = ""
    bm = re.search(r"^(?:On branch |## )(\S+)", text, re.MULTILINE)
    if bm:
        branch = f"branch {bm.group(1).split('...')[0]}; "

    parts = [f"{branch}{sum(len(v) for v in grouped.values())} path(s) changed"]
    for label in sorted(grouped):
        paths = grouped[label]
        shown = ", ".join(paths[:5])
        more = f", +{len(paths) - 5} more" if len(paths) > 5 else ""
        parts.append(f"  {label} ({len(paths)}): {shown}{more}")
    return "\n".join(parts)


def looks_like_pytest(text: str) -> bool:
    return bool(re.search(r"=+ (test session starts|FAILURES|short test summary)", text)) or bool(
        re.search(r"^\d+ (passed|failed)", text, re.MULTILINE)
    )


def compress_pytest(text: str) -> str:
    """Keep failures and the summary. A wall of dots for passing tests tells the reader
    nothing they can act on."""
    lines = text.splitlines()
    failures = [ln for ln in lines if re.match(r"^(FAILED|ERROR)\s", ln)]
    summary = [ln for ln in lines if re.search(r"\d+ (passed|failed|error|skipped|deselected)", ln)]

    kept: list[str] = []
    if failures:
        kept.append(f"{len(failures)} failing:")
        kept.extend(f"  {f}" for f in failures[:20])
        if len(failures) > 20:
            kept.append(f"  ... +{len(failures) - 20} more")

    # the assertion detail for the first failure is usually what gets acted on
    in_block = False
    detail: list[str] = []
    for ln in lines:
        if re.match(r"^_{3,}.*_{3,}$", ln):
            if in_block:
                break
            in_block = True
            continue
        if in_block and ln.startswith("E "):
            detail.append(ln)
    if detail:
        kept.append("first failure:")
        kept.extend(f"  {d}" for d in detail[:8])

    kept.extend(summary[-2:] if summary else [])
    return "\n".join(kept) if kept else text


def looks_like_install_log(text: str) -> bool:
    return bool(
        re.search(r"^(Collecting|Downloading|Requirement already satisfied|added \d+ packages)", text, re.MULTILINE)
    )


def compress_install_log(text: str) -> str:
    lines = text.splitlines()
    installed = [ln for ln in lines if ln.startswith("Successfully installed")]
    errors = [ln for ln in lines if re.search(r"^(ERROR|error)", ln)]
    collecting = sum(1 for ln in lines if ln.startswith("Collecting"))
    already = sum(1 for ln in lines if ln.startswith("Requirement already satisfied"))

    parts = []
    if collecting:
        parts.append(f"collected {collecting} package(s)")
    if already:
        parts.append(f"{already} already satisfied")
    parts.extend(errors[:10])
    parts.extend(installed)
    return "\n".join(parts) if parts else text


def looks_like_long_listing(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) > BASH_MAX_LINES and bool(
        re.match(r"^[-dlrwxst]{10}[\s@+]", lines[min(1, len(lines) - 1)])
    )


def compress(text: str, kind: str = "auto") -> str:
    """Route to a handler, then always apply the generic squeeze so unrecognized output
    still shrinks instead of passing through whole."""
    if not text.strip():
        return text

    if kind == "auto":
        if looks_like_git_status(text):
            kind = "git-status"
        elif looks_like_pytest(text):
            kind = "pytest"
        elif looks_like_install_log(text):
            kind = "install"
        else:
            kind = "generic"

    if kind == "git-status":
        result = compress_git_status(text)
    elif kind == "pytest":
        result = compress_pytest(text)
    elif kind == "install":
        result = compress_install_log(text)
    else:
        lines = _dedupe_adjacent(text.splitlines())
        result = "\n".join(_truncate(lines, BASH_MAX_LINES, BASH_TAIL_LINES))

    # Already-terse input can come out longer than it went in -- `git status --porcelain`
    # is denser than any per-state summary of it. A compressor that inflates its input is
    # worse than useless, so the original wins ties.
    return result if len(result) < len(text) else text
