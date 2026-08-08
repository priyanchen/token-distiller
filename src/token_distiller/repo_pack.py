"""Repomix-style repo packing: walk a directory, respect .gitignore, concatenate
files into one token-counted output. PDFs/images encountered along the way are
routed through the M1 distillation pipeline instead of being dumped raw."""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from token_distiller.config import DISTILLABLE_EXTENSIONS
from token_distiller.tokens import estimate_text_tokens

DEFAULT_EXCLUDES = [
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    "node_modules/",
    "dist/",
    "build/",
    "*.egg-info/",
    ".pytest_cache/",
]

BINARY_SKIP_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".dylib", ".bin",
    ".db", ".sqlite", ".sqlite3", ".mp4", ".mp3", ".mov", ".ico",
}


@dataclass
class PackedFile:
    path: str
    text: str
    tokens_est: int
    distilled: bool = False


@dataclass
class PackResult:
    files: list[PackedFile] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total_tokens_est(self) -> int:
        return sum(f.tokens_est for f in self.files)


def _load_spec(root: Path) -> pathspec.PathSpec:
    patterns = list(DEFAULT_EXCLUDES)
    gitignore = root / ".gitignore"
    if gitignore.exists():
        patterns += gitignore.read_text().splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _is_probably_text(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def pack(
    root_dir: str,
    include: str | None = None,
    exclude: str | None = None,
    allow_vision: bool = True,
    describe_figures: bool | None = None,
) -> PackResult:
    from token_distiller import pipeline  # lazy: only needed once a PDF/image shows up

    root = Path(root_dir).resolve()
    spec = _load_spec(root)
    result = PackResult()

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel_str = str(path.relative_to(root))

        if spec.match_file(rel_str):
            continue
        if include and not fnmatch.fnmatch(rel_str, include):
            continue
        if exclude and fnmatch.fnmatch(rel_str, exclude):
            continue

        suffix = path.suffix.lower()
        if suffix in DISTILLABLE_EXTENSIONS:
            try:
                dist_result, _, _ = pipeline.distill(
                    str(path),
                    allow_vision=allow_vision,
                    describe_figures=describe_figures,
                )
                result.files.append(
                    PackedFile(
                        path=rel_str,
                        text=dist_result.rendered_text,
                        tokens_est=dist_result.distilled_tokens_est,
                        distilled=True,
                    )
                )
            except Exception as exc:
                result.skipped.append(f"{rel_str} (distill failed: {exc})")
            continue

        if suffix in BINARY_SKIP_EXTENSIONS or not _is_probably_text(path):
            result.skipped.append(rel_str)
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        result.files.append(
            PackedFile(path=rel_str, text=text, tokens_est=estimate_text_tokens(text))
        )

    return result


def render_markdown(result: PackResult) -> str:
    parts = [f"# Repo pack ({len(result.files)} files, ~{result.total_tokens_est} tokens)\n"]
    for f in result.files:
        tag = " (distilled)" if f.distilled else ""
        parts.append(f"## {f.path}{tag}\n\n```\n{f.text}\n```\n")
    return "\n".join(parts)


def render_xml(result: PackResult) -> str:
    parts = ["<repo_pack>"]
    for f in result.files:
        distilled_attr = ' distilled="true"' if f.distilled else ""
        parts.append(f'  <file path="{f.path}"{distilled_attr}>')
        parts.append(f"    <![CDATA[{f.text}]]>")
        parts.append("  </file>")
    parts.append("</repo_pack>")
    return "\n".join(parts)


def render(result: PackResult, style: str = "markdown") -> str:
    if style == "xml":
        return render_xml(result)
    return render_markdown(result)
