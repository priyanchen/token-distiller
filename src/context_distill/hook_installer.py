"""Wires the PreToolUse Read-interception hook directly into a project's or the
global settings.json, for use without installing this as a formal Claude Code
plugin. Idempotent and backs up the original file before writing."""

import json
import shutil
import sys
from pathlib import Path

from context_distill.config import DISTILLABLE_EXTENSIONS

_MARKER = "distill hook-read"


def _distill_binary_path() -> str:
    return str(Path(sys.executable).parent / "distill")


def _build_hook_entries(command: str) -> list[dict]:
    entries = []
    for ext in sorted(DISTILLABLE_EXTENSIONS):
        pattern = ext.lstrip(".")
        entries.append(
            {
                "type": "command",
                "if": f"Read(*.{pattern})",
                "command": command,
                "timeout": 60000,
            }
        )
    return entries


def _already_installed(pretooluse: list) -> bool:
    for group in pretooluse:
        for hook in group.get("hooks", []):
            if _MARKER in hook.get("command", ""):
                return True
    return False


def resolve_target(target: str, project_dir: str | None) -> Path:
    if target == "global":
        return Path.home() / ".claude" / "settings.json"
    root = Path(project_dir) if project_dir else Path.cwd()
    return root / ".claude" / "settings.json"


def install(target_path: Path, dry_run: bool = False) -> str:
    settings: dict = {}
    if target_path.exists():
        settings = json.loads(target_path.read_text())

    hooks = settings.setdefault("hooks", {})
    pretooluse = hooks.setdefault("PreToolUse", [])

    if _already_installed(pretooluse):
        return f"already installed in {target_path}, no changes made"

    command = f'"{_distill_binary_path()}" hook-read'
    pretooluse.append({"matcher": "Read", "hooks": _build_hook_entries(command)})

    rendered = json.dumps(settings, indent=2) + "\n"

    if dry_run:
        return f"--- dry run, would write to {target_path} ---\n{rendered}"

    if target_path.exists():
        shutil.copy(target_path, target_path.with_suffix(target_path.suffix + ".bak"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(rendered)
    return f"installed hook into {target_path}"
