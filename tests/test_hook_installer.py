import json

from token_distiller import hook_installer


def _settings(tmp_path):
    return tmp_path / ".claude" / "settings.json"


def test_install_creates_settings_when_absent(tmp_path):
    target = _settings(tmp_path)
    hook_installer.install(target)
    assert target.exists()


def test_installed_hook_matches_the_read_tool(tmp_path):
    target = _settings(tmp_path)
    hook_installer.install(target)
    group = json.loads(target.read_text())["hooks"]["PreToolUse"][0]
    assert group["matcher"] == "Read"


def test_installed_hook_filters_on_file_patterns(tmp_path):
    """The `if` field is evaluated before the process spawns, so ordinary Reads never
    pay for the hook."""
    target = _settings(tmp_path)
    hook_installer.install(target)
    entries = json.loads(target.read_text())["hooks"]["PreToolUse"][0]["hooks"]
    assert all("if" in h for h in entries)
    assert "Read(*.pdf)" in [h["if"] for h in entries]


def test_install_preserves_unrelated_settings(tmp_path):
    target = _settings(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"model": "opus", "permissions": {"allow": ["Bash"]}}))
    hook_installer.install(target)
    settings = json.loads(target.read_text())
    assert settings["model"] == "opus"
    assert settings["permissions"] == {"allow": ["Bash"]}


def test_install_preserves_existing_hooks(tmp_path):
    target = _settings(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}}))
    hook_installer.install(target)
    matchers = [g["matcher"] for g in json.loads(target.read_text())["hooks"]["PreToolUse"]]
    assert "Bash" in matchers
    assert "Read" in matchers


def test_install_is_idempotent(tmp_path):
    target = _settings(tmp_path)
    hook_installer.install(target)
    message = hook_installer.install(target)
    groups = json.loads(target.read_text())["hooks"]["PreToolUse"]
    assert len(groups) == 1
    assert "already installed" in message


def test_install_backs_up_the_previous_file(tmp_path):
    target = _settings(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"model": "sonnet"}))
    hook_installer.install(target)
    backup = target.with_suffix(target.suffix + ".bak")
    assert json.loads(backup.read_text()) == {"model": "sonnet"}


def test_dry_run_writes_nothing(tmp_path):
    target = _settings(tmp_path)
    output = hook_installer.install(target, dry_run=True)
    assert not target.exists()
    assert "dry run" in output


def test_global_target_resolves_to_home(tmp_path):
    path = hook_installer.resolve_target("global", None)
    assert path.parts[-2:] == (".claude", "settings.json")


def test_project_target_resolves_under_the_project(tmp_path):
    path = hook_installer.resolve_target("project", str(tmp_path))
    assert path == tmp_path / ".claude" / "settings.json"


def test_default_install_covers_pdfs_only(tmp_path):
    """Intercepting an image read replaces the model's own vision on it with OCR text, so
    images are opt-in rather than default."""
    target = _settings(tmp_path)
    hook_installer.install(target)
    conditions = [h["if"] for h in json.loads(target.read_text())["hooks"]["PreToolUse"][0]["hooks"]]
    assert conditions == ["Read(*.pdf)"]


def test_images_flag_opts_into_image_extensions(tmp_path):
    target = _settings(tmp_path)
    hook_installer.install(target, include_images=True)
    conditions = [h["if"] for h in json.loads(target.read_text())["hooks"]["PreToolUse"][0]["hooks"]]
    assert "Read(*.pdf)" in conditions
    assert "Read(*.png)" in conditions
    assert "Read(*.heic)" in conditions


def test_hook_timeout_is_in_seconds_not_milliseconds(tmp_path):
    """The `timeout` field on a Claude Code hook is documented in seconds (default 600 for
    a command hook). This install wrote 60000 -- which reads as sixty seconds if you assume
    milliseconds, but is actually 60000 seconds (16.7 hours) -- and nothing caught it until
    it was checked directly against the docs. A sane upper bound here is what would have
    caught that: 60000 fails it outright, any real timeout in seconds does not.
    """
    target = _settings(tmp_path)
    hook_installer.install(target)
    entries = json.loads(target.read_text())["hooks"]["PreToolUse"][0]["hooks"]
    for h in entries:
        assert 0 < h["timeout"] <= 600, (
            f"timeout {h['timeout']} is not a plausible number of seconds "
            f"(Claude Code's own default for a command hook is 600)"
        )
