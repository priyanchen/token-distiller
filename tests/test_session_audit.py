from token_distiller.session_audit import (
    audit,
    audit_file,
    find_duplicates,
    find_orphaned_memory_files,
)

RULE = "Always run the full test suite before committing anything to the main branch."


def test_audit_file_reports_size_and_tokens(tmp_path):
    f = tmp_path / "CLAUDE.md"
    f.write_text("x" * 400)
    stats = audit_file(f)
    assert stats["bytes"] == 400
    assert stats["tokens_est"] == 100


def test_duplicate_paragraphs_are_detected(tmp_path):
    a, b = tmp_path / "CLAUDE.md", tmp_path / "MEMORY.md"
    a.write_text(f"# Rules\n\n{RULE}\n")
    b.write_text(f"# Memory\n\n{RULE}\n")
    duplicates = find_duplicates([a, b])
    assert len(duplicates) == 1
    assert duplicates[0]["occurrences"] == 2


def test_duplicate_detection_ignores_whitespace_differences(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text(RULE)
    b.write_text(RULE.replace(" ", "  "))
    assert len(find_duplicates([a, b])) == 1


def test_short_lines_are_not_flagged_as_duplicates(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("# Notes")
    b.write_text("# Notes")
    assert find_duplicates([a, b]) == []


def test_distinct_content_produces_no_duplicates(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("The first document says one thing entirely on its own.")
    b.write_text("The second document says something completely different instead.")
    assert find_duplicates([a, b]) == []


def test_unreferenced_memory_file_is_orphaned(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "linked.md").write_text("referenced note")
    (memory / "orphan.md").write_text("nothing points here")
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("See [[linked]] for details.")

    orphans = find_orphaned_memory_files(memory, [claude])
    assert any("orphan.md" in o for o in orphans)
    assert not any("linked.md" in o for o in orphans)


def test_memory_files_can_reference_each_other(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "hub.md").write_text("see [[spoke]]")
    (memory / "spoke.md").write_text("referenced by hub")
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("See [[hub]].")
    assert find_orphaned_memory_files(memory, [claude]) == []


def test_missing_memory_dir_yields_no_orphans(tmp_path):
    assert find_orphaned_memory_files(tmp_path / "nope", []) == []


def test_audit_reports_mode_and_findings(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "stray.md").write_text("unreferenced")
    (tmp_path / "CLAUDE.md").write_text(f"# Rules\n\n{RULE}\n")
    (tmp_path / "MEMORY.md").write_text(f"# Memory\n\n{RULE}\n")

    findings = audit(str(tmp_path), memory_dir=str(memory), mode="debug")
    assert findings["mode"] == "debug"
    assert len(findings["files"]) == 2
    assert findings["duplicates"]
    assert any("stray.md" in o for o in findings["orphaned_memory_files"])


def test_audit_tolerates_missing_files(tmp_path):
    findings = audit(str(tmp_path))
    assert findings["files"] == []
    assert findings["duplicates"] == []
