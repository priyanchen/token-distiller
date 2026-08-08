from token_distiller import repo_pack
from tests.conftest import make_pdf


def _project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 42\n")
    (tmp_path / "README.md").write_text("# Project readme\n")
    (tmp_path / ".gitignore").write_text("secrets/\n*.log\n")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key.txt").write_text("SHOULD NOT BE PACKED")
    (tmp_path / "debug.log").write_text("SHOULD NOT BE PACKED EITHER")
    return tmp_path


def test_source_files_are_packed(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)))
    assert any(f.path.endswith("app.py") for f in result.files)


def test_gitignored_directories_are_excluded(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)))
    assert not any("secrets" in f.path for f in result.files)


def test_gitignored_glob_patterns_are_excluded(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)))
    assert not any(f.path.endswith(".log") for f in result.files)


def test_default_excludes_apply_without_gitignore(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "keep.py").write_text("x = 1")
    result = repo_pack.pack(str(tmp_path))
    assert not any("__pycache__" in f.path for f in result.files)


def test_include_glob_narrows_the_pack(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)), include="*.md")
    assert [f.path for f in result.files] == ["README.md"]


def test_exclude_glob_removes_matches(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)), exclude="README.md")
    assert not any(f.path == "README.md" for f in result.files)


def test_embedded_pdf_is_distilled_not_dumped(tmp_path):
    """The differentiator over a plain repo packer: a PDF inside the tree becomes text
    rather than binary noise or a skipped file."""
    _project(tmp_path)
    make_pdf(tmp_path / "spec.pdf", [["REQUIREMENTS FROM THE EMBEDDED SPEC"]])
    result = repo_pack.pack(str(tmp_path))
    pdf_entries = [f for f in result.files if f.path.endswith(".pdf")]
    assert len(pdf_entries) == 1
    assert pdf_entries[0].distilled is True
    assert "REQUIREMENTS FROM THE EMBEDDED SPEC" in pdf_entries[0].text


def test_binary_files_are_skipped(tmp_path):
    _project(tmp_path)
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)))
    result = repo_pack.pack(str(tmp_path))
    assert any("blob.bin" in s for s in result.skipped)


def test_token_total_is_the_sum_of_files(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)))
    assert result.total_tokens_est == sum(f.tokens_est for f in result.files)


def test_markdown_render_labels_each_file(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)))
    rendered = repo_pack.render(result, style="markdown")
    assert "## src/app.py" in rendered
    assert "def main()" in rendered


def test_xml_render_is_well_formed(tmp_path):
    result = repo_pack.pack(str(_project(tmp_path)))
    rendered = repo_pack.render(result, style="xml")
    assert rendered.startswith("<repo_pack>")
    assert rendered.rstrip().endswith("</repo_pack>")


def test_pack_can_skip_figure_reading(tmp_path):
    """repo/index pack PDFs through the same pipeline, so they need the same escape hatch
    `distill file` has — otherwise a batch run has no way to avoid the OCR cost."""
    from token_distiller import repo_pack as rp

    make_pdf(tmp_path / "doc.pdf", [["A page with plenty of native text and no figures."]])
    result = rp.pack(str(tmp_path), describe_figures=False)
    assert any(f.path == "doc.pdf" and f.distilled for f in result.files)


def test_pack_defaults_describe_figures_to_none(tmp_path):
    """None means 'defer to config', so the caller isn't forced to know the default."""
    import inspect

    from token_distiller import repo_pack as rp

    assert inspect.signature(rp.pack).parameters["describe_figures"].default is None
