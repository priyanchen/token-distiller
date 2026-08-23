import io
import json
from argparse import Namespace
from pathlib import Path

from token_distiller import cli


def _run_hook(monkeypatch, capsys, file_path, session_id="S1"):
    payload = {"session_id": session_id, "tool_input": {"file_path": file_path}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    code = cli.cmd_hook_read(Namespace(no_vision=True))
    return code, capsys.readouterr().out


def _reason(out):
    return json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]


def test_non_distillable_file_passes_through_silently(monkeypatch, capsys, tmp_path):
    """Every Read in every session hits this path, so a .py file must produce no output
    and never trigger the heavy imports."""
    src = tmp_path / "module.py"
    src.write_text("print('hello')")
    code, out = _run_hook(monkeypatch, capsys, str(src))
    assert code == 0
    assert out == ""


def test_pdf_returns_a_deny_decision(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["intercepted document content"]])
    _, out = _run_hook(monkeypatch, capsys, path)
    hook = json.loads(out)["hookSpecificOutput"]
    assert hook["hookEventName"] == "PreToolUse"
    assert hook["permissionDecision"] == "deny"


def test_first_read_includes_the_distilled_text(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["intercepted document content here"]])
    _, out = _run_hook(monkeypatch, capsys, path)
    assert "intercepted document content" in _reason(out)


def test_first_read_reports_token_savings(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["some content to measure"]])
    _, out = _run_hook(monkeypatch, capsys, path)
    assert "[token-distiller]" in _reason(out)
    assert "tokens" in _reason(out)


def test_reread_in_same_session_collapses(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["a" * 400, "b" * 400]])
    _, first = _run_hook(monkeypatch, capsys, path, session_id="S_RE")
    _, second = _run_hook(monkeypatch, capsys, path, session_id="S_RE")
    assert len(_reason(second)) < len(_reason(first)) / 2


def test_collapsed_reread_offers_an_expand_handle(monkeypatch, capsys, pdf_factory):
    """A pointer is only acceptable because the full text stays reachable."""
    path = pdf_factory([["content that gets collapsed on reread"]])
    _run_hook(monkeypatch, capsys, path, session_id="S_RE2")
    _, second = _run_hook(monkeypatch, capsys, path, session_id="S_RE2")
    assert "distill expand" in _reason(second)


def test_a_different_session_still_gets_full_text(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["content for cross session check"]])
    _run_hook(monkeypatch, capsys, path, session_id="S_A")
    _, other = _run_hook(monkeypatch, capsys, path, session_id="S_B")
    assert "content for cross session check" in _reason(other)


def test_edited_file_is_not_collapsed(monkeypatch, capsys, tmp_path):
    """Collapse keys on content, not path — an edit must return the new text in full."""
    from tests.conftest import make_pdf

    target = tmp_path / "edited.pdf"
    make_pdf(target, [["ORIGINAL VERSION OF THE TEXT"]])
    _run_hook(monkeypatch, capsys, str(target), session_id="S_ED")

    make_pdf(target, [["REPLACED VERSION OF THE TEXT"]])
    _, out = _run_hook(monkeypatch, capsys, str(target), session_id="S_ED")
    assert "REPLACED VERSION" in _reason(out)


def test_large_document_defers_instead_of_truncating(monkeypatch, capsys, pdf_factory):
    """Bounded extraction means a large document is never cached (see pipeline.distill),
    so it has no handle -- the reply must point at `distill index` on the path instead of
    claiming an expand handle that doesn't exist."""
    monkeypatch.setattr("token_distiller.config.LARGE_DOC_TOKEN_THRESHOLD", 20)
    path = pdf_factory([["padding line " * 6] * 12 for _ in range(4)])
    _, out = _run_hook(monkeypatch, capsys, path, session_id="S_BIG")
    reason = _reason(out)
    assert "distill expand" not in reason
    assert "distill index" in reason
    assert "Nothing was discarded" in reason


def test_large_document_reply_states_pages_read_so_far(monkeypatch, capsys, pdf_factory):
    """The plan for this change was explicit: never state a partial page count as if it
    covered the document. The reply has to say how many of the total were actually read."""
    monkeypatch.setattr("token_distiller.config.LARGE_DOC_TOKEN_THRESHOLD", 20)
    path = pdf_factory([["padding line " * 6] * 12 for _ in range(4)])
    _, out = _run_hook(monkeypatch, capsys, path, session_id="S_PAGES")
    reason = _reason(out)
    assert "of 4 pages" in reason


def test_large_document_can_still_be_fully_reached_via_index_and_query(
    monkeypatch, capsys, pdf_factory
):
    """The whole justification for bounding the hook's extraction is that nothing is
    actually lost -- `distill index` always runs unbounded (repo_pack.pack never passes
    stop_after_tokens), so content past the deferred prefix must still be reachable."""
    from token_distiller import chunker, index_store, repo_pack, retrieval

    monkeypatch.setattr("token_distiller.config.LARGE_DOC_TOKEN_THRESHOLD", 20)
    path = pdf_factory(
        [["intro filler " * 6] * 12, ["intro filler " * 6] * 12,
         ["the needle sentence found only on this later page"],
         ["closing filler " * 6] * 12],
        name="big.pdf",
    )

    _, out = _run_hook(monkeypatch, capsys, path, session_id="S_INDEX")
    assert "distill index" in _reason(out)  # confirms it actually deferred

    result = repo_pack.pack(str(Path(path).parent))
    for f in result.files:
        index_store.clear_source(f.path)
        pieces = chunker.chunk_text(f.text)
        if pieces:
            index_store.add_chunks(f.path, pieces)

    hits = retrieval.query("needle sentence", top_k=3, use_semantic=False)
    assert any("needle sentence" in h["text"] for h in hits)


def test_unreadable_file_fails_open(monkeypatch, capsys, tmp_path):
    """A corrupt PDF must not block the Read — Claude should fall back to normal
    behaviour rather than lose access to the file entirely."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a valid pdf at all")
    code, out = _run_hook(monkeypatch, capsys, str(broken))
    assert code == 0
    assert "hookSpecificOutput" not in out


def test_hook_read_notes_pages_with_uncaptured_images(monkeypatch, capsys):
    """The hook-read header must surface this, since it's the surface Claude actually
    sees mid-session -- a note buried only in `distill file --json` would never reach
    the model reading the file."""
    from tests.conftest import make_pdf_with_images

    # figure reading off, so the image stays uncaptured and the gap note is what renders
    monkeypatch.setattr("token_distiller.pipeline.DESCRIBE_FIGURES", False)
    path = make_pdf_with_images(
        "/tmp/hook_img_test.pdf",
        pages=[
            ["Plain page with real text and nothing embedded on it at all here."],
            ["This page carries body text plus an embedded figure right beside it."],
        ],
        image_pages={1},
    )
    _, out = _run_hook(monkeypatch, capsys, path, session_id="S_IMG")
    reason = _reason(out)
    assert "embedded image" in reason
    assert "page" in reason.lower()


def test_hook_read_omits_the_note_when_no_pages_have_images(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["An entirely ordinary text-only page with nothing embedded."]])
    _, out = _run_hook(monkeypatch, capsys, path)
    reason = _reason(out)
    assert "embedded image" not in reason


def _run_hook_with(monkeypatch, capsys, tool_input, session_id="S1"):
    payload = {"session_id": session_id, "tool_input": tool_input}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    code = cli.cmd_hook_read(Namespace(no_vision=True))
    return code, capsys.readouterr().out


def test_page_ranged_read_passes_through(monkeypatch, capsys, pdf_factory):
    """A request for specific pages must not be answered with the whole document. Before
    this, hook-read ignored the range entirely and returned the full distilled text."""
    path = pdf_factory([[f"page {i} with enough text to stay on the native path"] for i in range(6)])
    code, out = _run_hook_with(monkeypatch, capsys, {"file_path": path, "pages": "2-4"})
    assert code == 0
    assert out == ""


def test_offset_read_passes_through(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["some page content that is comfortably long enough here"]])
    code, out = _run_hook_with(monkeypatch, capsys, {"file_path": path, "offset": 100})
    assert code == 0
    assert out == ""


def test_limit_read_passes_through(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["some page content that is comfortably long enough here"]])
    code, out = _run_hook_with(monkeypatch, capsys, {"file_path": path, "limit": 50})
    assert code == 0
    assert out == ""


def test_whole_file_read_is_still_intercepted(monkeypatch, capsys, pdf_factory):
    path = pdf_factory([["ordinary whole-file read of a page with plenty of text"]])
    _, out = _run_hook_with(monkeypatch, capsys, {"file_path": path})
    assert "hookSpecificOutput" in out
