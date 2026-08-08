from token_distiller.bash_compress import (
    _dedupe_adjacent,
    _truncate,
    compress,
    compress_git_status,
    compress_pytest,
    looks_like_git_status,
    looks_like_pytest,
)
from token_distiller.tokens import estimate_text_tokens

GIT_STATUS = """On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
 M src/app.py
 M src/utils.py
 M tests/test_app.py
 D old/legacy.py
?? scratch/notes.txt
?? scratch/other.txt
"""

PYTEST = """============================= test session starts ==============================
platform darwin -- Python 3.13.0, pytest-8.0.0
collected 240 items

tests/test_a.py ........................................................ [ 23%]
tests/test_b.py ........................................................ [ 46%]
tests/test_c.py .....F.................................................. [ 70%]
tests/test_d.py ........................................................ [100%]

=================================== FAILURES ===================================
_________________________________ test_totals _________________________________

    def test_totals():
>       assert total == 42
E       assert 41 == 42
E        +  where 41 = sum([20, 21])

tests/test_c.py:88: AssertionError
=========================== short test summary info ============================
FAILED tests/test_c.py::test_totals - assert 41 == 42
1 failed, 239 passed in 3.11s
"""


def test_git_status_is_detected():
    assert looks_like_git_status(GIT_STATUS)


def test_git_status_groups_by_state():
    out = compress_git_status(GIT_STATUS)
    assert "modified (3)" in out
    assert "deleted (1)" in out
    assert "untracked (2)" in out


def test_git_status_reports_the_branch():
    assert "branch main" in compress_git_status(GIT_STATUS)


def test_git_status_shrinks_output():
    out = compress_git_status(GIT_STATUS)
    assert estimate_text_tokens(out) < estimate_text_tokens(GIT_STATUS)


def test_pytest_is_detected():
    assert looks_like_pytest(PYTEST)


def test_pytest_keeps_the_failure():
    out = compress_pytest(PYTEST)
    assert "test_totals" in out


def test_pytest_keeps_the_assertion_detail():
    out = compress_pytest(PYTEST)
    assert "41 == 42" in out


def test_pytest_keeps_the_summary_line():
    out = compress_pytest(PYTEST)
    assert "239 passed" in out


def test_pytest_drops_the_passing_noise():
    out = compress_pytest(PYTEST)
    assert "........" not in out
    assert "test session starts" not in out


def test_pytest_compression_is_substantial():
    out = compress_pytest(PYTEST)
    before, after = estimate_text_tokens(PYTEST), estimate_text_tokens(out)
    assert after < before / 2


def test_adjacent_duplicates_collapse_with_a_count():
    out = _dedupe_adjacent(["same"] * 50)
    assert len(out) == 2
    assert "repeated 49" in out[1]


def test_non_adjacent_duplicates_are_preserved():
    assert _dedupe_adjacent(["a", "b", "a"]) == ["a", "b", "a"]


def test_truncation_keeps_head_and_tail():
    lines = [f"line {i}" for i in range(200)]
    out = _truncate(lines, max_lines=10, tail=3)
    assert out[0] == "line 0"
    assert out[-1] == "line 199"
    assert any("omitted" in ln for ln in out)


def test_short_output_is_not_truncated():
    lines = ["a", "b", "c"]
    assert _truncate(lines, max_lines=10, tail=3) == lines


def test_auto_routing_picks_pytest():
    assert "test_totals" in compress(PYTEST)


def test_auto_routing_picks_git_status():
    assert "modified (3)" in compress(GIT_STATUS)


def test_unrecognized_output_still_shrinks():
    noisy = "\n".join(["building module x"] * 300)
    out = compress(noisy)
    assert estimate_text_tokens(out) < estimate_text_tokens(noisy) / 5


def test_empty_input_is_returned_unchanged():
    assert compress("") == ""


def test_kind_override_forces_a_handler():
    """Forcing generic on pytest output must skip the pytest handler entirely."""
    out = compress(PYTEST, kind="generic")
    assert "test session starts" in out


def test_compression_never_inflates_already_terse_output():
    """`git status --porcelain` is denser than any per-state summary of it, so the
    original has to win."""
    porcelain = " M a.py\n?? b.py\n"
    assert len(compress(porcelain)) <= len(porcelain)


def test_compression_never_inflates_any_input():
    for sample in ["short", "a\nb\nc\n", GIT_STATUS, PYTEST, "x" * 50]:
        assert len(compress(sample)) <= len(sample), sample[:30]


def test_human_format_git_status_is_detected():
    """The default `git status` has no porcelain codes at all; missing this meant real
    git output passed through uncompressed."""
    human = (
        "On branch main\n\nChanges not staged for commit:\n"
        '  (use "git add <file>..." to update what will be committed)\n'
        "\tmodified:   src/app.py\n\nUntracked files:\n\tscratch/notes.txt\n"
    )
    assert looks_like_git_status(human)
    out = compress(human)
    assert "modified (1)" in out
    assert "untracked (1)" in out
    assert len(out) < len(human)
