from token_distiller.boilerplate import (
    find_boilerplate,
    render_manifest,
    strip_boilerplate,
)


def _pages(n, footer="(c) ACME CORP ALL RIGHTS RESERVED", every_other=None):
    out = []
    for i in range(n):
        lines = [f"Heading for page {i}", f"body content unique to page {i}"]
        if every_other and i % 2 == 0:
            lines.append(every_other)
        lines.append(footer)
        out.append("\n".join(lines))
    return out


def test_line_on_every_page_is_boilerplate():
    found = find_boilerplate(_pages(10))
    assert "(c) ACME CORP ALL RIGHTS RESERVED" in found


def test_line_on_half_the_pages_is_not_boilerplate():
    """The conservatism guarantee: a structural marker that appears on some pages
    carries position-specific meaning and must survive. On the real 25-page deck this
    is what keeps 'Example:' (15/25) while dropping the footer (25/25)."""
    found = find_boilerplate(_pages(10, every_other="Example:"))
    assert "Example:" not in found
    assert "(c) ACME CORP ALL RIGHTS RESERVED" in found


def test_unique_body_lines_are_never_boilerplate():
    found = find_boilerplate(_pages(10))
    assert not any("unique to page" in line for line in found)


def test_too_few_pages_disables_collapsing():
    assert find_boilerplate(_pages(2)) == []


def test_overly_long_lines_are_ignored():
    long_line = "x" * 500
    assert long_line not in find_boilerplate([long_line] * 10)


def test_strip_removes_boilerplate_from_every_page():
    stripped, manifest = strip_boilerplate(_pages(6))
    assert all("ACME CORP" not in page for page in stripped)
    assert manifest[0]["occurrences"] == 6


def test_strip_preserves_unique_content():
    stripped, _ = strip_boilerplate(_pages(6))
    for i, page in enumerate(stripped):
        assert f"body content unique to page {i}" in page


def test_strip_is_a_no_op_when_nothing_repeats():
    pages = [f"totally distinct page {i}" for i in range(5)]
    stripped, manifest = strip_boilerplate(pages)
    assert stripped == pages
    assert manifest == []


def test_manifest_restates_what_was_removed():
    _, manifest = strip_boilerplate(_pages(6))
    rendered = render_manifest(manifest)
    assert "ACME CORP" in rendered
    assert "6x" in rendered


def test_empty_manifest_renders_nothing():
    assert render_manifest([]) == ""
