from token_distiller import pdf_extract, pipeline


def test_two_column_page_extracts_left_column_then_right_column(pdf_columns_factory):
    """The bug this fixes: pdfplumber's default extract_text() sorts primarily by vertical
    position, so two columns at the same height get interleaved word-by-word into one
    corrupted line. Ten lines per column -- not two -- because the detector deliberately
    requires the same gap to repeat across several rows before it trusts it; see
    _detect_column_split's docstring for why a sparser page must not trigger a split."""
    left = [f"LeftLine{i}: column one content on row {i}." for i in range(10)]
    right = [f"RightLine{i}: column two content on row {i}." for i in range(10)]
    path = pdf_columns_factory([(left, right)])

    result, _, _ = pipeline.distill(path, use_cache=False)
    text = result.text

    left_positions = [text.find(f"LeftLine{i}:") for i in range(10)]
    right_positions = [text.find(f"RightLine{i}:") for i in range(10)]
    assert all(p >= 0 for p in left_positions + right_positions)
    assert left_positions == sorted(left_positions), "left column must read top to bottom"
    assert right_positions == sorted(right_positions), "right column must read top to bottom"
    assert max(left_positions) < min(right_positions), (
        "the entire left column must precede the entire right column -- interleaving by "
        "height is exactly the corruption this fix exists to prevent"
    )


def test_single_column_document_is_unaffected(pdf_factory):
    """The safety property that matters most: a single-column document -- the overwhelming
    majority of real PDFs, and everything the rest of the test suite already verifies --
    must take the exact same path as before column detection existed."""
    path = pdf_factory([[f"Ordinary paragraph line {i} of a normal single-column page." for i in range(15)]])
    result, _, _ = pipeline.distill(path, use_cache=False)
    lines = [f"Ordinary paragraph line {i} of a normal single-column page." for i in range(15)]
    assert result.text == "\n".join(lines)


def test_detector_does_not_fire_on_a_single_incidental_gap(pdf_factory):
    """One row with a wide gap (e.g. a lone table-like line on an otherwise prose page)
    must not be mistaken for a column boundary -- the detector requires the same gap
    position to repeat across several separate rows, not just appear once."""
    pages = [[
        "This is an ordinary paragraph with normal spacing throughout the line.",
        "Item                          Price",  # one incidental wide gap, single row
        "Another ordinary paragraph line follows immediately after that one.",
        "A third ordinary line continues the prose without any column layout.",
    ]]
    path = pdf_factory(pages)
    result, _, _ = pipeline.distill(path, use_cache=False)
    assert "Item                          Price" in result.text or "Item" in result.text
    # The paragraph lines must stay in original document order, not get split/reordered.
    text = result.text
    assert text.find("ordinary paragraph with normal spacing") < text.find("Another ordinary paragraph")
    assert text.find("Another ordinary paragraph") < text.find("A third ordinary line")


def test_detect_column_split_returns_none_below_the_word_count_floor(pdf_columns_factory):
    """A page with too little text to judge reliably must fall back rather than guess."""
    import pdfplumber

    path = pdf_columns_factory([(["Short left."], ["Short right."])])
    with pdfplumber.open(path) as pdf:
        assert pdf_extract._detect_column_split(pdf.pages[0]) is None
