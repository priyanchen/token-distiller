from token_distiller.models import DistillMethod, DistillResult, PageResult


def _result(pages):
    return DistillResult(source_path="/tmp/x.pdf", source_type="pdf", pages=pages)


def _page(text, raw, distilled, index=0):
    return PageResult(
        page_index=index,
        method=DistillMethod.OCR,
        text=text,
        raw_tokens_est=raw,
        distilled_tokens_est=distilled,
    )


def test_totals_sum_across_pages():
    result = _result([_page("a", 100, 10, 0), _page("b", 200, 20, 1)])
    assert result.raw_tokens_est == 300
    assert result.distilled_tokens_est == 30


def test_compression_ratio_is_raw_over_distilled():
    assert _result([_page("a", 100, 10)]).compression_ratio == 10.0


def test_fully_compressed_result_does_not_report_zero_compression():
    """A photo with no readable text distills to nothing. Dividing by zero used to
    report 0.0, which reads as 'no compression' for the maximally compressed case."""
    assert _result([_page("", 1440, 0)]).compression_ratio == 1440.0


def test_empty_result_has_no_ratio():
    assert _result([]).compression_ratio == 0.0


def test_text_joins_pages():
    assert _result([_page("first", 1, 1, 0), _page("second", 1, 1, 1)]).text == "first\n\nsecond"


def test_rendered_text_matches_text_without_boilerplate():
    result = _result([_page("body", 10, 1)])
    assert result.rendered_text == result.text


def test_rendered_text_restates_collapsed_boilerplate():
    result = _result([_page("body", 10, 1)])
    result.boilerplate = [{"line": "(c) ACME 2026", "occurrences": 9}]
    assert "(c) ACME 2026" in result.rendered_text
    assert "body" in result.rendered_text


def test_method_counts_tallies_pages():
    pages = [_page("a", 1, 1, 0), _page("b", 1, 1, 1)]
    pages[1].method = DistillMethod.NATIVE_TEXT
    assert _result(pages).method_counts() == {"ocr": 1, "native_text": 1}


def test_warnings_collect_from_all_pages():
    pages = [_page("a", 1, 1, 0), _page("b", 1, 1, 1)]
    pages[0].warnings = ["low confidence"]
    pages[1].warnings = ["no vision key"]
    assert _result(pages).warnings == ["low confidence", "no vision key"]
