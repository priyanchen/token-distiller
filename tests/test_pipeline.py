import pytest

from token_distiller import pipeline
from token_distiller.models import DistillMethod


def test_native_text_pdf_uses_text_layer(pdf_factory):
    path = pdf_factory([["Hello from the text layer of this document"]])
    result, _, _ = pipeline.distill(path)
    assert result.method_counts() == {"native_text": 1}
    assert "text layer" in result.text


def test_page_count_matches_document(pdf_factory):
    path = pdf_factory([[f"substantial page content for page number {i}"] for i in range(4)])
    result, _, _ = pipeline.distill(path)
    assert len(result.pages) == 4


def test_first_pass_is_a_cache_miss(pdf_factory):
    path = pdf_factory([["unique content here for the miss test"]])
    _, handle, cached = pipeline.distill(path)
    assert cached is False
    assert handle is not None


def test_second_pass_hits_cache(pdf_factory):
    path = pdf_factory([["content that will be read twice in a row"]])
    _, h1, _ = pipeline.distill(path)
    _, h2, cached = pipeline.distill(path)
    assert cached is True
    assert h1 == h2


def test_cached_result_is_identical_to_fresh(pdf_factory):
    """The core losslessness property: serving from cache must not alter a single byte."""
    path = pdf_factory([["identical output required", "second line of the page"]])
    fresh, _, _ = pipeline.distill(path)
    cached, _, was_cached = pipeline.distill(path)
    assert was_cached is True
    assert cached.rendered_text == fresh.rendered_text


def test_edited_file_is_not_served_from_cache(tmp_path):
    """Same path, different bytes: a stale hit here would hand back wrong content."""
    from tests.conftest import make_pdf

    target = tmp_path / "mutating.pdf"
    make_pdf(target, [["VERSION ONE ORIGINAL CONTENT"]])
    first, h1, _ = pipeline.distill(str(target))

    make_pdf(target, [["VERSION TWO REPLACEMENT CONTENT"]])
    second, h2, cached = pipeline.distill(str(target))

    assert cached is False
    assert h1 != h2
    assert "VERSION TWO" in second.text
    assert second.text != first.text


def test_cache_can_be_bypassed(pdf_factory):
    path = pdf_factory([["bypass the cache entirely please"]])
    pipeline.distill(path)
    _, handle, cached = pipeline.distill(path, use_cache=False)
    assert cached is False
    assert handle is None


def test_boilerplate_collapses_across_pages(pdf_factory):
    footer = "COPYRIGHT NOTICE EVERY PAGE"
    path = pdf_factory([[f"real content {i}", footer] for i in range(6)])
    result, _, _ = pipeline.distill(path)
    assert result.boilerplate
    assert result.text.count(footer) == 0


def test_collapsed_boilerplate_is_restated_once(pdf_factory):
    footer = "COPYRIGHT NOTICE EVERY PAGE"
    path = pdf_factory([[f"real content {i}", footer] for i in range(6)])
    result, _, _ = pipeline.distill(path)
    assert result.rendered_text.count(footer) == 1


def test_collapsing_never_removes_unique_content(pdf_factory):
    footer = "COPYRIGHT NOTICE EVERY PAGE"
    path = pdf_factory([[f"unique marker {i}", footer] for i in range(6)])
    result, _, _ = pipeline.distill(path)
    for i in range(6):
        assert f"unique marker {i}" in result.text


def test_boilerplate_reduces_token_count(pdf_factory):
    """Collapsing must land the document at the same size as if the footer were never
    there — that is the whole point of the transform."""
    footer = "COPYRIGHT NOTICE REPEATED ON ALL PAGES OF THIS DOCUMENT"
    body = [f"genuinely distinct page body number {i}" for i in range(8)]
    with_footer = pdf_factory([[b, footer] for b in body], name="with.pdf")
    without = pdf_factory([[b] for b in body], name="without.pdf")
    a, _, _ = pipeline.distill(with_footer)
    b, _, _ = pipeline.distill(without)
    assert a.distilled_tokens_est == b.distilled_tokens_est


def test_pdf_baseline_beats_text_only_estimate(pdf_factory):
    """Guards the M5 measurement fix: a text-native PDF must not report 1.0x."""
    path = pdf_factory([[f"a modest amount of page text, page {i}"] for i in range(3)])
    result, _, _ = pipeline.distill(path)
    assert result.raw_tokens_est > result.distilled_tokens_est
    assert result.compression_ratio > 1.0


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("plain text is not a distillable binary")
    with pytest.raises(ValueError):
        pipeline.distill(str(f))


def test_result_aggregates_page_tokens(pdf_factory):
    path = pdf_factory([["first page with enough text to stay native"],
                        ["second page with enough text to stay native"]])
    result, _, _ = pipeline.distill(path)
    assert result.raw_tokens_est == sum(p.raw_tokens_est for p in result.pages)
    assert result.distilled_tokens_est == sum(p.distilled_tokens_est for p in result.pages)


def test_method_enum_round_trips_through_cache(pdf_factory):
    path = pdf_factory([["method enum should survive serialization"]])
    pipeline.distill(path)
    cached, _, _ = pipeline.distill(path)
    assert cached.pages[0].method is DistillMethod.NATIVE_TEXT
