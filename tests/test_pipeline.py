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


def test_page_with_image_and_native_text_is_flagged(pdf_with_images_factory):
    """The core case this exists for: a page has enough body text to skip OCR/vision
    entirely, but also carries an embedded figure (diagram, illustration, chart) whose
    content native-text extraction cannot see. It must still be reported as
    native_text -- that's the correct method -- but flagged separately."""
    path = pdf_with_images_factory(
        pages=[
            ["Plenty of native body text on this page, easily enough to skip OCR."],
            ["This page has body text plus an embedded figure sitting beside it."],
        ],
        image_pages={1},
    )
    # describe_figures=False isolates the flagging behaviour: with figure reading on, a
    # readable figure is transcribed and the page stops counting as a gap.
    result, _, _ = pipeline.distill(path, describe_figures=False)
    assert result.method_counts() == {"native_text": 2}
    assert result.pages_with_uncaptured_images() == [1]
    assert result.pages[1].image_count == 1
    assert result.pages[0].image_count == 0


def test_page_without_images_is_not_flagged(pdf_with_images_factory):
    path = pdf_with_images_factory(
        pages=[["Ordinary text-only page, nothing embedded, nothing to flag here."]],
        image_pages=set(),
    )
    result, _, _ = pipeline.distill(path, describe_figures=False)
    assert result.pages_with_uncaptured_images() == []


def test_image_count_survives_cache_round_trip(pdf_with_images_factory):
    from token_distiller import cache

    path = pdf_with_images_factory(
        pages=[["A page with a figure embedded next to a full paragraph of text."]],
        image_pages={0},
    )
    _, handle, _ = pipeline.distill(path, describe_figures=False)
    reloaded = cache.get_by_id(handle)
    assert reloaded.pages[0].image_count == 1
    assert reloaded.pages_with_uncaptured_images() == [0]


@pytest.mark.ocr
def test_figure_on_native_text_page_is_read_and_labelled(tmp_path):
    """The point of M6: a diagram sitting next to body text is transcribed into the
    output instead of merely flagged. Marked ocr -- it drives real Tesseract."""
    from tests.conftest import make_pdf_with_legible_figure

    path = make_pdf_with_legible_figure(
        tmp_path / "figure_doc.pdf",
        page_text="Body prose that native extraction reads on its own without any OCR.",
        figure_lines=["QUARTERLY REVENUE", "Q1 340 Q2 580", "Q3 910 Q4 1240"],
    )
    result, _, _ = pipeline.distill(path, describe_figures=True, use_cache=False)

    assert result.pages[0].image_count == 1
    assert result.pages[0].figures, "figure produced no recovered text"
    assert result.figure_count == 1
    # once read, the page is no longer an uncaptured gap
    assert result.pages_with_uncaptured_images() == []
    assert result.pages_with_described_figures() == [0]
    # the figure's content reaches the distilled text, labelled as a figure
    assert "[figure 1 on page 1]" in result.text
    assert "REVENUE" in result.text.upper()
    # and the body prose is still there
    assert "Body prose" in result.text


@pytest.mark.ocr
def test_figures_survive_the_cache_round_trip(tmp_path):
    from tests.conftest import make_pdf_with_legible_figure
    from token_distiller import cache

    path = make_pdf_with_legible_figure(
        tmp_path / "figure_cached.pdf",
        page_text="Page prose alongside an embedded chart that must survive caching.",
        figure_lines=["MARKET SHARE", "NORTH 42", "SOUTH 58"],
    )
    fresh, handle, _ = pipeline.distill(path, describe_figures=True)
    reloaded = cache.get_by_id(handle)
    assert reloaded.pages[0].figures == fresh.pages[0].figures
    assert reloaded.text == fresh.text


@pytest.mark.ocr
def test_no_figures_flag_leaves_the_page_flagged(tmp_path):
    from tests.conftest import make_pdf_with_legible_figure

    path = make_pdf_with_legible_figure(
        tmp_path / "figure_skipped.pdf",
        page_text="Body prose present, but figure reading is turned off for this run.",
        figure_lines=["SHOULD NOT BE TRANSCRIBED"],
    )
    result, _, _ = pipeline.distill(path, describe_figures=False, use_cache=False)
    assert result.pages[0].figures == []
    assert result.pages_with_uncaptured_images() == [0]
    assert "SHOULD NOT BE TRANSCRIBED" not in result.text
