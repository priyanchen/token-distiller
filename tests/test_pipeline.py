import httpx
import pytest
from PIL import Image

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


def test_preprocess_produces_a_binary_image():
    from PIL import Image
    from token_distiller import ocr

    noisy = Image.new("RGB", (120, 60), (200, 180, 160))
    out = ocr.preprocess(noisy)
    assert out.mode == "1"


def test_preprocess_upscales_small_crops():
    from PIL import Image
    from token_distiller import ocr

    small = Image.new("RGB", (80, 40), "white")
    out = ocr.preprocess(small)
    assert out.width > small.width


def test_otsu_threshold_separates_two_peaks():
    from token_distiller.ocr import _otsu_threshold

    hist = [0] * 256
    hist[20] = 500   # dark cluster
    hist[230] = 500  # light cluster
    assert 20 < _otsu_threshold(hist) < 230


def test_otsu_threshold_handles_empty_histogram():
    from token_distiller.ocr import _otsu_threshold

    assert _otsu_threshold([0] * 256) == 128


def test_scoped_api_key_takes_precedence(monkeypatch):
    """The tool-specific variable must win, so enabling vision here cannot change how the
    host agent authenticates."""
    from token_distiller import vision_fallback

    monkeypatch.setenv("ANTHROPIC_API_KEY", "global-key")
    monkeypatch.setenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", "scoped-key")
    assert vision_fallback.resolve_api_key() == "scoped-key"


def test_falls_back_to_the_standard_api_key(monkeypatch):
    from token_distiller import vision_fallback

    monkeypatch.delenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "global-key")
    assert vision_fallback.resolve_api_key() == "global-key"


def test_no_key_at_all_resolves_to_none(monkeypatch):
    from token_distiller import vision_fallback

    monkeypatch.delenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert vision_fallback.resolve_api_key() is None


def test_missing_key_names_both_variables(monkeypatch):
    """The error is the only place a user learns which variable to set."""
    from PIL import Image
    from token_distiller import vision_fallback

    monkeypatch.delenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        vision_fallback.describe_image(Image.new("RGB", (10, 10), "white"))
    except vision_fallback.VisionUnavailable as exc:
        assert "TOKEN_DISTILLER_ANTHROPIC_API_KEY" in str(exc)
    else:
        raise AssertionError("expected VisionUnavailable")


def _fake_api_error():
    """A real anthropic.APIError, not a stand-in -- the fix must catch the actual
    exception type the SDK raises, not just something that looks like it."""
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, json={"error": {"message": "invalid"}})
    return anthropic.AuthenticationError(
        message="invalid x-api-key", response=response, body=None
    )


def test_page_level_vision_api_error_degrades_instead_of_crashing(monkeypatch):
    """A live API failure (bad key, rate limit, outage) must degrade the page to OCR text,
    the same as VisionUnavailable does -- not raise and abort the whole distillation."""
    from token_distiller import pipeline, vision_fallback

    monkeypatch.setattr(
        vision_fallback, "describe_image", lambda *a, **k: (_ for _ in ()).throw(_fake_api_error())
    )
    monkeypatch.setattr(
        pipeline.ocr_mod, "ocr_image_best", lambda image: ("", 0.0, 0)
    )  # forces needs_fallback=True regardless of real OCR

    result = pipeline._distill_ocr_or_vision(0, Image.new("RGB", (10, 10), "white"))

    assert result.method == DistillMethod.OCR_DEGRADED
    assert any("vision fallback unavailable" in w for w in result.warnings)


def test_figure_level_vision_api_error_leaves_page_flagged_not_raised(monkeypatch):
    """Same failure at the figure level: _read_figures must skip the unreadable figure and
    keep going, not let an API error propagate out of distill_pdf entirely."""
    from token_distiller import pipeline, vision_fallback

    monkeypatch.setattr(pipeline, "ocr_mod", pipeline.ocr_mod)
    monkeypatch.setattr(
        pipeline.ocr_mod, "ocr_image_best", lambda image: ("", 0.0, 0)
    )
    monkeypatch.setattr(
        vision_fallback, "describe_image", lambda *a, **k: (_ for _ in ()).throw(_fake_api_error())
    )
    monkeypatch.setattr(
        pipeline.pdf_extract, "rasterize_page", lambda *a, **k: Image.new("RGB", (100, 100), "white")
    )
    monkeypatch.setattr(
        pipeline.pdf_extract, "crop_region", lambda page, bbox, height_pt: page
    )

    recovered = pipeline._read_figures(
        "fake.pdf", 0, [(0.0, 0.0, 50.0, 50.0)], 100.0, allow_vision=True
    )

    assert recovered == []  # unreadable figure skipped, not raised


def test_stop_after_tokens_produces_a_partial_result(pdf_factory):
    """The actual point of bounding: a document that would cross the limit stops before
    reading every page, and says so."""
    from token_distiller import pipeline

    path = pdf_factory([[f"page {i} filler text " * 20] for i in range(10)])
    result = pipeline.distill_pdf(path, allow_vision=False, stop_after_tokens=50)

    assert result.is_partial is True
    assert result.total_page_count == 10
    assert len(result.pages) < 10


def test_stop_after_tokens_is_a_no_op_under_the_limit(pdf_factory):
    """A document that never crosses the limit must come out identical to an unbounded
    call -- same pages, not marked partial, boilerplate detection still runs."""
    from token_distiller import pipeline

    path = pdf_factory([["a short document well under any threshold"]])
    bounded = pipeline.distill_pdf(path, allow_vision=False, stop_after_tokens=50_000)
    unbounded = pipeline.distill_pdf(path, allow_vision=False)

    assert bounded.is_partial is False
    assert bounded.total_page_count is None
    assert len(bounded.pages) == len(unbounded.pages)
    assert bounded.distilled_tokens_est == unbounded.distilled_tokens_est


def test_partial_result_is_never_cached(pdf_factory):
    """cache.put's INSERT OR REPLACE keys on content hash alone -- caching a partial result
    would mean a handle silently covers less than what `distill expand` promises for that
    same file's hash. distill() must skip the cache write entirely, not just fail to find
    a stale entry."""
    from token_distiller import cache, pipeline

    path = pdf_factory([[f"page {i} filler text " * 20] for i in range(10)])
    result, handle, was_cached = pipeline.distill(
        path, allow_vision=False, stop_after_tokens=50
    )

    assert result.is_partial is True
    assert handle is None
    assert was_cached is False
    assert cache.get(cache.content_hash(path)) is None


def test_partial_result_skips_boilerplate_detection(pdf_factory):
    """find_boilerplate needs a real majority of the document's pages to tell a repeated
    footer from a structural marker; a partial prefix would confidently mislabel one for
    the other. A partial result must carry no boilerplate manifest at all."""
    from token_distiller import pipeline

    pages = [["Repeated Footer Line", f"unique content on page {i}"] for i in range(10)]
    path = pdf_factory(pages)

    partial = pipeline.distill_pdf(path, allow_vision=False, stop_after_tokens=30)
    complete = pipeline.distill_pdf(path, allow_vision=False)

    assert partial.is_partial is True
    assert partial.boilerplate == []
    assert complete.boilerplate != []  # confirms this fixture does trigger detection normally


def test_rtl_sampled_document_ignores_stop_after_tokens(monkeypatch, pdf_factory):
    """An RTL document is already fast in full (poppler reads it in one subprocess call
    regardless of how much is needed), so bounding it would add complexity for nothing --
    confirmed here by forcing the RTL classification on an ordinary fixture and checking
    the result is never marked partial even with an unreachably tiny limit."""
    from token_distiller import pdf_extract, pipeline

    monkeypatch.setattr(pdf_extract, "_sample_is_rtl", lambda _path: True)
    path = pdf_factory([[f"page {i} filler text " * 20] for i in range(10)])

    result = pipeline.distill_pdf(path, allow_vision=False, stop_after_tokens=1)

    assert result.is_partial is False
    assert len(result.pages) == 10


def test_stop_after_tokens_bounds_ocr_pages_too(monkeypatch):
    """The bound has to be on real distilled_tokens_est, not raw extracted text: a scanned
    page has no native text at all, so accumulating on text alone would never cross the
    limit and every page would still get OCR'd -- confirmed directly against an earlier
    version of this function, which read and OCR'd all 50 simulated pages here before this
    fix, never having bounded anything."""
    from token_distiller import pipeline
    from token_distiller.models import DistillMethod, PageResult

    calls = {"n": 0}

    def fake_ocr(i, image, allow_vision=True):
        calls["n"] += 1
        return PageResult(
            page_index=i, method=DistillMethod.OCR, text="x" * 2000,
            raw_tokens_est=500, distilled_tokens_est=500,
        )

    monkeypatch.setattr(pipeline, "_distill_ocr_or_vision", fake_ocr)
    monkeypatch.setattr(pipeline.pdf_extract, "rasterize_page", lambda *a, **k: object())
    monkeypatch.setattr(
        pipeline.pdf_extract,
        "iter_pages_with_figures",
        lambda path: iter([("", 600.0, 800.0, [])] * 50),
    )
    monkeypatch.setattr(pipeline.pdf_extract, "page_count", lambda path: 50)
    monkeypatch.setattr(pipeline.pdf_extract, "_sample_is_rtl", lambda path: False)

    pages, is_partial, total = pipeline._distill_pages_bounded(
        "fake.pdf", describe_figures=True, allow_vision=False, stop_after_tokens=1000
    )

    assert is_partial is True
    assert total == 50
    assert len(pages) < 50
    assert calls["n"] == len(pages)  # OCR ran only on the pages actually kept


def test_page_cap_stops_a_sparse_scanned_document_tokens_never_would(monkeypatch):
    """The gap this closes: a sparse scanned page can be nearly empty and still cost full
    OCR time, so a token-only bound never triggers on a long, sparse scanned document while
    wall-clock time keeps climbing. Simulates 500 pages each contributing almost nothing to
    the token total -- if only the token bound existed, this would read and OCR all 500."""
    from token_distiller import pipeline
    from token_distiller.models import DistillMethod, PageResult

    calls = {"n": 0}

    def fake_ocr(i, image, allow_vision=True):
        calls["n"] += 1
        return PageResult(
            page_index=i, method=DistillMethod.OCR, text="x",
            raw_tokens_est=500, distilled_tokens_est=1,  # near-zero, deliberately
        )

    monkeypatch.setattr(pipeline, "_distill_ocr_or_vision", fake_ocr)
    monkeypatch.setattr(pipeline.pdf_extract, "rasterize_page", lambda *a, **k: object())
    monkeypatch.setattr(
        pipeline.pdf_extract,
        "iter_pages_with_figures",
        lambda path: iter([("", 600.0, 800.0, [])] * 500),
    )
    monkeypatch.setattr(pipeline.pdf_extract, "page_count", lambda path: 500)
    monkeypatch.setattr(pipeline.pdf_extract, "_sample_is_rtl", lambda path: False)
    monkeypatch.setattr(pipeline, "LARGE_DOC_MAX_PAGES", 40)

    pages, is_partial, total = pipeline._distill_pages_bounded(
        "fake.pdf", describe_figures=True, allow_vision=False, stop_after_tokens=100_000
    )

    assert is_partial is True
    assert total == 500
    assert len(pages) == 40  # stopped by the page cap, not the (never-crossed) token bound
    assert calls["n"] == 40


def test_page_cap_never_triggers_on_a_dense_document(pdf_factory):
    """A document that crosses the token threshold well before the page cap must stop for
    that reason, not be affected by the cap at all -- confirms the cap is inert for the
    common case it was measured against."""
    from token_distiller import pipeline

    path = pdf_factory([[f"page {i} filler text " * 20] for i in range(10)])
    result = pipeline.distill_pdf(path, allow_vision=False, stop_after_tokens=50)

    assert result.is_partial is True
    assert len(result.pages) < 10  # stopped well short of any page-count concern
