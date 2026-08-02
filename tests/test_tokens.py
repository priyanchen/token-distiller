from token_distiller.config import IMAGE_MAX_EDGE_PX
from token_distiller.tokens import (
    downscaled_dimensions,
    estimate_image_tokens,
    estimate_pdf_page_host_tokens,
    estimate_text_tokens,
)


def test_empty_text_is_zero_tokens():
    assert estimate_text_tokens("") == 0


def test_text_tokens_follow_chars_per_four():
    assert estimate_text_tokens("a" * 400) == 100


def test_short_text_never_rounds_to_zero():
    assert estimate_text_tokens("hi") == 1


def test_small_images_are_not_downscaled():
    assert downscaled_dimensions(800, 600) == (800, 600)


def test_large_images_downscale_to_the_cap():
    w, h = downscaled_dimensions(2880, 1800)
    assert max(w, h) == IMAGE_MAX_EDGE_PX


def test_downscaling_preserves_aspect_ratio():
    w, h = downscaled_dimensions(2880, 1800)
    assert abs((w / h) - (2880 / 1800)) < 0.01


def test_retina_screenshot_priced_at_capped_resolution():
    """A 2880x1800 screenshot bills ~2049 tokens, not the 6912 an uncapped
    width*height/750 would suggest. Getting this wrong inflates reported savings."""
    uncapped = round(2880 * 1800 / 750)
    assert estimate_image_tokens(2880, 1800) < uncapped / 3


def test_zero_or_negative_dimensions_are_zero():
    assert estimate_image_tokens(0, 100) == 0
    assert estimate_image_tokens(-5, 100) == 0


def test_pdf_page_baseline_exceeds_its_text_alone():
    """The host renders each page to an image on top of extracting text, so a page's
    ingestion cost must be strictly greater than its text cost."""
    text = "some page text " * 20
    assert estimate_pdf_page_host_tokens(612, 792, text) > estimate_text_tokens(text)


def test_pdf_page_baseline_includes_text_contribution():
    blank = estimate_pdf_page_host_tokens(612, 792, "")
    filled = estimate_pdf_page_host_tokens(612, 792, "x" * 4000)
    assert filled - blank == estimate_text_tokens("x" * 4000)
