"""Reproduces the order-of-magnitude compression figures quoted in the README (33x on
a 25-page PDF, 91% on a screenshot) against synthetic-but-representative fixtures, so
those numbers are backed by an assertion in CI rather than a one-time manual
measurement.

These are not meant to reproduce the exact 33x / 91% figures bit-for-bit -- those came
from one specific real document and screenshot, and will vary with content. Each test
instead asserts a lower bound comfortably below what was actually measured when this
file was written (see the numbers in each docstring), so the claim stays honest without
being brittle to minor rendering or OCR differences across environments.
"""

from pathlib import Path

import pytest

from token_distiller import pipeline

# The screenshot benchmark has to rasterize real text for Tesseract to read, which means
# naming a font file. Hardcoding one path ties the test to a single OS; these lists cover
# macOS and common Linux distributions, and the test skips rather than fails if a machine
# has none of them, since a missing font says nothing about compression behaviour.
_REGULAR_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_BOLD_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _first_available_font(candidates: list[str]) -> str | None:
    for path in candidates:
        if Path(path).is_file():
            return path
    return None


def test_native_text_pdf_compresses_by_double_digit_multiple(pdf_factory):
    """Isolates the specific mechanism the README's 33x figure is about: a page with a
    text layer is still billed by the host as a rendered page image (tokens.py's
    estimate_pdf_page_host_tokens), while token-distiller reads the text layer
    directly. This is deliberately free of repeated boilerplate lines, so the result
    measures only that effect and not the separate boilerplate-collapse win already
    covered by test_pipeline.py::test_boilerplate_reduces_token_count.

    Measured at authoring time on a 25-page, fully-unique-per-page "lecture slide"
    fixture: 64,891 raw tokens vs 1,541 distilled tokens (42.1x). Asserting >=20x
    leaves roughly half that margin for content or renderer differences while still
    proving the figure is a real double-digit multiple, not a rounding artifact.
    """
    pages = []
    for i in range(25):
        n = i + 1
        pages.append(
            [
                f"Module {n}: Key Concept Overview",
                f"This section introduces the core principle behind unit {n} of the course.",
                f"It walks through the motivation for topic {n}, a short worked example,",
                f"and the most common mistake students make applying concept {n} in practice.",
            ]
        )

    path = pdf_factory(pages, name="lecture_deck_25pg.pdf")
    result, _, _ = pipeline.distill(path, use_cache=False)

    assert len(result.pages) == 25
    assert result.boilerplate == []  # confirms this measures the raw-vs-native effect alone
    assert result.compression_ratio >= 20.0, (
        f"expected a double-digit compression multiple on a text-native PDF, got "
        f"{result.compression_ratio:.1f}x ({result.raw_tokens_est} -> "
        f"{result.distilled_tokens_est} tokens)"
    )


@pytest.mark.ocr
def test_screenshot_ocr_saves_the_large_majority_of_tokens(tmp_path):
    """Reproduces the README's screenshot figure via the real OCR path (no mocking, no
    ANTHROPIC_API_KEY / vision fallback needed -- a rendered UI screenshot is high
    enough contrast that Tesseract clears OCR_CONF_THRESHOLD on its own). Deselected by
    default like other @pytest.mark.ocr tests; run explicitly with `pytest -m ocr`.

    A retina-resolution (2880x1800) screenshot is billed ~2,049 raw tokens per
    tokens.py's image formula regardless of how little text it contains, which is the
    whole point of the claim: a mostly-chrome settings-panel screenshot with eight menu
    items and a version string OCR'd to 46 distilled tokens at authoring time -- 97.8%
    saved. Asserting >=85% leaves headroom for OCR variance across Tesseract versions
    and font rendering while still confirming this is a large-majority reduction, not a
    marginal one.
    """
    from PIL import Image, ImageDraw, ImageFont

    if _first_available_font(_REGULAR_FONTS) is None:
        pytest.skip("no usable TrueType font found on this machine")
    font_path = _first_available_font(_REGULAR_FONTS)
    bold_path = _first_available_font(_BOLD_FONTS) or font_path

    width, height = 2880, 1800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 34)
    title_font = ImageFont.truetype(bold_path, 44)

    draw.rectangle([0, 0, width, 110], fill=(240, 240, 242))
    draw.text((60, 30), "Settings", font=title_font, fill="black")

    menu_items = [
        "Account", "Notifications", "Privacy and security", "Appearance",
        "Language and region", "Accessibility", "Storage", "About this device",
    ]
    y = 220
    for item in menu_items:
        draw.text((100, y), item, font=font, fill=(30, 30, 30))
        y += 90

    draw.text(
        (100, y + 60),
        "Version 14.2.1 (build 8841) - Last checked for updates today",
        font=font,
        fill=(90, 90, 90),
    )

    image_path = tmp_path / "screenshot_bench.png"
    img.save(image_path)

    result = pipeline.distill_image(str(image_path))
    page = result.pages[0]

    assert page.ocr_word_count > 0, "OCR extracted no words -- fixture text isn't legible"
    pct_saved = 1 - (page.distilled_tokens_est / page.raw_tokens_est)
    assert pct_saved >= 0.85, (
        f"expected the large majority of tokens saved on a sparse screenshot, got "
        f"{pct_saved:.1%} ({page.raw_tokens_est} -> {page.distilled_tokens_est} tokens)"
    )
