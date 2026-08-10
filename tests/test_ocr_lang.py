"""Covers the OCR language setting. Tesseract does not detect script automatically, so the
language it is asked for decides whether a page reads at all -- a Hebrew page OCR'd as
English returns near-nothing and then surfaces as "could not be read"."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from token_distiller import ocr
from token_distiller.config import OCR_LANG

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font() -> str | None:
    return next((p for p in _FONTS if Path(p).is_file()), None)


def test_default_language_is_english():
    assert OCR_LANG == "eng"


def test_language_is_read_from_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("TOKEN_DISTILLER_OCR_LANG", "eng+heb")
    from token_distiller import config

    try:
        assert importlib.reload(config).OCR_LANG == "eng+heb"
    finally:
        monkeypatch.delenv("TOKEN_DISTILLER_OCR_LANG", raising=False)
        importlib.reload(config)


def test_both_ocr_passes_use_the_same_language(monkeypatch):
    """The preprocessed retry must not silently fall back to English when a different
    language was requested -- that would make the retry read worse than the first pass for
    exactly the non-English pages it exists to rescue."""
    seen = []

    def fake_ocr_image(image, lang=OCR_LANG):
        seen.append(lang)
        return "", 0.0, 0  # zero words forces ocr_image_best to try the retry pass

    monkeypatch.setattr("token_distiller.ocr.ocr_image", fake_ocr_image)
    ocr.ocr_image_best(Image.new("L", (200, 80), "white"), lang="eng+heb")

    assert seen == ["eng+heb", "eng+heb"]


@pytest.mark.ocr
def test_an_explicit_language_still_reads_real_text():
    """Guards the wiring against real Tesseract: a bad `lang` value makes pytesseract raise,
    so this fails loudly if the argument is malformed or unsupported."""
    font_path = _font()
    if font_path is None:
        pytest.skip("no usable TrueType font on this machine")

    img = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 80), "Strategic clarity compounds", font=ImageFont.truetype(font_path, 48),
              fill="black")

    text, confidence, words = ocr.ocr_image(img, lang="eng")

    assert words >= 3, f"expected real words, got {words}: {text!r}"
    assert "clarity" in text.lower()
