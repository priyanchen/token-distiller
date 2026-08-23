"""Covers right-to-left text extraction.

pdfplumber returns glyphs in visual order, so Hebrew and Arabic come out with every word
reversed -- and unlike a bad OCR pass there is no confidence score to flag it, so the
reversed text reaches the model looking like ordinary text.
"""

import subprocess

import pytest

from token_distiller import pdf_extract


def test_detects_hebrew():
    assert pdf_extract.contains_rtl("עליון רמיית בית משפט")


def test_detects_arabic():
    assert pdf_extract.contains_rtl("مرحبا بالعالم")


def test_latin_is_not_rtl():
    assert not pdf_extract.contains_rtl("Strategic clarity compounds")
    assert not pdf_extract.contains_rtl("1234 -- ABC, xyz.")


def test_detects_rtl_even_when_reversed():
    """Detection has to work on pdfplumber's reversed output, which is the only form we
    ever see before deciding to re-extract."""
    assert pdf_extract.contains_rtl("טפשמ תיב תיימר ןוילע")


def test_bidi_controls_are_stripped():
    """Poppler wraps each directional run in embedding controls -- invisible, meaningless
    once the text is in logical order, and 7.6% of the characters on a real Hebrew page."""
    wrapped = "‫עליון‬ ‪Gmail‬"
    assert wrapped.translate(pdf_extract._BIDI_CONTROLS) == "עליון Gmail"


def test_latin_document_never_calls_pdftotext(monkeypatch, pdf_factory):
    """The whole safety argument for this change is that Latin documents take an identical
    path to before it existed."""
    def explode(*_a, **_k):
        raise AssertionError("pdftotext must not run for a Latin-script document")

    monkeypatch.setattr(pdf_extract, "pdftotext_pages", explode)
    path = pdf_factory([["Strategic clarity compounds over time."]], name="latin.pdf")
    pages = pdf_extract.extract_pages_with_figures(path)
    assert "Strategic clarity" in pages[0][0]


def test_falls_back_to_pdfplumber_when_poppler_is_missing(monkeypatch):
    """No text is worse than reversed text, so a missing or broken poppler must degrade to
    pdfplumber's output rather than losing the page."""
    def missing(*_a, **_k):
        raise FileNotFoundError("pdftotext")

    monkeypatch.setattr(subprocess, "run", missing)
    assert pdf_extract.pdftotext_pages("whatever.pdf") is None


def test_falls_back_when_poppler_times_out(monkeypatch):
    def slow(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="pdftotext", timeout=1)

    monkeypatch.setattr(subprocess, "run", slow)
    assert pdf_extract.pdftotext_pages("whatever.pdf") is None


def test_page_count_mismatch_aborts_substitution(monkeypatch):
    """Pairing one page's text with another page's figure boxes is a worse failure than
    reversed text, so a disagreement between the two extractors keeps pdfplumber's output."""
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: ["only one page"])
    pages = [("טפשמ תיב", 600.0, 800.0, []), ("עליון", 600.0, 800.0, [])]
    assert pdf_extract._reorder_rtl_pages("x.pdf", pages) == pages


def test_empty_replacement_keeps_the_original_text(monkeypatch):
    """A page poppler returns nothing for must keep whatever pdfplumber found."""
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: [""])
    pages = [("טפשמ תיב", 600.0, 800.0, [])]
    assert pdf_extract._reorder_rtl_pages("x.pdf", pages)[0][0] == "טפשמ תיב"


def test_substitution_replaces_only_rtl_pages(monkeypatch):
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: ["עליון", "REPLACED"])
    pages = [("ןוילע", 600.0, 800.0, []), ("Latin page", 600.0, 800.0, [])]
    out = pdf_extract._reorder_rtl_pages("x.pdf", pages)
    assert out[0][0] == "עליון"       # RTL page: substituted
    assert out[1][0] == "Latin page"  # Latin page: untouched


def test_disabling_the_feature_skips_substitution(monkeypatch):
    monkeypatch.setattr(pdf_extract, "RTL_REORDER_ENABLED", False)
    monkeypatch.setattr(
        pdf_extract, "pdftotext_pages", lambda _p: pytest.fail("should not be called")
    )
    pages = [("טפשמ תיב", 600.0, 800.0, [])]
    assert pdf_extract._reorder_rtl_pages("x.pdf", pages) == pages
