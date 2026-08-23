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


def _fake_pdftotext(monkeypatch, stdout: bytes):
    class Proc:
        pass

    proc = Proc()
    proc.stdout = stdout
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: proc)


def test_trailing_blank_pages_are_preserved(monkeypatch):
    """Poppler writes a form feed after every page, so only the single empty element after
    the last one may be dropped. rstrip("\f") also swallowed the feeds of genuinely blank
    trailing pages: a real 351-page book reported 349, which tripped the page-count guard
    and silently disabled RTL substitution for the entire document.
    """
    # two pages of text followed by two blank pages
    _fake_pdftotext(monkeypatch, "one\ftwo\f\f\f".encode())
    assert pdf_extract.pdftotext_pages("x.pdf") == ["one", "two", "", ""]


def test_single_page_document_yields_one_page(monkeypatch):
    _fake_pdftotext(monkeypatch, "only\f".encode())
    assert pdf_extract.pdftotext_pages("x.pdf") == ["only"]


def test_empty_output_yields_no_pages(monkeypatch):
    _fake_pdftotext(monkeypatch, b"")
    assert pdf_extract.pdftotext_pages("x.pdf") == []


def test_sample_is_rtl_true_for_a_document_with_hebrew_pages(monkeypatch):
    """The sample only needs one of the pages it looks at to contain RTL text. Mocked at
    the pdfplumber boundary rather than built as a real PDF: the hand-rolled fixture
    builder in conftest.py encodes page content as latin-1, so it cannot represent Hebrew
    at all -- confirmed directly, it raises UnicodeEncodeError rather than producing
    garbled-but-present text. The real end-to-end proof continues to be the Odyssey and
    the two Hebrew documents verified manually against this session's real files.
    """

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdf:
        pages = [FakePage("Latin front matter.")] * 3 + [FakePage("עליון רמיית בית משפט")]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdf_extract.pdfplumber, "open", lambda _p: FakePdf())
    assert pdf_extract._sample_is_rtl("mixed.pdf")


def test_sample_is_rtl_false_for_a_latin_only_document(pdf_factory):
    path = pdf_factory([["Strategic clarity compounds over time."]] * 5, name="latin.pdf")
    assert not pdf_extract._sample_is_rtl(path)


def test_sample_is_rtl_false_for_an_empty_document(monkeypatch):
    class FakePdf:
        pages = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdf_extract.pdfplumber, "open", lambda _p: FakePdf())
    assert not pdf_extract._sample_is_rtl("empty.pdf")


def test_sample_is_rtl_false_when_the_file_cannot_be_opened(monkeypatch):
    def explode(_p):
        raise Exception("corrupt")

    monkeypatch.setattr(pdf_extract.pdfplumber, "open", explode)
    assert not pdf_extract._sample_is_rtl("broken.pdf")


def test_extract_pages_fast_returns_none_on_page_count_mismatch(monkeypatch):
    """Same guard as _reorder_rtl_pages: a disagreement between pypdf and poppler about
    page count must abort the fast path rather than pair text with the wrong page."""

    class FakePage:
        rotation = 0
        mediabox = type("MB", (), {"width": 600, "height": 800})()

        def get(self, _key, default=None):
            return default

    class FakeReader:
        pages = [FakePage(), FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", lambda _p: FakeReader())
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: ["only one page"])

    assert pdf_extract._extract_pages_fast("x.pdf") is None


def test_extract_pages_fast_returns_none_when_poppler_fails(monkeypatch):
    class FakePage:
        rotation = 0
        mediabox = type("MB", (), {"width": 600, "height": 800})()

        def get(self, _key, default=None):
            return default

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", lambda _p: FakeReader())
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: None)

    assert pdf_extract._extract_pages_fast("x.pdf") is None


def test_extract_pages_fast_transposes_dimensions_for_rotated_pages(monkeypatch):
    """pypdf's mediabox is the unrotated box; a /Rotate 90 page must still report the
    dimensions the rest of the pipeline expects (post-rotation), or the host-ingestion
    token baseline that feeds off page width/height would be computed sideways."""

    class FakePage:
        rotation = 90
        mediabox = type("MB", (), {"width": 600, "height": 800})()  # portrait, unrotated

        def get(self, _key, default=None):
            return default

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", lambda _p: FakeReader())
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: ["טקסט"])

    result = pdf_extract._extract_pages_fast("x.pdf")
    text, width, height, boxes = result[0]
    assert (width, height) == (800.0, 600.0)  # transposed: landscape after rotation


def test_extract_pages_fast_only_visits_pdfplumber_for_image_bearing_pages(monkeypatch):
    """The entire point of the fast path: pdfplumber must not be asked to look at a page
    pypdf already determined has no images -- that per-page access is what actually costs
    time (measured: 0.50s for 28 pages vs 10.42s for all 351 on the same real book), not
    the pdfplumber.open() call itself, so the test has to prove which page indices were
    touched, not just that open() happened."""

    class FakeXObjectImage:
        def get_object(self):
            return {"/Subtype": "/Image"}

    class FakePageWithImage:
        rotation = 0
        mediabox = type("MB", (), {"width": 600, "height": 800})()

        def get(self, key, default=None):
            if key == "/Resources":
                return {"/XObject": self}
            return default

        def get_object(self):
            return {"x": FakeXObjectImage()}

    class FakePageWithoutImage:
        rotation = 0
        mediabox = type("MB", (), {"width": 600, "height": 800})()

        def get(self, _key, default=None):
            return default

    class FakeReader:
        pages = [FakePageWithImage(), FakePageWithoutImage()]

    accessed_indices = []

    class FakePdfplumberPage:
        def __init__(self, index):
            self._index = index

        @property
        def images(self):
            accessed_indices.append(self._index)
            return []

    class FakePdfplumberPages(list):
        def __getitem__(self, i):
            page = super().__getitem__(i)
            return page

    class FakePdfplumber:
        pages = FakePdfplumberPages([FakePdfplumberPage(0), FakePdfplumberPage(1)])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("pypdf.PdfReader", lambda _p: FakeReader())
    monkeypatch.setattr(pdf_extract, "pdftotext_pages", lambda _p: ["a", "b"])
    monkeypatch.setattr(pdf_extract.pdfplumber, "open", lambda _p: FakePdfplumber())

    result = pdf_extract._extract_pages_fast("x.pdf")

    assert result is not None
    assert accessed_indices == [0]  # only the image-bearing page's .images was touched
