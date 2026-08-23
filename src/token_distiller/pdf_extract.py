import re
import subprocess

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

from token_distiller.config import (
    FIGURE_MIN_SIDE_PT,
    FIGURE_RENDER_DPI,
    MIN_NATIVE_TEXT_CHARS,
    PDFTOTEXT_TIMEOUT_S,
    RENDER_DPI,
    RTL_REORDER_ENABLED,
)

PDF_POINTS_PER_INCH = 72.0

# Hebrew, Arabic, Syriac, Thaana, N'Ko, and the Hebrew/Arabic presentation-form blocks.
# Matching on codepoint works even when the text is reversed, since reversal changes the
# order of characters, not which characters they are.
_RTL_RE = re.compile(
    "[֐-׿؀-ۿ܀-ݏݐ-ݿހ-޿߀-߿"
    "ࢠ-ࣿיִ-﷿ﹰ-﻿]"
)

# Poppler wraps every directional run in BiDi embedding controls. They are invisible and
# carry no meaning once the text is already in logical order, but cost real tokens --
# measured at 156 of 2054 characters (7.6%) on one Hebrew page.
_BIDI_CONTROLS = dict.fromkeys(
    [0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)]
)


def contains_rtl(text: str) -> bool:
    return _RTL_RE.search(text) is not None


def _sample_is_rtl(pdf_path: str) -> bool:
    """Classifies a document as RTL from a handful of spread pages rather than reading all
    of them, so a large RTL document can skip pdfplumber's full-document extract_text()
    pass entirely -- that per-page call, not text extraction itself, is what dominates its
    cost (measured: 24.75s for 447 pages even with extract_text skipped). Sampling first,
    fourth, middle, three-quarter, and last page took ~0.2s on every document measured this
    session, correctly separating Hebrew and Latin documents in every case tried.

    A false negative here is not a correctness risk: any page pdfplumber's full pass later
    finds to contain RTL still goes through _reorder_rtl_pages regardless of this sample.
    It only costs the speedup this classification exists to provide.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)
            if n == 0:
                return False
            idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
            return any(contains_rtl(pdf.pages[i].extract_text() or "") for i in idx)
    except Exception:
        return False


def pdftotext_pages(pdf_path: str) -> list[str] | None:
    """Page texts via poppler, which implements the Unicode bidirectional algorithm.

    Returns None when poppler is unavailable or fails, so the caller keeps pdfplumber's
    text rather than losing the page entirely: reversed text is bad, no text is worse.
    """
    try:
        proc = subprocess.run(
            ["pdftotext", "-q", pdf_path, "-"],
            capture_output=True,
            timeout=PDFTOTEXT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = proc.stdout.decode("utf-8", errors="replace")
    parts = text.split("\f")
    # Poppler writes a form feed after every page including the last, so the split leaves one
    # empty trailing element. Only that one may be dropped: rstrip("\f") would also swallow
    # the feeds belonging to genuinely blank trailing pages, and the resulting page-count
    # mismatch silently disables substitution for the whole document.
    if parts and not parts[-1]:
        parts.pop()
    return [page.translate(_BIDI_CONTROLS).strip() for page in parts]


def extract_native_pages(pdf_path: str) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [(page.extract_text() or "").strip() for page in pdf.pages]


def extract_pages_with_dimensions(pdf_path: str) -> list[tuple[str, float, float, int]]:
    """(text, width_pt, height_pt, embedded_image_count) per page — dimensions feed the
    host-ingestion baseline; image_count feeds PageResult.image_count so a page that
    has enough native text to skip OCR/vision entirely can still be flagged when it
    also carries a figure/diagram/illustration that native-text extraction can't see.
    Read from the same pdfplumber pass already extracting text, so this costs nothing
    extra on top of the (already dominant) per-page extract_text() call."""
    return [(text, w, h, len(boxes)) for text, w, h, boxes in extract_pages_with_figures(pdf_path)]


def extract_pages_with_figures(
    pdf_path: str,
) -> list[tuple[str, float, float, list[tuple[float, float, float, float]]]]:
    """(text, width_pt, height_pt, [figure_bbox...]) per page.

    Bounding boxes are in PDF points with the origin at the page's bottom-left, which is
    what `rasterize_region` expects. Decorative rules and hairline spacers are filtered
    out by FIGURE_MIN_SIDE_PT: describing a 2pt-tall background strip costs a vision call
    and returns nothing useful.
    """
    if _sample_is_rtl(pdf_path):
        fast = _extract_pages_fast(pdf_path)
        if fast is not None:
            return fast
        # poppler failed, or the two extractors disagreed on page count -- fall through to
        # the pdfplumber path below, which still corrects any RTL page it finds.

    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page in pdf.pages:
            boxes = [
                (float(im["x0"]), float(im["y0"]), float(im["x1"]), float(im["y1"]))
                for im in page.images
                if (im["x1"] - im["x0"]) >= FIGURE_MIN_SIDE_PT
                and (im["y1"] - im["y0"]) >= FIGURE_MIN_SIDE_PT
            ]
            pages.append(
                ((page.extract_text() or "").strip(), float(page.width), float(page.height), boxes)
            )
    return _reorder_rtl_pages(pdf_path, pages)


def _image_bearing_pages(reader) -> tuple[list[tuple[float, float]], list[int]]:
    """Page dimensions and the indices of pages carrying an image XObject, from pypdf.

    Rotation matters: pdfplumber's width/height are post-rotation, while mediabox is the
    unrotated box, so a /Rotate 90 page would otherwise report its dimensions transposed
    and skew the raw-token baseline that feeds off them.
    """
    dims: list[tuple[float, float]] = []
    image_pages: list[int] = []
    for i, page in enumerate(reader.pages):
        box = page.mediabox
        width, height = float(box.width), float(box.height)
        if (page.rotation or 0) % 180 == 90:
            width, height = height, width
        dims.append((width, height))

        xobjects = (page.get("/Resources") or {}).get("/XObject")
        if not xobjects:
            continue
        try:
            resolved = xobjects.get_object().values()
            if any(x.get_object().get("/Subtype") == "/Image" for x in resolved):
                image_pages.append(i)
        except Exception:
            # A malformed resource dict must not lose the page; treat it as image-bearing
            # so pdfplumber gets a chance to look properly.
            image_pages.append(i)
    return dims, image_pages


def _extract_pages_fast(pdf_path: str):
    """Geometry from pypdf, text from poppler, figure boxes from pdfplumber -- and only for
    the pages that actually have figures. Only called for documents _sample_is_rtl has
    already classified as RTL, so this is never on the path for a Latin-script document.

    pdfplumber's cost is parsing a page at all rather than extracting its text (24.75s for
    447 pages even with extract_text skipped), so the saving comes from not visiting pages
    that have nothing to look at. Poppler's text is used directly -- not run back through
    _reorder_rtl_pages -- since pdftotext_pages already returns logically-ordered,
    BiDi-stripped text; substituting it again would just repeat the same subprocess call.

    Returns None if anything is inconsistent, so the caller falls back to the pdfplumber
    path instead of guessing.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_path)
        dims, image_pages = _image_bearing_pages(reader)
    except Exception:
        return None

    texts = pdftotext_pages(pdf_path)
    if texts is None or len(texts) != len(dims):
        return None

    boxes_by_page: dict[int, list] = {}
    if image_pages:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i in image_pages:
                    boxes_by_page[i] = [
                        (float(im["x0"]), float(im["y0"]), float(im["x1"]), float(im["y1"]))
                        for im in pdf.pages[i].images
                        if (im["x1"] - im["x0"]) >= FIGURE_MIN_SIDE_PT
                        and (im["y1"] - im["y0"]) >= FIGURE_MIN_SIDE_PT
                    ]
        except Exception:
            return None

    return [
        (texts[i], dims[i][0], dims[i][1], boxes_by_page.get(i, []))
        for i in range(len(dims))
    ]


def _reorder_rtl_pages(
    pdf_path: str, pages: list[tuple[str, float, float, list]]
) -> list[tuple[str, float, float, list]]:
    """Replace the text of RTL pages with poppler's logically-ordered version.

    Substitution is per page and only where pdfplumber actually saw RTL characters, so a
    Latin-script document takes a byte-identical path to before this existed. A page-count
    mismatch aborts the whole substitution: the two extractors would then disagree about the
    document's structure, and pairing text with the wrong page's figures is a worse failure
    than reversed text.
    """
    if not RTL_REORDER_ENABLED or not any(contains_rtl(text) for text, _, _, _ in pages):
        return pages

    reordered = pdftotext_pages(pdf_path)
    if reordered is None or len(reordered) != len(pages):
        return pages

    return [
        (reordered[i] if contains_rtl(text) and reordered[i] else text, w, h, boxes)
        for i, (text, w, h, boxes) in enumerate(pages)
    ]


def page_count(pdf_path: str) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def has_native_text(text: str) -> bool:
    return len(text) >= MIN_NATIVE_TEXT_CHARS


def rasterize_page(pdf_path: str, page_index: int, dpi: int = RENDER_DPI) -> Image.Image:
    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=page_index + 1, last_page=page_index + 1
    )
    return images[0]


def crop_region(
    page: Image.Image,
    bbox_pt: tuple[float, float, float, float],
    page_height_pt: float,
    dpi: int = FIGURE_RENDER_DPI,
) -> Image.Image:
    """Cut one figure out of an already-rendered page.

    Cropping matters for both cost and quality: the surrounding body text is already
    captured losslessly by native extraction, so including it would pay vision tokens to
    re-read text we already have, and dilute the model's attention on the actual diagram.
    PDF y-coordinates grow upward from the bottom-left while PIL's grow downward from the
    top-left, so the vertical axis is flipped here.
    """
    scale = dpi / PDF_POINTS_PER_INCH
    x0, y0, x1, y1 = bbox_pt
    left = max(0, int(x0 * scale))
    right = min(page.width, int(x1 * scale))
    top = max(0, int((page_height_pt - y1) * scale))
    bottom = min(page.height, int((page_height_pt - y0) * scale))
    if right <= left or bottom <= top:
        return page
    return page.crop((left, top, right, bottom))


def rasterize_region(
    pdf_path: str,
    page_index: int,
    bbox_pt: tuple[float, float, float, float],
    page_height_pt: float,
    dpi: int = FIGURE_RENDER_DPI,
) -> Image.Image:
    """Render a page and return one figure from it. Callers handling several figures on the
    same page should render once with rasterize_page and use crop_region instead — this
    convenience wrapper re-renders per call."""
    page = rasterize_page(pdf_path, page_index, dpi=dpi)
    return crop_region(page, bbox_pt, page_height_pt, dpi=dpi)
