import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

from token_distiller.config import (
    FIGURE_MIN_SIDE_PT,
    FIGURE_RENDER_DPI,
    MIN_NATIVE_TEXT_CHARS,
    RENDER_DPI,
)

PDF_POINTS_PER_INCH = 72.0


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
        return pages


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


def rasterize_region(
    pdf_path: str,
    page_index: int,
    bbox_pt: tuple[float, float, float, float],
    page_height_pt: float,
    dpi: int = FIGURE_RENDER_DPI,
) -> Image.Image:
    """Render one figure, not the whole page.

    Cropping matters for both cost and quality: the surrounding body text is already
    captured losslessly by native extraction, so including it would pay vision tokens to
    re-read text we already have, and dilute the model's attention on the actual diagram.
    PDF y-coordinates grow upward from the bottom-left while PIL's grow downward from the
    top-left, so the vertical axis is flipped here.
    """
    page = rasterize_page(pdf_path, page_index, dpi=dpi)
    scale = dpi / PDF_POINTS_PER_INCH
    x0, y0, x1, y1 = bbox_pt
    left = max(0, int(x0 * scale))
    right = min(page.width, int(x1 * scale))
    top = max(0, int((page_height_pt - y1) * scale))
    bottom = min(page.height, int((page_height_pt - y0) * scale))
    if right <= left or bottom <= top:
        return page
    return page.crop((left, top, right, bottom))
