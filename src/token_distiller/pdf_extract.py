import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

from token_distiller.config import MIN_NATIVE_TEXT_CHARS, RENDER_DPI


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
    with pdfplumber.open(pdf_path) as pdf:
        return [
            (
                (page.extract_text() or "").strip(),
                float(page.width),
                float(page.height),
                len(page.images),
            )
            for page in pdf.pages
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
