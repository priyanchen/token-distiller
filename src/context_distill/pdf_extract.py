import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

from context_distill.config import MIN_NATIVE_TEXT_CHARS, RENDER_DPI


def extract_native_pages(pdf_path: str) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [(page.extract_text() or "").strip() for page in pdf.pages]


def extract_pages_with_dimensions(pdf_path: str) -> list[tuple[str, float, float]]:
    """(text, width_pt, height_pt) per page — dimensions feed the host-ingestion baseline."""
    with pdfplumber.open(pdf_path) as pdf:
        return [
            ((page.extract_text() or "").strip(), float(page.width), float(page.height))
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
