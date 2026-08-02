"""Token estimation. Claude's exact tokenizer isn't public, so these are documented
approximations: chars/4 for text (Anthropic's own rough-estimate guidance), and the
published image-token pixel formula for image content.

The "raw" side deliberately models what the *host* pays to ingest the file, not what
the file's text alone would cost. A PDF read natively is rendered to a page image and
billed for those pixels on top of its text, so scoring a text-native PDF as
raw == distilled would report 1.0x on a file we in fact compress heavily.
"""

from token_distiller.config import (
    CHARS_PER_TOKEN,
    HOST_PDF_RENDER_DPI,
    IMAGE_MAX_EDGE_PX,
    IMAGE_TOKEN_DIVISOR,
)

PDF_POINTS_PER_INCH = 72.0


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def downscaled_dimensions(width_px: int, height_px: int) -> tuple[int, int]:
    longest = max(width_px, height_px)
    if longest <= IMAGE_MAX_EDGE_PX:
        return width_px, height_px
    scale = IMAGE_MAX_EDGE_PX / longest
    return max(1, round(width_px * scale)), max(1, round(height_px * scale))


def estimate_image_tokens(width_px: int, height_px: int) -> int:
    """Anthropic downscales any image whose long edge exceeds 1568px before billing, so
    a retina screenshot is not priced at its stored resolution."""
    if width_px <= 0 or height_px <= 0:
        return 0
    w, h = downscaled_dimensions(width_px, height_px)
    return max(1, round((w * h) / IMAGE_TOKEN_DIVISOR))


def estimate_pdf_page_host_tokens(
    width_pt: float, height_pt: float, page_text: str, dpi: int = HOST_PDF_RENDER_DPI
) -> int:
    """What a host pays to ingest one PDF page: the rendered page image plus its text."""
    scale = dpi / PDF_POINTS_PER_INCH
    width_px = max(1, round(width_pt * scale))
    height_px = max(1, round(height_pt * scale))
    return estimate_image_tokens(width_px, height_px) + estimate_text_tokens(page_text)
