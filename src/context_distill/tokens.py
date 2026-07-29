"""Token estimation. Claude's exact tokenizer isn't public, so these are documented
approximations: chars/4 for text (Anthropic's own rough-estimate guidance), and the
published image-token pixel formula for image-only content."""

from context_distill.config import CHARS_PER_TOKEN, IMAGE_TOKEN_DIVISOR


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def estimate_image_tokens(width_px: int, height_px: int) -> int:
    if width_px <= 0 or height_px <= 0:
        return 0
    return max(1, round((width_px * height_px) / IMAGE_TOKEN_DIVISOR))
