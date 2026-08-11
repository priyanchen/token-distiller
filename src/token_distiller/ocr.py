import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from token_distiller.config import (
    OCR_LANG,
    OCR_MIN_UPSCALE_PX,
    OCR_RETRY_CONF_THRESHOLD,
    OCR_RETRY_MIN_WORDS,
    OCR_UPSCALE_FACTOR,
)


def _otsu_threshold(histogram: list[int]) -> int:
    """Otsu's method: pick the grey level that maximises between-class variance. PIL has no
    built-in, and a fixed threshold fails badly on figures with tinted or shaded
    backgrounds, which is most of them."""
    total = sum(histogram)
    if total == 0:
        return 128
    sum_all = sum(i * h for i, h in enumerate(histogram))
    weight_bg = 0
    sum_bg = 0
    best_variance = -1.0
    first_best = last_best = 128
    for level in range(256):
        weight_bg += histogram[level]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += level * histogram[level]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            first_best = last_best = level
        elif variance == best_variance:
            last_best = level
    # Every empty grey level between two clusters scores identically, so the winning
    # variance is usually a plateau. Taking its midpoint keeps the threshold away from
    # sitting directly on a peak, where small shifts in exposure flip pixels.
    return (first_best + last_best) // 2


def preprocess(image: Image.Image) -> Image.Image:
    """Grey, stretch contrast, upscale if small, then binarize.

    Tesseract is tuned for roughly 300-DPI black-on-white text. Figures cropped out of a
    PDF are often smaller than that and sit on a coloured panel, which is exactly the case
    raw OCR returns nothing for.
    """
    gray = ImageOps.autocontrast(image.convert("L"))

    if min(gray.size) < OCR_MIN_UPSCALE_PX:
        gray = gray.resize(
            (gray.width * OCR_UPSCALE_FACTOR, gray.height * OCR_UPSCALE_FACTOR),
            Image.LANCZOS,
        )

    threshold = _otsu_threshold(gray.histogram())
    return gray.point(lambda p: 255 if p > threshold else 0, mode="1")


def ocr_image(image: Image.Image, lang: str = OCR_LANG) -> tuple[str, float, int]:
    """Returns (text, mean_word_confidence, word_count). Tesseract marks
    non-text boxes with conf=-1; those are excluded from both the text and the score."""
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)

    words: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_value < 0:
            continue
        words.append(text)
        confidences.append(conf_value)

    full_text = " ".join(words)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return full_text, mean_confidence, len(words)


def ocr_image_best(image: Image.Image, lang: str = OCR_LANG) -> tuple[str, float, int]:
    """OCR the image as-is, then retry preprocessed and keep whichever read better.

    Retrying rather than always preprocessing matters: binarizing can destroy anti-aliased
    body text that reads perfectly well raw, so preprocessing must be able to lose. "Better"
    is judged on word count first — a pass that finds no words is useless however confident
    it claims to be — then on confidence.
    """
    raw = ocr_image(image, lang=lang)
    if raw[1] >= OCR_RETRY_CONF_THRESHOLD and raw[2] >= OCR_RETRY_MIN_WORDS:
        return raw

    try:
        retried = ocr_image(preprocess(image), lang=lang)
    except Exception:
        return raw

    if (retried[2], retried[1]) > (raw[2], raw[1]):
        return retried
    return raw
