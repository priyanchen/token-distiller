import pytesseract
from PIL import Image
from pytesseract import Output


def ocr_image(image: Image.Image) -> tuple[str, float, int]:
    """Returns (text, mean_word_confidence, word_count). Tesseract marks
    non-text boxes with conf=-1; those are excluded from both the text and the score."""
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

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
