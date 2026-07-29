import time
from pathlib import Path

from context_distill import image_ingest, ocr as ocr_mod, pdf_extract, vision_fallback
from context_distill.config import OCR_CONF_THRESHOLD, OCR_MIN_WORD_COUNT, PDF_EXTENSIONS
from context_distill.models import DistillMethod, DistillResult, PageResult
from context_distill.tokens import estimate_image_tokens, estimate_text_tokens


def _distill_ocr_or_vision(page_index: int, image, allow_vision: bool = True) -> PageResult:
    text, mean_conf, word_count = ocr_mod.ocr_image(image)
    raw_tokens = estimate_image_tokens(*image.size)
    needs_fallback = mean_conf < OCR_CONF_THRESHOLD or word_count < OCR_MIN_WORD_COUNT

    if needs_fallback and allow_vision:
        try:
            vision_text = vision_fallback.describe_image(image)
            return PageResult(
                page_index=page_index,
                method=DistillMethod.VISION,
                text=vision_text,
                ocr_confidence=mean_conf,
                ocr_word_count=word_count,
                raw_tokens_est=raw_tokens,
                distilled_tokens_est=estimate_text_tokens(vision_text),
            )
        except vision_fallback.VisionUnavailable as exc:
            return PageResult(
                page_index=page_index,
                method=DistillMethod.OCR_DEGRADED,
                text=text,
                ocr_confidence=mean_conf,
                ocr_word_count=word_count,
                raw_tokens_est=raw_tokens,
                distilled_tokens_est=estimate_text_tokens(text),
                warnings=[
                    f"low OCR confidence ({mean_conf:.1f}) and no vision fallback available: {exc}"
                ],
            )

    return PageResult(
        page_index=page_index,
        method=DistillMethod.OCR,
        text=text,
        ocr_confidence=mean_conf,
        ocr_word_count=word_count,
        raw_tokens_est=raw_tokens,
        distilled_tokens_est=estimate_text_tokens(text),
    )


def distill_pdf(path: str, allow_vision: bool = True) -> DistillResult:
    start = time.monotonic()
    native_pages = pdf_extract.extract_native_pages(path)
    pages: list[PageResult] = []

    for i, native_text in enumerate(native_pages):
        if pdf_extract.has_native_text(native_text):
            tok = estimate_text_tokens(native_text)
            pages.append(
                PageResult(
                    page_index=i,
                    method=DistillMethod.NATIVE_TEXT,
                    text=native_text,
                    raw_tokens_est=tok,
                    distilled_tokens_est=tok,
                )
            )
        else:
            image = pdf_extract.rasterize_page(path, i)
            pages.append(_distill_ocr_or_vision(i, image, allow_vision=allow_vision))

    duration_ms = round((time.monotonic() - start) * 1000)
    return DistillResult(source_path=str(path), source_type="pdf", pages=pages, duration_ms=duration_ms)


def distill_image(path: str, allow_vision: bool = True) -> DistillResult:
    start = time.monotonic()
    image = image_ingest.load_image(path)
    page = _distill_ocr_or_vision(0, image, allow_vision=allow_vision)
    duration_ms = round((time.monotonic() - start) * 1000)
    return DistillResult(source_path=str(path), source_type="image", pages=[page], duration_ms=duration_ms)


def distill(path: str, allow_vision: bool = True) -> DistillResult:
    suffix = Path(path).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return distill_pdf(path, allow_vision=allow_vision)
    if image_ingest.is_image(path):
        return distill_image(path, allow_vision=allow_vision)
    raise ValueError(f"unsupported file type: {suffix}")
