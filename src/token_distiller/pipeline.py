import time
from pathlib import Path

import anthropic

from token_distiller import (
    boilerplate as boilerplate_mod,
    cache,
    image_ingest,
    ocr as ocr_mod,
    pdf_extract,
    vision_fallback,
)
from token_distiller.config import (
    BOILERPLATE_ENABLED,
    CACHE_ENABLED,
    DESCRIBE_FIGURES,
    FIGURE_PROMPT,
    FIGURE_RENDER_DPI,
    OCR_CONF_THRESHOLD,
    OCR_MIN_WORD_COUNT,
    PDF_EXTENSIONS,
)
from token_distiller.models import DistillMethod, DistillResult, PageResult
from token_distiller.tokens import (
    estimate_image_tokens,
    estimate_pdf_page_host_tokens,
    estimate_text_tokens,
)


def _distill_ocr_or_vision(page_index: int, image, allow_vision: bool = True) -> PageResult:
    text, mean_conf, word_count = ocr_mod.ocr_image_best(image)
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
        except (vision_fallback.VisionUnavailable, anthropic.APIError) as exc:
            # A key configuration problem (VisionUnavailable) and a live API failure
            # (rate limit, auth, network, outage) both mean "no vision fallback for this
            # page" -- degrading to the OCR text either way beats losing the whole
            # distillation to one page's transient API error.
            return PageResult(
                page_index=page_index,
                method=DistillMethod.OCR_DEGRADED,
                text=text,
                ocr_confidence=mean_conf,
                ocr_word_count=word_count,
                raw_tokens_est=raw_tokens,
                distilled_tokens_est=estimate_text_tokens(text),
                warnings=[
                    f"low OCR confidence ({mean_conf:.1f}) and vision fallback unavailable: {exc}"
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


def _apply_boilerplate(pages: list[PageResult]) -> list[dict]:
    if not BOILERPLATE_ENABLED or len(pages) < 2:
        return []
    stripped, manifest = boilerplate_mod.strip_boilerplate([p.text for p in pages])
    if not manifest:
        return []
    for page, new_text in zip(pages, stripped):
        page.text = new_text
        # figure text is not part of the page's text layer, so it survives boilerplate
        # stripping untouched and must still be counted
        page.distilled_tokens_est = estimate_text_tokens(new_text) + sum(
            estimate_text_tokens(f) for f in page.figures
        )
    return manifest


def _read_figures(
    path: str,
    page_index: int,
    boxes: list[tuple[float, float, float, float]],
    page_height_pt: float,
    allow_vision: bool,
) -> list[str]:
    """Crop each embedded figure and put it through the same OCR -> vision chain used for
    scanned pages. Failures are skipped rather than raised: a figure we cannot read leaves
    the page flagged as uncaptured, which is the honest outcome, and must never take down
    the extraction of the page's text."""
    try:
        # rendered once per page, not once per figure
        page_image = pdf_extract.rasterize_page(path, page_index, dpi=FIGURE_RENDER_DPI)
    except Exception:
        return []

    recovered: list[str] = []
    for bbox in boxes:
        try:
            crop = pdf_extract.crop_region(page_image, bbox, page_height_pt)
        except Exception:
            continue

        text, mean_conf, word_count = ocr_mod.ocr_image_best(crop)
        if mean_conf < OCR_CONF_THRESHOLD or word_count < OCR_MIN_WORD_COUNT:
            if allow_vision:
                try:
                    text = vision_fallback.describe_image(crop, prompt=FIGURE_PROMPT)
                except (vision_fallback.VisionUnavailable, anthropic.APIError):
                    # Same reasoning as the page-level fallback: a live API failure here
                    # must not abort the rest of the document. The figure stays unread and
                    # the page stays flagged via pages_with_uncaptured_images() -- the
                    # honest outcome _read_figures already promises for any failure.
                    pass
        if text and text.strip():
            recovered.append(text.strip())
    return recovered


def distill_pdf(
    path: str, allow_vision: bool = True, describe_figures: bool | None = None
) -> DistillResult:
    start = time.monotonic()
    if describe_figures is None:
        describe_figures = DESCRIBE_FIGURES
    native_pages = pdf_extract.extract_pages_with_figures(path)
    pages: list[PageResult] = []

    for i, (native_text, width_pt, height_pt, boxes) in enumerate(native_pages):
        if pdf_extract.has_native_text(native_text):
            figures = (
                _read_figures(path, i, boxes, height_pt, allow_vision)
                if (describe_figures and boxes)
                else []
            )
            pages.append(
                PageResult(
                    page_index=i,
                    method=DistillMethod.NATIVE_TEXT,
                    text=native_text,
                    raw_tokens_est=estimate_pdf_page_host_tokens(
                        width_pt, height_pt, native_text
                    ),
                    distilled_tokens_est=estimate_text_tokens(native_text)
                    + sum(estimate_text_tokens(f) for f in figures),
                    image_count=len(boxes),
                    figures=figures,
                )
            )
        else:
            image = pdf_extract.rasterize_page(path, i)
            pages.append(_distill_ocr_or_vision(i, image, allow_vision=allow_vision))

    manifest = _apply_boilerplate(pages)
    duration_ms = round((time.monotonic() - start) * 1000)
    return DistillResult(
        source_path=str(path),
        source_type="pdf",
        pages=pages,
        duration_ms=duration_ms,
        boilerplate=manifest,
    )


def distill_image(path: str, allow_vision: bool = True) -> DistillResult:
    start = time.monotonic()
    image = image_ingest.load_image(path)
    page = _distill_ocr_or_vision(0, image, allow_vision=allow_vision)
    duration_ms = round((time.monotonic() - start) * 1000)
    return DistillResult(
        source_path=str(path), source_type="image", pages=[page], duration_ms=duration_ms
    )


def _distill_uncached(
    path: str, allow_vision: bool = True, describe_figures: bool | None = None
) -> DistillResult:
    suffix = Path(path).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return distill_pdf(path, allow_vision=allow_vision, describe_figures=describe_figures)
    if image_ingest.is_image(path):
        return distill_image(path, allow_vision=allow_vision)
    raise ValueError(f"unsupported file type: {suffix}")


def distill(
    path: str,
    allow_vision: bool = True,
    use_cache: bool = True,
    describe_figures: bool | None = None,
) -> tuple[DistillResult, int | None, bool]:
    """Returns (result, cache_handle, was_cache_hit). The handle is what `distill expand`
    resolves, so any caller that shortens its output can still point at the full text."""
    if not (use_cache and CACHE_ENABLED):
        return (
            _distill_uncached(
                path, allow_vision=allow_vision, describe_figures=describe_figures
            ),
            None,
            False,
        )

    hash_value = cache.content_hash(path)
    hit = cache.get(hash_value)
    if hit is not None:
        handle, result = hit
        result.source_path = str(path)
        return result, handle, True

    result = _distill_uncached(
        path, allow_vision=allow_vision, describe_figures=describe_figures
    )
    handle = cache.put(hash_value, result)
    return result, handle, False
