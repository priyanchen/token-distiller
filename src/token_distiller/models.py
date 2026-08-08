from dataclasses import dataclass, field
from enum import Enum


class DistillMethod(str, Enum):
    NATIVE_TEXT = "native_text"
    OCR = "ocr"
    OCR_DEGRADED = "ocr_degraded"
    VISION = "vision"


@dataclass
class PageResult:
    page_index: int
    method: DistillMethod
    text: str
    ocr_confidence: float | None = None
    ocr_word_count: int | None = None
    raw_tokens_est: int = 0
    distilled_tokens_est: int = 0
    warnings: list[str] = field(default_factory=list)
    # Embedded images (diagrams, figures, illustrations) found on this page by the
    # PDF's object structure, independent of whether the page also has a text layer.
    # Only meaningful for method == NATIVE_TEXT: the OCR/vision paths already rasterize
    # and read the whole page image, so nothing on it is missed. A native-text page
    # with image_count > 0 is the case that matters -- text extraction reads the text
    # layer only, so any information carried solely by the image is not represented
    # anywhere in the distilled output.
    image_count: int = 0
    # Text recovered from this page's embedded figures via the OCR/vision chain. One entry
    # per figure that was successfully read; empty when figure reading is disabled or when
    # every attempt failed, which is what keeps pages_with_uncaptured_images() honest.
    figures: list[str] = field(default_factory=list)

    @property
    def text_with_figures(self) -> str:
        """Figure text is labelled rather than spliced silently into the body, so a reader
        can tell prose from a transcribed diagram."""
        if not self.figures:
            return self.text
        blocks = [
            f"[figure {i} on page {self.page_index + 1}] {fig}"
            for i, fig in enumerate(self.figures, start=1)
        ]
        return "\n\n".join([self.text, *blocks]) if self.text else "\n\n".join(blocks)


@dataclass
class DistillResult:
    source_path: str
    source_type: str  # "pdf" | "image"
    pages: list[PageResult]
    duration_ms: int = 0
    boilerplate: list[dict] = field(default_factory=list)

    @property
    def raw_tokens_est(self) -> int:
        return sum(p.raw_tokens_est for p in self.pages)

    @property
    def distilled_tokens_est(self) -> int:
        return sum(p.distilled_tokens_est for p in self.pages)

    @property
    def compression_ratio(self) -> float:
        # A page that distilled to nothing (a photo with no readable text) compressed
        # completely; reporting 0.0 there would read as "no compression at all".
        return self.raw_tokens_est / max(1, self.distilled_tokens_est)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text_with_figures for p in self.pages)

    @property
    def rendered_text(self) -> str:
        """Text as a consumer should see it: collapsed boilerplate is restated once up
        front rather than silently dropped."""
        if not self.boilerplate:
            return self.text
        from token_distiller.boilerplate import render_manifest

        return f"{render_manifest(self.boilerplate)}\n\n{self.text}"

    @property
    def warnings(self) -> list[str]:
        return [w for p in self.pages for w in p.warnings]

    def method_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.pages:
            counts[p.method.value] = counts.get(p.method.value, 0) + 1
        return counts

    def pages_with_uncaptured_images(self) -> list[int]:
        """Native-text pages carrying embedded images that were *not* read.

        A page qualifies only when no figure text was recovered for it — figure reading
        disabled, or every crop failed. Once a figure has been described, its content is
        in the distilled output and the page is no longer a gap.
        """
        return [
            p.page_index
            for p in self.pages
            if p.method == DistillMethod.NATIVE_TEXT and p.image_count > 0 and not p.figures
        ]

    def pages_with_described_figures(self) -> list[int]:
        return [p.page_index for p in self.pages if p.figures]

    @property
    def figure_count(self) -> int:
        return sum(len(p.figures) for p in self.pages)
