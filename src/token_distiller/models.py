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
        return "\n\n".join(p.text for p in self.pages)

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
        """Page indices where native-text extraction ran -- so nothing ever rasterized
        or OCR'd the page -- but the page also contains one or more embedded images.
        Their content (a diagram, a figure, an illustration) is not lost or deferred
        like large-document text; it was simply never read in the first place. This is
        the one gap in the "nothing is discarded" guarantee: that guarantee covers text
        that gets shortened, not image content on an otherwise-native-text page."""
        return [
            p.page_index
            for p in self.pages
            if p.method == DistillMethod.NATIVE_TEXT and p.image_count > 0
        ]
