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
        if self.distilled_tokens_est == 0:
            return 0.0
        return self.raw_tokens_est / self.distilled_tokens_est

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def rendered_text(self) -> str:
        """Text as a consumer should see it: collapsed boilerplate is restated once up
        front rather than silently dropped."""
        if not self.boilerplate:
            return self.text
        from context_distill.boilerplate import render_manifest

        return f"{render_manifest(self.boilerplate)}\n\n{self.text}"

    @property
    def warnings(self) -> list[str]:
        return [w for p in self.pages for w in p.warnings]

    def method_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.pages:
            counts[p.method.value] = counts.get(p.method.value, 0) + 1
        return counts
