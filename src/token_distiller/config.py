"""Central place for every tunable default. Nothing else in the package should hardcode a constant."""

import os
from pathlib import Path

HOME = Path(os.environ.get("TOKEN_DISTILLER_HOME", Path.home() / ".token-distiller"))
DB_PATH = HOME / "distill.db"

# --- M1: distillation thresholds ---
MIN_NATIVE_TEXT_CHARS = 20  # below this, treat a PDF page as image-only
OCR_CONF_THRESHOLD = 70.0  # mean Tesseract word confidence (0-100)
OCR_MIN_WORD_COUNT = 5

# A raw OCR pass this weak earns a second attempt on a preprocessed copy. Set high enough
# that anything already reading cleanly skips the retry entirely.
OCR_RETRY_CONF_THRESHOLD = 80.0
OCR_RETRY_MIN_WORDS = 3
# Tesseract wants roughly 300-DPI text; crops smaller than this get scaled up first.
OCR_MIN_UPSCALE_PX = 600
OCR_UPSCALE_FACTOR = 2

# Tesseract's own default is English, and it does not detect script automatically -- a
# Hebrew or Arabic page OCR'd as "eng" returns near-nothing, which then surfaces as
# "could not be read" rather than "wrong language". Accepts Tesseract's multi-language
# form ("eng+heb") to cover a mixed-language corpus in one pass. `tesseract --list-langs`
# shows what is installed locally.
OCR_LANG = os.environ.get("TOKEN_DISTILLER_OCR_LANG", "eng")

# pdfplumber returns glyphs in visual order, so a right-to-left script comes out with every
# word reversed -- silently, with no confidence score to flag it. Poppler's pdftotext
# implements the Unicode bidirectional algorithm and returns logical order, so it is used
# for pages that actually contain RTL text. Poppler is already required (pdf2image).
RTL_REORDER_ENABLED = os.environ.get("TOKEN_DISTILLER_RTL_REORDER", "1") != "0"
PDFTOTEXT_TIMEOUT_S = 120

RENDER_DPI = 200  # rasterization DPI for text-less PDF pages

VISION_MODEL = os.environ.get("TOKEN_DISTILLER_VISION_MODEL", "claude-sonnet-5")
VISION_PROMPT = (
    "Transcribe all readable text from this image verbatim. If it contains a "
    "chart, diagram, or figure with no transcribable text, describe its content "
    "and any labeled data concisely instead. Do not add commentary."
)

# Anthropic's own published rough-estimate guidance: ~4 chars per token for English text.
CHARS_PER_TOKEN = 4.0
# Anthropic's published image-token approximation: tokens ~= (width_px * height_px) / 750
IMAGE_TOKEN_DIVISOR = 750.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
DISTILLABLE_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
# Checked first, so this tool can be given a key without exporting ANTHROPIC_API_KEY
# globally. That matters because the host agent may also read ANTHROPIC_API_KEY and
# switch from subscription auth to per-token API billing if it finds one.
SCOPED_API_KEY_ENV = "TOKEN_DISTILLER_ANTHROPIC_API_KEY"
VOYAGE_API_KEY_ENV = "VOYAGE_API_KEY"

# --- M3: retrieval ---
CHUNK_TOKEN_SIZE = 500
CHUNK_TOKEN_OVERLAP = 50
DEFAULT_TOP_K = 5

# --- M4: activity mode ---
ACTIVITY_WINDOW_SIZE = 10

# --- Bash output compression ---
BASH_MAX_LINES = int(os.environ.get("TOKEN_DISTILLER_BASH_MAX_LINES", "40"))
BASH_TAIL_LINES = 5

# --- M5: cache, boilerplate, large-document handling ---
CACHE_ENABLED = os.environ.get("TOKEN_DISTILLER_CACHE", "1") != "0"
# Re-reading an unchanged file in the same session returns a pointer instead of the
# full text. The text is never discarded — `distill expand <handle>` returns it.
REREAD_COLLAPSE_ENABLED = os.environ.get("TOKEN_DISTILLER_REREAD_COLLAPSE", "1") != "0"

# A line must appear on at least this fraction of pages to count as boilerplate.
# Deliberately high: at 0.8 a 25-page deck's copyright footer (25/25) collapses while a
# structural marker like "Example:" (15/25) survives.
BOILERPLATE_PAGE_FRACTION = 0.8
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MAX_LINE_CHARS = 120
BOILERPLATE_ENABLED = os.environ.get("TOKEN_DISTILLER_BOILERPLATE", "1") != "0"

# Beyond this, the hook returns an outline + head + expand handle rather than the whole
# document. Deferred, not dropped: the full text stays retrievable via `distill expand`.
LARGE_DOC_TOKEN_THRESHOLD = int(os.environ.get("TOKEN_DISTILLER_LARGE_DOC_TOKENS", "8000"))
LARGE_DOC_HEAD_TOKENS = 1500
# A second, independent bound on the same loop: a scanned page can be nearly empty and
# still cost full OCR time (measured ~2s/page on a real document), so a token-only bound
# never triggers on a long, sparse scanned document while wall-clock time keeps climbing.
# 100 pages x ~2s/page is comfortably under the hook's 300s timeout with margin to spare,
# and dense documents never reach it -- they already cross LARGE_DOC_TOKEN_THRESHOLD well
# before page 100 (measured: page 11-21 on two real books).
LARGE_DOC_MAX_PAGES = int(os.environ.get("TOKEN_DISTILLER_LARGE_DOC_MAX_PAGES", "100"))

# --- M6: reading figures embedded in otherwise-text pages ---
# Native text extraction cannot see a diagram, so each embedded figure is cropped out and
# put through the same OCR -> vision chain used for scanned pages. On by default: a
# flagged-but-unread diagram was the one real gap in the "nothing is discarded" promise.
DESCRIBE_FIGURES = os.environ.get("TOKEN_DISTILLER_DESCRIBE_FIGURES", "1") != "0"
# Ignore hairline rules, borders and background strips — describing a 2pt spacer spends a
# vision call to learn nothing.
FIGURE_MIN_SIDE_PT = float(os.environ.get("TOKEN_DISTILLER_FIGURE_MIN_SIDE_PT", "48"))
FIGURE_RENDER_DPI = 200
FIGURE_PROMPT = (
    "This is a figure cropped from a document page; the surrounding body text is already "
    "captured separately. Transcribe every label, axis, number, and caption verbatim, then "
    "state in one or two sentences what the figure shows. Do not describe visual styling."
)

# Anthropic downscales images whose long edge exceeds this before billing.
IMAGE_MAX_EDGE_PX = 1568
# Rasterization DPI a host uses when turning a PDF page into an image for the model.
HOST_PDF_RENDER_DPI = 150


def ensure_home() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME
