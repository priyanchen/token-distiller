"""Central place for every tunable default. Nothing else in the package should hardcode a constant."""

import os
from pathlib import Path

HOME = Path(os.environ.get("CONTEXT_DISTILL_HOME", Path.home() / ".context-distill"))
DB_PATH = HOME / "distill.db"

# --- M1: distillation thresholds ---
MIN_NATIVE_TEXT_CHARS = 20  # below this, treat a PDF page as image-only
OCR_CONF_THRESHOLD = 70.0  # mean Tesseract word confidence (0-100)
OCR_MIN_WORD_COUNT = 5
RENDER_DPI = 200  # rasterization DPI for text-less PDF pages

VISION_MODEL = os.environ.get("CONTEXT_DISTILL_VISION_MODEL", "claude-sonnet-5")
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
VOYAGE_API_KEY_ENV = "VOYAGE_API_KEY"

# --- M3: retrieval ---
CHUNK_TOKEN_SIZE = 500
CHUNK_TOKEN_OVERLAP = 50
DEFAULT_TOP_K = 5

# --- M4: activity mode ---
ACTIVITY_WINDOW_SIZE = 10


def ensure_home() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME
