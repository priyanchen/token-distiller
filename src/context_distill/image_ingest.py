from pathlib import Path

import pillow_heif
from PIL import Image

from context_distill.config import IMAGE_EXTENSIONS

pillow_heif.register_heif_opener()


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")
