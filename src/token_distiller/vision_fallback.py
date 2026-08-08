import base64
import io
import os

from PIL import Image

from token_distiller.config import (
    ANTHROPIC_API_KEY_ENV,
    SCOPED_API_KEY_ENV,
    VISION_MODEL,
    VISION_PROMPT,
)


class VisionUnavailable(Exception):
    pass


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def resolve_api_key() -> str | None:
    """Scoped variable wins. Exporting ANTHROPIC_API_KEY globally can change how the host
    agent authenticates, so a tool-specific name lets this feature be enabled in isolation."""
    return os.environ.get(SCOPED_API_KEY_ENV) or os.environ.get(ANTHROPIC_API_KEY_ENV)


def describe_image(image: Image.Image, prompt: str = VISION_PROMPT) -> str:
    api_key = resolve_api_key()
    if not api_key:
        raise VisionUnavailable(
            f"no API key: set {SCOPED_API_KEY_ENV} (preferred) or {ANTHROPIC_API_KEY_ENV}"
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _image_to_base64_png(image),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")
