import base64
import io
import os

from PIL import Image

from context_distill.config import ANTHROPIC_API_KEY_ENV, VISION_MODEL, VISION_PROMPT


class VisionUnavailable(Exception):
    pass


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def describe_image(image: Image.Image) -> str:
    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
    if not api_key:
        raise VisionUnavailable(f"{ANTHROPIC_API_KEY_ENV} not set")

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
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")
