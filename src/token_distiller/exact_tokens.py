from token_distiller.config import ANTHROPIC_API_KEY_ENV, SCOPED_API_KEY_ENV, VISION_MODEL
from token_distiller.vision_fallback import resolve_api_key


class ExactCountUnavailable(Exception):
    pass


def count_tokens_exact(text: str, model: str = VISION_MODEL) -> int:
    api_key = resolve_api_key()
    if not api_key:
        raise ExactCountUnavailable(
            f"no API key: set {SCOPED_API_KEY_ENV} (preferred) or {ANTHROPIC_API_KEY_ENV}"
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens
