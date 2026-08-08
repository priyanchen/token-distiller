class _FakeResponse:
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens


class _FakeMessages:
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens
        self.calls = []

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.input_tokens)


class _FakeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = _FakeMessages(input_tokens=42)


def test_returns_input_tokens_from_the_api(monkeypatch):
    from token_distiller import exact_tokens

    monkeypatch.setenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    assert exact_tokens.count_tokens_exact("hello world") == 42


def test_passes_model_and_text_through_as_a_single_user_message(monkeypatch):
    from token_distiller import exact_tokens

    monkeypatch.setenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    fake_client_holder = {}

    def make_client(api_key=None):
        client = _FakeClient(api_key=api_key)
        fake_client_holder["client"] = client
        return client

    monkeypatch.setattr(anthropic, "Anthropic", make_client)

    exact_tokens.count_tokens_exact("hi there", model="claude-opus-5")

    call = fake_client_holder["client"].messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["messages"] == [{"role": "user", "content": "hi there"}]


def test_scoped_key_is_used_over_the_global_one(monkeypatch):
    from token_distiller import exact_tokens

    monkeypatch.setenv("ANTHROPIC_API_KEY", "global-key")
    monkeypatch.setenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", "scoped-key")
    import anthropic

    fake_client_holder = {}

    def make_client(api_key=None):
        client = _FakeClient(api_key=api_key)
        fake_client_holder["client"] = client
        return client

    monkeypatch.setattr(anthropic, "Anthropic", make_client)

    exact_tokens.count_tokens_exact("hi")
    assert fake_client_holder["client"].api_key == "scoped-key"


def test_without_a_key_raises_naming_both_variables(monkeypatch):
    from token_distiller import exact_tokens

    monkeypatch.delenv("TOKEN_DISTILLER_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        exact_tokens.count_tokens_exact("hi")
    except exact_tokens.ExactCountUnavailable as exc:
        assert "TOKEN_DISTILLER_ANTHROPIC_API_KEY" in str(exc)
        assert "ANTHROPIC_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ExactCountUnavailable")
