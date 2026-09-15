import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from token_yield.foundry_count import (
    CountProtocolError,
    FoundryCountError,
    FoundryInputTokenCounter,
    TiktokenDependencyError,
    plain_text_token_lower_bound,
)


class FakePost:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, dict(headers), body, timeout))
        return self.result


def test_counts_exact_dispatch_components_and_records_audit_fields():
    raw = b'{"input_tokens":73,"id":"count_1","detail":{"cached":0}}'
    post = FakePost((200, raw, {"x-request-id": "request_1"}))
    counter = FoundryInputTokenCounter(
        "https://example.services.ai.azure.com/openai/v1/",
        model="gpt-5-mini",
        token_provider=lambda: "secret-token",
        http_post=post,
    )
    tools = [{"type": "function", "name": "lookup", "parameters": {}}]
    result = counter.count(
        [{"role": "user", "content": "hello"}],
        instructions="Be brief.",
        tools=tools,
        tool_choice="auto",
        reasoning={"effort": "minimal"},
        text={"verbosity": "low"},
        max_output_tokens=512,
    )

    url, headers, body, timeout = post.requests[0]
    assert url == (
        "https://example.services.ai.azure.com/openai/v1/"
        "responses/input_tokens"
    )
    assert headers["Authorization"].startswith("Bearer ")
    assert timeout == 120
    assert json.loads(body) == {
        "model": "gpt-5-mini",
        "input": [{"role": "user", "content": "hello"}],
        "instructions": "Be brief.",
        "tools": tools,
        "tool_choice": "auto",
        "reasoning": {"effort": "minimal"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 512,
    }
    assert result.input_tokens == 73
    assert result.model == "gpt-5-mini"
    assert result.endpoint == url
    assert result.response_id == "count_1"
    assert result.request_id == "request_1"
    assert result.request_sha256 == hashlib.sha256(body).hexdigest()
    assert result.response_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.raw_fields == json.loads(raw)
    assert "secret-token" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.input_tokens = 1
    with pytest.raises(TypeError):
        result.raw_fields["input_tokens"] = 1


def test_accepts_full_count_path_and_omits_unsupplied_optional_fields():
    post = FakePost((200, b'{"input_tokens":0}'))
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1/responses/input_tokens/",
        token_provider=lambda: "token",
        http_post=post,
    )
    assert counter("plain").input_tokens == 0
    assert json.loads(post.requests[0][2]) == {
        "model": "gpt-5-mini",
        "input": "plain",
    }


def test_counts_an_exact_continuation_payload():
    post = FakePost((200, b'{"input_tokens":9}'))
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1",
        token_provider=lambda: "token",
        http_post=post,
    )
    payload = {
        "model": "gpt-5-mini",
        "previous_response_id": "resp_1",
        "input": [{"type": "function_call_output", "output": "{}"}],
    }
    assert counter.count_payload(payload).input_tokens == 9
    assert json.loads(post.requests[0][2]) == payload


def test_dispatch_projection_removes_only_output_configuration():
    post = FakePost((200, b'{"input_tokens":9}'))
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1",
        token_provider=lambda: "token",
        http_post=post,
    )
    counter.count_dispatch_payload({
        "model": "gpt-5-mini",
        "input": "hello",
        "tools": [{"type": "function", "name": "fetch"}],
        "tool_choice": "auto",
        "max_output_tokens": 512,
        "reasoning": {"effort": "minimal"},
        "text": {"verbosity": "low"},
    })
    assert json.loads(post.requests[0][2]) == {
        "model": "gpt-5-mini",
        "input": "hello",
        "tools": [{"type": "function", "name": "fetch"}],
        "tool_choice": "auto",
    }


@pytest.mark.parametrize("status", [199, 300, 401, 500])
def test_non_2xx_is_an_explicit_failure(status):
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1",
        token_provider=lambda: "token",
        http_post=FakePost((status, b'{"input_tokens":1}')),
    )
    with pytest.raises(FoundryCountError, match=f"HTTP {status}"):
        counter.count("hello")


def test_non_2xx_includes_bounded_provider_error_detail():
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1",
        token_provider=lambda: "token",
        http_post=FakePost((
            400,
            b'{"error":{"type":"invalid_request_error",'
            b'"param":"input","message":"must not be empty"}}',
        )),
    )
    with pytest.raises(
        FoundryCountError, match="invalid_request_error.*must not be empty"
    ):
        counter.count("")


@pytest.mark.parametrize(
    "body,match",
    [
        (b"not-json", "invalid JSON"),
        (b"[]", "must be an object"),
        (b"{}", "non-negative integer"),
        (b'{"input_tokens":-1}', "non-negative integer"),
        (b'{"input_tokens":true}', "non-negative integer"),
        (b'{"input_tokens":1.5}', "non-negative integer"),
    ],
)
def test_rejects_malformed_or_absent_count(body, match):
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1",
        token_provider=lambda: "token",
        http_post=FakePost((200, body)),
    )
    with pytest.raises(CountProtocolError, match=match):
        counter.count("hello")


def test_rejects_empty_injected_token_before_http():
    post = FakePost((200, b'{"input_tokens":1}'))
    counter = FoundryInputTokenCounter(
        "https://example/openai/v1",
        token_provider=lambda: " ",
        http_post=post,
    )
    with pytest.raises(Exception, match="empty access token"):
        counter.count("hello")
    assert post.requests == []


def test_plain_text_lower_bound_uses_o200k_base(monkeypatch):
    requested = []

    class Encoding:
        def encode(self, text):
            assert text == "hello"
            return [1, 2]

    class Tiktoken:
        def get_encoding(self, name):
            requested.append(name)
            return Encoding()

    monkeypatch.setattr(
        "token_yield.foundry_count.importlib.import_module",
        lambda name: Tiktoken(),
    )
    assert plain_text_token_lower_bound("hello") == 2
    assert requested == ["o200k_base"]


def test_plain_text_lower_bound_has_clear_lazy_dependency_error(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(
        "token_yield.foundry_count.importlib.import_module", missing
    )
    with pytest.raises(TiktokenDependencyError, match="tiktoken"):
        plain_text_token_lower_bound("hello")
