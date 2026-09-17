import hashlib
import json
from types import SimpleNamespace

import pytest

from token_yield.foundry_dispatch import (
    AuthenticationError,
    FoundryDispatcher,
    HttpRequestSpec,
    ResponseProtocolError,
    RunawayToolCallsError,
    ToolPolicyError,
    acquire_entra_token,
)


def response(response_id, output, *, usage=None, status="completed"):
    return {
        "id": response_id,
        "status": status,
        "model": "gpt-5-mini-2026-01-01",
        "output": output,
        "usage": usage
        or {
            "input_tokens": 10,
            "output_tokens": 6,
            "total_tokens": 16,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }


class FakePost:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, dict(headers), json.loads(body), timeout))
        return 200, json.dumps(self.responses.pop(0)).encode()


def test_acquires_token_with_azure_cli_without_persisting_it():
    invocations = []

    def runner(command, **kwargs):
        invocations.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="secret-token\n", stderr="")

    assert acquire_entra_token(
        runner, "https://resource.example", environment={}
    ) == "secret-token"
    command, kwargs = invocations[0]
    assert command == [
        "az",
        "account",
        "get-access-token",
        "--resource",
        "https://resource.example",
        "--query",
        "accessToken",
        "--output",
        "tsv",
    ]
    assert kwargs["capture_output"] is True


def test_acquires_token_from_entra_environment_without_cli():
    calls = []

    def oauth(url, body, timeout):
        calls.append((url, body, timeout))
        return b'{"access_token":"secret-token"}'

    token = acquire_entra_token(
        resource="https://resource.example",
        environment={
            "AZURE_TENANT_ID": "tenant",
            "AZURE_CLIENT_ID": "client",
            "AZURE_CLIENT_SECRET": "secret",
        },
        oauth_post=oauth,
    )
    assert token == "secret-token"
    assert calls[0][0].endswith("/tenant/oauth2/v2.0/token")
    assert b"scope=https%3A%2F%2Fresource.example%2F.default" in calls[0][1]


def test_scoped_auth_uses_subscription_and_verifies_returned_tenant():
    invocations = []

    def runner(command, **kwargs):
        invocations.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "accessToken": "scoped-secret", "tenant": "tenant", "subscription": "subscription",
        }))

    assert acquire_entra_token(
        runner, subscription_id="subscription", tenant_id="tenant",
        environment={"AZURE_TENANT_ID": "other", "AZURE_CLIENT_ID": "ambient",
                     "AZURE_CLIENT_SECRET": "must-not-use"},
        oauth_post=lambda *args: pytest.fail("scoped authentication must use the selected CLI account"),
    ) == "scoped-secret"
    assert invocations == [[
        "az", "account", "get-access-token", "--resource",
        "https://cognitiveservices.azure.com", "--subscription", "subscription",
        "--output", "json",
    ]]


@pytest.mark.parametrize("metadata", [
    {}, None, [], {"tenant": "other", "subscription": "subscription", "accessToken": "secret"},
    {"tenant": "tenant", "subscription": "other", "accessToken": "secret"},
    {"tenant": "tenant", "subscription": "subscription"},
    {"tenant": "tenant", "subscription": "subscription", "accessToken": ""},
])
def test_scoped_auth_rejects_missing_or_wrong_identity(metadata):
    with pytest.raises(AuthenticationError):
        acquire_entra_token(
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(metadata)),
            subscription_id="subscription", tenant_id="tenant",
        )


def test_scoped_auth_rejects_invalid_json():
    with pytest.raises(AuthenticationError, match="invalid authentication metadata"):
        acquire_entra_token(
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not JSON"),
            subscription_id="subscription", tenant_id="tenant",
        )


@pytest.mark.parametrize("scope", [
    {"subscription_id": "subscription"}, {"tenant_id": "tenant"},
    {"subscription_id": "", "tenant_id": "tenant"},
])
def test_scoped_auth_never_falls_back_on_incomplete_scope(scope):
    with pytest.raises(ValueError, match="requires subscription_id and tenant_id"):
        acquire_entra_token(
            lambda *args, **kwargs: pytest.fail("must not invoke CLI with incomplete scope"),
            **scope,
        )


def test_dispatches_tool_loop_records_calls_fetch_and_usage_by_target():
    first = response(
        "resp_1",
        [
            {
                "type": "function_call",
                "name": "fetch_public_api",
                "call_id": "call_1",
                "arguments": '{"manifest_id":"prices"}',
            }
        ],
    )
    second = response(
        "resp_2",
        [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
        usage={
            "input_tokens": 8,
            "output_tokens": 4,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    )
    post = FakePost([first, second])
    spec = HttpRequestSpec("GET", "https://public.example/data")
    fetched = []

    def fetch_executor(received):
        fetched.append(received)
        return 201, b'{"value":42}'

    dispatcher = FoundryDispatcher(
        "https://generic.services.ai.azure.com/openai/v1",
        {"prices": spec},
        fetch_executor,
        token_provider=lambda: "secret-token",
        http_post=post,
    )
    result = dispatcher.dispatch("Find it", target="target-a")

    assert result.output == "done"
    assert result.response_id == "resp_2"
    assert len(result.response_calls) == 2
    assert result.usage.input_tokens == 18
    assert result.usage.output_tokens == 10
    assert result.usage.reasoning_tokens == 3
    assert result.usage.cached_tokens == 4
    assert fetched == [spec]
    assert result.fetches[0].status == 201
    assert result.fetches[0].response_bytes == 12
    assert result.fetches[0].response_sha256 == hashlib.sha256(
        b'{"value":42}'
    ).hexdigest()

    first_request, second_request = post.requests
    assert first_request[0].endswith("/openai/v1/responses")
    assert first_request[1]["Authorization"] == "Bearer secret-token"
    assert first_request[2]["model"] == "gpt-5-mini"
    assert first_request[2]["max_output_tokens"] == 512
    assert first_request[2]["reasoning"] == {"effort": "minimal"}
    assert first_request[2]["text"] == {"verbosity": "low"}
    assert second_request[2]["previous_response_id"] == "resp_1"
    assert second_request[2]["tool_choice"] == "none"
    tool_output = second_request[2]["input"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call_1"
    assert json.loads(tool_output["output"]) == {
        "status": 201,
        "body": '{"value":42}',
    }
    snapshot = dispatcher.ledger.snapshot()
    assert snapshot["response_calls"] == 2
    assert snapshot["targets"]["target-a"]["total_tokens"] == 28
    assert snapshot["total"]["cached_tokens"] == 4
    assert len(result.response_calls[0].request_sha256) == 64
    assert len(result.response_calls[0].response_sha256) == 64


def test_dispatch_without_manifests_omits_empty_tool_schema():
    post = FakePost([response(
        "r", [{"type": "message", "content": [
            {"type": "output_text", "text": "done"}
        ]}]
    )])
    FoundryDispatcher(
        "https://generic.example/v1",
        token_provider=lambda: "token",
        http_post=post,
    ).dispatch("prompt")
    assert "tools" not in post.requests[0][2]


def test_initial_payload_is_the_payload_sent_first():
    post = FakePost([response(
        "r", [{"type": "message", "content": [
            {"type": "output_text", "text": "done"}
        ]}]
    )])
    dispatcher = FoundryDispatcher(
        "https://generic.example/v1",
        token_provider=lambda: "token",
        http_post=post,
        reasoning_effort="medium",
        text_verbosity="medium",
    )
    expected = dispatcher.initial_payload("prompt")
    dispatcher.dispatch("prompt")
    assert post.requests[0][2] == expected


def test_pre_dispatch_hook_receives_every_exact_request_payload():
    post = FakePost([
        response("r1", [{
            "type": "function_call",
            "name": "fetch_public_api",
            "call_id": "call_1",
            "arguments": '{"manifest_id":"known"}',
        }]),
        response("r2", [{
            "type": "message",
            "content": [{"type": "output_text", "text": "done"}],
        }]),
    ])
    observed = []
    FoundryDispatcher(
        "https://generic.example/v1",
        {"known": HttpRequestSpec("GET", "https://public.example")},
        lambda spec: (200, b"{}"),
        token_provider=lambda: "token",
        http_post=post,
        require_tool=True,
        max_response_calls=2,
        pre_dispatch_hook=lambda payload, index: observed.append(
            (index, dict(payload))
        ),
    ).dispatch("prompt")
    assert [index for index, _ in observed] == [0, 1]
    assert observed[0][1] == post.requests[0][2]
    assert observed[1][1] == post.requests[1][2]


def test_required_tool_cannot_be_silently_skipped():
    post = FakePost([response(
        "r", [{"type": "message", "content": [
            {"type": "output_text", "text": "done"}
        ]}]
    )])
    dispatcher = FoundryDispatcher(
        "https://generic.example/v1",
        {"known": HttpRequestSpec("GET", "https://public.example")},
        lambda spec: (200, b"ok"),
        token_provider=lambda: "token",
        http_post=post,
        require_tool=True,
    )
    with pytest.raises(ToolPolicyError, match="required"):
        dispatcher.dispatch("prompt")


@pytest.mark.parametrize(
    "item,error",
    [
        (
            {
                "type": "function_call",
                "name": "other_tool",
                "call_id": "c",
                "arguments": "{}",
            },
            "unknown function tool",
        ),
        (
            {
                "type": "function_call",
                "name": "fetch_public_api",
                "call_id": "c",
                "arguments": '{"manifest_id":"unknown"}',
            },
            "unknown manifest",
        ),
    ],
)
def test_rejects_unknown_tools_and_manifests(item, error):
    dispatcher = FoundryDispatcher(
        "https://generic.example/v1",
        {},
        lambda spec: (200, b""),
        token_provider=lambda: "token",
        http_post=FakePost([response("r", [item])]),
    )
    with pytest.raises(ToolPolicyError, match=error):
        dispatcher.dispatch("prompt")


@pytest.mark.parametrize(
    "value,match",
    [
        (response("r", [], status="incomplete"), "incomplete status"),
        (
            {
                "id": "r",
                "status": "completed",
                "model": "gpt-5-mini",
                "output": [],
            },
            "missing usage",
        ),
    ],
)
def test_rejects_incomplete_or_usage_less_responses(value, match):
    dispatcher = FoundryDispatcher(
        "https://generic.example/v1",
        token_provider=lambda: "token",
        http_post=FakePost([value]),
    )
    with pytest.raises(ResponseProtocolError, match=match):
        dispatcher.dispatch("prompt")


def test_rejects_runaway_function_calls():
    call = {
        "type": "function_call",
        "name": "fetch_public_api",
        "call_id": "one",
        "arguments": '{"manifest_id":"known"}',
    }
    dispatcher = FoundryDispatcher(
        "https://generic.example/v1",
        {"known": HttpRequestSpec("GET", "https://public.example")},
        lambda spec: (200, b"ok"),
        token_provider=lambda: "token",
        http_post=FakePost([response("r1", [call])]),
        max_response_calls=1,
    )
    with pytest.raises(RunawayToolCallsError, match="limit"):
        dispatcher.dispatch("prompt")


def test_rejects_too_many_tools_in_one_response_before_fetching():
    calls = [
        {
            "type": "function_call",
            "name": "fetch_public_api",
            "call_id": "one",
            "arguments": '{"manifest_id":"known"}',
        },
        {
            "type": "function_call",
            "name": "fetch_public_api",
            "call_id": "two",
            "arguments": '{"manifest_id":"known"}',
        },
    ]
    fetched = []
    dispatcher = FoundryDispatcher(
        "https://generic.example/v1/responses",
        {"known": HttpRequestSpec("GET", "https://public.example")},
        lambda spec: fetched.append(spec) or (200, b"ok"),
        token_provider=lambda: "token",
        http_post=FakePost([response("r1", calls)]),
        max_tool_calls=1,
    )
    with pytest.raises(RunawayToolCallsError, match="tool call limit"):
        dispatcher.dispatch("prompt")
    assert fetched == []
