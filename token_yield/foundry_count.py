"""Exact pre-dispatch input-token counting for Microsoft Foundry Responses."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from token_yield.foundry_dispatch import (
    DEFAULT_RESOURCE,
    AuthenticationError,
    acquire_entra_token,
)


class FoundryCountError(RuntimeError):
    """The provider count request could not be completed."""


class CountProtocolError(FoundryCountError):
    """The provider returned an invalid count response."""


class TiktokenDependencyError(ImportError):
    """The optional local tokenizer is not installed."""


@dataclass(frozen=True)
class InputTokenCount:
    """Auditable result from Foundry's exact input-token count endpoint."""

    input_tokens: int
    endpoint: str
    model: str
    request_sha256: str
    response_sha256: str
    response_id: Optional[str]
    request_id: Optional[str]
    latency_ms: int
    raw_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_fields", MappingProxyType(dict(self.raw_fields)))


def _default_http_post(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> Tuple[int, bytes, Mapping[str, str]]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return (
                int(response.getcode()),
                response.read(),
                dict(response.headers.items()),
            )
    except HTTPError as exc:
        headers = dict(exc.headers.items()) if exc.headers is not None else {}
        return int(exc.code), exc.read(), headers


def _normalize_http_result(
    result: Any,
) -> Tuple[int, bytes, Mapping[str, str]]:
    headers: Any = {}
    if isinstance(result, tuple) and len(result) in (2, 3):
        status, body = result[:2]
        if len(result) == 3:
            headers = result[2]
    elif isinstance(result, Mapping):
        status = result.get("status")
        body = result.get("body")
        headers = result.get("headers", {})
    else:
        status = getattr(result, "status", None)
        if status is None and hasattr(result, "getcode"):
            status = result.getcode()
        body = result.read() if hasattr(result, "read") else getattr(result, "body", None)
        headers = getattr(result, "headers", {})
    if not isinstance(status, int) or isinstance(status, bool):
        raise CountProtocolError("HTTP layer returned no integer status")
    if isinstance(body, str):
        body = body.encode("utf-8")
    if not isinstance(body, (bytes, bytearray)):
        raise CountProtocolError("HTTP layer returned no byte response body")
    if hasattr(headers, "items"):
        headers = dict(headers.items())
    if not isinstance(headers, Mapping):
        raise CountProtocolError("HTTP layer returned invalid response headers")
    return status, bytes(body), {
        str(key).lower(): str(value) for key, value in headers.items()
    }


def _count_url(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty URL")
    value = endpoint.strip().rstrip("/")
    if value.endswith("/responses/input_tokens"):
        return value
    if value.endswith("/openai/v1"):
        return value + "/responses/input_tokens"
    raise ValueError(
        "endpoint must end with /openai/v1 or /responses/input_tokens"
    )


def _optional_identifier(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _error_detail(raw: bytes) -> str:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    error = value.get("error") if isinstance(value, Mapping) else None
    if not isinstance(error, Mapping):
        return ""
    fields = [
        str(error[key]) for key in ("type", "code", "param", "message")
        if error.get(key) not in (None, "")
    ]
    return ": ".join(fields)[:500]


class FoundryInputTokenCounter:
    """Call Foundry's exact Responses input-token count endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "gpt-5-mini",
        resource: str = DEFAULT_RESOURCE,
        command_runner: Callable[..., Any] = subprocess.run,
        token_provider: Optional[Callable[[], str]] = None,
        http_post: Callable[
            [str, Mapping[str, str], bytes, float], Any
        ] = _default_http_post,
        timeout_seconds: float = 120,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = _count_url(endpoint)
        self.model = model
        self.resource = resource
        self.command_runner = command_runner
        self.token_provider = token_provider
        self.http_post = http_post
        self.timeout_seconds = timeout_seconds

    def _token(self) -> str:
        token = (
            self.token_provider()
            if self.token_provider is not None
            else acquire_entra_token(self.command_runner, self.resource)
        )
        if not isinstance(token, str) or not token.strip():
            raise AuthenticationError("token provider returned an empty access token")
        return token.strip()

    def __call__(self, input: Any, **kwargs: Any) -> InputTokenCount:
        return self.count(input, **kwargs)

    def count(
        self,
        input: Any,
        *,
        instructions: Optional[str] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        reasoning: Optional[Any] = None,
        text: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
    ) -> InputTokenCount:
        """Count the exact payload that will be sent to ``/responses``.

        ``reasoning`` and ``text`` are omitted unless supplied, allowing callers
        to include them only for count-API versions which accept those fields.
        """

        payload: Dict[str, Any] = {"model": self.model, "input": input}
        if instructions is not None:
            payload["instructions"] = instructions
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if text is not None:
            payload["text"] = text
        if max_output_tokens is not None:
            if (
                not isinstance(max_output_tokens, int)
                or isinstance(max_output_tokens, bool)
                or max_output_tokens < 1
            ):
                raise ValueError("max_output_tokens must be a positive integer")
            payload["max_output_tokens"] = max_output_tokens
        return self.count_payload(payload)

    def count_payload(self, payload: Mapping[str, Any]) -> InputTokenCount:
        """Count an exact Responses request object before dispatch."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if payload.get("model") != self.model:
            raise ValueError("payload model must match the configured model")
        try:
            body = json.dumps(
                dict(payload), separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("count payload must be JSON serializable") from exc

        request_sha256 = hashlib.sha256(body).hexdigest()
        headers = {
            "Authorization": "Bearer " + self._token(),
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            status, raw, response_headers = _normalize_http_result(
                self.http_post(
                    self.endpoint, headers, body, self.timeout_seconds
                )
            )
        except FoundryCountError:
            raise
        except Exception as exc:
            raise FoundryCountError("input-token count request failed") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        if status < 200 or status >= 300:
            detail = _error_detail(raw)
            raise FoundryCountError(
                f"input-token count API returned HTTP {status}"
                + (f": {detail}" if detail else "")
            )
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CountProtocolError(
                "input-token count API returned invalid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise CountProtocolError(
                "input-token count API response must be an object"
            )
        count = parsed.get("input_tokens")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise CountProtocolError(
                "input-token count response is missing non-negative "
                "integer input_tokens"
            )
        response_id = _optional_identifier(parsed.get("id"))
        request_id = _optional_identifier(parsed.get("request_id"))
        if request_id is None:
            for name in (
                "x-request-id",
                "x-ms-request-id",
                "request-id",
                "apim-request-id",
            ):
                request_id = _optional_identifier(response_headers.get(name))
                if request_id is not None:
                    break
        return InputTokenCount(
            input_tokens=count,
            endpoint=self.endpoint,
            model=self.model,
            request_sha256=request_sha256,
            response_sha256=hashlib.sha256(raw).hexdigest(),
            response_id=response_id,
            request_id=request_id,
            latency_ms=latency_ms,
            raw_fields=parsed,
        )

    def count_dispatch_payload(
        self, payload: Mapping[str, Any]
    ) -> InputTokenCount:
        """Count the provider-supported, token-bearing dispatch projection.

        The input-token endpoint rejects output-only Responses fields even
        though they are valid on the generation endpoint.
        """

        excluded = {"max_output_tokens", "reasoning", "text"}
        projected = {
            key: value for key, value in payload.items()
            if key not in excluded
        }
        return self.count_payload(projected)


def plain_text_token_lower_bound(text: str) -> int:
    """Count plain text with ``o200k_base``; excludes all tool/schema framing."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ImportError as exc:
        raise TiktokenDependencyError(
            "plain-text token counting requires the 'tiktoken' package"
        ) from exc
    return len(tiktoken.get_encoding("o200k_base").encode(text))


# Concise aliases for callers that prefer function-oriented naming.
FoundryCounter = FoundryInputTokenCounter
CountResult = InputTokenCount
local_plain_text_token_lower_bound = plain_text_token_lower_bound
