"""Small, auditable adapter for the Microsoft Foundry Responses API."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_RESOURCE = "https://cognitiveservices.azure.com"
TOOL_NAME = "fetch_public_api"


class FoundryDispatchError(RuntimeError):
    """Base error for a dispatch that cannot be safely completed."""


class AuthenticationError(FoundryDispatchError):
    pass


class ResponseProtocolError(FoundryDispatchError):
    pass


class ToolPolicyError(FoundryDispatchError):
    pass


class RunawayToolCallsError(FoundryDispatchError):
    pass


@dataclass(frozen=True)
class HttpRequestSpec:
    """An immutable, pre-approved HTTP request."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", self.method.upper())
        object.__setattr__(
            self, "headers", MappingProxyType(dict(self.headers))
        )
        if self.body is not None:
            object.__setattr__(self, "body", bytes(self.body))


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.cached_tokens + other.cached_tokens,
            self.total_tokens + other.total_tokens,
        )


ZERO_USAGE = TokenUsage(0, 0, 0, 0, 0)


@dataclass(frozen=True)
class ResponseCall:
    target: str
    response_id: str
    status: str
    model: str
    elapsed_ms: int
    usage: TokenUsage
    request_sha256: str
    response_sha256: str


@dataclass(frozen=True)
class FetchRecord:
    manifest_id: str
    status: int
    latency_ms: int
    response_bytes: int
    response_sha256: str


@dataclass(frozen=True)
class DispatchResult:
    output: str
    response_id: str
    status: str
    model: str
    elapsed_ms: int
    usage: TokenUsage
    response_calls: Tuple[ResponseCall, ...]
    fetches: Tuple[FetchRecord, ...]


class UsageLedger:
    """Thread-safe usage totals, retained per logical target and overall."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: Dict[str, List[ResponseCall]] = {}

    def record(self, call: ResponseCall) -> None:
        with self._lock:
            self._calls.setdefault(call.target, []).append(call)

    def calls(self, target: Optional[str] = None) -> Tuple[ResponseCall, ...]:
        with self._lock:
            if target is not None:
                return tuple(self._calls.get(target, ()))
            return tuple(
                call for calls in self._calls.values() for call in calls
            )

    def totals(self) -> Dict[str, TokenUsage]:
        with self._lock:
            result: Dict[str, TokenUsage] = {}
            overall = ZERO_USAGE
            for target, calls in self._calls.items():
                total = ZERO_USAGE
                for call in calls:
                    total = total + call.usage
                result[target] = total
                overall = overall + total
            result["__all__"] = overall
            return result

    def snapshot(self) -> Dict[str, Any]:
        totals = self.totals()
        return {
            "targets": {
                target: asdict(usage)
                for target, usage in totals.items()
                if target != "__all__"
            },
            "total": asdict(totals["__all__"]),
            "response_calls": len(self.calls()),
        }


def _default_oauth_post(url: str, body: bytes, timeout: float) -> bytes:
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def acquire_entra_token(
    command_runner: Callable[..., Any] = subprocess.run,
    resource: str = DEFAULT_RESOURCE,
    *,
    environment: Optional[Mapping[str, str]] = None,
    oauth_post: Callable[[str, bytes, float], bytes] = _default_oauth_post,
) -> str:
    """Acquire an in-memory bearer token from Entra environment or Azure CLI."""

    env = os.environ if environment is None else environment
    tenant = env.get("AZURE_TENANT_ID")
    client = env.get("AZURE_CLIENT_ID")
    secret = env.get("AZURE_CLIENT_SECRET")
    if tenant and client and secret:
        token_url = (
            f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        )
        body = urlencode({
            "grant_type": "client_credentials",
            "client_id": client,
            "client_secret": secret,
            "scope": resource.rstrip("/") + "/.default",
        }).encode("utf-8")
        try:
            response = json.loads(oauth_post(token_url, body, 30))
            token = (
                response.get("access_token")
                if isinstance(response, dict) else None
            )
            if isinstance(token, str) and token:
                return token
        except (OSError, HTTPError, UnicodeDecodeError, json.JSONDecodeError):
            pass

    executable = (
        "az" if command_runner is not subprocess.run
        else shutil.which("az") or shutil.which("az.cmd") or "az"
    )
    command = [
        executable,
        "account",
        "get-access-token",
        "--resource",
        resource,
        "--query",
        "accessToken",
        "--output",
        "tsv",
    ]
    try:
        completed = command_runner(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuthenticationError("Azure CLI token acquisition failed") from exc
    if getattr(completed, "returncode", 1) != 0:
        raise AuthenticationError("Azure CLI token acquisition failed")
    token = str(getattr(completed, "stdout", "")).strip()
    if not token:
        raise AuthenticationError("Azure CLI returned an empty access token")
    return token


def _default_http_post(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> Tuple[int, bytes]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.getcode()), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()


def _normalize_http_result(result: Any) -> Tuple[int, bytes]:
    if isinstance(result, tuple) and len(result) == 2:
        status, body = result
    elif isinstance(result, Mapping):
        status, body = result.get("status"), result.get("body")
    else:
        status = getattr(result, "status", None)
        if status is None and hasattr(result, "getcode"):
            status = result.getcode()
        body = result.read() if hasattr(result, "read") else getattr(result, "body", None)
    if not isinstance(status, int):
        raise ResponseProtocolError("HTTP layer returned no integer status")
    if isinstance(body, str):
        body = body.encode("utf-8")
    if not isinstance(body, (bytes, bytearray)):
        raise ResponseProtocolError("HTTP layer returned no byte response body")
    return status, bytes(body)


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResponseProtocolError(f"usage is missing non-negative integer {label}")
    return value


def parse_usage(response: Mapping[str, Any]) -> TokenUsage:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ResponseProtocolError("response is missing usage")
    input_tokens = _non_negative_int(usage.get("input_tokens"), "input_tokens")
    output_tokens = _non_negative_int(usage.get("output_tokens"), "output_tokens")
    input_details = usage.get("input_tokens_details", {})
    output_details = usage.get("output_tokens_details", {})
    if not isinstance(input_details, Mapping) or not isinstance(
        output_details, Mapping
    ):
        raise ResponseProtocolError("usage token details must be objects")
    cached = _non_negative_int(input_details.get("cached_tokens", 0), "cached_tokens")
    reasoning = _non_negative_int(
        output_details.get("reasoning_tokens", 0), "reasoning_tokens"
    )
    total = _non_negative_int(
        usage.get("total_tokens", input_tokens + output_tokens), "total_tokens"
    )
    if total != input_tokens + output_tokens:
        raise ResponseProtocolError("usage total_tokens is inconsistent")
    if cached > input_tokens or reasoning > output_tokens:
        raise ResponseProtocolError("usage detail tokens exceed their channel")
    return TokenUsage(input_tokens, output_tokens, reasoning, cached, total)


def _response_text(response: Mapping[str, Any]) -> str:
    top_level = response.get("output_text")
    if isinstance(top_level, str):
        return top_level
    parts: List[str] = []
    output = response.get("output", [])
    if not isinstance(output, list):
        raise ResponseProtocolError("response output must be an array")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            raise ResponseProtocolError("message content must be an array")
        for value in content:
            if isinstance(value, Mapping) and value.get("type") == "output_text":
                text = value.get("text")
                if not isinstance(text, str):
                    raise ResponseProtocolError("output_text is missing text")
                parts.append(text)
    return "".join(parts)


class FoundryDispatcher:
    """Call a fixed Responses endpoint and control its sole function tool."""

    def __init__(
        self,
        endpoint: str,
        manifests: Optional[Mapping[str, HttpRequestSpec]] = None,
        fetch_executor: Optional[Callable[[HttpRequestSpec], Any]] = None,
        *,
        model: str = "gpt-5-mini",
        resource: str = DEFAULT_RESOURCE,
        command_runner: Callable[..., Any] = subprocess.run,
        token_provider: Optional[Callable[[], str]] = None,
        http_post: Callable[[str, Mapping[str, str], bytes, float], Any] = _default_http_post,
        timeout_seconds: float = 120,
        max_response_calls: int = 8,
        max_tool_calls: int = 8,
        max_output_tokens: int = 512,
        reasoning_effort: str = "minimal",
        text_verbosity: str = "low",
        require_tool: bool = False,
        ledger: Optional[UsageLedger] = None,
        pre_dispatch_hook: Optional[
            Callable[[Mapping[str, Any], int], None]
        ] = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty URL")
        if max_response_calls < 1:
            raise ValueError("max_response_calls must be positive")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if require_tool and not manifests:
            raise ValueError("require_tool needs at least one manifest")
        base_url = endpoint.rstrip("/")
        self.responses_url = (
            base_url if base_url.endswith("/responses") else base_url + "/responses"
        )
        self.model = model
        self.resource = resource
        self.command_runner = command_runner
        self.token_provider = token_provider
        self.http_post = http_post
        self.timeout_seconds = timeout_seconds
        self.max_response_calls = max_response_calls
        self.max_tool_calls = max_tool_calls
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.require_tool = require_tool
        self.manifests = MappingProxyType(dict(manifests or {}))
        self.fetch_executor = fetch_executor
        self.ledger = ledger or UsageLedger()
        self.pre_dispatch_hook = pre_dispatch_hook

    def __call__(self, prompt: str, target: str = "default") -> DispatchResult:
        return self.dispatch(prompt, target=target)

    def _token(self) -> str:
        token = (
            self.token_provider()
            if self.token_provider is not None
            else acquire_entra_token(self.command_runner, self.resource)
        )
        if not isinstance(token, str) or not token.strip():
            raise AuthenticationError("token provider returned an empty access token")
        return token.strip()

    def _post(
        self, payload: Mapping[str, Any], token: str
    ) -> Tuple[Mapping[str, Any], int, str, str]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_sha256 = hashlib.sha256(body).hexdigest()
        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            status, raw = _normalize_http_result(
                self.http_post(
                    self.responses_url, headers, body, self.timeout_seconds
                )
            )
        except FoundryDispatchError:
            raise
        except Exception as exc:
            raise FoundryDispatchError("Responses API request failed") from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if status < 200 or status >= 300:
            raise FoundryDispatchError(f"Responses API returned HTTP {status}")
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponseProtocolError("Responses API returned invalid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ResponseProtocolError("Responses API response must be an object")
        return (
            parsed,
            elapsed_ms,
            request_sha256,
            hashlib.sha256(raw).hexdigest(),
        )

    def _record_response(
        self,
        response: Mapping[str, Any],
        elapsed_ms: int,
        target: str,
        request_sha256: str,
        response_sha256: str,
    ) -> ResponseCall:
        response_id = response.get("id")
        status = response.get("status")
        model = response.get("model")
        if not all(isinstance(value, str) and value for value in (response_id, status, model)):
            raise ResponseProtocolError("response is missing id, status, or model")
        call = ResponseCall(
            target=target,
            response_id=response_id,
            status=status,
            model=model,
            elapsed_ms=elapsed_ms,
            usage=parse_usage(response),
            request_sha256=request_sha256,
            response_sha256=response_sha256,
        )
        self.ledger.record(call)
        if status != "completed":
            raise ResponseProtocolError(f"response has incomplete status {status!r}")
        return call

    def _function_calls(self, response: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        output = response.get("output")
        if not isinstance(output, list):
            raise ResponseProtocolError("response output must be an array")
        calls: List[Mapping[str, Any]] = []
        for item in output:
            if not isinstance(item, Mapping):
                raise ResponseProtocolError("response output item must be an object")
            item_type = item.get("type")
            if item_type == "function_call":
                if item.get("name") != TOOL_NAME:
                    raise ToolPolicyError(f"unknown function tool {item.get('name')!r}")
                calls.append(item)
            elif isinstance(item_type, str) and "call" in item_type:
                raise ToolPolicyError(f"unknown tool output type {item_type!r}")
        return calls

    def _execute_call(self, call: Mapping[str, Any]) -> Tuple[Dict[str, str], FetchRecord]:
        call_id = call.get("call_id")
        arguments = call.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ResponseProtocolError("function call is missing call_id")
        if not isinstance(arguments, str):
            raise ResponseProtocolError("function call arguments must be JSON text")
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolPolicyError("function call arguments are invalid JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"manifest_id"}:
            raise ToolPolicyError("fetch_public_api requires only manifest_id")
        manifest_id = parsed["manifest_id"]
        if not isinstance(manifest_id, str) or manifest_id not in self.manifests:
            raise ToolPolicyError(f"unknown manifest {manifest_id!r}")
        if self.fetch_executor is None:
            raise ToolPolicyError("no fetch executor is configured")
        started = time.perf_counter()
        try:
            status, body = _normalize_http_result(
                self.fetch_executor(self.manifests[manifest_id])
            )
        except FoundryDispatchError:
            raise
        except Exception as exc:
            raise FoundryDispatchError(
                f"fetch executor failed for manifest {manifest_id!r}"
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        record = FetchRecord(
            manifest_id=manifest_id,
            status=status,
            latency_ms=latency_ms,
            response_bytes=len(body),
            response_sha256=hashlib.sha256(body).hexdigest(),
        )
        output = json.dumps(
            {
                "status": status,
                "body": body.decode("utf-8", errors="replace"),
            },
            separators=(",", ":"),
        )
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }, record

    def _tool_definitions(self) -> List[Dict[str, Any]]:
        if not self.manifests:
            return []
        return [
            {
                "type": "function",
                "name": TOOL_NAME,
                "description": "Fetch one pre-approved public API manifest.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "manifest_id": {
                            "type": "string",
                            "enum": sorted(self.manifests),
                        }
                    },
                    "required": ["manifest_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

    def initial_payload(self, prompt: str) -> Dict[str, Any]:
        """Build the exact first Responses payload without dispatching it."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        tools = self._tool_definitions()
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"verbosity": self.text_verbosity},
        }
        if tools:
            payload["tools"] = tools
            if self.require_tool:
                payload["tool_choice"] = {
                    "type": "function",
                    "name": TOOL_NAME,
                }
        return payload

    def dispatch(self, prompt: str, *, target: str = "default") -> DispatchResult:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        if target == "__all__":
            raise ValueError("target name '__all__' is reserved")
        token = self._token()
        tools = self._tool_definitions()
        payload = self.initial_payload(prompt)
        calls: List[ResponseCall] = []
        fetches: List[FetchRecord] = []
        seen_call_ids = set()
        started = time.perf_counter()

        for index in range(self.max_response_calls):
            if self.pre_dispatch_hook is not None:
                self.pre_dispatch_hook(MappingProxyType(dict(payload)), index)
            (
                response,
                elapsed_ms,
                request_sha256,
                response_sha256,
            ) = self._post(payload, token)
            response_call = self._record_response(
                response,
                elapsed_ms,
                target,
                request_sha256,
                response_sha256,
            )
            calls.append(response_call)
            function_calls = self._function_calls(response)
            if not function_calls:
                if index == 0 and self.require_tool:
                    raise ToolPolicyError("required fetch_public_api call was omitted")
                usage = ZERO_USAGE
                for call in calls:
                    usage = usage + call.usage
                return DispatchResult(
                    output=_response_text(response),
                    response_id=response_call.response_id,
                    status=response_call.status,
                    model=response_call.model,
                    elapsed_ms=round((time.perf_counter() - started) * 1000),
                    usage=usage,
                    response_calls=tuple(calls),
                    fetches=tuple(fetches),
                )
            if len(seen_call_ids) + len(function_calls) > self.max_tool_calls:
                raise RunawayToolCallsError(
                    f"tool call limit ({self.max_tool_calls}) exceeded"
                )
            outputs = []
            for function_call in function_calls:
                call_id = function_call.get("call_id")
                if call_id in seen_call_ids:
                    raise ToolPolicyError(f"duplicate function call id {call_id!r}")
                seen_call_ids.add(call_id)
                output, fetch = self._execute_call(function_call)
                outputs.append(output)
                fetches.append(fetch)
            payload = {
                "model": self.model,
                "previous_response_id": response_call.response_id,
                "input": outputs,
                "tools": tools,
                "tool_choice": "none",
                "max_output_tokens": self.max_output_tokens,
                "reasoning": {"effort": self.reasoning_effort},
                "text": {"verbosity": self.text_verbosity},
            }

        raise RunawayToolCallsError(
            f"response call limit ({self.max_response_calls}) exceeded"
        )
