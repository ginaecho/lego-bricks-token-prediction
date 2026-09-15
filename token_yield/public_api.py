"""Allowlisted public-API execution for the Fetch brick."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class ApiManifest:
    manifest_id: str
    method: str
    url: str
    headers: Dict[str, str]
    required_fields: tuple[str, ...]
    max_response_bytes: int
    snapshot_path: str
    snapshot_sha256: str
    root_type: str = "object"
    item_required_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class FetchEvidence:
    manifest_id: str
    method: str
    url: str
    status: int
    latency_ms: int
    response_bytes: int
    response_sha256: str
    body: str

    def record(self) -> Dict[str, Any]:
        return asdict(self)


def load_manifests(path: Path) -> Dict[str, ApiManifest]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("API manifest file must be a non-empty object")
    manifests = {}
    prohibited_headers = {"authorization", "cookie", "proxy-authorization"}
    for manifest_id, value in raw.items():
        method = str(value["method"]).upper()
        url = str(value["url"])
        headers = {str(k): str(v) for k, v in value.get("headers", {}).items()}
        maximum = int(value["max_response_bytes"])
        if method != "GET":
            raise ValueError(f"{manifest_id}: only GET is allowed in this pilot")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username:
            raise ValueError(f"{manifest_id}: URL must be public HTTPS")
        if {key.casefold() for key in headers} & prohibited_headers:
            raise ValueError(f"{manifest_id}: secret-bearing headers are prohibited")
        if maximum <= 0 or maximum > 1_000_000:
            raise ValueError(f"{manifest_id}: invalid maximum response size")
        manifests[manifest_id] = ApiManifest(
            manifest_id=manifest_id,
            method=method,
            url=url,
            headers=headers,
            required_fields=tuple(value["expected_schema"].get("required", ())),
            max_response_bytes=maximum,
            snapshot_path=str(value["snapshot_path"]),
            snapshot_sha256=str(value["snapshot_sha256"]),
            root_type=str(value["expected_schema"].get("root_type", "object")),
            item_required_fields=tuple(
                value["expected_schema"].get("item_required", ())
            ),
        )
    return manifests


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_without_redirect(request: Request, timeout: float):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def execute_manifest(
    manifest: ApiManifest,
    opener: Callable[..., Any] = _open_without_redirect,
    timeout_seconds: float = 30,
) -> FetchEvidence:
    """Execute one immutable request and validate size, URL, and JSON schema."""

    request = Request(
        manifest.url, method=manifest.method, headers=manifest.headers
    )
    started = time.perf_counter()
    with opener(request, timeout=timeout_seconds) as response:
        status = int(response.getcode())
        final_url = response.geturl()
        body = response.read(manifest.max_response_bytes + 1)
    latency_ms = round((time.perf_counter() - started) * 1000)
    if final_url != manifest.url:
        raise ValueError(f"{manifest.manifest_id}: redirect is not allowlisted")
    if status != 200:
        raise RuntimeError(f"{manifest.manifest_id}: HTTP status {status}")
    if len(body) > manifest.max_response_bytes:
        raise ValueError(f"{manifest.manifest_id}: response exceeds byte limit")
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{manifest.manifest_id}: response is not JSON") from exc
    if manifest.root_type == "object" and not isinstance(parsed, dict):
        raise ValueError(f"{manifest.manifest_id}: response must be an object")
    if manifest.root_type == "array" and not isinstance(parsed, list):
        raise ValueError(f"{manifest.manifest_id}: response must be an array")
    if manifest.root_type not in {"object", "array"}:
        raise ValueError(f"{manifest.manifest_id}: unsupported root type")
    missing = (
        [field for field in manifest.required_fields if field not in parsed]
        if isinstance(parsed, dict) else []
    )
    if missing:
        raise ValueError(
            f"{manifest.manifest_id}: missing required fields: {missing}"
        )
    if isinstance(parsed, list):
        invalid = [
            index for index, item in enumerate(parsed)
            if not isinstance(item, dict)
            or any(field not in item for field in manifest.item_required_fields)
        ]
        if invalid:
            raise ValueError(
                f"{manifest.manifest_id}: invalid array items: {invalid}"
            )
    text = body.decode("utf-8")
    return FetchEvidence(
        manifest_id=manifest.manifest_id,
        method=manifest.method,
        url=manifest.url,
        status=status,
        latency_ms=latency_ms,
        response_bytes=len(body),
        response_sha256=hashlib.sha256(body).hexdigest(),
        body=text,
    )
