"""Tests for the exact allowlisted Fetch controller."""

import io
import json
from pathlib import Path

import pytest

from token_yield.public_api import ApiManifest, execute_manifest, load_manifests

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, body, url="https://example.test/x", status=200):
        self._body = io.BytesIO(body)
        self._url = url
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def getcode(self):
        return self._status

    def geturl(self):
        return self._url

    def read(self, size):
        return self._body.read(size)


def manifest(**overrides):
    values = {
        "manifest_id": "x",
        "method": "GET",
        "url": "https://example.test/x",
        "headers": {"Accept": "application/json"},
        "required_fields": ("id",),
        "max_response_bytes": 100,
        "snapshot_path": "x.json",
        "snapshot_sha256": "0" * 64,
    }
    values.update(overrides)
    return ApiManifest(**values)


def test_committed_manifests_are_safe_and_complete():
    values = load_manifests(
        ROOT / "experiments" / "foundry_wave2" / "api_manifests.json"
    )
    assert len(values) == 3
    assert all(item.url.startswith("https://www.federalregister.gov/api/")
               for item in values.values())


def test_executor_records_response_evidence():
    body = json.dumps({"id": 7}).encode()
    result = execute_manifest(
        manifest(),
        opener=lambda request, timeout: Response(body),
    )
    assert result.status == 200
    assert result.response_bytes == len(body)
    assert len(result.response_sha256) == 64


def test_executor_rejects_redirect_oversize_and_missing_schema():
    with pytest.raises(ValueError, match="redirect"):
        execute_manifest(
            manifest(),
            opener=lambda request, timeout: Response(
                b'{"id":7}', url="https://other.test/x"
            ),
        )
    with pytest.raises(ValueError, match="byte limit"):
        execute_manifest(
            manifest(max_response_bytes=2),
            opener=lambda request, timeout: Response(b'{"id":7}'),
        )
    with pytest.raises(ValueError, match="missing required"):
        execute_manifest(
            manifest(),
            opener=lambda request, timeout: Response(b'{"other":7}'),
        )


def test_executor_accepts_array_root_with_required_item_fields():
    result = execute_manifest(
        manifest(
            root_type="array",
            required_fields=(),
            item_required_fields=("id",),
        ),
        opener=lambda request, timeout: Response(b'[{"id":"x"}]'),
    )
    assert result.response_bytes == 12
