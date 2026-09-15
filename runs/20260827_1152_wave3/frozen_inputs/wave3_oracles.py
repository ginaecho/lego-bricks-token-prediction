"""Frozen, normalized structural and semantic acceptance oracles for wave 3."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
)
_TYPE_MAP = {
    "object": dict, "array": list, "string": str, "integer": int,
    "number": (int, float), "boolean": bool, "null": type(None),
}


def normalize_text(value: str) -> str:
    """Normalize Unicode, aliases, dates, decimals, whitespace, and punctuation."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\s*&\s*", " and ", text)
    # Strip punctuation around dates before parsing each plausible date phrase.
    date_pattern = re.compile(
        r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:jan(?:uary)?|"
        r"feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+\d{4})\b", re.IGNORECASE)

    def date_replace(match: re.Match[str]) -> str:
        candidate = match.group(0).replace(",", "")
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                pass
        return candidate

    text = date_pattern.sub(date_replace, text)

    def decimal_replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        # ISO dates have already been canonicalized; do not reinterpret their
        # year/month/day components as independent decimal quantities.
        before = text[match.start() - 1:match.start()]
        after = text[match.end():match.end() + 1]
        if before == "-" or after == "-":
            return raw
        negative = raw.strip().startswith("(") and raw.strip().endswith(")")
        number = re.sub(r"[^0-9.+-]", "", raw.replace(",", ""))
        try:
            normalized = format(Decimal(number).normalize(), "f")
            return ("-" if negative else "") + normalized
        except InvalidOperation:
            return raw

    text = re.sub(
        r"(?<!\w)(?:[$€£]\s*)?(?:\(\s*)?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*\))?(?!\w)",
        decimal_replace, text,
    )
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def normalize_value(value: Any) -> Any:
    """Recursively normalize strings while preserving JSON structure."""

    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    if isinstance(value, float):
        return format(Decimal(str(value)).normalize(), "f")
    return value


def _is_type(value: Any, kind: str) -> bool:
    expected = _TYPE_MAP.get(kind)
    if expected is None:
        raise ValueError(f"unsupported schema type: {kind}")
    if kind in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> List[str]:
    """Validate the required JSON-schema-like subset used by frozen oracles."""

    errors: List[str] = []
    kind = schema.get("type")
    if kind and not _is_type(value, kind):
        return [f"{path}: expected {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value outside frozen enum")
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        for key, child in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate_schema(value[key], child, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(schema.get("properties", {}))
            errors.extend(f"{path}.{key}: additional field" for key in sorted(extras))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        child = schema.get("items")
        if child:
            for index, item in enumerate(value):
                errors.extend(validate_schema(item, child, f"{path}[{index}]"))
    return errors


def _field(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def validate_semantic_facts(
    value: Mapping[str, Any], required_facts: Sequence[Mapping[str, Any]]
) -> Tuple[bool, List[str]]:
    """Match all facts against frozen patterns and aliases after normalization."""

    missing: List[str] = []
    for fact in required_facts:
        field = str(fact["field"])
        actual = _field(value, field)
        if actual is None:
            missing.append(field)
            continue
        haystack = normalize_text(
            " ".join(map(str, actual)) if isinstance(actual, list) else str(actual)
        )
        accepted = []
        if "pattern" in fact:
            accepted.append(str(fact["pattern"]))
        accepted.extend(map(str, fact.get("aliases", ())))
        # Patterns are frozen inputs. Match normalized patterns as regexes;
        # aliases are escaped so ordinary punctuation cannot become regex syntax.
        pattern_hit = False
        if "pattern" in fact:
            try:
                pattern_hit = re.search(str(fact["pattern"]), haystack, re.IGNORECASE) is not None
            except re.error as exc:
                raise ValueError(f"invalid frozen pattern for {field}: {exc}") from exc
        alias_hit = any(normalize_text(alias) in haystack for alias in fact.get("aliases", ()))
        if not (pattern_hit or alias_hit):
            missing.append(field)
    return not missing, missing


def _parse_object(output: Any) -> Mapping[str, Any]:
    if isinstance(output, Mapping):
        return output
    if not isinstance(output, str):
        raise ValueError("response must be a JSON object or string")
    text = output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def evaluate_oracle(
    output: Any,
    oracle: Mapping[str, Any],
    *,
    telemetry_complete: bool,
    provider_incomplete: bool = False,
) -> Dict[str, Any]:
    """Evaluate independently recorded telemetry, structure, and semantics.

    Incomplete provider delivery makes content verdicts ``None`` rather than
    falsely labelling a truncated response as semantically wrong.
    """

    if provider_incomplete or not telemetry_complete:
        return {
            "telemetry_complete": bool(telemetry_complete),
            "provider_incomplete": bool(provider_incomplete),
            "structural_acceptance": None,
            "semantic_acceptance": None,
            "overall_acceptance": False,
            "errors": ["provider response incomplete"] if provider_incomplete
                      else ["usage telemetry incomplete"],
        }
    try:
        value = _parse_object(output)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "telemetry_complete": True, "provider_incomplete": False,
            "structural_acceptance": False, "semantic_acceptance": None,
            "overall_acceptance": False, "errors": [str(exc)],
        }
    schema_errors = validate_schema(value, oracle.get("schema", {"type": "object"}))
    structural = not schema_errors
    semantic: Optional[bool] = None
    errors = list(schema_errors)
    if structural:
        semantic, missing = validate_semantic_facts(
            value, oracle.get("required_facts", ())
        )
        errors.extend(f"semantic fact missing: {field}" for field in missing)
        for field, expected in oracle.get("equals", {}).items():
            actual = _field(value, str(field))
            if normalize_value(actual) != normalize_value(expected):
                semantic = False
                errors.append(f"{field}: value differs from frozen expected value")
        cited_field = oracle.get("cited_ids_field")
        if cited_field:
            cited = _field(value, str(cited_field))
            allowed = set(oracle.get("input_ids", ()))
            if not isinstance(cited, list):
                semantic = False
                errors.append(f"{cited_field}: cited IDs must be an array")
            else:
                unknown = sorted(set(cited) - allowed)
                if unknown:
                    semantic = False
                    errors.append("cited IDs not in input: " + ", ".join(map(str, unknown)))
    return {
        "telemetry_complete": True,
        "provider_incomplete": False,
        "structural_acceptance": structural,
        "semantic_acceptance": semantic,
        "overall_acceptance": bool(structural and semantic),
        "errors": errors,
    }
