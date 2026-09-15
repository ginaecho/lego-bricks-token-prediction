"""Source-backed business cases for measuring atomic and composed task bricks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from urllib.parse import urlparse

from .tasks import ORDER, PRIMITIVES


@dataclass(frozen=True)
class AcceptanceRule:
    """One deterministic condition an agent output must satisfy."""

    kind: str
    value: Any


@dataclass(frozen=True)
class BusinessCase:
    """A reproducible business task tied to evidence and an acceptance gate."""

    case_id: str
    title: str
    tier: str
    business_domain: str
    provenance: str
    source_urls: Tuple[str, ...]
    source_note: str
    input_driver: str
    input_size: int
    context_bytes: int
    counts: Dict[str, int]
    prompt: str
    acceptance: Tuple[AcceptanceRule, ...]
    held_out: bool = False

    @property
    def arity(self) -> int:
        return sum(1 for count in self.counts.values() if count)

    @property
    def total_units(self) -> int:
        return sum(self.counts.values())

    def notation(self) -> str:
        bits = []
        for slug in ORDER:
            count = self.counts.get(slug, 0)
            if count:
                name = PRIMITIVES[slug].name
                bits.append(f"{count}x{name}" if count > 1 else name)
        return " + ".join(bits)


@dataclass(frozen=True)
class AcceptanceResult:
    """The independently checkable verdict for one completed case."""

    accepted: bool
    checks: Tuple[Tuple[str, bool, str], ...]


def _is_public_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def validate_case(case: BusinessCase) -> None:
    """Reject cases that cannot support a credible measurement."""

    if not case.case_id or not case.title or not case.prompt.strip():
        raise ValueError("case_id, title, and prompt are required")
    if case.tier not in {"base", "composite"}:
        raise ValueError(f"{case.case_id}: tier must be base or composite")
    if case.provenance not in {"source-backed", "created"}:
        raise ValueError(f"{case.case_id}: unsupported provenance")
    if not case.source_urls or not all(_is_public_https_url(u) for u in case.source_urls):
        raise ValueError(f"{case.case_id}: at least one public HTTPS source is required")
    if not case.source_note.strip():
        raise ValueError(f"{case.case_id}: source_note is required")
    allowed_drivers = {
        "input_bytes", "output_units", "corpus_bytes", "mixed",
        "external_calls", "alternatives_x_criteria", "conditions_x_points",
        "subtask_count", "recipients_x_fields", "gates_x_evidence",
        "field_mappings",
    }
    if case.input_driver not in allowed_drivers:
        raise ValueError(
            f"{case.case_id}: input_driver must be one of {sorted(allowed_drivers)}"
        )
    if not isinstance(case.input_size, int) or case.input_size < 0:
        raise ValueError(f"{case.case_id}: input_size must be a non-negative integer")
    if not isinstance(case.context_bytes, int) or case.context_bytes < 0:
        raise ValueError(
            f"{case.case_id}: context_bytes must be a non-negative integer"
        )
    if case.input_driver in {"input_bytes", "corpus_bytes", "mixed"} and not case.input_size:
        raise ValueError(f"{case.case_id}: byte-driven cases require input_size")
    if case.input_driver not in {"input_bytes", "corpus_bytes", "mixed"}:
        if not case.input_size:
            raise ValueError(f"{case.case_id}: count-driven cases require input_size")
    if not case.acceptance:
        raise ValueError(f"{case.case_id}: acceptance criteria are required")
    unknown = set(case.counts) - set(ORDER)
    if unknown:
        raise ValueError(f"{case.case_id}: unknown primitives: {sorted(unknown)}")
    if any(not isinstance(n, int) or n < 0 for n in case.counts.values()):
        raise ValueError(f"{case.case_id}: primitive counts must be non-negative integers")
    expected_arity = 1 if case.tier == "base" else 2
    if case.arity < expected_arity:
        raise ValueError(f"{case.case_id}: {case.tier} case has arity {case.arity}")


def _case_from_dict(data: Mapping[str, Any]) -> BusinessCase:
    counts = {slug: 0 for slug in ORDER}
    counts.update(data["counts"])
    case = BusinessCase(
        case_id=data["case_id"],
        title=data["title"],
        tier=data["tier"],
        business_domain=data["business_domain"],
        provenance=data["provenance"],
        source_urls=tuple(data["source_urls"]),
        source_note=data["source_note"],
        input_driver=data["input_driver"],
        input_size=data["input_size"],
        context_bytes=data["context_bytes"],
        counts=counts,
        prompt=data["prompt"],
        acceptance=tuple(
            AcceptanceRule(rule["kind"], rule["value"])
            for rule in data["acceptance"]
        ),
        held_out=bool(data.get("held_out", False)),
    )
    validate_case(case)
    return case


def load_cases(path: str) -> List[BusinessCase]:
    """Load and validate a JSONL business-case catalog."""

    cases = []
    seen = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = _case_from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if case.case_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate case_id {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    return cases


def evaluate_output(case: BusinessCase, output: str) -> AcceptanceResult:
    """Apply the catalog's deterministic acceptance rules to an agent output."""

    checks = []
    parsed_json: Any = None
    json_error = ""
    for rule in case.acceptance:
        if rule.kind == "contains_all":
            required = [str(v) for v in rule.value]
            missing = [v for v in required if v.casefold() not in output.casefold()]
            checks.append((rule.kind, not missing,
                           "all required text present" if not missing
                           else "missing: " + ", ".join(missing)))
        elif rule.kind == "valid_json_fields":
            if parsed_json is None and not json_error:
                try:
                    match = re.search(r"\{.*\}", output, re.DOTALL)
                    parsed_json = json.loads(match.group(0) if match else output)
                except json.JSONDecodeError as exc:
                    json_error = str(exc)
            fields = [str(v) for v in rule.value]
            missing = fields if not isinstance(parsed_json, dict) else [
                field for field in fields if field not in parsed_json
            ]
            passed = not json_error and not missing
            detail = ("valid JSON with required fields" if passed else
                      json_error or "missing fields: " + ", ".join(missing))
            checks.append((rule.kind, passed, detail))
        elif rule.kind == "json_equals":
            if parsed_json is None and not json_error:
                try:
                    match = re.search(r"\{.*\}", output, re.DOTALL)
                    parsed_json = json.loads(match.group(0) if match else output)
                except json.JSONDecodeError as exc:
                    json_error = str(exc)
            expected = dict(rule.value)
            mismatched = [
                field for field, value in expected.items()
                if not isinstance(parsed_json, dict) or parsed_json.get(field) != value
            ]
            passed = not json_error and not mismatched
            detail = ("expected field values present" if passed else
                      json_error or "mismatched fields: " + ", ".join(mismatched))
            checks.append((rule.kind, passed, detail))
        elif rule.kind == "max_words":
            actual = len(output.split())
            limit = int(rule.value)
            checks.append((rule.kind, actual <= limit, f"{actual}/{limit} words"))
        elif rule.kind == "min_lines":
            actual = len([line for line in output.splitlines() if line.strip()])
            minimum = int(rule.value)
            checks.append((rule.kind, actual >= minimum, f"{actual}/{minimum} lines"))
        elif rule.kind == "json_types":
            if parsed_json is None and not json_error:
                try:
                    match = re.search(r"\{.*\}", output, re.DOTALL)
                    parsed_json = json.loads(match.group(0) if match else output)
                except json.JSONDecodeError as exc:
                    json_error = str(exc)
            type_map = {
                "string": str, "number": (int, float), "integer": int,
                "boolean": bool, "object": dict, "array": list,
            }
            invalid = []
            for field, type_name in dict(rule.value).items():
                expected = type_map.get(str(type_name))
                actual = parsed_json.get(field) if isinstance(parsed_json, dict) else None
                if expected is None:
                    raise ValueError(
                        f"{case.case_id}: unknown JSON type {type_name}"
                    )
                if (
                    not isinstance(actual, expected)
                    or type_name in {"number", "integer"} and isinstance(actual, bool)
                ):
                    invalid.append(field)
            passed = not json_error and not invalid
            detail = ("expected JSON field types present" if passed else
                      json_error or "invalid fields: " + ", ".join(invalid))
            checks.append((rule.kind, passed, detail))
        else:
            raise ValueError(f"{case.case_id}: unknown acceptance rule {rule.kind}")
    return AcceptanceResult(all(passed for _, passed, _ in checks), tuple(checks))


def to_run(case: BusinessCase, result: Any):
    """Bridge a measured case result into the composition model's training row."""

    from .compose import Run

    def value(name: str, default: Any = None) -> Any:
        if isinstance(result, Mapping):
            return result.get(name, default)
        return getattr(result, name, default)

    if value("proof_type") == "acceptance-only" or value("provenance") != "measured":
        raise ValueError("only measured run records can enter model training")
    tokens = value("total_tokens", 0)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("a positive measured total_tokens value is required")
    tool_uses = value("tool_uses")
    if (
        tool_uses is not None
        and (
            not isinstance(tool_uses, int)
            or isinstance(tool_uses, bool)
            or tool_uses < 0
        )
    ):
        raise ValueError("tool_uses must be null or a non-negative integer")
    return Run(
        label=case.case_id,
        notation=case.notation(),
        counts=dict(case.counts),
        context_bytes=case.context_bytes,
        arity=case.arity,
        tokens=tokens,
        tool_uses=tool_uses,
        held_out=case.held_out,
    )


def catalog_summary(cases: Iterable[BusinessCase]) -> Dict[str, Any]:
    """Return coverage facts suitable for experiment reports."""

    cases = list(cases)
    covered = {slug for case in cases for slug, count in case.counts.items() if count}
    return {
        "cases": len(cases),
        "base_cases": sum(case.tier == "base" for case in cases),
        "composite_cases": sum(case.tier == "composite" for case in cases),
        "primitives_covered": len(covered),
        "missing_primitives": [slug for slug in ORDER if slug not in covered],
        "domains": sorted({case.business_domain for case in cases}),
    }
