"""Frozen case construction and quality oracles for the Foundry wave-2 pilot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    family: str
    group_id: str
    prompt: str
    expected: Any
    context_bytes: int
    units: int
    max_output_tokens: int
    manifest_id: Optional[str] = None
    arm: Optional[str] = None

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


def _json_object(output: str) -> Mapping[str, Any]:
    match = re.search(r"\{.*\}", output, re.DOTALL)
    value = json.loads(match.group(0) if match else output)
    if not isinstance(value, dict):
        raise ValueError("output is not a JSON object")
    return value


def evaluate_pilot_output(case: PilotCase, output: str) -> Dict[str, Any]:
    """Apply a semantic or exact structured oracle without length scoring."""

    if case.family == "control":
        passed = output.strip() == "DONE"
        return {"accepted": passed, "detail": "exact DONE" if passed else "not DONE"}
    try:
        value = _json_object(output)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"accepted": False, "detail": str(exc)}
    if case.family == "summarise":
        expected = dict(case.expected)
        mismatched = [
            key for key, item in expected.items() if value.get(key) != item
        ]
        summary = value.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            mismatched.append("summary")
        return {
            "accepted": not mismatched,
            "detail": "semantic facts preserved" if not mismatched
            else "mismatched: " + ", ".join(mismatched),
        }
    mismatched = [
        key for key, item in dict(case.expected).items() if value.get(key) != item
    ]
    return {
        "accepted": not mismatched,
        "detail": "exact mapped facts" if not mismatched
        else "mismatched: " + ", ".join(mismatched),
    }


def _summarise_case(
    case_id: str, size: int, source: bytes, replicate: int
) -> PilotCase:
    context = source[:size].decode("utf-8", errors="ignore")
    actual_bytes = len(context.encode("utf-8"))
    expected = {
        "document_title": (
            "Medicare Program; Inpatient Rehabilitation Facility Prospective "
            "Payment System for Federal Fiscal Year 2025 and Updates to the "
            "Inpatient Rehabilitation Facility Quality Reporting Program; "
            "Proposed Rule"
        ),
        "agency": "Centers for Medicare & Medicaid Services",
        "publication_date": "2024-03-29",
    }
    prompt = (
        "Summarise the public Federal Register excerpt below. Return one JSON "
        "object with document_title, agency, publication_date, and summary. "
        "Preserve those three facts exactly; summary must faithfully compress "
        "only supplied content and must not add advice.\n\n"
        f"EXCERPT ({actual_bytes} UTF-8 bytes):\n{context}"
    )
    return PilotCase(
        case_id, "summarise", f"summarise-{size}", prompt, expected,
        actual_bytes, 1, 512,
    )


def _transform_case(case_id: str, fields: int, replicate: int) -> PilotCase:
    source = {
        f"source_{index:03d}": f"value-{index:03d}"
        for index in range(1, fields + 1)
    }
    mapping = {
        f"source_{index:03d}": f"target_{index:03d}"
        for index in range(1, fields + 1)
    }
    expected = {target: source[src] for src, target in mapping.items()}
    prompt = (
        "Transform SOURCE using MAPPING. Return only one JSON object with "
        "exactly the target fields and copied values; do not infer or omit.\n"
        f"SOURCE={json.dumps(source, sort_keys=True)}\n"
        f"MAPPING={json.dumps(mapping, sort_keys=True)}"
    )
    return PilotCase(
        case_id, "transform", f"transform-{fields}", prompt, expected,
        len(prompt.encode("utf-8")), fields, 1024,
    )


def _fetch_case(
    case_id: str,
    document_id: str,
    arm: str,
    snapshot: str,
) -> PilotCase:
    manifest_id = f"federal-register-{document_id}"
    source = json.loads(snapshot)
    agencies = [item["name"] for item in source["agencies"]]
    expected = {
        "document_number": source["document_number"],
        "title": source["title"],
        "publication_date": source["publication_date"],
        "agencies": agencies,
        "html_url": source["html_url"],
    }
    instruction = (
        "Return only one JSON object containing document_number, title, "
        "publication_date, agencies (array of agency names), and html_url. "
        "Copy values exactly."
    )
    if arm == "snapshot":
        prompt = f"{instruction}\nFROZEN_API_RESPONSE={snapshot}"
        manifest = None
        context_bytes = len(snapshot.encode("utf-8"))
    else:
        prompt = (
            f"{instruction}\nCall fetch_public_api exactly once with "
            f'manifest_id "{manifest_id}". Do not answer before using the tool.'
        )
        manifest = manifest_id
        context_bytes = 0
    return PilotCase(
        case_id, "fetch", f"fetch-{document_id}", prompt, expected,
        context_bytes, 1, 512, manifest, arm,
    )


def load_pilot_cases(experiment_dir: Path) -> List[PilotCase]:
    """Build cases in the exact preregistered dispatch order."""

    prereg = json.loads(
        (experiment_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    summary_source = (
        experiment_dir / "snapshots" / "federal-register-2024-06550.txt"
    ).read_bytes()
    cases: Dict[str, PilotCase] = {}
    for size in (1024, 10240, 102400):
        for replicate in range(1, 4):
            case_id = f"summarise-{size}-r{replicate}"
            cases[case_id] = _summarise_case(
                case_id, size, summary_source, replicate
            )
    for fields in (2, 8, 32):
        for replicate in range(1, 4):
            case_id = f"transform-{fields}-r{replicate}"
            cases[case_id] = _transform_case(case_id, fields, replicate)
    for document_id in ("2019-24499", "2024-06550", "2025-01358"):
        snapshot = (
            experiment_dir / "snapshots"
            / f"federal-register-{document_id}.json"
        ).read_text(encoding="utf-8")
        for arm in ("snapshot", "live"):
            case_id = f"fetch-{document_id}-{arm}"
            cases[case_id] = _fetch_case(
                case_id, document_id, arm, snapshot
            )
    for replicate in range(1, 7):
        case_id = f"control-r{replicate}"
        cases[case_id] = PilotCase(
            case_id, "control", "control", "Reply with exactly: DONE",
            "DONE", 0, 0, 32,
        )
    order = [row["case_id"] for row in prereg["order"]]
    if len(order) != 30 or len(set(order)) != 30 or set(order) != set(cases):
        raise ValueError("preregistered pilot order is not a complete 30-case set")
    return [cases[case_id] for case_id in order]
