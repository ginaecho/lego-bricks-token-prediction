"""Frozen case construction for the 36-session wave-3 repair block."""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class RepairCase:
    case_id: str
    family: str
    group_id: str
    source_id: str
    prompt: str
    oracle: Mapping[str, Any]
    effort_level: str
    verbosity_level: str
    cache_warm: bool
    output_bound_tokens: int
    output_spec_units: int
    k_declared: int
    k_free_allowed: bool
    attempt_index: int
    arm: Optional[str] = None
    manifest_id: Optional[str] = None
    declared_fetch_bytes: int = 0

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def _source_excerpt(path: Path, spec: Mapping[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "html_text_window":
        parser = _TextExtractor()
        parser.feed(path.read_text(encoding="utf-8"))
        text = " ".join(parser.parts)
    elif kind == "xml_text_window":
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        text = " ".join(" ".join(value.split()) for value in root.itertext())
    elif kind == "json_full":
        value = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    else:
        raise ValueError(f"unsupported source extraction kind {kind}")
    start = str(spec.get("start", ""))
    index = text.find(start) if start else 0
    if index < 0:
        raise ValueError(f"source extraction marker not found: {start}")
    excerpt = html.unescape(text[index:index + int(spec["max_chars"])])
    missing = [
        phrase for phrase in spec["required_phrases"]
        if phrase.casefold() not in excerpt.casefold()
    ]
    if missing:
        raise ValueError(f"source excerpt is missing required facts: {missing}")
    return excerpt


def _summary_case(
    source_id: str,
    excerpt: str,
    required_phrases: List[str],
    effort: str,
    replicate: int,
) -> RepairCase:
    verbosity = "low" if effort == "minimal" else "medium"
    prompt = (
        "Summarise the frozen official-source excerpt below. Return one JSON "
        "object with source_id, summary, and facts. source_id must equal "
        f'"{source_id}". summary must be a faithful concise paragraph. facts '
        "must be an array of short statements covering every REQUIRED_FACT; "
        "do not add facts or advice.\n"
        f"REQUIRED_FACTS={json.dumps(required_phrases)}\n"
        f"OFFICIAL_EXCERPT={excerpt}"
    )
    return RepairCase(
        case_id=f"summarise-{source_id}-{effort}-r{replicate}",
        family="summarise",
        group_id=f"summarise:{source_id}:{effort}",
        source_id=source_id,
        prompt=prompt,
        oracle={
            "schema": {
                "type": "object",
                "required": ["source_id", "summary", "facts"],
                "properties": {
                    "source_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": len(required_phrases),
                    },
                },
                "additionalProperties": False,
            },
            "equals": {"source_id": source_id},
            "required_facts": [
                {"field": "facts", "aliases": [phrase]}
                for phrase in required_phrases
            ],
        },
        effort_level=effort,
        verbosity_level=verbosity,
        cache_warm=replicate > 1,
        output_bound_tokens=768,
        output_spec_units=len(required_phrases),
        k_declared=1,
        k_free_allowed=False,
        attempt_index=1,
    )


def _fetch_expected(source_id: str, value: Any) -> Dict[str, Any]:
    if source_id == "sec-apple-revenue":
        return {
            "source_id": source_id,
            "entity_name": value["entityName"],
            "concept_tag": value["tag"],
            "unit_names": sorted(value["units"]),
            "record_count": len(value["units"]["USD"]),
        }
    if source_id == "usaspending-budget-functions":
        return {
            "source_id": source_id,
            "first_code": value["results"][0]["budget_function_code"],
            "first_title": value["results"][0]["budget_function_title"],
            "record_count": len(value["results"]),
        }
    if source_id == "cms-provider-enrollment":
        return {
            "source_id": source_id,
            "first_npi": value[0]["NPI"],
            "first_state": value[0]["STATE_CD"],
            "record_count": len(value),
        }
    raise KeyError(source_id)


def _fetch_case(
    source_id: str,
    snapshot: str,
    arm: str,
    replicate: int,
) -> RepairCase:
    expected = _fetch_expected(source_id, json.loads(snapshot))
    fields = list(expected)
    instruction = (
        "Return one JSON object with exactly these fields: "
        f"{', '.join(fields)}. Set source_id exactly to \"{source_id}\". "
        "Copy every other value from the official response without inference."
    )
    if arm == "snapshot":
        prompt = f"{instruction}\nOFFICIAL_RESPONSE={snapshot}"
        manifest_id = None
        k_declared = 1
    else:
        prompt = (
            f"{instruction}\nCall fetch_public_api exactly once with "
            f'manifest_id "{source_id}" before answering.'
        )
        manifest_id = source_id
        k_declared = 2
    properties = {
        key: {
            "type": (
                "array" if isinstance(value, list)
                else "integer" if isinstance(value, int)
                else "string"
            )
        }
        for key, value in expected.items()
    }
    if "unit_names" in properties:
        properties["unit_names"]["items"] = {"type": "string"}
    return RepairCase(
        case_id=f"fetch-{source_id}-{arm}-r{replicate}",
        family="fetch",
        group_id=f"fetch:{source_id}:{arm}",
        source_id=source_id,
        prompt=prompt,
        oracle={
            "schema": {
                "type": "object",
                "required": fields,
                "properties": properties,
                "additionalProperties": False,
            },
            "equals": expected,
            "required_facts": [],
        },
        effort_level="minimal",
        verbosity_level="low",
        cache_warm=replicate > 1,
        output_bound_tokens=512,
        output_spec_units=len(fields),
        k_declared=k_declared,
        k_free_allowed=False,
        attempt_index=1,
        arm=arm,
        manifest_id=manifest_id,
        declared_fetch_bytes=len(snapshot.encode("utf-8")),
    )


def _control_case(effort: str, replicate: int) -> RepairCase:
    return RepairCase(
        case_id=f"control-{effort}-r{replicate}",
        family="control",
        group_id=f"control:{effort}",
        source_id="none",
        prompt='Return exactly this JSON object: {"status":"DONE"}',
        oracle={
            "schema": {
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
                "additionalProperties": False,
            },
            "equals": {"status": "DONE"},
            "required_facts": [],
        },
        effort_level=effort,
        verbosity_level="low" if effort == "minimal" else "medium",
        cache_warm=replicate > 1,
        output_bound_tokens=32,
        output_spec_units=1,
        k_declared=1,
        k_free_allowed=False,
        attempt_index=1,
    )


def _tokenizer_confirmation_case(size: str, repetitions: int) -> RepairCase:
    payload = " ".join(["alpha beta gamma delta"] * repetitions)
    prompt = (
        "This is a tokenizer/null calibration payload. Ignore its content and "
        'return exactly this JSON object: {"status":"DONE"}.\n'
        f"CALIBRATION_PAYLOAD={payload}"
    )
    return RepairCase(
        case_id=f"tokenizer-confirmation-{size}",
        family="tokenizer_confirmation",
        group_id=f"tokenizer_confirmation:{size}",
        source_id="none",
        prompt=prompt,
        oracle={
            "schema": {
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
                "additionalProperties": False,
            },
            "equals": {"status": "DONE"},
            "required_facts": [],
        },
        effort_level="minimal",
        verbosity_level="low",
        cache_warm=False,
        output_bound_tokens=32,
        output_spec_units=1,
        k_declared=1,
        k_free_allowed=False,
        attempt_index=1,
    )


def _randomized_topological_order(
    cases: List[RepairCase], seed: int
) -> List[RepairCase]:
    """Randomize groups while ensuring every warm replicate follows its seed."""

    by_group: Dict[str, List[RepairCase]] = {}
    for case in cases:
        by_group.setdefault(case.group_id, []).append(case)
    for values in by_group.values():
        values.sort(key=lambda case: case.case_id)
        values.sort(key=lambda case: case.cache_warm)
    rng = random.Random(seed)
    output = []
    active = list(by_group)
    while active:
        group = rng.choice(active)
        output.append(by_group[group].pop(0))
        if not by_group[group]:
            active.remove(group)
    return output


def load_repair_cases(experiment_dir: Path, seed: int = 260827) -> List[RepairCase]:
    sources = json.loads(
        (experiment_dir / "source_registry.json").read_text(encoding="utf-8")
    )["summary_sources"]
    cases: List[RepairCase] = []
    for source_id, source in sources.items():
        snapshot_path = experiment_dir / source["snapshot_path"]
        excerpt = _source_excerpt(snapshot_path, source["extract"])
        for effort in ("minimal", "medium"):
            for replicate in (1, 2):
                cases.append(_summary_case(
                    source_id,
                    excerpt,
                    list(source["extract"]["required_phrases"]),
                    effort,
                    replicate,
                ))
    manifests = json.loads(
        (experiment_dir / "api_manifests.json").read_text(encoding="utf-8")
    )
    for source_id, manifest in manifests.items():
        snapshot = (
            experiment_dir / manifest["snapshot_path"]
        ).read_text(encoding="utf-8")
        for arm in ("snapshot", "live"):
            for replicate in (1, 2, 3):
                cases.append(_fetch_case(source_id, snapshot, arm, replicate))
    for effort in ("minimal", "medium"):
        for replicate in (1, 2):
            cases.append(_control_case(effort, replicate))
    cases.extend((
        _tokenizer_confirmation_case("small", 32),
        _tokenizer_confirmation_case("large", 512),
    ))
    if len(cases) != 36:
        raise AssertionError(f"repair matrix has {len(cases)} cases, expected 36")
    ordered = _randomized_topological_order(cases, seed)
    if len({case.case_id for case in ordered}) != 36:
        raise AssertionError("repair case IDs must be unique")
    return ordered
