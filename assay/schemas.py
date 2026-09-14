"""Record contracts and their JSONL storage.

Three rules are enforced here rather than trusted:

* ``billable_units`` and ``cost_usd`` are **recomputed on load** from the raw token counts
  and the pricing record. A value written to the file is never believed.
* A run whose usage is missing is an *excluded measurement*, never a zero.
* Storage is append-only. Nothing in this module can rewrite or delete a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

from assay.errors import SchemaError
from assay.evidence import EvidenceClass
from assay.pricing import Pricing


class Tier(str, Enum):
    NULL = "null"
    LADDER = "ladder"
    BASE = "base"
    COMPOSITE = "composite"
    BLIND = "blind"
    BATCHING = "batching"
    LIVE = "live"


class Split(str, Enum):
    FIT = "fit"
    BLIND = "blind"
    BATCHING = "batching"
    PROBE = "probe"


class Status(str, Enum):
    OK = "ok"
    FAILED = "failed"
    EXCLUDED = "excluded"


class ExclusionCode(str, Enum):
    MODEL_MISMATCH = "model_mismatch"
    CACHE_NONZERO = "cache_nonzero"
    REASONING_NONZERO = "reasoning_nonzero"
    TRUNCATED = "truncated"
    MAX_TURNS = "max_turns"
    TRANSPORT_ERROR = "transport_error"
    PATH_VIOLATION = "path_violation"
    SCHEMA_INVALID = "schema_invalid"
    USAGE_MISSING = "usage_missing"

    @property
    def is_transport(self) -> bool:
        """Transport failures are retried under a fresh run_id and do not count against
        the 10% exclusion ceiling; everything else does."""
        return self is ExclusionCode.TRANSPORT_ERROR


FITTABLE_TIERS = frozenset({Tier.NULL, Tier.BASE, Tier.COMPOSITE})
"""Ladder probes shape the design; they are never fitted. Blind and batching never are either."""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def contract_violation(self) -> ExclusionCode | None:
        """Cache and hidden reasoning both break the measurement, in opposite directions."""
        if self.cache_read_tokens or self.cache_write_tokens:
            return ExclusionCode.CACHE_NONZERO
        if self.reasoning_tokens:
            return ExclusionCode.REASONING_NONZERO
        return None


@dataclass
class Run:
    run_id: str
    campaign_id: str
    evidence_class: EvidenceClass
    split: Split
    tier: Tier
    probe_id: str
    replicate: int
    instruction_sha256: str
    units: dict[str, int]
    context_files: list[str]
    context_bytes: int
    rendered_prompt_bytes: int
    model_sent: str
    model_echoed: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    elapsed_s: float = 0.0
    tool_calls: int = 0
    tool_result_bytes: int = 0
    status: Status = Status.OK
    exclusion_code: ExclusionCode | None = None
    had_retry: bool = False
    quality_score: float | None = None
    bundle_id: str | None = None
    output_sha256: str = ""
    runtime_hash: str = ""
    pricing_hash: str = ""
    git_sha: str = ""
    timestamp: str = ""
    notes: str = ""

    # Derived. Never read from disk; always recomputed by `attach_pricing`.
    billable_units: float = field(default=0.0, compare=False)
    cost_usd: float = field(default=0.0, compare=False)

    @property
    def usage(self) -> Usage:
        return Usage(
            self.prompt_tokens,
            self.completion_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.reasoning_tokens,
        )

    @property
    def total_units(self) -> int:
        return sum(self.units.values())

    @property
    def accepted(self) -> bool:
        return self.status is Status.OK and self.exclusion_code is None

    @property
    def fittable(self) -> bool:
        return self.accepted and self.tier in FITTABLE_TIERS and self.split is Split.FIT

    def attach_pricing(self, pricing: Pricing) -> "Run":
        self.billable_units = pricing.billable_units(self.prompt_tokens, self.completion_tokens)
        self.cost_usd = pricing.cost_usd(self.prompt_tokens, self.completion_tokens)
        return self

    def enforce_contract(self, expected_model: str | None = None) -> "Run":
        """Apply the locked contract. Returns self, mutated to excluded where it fails."""
        if self.exclusion_code is not None:
            self.status = Status.EXCLUDED
            return self
        code = self.usage.contract_violation()
        if code is None and expected_model and self.model_echoed != expected_model:
            code = ExclusionCode.MODEL_MISMATCH
        if code is not None:
            self.exclusion_code = code
            self.status = Status.EXCLUDED
        return self

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["evidence_class"] = self.evidence_class.value
        data["split"] = self.split.value
        data["tier"] = self.tier.value
        data["status"] = self.status.value
        data["exclusion_code"] = self.exclusion_code.value if self.exclusion_code else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Run":
        payload = dict(data)
        payload.pop("billable_units", None)
        payload.pop("cost_usd", None)
        try:
            payload["evidence_class"] = EvidenceClass(payload["evidence_class"])
            payload["split"] = Split(payload["split"])
            payload["tier"] = Tier(payload["tier"])
            payload["status"] = Status(payload.get("status", "ok"))
            code = payload.get("exclusion_code")
            payload["exclusion_code"] = ExclusionCode(code) if code else None
        except (KeyError, ValueError) as exc:
            raise SchemaError(f"run record has an invalid enum field: {exc}") from exc

        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise SchemaError(f"run record has unknown fields: {sorted(unknown)}")
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise SchemaError(f"run record is missing required fields: {exc}") from exc


def make_run_id(
    probe_id: str,
    context_hashes: Iterable[str],
    runtime_hash: str,
    pricing_hash: str,
    replicate: int,
) -> str:
    """Deterministic identity so a killed campaign resumes without duplicating spend."""
    payload = "|".join(
        [probe_id, ",".join(sorted(context_hashes)), runtime_hash, pricing_hash, str(replicate)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- append-only JSONL ------------------------------------------------------------


def append_run(path: str | Path, run: Run) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")


def read_runs(path: str | Path, pricing: Pricing) -> list[Run]:
    p = Path(path)
    if not p.exists():
        return []
    runs: list[Run] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                run = Run.from_dict(json.loads(line))
            except (json.JSONDecodeError, SchemaError) as exc:
                raise SchemaError(f"{p}:{lineno}: {exc}") from exc
            runs.append(run.attach_pricing(pricing))
    return runs


def existing_run_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    ids: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["run_id"])
    return ids


def iter_accepted(runs: Iterable[Run]) -> Iterator[Run]:
    return (r for r in runs if r.accepted)


def exclusion_rate(runs: Iterable[Run]) -> tuple[float, int, int]:
    """Non-transport exclusion rate. Above 10% the campaign is not interpretable."""
    runs = list(runs)
    considered = [r for r in runs if not (r.exclusion_code and r.exclusion_code.is_transport)]
    if not considered:
        return 0.0, 0, 0
    excluded = sum(1 for r in considered if not r.accepted)
    return excluded / len(considered) * 100.0, excluded, len(considered)
