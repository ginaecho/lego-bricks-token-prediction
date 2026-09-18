"""Execute a campaign: dispatch probes, record runs, refuse to do anything unrecorded.

Four guarantees, each of which exists because of a specific way campaigns go wrong:

* **Resumable.** ``run_id`` is a hash of what was asked, not a counter, so a killed
  campaign restarts without paying twice for the same probe.
* **Append-only.** Nothing here can rewrite or delete a run. A bad run becomes an
  exclusion row with a code, never a hole in the file.
* **Budget-fenced.** The dispatch envelope is checked before each call, not tallied
  afterwards. The cap is not a cost control -- the campaign is cheap -- it is what stops
  the study quietly turning into a search for a flattering result.
* **Seal-respecting.** Blind probes refuse to dispatch unless blind intent is declared.
* **Cache-fenced.** The adapter must declare that prompt caching is off *before* the first
  call. A cache hit is caught after the fact by the usage contract, but by then it is a
  paid-for exclusion, and four of them void a 24-task blind set.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from assay.corpus import SealedCorpus
from assay.errors import AdapterError, BudgetError, SealError
from assay.harness.adapter import AgentAdapter
from assay.harness.probes import Probe
from assay.schemas import (
    ExclusionCode,
    Run,
    Split,
    Status,
    append_run,
    existing_run_ids,
    make_run_id,
)
from assay.pricing import Pricing


@dataclass
class Budget:
    max_dispatches: int
    spent: int = 0

    @property
    def remaining(self) -> int:
        return self.max_dispatches - self.spent

    def reserve(self, n: int = 1) -> None:
        if self.remaining < n:
            raise BudgetError(
                f"dispatch would exceed the frozen envelope "
                f"({self.spent}/{self.max_dispatches} already spent)"
            )
        self.spent += n


@dataclass
class CampaignReport:
    campaign_id: str
    dispatched: int = 0
    skipped: int = 0
    excluded: int = 0
    runs: list[Run] = field(default_factory=list)
    exclusions: list[tuple[str, ExclusionCode]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "dispatched": self.dispatched,
            "skipped_already_present": self.skipped,
            "excluded": self.excluded,
            "exclusion_codes": [
                {"run_id": rid, "code": code.value} for rid, code in self.exclusions
            ],
        }


def git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or default
    except (OSError, subprocess.SubprocessError):
        return default


def assert_caching_disabled(adapter: AgentAdapter) -> None:
    """Refuse to start a campaign against an adapter that has not disabled prompt caching.

    ``Usage.contract_violation`` already excludes a run whose usage reports cache tokens,
    but that is a post-mortem: the dispatch is paid for, and on the 24-task blind set the
    fourth such exclusion voids the entire set rather than shrinking it. Requiring the
    declaration up front turns an expensive discovery into a cheap refusal.
    """
    declared = getattr(adapter, "caching_disabled", None)
    if declared is not True:
        raise AdapterError(
            f"{getattr(adapter, 'name', adapter)!r} does not declare caching_disabled=True. "
            "Prompt caching makes a run a measurement of what was asked before it, not of "
            "the task; disable it at the provider and declare it on the adapter."
        )


def run_campaign(
    probes: Sequence[Probe],
    adapter: AgentAdapter,
    out_path: str | Path,
    *,
    campaign_id: str,
    pricing: Pricing,
    corpus: SealedCorpus,
    runtime_hash: str,
    expected_model: str | None = None,
    budget: Budget | None = None,
    allow_blind: bool = False,
    temperature: float = 0.0,
    on_dispatch: Callable[[Probe], None] | None = None,
) -> CampaignReport:
    report = CampaignReport(campaign_id=campaign_id)
    assert_caching_disabled(adapter)
    already = existing_run_ids(out_path)
    pricing_hash = pricing.hash()
    sha = git_sha()

    for probe in probes:
        if probe.split is Split.BLIND and not allow_blind:
            raise SealError(
                f"{probe.probe_id} is a blind probe; dispatch it through the blind runner"
            )

        doc_hashes = [corpus.manifest.by_name(n).sha256 for n in probe.context_files]
        run_id = make_run_id(
            probe.probe_id, doc_hashes, runtime_hash, pricing_hash, probe.replicate
        )
        if run_id in already:
            report.skipped += 1
            continue

        paths = [Path(corpus.root) / n for n in probe.context_files]
        context_bytes = sum(corpus.size(n) for n in probe.context_files)

        register = getattr(adapter, "register", None)
        if register is not None:
            register(probe.instruction, probe.units)

        if budget is not None:
            budget.reserve(1)
        if on_dispatch is not None:
            on_dispatch(probe)

        exclusion: ExclusionCode | None = None
        try:
            result = adapter.run(probe.instruction, paths, temperature=temperature)
        except AdapterError:
            exclusion = ExclusionCode.TRANSPORT_ERROR
            result = None
        except Exception:  # noqa: BLE001 -- an unclassified failure is still a failure, not a zero
            exclusion = ExclusionCode.SCHEMA_INVALID
            result = None

        run = Run(
            run_id=run_id,
            campaign_id=campaign_id,
            evidence_class=adapter.evidence_class,
            split=probe.split,
            tier=probe.tier,
            probe_id=probe.probe_id,
            replicate=probe.replicate,
            instruction_sha256=probe.instruction_sha256,
            units=dict(probe.units),
            context_files=list(probe.context_files),
            context_bytes=context_bytes,
            rendered_prompt_bytes=len(probe.instruction.encode("utf-8")) + context_bytes,
            model_sent=adapter.model,
            model_echoed=result.model_echoed if result else "",
            prompt_tokens=result.prompt_tokens if result else 0,
            completion_tokens=result.completion_tokens if result else 0,
            cache_read_tokens=result.cache_read_tokens if result else 0,
            cache_write_tokens=result.cache_write_tokens if result else 0,
            reasoning_tokens=result.reasoning_tokens if result else 0,
            elapsed_s=result.elapsed_s if result else 0.0,
            tool_calls=result.tool_calls if result else 0,
            tool_result_bytes=result.tool_result_bytes if result else 0,
            status=Status.OK if result else Status.FAILED,
            exclusion_code=exclusion,
            bundle_id=probe.bundle_id,
            runtime_hash=runtime_hash,
            pricing_hash=pricing_hash,
            git_sha=sha,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            notes=probe.notes,
        )
        run.enforce_contract(expected_model)
        run.attach_pricing(pricing)

        append_run(out_path, run)
        report.runs.append(run)
        report.dispatched += 1
        if not run.accepted:
            report.excluded += 1
            if run.exclusion_code:
                report.exclusions.append((run.run_id, run.exclusion_code))

    return report
