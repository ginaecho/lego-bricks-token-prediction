"""Frozen research measurements; six existing briefs, no delivery or fresh test set.

Preparation is offline. Execution claims the additional approval once, persists
reservations before dispatch, and never resumes or retries an uncertain attempt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .budget import HardBudget, SafetyRateCard
from .customer_decomposition import (
    SCOPING_BRICKS, batch_candidates, build_prompt, canonical_json, check_output, compile_plan,
    content_hash, load_catalog, load_template, quote_features, validate_batches,
)
from .customer_pilot import (
    _audit_deployment, _default_transport, _dispatcher, _now,
    _require_measured_channels, _write_json, load_config,
)
from .economics import Pricing
from .foundry_dispatch import UsageLedger, acquire_entra_token


VERSION = "marketplace-measurements-v1"
APPROVAL_ID = "marketplace-additional-usd50-v1"
PINNED_INPUTS = {
    "catalog.json": "62b4673545fa88adb464c61ec3dcd14d29702fdf81838cf712f378fd05a315d3",
    "template.json": "cc47f80157ee9a5ddb2bca83d44244388ff863a6f6eb9a4f0afa390ea19e3f6c",
}
RUNTIME_FILES = (
    "marketplace_measurements.py", "customer_decomposition.py", "customer_pilot.py",
    "foundry_dispatch.py", "budget.py", "economics.py",
)
LAYOUTS = {
    "standalone-v1": tuple((slug,) for slug in SCOPING_BRICKS),
    "paired-v1": (SCOPING_BRICKS[:2], SCOPING_BRICKS[2:]),
    "batched-v1": (SCOPING_BRICKS,),
}
for _index, _batches in enumerate(batch_candidates()):
    if _batches not in LAYOUTS.values():
        LAYOUTS[f"intermediate-mask-{_index}-v1"] = _batches
UNSUPPORTED_VARIANT = "sequential-handoff-v2"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_campaign(root: Path) -> dict:
    """Build a deterministic manifest and exact requests from pinned local inputs."""
    root = Path(root).resolve()
    source = root / "experiments" / "customer_requests"
    for filename, digest in PINNED_INPUTS.items():
        if _sha((source / filename).read_bytes()) != digest:
            raise ValueError("approved six-brief catalog or original template hash changed")
    catalog = load_catalog(source / "catalog.json")
    template = load_template(source / "template.json")
    original = load_config(source / "pilot.json")
    config = {key: original[key] for key in (
        "endpoint", "deployment", "expected_response_model", "deployment_version",
        "deployment_sku", "reasoning_effort", "text_verbosity", "pricing", "pricing_source",
    )}
    config.update(cap_usd=50.0, stop_usd=48.0)
    rates, pricing = SafetyRateCard(), Pricing(**config["pricing"])
    calls, projects = [], []
    for project in catalog["projects"]:
        plan = compile_plan(project, template)
        group = "train" if project["split"] == "train" else "historical_holdout"
        projects.append({
            "project_id": project["id"], "deduplication_group": project["id"],
            "group": group, "original_split": project["split"],
            "source_url": project["source_url"], "project_sha256": content_hash(project),
            "plan_sha256": plan.sha256, "requirement_count": len(project["requirements"]),
            "context_bytes": quote_features(project, template, SCOPING_BRICKS)["context_bytes"],
            "counts": dict(plan.counts),
        })
        # Scheduling is serial, but tasks never consume another generated answer.
        # Rotate layout order deterministically across projects; this is not
        # a randomized causal trial or balanced repeated-measures design.
        variants = list(LAYOUTS.items())
        offset = (len(projects) - 1) % len(variants)
        for variant, batches in variants[offset:] + variants[:offset]:
            validate_batches(batches)
            for batch_index, operations in enumerate(batches):
                quote = quote_features(project, template, operations)
                prompt = build_prompt(project, template, operations)
                output_cap = quote["planned_output_tokens"]
                # UTF-8 bytes plus framing slack is a conservative local proxy,
                # not a provider token count. Breaches stop subsequent dispatch.
                input_bound = quote["prompt_bytes"] + 4096
                if input_bound > 32768:
                    raise ValueError("approved prompt exceeds input reservation ceiling")
                payload = {
                    "model": config["deployment"], "input": prompt,
                    "max_output_tokens": output_cap,
                    "reasoning": {"effort": config["reasoning_effort"]},
                    "text": {"verbosity": config["text_verbosity"]},
                }
                calls.append({
                    "call_id": f"{project['id']}-{variant}-r0-{batch_index}",
                    "project_id": project["id"], "deduplication_group": project["id"],
                    "group": group, "variant": variant, "replicate": 0,
                    "operations": list(operations), "quote": quote,
                    "service_selections": [{
                        "service_id": slug,
                        "version": f"marketplace-services-v1:{content_hash(template)}",
                        "quantity": quote["counts"][slug],
                    } for slug in operations],
                    "plan_sha256": plan.sha256, "prompt_sha256": _sha(prompt.encode("utf-8")),
                    "payload_sha256": content_hash(payload), "payload": payload,
                    "max_input_tokens": input_bound, "max_output_tokens": output_cap,
                    "reservation_usd": rates.price(input_bound, 2 * output_cap),
                    "uncached_retail_bound_usd": pricing.attempt_cost(
                        input_tokens=input_bound, cached_input_tokens=0, output_tokens=output_cap),
                })
    manifest = {
        "schema_version": VERSION,
        "authorization": {
            "approval_id": APPROVAL_ID, "additional_cap_usd": 50.0,
            "operational_stop_usd": 48.0, "historical_spend_charged_to_this_approval": False,
            "user_approval": ["Up to USD 50",
                              "Reuse the six approved paraphrased briefs for execution-variant measurements"],
            "inputs": "Only the six existing local paraphrases, annotations and exclusions.",
            "permissions": "Permission to use local paraphrases only; no source license expansion.",
            "prohibited": ["source downloads", "private inputs", "tools", "customer contact",
                           "cloud resource changes", "full engagement execution"],
            "execution_gate": "--execute required; single-use approval claim across run directories",
            "retries": "No automatic retries or resume; unknown charges retain full reservations.",
        },
        "provenance": {
            "catalog_path": "experiments\\customer_requests\\catalog.json",
            "source_file_sha256": {
                **PINNED_INPUTS, "pilot.json": _sha((source / "pilot.json").read_bytes()),
            },
            "catalog_sha256": content_hash(catalog), "template_sha256": content_hash(template),
            "runtime_file_sha256": {
                name: _sha((root / "token_yield" / name).read_bytes()) for name in RUNTIME_FILES
            },
        },
        "config": config, "catalog": catalog, "template": template,
        "services": [{"service_id": op["id"],
                      "service_version": f"marketplace-services-v1:{content_hash(template)}",
                      "contract": op, "depends_on": []} for op in template["operations"]],
        "projects": projects,
        "grouping": {
            "unit": "project_id; all variants/repeats of a brief stay together",
            "train_projects": 4, "historical_holdout_projects": 2,
            "calibration_projects": 0, "fresh_test_projects": 0,
            "restriction": "Historical holdout labels must not train, calibrate or select models.",
        },
        "variation_matrix": {
            "replicates": 1, "input_sizes": "original briefs only; no truncation or augmentation",
            "ordering": "Projects retain original order; layout order rotates by project index.",
            "quantities": "original listed requirements only; no synthetic quantity scaling",
            "layouts": [{"variant": name, "batches": [list(batch) for batch in batches],
                         "status": "supported", "semantics": "independent-original-brief"}
                        for name, batches in LAYOUTS.items()],
            "unsupported": [{
                "variant": UNSUPPORTED_VARIANT, "status": "blocked",
                "reason": "Handoffs change v1 independent semantics. A separate validated "
                          "template, contracts and research authorization are required.",
            }],
        },
        "acceptance_criteria": [
            "Every layout preserves all four operations and per-service unit counts exactly once.",
            "Completed response with all five measured usage channels and pinned model identity.",
            "Original check_output contract must pass; failure halts rather than retrying.",
            "Failed/incomplete attempts remain evidence and never become zero-cost successes.",
            "Human draft-quality review required; automatic checks do not prove noninferiority.",
            "No new unseen-project, calibration, coverage, universal discount or delivery claims.",
        ],
        "calls": [{key: value for key, value in call.items() if key != "payload"}
                  for call in calls],
        "preview": {
            "planned_calls": len(calls), "planned_project_layouts": len(projects) * len(LAYOUTS),
            "max_calls_per_dispatch": 1, "tools": 0, "automatic_retries": 0,
            "total_reservation_usd": sum(call["reservation_usd"] for call in calls),
            "uncached_retail_bound_usd": sum(call["uncached_retail_bound_usd"] for call in calls),
            "expected_cost_usd": None,
            "cost_note": "Bounds, not a prediction or invoice. Output includes per-operation caps; "
                         "safety reserves output and reasoning separately. Runtime may stop before "
                         "the whole matrix completes; USD48 guard takes precedence.",
        },
        "limitations": [
            "Research variants of the same six briefs; not independent calibration or fresh testing.",
            "All eight contiguous independent layouts are planned once per brief; no repeated "
            "measurements, independent replication or randomized causal claim.",
            "No workflow predictor fitting until new measurements exist; historical percentages "
            "are not discounts.",
            "Retail prices are inherited from the local 2026-09-14 snapshot, not re-queried.",
            "Input reservations are local byte bounds, not provider-attested counts; "
            "a provider invoice is not controlled by this application-side safety guard.",
        ],
    }
    # Normalize tuples once so JSON reloading retains exact equality.
    manifest = json.loads(canonical_json(manifest))
    return {"manifest": manifest, "manifest_sha256": content_hash(manifest), "calls": calls}


def prepare_campaign(root: Path, variant: str | None = None) -> dict:
    """Verify the checked-in manifest without network calls or file writes."""
    if variant is not None and variant not in LAYOUTS:
        raise ValueError(f"unsupported variant: {variant}; sequential handoffs are blocked")
    campaign = build_campaign(root)
    path = Path(root) / "experiments" / "marketplace" / "manifest.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen != manifest_record(campaign):
        raise ValueError("frozen marketplace manifest differs from local inputs/runtime/contracts")
    # Subsetting would silently change the approved matrix or its evidence denominator.
    if variant is not None:
        raise ValueError("supported layouts must execute as the complete frozen matrix")
    return campaign


def manifest_record(campaign: dict) -> dict:
    """Compact review surface; its campaign digest binds every exact payload."""
    manifest = campaign["manifest"]
    return {
        key: manifest[key] for key in (
            "schema_version", "authorization", "provenance", "config", "projects",
            "grouping", "variation_matrix", "acceptance_criteria", "preview", "limitations",
        )
    } | {
        "campaign_sha256": campaign["manifest_sha256"],
        "service_versions": {op["service_id"]: op["service_version"] for op in manifest["services"]},
        "template_version": manifest["template"]["schema_version"],
        "calls_sha256": content_hash(manifest["calls"]),
    }


def _summary(campaign: dict, rows: list[dict], budget: HardBudget, status: str) -> dict:
    workflows = []
    for project in campaign["manifest"]["projects"]:
        for variant in LAYOUTS:
            planned = [call for call in campaign["calls"]
                       if call["project_id"] == project["project_id"] and call["variant"] == variant]
            observed = [row for row in rows if row["call_id"] in {c["call_id"] for c in planned}]
            complete = len(observed) == len(planned) and all(
                row["status"] == "completed" for row in observed)
            workflows.append({
                "project_id": project["project_id"], "group": project["group"], "variant": variant,
                "planned_calls": len(planned), "attempt_records": len(observed),
                "complete": complete,
                "rated_cost_usd": sum(row["rated_cost_usd"] for row in observed) if complete else None,
                "all_contract_checks_passed": complete and all(
                    row["quality"]["contract_passed"] for row in observed),
            })
    return {
        "status": status, "manifest_sha256": campaign["manifest_sha256"],
        "planned_calls": len(campaign["calls"]), "attempt_records": len(rows),
        "completed_calls": sum(row["status"] == "completed" for row in rows),
        "measured_rated_cost_usd": sum(row["rated_cost_usd"] or 0 for row in rows),
        "unknown_cost_attempts": sum(row["usage"] is None and row["dispatched"] for row in rows),
        "total_cost_known": not budget.snapshot()["active_reservations"],
        "budget": budget.snapshot(), "workflows": workflows,
        "quality_noninferiority_established": False, "production_recommendation": None,
        "predictor_status": "blocked_pending_measurements_and_project_grouped_evaluation",
        "unsupported_variants": campaign["manifest"]["variation_matrix"]["unsupported"],
    }


def execute_campaign(
    campaign: dict, root: Path, run_dir: Path, *, execute: bool = False,
    token_provider: Callable = acquire_entra_token, transport: Callable = _default_transport,
) -> dict:
    """Execute a frozen matrix once; failures consume the single-use approval claim."""
    if not execute:
        raise ValueError("paid execution requires explicit execute=True / --execute")
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if campaign != prepare_campaign(root):
        raise ValueError("campaign differs from the frozen manifest")
    if run_dir.parent != root / "runs":
        raise ValueError("run directory must be a new direct child of repository runs")
    if run_dir.exists():
        raise FileExistsError(run_dir)
    approval = root / "experiments" / "marketplace" / (APPROVAL_ID + ".claim")
    # Exclusive mkdir prevents parallel processes or a new run name resetting the cap.
    approval.mkdir(exist_ok=False)
    _write_json(approval / "claim.json", {
        "approval_id": APPROVAL_ID, "run_name": run_dir.name,
        "manifest_sha256": campaign["manifest_sha256"], "claimed_at": _now(),
        "retry_allowed": False, "reconciliation_required_before_any_new_authorization": True,
    })
    run_dir.mkdir(parents=True, exist_ok=False)
    config = campaign["manifest"]["config"]
    budget = HardBudget(config["cap_usd"], config["stop_usd"])
    pricing, ledger = Pricing(**config["pricing"]), UsageLedger()
    rows = []
    projects = {p["id"]: p for p in campaign["manifest"]["catalog"]["projects"]}
    _write_json(run_dir / "protocol.json", campaign)

    def persist(status: str) -> dict:
        _write_json(run_dir / "records.json", rows)
        _write_json(approval / "budget.json", budget.snapshot())
        _write_json(run_dir / "budget.json", budget.snapshot())
        summary = _summary(campaign, rows, budget, status)
        _write_json(run_dir / "analysis.json", summary)
        _write_json(run_dir / "state.json", {
            "status": status, "at": _now(), "no_automatic_retry": True,
        })
        return summary

    persist("preflight")
    try:
        token = token_provider()
        if not isinstance(token, str) or not token.strip():
            raise ValueError("authentication returned no token")
        _audit_deployment(config, run_dir, "before")
        for call in campaign["calls"]:
            row = {**{k: v for k, v in call.items() if k != "payload"},
                   "started_at": _now(), "status": "pending_reservation",
                   "dispatched": False, "usage": None, "rated_cost_usd": None,
                   "quality": {"contract_passed": False, "human_review_required": True}}
            rows.append(row)
            try:
                budget.reserve(call["call_id"], call["max_input_tokens"],
                               2 * call["max_output_tokens"])
            except RuntimeError:
                row.update(status="budget_blocked", finished_at=_now())
                return persist("budget_stopped")
            row["status"] = "reserved"
            persist("dispatching")

            def audited_transport(url, headers, body, timeout):
                if (json.loads(body) != call["payload"]
                        or url != config["endpoint"] + "/responses" or row["dispatched"]):
                    raise ValueError("request differs from frozen contract or repeats an attempt")
                (run_dir / (call["call_id"] + ".request.json")).write_bytes(body)
                row.update(dispatched=True, dispatched_at=_now(), request_sha256=_sha(body))
                persist("dispatching")
                status, raw = transport(url, headers, body, timeout)
                # Headers and arbitrary exception strings are never persisted. Redact
                # even an unexpected credential echo from provider response evidence.
                safe_raw = raw.replace(token.encode("utf-8"), b"[REDACTED_CREDENTIAL]")
                (run_dir / (call["call_id"] + ".response.json")).write_bytes(safe_raw)
                row.update(http_status=status, response_received_at=_now(),
                           response_sha256=_sha(raw), stored_response_sha256=_sha(safe_raw),
                           response_redacted=safe_raw != raw)
                if 200 <= status < 300:
                    _require_measured_channels(raw)
                return status, raw

            try:
                result = _dispatcher(config, call["max_output_tokens"], lambda: token,
                                     audited_transport, ledger)(call["payload"]["input"],
                                                                target=call["call_id"])
                row.update(response_status=result.status, response_model=result.model,
                           output=result.output.replace(token, "[REDACTED_CREDENTIAL]"))
                if result.model != config["expected_response_model"]:
                    raise ValueError("response model identity mismatch")
                row["quality"] = check_output(
                    projects[call["project_id"]], tuple(call["operations"]), row["output"])
                if not row["quality"]["contract_passed"]:
                    raise ValueError("output failed the original scoping contract")
                row["status"] = "completed"
            except (Exception, KeyboardInterrupt) as exc:
                row.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                try:
                    measured = ledger.calls(call["call_id"])
                    if measured:
                        usage = ledger.totals()[call["call_id"]]
                        row.update(usage=asdict(usage),
                                   response_calls=[asdict(item) for item in measured])
                        row["rated_cost_usd"] = pricing.attempt_cost(
                            input_tokens=usage.input_tokens, cached_input_tokens=usage.cached_tokens,
                            output_tokens=usage.output_tokens)
                        budget.settle(call["call_id"], row["usage"], row["rated_cost_usd"])
                        if (usage.input_tokens > call["max_input_tokens"]
                                or usage.output_tokens > call["max_output_tokens"]
                                or usage.reasoning_tokens > call["max_output_tokens"]):
                            row["status"] = "reservation_bound_violation"
                            raise RuntimeError("measured usage exceeds reservation bounds")
                except (Exception, KeyboardInterrupt) as exc:
                    if row["status"] != "reservation_bound_violation":
                        row["status"] = "settlement_failed"
                    row["error_type"] = type(exc).__name__
                    raise
                finally:
                    row["finished_at"] = _now()
                    persist("running")
        _audit_deployment(config, run_dir, "after")
        return persist("completed_research_only")
    except (Exception, KeyboardInterrupt) as exc:
        persist("halted")
        _write_json(run_dir / "failure.json", {
            "error_type": type(exc).__name__, "at": _now(),
            "detail": "Stopped without retry. Inspect raw evidence; reconcile active reservations.",
        })
        raise
