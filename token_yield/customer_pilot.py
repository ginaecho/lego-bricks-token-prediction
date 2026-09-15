"""Frozen, capped scoping pilot over authentic public customer requests.

No network access occurs during planning. Execution is single-use per run
directory: ambiguous attempts are never retried automatically. Retail-rate
costs and conservative spending reservations are retained separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .budget import HardBudget, SafetyRateCard
from .customer_decomposition import (
    SCOPING_BRICKS, arm_batches, batch_candidates, build_prompt, canonical_json,
    check_output, compile_plan, content_hash, delivery_decomposition,
    load_catalog, load_template, quote_features,
)
from .economics import Pricing
from .foundry_count import FoundryCountError, FoundryInputTokenCounter
from .foundry_dispatch import (
    FoundryDispatchError, FoundryDispatcher, ResponseProtocolError, UsageLedger,
    acquire_entra_token,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _require_measured_channels(raw: bytes) -> None:
    try:
        response = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseProtocolError("response contains invalid JSON") from exc
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        raise ResponseProtocolError("response is missing measured usage")
    for path in (
        ("input_tokens",), ("output_tokens",), ("total_tokens",),
        ("input_tokens_details", "cached_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ):
        node = usage
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise ResponseProtocolError(f"missing measured usage channel: {'.'.join(path)}")
            node = node[key]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "customer-pilot-v1":
        raise ValueError("unsupported pilot configuration")
    if config.get("endpoint") != "https://foundary-tzuc06.openai.azure.com/openai/v1":
        raise ValueError("this approval covers only the specified Azure endpoint")
    if (config.get("deployment") != "gpt-5.4"
            or config.get("expected_response_model") != "gpt-5.4"
            or config.get("deployment_version") != "2026-03-05"
            or config.get("deployment_sku") != "GlobalStandard"):
        raise ValueError("pilot requires the approved pinned GPT-5.4 deployment")
    cap, stop = config.get("cap_usd"), config.get("stop_usd")
    if any(type(n) not in (int, float) or not math.isfinite(n) for n in (cap, stop)):
        raise ValueError("budget values must be finite numbers")
    if not 0 < stop <= cap <= 50:
        raise ValueError("budget must satisfy 0 < stop <= cap <= approved USD 50")
    prior = config["prior_attempt"]
    prior_cost, prior_safety = prior["rated_cost_usd"], prior["settled_safety_usd"]
    if (any(type(n) not in (int, float) or not math.isfinite(n)
            for n in (prior_cost, prior_safety))
            or not 0 <= prior_cost <= prior_safety
            or prior["active_reserved_usd"] != 0
            or cap + prior_safety > 50 or stop + prior_safety > 48):
        raise ValueError("reconciled prior spend must fit the cumulative approval and stop")
    if (type(config.get("replicates")) is not int or config["replicates"] != 2
            or type(config.get("seed")) is not int
            or type(config.get("max_input_tokens_per_call")) is not int
            or config["max_input_tokens_per_call"] != 32768):
        raise ValueError("v1 requires two replicates and a 32768-token input ceiling")
    if config.get("reasoning_effort") != "none" or config.get("text_verbosity") != "low":
        raise ValueError("v1 reasoning/verbosity must be none/low")
    pricing = Pricing(**config["pricing"])
    safety = SafetyRateCard()
    if not (0 < pricing.input_per_million <= safety.input_per_million_usd
            and 0 < pricing.effective_cached_input_per_million <= pricing.input_per_million
            and 0 < pricing.output_per_million <= safety.output_per_million_usd):
        raise ValueError("verified pricing must be positive and covered by safety rates")
    return config


def _deployment_probe() -> dict:
    az = shutil.which("az")
    if az is None:
        raise RuntimeError("Azure CLI is required to verify the deployment version")
    try:
        result = subprocess.run([
            az, "cognitiveservices", "account", "deployment", "show",
            "--name", "foundary-tzuc06", "--deployment-name", "gpt-5.4",
            "--resource-group", "rg-tzuc06",
            "--subscription", "ef669702-542a-4abc-95a6-edf9f972cd3c",
            "--query", "{model:properties.model,sku:sku.name,state:properties.provisioningState,"
            "versionUpgradeOption:properties.versionUpgradeOption}", "--output", "json",
        ], capture_output=True, text=True, check=True, timeout=90)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Azure deployment verification failed") from exc
    return json.loads(result.stdout)


def _audit_deployment(config: dict, run_dir: Path, phase: str) -> None:
    metadata = _deployment_probe()
    _write_json(run_dir / f"deployment-{phase}.json", {
        "observed_at": _now(), "metadata": metadata,
        "limitation": "Control-plane observations bracket execution; response alias "
        "does not independently attest the backend version on each call.",
    })
    model = metadata.get("model")
    if (not isinstance(model, dict) or model.get("name") != config["deployment"]
            or model.get("version") != config["deployment_version"]
            or metadata.get("sku") != config["deployment_sku"]
            or metadata.get("state") != "Succeeded"):
        raise RuntimeError("Azure deployment differs from approved model/version/SKU")


def prepare_campaign(experiment_dir: Path) -> dict:
    """Create deterministic calls and all candidate layouts without executing."""
    catalog = load_catalog(experiment_dir / "catalog.json")
    template = load_template(experiment_dir / "template.json")
    config = load_config(experiment_dir / "pilot.json")
    projects = catalog["projects"]
    if (len(projects) != 6
            or sum(p["split"] == "train" for p in projects) != 4):
        raise ValueError("v1 requires four training and two outcome-holdout projects")
    plans = []
    calls = []
    rng = random.Random(config["seed"])
    for project in projects:
        compiled = compile_plan(project, template)
        plans.append({
            **asdict(compiled), "plan_sha256": compiled.sha256,
            "delivery_counts": delivery_decomposition(project).counts,
            "delivery_workload": "unknown; counts are catalog deliverables only",
            "candidates": [
                {"batches": batches, "quotes": [
                    quote_features(project, template, batch) for batch in batches
                ]}
                for batches in batch_candidates()
            ],
        })
    for split in ("train", "holdout"):
        jobs = [(p, repeat, arm) for p in projects if p["split"] == split
                for repeat in range(config["replicates"]) for arm in ("split", "batched")]
        rng.shuffle(jobs)
        for project, repeat, arm in jobs:
            plan = compile_plan(project, template)
            for batch_index, operations in enumerate(arm_batches(arm)):
                quote = quote_features(project, template, operations)
                prompt = build_prompt(project, template, operations)
                if quote["prompt_bytes"] + 4096 > config["max_input_tokens_per_call"]:
                    raise ValueError("prompt is too large for conservative input reservation")
                calls.append({
                    "call_id": f"{project['id']}-{repeat}-{arm}-{batch_index}",
                    "project_id": project["id"], "split": split, "replicate": repeat,
                    "arm": arm, "operations": list(operations), "quote": quote,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "plan_sha256": plan.sha256,
                })
    return {
        "schema_version": "customer-campaign-v1",
        "catalog": catalog, "template": template, "config": config,
        "catalog_sha256": content_hash(catalog), "template_sha256": content_hash(template),
        "config_sha256": content_hash(config), "plans": plans, "calls": calls,
        "limitations": [
            "Scoping drafts only, not execution of customer engagements.",
            "Hand-annotated deliverable units are not known full-project workloads.",
            "Shared template prevents a source/template-independent confirmatory claim.",
            "Four train and two outcome-holdout projects cannot calibrate useful tails.",
            "Automatic contract checks do not establish semantic quality of free text.",
            "Rated costs use published prices; invoice reconciliation is outstanding.",
        ],
    }


def _default_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def _dispatcher(config: dict, output_cap: int, token_provider: Callable[[], str],
                transport: Callable, ledger: UsageLedger) -> FoundryDispatcher:
    return FoundryDispatcher(
        config["endpoint"], model=config["deployment"], token_provider=token_provider,
        http_post=transport, max_response_calls=1, max_tool_calls=0,
        max_output_tokens=output_cap, reasoning_effort=config["reasoning_effort"],
        text_verbosity=config["text_verbosity"], ledger=ledger,
    )


def _freeze_predictions(campaign: dict, rows: list[dict], run_dir: Path) -> dict:
    from .customer_models import fit_models, predict_cost

    models = fit_models(rows)
    _write_json(run_dir / "models.json", models)
    predictions = [
        {"call_id": call["call_id"], "predictions": {
            form: predict_cost(models, call["quote"], form)
            for form in models["forms"]
        }}
        for call in campaign["calls"] if call["split"] == "holdout"
    ]
    artifact = {
        "frozen_at": _now(), "models_sha256": content_hash(models),
        "selected_form": models["selected_form"], "predictions": predictions,
        "candidates": [
            {"project_id": plan["project_id"], "plan_sha256": plan["plan_sha256"],
             "layouts": [
                 {"batches": candidate["batches"],
                  "predicted_rated_cost_usd": sum(
                      predict_cost(models, quote) for quote in candidate["quotes"]),
                  "quality_evidence": "unmeasured_layout" if len(candidate["batches"])
                  not in (1, 4) else "await_paired_measurements"}
                 for candidate in plan["candidates"]
             ]}
            for plan in campaign["plans"]
        ],
    }
    _write_json(run_dir / "predictions.json", artifact)
    return artifact


def summarize_batching(campaign: dict, rows: list[dict]) -> dict:
    """Rank only complete paired layouts; retain a human quality-review gate."""
    indexed = {row["call_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate measured call ids")
    expected = {call["call_id"]: call for call in campaign["calls"]}
    if set(indexed) - set(expected):
        raise ValueError("measurements contain calls outside the frozen campaign")
    for row in rows:
        for field in ("project_id", "arm", "replicate", "plan_sha256", "operations"):
            if row[field] != expected[row["call_id"]][field]:
                raise ValueError("measurements disagree with the frozen layout")
    projects = []
    for project in campaign["catalog"]["projects"]:
        arms = {}
        for arm in ("split", "batched"):
            planned = [call for call in campaign["calls"]
                       if call["project_id"] == project["id"] and call["arm"] == arm]
            observed = [indexed[call["call_id"]] for call in planned
                        if call["call_id"] in indexed]
            complete = (len(observed) == len(planned) and all(
                row["status"] == "completed" and row["rated_cost_usd"] is not None
                for row in observed))
            passed = complete and all(row["quality"]["contract_passed"] for row in observed)
            arms[arm] = {
                "complete": complete, "all_contract_checks_passed": passed,
                "mean_rated_cost_usd": sum(row["rated_cost_usd"] for row in observed)
                / campaign["config"]["replicates"] if complete else None,
            }
        comparable = all(value["complete"] for value in arms.values())
        contract_ok = all(value["all_contract_checks_passed"] for value in arms.values())
        savings = None
        candidate = None
        if comparable:
            split_cost = arms["split"]["mean_rated_cost_usd"]
            batched_cost = arms["batched"]["mean_rated_cost_usd"]
            savings = 1 - batched_cost / split_cost if split_cost else None
            if contract_ok:
                candidate = "batched" if batched_cost < split_cost else "split"
        projects.append({
            "project_id": project["id"], "arms": arms,
            "observed_savings_fraction": savings,
            "cheapest_contract_passing_arm": candidate,
            "production_recommendation": None,
            "reason": "human_review_of_draft_quality_required" if contract_ok
            else "incomplete_or_failed_contract_checks",
        })
    return {"projects": projects, "quality_noninferiority_established": False,
            "unmeasured_layouts_eligible": False}


def execute_campaign(
    campaign: dict,
    run_dir: Path,
    *,
    token_provider: Callable[[], str] = acquire_entra_token,
    transport: Callable = _default_transport,
    count_probe: Callable | None = None,
) -> dict:
    """Execute once, saving a reservation before every possible billable call.

    A failed or interrupted campaign cannot be restarted in this directory.
    Unknown charges keep their full reservation. There are no automatic retries.
    """
    run_dir.mkdir(parents=True, exist_ok=False)
    config = campaign["config"]
    pricing = Pricing(**config["pricing"])
    budget = HardBudget(config["cap_usd"], config["stop_usd"])
    rows: list[dict] = []
    predictions = None
    projects = {p["id"]: p for p in campaign["catalog"]["projects"]}
    code_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("customer_pilot.py", "customer_decomposition.py", "customer_models.py",
                     "robust.py", "foundry_dispatch.py", "foundry_count.py",
                     "budget.py", "economics.py")
    }
    _write_json(run_dir / "protocol.json", {
        **campaign, "frozen_at": _now(), "code_sha256": code_hashes,
    })
    _write_json(run_dir / "budget.json", budget.snapshot())
    _write_json(run_dir / "state.json", {"status": "preflight", "at": _now()})
    ledger = UsageLedger()
    try:
        # Authenticate before reserving any generation spend. Tokens stay in memory.
        token = token_provider()
        if not isinstance(token, str) or not token.strip():
            raise ValueError("authentication returned no token")
        _audit_deployment(config, run_dir, "before")
        payload = {"model": config["deployment"], "input": campaign["calls"][0]["prompt"]}
        counter = FoundryInputTokenCounter(config["endpoint"], model=config["deployment"],
                                          token_provider=lambda: token)
        try:
            count = count_probe(payload) if count_probe is not None else counter.count_payload(payload)
        except FoundryCountError as exc:
            # Only unsupported capability is eligible for diagnostic continuation.
            message = str(exc)
            if "not supported" not in message.lower() and "unsupported" not in message.lower():
                raise
            _write_json(run_dir / "input_count.json", {
                "available": False, "error": message, "exact_input_baseline": None,
                "promotion_eligible": False,
            })
        else:
            _write_json(run_dir / "input_count.json", {
                "available": True, "preflight_first_request_tokens": count.input_tokens,
                "probe_payload": payload,
                "note": "Input-only capability probe; generation settings omitted. "
                "Not an exact generation-request baseline.",
            })
        for call in campaign["calls"]:
            if call["split"] == "holdout" and predictions is None:
                _audit_deployment(config, run_dir, "before-holdout")
                predictions = _freeze_predictions(campaign, rows, run_dir)
            call_id = call["call_id"]
            output_cap = call["quote"]["planned_output_tokens"]
            reserve = budget.reserve(call_id, config["max_input_tokens_per_call"],
                                     2 * output_cap)
            _write_json(run_dir / "budget.json", budget.snapshot())
            row = {**call, "started_at": _now(), "status": "reserved",
                   "reservation_usd": reserve, "usage": None, "rated_cost_usd": None,
                   "quality": {"contract_passed": False, "human_review_required": True,
                               "errors": ["not completed"]}}
            rows.append(row)
            _write_json(run_dir / "records.json", rows)
            _write_json(run_dir / "state.json", {
                "status": "dispatching", "call_id": call_id, "at": _now(),
            })

            def audited_transport(url: str, headers: dict, body: bytes,
                                  timeout: float) -> tuple[int, bytes]:
                status, raw = transport(url, headers, body, timeout)
                # Never persist headers: they contain the bearer token.
                (run_dir / f"{call_id}.request.json").write_bytes(body)
                (run_dir / f"{call_id}.response.json").write_bytes(raw)
                row["http_status"] = status
                if 200 <= status < 300:
                    _require_measured_channels(raw)
                return status, raw

            dispatcher = _dispatcher(config, output_cap, lambda: token,
                                     audited_transport, ledger)
            try:
                result = dispatcher(call["prompt"], target=call_id)
            except FoundryDispatchError as exc:
                row.update(status="failed", error=str(exc))
                raise
            else:
                row.update(output=result.output, response_model=result.model,
                           status="completed",
                           quality=check_output(projects[call["project_id"]],
                                                tuple(call["operations"]), result.output))
            finally:
                try:
                    measured = ledger.calls(call_id)
                    if measured:
                        usage = ledger.totals()[call_id]
                        row["usage"] = asdict(usage)
                        row["response_calls"] = [asdict(item) for item in measured]
                        row["rated_cost_usd"] = pricing.attempt_cost(
                            input_tokens=usage.input_tokens, cached_input_tokens=usage.cached_tokens,
                            output_tokens=usage.output_tokens)
                        budget.settle(call_id, row["usage"], row["rated_cost_usd"])
                except (ValueError, RuntimeError) as exc:
                    row.update(status="settlement_failed", error=str(exc))
                    raise
                finally:
                    row["finished_at"] = _now()
                    _write_json(run_dir / "records.json", rows)
                    _write_json(run_dir / "budget.json", budget.snapshot())
            if result.model != config["expected_response_model"]:
                row["status"] = "identity_mismatch"
                _write_json(run_dir / "records.json", rows)
                raise RuntimeError("response model differs from pinned model; campaign stopped")
            if (result.usage.input_tokens > config["max_input_tokens_per_call"]
                    or result.usage.output_tokens > output_cap):
                row["status"] = "reservation_bound_violation"
                _write_json(run_dir / "records.json", rows)
                raise RuntimeError("provider usage exceeded reservation bounds")
        _audit_deployment(config, run_dir, "after")
        from .customer_models import evaluate_predictions

        if predictions is None:
            raise ValueError("campaign has no holdout prediction phase")
        evaluation = evaluate_predictions(
            [row for row in rows if row["split"] == "holdout"],
            predictions["predictions"])
        summary = {
            "status": "complete_exploratory", "calls": len(rows),
            "total_rated_cost_usd": sum(row["rated_cost_usd"] for row in rows),
            "prior_attempt": config["prior_attempt"],
            "cumulative_rated_cost_usd": sum(row["rated_cost_usd"] for row in rows)
            + config["prior_attempt"]["rated_cost_usd"],
            "contract_passed_calls": sum(row["quality"]["contract_passed"] for row in rows),
            "selected_form": predictions["selected_form"],
            "holdout": evaluation, "batching": summarize_batching(campaign, rows),
            "budget": budget.snapshot(), "limitations": campaign["limitations"],
        }
        _write_json(run_dir / "analysis.json", summary)
        _write_json(run_dir / "state.json", {"status": "complete_exploratory", "at": _now()})
        return summary
    except (FoundryDispatchError, FoundryCountError, OSError, ValueError, RuntimeError) as exc:
        _write_json(run_dir / "state.json", {
            "status": "halted", "error_type": type(exc).__name__, "error": str(exc),
            "at": _now(), "no_automatic_retry": True,
        })
        raise
