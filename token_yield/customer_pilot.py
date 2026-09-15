"""Frozen, capped scoping pilot over authentic public customer requests.

No network access occurs during planning. Execution is single-use per run
directory: ambiguous attempts are never retried automatically. Retail-rate
costs and conservative spending reservations are retained separately.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
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
    load_catalog, load_template, quote_features, validate_project,
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
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            # Windows scanners can briefly lock snapshots; never retry an API request.
            time.sleep(0.05 * (attempt + 1))


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
    return _validate_config(config)


def _validate_config(config: dict) -> dict:
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
    return _prepare_campaign(catalog, template, config)


def _prepare_campaign(catalog: dict, template: dict, config: dict) -> dict:
    projects = catalog["projects"]
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
    prospective = campaign.get("prospective")
    if prospective is not None:
        _validate_prospective(campaign, run_dir)
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
    if prospective is not None:
        _write_json(run_dir / "models.json", prospective["model"])
        predictions = prospective["predictions"]
        _write_json(run_dir / "predictions.json", {
            "frozen_at": _now(), "model_sha256": prospective["model_sha256"],
            "catalog_sha256": campaign["catalog_sha256"], "predictions": predictions,
        })
        _prospective_summary(campaign, rows, run_dir, "preflight")
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
                if prospective is not None:
                    expected_payload = {
                        "model": config["deployment"], "input": call["prompt"],
                        "max_output_tokens": output_cap,
                        "reasoning": {"effort": config["reasoning_effort"]},
                        "text": {"verbosity": config["text_verbosity"]},
                    }
                    if (json.loads(body) != expected_payload
                            or url != config["endpoint"] + "/responses"):
                        raise ValueError("request differs from frozen no-tool payload")
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
        if prospective is not None:
            summary = _prospective_summary(campaign, rows, run_dir, "completed")
            _write_json(run_dir / "analysis.json", summary)
            _write_json(run_dir / "state.json", {"status": summary["status"], "at": _now()})
            return summary
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
    except (FoundryDispatchError, FoundryCountError, OSError, ValueError, RuntimeError,
            KeyboardInterrupt) as exc:
        if prospective is not None:
            _prospective_summary(campaign, rows, run_dir, "halted", error=str(exc))
        _write_json(run_dir / "state.json", {
            "status": "halted", "error_type": type(exc).__name__, "error": str(exc),
            "at": _now(), "no_automatic_retry": True,
        })
        raise


def _project_quote(calls: list[dict]) -> dict:
    quotes = [call["quote"] for call in calls]
    return {
        "context_bytes": quotes[0]["context_bytes"],
        "prompt_bytes": sum(value["prompt_bytes"] for value in quotes),
        "planned_output_tokens": sum(value["planned_output_tokens"] for value in quotes),
        "counts": {name: sum(value["counts"][name] for value in quotes)
                   for name in SCOPING_BRICKS},
        "planned_calls": len(calls),
        "max_operations_per_call": max(len(call["operations"]) for call in calls),
    }


def _runtime_compatibility(model: dict, campaign: dict) -> dict:
    """Audit unchanged generation components without claiming whole-runtime identity."""
    original = model["runtime"]
    hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in original["execution_code_sha256"]
    }
    critical = ("customer_decomposition.py", "foundry_dispatch.py", "foundry_count.py",
                "budget.py", "economics.py", "robust.py")
    matches = {name: hashes.get(name) == original["execution_code_sha256"].get(name)
               for name in critical}
    if not all(matches.values()):
        raise ValueError("critical execution component differs from frozen runtime")
    function_matches = {}
    verification_note = "Historical controller source did not match the saved execution hash."
    source = Path(__file__)
    try:
        historical = subprocess.run(
            ["git", "show", "HEAD:token_yield/customer_pilot.py"],
            cwd=source.parents[1], capture_output=True, check=True, timeout=10,
        ).stdout
        # Git stores LF; the execution snapshot may have been checked out as CRLF.
        variants = (historical, historical.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        expected = original["execution_code_sha256"]["customer_pilot.py"]
        baseline = next((raw for raw in variants
                         if hashlib.sha256(raw).hexdigest() == expected), None)
        if baseline is not None:
            def functions(raw: bytes) -> dict:
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}

            before, after = functions(baseline), functions(source.read_bytes())
            function_matches = {name: before.get(name) == after.get(name) for name in (
                "_dispatcher", "_default_transport", "_require_measured_channels",
                "_audit_deployment", "_deployment_probe",
            )}
            if not all(function_matches.values()):
                raise ValueError("critical pilot dispatch function differs from frozen runtime")
            verification_note = "Critical controller functions match the historical source."
    except (OSError, subprocess.SubprocessError) as exc:
        verification_note = f"Historical controller verification unavailable: {exc}"
    config = campaign["config"]
    actual_runtime = {
        "deployment": config["deployment"], "deployment_version": config["deployment_version"],
        "reasoning_effort": config["reasoning_effort"], "text_verbosity": config["text_verbosity"],
        "template_sha256": campaign["template_sha256"], "execution_code_sha256": hashes,
        "execution_policy": config["authorization"],
        "max_input_tokens_per_call": config["max_input_tokens_per_call"],
        "pricing": config["pricing"], "pricing_source": config["pricing_source"],
    }
    return {
        "status": "controller_revision_transfer_evaluation",
        "exact_runtime_identity": actual_runtime == original,
        "critical_file_matches": matches, "critical_controller_function_matches": function_matches,
        "historical_controller_verified": bool(function_matches),
        "historical_controller_verification_note": verification_note,
        "controller_code_sha256": hashes, "actual_runtime": actual_runtime,
        "scope": "Unchanged compiler/dispatcher/safety components; revised evaluation controller "
                 "and approved input catalog. This is not exact original-runtime identity.",
    }


def _load_prospective_catalog(path: Path) -> tuple[dict, dict]:
    """Keep approved research metadata outside the unchanged compiler input."""
    catalog = json.loads(path.read_text(encoding="utf-8"))
    metadata = {key: catalog.pop(key) for key in (
        "scope", "selection_notes", "rejected_projects") if key in catalog}
    notes = metadata.get("selection_notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("selection_notes must be a list of research caveats")
    if (set(catalog) != {"schema_version", "verified_on", "projects"}
            or catalog["schema_version"] != "customer-requests-v1"
            or not isinstance(catalog["verified_on"], str)
            or not catalog["verified_on"].strip()
            or not isinstance(catalog["projects"], list)):
        raise ValueError("unsupported prospective catalog fields or schema")
    project_metadata, seen = {}, set()
    for project in catalog["projects"]:
        extra = {key: project.pop(key) for key in (
            "compatibility", "source_sections") if key in project}
        validate_project(project)
        if project["id"] in seen:
            raise ValueError("duplicate project id")
        seen.add(project["id"])
        if extra:
            project_metadata[project["id"]] = extra
    return catalog, {**metadata, "projects": project_metadata}


def prepare_prospective_campaign(model_run: Path, catalog_path: Path) -> dict:
    """Freeze all project/channel/form forecasts offline; never fit on new labels."""
    from .customer_models import _project_prediction, forecast_project
    from urllib.parse import urlsplit, urlunsplit

    model_run, catalog_path = model_run.resolve(), catalog_path.resolve()
    model_raw = (model_run / "models.json").read_bytes()
    model = json.loads(model_raw)
    if model.get("schema_version") != "customer-project-model-v1":
        raise ValueError("test-model requires a frozen project model")
    source_name = model["provenance"]["source_run"]
    if Path(source_name).name != source_name:
        raise ValueError("source run must be a sibling run name")
    source_run = model_run.parent / source_name
    protocol_raw = (source_run / "protocol.json").read_bytes()
    if hashlib.sha256(protocol_raw).hexdigest() != model["provenance"]["source_file_sha256"]["protocol.json"]:
        raise ValueError("source protocol hash mismatch")
    source = json.loads(protocol_raw)
    for section in ("catalog", "template", "config"):
        if content_hash(source[section]) != source[f"{section}_sha256"]:
            raise ValueError(f"source {section} hash mismatch")
    config, template = copy.deepcopy(source["config"]), source["template"]
    _validate_config(config)
    runtime = model["runtime"]
    if set(runtime) != {
        "deployment", "deployment_version", "reasoning_effort", "text_verbosity",
        "template_sha256", "execution_code_sha256", "execution_policy",
        "max_input_tokens_per_call", "pricing", "pricing_source",
    }:
        raise ValueError("unknown frozen model runtime settings")
    if (runtime["execution_code_sha256"] != source["code_sha256"]
            or runtime["template_sha256"] != source["template_sha256"]
            or runtime["execution_policy"] != config["authorization"]
            or any(runtime[key] != config[key] for key in (
                "deployment", "deployment_version", "reasoning_effort", "text_verbosity",
                "max_input_tokens_per_call", "pricing", "pricing_source"))
            or template["max_output_tokens_per_operation"] != 1600):
        raise ValueError("model settings differ from frozen source execution")
    if set(runtime["execution_code_sha256"]) != {
        "customer_pilot.py", "customer_decomposition.py", "customer_models.py",
        "robust.py", "foundry_dispatch.py", "foundry_count.py", "budget.py", "economics.py",
    }:
        raise ValueError("unknown execution components")
    catalog, source_metadata = _load_prospective_catalog(catalog_path)
    projects = catalog["projects"]
    if not 4 <= len(projects) <= 6 or any(p["split"] != "holdout" for p in projects):
        raise ValueError("prospective testing requires four to six all-holdout projects")

    def normalized_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                           parts.path.rstrip("/"), parts.query, ""))

    old_ids = {p["id"] for p in source["catalog"]["projects"]} | set(model["training_project_ids"])
    old_urls = {normalized_url(p["source_url"]) for p in source["catalog"]["projects"]}
    new_urls = [normalized_url(p["source_url"]) for p in projects]
    if (old_ids & {p["id"] for p in projects} or old_urls & set(new_urls)
            or len(set(new_urls)) != len(new_urls)):
        raise ValueError("prospective projects require new, distinct IDs and source URLs")
    config.update(cap_usd=10.0, stop_usd=9.5, prior_attempt={
        "run": "separate_prospective_approval", "rated_cost_usd": 0.0,
        "settled_safety_usd": 0.0, "active_reserved_usd": 0,
    })
    config["authorization"]["inputs"] = (
        "Only this frozen prospective public-request catalog and the unchanged scoping template.")
    _validate_config(config)
    campaign = _prepare_campaign(catalog, template, config)
    compatibility = _runtime_compatibility(model, campaign)
    predictions = []
    for project in projects:
        for repeat in range(config["replicates"]):
            for arm in ("split", "batched"):
                calls = [call for call in campaign["calls"] if (
                    call["project_id"], call["replicate"], call["arm"]
                ) == (project["id"], repeat, arm)]
                quote = _project_quote(calls)
                actual = forecast_project(model, quote, compatibility["actual_runtime"])
                conditional = forecast_project(model, quote, runtime)
                support = actual["support"]
                if support["status"] == "unsupported":
                    support["reasons"] += conditional["support"]["reasons"]
                predictions.append({
                    "observation_id": f"{project['id']}-{repeat}-{arm}",
                    "project_id": project["id"], "arm": arm, "replicate": repeat, "quote": quote,
                    "predicted": conditional["point_estimate"],
                    "predictions_by_form": {
                        target: {form: _project_prediction(channel, quote, form)
                                 for form in channel["forms"]}
                        for target, channel in model["channels"].items()
                    },
                    "support": support,
                    "prediction_basis": "conditional_original_runtime_transfer_forecast",
                    "actual_runtime_forecast": actual,
                })
    campaign["limitations"] = [
        item for item in campaign["limitations"]
        if not item.startswith("Four train")
    ] + model["limitations"] + [
        "New source holdout, same shared scoping template; not full customer delivery.",
        "Project IDs and buyer labels are not independent organizations. Requests from "
        "the same buyer and a shared template can have correlated errors.",
        "Legacy requirement brick tags describe proposed deliverables, not executed operations. "
        "Only extract/classify/plan/report scoping projections are executed and evaluated.",
        "Weights and selected forms remain frozen; no calibration or prediction intervals.",
        "Controller/input-policy revisions prevent exact runtime identity; scores evaluate "
        "conditional original-runtime forecasts as an explicitly labeled transfer test.",
        "No source fetching, tools, retries, or paid generation occurs during preview.",
    ] + source_metadata.get("selection_notes", [])
    campaign["prospective"] = {
        "source_model_run": str(model_run), "source_measured_run": str(source_run),
        "catalog_path": str(catalog_path),
        "source_catalog_metadata": source_metadata,
        "model_sha256": hashlib.sha256(model_raw).hexdigest(),
        "catalog_file_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "model": model, "compatibility": compatibility, "predictions": predictions,
        "selected_forms": {target: channel["selected_form"]
                           for target, channel in model["channels"].items()},
    }
    campaign["prospective"]["campaign_sha256"] = content_hash(campaign)
    return campaign


def _validate_prospective(campaign: dict, run_dir: Path) -> None:
    prospective = campaign["prospective"]
    for root in ("source_model_run", "source_measured_run"):
        source = Path(prospective[root]).resolve()
        if run_dir.resolve() == source or source in run_dir.resolve().parents:
            raise ValueError("prospective output must be outside frozen source runs")
    frozen = copy.deepcopy(campaign)
    expected = frozen["prospective"].pop("campaign_sha256")
    if content_hash(frozen) != expected:
        raise ValueError("prospective campaign changed after prediction freeze")
    current = prepare_prospective_campaign(
        Path(prospective["source_model_run"]), Path(prospective["catalog_path"]))
    if canonical_json(current) != canonical_json(campaign):
        raise ValueError("frozen model, catalog, or execution code changed after preview")


def _prospective_summary(campaign: dict, rows: list[dict], run_dir: Path,
                         status: str, *, error: str | None = None) -> dict:
    from .customer_models import evaluate_project_models

    frozen = campaign["prospective"]
    observations, scores = [], {}
    if status == "completed":
        grouped = aggregate_project_rows(campaign, rows)
        indexed = {row["call_id"]: row for row in grouped}
        for prediction in frozen["predictions"]:
            row = indexed[prediction["observation_id"]]
            if row["quote"] != prediction["quote"]:
                raise ValueError("observed quote differs from pre-execution prediction")
            observations.append({
                **prediction, "actual": {
                    **{name: row["usage"][name] for name in (
                        "input_tokens", "output_tokens", "total_tokens")},
                    "rated_cost_usd": row["rated_cost_usd"],
                }, "contract_passed": row["contract_passed"],
            })
        scores = evaluate_project_models(frozen["model"], grouped)["scores"]
        _write_json(run_dir / "project_records.json", grouped)
    summary = {
        "status": status, "method": "prospective_frozen_model_transfer_evaluation",
        "model_sha256": frozen["model_sha256"], "catalog_sha256": campaign["catalog_sha256"],
        "catalog_file_sha256": frozen["catalog_file_sha256"],
        "source_model_run": Path(frozen["source_model_run"]).name,
        "n_projects": len(campaign["catalog"]["projects"]), "n_observations": len(observations),
        "planned_observations": len(frozen["predictions"]),
        "selected_forms": frozen["selected_forms"], "scores": scores, "observations": observations,
        "totals": {
            "rated_cost_usd": sum(row["rated_cost_usd"] or 0 for row in rows),
            **{name: sum((row.get("usage") or {}).get(name, 0) for row in rows)
               for name in ("input_tokens", "output_tokens")},
        },
        "runtime_compatibility": frozen["compatibility"],
        "limitations": campaign["limitations"], "error": error,
        "spend_scope": "Metered attempts only; budget.json retains uncertain reservations.",
    }
    _write_json(run_dir / "prospective_evaluation.json", summary)
    return summary


def aggregate_project_rows(campaign: dict, rows: list[dict]) -> list[dict]:
    """Combine complete declared executions, never partial or successful-only costs."""
    from .customer_models import _completed_rows

    if campaign.get("schema_version") != "customer-campaign-v1":
        raise ValueError("unsupported customer campaign schema")
    for section in ("catalog", "template", "config"):
        if content_hash(campaign[section]) != campaign[f"{section}_sha256"]:
            raise ValueError(f"frozen {section} hash mismatch")
    completed, excluded = _completed_rows(rows, training=False)
    if excluded:
        raise ValueError(f"incomplete project executions cannot be training targets: {excluded}")
    actual = {row["call_id"]: row for row in completed}
    declared = {call["call_id"]: call for call in campaign["calls"]}
    if len(declared) != len(campaign["calls"]) or set(actual) != set(declared):
        raise ValueError("measured call IDs must exactly match the frozen campaign")
    projects = {project["id"]: project for project in campaign["catalog"]["projects"]}
    groups: dict[tuple, list[dict]] = {}
    pricing = Pricing(**campaign["config"]["pricing"])
    for call_id, call in declared.items():
        project = projects[call["project_id"]]
        if type(call["replicate"]) is not int or call["replicate"] < 0:
            raise ValueError("replicate must be a nonnegative integer")
        batches = arm_batches(call["arm"])
        operations = tuple(call["operations"])
        if operations not in batches:
            raise ValueError("operations differ from declared arm")
        expected_id = f"{project['id']}-{call['replicate']}-{call['arm']}-{batches.index(operations)}"
        if call_id != expected_id or call["split"] != project["split"]:
            raise ValueError("call identity or split differs from frozen project")
        expected_quote = quote_features(project, campaign["template"], operations)
        prompt = build_prompt(project, campaign["template"], operations)
        if (canonical_json(call["quote"]) != canonical_json(expected_quote)
                or call["prompt"] != prompt
                or call["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                or call["plan_sha256"] != compile_plan(project, campaign["template"]).sha256):
            raise ValueError("declared call differs from frozen quote, prompt, or plan")
        row = actual[call_id]
        if any(canonical_json(row.get(key)) != canonical_json(value)
               for key, value in call.items()):
            raise ValueError(f"measurement differs from frozen call: {call_id}")
        if row.get("response_model") != campaign["config"]["expected_response_model"]:
            raise ValueError("measured model differs from frozen runtime")
        quality = row.get("quality", {}).get("contract_passed")
        if type(quality) is not bool:
            raise ValueError("measured contract_passed must be boolean")
        usage = row["usage"]
        rated = pricing.attempt_cost(
            input_tokens=usage["input_tokens"], cached_input_tokens=usage["cached_tokens"],
            output_tokens=usage["output_tokens"])
        if not math.isclose(rated, row["rated_cost_usd"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("recorded cost does not reconcile with usage and frozen rates")
        if (usage["output_tokens"] > expected_quote["planned_output_tokens"]
                or usage["input_tokens"] > campaign["config"]["max_input_tokens_per_call"]):
            raise ValueError("measured usage exceeds declared execution limits")
        key = (project["id"], call["arm"], call["replicate"])
        groups.setdefault(key, []).append(call)
    expected_groups = {
        (project_id, arm, replicate) for project_id in projects
        for arm in ("split", "batched")
        for replicate in range(campaign["config"]["replicates"])
    }
    if set(groups) != expected_groups:
        raise ValueError("campaign is missing complete project/arm/replicate groups")
    result = []
    for (project_id, arm, replicate), calls in sorted(groups.items()):
        calls = sorted(calls, key=lambda call: call["call_id"])
        if sorted(tuple(call["operations"]) for call in calls) != sorted(arm_batches(arm)):
            raise ValueError("project group must execute every declared operation exactly once")
        measurements = [actual[call["call_id"]] for call in calls]
        project = projects[project_id]
        quote = _project_quote(calls)
        result.append({
            "call_id": f"{project_id}-{replicate}-{arm}",
            "project_id": project_id, "split": project["split"],
            "arm": arm, "replicate": replicate, "status": "completed",
            "source_url": project["source_url"], "quote": quote,
            "source_call_ids": [call["call_id"] for call in calls],
            "usage": {name: sum(row["usage"][name] for row in measurements)
                      for name in ("input_tokens", "output_tokens", "total_tokens",
                                   "cached_tokens", "reasoning_tokens")},
            "rated_cost_usd": sum(row["rated_cost_usd"] for row in measurements),
            "contract_passed": all(row["quality"]["contract_passed"] for row in measurements),
            "human_quality_accepted": None,
            "label_provenance": "provider_usage_and_price_derived_cost",
        })
    return result


def _verify_training_responses(source_run: Path, campaign: dict, rows: list[dict]) -> dict:
    hashes = {}
    for row in rows:
        call_id = row["call_id"]
        request_path = source_run / f"{call_id}.request.json"
        response_path = source_run / f"{call_id}.response.json"
        request_raw, response_raw = request_path.read_bytes(), response_path.read_bytes()
        _require_measured_channels(response_raw)
        request, response = json.loads(request_raw), json.loads(response_raw)
        if (request["input"] != row["prompt"]
                or request["model"] != campaign["config"]["deployment"]
                or request["max_output_tokens"] != row["quote"]["planned_output_tokens"]
                or request.get("tools")
                or request["reasoning"]["effort"] != campaign["config"]["reasoning_effort"]
                or request["text"]["verbosity"] != campaign["config"]["text_verbosity"]):
            raise ValueError(f"raw request differs from frozen execution: {call_id}")
        usage = response["usage"]
        observed = {
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_tokens": usage["input_tokens_details"]["cached_tokens"],
            "reasoning_tokens": usage["output_tokens_details"]["reasoning_tokens"],
        }
        if (canonical_json(observed) != canonical_json(row["usage"])
                or response["model"] != row["response_model"]
                or response["status"] != "completed"):
            raise ValueError(f"raw response differs from recorded execution: {call_id}")
        ledger = row["response_calls"]
        if len(ledger) != 1:
            raise ValueError("scoping execution must contain exactly one response per call")
        for path, raw, field in (
            (request_path, request_raw, "request_sha256"),
            (response_path, response_raw, "response_sha256"),
        ):
            digest = hashlib.sha256(raw).hexdigest()
            if digest != ledger[0][field]:
                raise ValueError(f"response ledger hash mismatch: {path.name}")
            hashes[path.name] = digest
    return hashes


def train_project_forecast(source_run: Path, run_dir: Path) -> dict:
    """Train offline from existing measured calls, preserving all historical files."""
    from .customer_models import evaluate_project_models, fit_project_models, forecast_project

    source_run, run_dir = source_run.resolve(), run_dir.resolve()
    if run_dir == source_run or source_run in run_dir.parents:
        raise ValueError("training output must be outside the frozen source run")
    if run_dir.exists():
        raise FileExistsError(f"training output already exists: {run_dir}")
    raw = {name: (source_run / name).read_bytes() for name in ("protocol.json", "records.json")}
    campaign, measured = json.loads(raw["protocol.json"]), json.loads(raw["records.json"])
    rows = aggregate_project_rows(campaign, measured)
    hashes = _verify_training_responses(source_run, campaign, measured)
    hashes.update({name: hashlib.sha256(value).hexdigest() for name, value in raw.items()})
    config = campaign["config"]
    runtime = {
        "deployment": config["deployment"], "deployment_version": config["deployment_version"],
        "reasoning_effort": config["reasoning_effort"], "text_verbosity": config["text_verbosity"],
        "template_sha256": campaign["template_sha256"],
        "execution_code_sha256": campaign["code_sha256"],
        "execution_policy": config["authorization"],
        "max_input_tokens_per_call": config["max_input_tokens_per_call"],
        "pricing": config["pricing"], "pricing_source": config["pricing_source"],
    }
    training = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    models = fit_project_models(training, runtime)
    models["provenance"] = {
        "source_run": source_run.name, "source_file_sha256": hashes,
        "catalog_sha256": campaign["catalog_sha256"],
        "training_code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("customer_models.py", "customer_pilot.py", "customer_decomposition.py",
                         "robust.py", "economics.py")
        },
    }
    predictions = [
        {"observation_id": row["call_id"], "project_id": row["project_id"],
         "split": row["split"], "arm": row["arm"],
         "evaluation_role": "in_sample" if row["split"] == "train" else "historical_holdout",
         **forecast_project(models, row["quote"], runtime)}
        for row in rows
    ]
    evaluation = evaluate_project_models(models, holdout)
    summary = {
        "status": "trained_exploratory", "source_run": source_run.name,
        "measured_calls": len(measured), "project_observations": len(rows),
        "training_projects": len(models["training_project_ids"]),
        "holdout_projects": evaluation["n_projects"],
        "training_observations": len(training), "holdout_observations": len(holdout),
        "selected_forms": evaluation["selected_forms"],
        "holdout": evaluation, "spent_usd": 0,
        "scope": models["scope"], "calibration_status": models["calibration_status"],
        "calibrated_interval": None, "upper_budget_bound": None,
        "total_recorded_rated_cost_usd": sum(row["rated_cost_usd"] for row in rows),
        "contract_passed_project_observations": sum(row["contract_passed"] for row in rows),
        "limitations": campaign["limitations"] + models["limitations"] + [
            "Historical holdout results were previously examined; this is retrospective evaluation.",
            "Training CV chooses the form; its winning CV error is not an unbiased new-test estimate.",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("project_records.json", rows), ("models.json", models),
                        ("predictions.json", predictions), ("analysis.json", summary)):
        _write_json(run_dir / name, value)
    return summary
