"""Measured composition regression with an untouched, unseen-combination test set."""

from __future__ import annotations

import json
import math
import random
import uuid
from copy import deepcopy
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .marketplace_agent_contracts import (
    ATOMS, MAX_COMPOSITION_STEPS, canonical, composition_contract, contracts,
    fingerprint, numeric_features, source_documents, workload_prompt,
    validate_message,
)

PROTOCOL = "compact-keyed-composition-generalization-v2"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
FEATURES = (
    *ATOMS, "prompt_bytes", "source_bytes", "document_count", "component_count",
    "shared_operations", "handoffs",
    *(f"position_{atom}" for atom in ATOMS),
)
GATES = {
    "input_mape": 0.15, "output_mape": 0.30,
    "per_subtype_input_mape": 0.25, "per_subtype_output_mape": 0.45,
    "baseline_improvement": 0.10,
}


def selected_contract(ids: list[str]) -> dict:
    catalog = {item["id"]: item for item in contracts()}
    if (not isinstance(ids, list) or not 2 <= len(ids) <= len(catalog)
            or any(not isinstance(item, str) or item not in catalog for item in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Select two or more distinct known marketplace subtype IDs")
    return composition_contract([{**catalog[item], "quantity": 1} for item in ids])


def prompt_for(brick: dict, docs: list[dict]) -> str:
    payload = json.loads(workload_prompt(brick, docs))
    if brick.get("steps"):
        payload["output_contract"]["atom_results"] = {
            step["id"]: "source-only step result, at most 8 words; state missing evidence explicitly"
            for step in brick["steps"]
        }
    return canonical(payload)


def features_for(brick: dict, docs: list[dict]) -> list[float]:
    values = numeric_features(brick, docs)
    values["prompt_bytes"] = len(prompt_for(brick, docs).encode("utf-8"))
    steps = brick.get("steps", [])
    count = len(brick.get("component_contracts", [])) or 1
    values.update(component_count=count, handoffs=count - 1,
                  shared_operations=sum(max(0, n - 1) for n in brick["atoms"].values()))
    for atom in ATOMS:
        values[f"position_{atom}"] = sum(
            (index + 1) / len(steps) for index, step in enumerate(steps)
            if step["operation"] == atom
        ) if steps else 0
    return [float(values[name]) for name in FEATURES]


def dataset_plan(previous: dict | None = None) -> dict:
    """Freeze identities without seeing outcomes; held-out memberships are novel."""
    ids = [item["id"] for item in contracts()]
    seed = previous["seed"] + 1 if previous else 20260921
    rng = random.Random(seed)
    sizes = {"train": [2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8],
             "holdout": [2, 2, 3, 3, 4, 5, 6, 7]}
    for _ in range(1000):
        excluded = {tuple(sorted(item["ids"])) for items in previous["groups"].values()
                    for item in items} if previous else set()
        if previous:
            excluded.update(tuple(ids) for ids in previous.get("excluded_memberships", []))
        groups, seen = {}, set(excluded)
        for split, counts in sizes.items():
            if previous and split == "train":
                groups[split] = previous["groups"]["train"]
                continue
            chosen = []
            for count in counts:
                for _ in range(10000):
                    members = rng.sample(ids, count)
                    identity = tuple(sorted(members))
                    if identity in seen:
                        continue
                    try:
                        brick = selected_contract(members)
                    except ValueError:
                        continue
                    seen.add(identity)
                    chosen.append({"ids": members, "contract": brick})
                    break
                else:
                    raise RuntimeError("Could not construct bounded composition design")
            groups[split] = chosen
        if all(set(ids) == {member for item in items for member in item["ids"]}
               for items in groups.values()):
            return {"protocol": PROTOCOL, "seed": seed, "groups": groups,
                    "features": list(FEATURES), "gates": GATES, "alphas": list(ALPHAS),
                    "target_transforms": ["identity", "log1p"],
                    "reuse_training": bool(previous), "holdout_repeats": 1 if previous else 2,
                    "excluded_memberships": [list(ids) for ids in sorted(excluded)],
                    "max_steps": MAX_COMPOSITION_STEPS,
                    "labels": "Measured provider usage only; no ROI or synthetic token labels",
                    "split": "Disjoint subtype combinations, including reversed memberships; "
                             "fresh held-out documents dispatched only after parameter freeze"}
    raise RuntimeError("Composition design did not cover every subtype in both splits")


def _fit(rows: list[dict], alpha: float, target: str, transform: str) -> dict:
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Only training rows may fit preprocessing or regressions")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    labels = [row[target] for row in rows]
    model.fit([row["x"] for row in rows], np.log1p(labels) if transform == "log1p" else labels)
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    return {"target_transform": transform, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}


def _predict(model: dict, vector: list[float]) -> float:
    if not len(vector) == len(FEATURES) == len(model["coefficients"]) == len(model["mean"]) == len(model["scale"]):
        raise ValueError("Composition feature schema mismatch")
    value = model["intercept"] + sum(
        ((x - mean) / scale) * coefficient
        for x, mean, scale, coefficient in zip(
            vector, model["mean"], model["scale"], model["coefficients"])
    )
    logarithmic = model.get("target_transform", "log1p") == "log1p"
    if not math.isfinite(value) or (logarithmic and value > 20):
        raise ValueError("Composition regression exceeds numeric support")
    prediction = math.expm1(value) if logarithmic else value
    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("Composition regression produced an invalid token estimate")
    return prediction


def fit_training(rows: list[dict]) -> dict:
    """Tune on held-out training combinations; never on the acceptance holdout."""
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Acceptance holdouts cannot enter training or tuning")
    groups = sorted({row["group"] for row in rows if len(row["ids"]) > 1})
    if len(groups) < 4:
        raise ValueError("At least four measured training combinations are required")
    candidates = []
    for transform, alpha in [(transform, alpha) for transform in ("identity", "log1p") for alpha in ALPHAS]:
        residuals = {"input": [], "output": []}
        for fold in range(4):
            excluded = set(groups[fold::4])
            train = [row for row in rows if row["group"] not in excluded]
            test = [row for row in rows if row["group"] in excluded]
            for target in residuals:
                model = _fit(train, alpha, target, transform)
                residuals[target].extend(
                    abs(_predict(model, row["x"]) - row[target]) / max(1, row[target])
                    for row in test
                )
        candidates.append({"alpha": alpha, "target_transform": transform, **{
            target + "_mape": float(np.mean(errors)) for target, errors in residuals.items()
        }})
    chosen = min(candidates, key=lambda item: item["input_mape"] + item["output_mape"])
    return {"protocol": PROTOCOL, "features": list(FEATURES), "alpha": chosen["alpha"],
            "tuning": candidates, "models": {
                target: _fit(rows, chosen["alpha"], target, chosen["target_transform"]) for target in ("input", "output")
            }, "train_ids": [row["id"] for row in rows],
            "train_groups": groups, "holdout_used_for_selection": False}


def evaluate(model: dict, train: list[dict], holdout: list[dict]) -> dict:
    if not holdout or any(row["split"] != "holdout" for row in holdout):
        raise ValueError("An untouched acceptance holdout is required")
    train_memberships = {tuple(sorted(row["ids"])) for row in train}
    if any(tuple(sorted(row["ids"])) in train_memberships for row in holdout):
        raise ValueError("Acceptance combination leaked into training")
    wanted = {item["id"] for item in contracts()}
    coverage = {member for row in holdout for member in row["ids"]}
    if coverage != wanted:
        raise ValueError("Acceptance holdout must exercise every marketplace subtype")
    predictions = [{
        "id": row["id"], "ids": row["ids"],
        **{target: _predict(model["models"][target], row["x"]) for target in ("input", "output")},
    } for row in holdout]
    report = {"gates": GATES, "predictions": predictions, "metrics": {}, "per_subtype": {}}
    accepted = True
    for target in ("input", "output"):
        errors = [abs(pred[target] - row[target]) for pred, row in zip(predictions, holdout)]
        relative = [error / max(1, row[target]) for error, row in zip(errors, holdout)]
        # A composition-only training mean is harder to beat than a mostly-singleton mean.
        mean = float(np.mean([row[target] for row in train if len(row["ids"]) > 1]))
        baseline = float(np.mean([abs(mean - row[target]) for row in holdout]))
        mae, mape = float(np.mean(errors)), float(np.mean(relative))
        training_error = float(np.mean([
            abs(_predict(model["models"][target], row["x"]) - row[target])
            / max(1, row[target]) for row in train if len(row["ids"]) > 1
        ]))
        report["metrics"].update({
            target + "_mae": mae, target + "_mape": mape,
            target + "_baseline_mae": baseline, target + "_training_mape": training_error,
            target + "_generalization_gap": mape - training_error,
        })
        accepted &= mape <= GATES[target + "_mape"] and mae <= baseline * (1 - GATES["baseline_improvement"])
        for subtype in sorted(wanted):
            values = [value for value, row in zip(relative, holdout) if subtype in row["ids"]]
            score = float(np.mean(values))
            report["per_subtype"].setdefault(subtype, {})[target + "_mape"] = score
            accepted &= score <= GATES[f"per_subtype_{target}_mape"]
    report.update(accepted=bool(accepted), unseen_combinations=len({row["group"] for row in holdout}),
                  subtypes_tested=len(coverage), refit_on_holdout=False)
    return report


def load_published(runtime) -> dict | None:
    pointer = runtime.state_dir / "composition-current.json"
    if not pointer.exists():
        return None
    entry = json.loads(pointer.read_text(encoding="utf-8"))
    version = entry.get("version", "")
    if len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid composition model pointer")
    model = json.loads((runtime.state_dir / "composition-versions" / f"{version}.json").read_text(encoding="utf-8"))
    if fingerprint(model) != entry["sha256"]:
        raise ValueError("Composition model fingerprint mismatch")
    if (model["compatibility"] != runtime._compatibility or model["protocol"] != PROTOCOL
            or model["catalog_hash"] != fingerprint(contracts())):
        return None
    if not model["evaluation"]["accepted"]:
        raise ValueError("Unaccepted composition model cannot be published")
    expected = "mocked-test-provider" if runtime._dispatch_hook else "measured-foundry"
    if model["source"] != expected:
        raise ValueError("Composition model provider provenance does not match runtime")
    return model


def forecast(runtime, ids: list[str]) -> dict:
    brick = selected_contract(ids)
    model = load_published(runtime)
    if model is None:
        return {"supported": False, "reason": "Marketplace-wide composition training has not passed acceptance.",
                "input_tokens": None, "output_tokens": None}
    vector = features_for(brick, source_documents("train-0", 0))
    predicted = {target: _predict(model["models"][target], vector) for target in ("input", "output")}
    if predicted["input"] > 32768:
        return {"supported": False, "reason": "Predicted workflow exceeds the measured token protocol limits.",
                "input_tokens": None, "output_tokens": None}
    raw_output = predicted["output"]
    predicted["output"] = min(raw_output, 1536)
    return {**brick, "supported": True, "input_tokens": predicted["input"],
            "output_tokens": predicted["output"], "total_tokens": sum(predicted.values()),
            "usd_per_run": runtime._cost(predicted["input"], predicted["output"]),
            "pilot_version": model["version"], "model_id": "gpt-5.4", "source": model["source"],
            "forecast_mode": "measured-composition-model", "standalone_forecasts_summed": False,
            "uncapped_output_tokens": raw_output, "output_token_ceiling": 1536,
            "output_ceiling_applied": raw_output > 1536,
            "reason": ("Output-usage estimate at the configured provider ceiling; complete execution is not guaranteed."
                       if raw_output > 1536 else
                       "Generalizing compact source-only composition pilot; not a production guarantee."),
            "evaluation": model["evaluation"]["metrics"]}


class PublishedMarketplace:
    """Read-only shipped model; no credentials, campaign ledger, or dispatch."""

    _dispatch_hook = None

    def __init__(self, directory: Path):
        self.state_dir = directory
        snapshot = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        self._compatibility = snapshot["compatibility"]
        self.config = {"pricing": snapshot["pricing"]}
        self._catalog = snapshot["catalog"]
        if fingerprint(self._catalog) != snapshot["catalog_sha256"]:
            raise ValueError("Published catalog fingerprint mismatch")
        model = load_published(self)
        if (model is None or self._catalog["source"] != "measured-foundry"
                or self._catalog["composition_model"]["version"] != model["version"]):
            raise ValueError("Published catalog requires its compatible accepted measured model")

    def catalog(self) -> dict:
        return deepcopy(self._catalog)

    def _cost(self, inputs: float, outputs: float) -> float:
        rates = self.config["pricing"]
        return (inputs * rates["input_per_million"] + outputs * rates["output_per_million"]) / 1_000_000

    def forecast_selection(self, selections: list[str]) -> dict:
        brick = selected_contract(selections)
        exact = next((item for item in self._catalog["items"]
                      if item["contract_hash"] == brick["contract_hash"] and item["supported"]), None)
        return deepcopy(exact) if exact else forecast(self, selections)


def reuse_training(runtime, parent_dir, previous: dict) -> list[dict]:
    """Re-audit training evidence; previous acceptance labels remain excluded."""
    parameters = json.loads((parent_dir / "agent-artifacts" / "parameters-before-unseen-holdout.json").read_text(encoding="utf-8"))
    if (parameters["protocol"] != PROTOCOL or parameters["compatibility"] != runtime._compatibility
            or parameters["catalog_hash"] != fingerprint(contracts())
            or parameters["plan_sha256"] != fingerprint(previous) or previous["gates"] != GATES):
        raise ValueError("Parent training protocol is incompatible")
    source = "mocked-test-provider" if runtime._dispatch_hook else "measured-foundry"
    result = json.loads((parent_dir / "result.json").read_text(encoding="utf-8"))
    wanted = {item["contract"]["id"]: item for item in previous["groups"]["train"]}
    rows = []
    for row in result["training"]["rows"]:
        if row["split"] != "train" or len(row["ids"]) == 1:
            continue
        if row["group"] not in wanted or row["id"] not in parameters["train_ids"]:
            raise ValueError("Parent row was not in the frozen training partition")
        brick = {**wanted[row["group"]]["contract"], "result_format": "keyed-steps-v1"}
        evidence_path = (parent_dir / row["evidence"]).resolve()
        if not evidence_path.is_relative_to(parent_dir.parent.resolve()):
            raise ValueError("Reused evidence must remain inside the campaign run directory")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        docs = json.loads(evidence["prompt"])["documents"]
        observed = row["measurement"]
        if (evidence["status"] != "validated" or evidence["kind"] != "workload"
                or evidence["source"] != source or row["source"] != source
                or evidence["model_id"] != "gpt-5.4" or evidence["id"] != row["id"]
                or evidence["prompt"] != prompt_for(brick, docs)
                or fingerprint(evidence["prompt"]) != observed["prompt_sha256"]
                or evidence["usage"] != observed["usage"]
                or evidence["public_output"] != observed["public_output"]
                or row["ids"] != wanted[row["group"]]["ids"]
                or row["x"] != features_for(brick, docs)
                or any(row[target] != evidence["usage"][target + "_tokens"] for target in ("input", "output"))
                or ("evidence_sha256" in row and row["evidence_sha256"] != fingerprint(evidence))):
            raise ValueError("Parent composition evidence or labels changed")
        validate_message("workload", evidence["public_output"], docs, [brick])
        rows.append({**row, "evidence": str(evidence_path), "evidence_sha256": fingerprint(evidence),
                     "reused_from_run": parent_dir.name})
    if (len({row["id"] for row in rows}) != 24
            or Counter(row["group"] for row in rows) != Counter({group: 2 for group in wanted})):
        raise ValueError("All 24 original composition training rows are required")
    return rows


def train_marketplace(runtime, request, run_dir, run_id, call, enter, event, write, cancel) -> dict:
    if runtime.campaign or runtime.source_fixture:
        raise ValueError("Generalizing marketplace training requires its explicitly approved unrestricted pilot")
    source = "mocked-test-provider" if runtime._dispatch_hook else "measured-foundry"
    parent_id = request.get("training_parent_run")
    parent_dir = run_dir.parent / parent_id if parent_id else None
    previous = json.loads((parent_dir / "agent-artifacts" / "generalization-plan.json").read_text(encoding="utf-8")) if parent_dir else None
    plan = dataset_plan(previous)
    artifacts = run_dir / "agent-artifacts"
    write(artifacts / "generalization-plan.json", plan)
    catalog = contracts()
    by_id = {item["id"]: item for item in catalog}
    enter("features")
    event("Frozen marketplace-wide design: all subtypes; disjoint acceptance combinations.",
          {"features": list(FEATURES), "training_combinations": 12, "unseen_combinations": 8,
           "subtypes": 16, "new_workload_calls": 8 if parent_dir else 40, "gates": GATES,
           "parent_run": parent_id, "previous_holdouts_excluded": bool(parent_id)})
    rows, measurements = [], []
    for path in sorted((runtime.state_dir / "rows").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("split") != "train" or row.get("source") != source
                or row.get("compatibility") != runtime._compatibility or row.get("brick_id") not in by_id):
            continue
        brick = by_id[row["brick_id"]]
        runtime._validate_reused_row(row, brick)
        docs = source_documents(row["group"], int(row["group"][-1]))
        rows.append({"id": row["id"], "ids": [brick["id"]], "group": brick["id"],
                     "split": "train", "source": source, "x": features_for(brick, docs),
                     "input": row["input_tokens"], "output": row["output_tokens"],
                     "evidence": row["evidence"], "evidence_sha256": row["evidence_sha256"]})
    support = Counter(member for row in rows for member in row["ids"])
    if any(support[item] < 4 for item in by_id):
        raise ValueError("First train all sixteen standalone subtypes with audited provider rows")

    def measure(split):
        result = []
        for item_index, item in enumerate(plan["groups"][split]):
            brick = {**item["contract"], "result_format": "keyed-steps-v1"}
            indices = (0, 3) if split == "train" else ((4 + item_index % 2,) if parent_dir else (4, 5))
            for index in indices:
                cancel()
                docs = source_documents(f"generalization-{run_id}-{split}-{brick['id']}-{index}", index)
                observed = call("workload", brick["id"], prompt_for(brick, docs), docs, [brick], workload=True)
                measurements.append(observed)
                row = {"id": observed["id"], "ids": item["ids"], "group": brick["id"],
                       "split": split, "source": source, "x": features_for(brick, docs),
                       "input": observed["usage"]["input_tokens"], "output": observed["usage"]["output_tokens"],
                       "evidence": observed["evidence"], "measurement": observed}
                write(artifacts / "composition-rows" / f"{row['id']}.json", row)
                event("Validated measured composition row persisted.",
                      {"split": split, "group": brick["id"], "input_tokens": row["input"],
                       "output_tokens": row["output"], "subtypes": item["ids"]})
                result.append(row)
        return result

    enter("measure")
    rows.extend(reuse_training(runtime, parent_dir, previous) if parent_dir else measure("train"))
    enter("train")
    model = fit_training(rows)
    model.update(version=uuid.uuid4().hex, compatibility=runtime._compatibility, source=source,
                 catalog_hash=fingerprint(catalog), plan_sha256=fingerprint(plan),
                 run_id=run_id, production_promoted=False)
    write(artifacts / "parameters-before-unseen-holdout.json", model)
    event("Parameters frozen before unseen-combination acceptance measurements.",
          {"train_count": len(rows), "alpha": model["alpha"], "version": model["version"],
           "train_only_cross_validation": model["tuning"]})
    enter("evaluate")
    holdout = measure("holdout")
    evaluation = evaluate(model, rows, holdout)
    model["evaluation"] = evaluation
    write(artifacts / "composition-candidate.json", model)
    event("Unseen-combination acceptance evaluated without refitting.", evaluation)
    enter("predict_after")
    metrics = evaluation["metrics"]
    result = {
        "run_id": run_id, "request": request, "runtime": "foundry", "source": source,
        "model_id": "gpt-5.4", "deployment": "gpt-5.4", "bricks": catalog,
        "documents": source_documents("train-0", 0), "requirements": [],
        "features": dict(zip(FEATURES, rows[-1]["x"])), "feature_names": list(FEATURES),
        "training": {"rows": rows + holdout, "train_count": len(rows), "test_count": len(holdout),
                     "source": source, "version": model["version"], "alpha": model["alpha"],
                     "input_mae": metrics["input_mae"], "output_mae": metrics["output_mae"],
                     "pilot_published": False, "production_promoted": False,
                     "generalization": evaluation},
        "before": {"supported": False, "reason": "Previous exact-workflow coverage was incomplete."},
        "after": {"supported": False, "reason": "Catalog-wide training, not a customer project quote.",
                  "source": source, "model_id": "gpt-5.4", "forecast_mode": "measured-composition-model",
                  "input_tokens": None, "output_tokens": None},
        "catalog": runtime.catalog()["items"], "agents": [], "measurements": measurements,
        "orchestration": {"calls": 0},
        "workload": {"calls": len(measurements),
                     "input_tokens": sum(row["usage"]["input_tokens"] for row in measurements),
                     "output_tokens": sum(row["usage"]["output_tokens"] for row in measurements)},
        "budget": json.loads((runtime.state_dir / "budget.json").read_text())["budget"],
        "rates": runtime.config["pricing"],
        "limitations": ["Small source-template pilot; compact step results; not a production guarantee.",
                        "Held-out unseen combinations test generalization, not every possible permutation."],
        "adjudication": {"unsupported": [], "summary": "Deterministic marketplace-wide measurement design."},
        "dissent": [], "composition": {"kind": "generalizing-model", "standalone_forecasts_summed": False},
        "generalization": evaluation,
    }
    enter("complete")
    write(run_dir / "result.json", result)
    if evaluation["accepted"]:
        write(runtime.state_dir / "composition-versions" / f"{model['version']}.json", model)
        cancel()
        write(runtime.state_dir / "composition-current.json", {
            "version": model["version"], "sha256": fingerprint(model), "run_id": run_id,
        })
        result["training"]["pilot_published"] = True
        write(run_dir / "result.json", result)
    event("Marketplace composition acceptance finished.", {
        "published": evaluation["accepted"], "version": model["version"],
        "all_subtypes_tested": evaluation["subtypes_tested"] == 16,
        "unseen_combinations": evaluation["unseen_combinations"], "production_promoted": False,
    })
    return result
