"""Bounded orchestration adapter for real executions and calibration rewards.

The fixed compound executes two validated source-only calls in order against
the same documents. It has no output handoff or cross-brick context optimization.
It is experimental measurement evidence, not a marketplace combination quote.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict

from .marketplace_agent_contracts import fingerprint, numeric_features, workload_prompt
from .measurement_policy import MeasurementPolicy, POLICY_VERSION, ROUNDS

ACTION_SELECTION_VERSION = "requested-novel-first-v1"


def select_actions(bricks: list[dict], requested_id: str | None = None) -> tuple[list[dict], dict]:
    """Keep required capability coverage within two singles, independent of input order."""
    selected = {}
    for brick in bricks:
        existing = selected.get(brick["id"])
        if existing and existing["contract_hash"] != brick["contract_hash"]:
            raise ValueError("conflicting selected contract identities")
        selected[brick["id"]] = brick
    if not selected:
        raise ValueError("measurement policy requires a supported selected contract")
    if requested_id is not None and requested_id not in selected:
        raise ValueError("requested measurement capability is not in the supported selection")
    required = ({requested_id} if requested_id is not None else
                {key for key, value in selected.items() if value.get("novel")})
    if len(required) > 2:
        raise ValueError("more than two required novel actions; narrow scope rather than discard capabilities")
    ranked = sorted(selected, key=lambda key: (
        key not in required, not bool(selected[key].get("novel")), key))
    included = sorted(ranked[:2])
    audit = {
        "version": ACTION_SELECTION_VERSION, "requested_id": requested_id,
        "eligible": sorted(selected), "required": sorted(required), "included": included,
        "excluded": [{"id": key, "reason": "two-single action bound after required/novel priority"}
                     for key in sorted(selected) if key not in included],
        "ranking": "required capability, other novel capability, lexical contract ID",
        "pair_order": "lexical contract ID; same original documents per independent call; no handoff",
    }
    return [selected[key] for key in included], audit


class AdaptiveMeasurements:
    """Run an opt-in bandit using the existing metered call and ridge fit helpers."""

    def __init__(self, *, runtime, bricks, names, source, run_id, run_dir, call, event,
                 cancel, write, read, requested_id: str | None = None):
        self.runtime, self.names, self.source = runtime, names, source
        self.run_id, self.run_dir = run_id, run_dir
        self.call, self.event, self.cancel, self.write, self.read = call, event, cancel, write, read
        singles, selection = select_actions(bricks, requested_id)
        self.actions = {b["id"]: [b] for b in singles}
        self.compound_id = None
        if len(singles) == 2:
            self.compound_id = "ordered_" + fingerprint([b["contract_hash"] for b in singles])[:16]
            self.actions[self.compound_id] = singles
        self.spec = {"version": POLICY_VERSION, "actions": {
            key: [{"id": b["id"], "contract_hash": b["contract_hash"]} for b in steps]
            for key, steps in self.actions.items()}, "ordered_same_sources": True,
            "output_handoff": False, "marketplace_combination_quote": False,
            "selection": selection}
        partition = fingerprint({
            "runtime": runtime._compatibility,
            "spec": {key: value for key, value in self.spec.items() if key != "selection"},
            "selection_version": ACTION_SELECTION_VERSION, "names": names,
        })
        self.policy = MeasurementPolicy(
            runtime.state_dir / "measurement-policy" / f"{partition}.sqlite",
            compatibility=partition, source=source, actions=list(self.actions))
        if self.policy.snapshot()["pending"]:
            raise RuntimeError("unsettled policy decision from prior execution; manual audit required")
        self.artifacts = run_dir / "agent-artifacts" / "measurement-policy"
        self.write(self.artifacts / "action-contracts.json", self.spec)
        self.event("Measurement action availability resolved.", {
            "source": source, "selection": selection, "actions": self.spec["actions"]})
        self.calibration = []
        self.observations = []

    def measure(self, action: str, group: str, index: int, split: str) -> dict:
        docs = self.runtime._documents(group, index)
        steps = self.actions[action]
        features = {name: sum(numeric_features(b, docs)[name] for b in steps)
                    for name in numeric_features(steps[0], docs)}
        records = []
        for ordinal, brick in enumerate(steps):
            self.cancel()
            record = self.call("workload", brick["id"], workload_prompt(brick, docs),
                               docs, [brick], workload=True)
            evidence = self.read(self.run_dir / record["evidence"])
            if (evidence["status"] != "validated" or evidence["source"] != self.source
                    or evidence["prompt_sha256"] != fingerprint(workload_prompt(brick, docs))):
                raise ValueError("compound step lacks matching validated execution evidence")
            records.append(evidence)
            self.event("Ordered measurement step validated.",
                       {"action": action, "step": ordinal + 1, "brick_id": brick["id"],
                        "call_id": record["id"], "source": self.source,
                        "usage": record["usage"], "rated_usd": record["rated_usd"],
                        "safety_usd": record["safety_usd"], "split": split, "group": group})
        row = {"id": uuid.uuid4().hex, "run_id": self.run_id, "brick_id": action,
               "group": group, "split": split, "source": self.source,
               "compatibility": self.runtime._compatibility, "features": features,
               "input_tokens": sum(r["usage"]["input_tokens"] for r in records),
               "output_tokens": sum(r["usage"]["output_tokens"] for r in records),
               "rated_usd": sum(r["rated_usd"] for r in records),
               "safety_usd": sum(r["safety_usd"] for r in records),
               "execution_ids": [r["id"] for r in records],
               "contract_hash": fingerprint(self.spec["actions"][action]),
               "measurements": records}
        self.write(self.artifacts / "rows" / f"{row['id']}.json", row)
        self.event("Selected workload measured; no forecast used as a label.",
                   {k: row[k] for k in ("id", "brick_id", "source", "split", "group",
                                       "input_tokens", "output_tokens", "rated_usd", "safety_usd")})
        return row

    def _model(self, rows: list[dict]) -> tuple[dict, str, float]:
        models = {t: self.runtime._fit(rows, self.names, t, 1.0) for t in ("input", "output")}
        payload = {"models": {t: asdict(m) for t, m in models.items()},
                   "train_ids": [r["id"] for r in rows], "alpha": 1.0,
                   "names": self.names, "source": self.source}
        version = fingerprint(payload)
        self.write(self.artifacts / "predictors" / f"{version}.json", payload)
        errors = [abs(models[t].predict([r["features"][n] for n in self.names]) - r[t + "_tokens"])
                  for r in self.calibration for t in ("input", "output")]
        if not errors or not all(math.isfinite(e) for e in errors):
            raise ValueError("finite calibration metrics required")
        return models, version, sum(errors) / len(errors)

    def run(self, train_rows: list[dict]) -> list[dict]:
        if any(r["split"] != "train" or r["group"] not in {"train-0", "train-1", "train-2"}
               for r in train_rows):
            raise ValueError("calibration/final holdout cannot enter policy fitting")
        rows = list(train_rows)
        if self.compound_id:
            rows.append(self.measure(self.compound_id, "train-0", 0, "train"))
        for action in self.actions:
            self.calibration.append(self.measure(action, f"calibration-{self.run_id}-3", 3, "calibration"))
        calibration_ids = [call_id for row in self.calibration for call_id in row["execution_ids"]]
        self.write(self.artifacts / "calibration.json",
                   {"rows": self.calibration, "never_final_acceptance": True})
        for index in range(ROUNDS):
            self.cancel()
            _, before_version, before_mae = self._model(rows)
            decision = self.policy.choose(f"{self.run_id}:{index}", {
                "run_id": self.run_id, "round": index, "max_rounds": ROUNDS,
                "predictor_before": before_version, "calibration_mae": before_mae,
                "train_count": len(rows), "calibration_ids": calibration_ids})
            self.event("Measurement policy decision.", decision)
            try:
                row = self.measure(decision["action"], f"train-{index % 3}", index % 3, "train")
                next_rows = rows + [row]
                _, after_version, after_mae = self._model(next_rows)
                self.cancel()
                observation = self.policy.feedback(
                    decision["decision_id"], before_mae=before_mae, after_mae=after_mae,
                    cost_usd=row["rated_usd"], measurements=row["measurements"],
                    predictor_before=before_version, predictor_after=after_version,
                    calibration_ids=calibration_ids)
            except BaseException as exc:
                rejected = self.policy.reject(decision["decision_id"], f"{type(exc).__name__}: {exc}")
                self.event("Measurement policy stopped; reward unavailable.", rejected)
                raise
            rows = next_rows
            self.observations.append(observation)
            self.event("Calibration reward recorded.", {
                k: observation[k] for k in ("decision_id", "reward", "before_mae", "after_mae",
                                           "cost_usd", "source")})
            self.event("Measurement policy updated durably.", self.policy.snapshot())
            self.event("Supervised ridge refit recorded.", {"version": after_version,
                                                           "train_count": len(rows), "source": self.source})
        return rows

    def final_holdouts(self) -> list[dict]:
        if not self.compound_id:
            return []
        return [self.measure(self.compound_id, f"holdout-{self.run_id}-{i}", i, "holdout")
                for i in (4, 5)]

    def report(self) -> dict:
        return {**self.spec, "policy": self.policy.snapshot(), "rounds": len(self.observations),
                "calibration_ids": [r["id"] for r in self.calibration],
                "calibration_rated_usd": sum(r["rated_usd"] for r in self.calibration),
                "observations": self.observations,
                "limitations": [
                    "Non-contextual bandit; audited context is not a learned context model.",
                    "Signed clipped calibration improvement per marginal measurement dollar.",
                    "Seed/calibration overhead is separately metered, not hidden in marginal reward.",
                    "Calibration is adaptively optimized; final holdouts never update the policy.",
                    "Templates are a small pilot, not a general accuracy or efficiency benchmark.",
                    "One fixed ordered same-source compound; no output handoff or marketplace combo quote.",
                ]}
