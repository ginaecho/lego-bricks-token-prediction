"""Post-delivery feedback bound to purchased scope, with automatic shadow learning.

The original selection, not a newly sampled action, receives the outcome.
Only prospectively logged suggestions can support off-policy evaluation.
Human promotion remains separate from the customer form.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from .customer_outcomes import (
    EPSILON, SOURCES, OutcomeStore, _bool, _enum, _keys, _mapping, _text, canonical, feedback_themes,
)
from .customer_models import _count, _number
from .robust import Record, RidgeLinearModel


REWARD_VERSION = "delivered-experience-v1"
KEEP = {"all": 1., "some": .5, "none": 0., "unknown": None}
USE = {"yes": 1., "partly": .5, "no": 0., "unknown": None}
OPTIONAL_NUMBERS = ("benefit_usd", "cost_usd", "revenue_usd", "provider_cost_usd",
                    "autonomy_pct", "kept_pct", "useful_pct", "actual_tokens")
TARGETS = ("satisfaction", "sow_quality", "staffing_fit", "reported_roi",
           "reported_profit_usd", "reported_aes")


def _optional(value: object, name: str, maximum: float = 1e9) -> float | None:
    if value is None:
        return None
    number = _number(value, name)
    if number > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return number


def action_key(function: dict) -> str:
    return canonical([function["variant_id"], function["model_id"]])


def validate_handoff(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("a marketplace handoff is required")
    _keys(value, {"schema", "receipt_id", "title", "description", "flow", "quote_basis",
                  "functions", "staffing", "estimated_project_tokens"})
    if value["schema"] != "token-yield-delivery-handoff-v1":
        raise ValueError("unsupported marketplace handoff schema")
    receipt_id = _text(value["receipt_id"], "receipt_id")
    if len(receipt_id) > 100:
        raise ValueError("receipt_id is too long")
    functions = value["functions"]
    if not isinstance(functions, list) or not 1 <= len(functions) <= 50:
        raise ValueError("handoff must include 1 to 50 selected functionalities")
    cleaned = []
    for item in functions:
        if not isinstance(item, dict):
            raise ValueError("selected functionality must be an object")
        _keys(item, {"id", "feature_id", "variant_id", "name", "model_id", "model_name",
                     "estimated_tokens", "runs", "decision_id"})
        function: dict = {key: _text(item[key], key) for key in
                          ("id", "feature_id", "variant_id", "name", "model_id", "model_name")}
        function["estimated_tokens"] = _optional(item["estimated_tokens"], "estimated_tokens")
        function["runs"] = _count(item["runs"], "runs")
        if function["runs"] > 1_000_000_000:
            raise ValueError("runs exceeds 1 billion")
        function["decision_id"] = (_text(item["decision_id"], "decision_id")
                                   if item["decision_id"] is not None else None)
        cleaned.append(function)
    if len({f["id"] for f in cleaned}) != len(cleaned):
        raise ValueError("duplicate selected functionality")
    staffing = value["staffing"]
    if not isinstance(staffing, list) or len(staffing) > 20:
        raise ValueError("staffing must be a list of at most 20 planned roles")
    roles = []
    for role in staffing:
        role = _mapping(role, "staffing role")
        _keys(role, {"role", "hours"})
        roles.append({"role": _text(role["role"], "role"),
                      "hours": _optional(role["hours"], "hours")})
    return {
        "schema": value["schema"], "receipt_id": receipt_id,
        "title": _text(value["title"], "title"),
        "description": _text(value["description"], "description", optional=True),
        "flow": _enum(value["flow"], ("manual", "custom"), "flow"),
        "quote_basis": _text(value["quote_basis"], "quote_basis"),
        "functions": cleaned, "staffing": roles,
        "estimated_project_tokens": _optional(value["estimated_project_tokens"], "estimated_project_tokens"),
    }


def extract_signals(feedback: dict, sow: float | None, staffing: float | None) -> dict:
    """Keep categorical experience reward distinct from financial or AES proof."""
    sat = _number(feedback["satisfaction"], "satisfaction")
    if not 1 <= sat <= 5:
        raise ValueError("satisfaction must be 1 to 5")
    kept = KEEP[_enum(feedback["kept"], tuple(KEEP), "kept")]
    useful = USE[_enum(feedback["usefulness"], tuple(USE), "usefulness")]
    details = {key: _optional(feedback[key], key, 100 if key.endswith("_pct") else 1e9)
               for key in OPTIONAL_NUMBERS}
    benefit, cost = details["benefit_usd"], details["cost_usd"]
    revenue, provider = details["revenue_usd"], details["provider_cost_usd"]
    a, k, u = (details[name] for name in ("autonomy_pct", "kept_pct", "useful_pct"))
    aes = 0. if a == 0 or k == 0 else a * k * u / 10000 if (
        a is not None and k is not None and u is not None) else None
    reward = (2 * (.4 * (sat - 1) / 4 + .3 * kept + .3 * useful) - 1
              if kept is not None and useful is not None else None)
    return {
        "satisfaction": sat, "sow_quality": sow, "staffing_fit": staffing,
        "experience_reward": reward, "reward_version": REWARD_VERSION,
        "reward_pending": reward is None,
        "reported_aes": aes,
        "reported_roi": (benefit - cost) / cost if benefit is not None and cost else None,
        "reported_profit_usd": revenue - provider if revenue is not None and provider is not None else None,
        "financial_status": "customer_reported_not_verified",
        "themes": feedback_themes(feedback["comments"]),
        "actual_tokens": details["actual_tokens"],
    }


def distribution(policy: dict, context: str, options: list[str]) -> dict[str, float]:
    values = {key: policy.get(context, {}).get(key, {}).get("q", 0.) for key in options}
    best = max(values.values())
    greedy = [key for key, value in values.items() if value == best]
    return {key: EPSILON / len(options) + ((1 - EPSILON) / len(greedy)
            if key in greedy else 0.) for key in options}


class DeliveryFeedback:
    def __init__(self, store: OutcomeStore):
        self.store = store
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS delivered_feedback "
                       "(id TEXT PRIMARY KEY, source TEXT NOT NULL, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS delivery_choices "
                       "(id TEXT PRIMARY KEY, source TEXT NOT NULL, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS delivery_models "
                       "(version INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, body TEXT NOT NULL)")

    @staticmethod
    def _active(db, source: str) -> dict:
        row = db.execute("SELECT value FROM meta WHERE key=?", (f"delivery-active:{source}",)).fetchone()
        if not row:
            return {"version": 0, "policy": {}}
        return json.loads(db.execute("SELECT body FROM delivery_models WHERE version=?",
                                    (int(row[0]),)).fetchone()[0])

    @staticmethod
    def _latest(db, source: str) -> dict | None:
        row = db.execute("SELECT body FROM delivery_models WHERE source=? ORDER BY version DESC LIMIT 1",
                         (source,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _records(db, source: str) -> list[dict]:
        return [json.loads(row[0]) for row in db.execute(
            "SELECT body FROM delivered_feedback WHERE source=? ORDER BY rowid", (source,))]

    def listing(self, *, include_examples: bool = False) -> dict:
        with self.store.connection() as db:
            return {"deliveries": [
                {"receipt_id": value["handoff"]["receipt_id"], "title": value["handoff"]["title"],
                 "source": value["source"], "updated_at": value["updated_at"],
                 "function_count": len(value["handoff"]["functions"])}
                for row in db.execute(
                    "SELECT body FROM delivered_feedback WHERE ? OR NOT EXISTS "
                    "(SELECT 1 FROM meta WHERE key='delivery-training-example:' || delivered_feedback.id) "
                    "ORDER BY rowid DESC", (include_examples,))
                for value in [json.loads(row[0])]]}

    def get(self, receipt_id: str) -> dict:
        with self.store.connection() as db:
            row = db.execute("SELECT body FROM delivered_feedback WHERE id=?",
                             (_text(receipt_id, "receipt_id"),)).fetchone()
            if not row:
                raise ValueError("delivered project not found")
            return json.loads(row[0])

    def suggest(self, source: str, feature_id: str, options: list[dict], *, rng=None) -> dict:
        _enum(source, SOURCES, "source")
        context = _text(feature_id, "feature_id")
        if not isinstance(options, list) or not 1 <= len(options) <= 20:
            raise ValueError("provide 1 to 20 existing marketplace options")
        for option in options:
            option = _mapping(option, "marketplace option")
            _keys(option, {"variant_id", "model_id"})
            for key, value in option.items():
                _text(value, key)
        keys = [action_key(o) for o in options]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate marketplace options")
        with self.store.connection() as db:
            active = self._active(db, source)
            probs = distribution(active["policy"], context, keys)
            draw = (rng or random.SystemRandom()).random()
            if not 0 <= draw < 1:
                raise ValueError("invalid random draw")
            selected, cumulative = keys[-1], 0.
            for key, probability in probs.items():
                cumulative += probability
                if draw < cumulative:
                    selected = key
                    break
            value = {"id": uuid.uuid4().hex, "source": source, "feature_id": context,
                     "action": selected, "option": options[keys.index(selected)],
                     "probabilities": probs, "policy_version": active["version"],
                     "reward_version": REWARD_VERSION}
            db.execute("INSERT INTO delivery_choices VALUES(?,?,?)",
                       (value["id"], source, canonical(value)))
            self.store.audit(db, "marketplace_suggestion", value)
            return value

    def submit(self, payload: dict) -> dict:
        payload = _mapping(payload, "feedback submission")
        _keys(payload, {"handoff", "source", "respondent", "delivered", "consent",
                        "sow_quality", "staffing_fit", "observation_days", "feedback"})
        handoff = validate_handoff(payload["handoff"])
        source = _enum(payload["source"], SOURCES, "source")
        if not _bool(payload["delivered"], "delivered"):
            raise ValueError("confirm delivery before leaving post-delivery feedback")
        if not _bool(payload["consent"], "consent"):
            raise ValueError("permission to use this feedback for learning is required")
        sow = _optional(payload["sow_quality"], "sow_quality", 5)
        staff = _optional(payload["staffing_fit"], "staffing_fit", 5)
        if any(value is not None and value < 1 for value in (sow, staff)):
            raise ValueError("SOW and staffing ratings must be 1 to 5 or unknown")
        days = _count(payload["observation_days"], "observation_days")
        if not 1 <= days <= 3650:
            raise ValueError("observation_days must be 1 to 3650")
        responses = payload["feedback"]
        if not isinstance(responses, list):
            raise ValueError("feedback must be a list")
        cleaned = []
        for item in responses:
            item = _mapping(item, "functionality feedback")
            _keys(item, {"function_id", "satisfaction", "kept", "usefulness", "comments",
                         *OPTIONAL_NUMBERS})
            item = {**item, "function_id": _text(item["function_id"], "function_id"),
                    "comments": _text(item["comments"], "comments", optional=True)}
            signals = extract_signals(item, sow, staff)
            cleaned.append({**item, "signals": signals})
        expected = {f["id"] for f in handoff["functions"]}
        received = [item["function_id"] for item in cleaned]
        if len(received) != len(set(received)) or set(received) != expected:
            raise ValueError("feedback must match exactly the selected functionalities, once each")
        receipt = handoff["receipt_id"]
        split = "validation" if int(hashlib.sha256(receipt.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
        value = {
            "handoff": handoff, "source": source, "respondent": _text(payload["respondent"], "respondent"),
            "delivered": True, "consent": True, "sow_quality": sow, "staffing_fit": staff,
            "observation_days": days, "feedback": cleaned, "split": split,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.store.connection() as db:
            prior = db.execute("SELECT body FROM delivered_feedback WHERE id=?", (receipt,)).fetchone()
            if prior:
                existing = json.loads(prior[0])
                if existing["handoff"] != handoff or existing["source"] != source:
                    raise ValueError("the delivered selection and evidence source are frozen")
                if existing["observation_days"] != days:
                    raise ValueError("the observation window is frozen for this delivery")
                same = {k: v for k, v in existing.items() if k != "updated_at"} == {
                    k: v for k, v in value.items() if k != "updated_at"}
                if same:
                    return self._receipt_result(db, existing)
                for row in db.execute("SELECT body FROM delivery_models WHERE source=?", (source,)):
                    model = json.loads(row[0])
                    if model.get("promoted_by") or receipt in model.get("evaluation_ids", []):
                        if receipt in model["training_ids"] or receipt in model.get("evaluation_ids", []):
                            raise ValueError("feedback used by a promoted/evaluated policy is locked for audit")
            used_choices = {}
            for record in self._records(db, source):
                if record["handoff"]["receipt_id"] != receipt:
                    for function in record["handoff"]["functions"]:
                        if function["decision_id"]:
                            used_choices[function["decision_id"]] = record["handoff"]["receipt_id"]
            for function in handoff["functions"]:
                decision_id = function["decision_id"]
                if not decision_id:
                    continue
                if decision_id in used_choices:
                    raise ValueError("suggestion already credited to a functionality or delivery")
                used_choices[decision_id] = receipt
                row = db.execute("SELECT body FROM delivery_choices WHERE id=?", (decision_id,)).fetchone()
                if not row:
                    raise ValueError("unknown pre-delivery suggestion")
                choice = json.loads(row[0])
                if (choice["source"] != source or choice["feature_id"] != function["feature_id"]
                        or choice["action"] != action_key(function)):
                    raise ValueError("feedback does not match the original suggested action/source")
            db.execute("INSERT OR REPLACE INTO delivered_feedback VALUES(?,?,?)",
                       (receipt, source, canonical(value)))
            self.store.audit(db, "delivered_customer_feedback", value)
            self._train(db, source)
            return self._receipt_result(db, value)

    def _receipt_result(self, db, value: dict) -> dict:
        model = self._latest(db, value["source"])
        return {"receipt_id": value["handoff"]["receipt_id"], "saved": True,
                "source": value["source"], "signals": [
                    {"function_id": item["function_id"], **item["signals"]} for item in value["feedback"]],
                "learning": {"candidate_version": model["version"] if model else None,
                             "reward_updates": model["reward_updates"] if model else 0,
                             "status": "protected_validation" if value["split"] == "validation" else "candidate_updated",
                             "active_policy_changed": False}}

    def _train(self, db, source: str) -> dict:
        records = [r for r in self._records(db, source) if r["split"] == "train"]
        digest = hashlib.sha256(canonical(records).encode()).hexdigest()
        latest = self._latest(db, source)
        if latest and latest["training_digest"] == digest:
            return latest
        policy, targets = {}, {}
        updates = 0
        for record in records:
            replies = {f["function_id"]: f for f in record["feedback"]}
            for function in record["handoff"]["functions"]:
                context, action = function["feature_id"], action_key(function)
                signals = replies[function["id"]]["signals"]
                reward = signals["experience_reward"]
                if reward is not None:
                    cell = policy.setdefault(context, {}).setdefault(action, {"n": 0, "q": 0.})
                    cell["n"] += 1
                    cell["q"] += (reward - cell["q"]) / cell["n"]
                    updates += 1
                inputs = (math.log1p(function["estimated_tokens"] or 0),
                          float(function["estimated_tokens"] is not None),
                          math.log1p(function["runs"]), math.log1p(record["observation_days"]))
                for target in TARGETS:
                    if signals[target] is not None:
                        key = canonical([context, action, target])
                        targets.setdefault(key, []).append(
                            Record(inputs, signals[target], record["handoff"]["receipt_id"]))
        fits = {key: {"fit": asdict(RidgeLinearModel.fit(rows, alpha=5.)),
                      "independent_projects": len({r.group for r in rows}),
                      "observed_ranges": [[min(r.features[j] for r in rows),
                                           max(r.features[j] for r in rows)] for j in range(4)]}
                for key, rows in targets.items() if len({r.group for r in rows}) >= 3}
        model = {"source": source, "policy": policy, "predictors": fits,
                 "training_ids": [r["handoff"]["receipt_id"] for r in records],
                 "training_snapshots": records,
                 "training_digest": digest, "reward_updates": updates,
                 "reward_version": REWARD_VERSION, "evaluation": None, "evaluation_ids": [],
                 "base_policy_version": self._active(db, source)["version"]}
        cursor = db.execute("INSERT INTO delivery_models(source,body) VALUES(?,?)", (source, "{}"))
        model["version"] = cursor.lastrowid
        db.execute("UPDATE delivery_models SET body=? WHERE version=?", (canonical(model), model["version"]))
        self.store.audit(db, "delivery_candidate_trained", {
            "version": model["version"], "source": source, "reward_updates": updates,
            "training_ids": model["training_ids"]})
        return model

    def learning(self, source: str) -> dict:
        _enum(source, SOURCES, "source")
        with self.store.connection() as db:
            return {"candidate": self._latest(db, source), "active": self._active(db, source)}

    def predict(self, source: str, handoff: dict, observation_days: int) -> dict:
        """Return supported, in-range estimates from the human-approved model."""
        _enum(source, SOURCES, "source")
        scope = validate_handoff(handoff)
        days = _count(observation_days, "observation_days")
        if not 1 <= days <= 3650:
            raise ValueError("observation_days must be 1 to 3650")
        with self.store.connection() as db:
            model = self._active(db, source)
        estimates = []
        for function in scope["functions"]:
            inputs = (math.log1p(function["estimated_tokens"] or 0),
                      float(function["estimated_tokens"] is not None),
                      math.log1p(function["runs"]), math.log1p(days))
            targets = {}
            for target in TARGETS:
                fit = model.get("predictors", {}).get(canonical(
                    [function["feature_id"], action_key(function), target]))
                if not fit:
                    targets[target] = {"estimate": None, "reason": "insufficient approved evidence"}
                elif any(not low - 1e-9 <= value <= high + 1e-9
                         for value, (low, high) in zip(inputs, fit["observed_ranges"])):
                    targets[target] = {"estimate": None, "reason": "outside observed input ranges"}
                else:
                    predicted = RidgeLinearModel(**fit["fit"]).predict(inputs)
                    if target in ("satisfaction", "sow_quality", "staffing_fit"):
                        predicted = max(1., min(5., predicted))
                    elif target == "reported_aes":
                        predicted = max(0., min(100., predicted))
                    targets[target] = {"estimate": predicted,
                                       "independent_projects": fit["independent_projects"],
                                       "evidence": "customer_reported_not_verified"}
            estimates.append({"function_id": function["id"], "targets": targets})
        return {"source": source, "policy_version": model["version"], "functions": estimates}

    def promote(self, source: str, approver: str) -> dict:
        """Evaluate once on fresh prospective validation choices, then human-gate."""
        _enum(source, SOURCES, "source")
        approver = _text(approver, "approver")
        with self.store.connection() as db:
            model = self._latest(db, source)
            if model is None:
                raise ValueError("no delivery feedback candidate")
            if model.get("promoted_by"):
                return model
            active = self._active(db, source)
            if model["base_policy_version"] != active["version"]:
                raise ValueError("candidate baseline changed")
            if model["evaluation"] is None:
                used = set()
                for row in db.execute("SELECT body FROM delivery_models WHERE source=?", (source,)):
                    used.update(json.loads(row[0]).get("evaluation_ids", []))
                groups, weights, ids = [], [], []
                for record in self._records(db, source):
                    receipt = record["handoff"]["receipt_id"]
                    if record["split"] != "validation" or receipt in used:
                        continue
                    rewards = {r["function_id"]: r["signals"]["experience_reward"] for r in record["feedback"]}
                    gains, ratios = [], []
                    for function in record["handoff"]["functions"]:
                        reward = rewards[function["id"]]
                        if not function["decision_id"] or reward is None:
                            continue
                        choice = json.loads(db.execute("SELECT body FROM delivery_choices WHERE id=?",
                                                       (function["decision_id"],)).fetchone()[0])
                        action = choice["action"]
                        options = list(choice["probabilities"])
                        candidate = distribution(model["policy"], function["feature_id"], options)[action]
                        baseline = distribution(active["policy"], function["feature_id"], options)[action]
                        logged = choice["probabilities"][action]
                        gains.append((candidate - baseline) / logged * reward)
                        ratios.append(candidate / logged)
                    if gains:
                        groups.append(sum(gains) / len(gains))
                        weights.append(sum(ratios) / len(ratios))
                        ids.append(receipt)
                n = len(groups)
                if n < 20:
                    raise ValueError("promotion pending: need 20 fresh, prospectively logged validation projects")
                mean = sum(groups) / n
                se = math.sqrt(sum((g - mean) ** 2 for g in groups) / (n - 1) / n)
                ess = sum(weights) ** 2 / sum(w * w for w in weights)
                evaluation = {"projects": n, "gain": mean, "lower_bound": mean - 1.96 * se,
                              "effective_sample_size": ess}
                evaluation["eligible"] = (evaluation["lower_bound"] > 0 and ess >= 10
                                           and len(model["training_ids"]) >= 20)
                model.update(evaluation=evaluation, evaluation_ids=ids)
            if model["evaluation"]["eligible"]:
                model["promoted_by"] = approver
                db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                           (f"delivery-active:{source}", str(model["version"])))
            db.execute("UPDATE delivery_models SET body=? WHERE version=?", (canonical(model), model["version"]))
            self.store.audit(db, "delivery_policy_review", {
                "version": model["version"], "approver": approver, "evaluation": model["evaluation"]})
            return model
