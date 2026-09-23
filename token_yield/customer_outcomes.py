"""Local outcome learning and an audited, human-gated contextual bandit.

Ratings train outcome regressions. Reviewed, complete outcome rewards train a
tabular contextual bandit over approved SOW/staffing packages, not LLM weights.
Synthetic and customer data have separate models, policies and evaluation sets.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .customer_models import _count, _identifier, _mapping, _number
from .robust import Record, RidgeLinearModel


ACTIONS = [
    {"id": "lean", "name": "Lean delivery",
     "description": "Focused SOW, generalist staffing, standard acceptance review."},
    {"id": "balanced", "name": "Balanced delivery",
     "description": "Explicit acceptance criteria, specialist checkpoint, planned review."},
    {"id": "assured", "name": "Assured delivery",
     "description": "Detailed SOW, specialist staffing, independent quality review."},
]
KINDS = ("extract", "classify", "plan", "report", "sow", "staffing", "custom")
COMPLEXITIES = ("low", "medium", "high")
SOURCES = ("real", "synthetic")
RATINGS = ("satisfaction", "sow_quality", "staffing_fit", "impact")
MONEY = ("customer_benefit_usd", "customer_cost_usd",
         "provider_revenue_usd", "provider_cost_usd")
COUNTS = ("completed_units", "kept_units", "useful_units", "actual_tokens")
TARGETS = (*RATINGS, "aes", "customer_roi", "provider_profit_usd", "net_benefit_usd")
EPSILON = 0.2
SCHEMA = "customer-outcomes-v1"
REWARD_SPEC = "0.4*AES/100+0.2*satisfaction/5+0.1*SOW/5+0.1*staffing/5+0.2*clip(net/reference,-1,1)"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, name: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    text = _identifier(value, name).strip()
    if len(text) > 12000:
        raise ValueError(f"{name} exceeds 12000 characters")
    return text


def _enum(value: object, choices: tuple, name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _keys(value: dict, required: set) -> None:
    if set(value) != required:
        raise ValueError(f"fields differ: missing={sorted(required-set(value))}, "
                         f"unexpected={sorted(set(value)-required)}")


def validate_function(value: dict) -> dict:
    value = _mapping(value, "function")
    fields = {"id", "name", "kind", "complexity", "assigned_units", "estimated_tokens",
              "customer_budget_usd", "provider_budget_usd", "reference_benefit_usd",
              "observation_days"}
    _keys(value, fields)
    result: dict = {key: _text(value[key], key) for key in ("id", "name")}
    result["kind"] = _enum(value["kind"], KINDS, "kind")
    result["complexity"] = _enum(value["complexity"], COMPLEXITIES, "complexity")
    for key in ("assigned_units", "estimated_tokens", "observation_days"):
        result[key] = _count(value[key], key)
        if not 1 <= result[key] <= 1_000_000_000:
            raise ValueError(f"{key} must be positive and at most 1 billion")
    for key in ("customer_budget_usd", "provider_budget_usd", "reference_benefit_usd"):
        result[key] = _number(value[key], key)
        if not 0 < result[key] <= 1_000_000_000:
            raise ValueError(f"{key} must be positive and at most 1 billion USD")
    return result


def validate_feedback(value: dict, function: dict) -> dict:
    required = {"decision_id", "respondent", *COUNTS, *RATINGS, *MONEY,
                "financial_evidence", "window_closed", "safety_ok", "budget_ok",
                "evidence", "comments"}
    _keys(value, required)
    result: dict = {key: _text(value[key], key) for key in ("decision_id", "respondent")}
    for key in ("evidence", "comments"):
        result[key] = _text(value[key], key, optional=True)
    for key in COUNTS:
        result[key] = None if value[key] is None else _count(value[key], key)
        if result[key] is not None and result[key] > 1_000_000_000:
            raise ValueError(f"{key} exceeds 1 billion")
    previous = function["assigned_units"]
    for key in COUNTS[:3]:
        current = result[key]
        if current is not None and (previous is None or current > previous):
            raise ValueError("counts must follow assigned >= autonomous completed >= kept >= useful")
        previous = current
    for key in RATINGS:
        result[key] = None if value[key] is None else _number(value[key], key)
        if result[key] is not None and not 1 <= result[key] <= 5:
            raise ValueError(f"{key} must be between 1 and 5")
    for key in MONEY:
        result[key] = None if value[key] is None else _number(value[key], key)
        if result[key] is not None and result[key] > 1_000_000_000:
            raise ValueError(f"{key} exceeds 1 billion USD")
    result["financial_evidence"] = _enum(
        value["financial_evidence"], ("reported", "verified", "unknown"), "financial_evidence")
    for key in ("window_closed", "safety_ok", "budget_ok"):
        result[key] = _bool(value[key], key)
    if result["financial_evidence"] == "verified" and not result["evidence"]:
        raise ValueError("verified finances require an evidence reference")
    return result


def outcome_signals(function: dict, feedback: dict) -> dict:
    """Score nested autonomous work units; absence is unknown, never success."""
    completed, kept, useful = (feedback[key] for key in COUNTS[:3])
    if completed == 0 or kept == 0:
        useful = 0
    assigned = function["assigned_units"]
    closed = feedback["window_closed"]
    aes = 100 * useful / assigned if useful is not None and closed else None
    tokens = feedback["actual_tokens"]
    benefit, cost = (feedback[key] for key in MONEY[:2])
    verified = feedback["financial_evidence"] == "verified" and closed
    net = benefit - cost if benefit is not None and cost is not None and verified else None
    revenue, delivery_cost = (feedback[key] for key in MONEY[2:])
    profit = (revenue - delivery_cost if revenue is not None and delivery_cost is not None
              and verified else None)
    pending = []
    if aes is None:
        pending.append("closed autonomous outcome evidence")
    for name in ("satisfaction", "sow_quality", "staffing_fit"):
        if feedback[name] is None:
            pending.append(name)
    if net is None:
        pending.append("verified customer benefit and total cost")
    if not feedback["evidence"]:
        pending.append("delivery/outcome evidence reference")
    if tokens is None or tokens == 0:
        pending.append("positive actual token accounting including retries")
    if not feedback["safety_ok"]:
        pending.append("safety gate failed")
    financial_overrun = (
        cost is not None and cost > function["customer_budget_usd"]
        or delivery_cost is not None and delivery_cost > function["provider_budget_usd"])
    if not feedback["budget_ok"] or financial_overrun:
        pending.append("budget gate failed")
    reward = None
    if not pending:
        assert aes is not None and net is not None
        reward = (
            .4 * aes / 100 + .2 * feedback["satisfaction"] / 5
            + .1 * feedback["sow_quality"] / 5 + .1 * feedback["staffing_fit"] / 5
            + .2 * max(-1., min(1., net / function["reference_benefit_usd"])))
    themes = feedback_themes(feedback["comments"])
    return {
        "aes": aes, "completion_rate": completed / assigned if completed is not None else None,
        "retention_rate": kept / completed if kept is not None and completed else None,
        "usefulness_rate": useful / kept if useful is not None and kept else None,
        "token_yield_per_1000": useful * 1000 / tokens
        if useful is not None and tokens and closed else None,
        "customer_roi": net / cost if net is not None and cost else None,
        "provider_profit_usd": profit, "net_benefit_usd": net,
        "reward": reward, "reward_pending": pending, "themes": themes,
        "financial_status": "verified" if verified else feedback["financial_evidence"],
        "evidence_coverage": sum(feedback[k] is not None for k in COUNTS[:3]) / 3,
    }


def feedback_themes(comments: str) -> list[str]:
    """Extract transparent discussion topics, not invented sentiment or ROI."""
    return [
        label for label, pattern in (
            ("SOW / scope", r"\b(scope|sow|requirement|acceptance)\b"),
            ("Staffing / expertise", r"\b(staff|staffing|expert|skill|team)\b"),
            ("Quality / rework", r"\b(error|quality|rework|incorrect|bug)\b"),
            ("Cost / value", r"\b(cost|expensive|roi|profit|saving|value)\b"),
            ("Delivery / timing", r"\b(slow|late|delay|fast|time)\b"),
        ) if re.search(pattern, comments, re.IGNORECASE)
    ]


def context_key(function: dict) -> str:
    return f"{function['kind']}:{function['complexity']}"


def probabilities(policy: dict, function: dict) -> dict[str, float]:
    scores = policy.get(context_key(function), {})
    values = {a["id"]: scores.get(a["id"], {}).get("q", 0.) for a in ACTIONS}
    best = max(values.values())
    greedy = [action for action, value in values.items() if value == best]
    return {action: EPSILON / len(ACTIONS) + ((1 - EPSILON) / len(greedy)
            if action in greedy else 0.) for action in values}


def features(function: dict, action: str) -> tuple[float, ...]:
    context = [float(function["kind"] == kind) for kind in KINDS]
    complexity = [float(function["complexity"] == level) for level in COMPLEXITIES]
    package = [float(action == a["id"]) for a in ACTIONS]
    return tuple([
        *context, *complexity, *package,
        *(x * y for x in complexity for y in package),
        *(x * y for x in context for y in package),
        math.log1p(function["estimated_tokens"]), math.log1p(function["assigned_units"]),
        math.log1p(function["customer_budget_usd"]), math.log1p(function["provider_budget_usd"]),
        math.log1p(function["reference_benefit_usd"]), math.log1p(function["observation_days"]),
    ])


class OutcomeStore:
    """Transactional local pilot store; identity fields are attestations, not auth."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS models(
                    version INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
                    body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit(
                    seq INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
                    BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
                    BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
            """)
            row = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if row and row[0] != SCHEMA:
                raise ValueError("unsupported outcomes database schema")
            db.execute("INSERT OR IGNORE INTO meta VALUES('schema',?)", (SCHEMA,))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except (Exception, KeyboardInterrupt):
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def audit(db, kind: str, body: dict) -> None:
        db.execute("INSERT INTO audit(at,kind,body) VALUES(?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), kind, canonical(body)))

    @staticmethod
    def projects(db) -> list[dict]:
        return [json.loads(row[0]) for row in db.execute("SELECT body FROM projects ORDER BY rowid")]

    @staticmethod
    def decisions(db) -> list[dict]:
        return [json.loads(row[0]) for row in db.execute("SELECT body FROM decisions ORDER BY rowid")]

    @staticmethod
    def decision(db, decision_id: str) -> dict:
        decision_id = _text(decision_id, "decision_id")
        row = db.execute("SELECT body FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if row is None:
            raise ValueError("unknown decision")
        return json.loads(row[0])

    @staticmethod
    def function(db, project_id: str, function_id: str) -> tuple[dict, dict]:
        project_id = _text(project_id, "project_id")
        function_id = _text(function_id, "function_id")
        row = db.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise ValueError("unknown project")
        project = json.loads(row[0])
        for function in project["functions"]:
            if function["id"] == function_id:
                return project, function
        raise ValueError("unknown functionality in this project")

    @staticmethod
    def save_decision(db, value: dict) -> None:
        db.execute("UPDATE decisions SET body=? WHERE id=?", (canonical(value), value["id"]))

    @staticmethod
    def latest(db, source: str) -> dict | None:
        row = db.execute("SELECT body FROM models WHERE source=? ORDER BY version DESC LIMIT 1",
                         (source,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def active(db, source: str) -> dict:
        row = db.execute("SELECT value FROM meta WHERE key=?", (f"active:{source}",)).fetchone()
        if row:
            body = db.execute("SELECT body FROM models WHERE version=?", (int(row[0]),)).fetchone()
            return json.loads(body[0])
        return {"version": 0, "policy": {}, "policy_updates": 0}

    def create_project(self, payload: dict) -> dict:
        _keys(payload, {"title", "description", "source", "functions"})
        source = _enum(payload["source"], SOURCES, "source")
        items = payload["functions"]
        if not isinstance(items, list) or not 1 <= len(items) <= 50:
            raise ValueError("provide 1 to 50 explicit functionalities")
        items = [validate_function(item) for item in items]
        if len({f["id"] for f in items}) != len(items):
            raise ValueError("functionality IDs must be unique in a project")
        project = {"id": uuid.uuid4().hex, "title": _text(payload["title"], "title"),
                   "description": _text(payload["description"], "description"),
                   "source": source, "functions": items}
        with self.connection() as db:
            db.execute("INSERT INTO projects VALUES(?,?)", (project["id"], canonical(project)))
            self.audit(db, "project_created", project)
        return project

    def recommend(self, project_id: str, function_id: str, split: str = "train",
                  *, rng=None) -> dict:
        split = _enum(split, ("train", "validation"), "split")
        with self.connection() as db:
            project, function = self.function(db, project_id, function_id)
            prior = [d for d in self.decisions(db) if d["project_id"] == project_id]
            if any(d["split"] != split for d in prior):
                raise ValueError("all functionality decisions in a project must share the frozen split")
            if any(d["function_id"] == function_id and d["status"] != "rejected" for d in prior):
                raise ValueError("this work cohort already has a decision; create a new project for new work")
            active = self.active(db, project["source"])
            probs = probabilities(active["policy"], function)
            draw = (rng or random.SystemRandom()).random()
            if not 0 <= draw < 1:
                raise ValueError("random draw must be in [0,1)")
            cumulative, chosen = 0., ACTIONS[-1]["id"]
            for action, probability in probs.items():
                cumulative += probability
                if draw < cumulative:
                    chosen = action
                    break
            value = {
                "id": uuid.uuid4().hex, "project_id": project_id, "function_id": function_id,
                "source": project["source"], "action": chosen, "probabilities": probs,
                "policy_version": active["version"], "split": split,
                "status": "proposed", "approver": None, "feedback": None,
                "reward_spec": REWARD_SPEC, "context": context_key(function),
            }
            db.execute("INSERT INTO decisions VALUES(?,?)", (value["id"], canonical(value)))
            self.audit(db, "recommendation", value)
            return value

    def approve(self, decision_id: str, approver: str, approved: bool) -> dict:
        approver = _text(approver, "approver")
        _bool(approved, "approved")
        with self.connection() as db:
            decision = self.decision(db, decision_id)
            if decision["status"] != "proposed":
                raise ValueError("only proposed decisions may be approved or rejected")
            decision.update(status="approved" if approved else "rejected", approver=approver)
            self.save_decision(db, decision)
            self.audit(db, "decision_approval", decision)
            return decision

    @staticmethod
    def _frozen(db, decision_id: str) -> bool:
        for row in db.execute("SELECT body FROM models"):
            model = json.loads(row[0])
            if (decision_id in model.get("reward_ids", [])
                    or decision_id in model.get("evaluation_ids", [])):
                return True
        return False

    def feedback(self, payload: dict) -> dict:
        with self.connection() as db:
            decision = self.decision(db, _text(payload.get("decision_id"), "decision_id"))
            if decision["status"] != "approved":
                raise ValueError("feedback requires an approved delivery option")
            if self._frozen(db, decision["id"]):
                raise ValueError("reward/evaluation evidence is immutable; record a new work cohort")
            _, function = self.function(db, decision["project_id"], decision["function_id"])
            feedback = validate_feedback(payload, function)
            feedback["review"] = None
            feedback["signals"] = outcome_signals(function, feedback)
            decision["feedback"] = feedback
            self.save_decision(db, decision)
            self.audit(db, "feedback_revision", decision)
            return decision

    def review(self, decision_id: str, reviewer: str, approved: bool, notes: str) -> dict:
        reviewer, notes = _text(reviewer, "reviewer"), _text(notes, "notes")
        _bool(approved, "approved")
        with self.connection() as db:
            decision = self.decision(db, decision_id)
            if self._frozen(db, decision_id):
                raise ValueError("reward/evaluation review is immutable")
            if not decision["feedback"]:
                raise ValueError("feedback must exist before review")
            decision["feedback"]["review"] = {
                "reviewer": reviewer, "approved": approved, "notes": notes}
            self.save_decision(db, decision)
            self.audit(db, "outcome_review", decision)
            return decision

    def _rows(self, db, source: str, split: str) -> list[dict]:
        rows = []
        for decision in self.decisions(db):
            feedback = decision["feedback"]
            if (decision["source"] != source or decision["split"] != split or not feedback
                    or not feedback["review"] or not feedback["review"]["approved"]):
                continue
            _, function = self.function(db, decision["project_id"], decision["function_id"])
            rows.append({**decision, "function": function,
                         "targets": {**{key: feedback[key] for key in RATINGS},
                                     **feedback["signals"]}})
        return rows

    def train(self, source: str) -> dict:
        _enum(source, SOURCES, "source")
        with self.connection() as db:
            rows = self._rows(db, source, "train")
            if not rows:
                raise ValueError("no human-reviewed training feedback; validation never trains")
            digest = hashlib.sha256(canonical(rows).encode()).hexdigest()
            latest = self.latest(db, source)
            if latest and latest["training_digest"] == digest:
                return latest
            predictors, metrics, policy = {}, {}, {}
            for target in TARGETS:
                usable = [row for row in rows if row["targets"][target] is not None]
                records = [Record(features(row["function"], row["action"]),
                                  row["targets"][target], row["project_id"]) for row in usable]
                if records:
                    model = RidgeLinearModel.fit(records, alpha=5.)
                    predictors[target] = asdict(model)
                    metrics[target] = {"count": len(records), "mae": None, "baseline_mae": None,
                                       "training_mean": sum(r.target for r in records) / len(records)}
            reward_rows = [row for row in rows if row["targets"]["reward"] is not None]
            for row in reward_rows:
                cell = policy.setdefault(row["context"], {})
                score = cell.setdefault(row["action"], {"n": 0, "q": 0.})
                score["n"] += 1
                score["q"] += (row["targets"]["reward"] - score["q"]) / score["n"]
            value = {
                "source": source, "training_digest": digest, "training_count": len(rows),
                "trained_ids": [r["id"] for r in rows], "policy_updates": len(reward_rows),
                "reward_ids": [r["id"] for r in reward_rows],
                "training_rows": rows,
                "policy": policy, "predictors": predictors, "predictor_metrics": metrics,
                "status": "candidate", "evaluation": None, "evaluation_ids": [],
                "base_policy_version": self.active(db, source)["version"],
                "reward_spec": REWARD_SPEC,
            }
            cursor = db.execute("INSERT INTO models(source,body) VALUES(?,?)", (source, "{}"))
            value["version"] = cursor.lastrowid
            db.execute("UPDATE models SET body=? WHERE version=?",
                       (canonical(value), value["version"]))
            self.audit(db, "candidate_trained", value)
            return value

    def predict(self, project_id: str, function_id: str) -> dict:
        with self.connection() as db:
            project, function = self.function(db, project_id, function_id)
            model = self.latest(db, project["source"])
            active = self.active(db, project["source"])
            probs = probabilities(active["policy"], function)
            rows = model["training_rows"] if model else []
            options = []
            for action in probs:
                similar = [r for r in rows if r["action"] == action
                           and r["context"] == context_key(function)]
                groups = len({r["project_id"] for r in similar})
                numeric = ("estimated_tokens", "assigned_units", "customer_budget_usd",
                           "provider_budget_usd", "reference_benefit_usd", "observation_days")
                outside = [key for key in numeric if similar and not (
                    min(r["function"][key] for r in similar) <= function[key]
                    <= max(r["function"][key] for r in similar))]
                estimates: dict[str, float | None] = {target: None for target in TARGETS}
                if model and groups >= 3 and not outside:
                    for target, fitted in model["predictors"].items():
                        target_rows = [r for r in similar if r["targets"][target] is not None]
                        if len({r["project_id"] for r in target_rows}) < 3:
                            continue
                        point = RidgeLinearModel(**fitted).predict(features(function, action))
                        if target in RATINGS:
                            point = max(1., min(5., point))
                        elif target == "aes":
                            point = max(0., min(100., point))
                        estimates[target] = point
                options.append({
                    "action": action, "probability": probs[action], "estimates": estimates,
                    "support": {"status": "supported" if groups >= 10 and not outside
                                else "limited" if groups else "unseen",
                                "training_examples": len(similar), "independent_projects": groups,
                                "outside_observed_ranges": outside},
                    "uncertainty": "Associational pilot estimates; no calibrated intervals or causal ROI claim. "
                                   "At least 3 matching projects per target required. Compare held-out MAE.",
                })
            return {"source": project["source"], "model_version": model["version"] if model else None,
                    "options": options}

    def evaluate(self, source: str) -> dict:
        """Use each protected validation cohort once, never as a training label."""
        _enum(source, SOURCES, "source")
        with self.connection() as db:
            model = self.latest(db, source)
            if model is None:
                raise ValueError("train a candidate first")
            if model["evaluation"] is not None:
                return model
            used_projects = set()
            for row in db.execute("SELECT body FROM models WHERE source=?", (source,)):
                used_projects.update(json.loads(row[0]).get("evaluation_projects", []))
            rows = [r for r in self._rows(db, source, "validation")
                    if r["project_id"] not in used_projects and r["targets"]["reward"] is not None]
            if not rows:
                raise ValueError("no fresh reward-complete validation cohorts; collect or finish protected projects")
            active = self.active(db, source)
            if active["version"] != model["base_policy_version"]:
                raise ValueError("candidate baseline changed; retrain with new feedback")
            gains, weights = {}, {}
            rewarded = [r for r in rows if r["targets"]["reward"] is not None]
            for row in rewarded:
                action = row["action"]
                candidate = probabilities(model["policy"], row["function"])[action]
                baseline = probabilities(active["policy"], row["function"])[action]
                logged = row["probabilities"][action]
                gain = (candidate - baseline) / logged * row["targets"]["reward"]
                gains.setdefault(row["project_id"], []).append(gain)
                weights.setdefault(row["project_id"], []).append(candidate / logged)
            grouped_gains = [sum(v) / len(v) for v in gains.values()]
            grouped_weights = [sum(v) / len(v) for v in weights.values()]
            n = len(grouped_gains)
            mean = sum(grouped_gains) / n if n else None
            stderr = (math.sqrt(sum((v - mean) ** 2 for v in grouped_gains) / (n - 1) / n)
                      if n > 1 and mean is not None else None)
            lower = mean - 1.96 * stderr if stderr is not None and mean is not None else None
            ess = (sum(grouped_weights) ** 2 / sum(w * w for w in grouped_weights)
                   if grouped_weights else 0.)
            eligible = bool(n >= 20 and ess >= 10 and lower is not None and lower > 0
                            and model["policy_updates"] >= 20)
            model["evaluation"] = {
                "eligible": eligible,
                "reason": "Offline gain gate passed; explicit human promotion still required." if eligible
                else "Need >=20 reward-bearing projects, ESS>=10, >=20 training rewards and positive gain lower bound.",
                "validation_count": n, "effective_sample_size": ess,
                "estimated_reward_gain": mean, "lower_bound": lower,
                "interpretation": "Project-clustered approximate 95% normal lower bound; "
                                  "IPS estimate applies to approved options only, not causal business proof.",
            }
            for target, metric in model["predictor_metrics"].items():
                actual, errors, baseline_errors = [], [], []
                fitted = RidgeLinearModel(**model["predictors"][target])
                for row in rows:
                    truth = row["targets"][target]
                    if truth is not None:
                        prediction = fitted.predict(features(row["function"], row["action"]))
                        actual.append(truth)
                        errors.append(abs(prediction - truth))
                        baseline_errors.append(abs(metric["training_mean"] - truth))
                metric.update(mae=sum(errors) / len(errors) if errors else None,
                              baseline_mae=sum(baseline_errors) / len(baseline_errors)
                              if baseline_errors else None, validation_count=len(actual))
            model["evaluation_ids"] = [r["id"] for r in rows]
            model["evaluation_projects"] = sorted({r["project_id"] for r in rows})
            db.execute("UPDATE models SET body=? WHERE version=?",
                       (canonical(model), model["version"]))
            self.audit(db, "candidate_evaluated", model)
            return model

    def promote(self, source: str, approver: str) -> dict:
        _enum(source, SOURCES, "source")
        approver = _text(approver, "approver")
        with self.connection() as db:
            model = self.latest(db, source)
            if not model or not model["evaluation"] or not model["evaluation"]["eligible"]:
                raise ValueError("policy promotion blocked: protected evaluation gate has not passed")
            if model["status"] == "active":
                return model
            if self.active(db, source)["version"] != model["base_policy_version"]:
                raise ValueError("active baseline changed")
            model.update(status="active", promoted_by=approver)
            db.execute("UPDATE models SET body=? WHERE version=?",
                       (canonical(model), model["version"]))
            db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                       (f"active:{source}", str(model["version"])))
            self.audit(db, "human_policy_promotion", model)
            return model

    def state(self, source: str) -> dict:
        _enum(source, SOURCES, "source")
        with self.connection() as db:
            active = self.active(db, source)
            model = self.latest(db, source)
            if model:
                model = {key: value for key, value in model.items()
                         if key not in ("predictors", "training_rows")}
            return {
                "source": source, "actions": ACTIONS, "kinds": KINDS,
                "projects": [p for p in self.projects(db) if p["source"] == source],
                "decisions": [d for d in self.decisions(db) if d["source"] == source],
                "model": model, "active_policy_version": active["version"],
                "policy_updates": active["policy_updates"], "reward_spec": REWARD_SPEC,
            }
