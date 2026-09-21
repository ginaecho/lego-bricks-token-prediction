"""Deterministic synthetic outcome episodes, never customer performance evidence."""

from __future__ import annotations

import random
import sqlite3
import uuid

from token_yield.customer_outcomes import (
    ACTIONS, COMPLEXITIES, REWARD_SPEC, OutcomeStore, canonical, context_key,
    outcome_signals, probabilities, validate_feedback, validate_function,
)


def seed_demo(store: OutcomeStore) -> dict:
    """Atomically seed validated fixtures; synthetic approvals are not people."""
    with store.connection() as db:
        if db.execute("SELECT value FROM meta WHERE key='synthetic-fixture-v1'").fetchone():
            return {"source": "synthetic", "inserted": 0, "message": "Synthetic fixture already exists."}
        rng = random.Random(721)
        for index in range(540):
            complexity = COMPLEXITIES[index % 3]
            kind = ("sow", "staffing")[index % 2]
            draw = rng.random()
            noise = rng.uniform(-.06, .06)
            _insert_fixture(store, db, index, complexity, kind, draw, noise)
        db.execute("INSERT INTO meta VALUES('synthetic-fixture-v1','540')")
    return {"source": "synthetic", "inserted": 540,
            "message": "Synthetic episodes only. Train, evaluate, and promote explicitly."}


def _insert_fixture(store: OutcomeStore, db: sqlite3.Connection, index: int,
                    complexity: str, kind: str, draw: float, noise: float) -> None:
    function = validate_function({
        "id": "scope" if kind == "sow" else "team",
        "name": "Statement of work" if kind == "sow" else "Staffing plan",
        "kind": kind, "complexity": complexity, "assigned_units": 100,
        "estimated_tokens": 10000, "customer_budget_usd": 3000,
        "provider_budget_usd": 2000, "reference_benefit_usd": 5000,
        "observation_days": 30,
    })
    project = {
        "id": uuid.uuid4().hex,
        "title": f"SYNTHETIC outcome trial {index + 1:03d}",
        "description": "SYNTHETIC 30-day business outcome scenario. "
        "Generated rewards demonstrate software behavior, not commercial results.",
        "source": "synthetic", "functions": [function]}
    decision = {
        "id": uuid.uuid4().hex, "project_id": project["id"], "function_id": function["id"],
        "source": "synthetic", "action": ACTIONS[min(2, int(draw * 3))]["id"],
        "probabilities": probabilities({}, function), "policy_version": 0,
        "split": "train" if index < 180 else "validation",
        "status": "approved", "approver": "Synthetic fixture approver",
        "reward_spec": REWARD_SPEC, "context": context_key(function),
    }
    optimal = ACTIONS[COMPLEXITIES.index(complexity)]["id"]
    match = decision["action"] == optimal
    quality = max(.04, min(.99, (.93 if match else .16) + noise))
    completed = int(100 * quality)
    kept = int(completed * quality)
    useful = int(kept * quality)
    rating = max(1., min(5., 1 + 4 * quality))
    feedback = validate_feedback({
        "decision_id": decision["id"], "respondent": "Synthetic customer",
        "completed_units": completed, "kept_units": kept, "useful_units": useful,
        "actual_tokens": 10000 + 1000 * ACTIONS.index(next(
            a for a in ACTIONS if a["id"] == decision["action"])),
        "satisfaction": rating, "sow_quality": rating, "staffing_fit": rating,
        "impact": rating, "customer_benefit_usd": 9000 * quality,
        "customer_cost_usd": 2500, "provider_revenue_usd": 2500,
        "provider_cost_usd": 1400 if match else 1900,
        "financial_evidence": "verified", "window_closed": True,
        "safety_ok": True, "budget_ok": True,
        "evidence": "SYNTHETIC fixture only; not independently verified business data",
        "comments": ("SOW scope and staffing matched the need; useful value." if match else
                     "SOW scope and staffing needed rework; low value."),
    }, function)
    feedback["signals"] = outcome_signals(function, feedback)
    feedback["review"] = {
        "reviewer": "Synthetic fixture reviewer", "approved": True,
        "notes": "Approved synthetic fixture; exclude from real customer models."}
    decision["feedback"] = feedback
    db.execute("INSERT INTO projects VALUES(?,?)", (project["id"], canonical(project)))
    db.execute("INSERT INTO decisions VALUES(?,?)", (decision["id"], canonical(decision)))
    store.audit(db, "synthetic_fixture", {"project": project, "decision": decision})
