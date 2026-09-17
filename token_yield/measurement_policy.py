"""Audited epsilon-greedy measurement bandit, not LLM reinforcement fine-tuning.

Use MeasurementPolicy.choose(), then feedback() or reject() exactly once.
The caller owns execution, telemetry validation and train/calibration/test splits.
SQLite serializes decisions and append-only hash-linked audit records.
"""

from __future__ import annotations

import math
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .marketplace_agent_contracts import canonical, fingerprint, strict_json

POLICY_VERSION = "measurement-epsilon-greedy-v1"
EPSILON = .2
ROUNDS = 4
REFERENCE_USD = .001
SPLIT_PROTOCOL = "train-012-repeat0-calibration3-final45-v1"
SOURCES = {"mocked-test-provider", "measured-foundry"}


def reward_value(before_mae: float, after_mae: float, cost_usd: float) -> float:
    """Return clipped fractional calibration improvement per reference dollar.

    Missing/zero cost and invalid metrics are errors, never neutral rewards.
    Negative rewards retain predictor regressions. Calibration is not a final
    test or a generalization claim.
    """
    for value in (before_mae, after_mae, cost_usd):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative metrics and cost required")
    if cost_usd == 0:
        raise ValueError("positive measured cost required; reward unavailable")
    efficiency = (before_mae - after_mae) / max(before_mae, 1.0) * REFERENCE_USD / cost_usd
    return max(-1.0, min(1.0, efficiency))


class MeasurementPolicy:
    """Persist sample-mean action rewards and exact decision propensities.

    This is a non-contextual bandit within a source/contract/model partition.
    Context is audited, not fitted by a contextual model. Unknown pending
    decisions block another choice until explicitly settled or rejected.
    """

    def __init__(self, path: Path, *, compatibility: str, source: str, actions: list[str]):
        if (source not in SOURCES or not isinstance(compatibility, str) or not compatibility
                or not 1 <= len(actions) <= 3 or len(set(actions)) != len(actions)
                or any(not isinstance(a, str) or not a for a in actions)):
            raise ValueError("finite unique actions and explicit compatibility/provenance required")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = {"version": POLICY_VERSION, "compatibility": compatibility, "source": source,
                       "actions": list(actions), "epsilon": EPSILON, "reference_usd": REFERENCE_USD,
                       "split_protocol": SPLIT_PROTOCOL,
                       "cost_basis": "simulated-rate-card" if source == "mocked-test-provider"
                       else "rated-provider-usage"}
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS events "
                       "(seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, "
                       "previous TEXT NOT NULL, digest TEXT NOT NULL)")
            db.execute("CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events "
                       "BEGIN SELECT RAISE(ABORT, 'immutable policy audit'); END")
            db.execute("CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events "
                       "BEGIN SELECT RAISE(ABORT, 'immutable policy audit'); END")
            events = self._events(db)
            if not events:
                self._append(db, "config", self.config)
            elif events[0] != ("config", self.config):
                raise ValueError("policy compatibility/provenance mismatch")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=0, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    @staticmethod
    def _events(db) -> list[tuple[str, dict]]:
        events, previous = [], ""
        for kind, text, parent, digest in db.execute(
                "SELECT kind, body, previous, digest FROM events ORDER BY seq"):
            body = strict_json(text)
            if parent != previous or digest != fingerprint(
                    {"kind": kind, "body": body, "previous": previous}):
                raise ValueError("policy audit integrity failure")
            events.append((kind, body))
            previous = digest
        return events

    @staticmethod
    def _append(db, kind: str, body: dict) -> None:
        last = db.execute("SELECT digest FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = last[0] if last else ""
        digest = fingerprint({"kind": kind, "body": body, "previous": previous})
        db.execute("INSERT INTO events(kind,body,previous,digest) VALUES(?,?,?,?)",
                   (kind, canonical(body), previous, digest))

    def _state(self, events: list[tuple[str, dict]]) -> tuple[dict, dict, dict]:
        scores = {a: {"n": 0, "q": 0.0} for a in self.config["actions"]}
        decisions, settled = {}, {}
        for kind, body in events[1:]:
            key = body["decision_id"]
            if kind == "decision":
                decisions[key] = body
            elif kind in ("feedback", "rejected"):
                settled[key] = (kind, body)
                if kind == "feedback":
                    score = scores[decisions[key]["action"]]
                    score["n"] += 1
                    score["q"] += (body["reward"] - score["q"]) / score["n"]
            else:
                raise ValueError("unknown policy audit event")
        return scores, decisions, settled

    def snapshot(self) -> dict:
        with self._connection() as db:
            events = self._events(db)
            scores, decisions, settled = self._state(events)
            return {**self.config, "scores": scores, "decisions": len(decisions),
                    "updates": sum(v["n"] for v in scores.values()),
                    "pending": sorted(set(decisions) - set(settled))}

    def choose(self, decision_id: str, context: dict, *, rng=None) -> dict:
        if not isinstance(decision_id, str) or not decision_id or not isinstance(context, dict):
            raise ValueError("decision identity and finite JSON context required")
        context = strict_json(canonical(context))
        with self._connection() as db:
            scores, decisions, settled = self._state(self._events(db))
            if decision_id in decisions:
                if decisions[decision_id]["context"] != context:
                    raise ValueError("decision identity reused with different context")
                return decisions[decision_id]
            if set(decisions) - set(settled):
                raise RuntimeError("unsettled measurement decision; no new choice permitted")
            actions = self.config["actions"]
            unseen = [a for a in actions if scores[a]["n"] == 0]
            if unseen:
                probabilities = {a: 1 / len(unseen) if a in unseen else 0 for a in actions}
                mode = "bounded-cold-start-exploration"
            else:
                best = max(s["q"] for s in scores.values())
                greedy = [a for a in actions if scores[a]["q"] == best]
                probabilities = {a: EPSILON / len(actions) +
                                 ((1 - EPSILON) / len(greedy) if a in greedy else 0)
                                 for a in actions}
                mode = "epsilon-greedy"
            draw = (rng or random.SystemRandom()).random()
            if not 0 <= draw < 1:
                raise ValueError("random draw must lie in [0,1)")
            cumulative, selected = 0.0, actions[-1]
            for action in actions:
                cumulative += probabilities[action]
                if draw < cumulative:
                    selected = action
                    break
            body = {"decision_id": decision_id, "context": context, "action": selected,
                    "propensity": probabilities[selected], "probabilities": probabilities,
                    "draw": draw, "mode": mode, "scores_before": scores,
                    "baseline": {"policy": "uniform-random",
                                 "probabilities": {a: 1 / len(actions) for a in actions}},
                    "source": self.config["source"], "cost_basis": self.config["cost_basis"]}
            self._append(db, "decision", body)
            return body

    def feedback(self, decision_id: str, *, before_mae: float, after_mae: float,
                 cost_usd: float, measurements: list[dict], predictor_before: str,
                 predictor_after: str, calibration_ids: list[str]) -> dict:
        reward = reward_value(before_mae, after_mae, cost_usd)
        if (not measurements or not predictor_before or not predictor_after
                or not calibration_ids or len(set(calibration_ids)) != len(calibration_ids)):
            raise ValueError("measurement, predictor and separate calibration evidence required")
        for item in measurements:
            if (item.get("source") != self.config["source"] or not item.get("id")
                    or item.get("status") not in ("measured", "validated")
                    or type(item.get("rated_usd")) not in (int, float)
                    or not math.isfinite(item["rated_usd"]) or item["rated_usd"] <= 0
                    or any(type(item.get("usage", {}).get(k)) is not int or item["usage"][k] < 0
                           for k in ("input_tokens", "output_tokens"))):
                raise ValueError("complete measured telemetry and matching provenance required")
        ids = [m["id"] for m in measurements]
        if len(set(ids)) != len(ids) or set(ids) & set(calibration_ids):
            raise ValueError("duplicate measurement or calibration leakage")
        if not math.isclose(sum(m["rated_usd"] for m in measurements), cost_usd,
                            rel_tol=1e-12, abs_tol=0):
            raise ValueError("reward cost must include all measured attempts")
        body = {"decision_id": decision_id, "before_mae": before_mae, "after_mae": after_mae,
                "cost_usd": cost_usd, "reward": reward, "measurements": measurements,
                "predictor_before": predictor_before, "predictor_after": predictor_after,
                "calibration_ids": calibration_ids, "source": self.config["source"]}
        return self._settle("feedback", body)

    def reject(self, decision_id: str, reason: str) -> dict:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("explicit failure/cancellation reason required")
        return self._settle("rejected", {"decision_id": decision_id, "reason": reason,
                                        "reward": None, "source": self.config["source"]})

    def _settle(self, kind: str, body: dict) -> dict:
        with self._connection() as db:
            _, decisions, settled = self._state(self._events(db))
            key = body["decision_id"]
            if key not in decisions:
                raise ValueError("feedback has no recorded decision")
            if key in settled:
                if settled[key] != (kind, body):
                    raise ValueError("conflicting duplicate feedback")
                return body
            if kind == "feedback":
                used = {m["id"] for old_kind, old_body in settled.values() if old_kind == "feedback"
                        for m in old_body["measurements"]}
                if used & {m["id"] for m in body["measurements"]}:
                    raise ValueError("measurement already credited to another decision")
            self._append(db, kind, body)
            return body
