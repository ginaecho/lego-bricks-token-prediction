"""Internal evidence rankings and requirement-constrained marketplace choices."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from .customer_outcomes import SOURCES, _enum, _keys, _mapping, _text, canonical
from .delivery_feedback import DeliveryFeedback, OPTIONAL_NUMBERS, action_key


MIN_PROJECTS = 3


def validate_catalog(catalog: list[dict]) -> list[dict]:
    if not isinstance(catalog, list) or not 1 <= len(catalog) <= 50:
        raise ValueError("catalog must contain 1 to 50 functionalities")
    cleaned = []
    for brick in catalog:
        brick = _mapping(brick, "catalog brick")
        _keys(brick, {"feature_id", "name", "options"})
        feature = _text(brick["feature_id"], "feature_id")
        name = _text(brick["name"], "name")
        if not isinstance(brick["options"], list) or not 1 <= len(brick["options"]) <= 20:
            raise ValueError("each brick must have 1 to 20 options")
        options = []
        for option in brick["options"]:
            option = _mapping(option, "brick option")
            _keys(option, {"variant_id", "model_id", "name"})
            options.append({key: _text(value, key) for key, value in option.items()})
        if len({action_key(option) for option in options}) != len(options):
            raise ValueError("duplicate brick option")
        cleaned.append({"feature_id": feature, "name": name, "options": options})
    if len({b["feature_id"] for b in cleaned}) != len(cleaned):
        raise ValueError("duplicate catalog functionality")
    return cleaned


def _score(groups: dict) -> tuple[int, float | None]:
    averages = [sum(values) / len(values) for values in groups.values()]
    n = len(averages)
    return n, (50 * (1 + sum(averages) / n) if n >= MIN_PROJECTS else None)


def _rank(rows: list[dict]) -> list[dict]:
    rows.sort(key=lambda row: -(row["score"] if row["score"] is not None else -1))
    previous, rank = None, None
    for index, row in enumerate(rows):
        if row["score"] is not None:
            if previous is None or abs(previous - row["score"]) > 1e-9:
                rank = index + 1
            previous = row["score"]
            row["rank"] = rank
        else:
            row["rank"] = None
    return rows


class BrickLearning:
    def __init__(self, deliveries: DeliveryFeedback):
        self.deliveries = deliveries

    def rankings(self, source: str, stage: str, catalog: list[dict]) -> dict:
        """Rank descriptive experience, not interchangeability or causal ROI."""
        _enum(source, SOURCES, "source")
        _enum(stage, ("candidate", "active"), "stage")
        catalog = validate_catalog(catalog)
        with self.deliveries.store.connection() as db:
            active = self.deliveries._active(db, source)
            model = self.deliveries._latest(db, source) if stage == "candidate" else active
            records = self.deliveries._records(db, source)
        snapshots = (model or {}).get("training_snapshots", [])
        functions, options = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        allowed = {brick["feature_id"]: {action_key(o) for o in brick["options"]} for brick in catalog}
        for record in snapshots:
            responses = {r["function_id"]: r["signals"]["experience_reward"] for r in record["feedback"]}
            for function in record["handoff"]["functions"]:
                feature, action = function["feature_id"], action_key(function)
                reward = responses[function["id"]]
                if reward is None or action not in allowed.get(feature, set()):
                    continue
                receipt = record["handoff"]["receipt_id"]
                functions[feature][receipt].append(reward)
                options[(feature, action)][receipt].append(reward)
        rows = []
        for brick in catalog:
            feature = brick["feature_id"]
            count, score = _score(functions[feature])
            variants = []
            for option in brick["options"]:
                support, option_score = _score(options[(feature, action_key(option))])
                variants.append({**option, "project_count": support, "score": option_score})
            rows.append({"feature_id": feature, "name": brick["name"], "score": score,
                         "project_count": count, "status": "learned" if score is not None else "insufficient_evidence",
                         "options": _rank(variants)})
        return {"source": source, "stage": stage, "model_version": (model or {}).get("version", 0),
                "active_version": active["version"], "minimum_projects": MIN_PROJECTS,
                "validation_projects": sum(r["split"] == "validation" for r in records),
                "rows": _rank(rows)}

    def recommend(self, source: str, catalog: list[dict], matches: list[dict]) -> dict:
        """Keep every declared requirement; sample only its eligible options."""
        _enum(source, SOURCES, "source")
        catalog = validate_catalog(catalog)
        if not isinstance(matches, list) or not 1 <= len(matches) <= 50:
            raise ValueError("provide 1 to 50 reviewed requirement matches from Studio")
        bricks = {b["feature_id"]: b for b in catalog}
        eligible, seen = [], set()
        for match in matches:
            match = _mapping(match, "requirement match")
            _keys(match, {"feature_id", "variant_ids", "reason"})
            feature = _text(match["feature_id"], "feature_id")
            reason = _text(match["reason"], "reason")
            variants = match["variant_ids"]
            if not isinstance(variants, list) or not variants or any(not isinstance(v, str) for v in variants):
                raise ValueError("variant_ids must be a nonempty string list")
            if feature not in bricks or not set(variants) <= {o["variant_id"] for o in bricks[feature]["options"]}:
                raise ValueError("requirement match contains unavailable functionality or variant")
            for variant in variants:
                if (feature, variant) in seen:
                    raise ValueError("overlapping requirement matches would double-count a variant")
                seen.add((feature, variant))
            eligible.append((bricks[feature], reason, [o for o in bricks[feature]["options"] if o["variant_id"] in variants]))
        ranked = self.rankings(source, "active", catalog)
        scores = {r["feature_id"]: r for r in ranked["rows"]}
        recommendations = []
        for brick, reason, options in eligible:
            choice = self.deliveries.suggest(source, brick["feature_id"], [
                {"variant_id": o["variant_id"], "model_id": o["model_id"]} for o in options])
            if choice["policy_version"] != ranked["active_version"]:
                raise ValueError("active policy changed during recommendation; request fresh suggestions")
            option = next(o for o in scores[brick["feature_id"]]["options"] if action_key(o) == choice["action"])
            recommendations.append({
                "feature_id": brick["feature_id"], "name": brick["name"], "reason": reason,
                **choice["option"], "option_name": option["name"], "decision_id": choice["id"],
                "probability": choice["probabilities"][choice["action"]],
                "score": option["score"], "project_count": option["project_count"],
            })
        recommendations.sort(key=lambda r: scores[r["feature_id"]]["rank"] or 1_000_000)
        return {"source": source, "policy_version": ranked["active_version"], "recommendations": recommendations}

    def seed_demo(self, catalog: list[dict]) -> dict:
        """Generate fictitious feedback through the real loop, without promotion."""
        catalog = validate_catalog(catalog)
        digest = hashlib.sha256(canonical(catalog).encode()).hexdigest()[:12]
        prefix = f"brick-demo-v1-{digest}"
        existing = {r["receipt_id"] for r in self.deliveries.listing(include_examples=True)["deliveries"]}
        rng = random.Random(491)
        seeded = 0

        def save(index: int, brick: dict, option: dict | None, split: str):
            nonlocal seeded
            suffix = 0
            while True:
                receipt = f"{prefix}-{split}-{index}-{suffix}"
                is_validation = int(hashlib.sha256(receipt.encode()).hexdigest()[:8], 16) % 5 == 0
                if is_validation == (split == "validation"):
                    break
                suffix += 1
            if receipt in existing:
                with self.deliveries.store.connection() as db:
                    db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                               (f"delivery-training-example:{receipt}", "true"))
                return
            decision = None
            if option is None:
                decision = self.deliveries.suggest("synthetic", brick["feature_id"], [
                    {"variant_id": o["variant_id"], "model_id": o["model_id"]} for o in brick["options"]], rng=rng)
                option = next(o for o in brick["options"] if action_key(o) == decision["action"])
            preferred_model = "gpt" if brick["feature_id"] == "research" else "gpt-mini"
            good = option["model_id"] == preferred_model
            satisfaction = 5 if good and brick["feature_id"] in ("research", "documents") else 4 if good else 1
            function = {
                "id": brick["feature_id"], "feature_id": brick["feature_id"], "variant_id": option["variant_id"],
                "name": brick["name"], "model_id": option["model_id"], "model_name": option["name"],
                "estimated_tokens": 5000, "runs": 100, "decision_id": decision["id"] if decision else None,
            }
            self.deliveries.submit({
                "source": "synthetic", "respondent": "Fictitious ranking walkthrough",
                "delivered": True, "consent": True, "sow_quality": satisfaction,
                "staffing_fit": satisfaction, "observation_days": 30,
                "handoff": {"schema": "token-yield-delivery-handoff-v1", "receipt_id": receipt,
                            "title": "SYNTHETIC ranking example", "description": "Not an actual customer delivery.",
                            "flow": "manual", "quote_basis": "synthetic demonstration",
                            "functions": [function], "staffing": [], "estimated_project_tokens": 500000},
                "feedback": [{"function_id": function["id"], "satisfaction": satisfaction,
                              "kept": "all" if good else "some", "usefulness": "yes" if good else "no",
                              "comments": "Fictitious outcome for the ranking walkthrough.",
                              **dict.fromkeys(OPTIONAL_NUMBERS)}],
            })
            with self.deliveries.store.connection() as db:
                db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                           (f"delivery-training-example:{receipt}", "true"))
            existing.add(receipt)
            seeded += 1

        index = 0
        for brick in catalog:
            for option in brick["options"]:
                if option["model_id"] not in ("gpt-mini", "gpt"):
                    continue
                for _ in range(3):
                    save(index, brick, option, "train")
                    index += 1
        for index in range(140):
            save(index, catalog[index % len(catalog)], None, "validation")
        return {"seeded": seeded, "source": "synthetic"}
