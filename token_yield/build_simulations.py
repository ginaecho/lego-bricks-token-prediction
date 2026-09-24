"""Construction-token corpus, separate from workload-execution measurements."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import sqlite3
from pathlib import Path

from .marketplace_agent_contracts import ATOMS, contracts, fingerprint

PROTOCOL = "agent-software-build-v1"
FAMILIES = {
    "recommend": "Personalized discovery", "research": "Research",
    "support": "AI customer support", "search": "Smart product search",
    "insights": "Customer insights", "documents": "Document automation",
    "onboard": "Guided onboarding",
}
REWARD_WEIGHTS = {
    "client_satisfaction": .30, "impact_recognized": .25,
    "roi_satisfaction": .20, "future_impact_potential": .10, "delivery_quality": .15,
}
STAFF = {
    "staff_data_scientist": .20, "staff_architect": .10,
    "staff_consultant": .10, "staff_software_engineer": .50,
}
BUILD_SCOPE = {
    "interests": "Preference ranking, exclusion enforcement and grounded explanations.",
    "behavior": "Recency-weighted browsing/purchase ranking with cold-start handling.",
    "journey": "Prerequisite-aware next actions and a validated two-step journey.",
    "normal": "Passage retrieval and extractive findings with exact source citations.",
    "web": "Allowlisted URL ingestion, retrieval and provenance-preserving findings.",
    "deep": "Multi-document evidence synthesis, disagreement and unresolved questions.",
    "faq": "Knowledge-base retrieval, grounded answers and explicit abstention.",
    "triage": "Configurable ticket categorization, priority and accountable routing.",
    "semantic": "Search index, relevance ranking and optional injected embedding interface.",
    "compare": "Attribute normalization, side-by-side comparison and preference ranking.",
    "feedback": "Feedback deduplication, themes and traceable supporting excerpts.",
    "sentiment": "Transparent sentiment scoring and severity-aware issue prioritization.",
    "extract": "Schema-driven field extraction, source spans and missing-field reporting.",
    "review": "Requirement/evidence checking and traceable gaps without certification claims.",
    "guided": "Validated prerequisite-driven onboarding steps and progress tracking.",
    "adaptive": "Experience/preference-aware onboarding with prerequisites and explanations.",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def point_id(types: list[str]) -> str:
    return ("single_" if len(types) == 1 else f"combo{len(types)}_") + "_".join(types)


def build_spec(types: list[str]) -> dict:
    registry = {item["id"]: item for item in contracts()}
    if (not 1 <= len(types) <= 4 or len(set(types)) != len(types)
            or any(item not in registry for item in types)):
        raise ValueError("Choose one to four distinct catalog types")
    selected = [registry[item] for item in types]
    if len({item["feature_id"] for item in selected}) != len(types):
        raise ValueError("This design combines distinct basic functionalities")
    count = len(types)
    months = .5 + .25 * (count - 1)
    features = {
        "basic_functionality_ids": [item["feature_id"] for item in selected],
        "basic_functionalities": [FAMILIES[item["feature_id"]] for item in selected],
        "types": types, "functionality_count": count,
        "integration_edge_count": count - 1,
        "shared_schema_count": 1, "shared_validation_layer_count": 1,
        "planned_acceptance_case_count": 6 + 3 * (count - 1),
        "artifact_kind": "python_cli_reference_implementation",
        "llm_integration": "injected_callable_validated_with_test_fixtures",
        "estimated_months_to_finish": months,
        **{name: round(value * (1 + .25 * (count - 1)), 3) for name, value in STAFF.items()},
        **{f"has_type_{item}": int(item in types) for item in registry},
        **{f"planned_operation_{atom}": sum(item["atoms"][atom] for item in selected) for atom in ATOMS},
    }
    group = fingerprint(sorted(types))
    bucket = int(group[:8], 16) % 10
    return {
        "id": point_id(types), "protocol": PROTOCOL,
        "origin": "catalog_designed_simulation", "catalog_origin": "independent_functionality_taxonomy",
        "source_case_ids": [], "input_features": features,
        "requirements": [BUILD_SCOPE[item] for item in types],
        "integration": {
            "ordered_types": types,
            "edges": [{"from": left, "to": right, "kind": "data_handoff"}
                      for left, right in zip(types, types[1:])],
            "instruction": "Build one integrated implementation, not a sum or concatenation of standalone token estimates.",
        },
        "planning_assumptions": {
            "source": "declared_scenario_not_observed_staffing",
            "staff_unit": "FTE", "working_hours_per_fte_month": 160,
            "staff_headcounts": {name: math.ceil(features[name]) for name in STAFF},
            "duration_rule": "0.5 + 0.25 * (functionality_count - 1) months; not measured delivery duration",
            "causal_status": "These staffing/duration values were not experimental interventions on the builders.",
        },
        "split": "test" if bucket >= 8 else "validation" if bucket == 7 else "train",
        "split_group": group,
        "split_rule": "Keep repeated/reordered builds of the same membership in one partition.",
    }


def combination_design() -> list[dict]:
    ids = [item["id"] for item in contracts()]
    result = []
    for size in range(1, 5):
        for members in itertools.combinations(ids, size):
            parents = {item["feature_id"] for item in contracts() if item["id"] in members}
            if len(parents) != size:
                continue
            spec = build_spec(list(members))
            result.append({"id": spec["id"], "types": list(members), "status": "planned_unmeasured",
                           "split": spec["split"], "split_group": spec["split_group"]})
    return result


def pseudo_outcomes(identifier: str) -> dict:
    rng = random.Random(int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16))
    scores = {name: rng.randint(0, 5) for name in REWARD_WEIGHTS}
    annual_cost = rng.randrange(15_000, 90_001, 5_000)
    annual_benefit = rng.randrange(10_000, 160_001, 5_000)
    return {
        "source": "synthetic_client_feedback", "is_pseudo": True,
        "feedback_schema": "build-outcomes-0to5-v1",
        "scores_0_to_5": scores, "reward_weights": REWARD_WEIGHTS,
        "reward_0_to_1": reward(scores),
        "pseudo_annual_cost_usd": annual_cost, "pseudo_annual_ai_value_usd": annual_benefit,
        "pseudo_net_impact_usd": annual_benefit - annual_cost,
        "pseudo_roi_pct": 100 * (annual_benefit - annual_cost) / annual_cost,
        "verified_client_roi_pct": None,
        "evidence": "Deterministic fictional scenario; no client delivery or financial verification.",
        "training_use": "Synthetic reward-policy experiments only; excluded from token-regression inputs and real-outcome policies.",
        "policy_promoted": False,
        "logged_action_propensity": None,
        "off_policy_evaluation_eligible": False,
    }


def reward(scores: dict) -> float:
    if set(scores) != set(REWARD_WEIGHTS):
        raise ValueError("Every declared reward dimension is required")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 5
           for value in scores.values()):
        raise ValueError("Feedback scores must be finite numbers from 0 to 5")
    return sum(scores[name] * weight / 5 for name, weight in REWARD_WEIGHTS.items())


def read_usage(database: Path, session_id: str, agent_id: str) -> list[dict]:
    """Read only the named builder's counters, never prompts or other sessions."""
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT id, agent_id, model, input_tokens, output_tokens, cache_read_tokens, "
            "cache_write_tokens, reasoning_tokens, token_details_json, duration_ms, created_at "
            "FROM assistant_usage_events WHERE session_id=? AND agent_id=? ORDER BY id",
            (session_id, agent_id),
        ).fetchall()
    if not rows:
        raise ValueError("No authoritative usage events for this builder; do not invent token labels")
    return [dict(row) for row in rows]


def measured_usage(rows: list[dict], agent_id: str) -> dict:
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Distinct authoritative usage events are required")
    cleaned = []
    for row in rows:
        if row["agent_id"] != agent_id:
            raise ValueError("Usage from another agent cannot be attributed to this build")
        names = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        if any(type(row[name]) is not int or row[name] < 0 for name in names):
            raise ValueError("Token counters must be nonnegative reported integers")
        details = json.loads(row["token_details_json"])
        input_count = sum(item["tokenCount"] for item in details
                          if item["tokenType"] in ("input", "cache_read", "cache_write"))
        output_count = sum(item["tokenCount"] for item in details if item["tokenType"] == "output")
        if input_count != row["input_tokens"] or output_count != row["output_tokens"]:
            raise ValueError("Counters disagree with authoritative token-type breakdown")
        for field, category in (("cache_read_tokens", "cache_read"), ("cache_write_tokens", "cache_write")):
            if row[field] != sum(item["tokenCount"] for item in details if item["tokenType"] == category):
                raise ValueError("Cache counters disagree with the token breakdown")
        cleaned.append({name: row[name] for name in
                        ("id", "agent_id", "model", *names, "reasoning_tokens", "duration_ms", "created_at")})
    totals = {name: sum(row[name] for row in cleaned) for name in names}
    return {
        "source": "copilot_runtime_assistant_usage_events", "builder_agent_id": agent_id,
        "builder_models": sorted({row["model"] for row in cleaned}),
        "provider_call_count": len(cleaned), **totals,
        "total_tokens": totals["input_tokens"] + totals["output_tokens"],
        "input_includes_cache_read_and_write": True,
        "reasoning_policy": "Reported separately as auxiliary detail; never added again to output tokens.",
        "measurement_scope": "Entire isolated build-agent session, including prompts, tool context, implementation, tests, repairs and final report.",
        "not_included": ["parent orchestration", "other agents", "future deployed workload inference"],
        "actual_billed_usd": None,
        "billing_basis": "Authenticated Copilot subscription/quota; not an Azure invoice or a claim of free compute.",
        "events": cleaned,
    }


def price_usage(usage: dict, card: dict) -> dict:
    no_cache, observed_cache = 0., 0.
    for event in usage["events"]:
        rates = card["short_context"]
        if event["input_tokens"] > card["short_context_max_input_tokens"]:
            rates = card.get("long_context")
            if rates is None:
                raise ValueError("No verified rate for this request's context tier")
        input_tokens, output_tokens = event["input_tokens"], event["output_tokens"]
        cached = event["cache_read_tokens"] if rates.get("cached_input_per_million") is not None else 0
        no_cache += (input_tokens * rates["input_per_million"]
                     + output_tokens * rates["output_per_million"]) / 1_000_000
        observed_cache += ((input_tokens - cached) * rates["input_per_million"]
                           + cached * (rates.get("cached_input_per_million") or 0)
                           + output_tokens * rates["output_per_million"]) / 1_000_000
    return {
        "target_model": card["model"], "currency": "USD",
        "no_cache_estimate_usd": no_cache, "observed_cache_profile_estimate_usd": observed_cache,
        "rate_source": card["source_url"], "rate_status": card["status"],
        "basis": "Counterfactual price of the observed token volume, not an execution of the target model.",
        "limitations": [
            "Other models/tokenizers can require different tokens and tool iterations.",
            "Azure cache eligibility may differ from observed Copilot cache usage.",
            "Not a tenant quote, actual subscription charge, tax estimate, or total delivery cost.",
        ],
    }


def planned_point(types: list[str]) -> dict:
    spec = build_spec(types)
    return {
        **spec, "spec_sha256": fingerprint(spec), "status": "planned_unmeasured",
        "build_token_usage": None, "azure_cost_scenarios": [],
        "pseudo_outcomes": pseudo_outcomes(spec["id"]), "build_evidence": None,
    }


def complete_point(point: dict, rows: list[dict], agent_id: str, cards: list[dict],
                   evidence: dict) -> dict:
    if point.get("spec_version") == "composition-levels-v3":
        from .build_waves_v3 import build_spec_v3
        features = point["input_features"]
        spec_keys = build_spec_v3(features["parts"], features["industry"]).keys()
    else:
        spec_keys = build_spec(point["input_features"]["types"]).keys()
    if fingerprint({key: point[key] for key in spec_keys}) != point["spec_sha256"]:
        raise ValueError("Build feature/specification record changed without a new version")
    if evidence.get("tests_passed") is not True or not evidence.get("artifact_sha256"):
        raise ValueError("A tested implementation and artifact hashes are required")
    usage = measured_usage(rows, agent_id)
    return {**point, "status": "measured_build_passed", "build_token_usage": usage,
            "azure_cost_scenarios": [price_usage(usage, card) for card in cards],
            "build_evidence": evidence}
