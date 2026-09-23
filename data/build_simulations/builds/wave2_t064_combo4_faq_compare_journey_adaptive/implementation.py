"""Synthetic, deterministic reference pipeline. Python standard library only."""

import copy
import json
import math
import re
import sys
from pathlib import Path


SCHEMA_VERSION = 1
PRODUCTS = {
    "atlas": {"name": "Synthetic Atlas", "price_usd": 480, "storage": "128 GB",
              "weight": "0.18 kg", "battery": "4 Ah"},
    "beacon": {"name": "Synthetic Beacon", "price_usd": 640, "storage": "0.256 TB",
               "weight": "210 g", "battery": "5000 mAh"},
    "comet": {"name": "Synthetic Comet", "price_usd": 390, "storage": "64 GB",
              "weight": "160 g", "battery": "3.5 Ah"},
}
KB = [
    {"id": "travel", "terms": ["travel", "roaming", "esim"],
     "text": "All three synthetic phones support eSIM for travel."},
    {"id": "returns", "terms": ["return", "returns", "refund"],
     "text": "Unopened synthetic phones can be returned within 14 days."},
    {"id": "storage", "terms": ["storage", "capacity", "photos"],
     "text": "Synthetic phone storage is fixed and cannot be expanded."},
]
PRIORITIES = ("price", "storage", "weight", "battery")
INTERESTS = ("photography", "travel", "productivity")
ACTIONS = {
    "choose_product": (),
    "learn_basics": ("choose_product",),
    "create_account": ("choose_product",),
    "configure_device": ("create_account",),
    "set_preferences": ("configure_device",),
    "explore_camera": ("configure_device",),
    "enable_travel": ("configure_device",),
    "enable_backup": ("configure_device",),
}
INTEREST_ACTION = dict(zip(INTERESTS, ("explore_camera", "enable_travel", "enable_backup")))
STAGE_NAMES = ("faq", "compare", "journey", "adaptive")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact_keys(value, keys, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(keys), label + " has missing or unknown fields")


def unique_strings(value, allowed, label, nonempty=False):
    require(isinstance(value, list), label + " must be a list")
    require(all(isinstance(x, str) for x in value), label + " must contain strings")
    require(len(value) == len(set(value)), label + " must not contain duplicates")
    require(all(x in allowed for x in value), label + " contains an unknown value")
    require(not nonempty or bool(value), label + " must not be empty")


def positive_number(value, label):
    require(type(value) in (int, float), label + " must be numeric")
    require(math.isfinite(value) and value > 0, label + " must be positive and finite")


def validate_request(request):
    exact_keys(request, ("schema_version", "synthetic_fixture", "question",
                         "product_ids", "preferences", "profile"), "input")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "unsupported schema_version")
    require(request["synthetic_fixture"] is True, "synthetic_fixture must be true")
    require(isinstance(request["question"], str) and 0 < len(request["question"].strip()) <= 1000,
            "question must contain 1 to 1000 characters")
    unique_strings(request["product_ids"], PRODUCTS, "product_ids", True)
    prefs = request["preferences"]
    exact_keys(prefs, ("budget_usd", "min_storage_gb", "priorities"), "preferences")
    positive_number(prefs["budget_usd"], "budget_usd")
    positive_number(prefs["min_storage_gb"], "min_storage_gb")
    unique_strings(prefs["priorities"], PRIORITIES, "priorities", True)
    profile = request["profile"]
    exact_keys(profile, ("experience", "interests", "completed_actions"), "profile")
    require(isinstance(profile["experience"], str)
            and profile["experience"] in ("novice", "intermediate", "expert"),
            "experience must be novice, intermediate, or expert")
    unique_strings(profile["interests"], INTERESTS, "interests")
    unique_strings(profile["completed_actions"], ACTIONS, "completed_actions")
    completed = set(profile["completed_actions"])
    for action in completed:
        require(set(ACTIONS[action]) <= completed, "completed_actions violates prerequisites")
    return request


def normalize_quantity(text, units):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", text)
    require(match is not None and match[2] in units, "invalid fixture quantity")
    return float(match[1]) * units[match[2]]


def normalized_product(product_id):
    product = PRODUCTS[product_id]
    return {
        "product_id": product_id, "name": product["name"],
        "price_usd": product["price_usd"],
        # Fixture uses decimal storage units: 1 TB = 1000 GB.
        "storage_gb": normalize_quantity(product["storage"], {"GB": 1, "TB": 1000}),
        "weight_g": normalize_quantity(product["weight"], {"g": 1, "kg": 1000}),
        "battery_mah": normalize_quantity(product["battery"], {"mAh": 1, "Ah": 1000}),
    }


def faq_stage(request, previous):
    tokens = set(re.findall(r"[a-z]+", request["question"].lower()))
    evidence = [{"id": fact["id"], "text": fact["text"]}
                for fact in KB if tokens.intersection(fact["terms"])]
    return {
        "status": "answered" if evidence else "abstained",
        "answer": " ".join(item["text"] for item in evidence) if evidence
                  else "I cannot answer this question from the synthetic knowledge base.",
        "citations": [item["id"] for item in evidence],
        "evidence": evidence,
        "product_ids": list(request["product_ids"]),
    }


def preference_key(row, preferences):
    values = {"price": row["price_usd"], "storage": -row["storage_gb"],
              "weight": row["weight_g"], "battery": -row["battery_mah"]}
    return tuple(values[p] for p in preferences["priorities"]) + (row["product_id"],)


def compare_stage(request, previous):
    prefs = request["preferences"]
    rows = []
    for product_id in previous["product_ids"]:
        row = normalized_product(product_id)
        reasons = []
        if row["price_usd"] > prefs["budget_usd"]:
            reasons.append("over_budget")
        if row["storage_gb"] < prefs["min_storage_gb"]:
            reasons.append("insufficient_storage")
        row.update(eligible=not reasons, exclusions=reasons)
        rows.append(row)
    eligible = sorted((row for row in rows if row["eligible"]),
                      key=lambda row: preference_key(row, prefs))
    return {
        "status": "matched" if eligible else "no_match",
        "faq_status": previous["status"],
        "citations": list(previous["citations"]),
        "columns": ["price_usd", "storage_gb", "weight_g", "battery_mah"],
        "rows": rows,
        "ranking": [row["product_id"] for row in eligible],
        "ranking_explanation": "Eligible products ranked lexicographically by "
                               + ", ".join(prefs["priorities"])
                               + "; product_id breaks ties.",
    }


def action_order(profile):
    order = ["choose_product"]
    if profile["experience"] == "novice":
        order.append("learn_basics")
    order += ["create_account", "configure_device", "set_preferences"]
    order += [INTEREST_ACTION[interest] for interest in profile["interests"]]
    return order


def action_record(action, profile, product_id):
    reason = f"For {profile['experience']} experience with {product_id}; "
    if action == "learn_basics":
        reason += "a guided introduction is recommended before account setup."
    elif action in INTEREST_ACTION.values():
        interest = next(k for k, v in INTEREST_ACTION.items() if v == action)
        reason += f"matches the stated {interest} preference."
    else:
        reason += "supports device setup and stated preferences."
    prerequisites = list(ACTIONS[action])
    reason += " Prerequisites: " + (", ".join(prerequisites) or "none") + "."
    return {"action_id": action, "product_id": product_id,
            "prerequisites": prerequisites, "explanation": reason}


def journey_stage(request, previous):
    profile = request["profile"]
    completed = set(profile["completed_actions"])
    product_id = previous["ranking"][0] if previous["ranking"] else None
    result = {
        "status": "blocked" if product_id is None else "ready",
        "product_id": product_id, "citations": list(previous["citations"]),
        "next_actions": [], "steps": [],
        "completed_after_journey": sorted(completed),
        "explanation": "",
    }
    if product_id is None:
        result["explanation"] = "No eligible product; relax budget or storage constraints."
        return result
    order = action_order(profile)
    result["next_actions"] = [
        action_record(a, profile, product_id) for a in order
        if a not in completed and set(ACTIONS[a]) <= completed
    ]
    for _ in range(2):
        available = [a for a in order if a not in completed and set(ACTIONS[a]) <= completed]
        if not available:
            break
        action = available[0]
        result["steps"].append(action_record(action, profile, product_id))
        completed.add(action)
    result["completed_after_journey"] = sorted(completed)
    result["status"] = "ready" if len(result["steps"]) == 2 else "exhausted"
    result["explanation"] = (
        "Two prerequisite-valid steps; completion below is simulated, not persisted."
        if len(result["steps"]) == 2 else
        "Fewer than two relevant actions remain; no duplicate or fabricated steps."
    )
    return result


def adaptive_stage(request, previous):
    profile = request["profile"]
    completed = set(previous["completed_after_journey"])
    product_id = previous["product_id"]
    steps = []
    if product_id is not None:
        for action in action_order(profile):
            if action not in completed:
                require(set(ACTIONS[action]) <= completed, "onboarding prerequisites unavailable")
                steps.append(action_record(action, profile, product_id))
                completed.add(action)
    return {
        "status": "blocked" if product_id is None else ("ready" if steps else "complete"),
        "product_id": product_id,
        "citations": list(previous["citations"]),
        "experience": profile["experience"], "interests": list(profile["interests"]),
        "starting_completed_actions": list(previous["completed_after_journey"]),
        "steps": steps,
        "explanation": "Onboarding assumes the proposed journey has been completed; "
                       "novices receive basics, experts skip optional basics, and "
                       "interest-specific features follow setup prerequisites."
                       if product_id is not None else "Blocked because comparison found no eligible product.",
    }


STAGES = {"faq": faq_stage, "compare": compare_stage,
          "journey": journey_stage, "adaptive": adaptive_stage}
OUTPUT_KEYS = {
    "faq": ("status", "answer", "citations", "evidence", "product_ids"),
    "compare": ("status", "faq_status", "citations", "columns", "rows", "ranking",
                "ranking_explanation"),
    "journey": ("status", "product_id", "citations", "next_actions", "steps",
                "completed_after_journey", "explanation"),
    "adaptive": ("status", "product_id", "citations", "experience", "interests",
                 "starting_completed_actions", "steps", "explanation"),
}


def validate_sequence(steps, initial_completed, product_id):
    require(isinstance(steps, list), "steps must be a list")
    completed = set(initial_completed)
    for step in steps:
        exact_keys(step, ("action_id", "product_id", "prerequisites", "explanation"), "step")
        action = step["action_id"]
        require(isinstance(action, str) and action in ACTIONS, "unknown action")
        require(action not in completed, "action already completed")
        require(step["product_id"] == product_id, "action product mismatch")
        require(step["prerequisites"] == list(ACTIONS[action]), "incorrect prerequisites")
        require(set(ACTIONS[action]) <= completed, "unsatisfied prerequisites")
        require(isinstance(step["explanation"], str) and bool(step["explanation"]),
                "missing action explanation")
        completed.add(action)
    return completed


def validate_stage(name, output, request, previous):
    """One validation boundary shared by every stage and cross-stage handoff."""
    require(name in STAGES, "unknown stage")
    exact_keys(output, OUTPUT_KEYS[name], name)
    # Canonical replay is intentionally bounded to tiny synthetic fixtures. It
    # rejects altered facts, rankings, and handoffs, not just malformed shapes.
    expected = STAGES[name](request, previous)
    require(json.dumps(output, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False),
            name + " output is inconsistent with grounded deterministic rules")
    if name == "journey":
        completed = validate_sequence(output["steps"], request["profile"]["completed_actions"],
                                      output["product_id"])
        require(sorted(completed) == output["completed_after_journey"],
                "journey completion mismatch")
        for action in output["next_actions"]:
            validate_sequence([action], request["profile"]["completed_actions"], output["product_id"])
    elif name == "adaptive":
        validate_sequence(output["steps"], previous["completed_after_journey"], output["product_id"])
    return output


def run_pipeline(request):
    request = copy.deepcopy(validate_request(request))
    stages = {}
    previous = None
    for name in STAGE_NAMES:
        output = STAGES[name](request, previous)
        stages[name] = validate_stage(name, output, request, previous)
        previous = copy.deepcopy(stages[name])
    return {"schema_version": SCHEMA_VERSION, "status": "ok", "synthetic_fixture": True,
            "input": request, "stage_order": list(STAGE_NAMES), "stages": stages}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        request = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                             object_pairs_hook=reject_duplicates,
                             parse_constant=reject_constant)
        result = run_pipeline(request)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
