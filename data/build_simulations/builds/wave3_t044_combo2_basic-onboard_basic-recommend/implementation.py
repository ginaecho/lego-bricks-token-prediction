"""Synthetic, deterministic onboarding -> personalized discovery reference CLI."""
import json
import math
import sys


STEPS = ("discover_preferences", "explore_matches", "choose_product")
LEVELS = ("beginner", "intermediate", "advanced")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    return value.strip()


def strings(value, path):
    require(isinstance(value, list), path + " must be a list")
    result = [text(item, path) for item in value]
    require(len(result) == len(set(result)), path + " must not contain duplicates")
    return result


def money(value, path):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            path + " must be a finite nonnegative number")
    return value


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def validate_input(data):
    """Normalize the shared input schema without mutating caller-owned objects."""
    fields(data, ("schema_version", "synthetic", "customer", "catalog", "limit"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "reference fixtures must be labeled synthetic")
    require(type(data["limit"]) is int and 1 <= data["limit"] <= 20, "limit must be 1..20")
    customer = data["customer"]
    fields(customer, ("id", "name", "goal", "experience", "interests", "budget",
                      "completed_steps"), "customer")
    normalized = {key: text(customer[key], "customer." + key)
                  for key in ("id", "name", "goal", "experience")}
    require(normalized["experience"] in LEVELS, "unknown experience")
    normalized["interests"] = strings(customer["interests"], "customer.interests")
    normalized["budget"] = money(customer["budget"], "customer.budget")
    completed = strings(customer["completed_steps"], "customer.completed_steps")
    require(completed == list(STEPS[:len(completed)]) and len(completed) <= len(STEPS),
            "completed_steps must be an ordered prefix of onboarding steps")
    normalized["completed_steps"] = completed
    require(isinstance(data["catalog"], list), "catalog must be a list")
    catalog = []
    seen = set()
    for item in data["catalog"]:
        fields(item, ("id", "name", "tags", "goals", "levels", "price", "available"), "product")
        product = {key: text(item[key], "product." + key) for key in ("id", "name")}
        require(product["id"] not in seen, "duplicate product id")
        seen.add(product["id"])
        for key in ("tags", "goals", "levels"):
            product[key] = strings(item[key], "product." + key)
        require(all(level in LEVELS for level in product["levels"]), "unknown product level")
        product["price"] = money(item["price"], "product.price")
        require(type(item["available"]) is bool, "product.available must be boolean")
        product["available"] = item["available"]
        catalog.append(product)
    return dict(schema_version=1, synthetic=True, customer=normalized,
                catalog=catalog, limit=data["limit"])


def onboarding_details(customer):
    count = len(customer["completed_steps"])
    step = STEPS[count] if count < len(STEPS) else "complete"
    actions = {
        "discover_preferences": "Confirm your interests and budget",
        "explore_matches": "Explore products matched to your preferences",
        "choose_product": "Compare your matches and choose a product",
        "complete": "Continue discovering products",
    }
    return {
        "customer_id": customer["id"],
        "completed": step == "complete",
        "next_step": {"id": step, "instruction":
                      f'{customer["name"]}, {actions[step].lower()} to pursue {customer["goal"]}.'},
        "discovery_context": {
            "customer_id": customer["id"], "goal": customer["goal"],
            "interests": list(customer["interests"]), "experience": customer["experience"],
            "max_price": customer["budget"],
        },
    }


def ranked_discovery(state):
    # Consume the validated onboarding handoff, not raw customer preferences.
    context = state["onboarding"]["discovery_context"]
    matches = []
    excluded = {"unavailable": 0, "over_budget": 0, "no_relevance": 0}
    for product in state["catalog"]:
        if not product["available"]:
            excluded["unavailable"] += 1
            continue
        if product["price"] > context["max_price"]:
            excluded["over_budget"] += 1
            continue
        interests = sorted(set(context["interests"]) & set(product["tags"]))
        goal_match = context["goal"] in product["goals"]
        if not interests and not goal_match:
            excluded["no_relevance"] += 1
            continue
        level_match = context["experience"] in product["levels"]
        reasons = (["Matches your goal"] if goal_match else [])
        reasons += ["Matches interest: " + interest for interest in interests]
        if level_match:
            reasons.append("Matches your experience")
        matches.append({
            "product_id": product["id"], "name": product["name"], "price": product["price"],
            "score": 4 * int(goal_match) + 2 * len(interests) + int(level_match),
            "reasons": reasons,
        })
    matches.sort(key=lambda item: (-item["score"], item["price"], item["product_id"]))
    return {
        "customer_id": context["customer_id"],
        "onboarding_step": state["onboarding"]["next_step"]["id"],
        "items": matches[:state["limit"]],
        "eligible_count": len(matches), "excluded": excluded,
        "message": "Personalized matches ready" if matches else
                   "No suitable products; review your interests or budget without exceeding it.",
    }


def validate_state(state, stage):
    """One validation boundary for both stage outputs and their shared handoff."""
    require(stage in ("onboarding", "discovery"), "unknown pipeline stage")
    base_keys = ("schema_version", "synthetic", "customer", "catalog", "limit")
    output_keys = ("status", "onboarding") + (("discovery",) if stage == "discovery" else ())
    fields(state, base_keys + output_keys, "pipeline state")
    base = validate_input({key: state[key] for key in base_keys})
    require(state["status"] == "ok", "invalid pipeline status")
    require(state["onboarding"] == onboarding_details(base["customer"]),
            "onboarding output does not match validated customer")
    validated = dict(base, status="ok", onboarding=onboarding_details(base["customer"]))
    if stage == "discovery":
        expected = ranked_discovery(validated)
        require(state["discovery"] == expected, "invalid discovery output")
        validated["discovery"] = expected
    return validated


def onboard(data):
    state = validate_input(data)
    state.update(status="ok", onboarding=onboarding_details(state["customer"]))
    return validate_state(state, "onboarding")


def recommend(onboarding_state):
    state = validate_state(onboarding_state, "onboarding")
    state["discovery"] = ranked_discovery(state)
    return validate_state(state, "discovery")


def run_pipeline(data):
    return recommend(onboard(data))


def reject_constant(value):
    raise ValidationError("nonfinite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
