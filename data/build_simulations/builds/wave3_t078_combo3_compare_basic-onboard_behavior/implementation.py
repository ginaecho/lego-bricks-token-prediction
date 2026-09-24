"""Synthetic, deterministic comparison -> onboarding -> discovery reference CLI."""

import copy
import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


ATTRS = ("price_usd", "weight_g", "battery_hours")
ALIASES = {
    "price_usd": ("price_usd", 1), "price": ("price_usd", 1),
    "weight_g": ("weight_g", 1), "weight_kg": ("weight_g", 1000),
    "battery_hours": ("battery_hours", 1), "battery_minutes": ("battery_hours", 1 / 60),
}
STEPS = ("profile", "learn", "compare", "select", "checkout")
GOALS = ("explore", "buy", "learn")


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing fields: " + ", ".join(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "Unexpected fields")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected nonempty text")
    return value


def number(value, minimum=0):
    require(type(value) in (int, float), "Expected a number, not a boolean")
    try:
        valid = math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    require(valid, "Expected a finite nonnegative number")
    return float(value)


def timestamp(value):
    text(value)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("Invalid ISO 8601 timestamp") from exc
    require(result.tzinfo is not None, "Timestamps must include a timezone")
    return result


def validate_customer(customer):
    fields(customer, ("name", "goal", "experience", "completed_steps"))
    text(customer["name"])
    require(customer["goal"] in GOALS, "Invalid customer goal")
    require(customer["experience"] in ("beginner", "experienced"), "Invalid experience")
    steps = customer["completed_steps"]
    require(isinstance(steps, list), "completed_steps must be a list")
    require(all(isinstance(step, str) and step in STEPS for step in steps), "Invalid completed step")
    require(len(set(steps)) == len(steps), "Duplicate completed step")


def validate_events(events, ids, now):
    require(isinstance(events, list), "Events must be a list")
    for event in events:
        fields(event, ("product_id", "kind", "at"))
        require(isinstance(event["product_id"], str) and event["product_id"] in ids, "Unknown event product")
        require(event["kind"] in ("browse", "purchase"), "Unknown event kind")
        require(timestamp(event["at"]) <= now, "Future events are not accepted")


def normalize(raw):
    fields(raw, ("schema_version", "synthetic", "as_of", "products", "preferences", "customer", "events"),
           ("half_life_days",))
    require(type(raw["schema_version"]) is int and raw["schema_version"] == 1, "Unsupported schema version")
    require(raw["synthetic"] is True, "Fixture data must be explicitly synthetic")
    now = timestamp(raw["as_of"])
    require(isinstance(raw["products"], list) and bool(raw["products"]), "Products must be a nonempty list")
    catalog = []
    ids = set()
    for product in raw["products"]:
        fields(product, ("id", "name", "attributes"))
        pid = text(product["id"])
        require(pid not in ids, "Duplicate product ID")
        ids.add(pid)
        text(product["name"])
        require(isinstance(product["attributes"], dict), "Attributes must be an object")
        attrs = {}
        for alias, value in product["attributes"].items():
            require(alias in ALIASES, "Unsupported attribute: " + alias)
            canonical, factor = ALIASES[alias]
            require(canonical not in attrs, "Ambiguous attribute aliases")
            attrs[canonical] = number(number(value) * factor)
        require(set(attrs) == set(ATTRS), "Each product needs price, weight and battery duration")
        catalog.append({"id": pid, "name": product["name"], "attributes": attrs})
    prefs = raw["preferences"]
    fields(prefs, ("weights",))
    fields(prefs["weights"], ATTRS)
    weights = {key: number(prefs["weights"][key]) for key in ATTRS}
    total = number(sum(weights.values()))
    require(total > 0, "At least one preference weight must be positive")
    weights = {key: value / total for key, value in weights.items()}
    customer = raw["customer"]
    validate_customer(customer)
    validate_events(raw["events"], ids, now)
    half_life = number(raw.get("half_life_days", 30))
    require(half_life > 0, "half_life_days must be positive")
    state = {
        "schema_version": 1, "status": "ok", "synthetic": True, "as_of": raw["as_of"],
        "catalog": catalog, "preferences": {"weights": weights},
        "customer": copy.deepcopy(customer), "events": copy.deepcopy(raw["events"]),
        "half_life_days": half_life,
    }
    validate(state, "normalized")
    return state


def validate(state, stage):
    """Shared contract validation at every pipeline boundary."""
    require(stage in ("normalized", "comparison", "onboarding", "behavior"), "Unknown stage")
    required = {"schema_version", "status", "synthetic", "as_of", "catalog",
                "preferences", "customer", "events", "half_life_days"}
    stages = ("comparison", "onboarding", "behavior")
    if stage != "normalized":
        required.update(stages[:stages.index(stage) + 1])
    fields(state, required)
    require(state["status"] == "ok" and state["synthetic"] is True, "Invalid envelope")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1, "Invalid version")
    now = timestamp(state["as_of"])
    require(isinstance(state["catalog"], list) and bool(state["catalog"]), "Invalid catalog")
    ids = []
    for product in state["catalog"]:
        fields(product, ("id", "name", "attributes"))
        ids.append(text(product["id"]))
        text(product["name"])
        fields(product["attributes"], ATTRS)
        for value in product["attributes"].values():
            number(value)
    require(len(set(ids)) == len(ids), "Duplicate catalog ID")
    validate_customer(state["customer"])
    validate_events(state["events"], ids, now)
    require(number(state["half_life_days"]) > 0, "half_life_days must be positive")
    fields(state["preferences"], ("weights",))
    fields(state["preferences"]["weights"], ATTRS)
    total = sum(number(value) for value in state["preferences"]["weights"].values())
    require(math.isclose(total, 1, rel_tol=1e-12), "Canonical weights must sum to one")
    if "comparison" in required:
        comparison = state["comparison"]
        fields(comparison, ("columns", "rows", "ranking"))
        require(comparison["columns"] == list(ATTRS), "Invalid comparison columns")
        require(comparison["rows"] == state["catalog"], "Comparison rows must match canonical catalog")
        validate_ranking(comparison["ranking"], ids, ("utilities",))
        for row in comparison["ranking"]:
            fields(row["utilities"], ATTRS)
            for value in row["utilities"].values():
                require(number(value) <= 1, "Invalid utility")
            expected = sum(row["utilities"][key] * state["preferences"]["weights"][key] for key in ATTRS)
            require(math.isclose(row["score"], expected, abs_tol=1e-12), "Preference score mismatch")
    if "onboarding" in required:
        onboarding = state["onboarding"]
        fields(onboarding, ("goal", "preferred_product_id", "comparison_order", "next_step", "complete"))
        require(onboarding["goal"] == state["customer"]["goal"], "Onboarding goal mismatch")
        order = [item["product_id"] for item in state["comparison"]["ranking"]]
        require(onboarding["comparison_order"] == order, "Lost comparison order")
        require(onboarding["preferred_product_id"] == order[0], "Lost comparison preference")
        fields(onboarding["next_step"], ("id", "message"))
        require(onboarding["next_step"]["id"] in STEPS + ("done",), "Invalid next step")
        text(onboarding["next_step"]["message"])
        require(type(onboarding["complete"]) is bool and
                onboarding["complete"] == (onboarding["next_step"]["id"] == "done"), "Invalid completion flag")
    if "behavior" in required:
        behavior = state["behavior"]
        fields(behavior, ("cold_start", "ranking", "onboarding_step", "seed_product_id"))
        require(type(behavior["cold_start"]) is bool, "Invalid cold-start flag")
        require(behavior["onboarding_step"] == state["onboarding"]["next_step"], "Lost next step")
        require(behavior["seed_product_id"] == state["onboarding"]["preferred_product_id"], "Lost onboarding seed")
        validate_ranking(behavior["ranking"], ids, ("preference_score", "activity_score", "onboarding_score"))
        for row in behavior["ranking"]:
            for key in ("preference_score", "activity_score", "onboarding_score"):
                require(number(row[key]) <= 1, "Invalid score component")
    return state


def validate_ranking(ranking, ids, extra):
    require(isinstance(ranking, list) and len(ranking) == len(ids), "Incomplete ranking")
    seen = []
    for index, row in enumerate(ranking):
        fields(row, ("product_id", "rank", "score") + extra)
        require(type(row["rank"]) is int and row["rank"] == index + 1, "Invalid rank position")
        require(number(row["score"]) <= 1, "Score must be in [0, 1]")
        seen.append(text(row["product_id"]))
    require(set(seen) == set(ids) and len(set(seen)) == len(ids), "Ranking IDs do not match catalog")
    require(ranking == sorted(ranking, key=lambda row: (-row["score"], row["product_id"])), "Ranking is unsorted")


def ranked(rows):
    rows.sort(key=lambda row: (-row["score"], row["product_id"]))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def compare(state):
    validate(state, "normalized")
    state = copy.deepcopy(state)
    ranges = {key: (min(p["attributes"][key] for p in state["catalog"]),
                    max(p["attributes"][key] for p in state["catalog"])) for key in ATTRS}
    rows = []
    for product in state["catalog"]:
        utilities = {}
        for key, (low, high) in ranges.items():
            value = product["attributes"][key]
            utilities[key] = 1.0 if low == high else (
                (value - low) / (high - low) if key == "battery_hours" else (high - value) / (high - low))
        score = sum(utilities[key] * state["preferences"]["weights"][key] for key in ATTRS)
        rows.append({"product_id": product["id"], "score": min(1.0, score), "utilities": utilities})
    state["comparison"] = {"columns": list(ATTRS), "rows": copy.deepcopy(state["catalog"]), "ranking": ranked(rows)}
    return validate(state, "comparison")


def onboard(state):
    validate(state, "comparison")
    state = copy.deepcopy(state)
    customer = state["customer"]
    order = [row["product_id"] for row in state["comparison"]["ranking"]]
    product = next(p for p in state["catalog"] if p["id"] == order[0])
    sequence = ["profile"]
    if customer["experience"] == "beginner" or customer["goal"] == "learn":
        sequence.append("learn")
    sequence += ["compare", "select"]
    if customer["goal"] == "buy":
        sequence.append("checkout")
    completed = set(customer["completed_steps"]) | {"compare"}
    step = next((step for step in sequence if step not in completed), "done")
    instructions = {
        "profile": "confirm your preferences", "learn": "review the getting-started guide",
        "select": "review and save this recommendation", "checkout": "review checkout details (no purchase is made)",
        "done": "continue exploring; your onboarding is complete",
    }
    message = f'{customer["name"]}, {instructions[step]} for {product["name"]} to support your {customer["goal"]} goal.'
    state["onboarding"] = {
        "goal": customer["goal"], "preferred_product_id": order[0], "comparison_order": order,
        "next_step": {"id": step, "message": message}, "complete": step == "done",
    }
    return validate(state, "onboarding")


def personalize(state):
    validate(state, "onboarding")
    state = copy.deepcopy(state)
    now = timestamp(state["as_of"])
    activity = {product["id"]: 0.0 for product in state["catalog"]}
    # Sorting makes floating-point accumulation independent of input event order.
    for event in sorted(state["events"], key=lambda e: (e["product_id"], e["at"], e["kind"])):
        days = (now - timestamp(event["at"])).total_seconds() / 86400
        activity[event["product_id"]] += (3 if event["kind"] == "purchase" else 1) * 2 ** (-days / state["half_life_days"])
    peak = max(activity.values())
    cold = peak == 0
    seed = state["onboarding"]["preferred_product_id"]
    rows = []
    for candidate in state["comparison"]["ranking"]:
        pid = candidate["product_id"]
        preference = candidate["score"]
        behavioral = activity[pid] / peak if peak else 0.0
        onboarding = float(pid == seed)
        score = (.9 * preference + .1 * onboarding) if cold else (
            .6 * preference + .3 * behavioral + .1 * onboarding)
        rows.append({"product_id": pid, "score": min(score, 1.0),
                     "preference_score": preference, "activity_score": behavioral,
                     "onboarding_score": onboarding})
    state["behavior"] = {
        "cold_start": cold, "ranking": ranked(rows),
        "onboarding_step": copy.deepcopy(state["onboarding"]["next_step"]), "seed_product_id": seed,
    }
    return validate(state, "behavior")


def run(raw):
    return personalize(onboard(compare(normalize(raw))))


def reject_constant(value):
    raise ValidationError("Nonfinite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as source:
            raw = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run(raw)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
