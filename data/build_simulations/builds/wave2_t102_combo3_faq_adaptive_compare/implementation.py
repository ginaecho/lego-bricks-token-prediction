"""Synthetic, deterministic FAQ -> onboarding -> comparison reference CLI.

Run: python -B implementation.py example_input.json
The same versioned envelope carries the request and validated stage results.
No network, model, third-party dependency, or persistent write is used.
"""

import copy
import json
import math
import re
import sys
from pathlib import Path


VERSION = "1.0"
ATTRIBUTES = ("price", "weight", "battery", "rating")
EXPERIENCES = ("beginner", "intermediate", "expert")
MODES = ("guided", "self_service")
STATES = ("ready", "faq_complete", "adaptive_complete", "ok")
STAGE_NAMES = ("faq", "adaptive", "compare")
ALIASES = {
    "price": "price", "cost": "price",
    "weight": "weight", "mass": "weight",
    "battery": "battery", "battery_life": "battery",
    "rating": "rating", "review_score": "rating",
}
UNITS = {
    "price": {"usd": 1.0, "$": 1.0},
    "weight": {"g": 1.0, "kg": 1000.0, "lb": 453.59237, "oz": 28.349523125},
    "battery": {"h": 1.0, "hours": 1.0, "min": 1.0 / 60},
    "rating": {"/5": 1.0, "/10": 0.5, "%": 0.05},
}
CANONICAL_UNITS = {"price": "USD", "weight": "g", "battery": "h", "rating": "/5"}
STOP_WORDS = set(
    "a an the is are was were do does did how what which why can i we you my "
    "me to for of and or in on with please tell about should".split()
)


class ValidationError(ValueError):
    """An invalid input or stage handoff."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, path):
    require(type(value) is dict, f"{path} must be an object")
    require(set(value) == set(names), f"{path} must have exactly: {', '.join(names)}")


def text(value, path, limit=10000):
    require(type(value) is str and bool(value.strip()) and len(value) <= limit,
            f"{path} must be nonempty text of at most {limit} characters")


def number(value, path, low=0, high=1e12):
    require(type(value) in (int, float), f"{path} must be a number, not a boolean")
    require(low <= value <= high and math.isfinite(value),
            f"{path} must be finite and between {low} and {high}")


def strings(value, path, allowed=None, limit=100):
    require(type(value) is list and len(value) <= limit, f"{path} must be a bounded list")
    for item in value:
        text(item, path, 100)
        if allowed is not None:
            require(item in allowed, f"{path} contains an unsupported value: {item}")
    require(len(value) == len(set(value)), f"{path} contains duplicates")


def records(value, path):
    require(type(value) is list and len(value) <= 100, f"{path} must contain at most 100 records")


def canonical_attribute(name):
    text(name, "attribute name", 100)
    key = name.strip().lower().replace(" ", "_")
    require(key in ALIASES, f"unsupported attribute: {name}")
    return ALIASES[key]


def normalize_attributes(raw):
    require(type(raw) is dict, "product attributes must be an object")
    require(len(raw) <= len(ATTRIBUTES), "too many product attributes")
    result = {}
    for name, measurement in raw.items():
        attribute = canonical_attribute(name)
        require(attribute not in result, f"duplicate normalized attribute: {attribute}")
        fields(measurement, ("value", "unit"), f"attributes.{name}")
        number(measurement["value"], f"attributes.{name}.value")
        text(measurement["unit"], f"attributes.{name}.unit", 20)
        unit = measurement["unit"].strip().lower()
        require(unit in UNITS[attribute], f"unsupported unit for {attribute}: {unit}")
        value = measurement["value"] * UNITS[attribute][unit]
        require(math.isfinite(value), "normalized attribute must be finite")
        if attribute == "rating":
            require(value <= 5, "normalized rating must be between 0 and 5")
        result[attribute] = value
    return result


def validate_request(request):
    fields(request, ("question", "profile", "knowledge_base", "onboarding_steps", "products"), "request")
    text(request["question"], "question", 1000)
    profile = request["profile"]
    fields(profile, ("experience", "onboarding_mode", "completed_steps", "priorities", "max_price_usd"),
           "profile")
    require(profile["experience"] in EXPERIENCES, "unsupported experience")
    require(profile["onboarding_mode"] in MODES, "unsupported onboarding mode")
    strings(profile["completed_steps"], "completed_steps")
    priorities = profile["priorities"]
    require(type(priorities) is dict, "priorities must be an object")
    require(all(key in ATTRIBUTES for key in priorities), "unsupported priority attribute")
    for name, weight in priorities.items():
        number(weight, f"priorities.{name}", low=1e-6, high=1e6)
    if profile["max_price_usd"] is not None:
        number(profile["max_price_usd"], "max_price_usd")

    docs = request["knowledge_base"]
    records(docs, "knowledge_base")
    doc_ids = set()
    for doc in docs:
        fields(doc, ("id", "question", "answer", "keywords", "topics", "attribute_hints"), "knowledge document")
        for key in ("id", "question", "answer"):
            text(doc[key], f"document.{key}", 100 if key == "id" else 10000)
        require(doc["id"] not in doc_ids, "duplicate knowledge document ID")
        doc_ids.add(doc["id"])
        strings(doc["keywords"], "document.keywords")
        strings(doc["topics"], "document.topics")
        strings(doc["attribute_hints"], "document.attribute_hints", ATTRIBUTES)

    steps = request["onboarding_steps"]
    records(steps, "onboarding_steps")
    step_map = {}
    for step in steps:
        fields(step, ("id", "title", "explanation", "prerequisites", "topics", "audiences"), "onboarding step")
        for key in ("id", "title", "explanation"):
            text(step[key], f"step.{key}", 100 if key == "id" else 10000)
        require(step["id"] not in step_map, "duplicate onboarding step ID")
        strings(step["prerequisites"], "prerequisites")
        strings(step["topics"], "step.topics")
        strings(step["audiences"], "audiences", EXPERIENCES)
        require(bool(step["audiences"]), "audiences must not be empty")
        step_map[step["id"]] = step
    for step in steps:
        require(all(key in step_map for key in step["prerequisites"]), "unknown prerequisite")
    require(all(key in step_map for key in profile["completed_steps"]), "unknown completed step")
    visiting, visited = set(), set()

    def visit(key):
        require(key not in visiting, "onboarding prerequisites contain a cycle")
        if key in visited:
            return
        visiting.add(key)
        for dependency in step_map[key]["prerequisites"]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in step_map:
        visit(key)

    products = request["products"]
    records(products, "products")
    product_ids = set()
    for product in products:
        fields(product, ("id", "name", "attributes"), "product")
        text(product["id"], "product.id", 100)
        text(product["name"], "product.name", 200)
        require(product["id"] not in product_ids, "duplicate product ID")
        product_ids.add(product["id"])
        normalize_attributes(product["attributes"])


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOP_WORDS


def _faq(request):
    query = tokens(request["question"])
    candidates = []
    for doc in request["knowledge_base"]:
        indexed = tokens(doc["question"] + " " + " ".join(doc["keywords"]))
        count = len(query & indexed)
        confidence = count / len(query) if query else 0.0
        if query and count >= min(2, len(query)) and confidence >= 0.34:
            candidates.append((confidence, doc))
    candidates.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    reason = "no_match"
    if candidates:
        best_score = candidates[0][0]
        best = [doc for score, doc in candidates if score == best_score]
        # Conflicting tied evidence is not resolved by an arbitrary document ID.
        signatures = {(doc["answer"], tuple(sorted(doc["topics"])),
                       tuple(sorted(doc["attribute_hints"]))) for doc in best}
        if len(signatures) == 1:
            doc = best[0]
            return {
                "status": "answered", "answer": doc["answer"], "sources": [doc["id"]],
                "confidence": round(best_score, 6), "reason": "matched",
                "topics": sorted(doc["topics"]), "attribute_hints": sorted(doc["attribute_hints"]),
            }
        reason = "ambiguous_match"
    return {"status": "abstained", "answer": None, "sources": [], "confidence": 0.0,
            "reason": reason, "topics": [], "attribute_hints": []}


def _adaptive(request, faq):
    profile = request["profile"]
    completed = set(profile["completed_steps"])
    step_map = {step["id"]: step for step in request["onboarding_steps"]}
    topics = set(faq["topics"])
    roots = sorted(
        key for key, step in step_map.items()
        if profile["experience"] in step["audiences"]
        and (not step["topics"] or bool(topics & set(step["topics"])))
        and key not in completed
    )
    plan, included = [], set()
    mode = profile["onboarding_mode"]

    def include(key):
        if key in completed or key in included:
            return
        step = step_map[key]
        for dependency in sorted(step["prerequisites"]):
            include(dependency)
        included.add(key)
        if key in roots:
            reason = f"Matches {profile['experience']} experience and " + (
                "answered FAQ topics." if step["topics"] else "general onboarding."
            )
        else:
            reason = "Required prerequisite; included even outside the preferred audience or topic."
        delivery = ("Follow the guided instructions and confirm completion."
                    if mode == "guided" else "Complete independently using the checklist.")
        plan.append({
            "id": key, "title": step["title"], "prerequisites": sorted(step["prerequisites"]),
            "explanation": step["explanation"], "selection_reason": reason,
            "delivery": delivery,
        })

    for key in roots:
        include(key)
    weights, reasons = {}, {}
    for attribute in ATTRIBUTES:
        if attribute in profile["priorities"]:
            weights[attribute] = profile["priorities"][attribute]
            reasons[attribute] = "Explicit user priority."
        elif attribute in faq["attribute_hints"]:
            weights[attribute] = 2.0
            reasons[attribute] = "Grounded FAQ attribute hint."
        else:
            weights[attribute] = 1.0
            reasons[attribute] = "Default balanced priority."
    total = sum(weights.values())
    return {
        "based_on_faq": faq["status"], "topics": faq["topics"][:],
        "mode": mode, "experience": profile["experience"], "steps": plan,
        "completed_steps": sorted(completed),
        "weights": {key: weights[key] / total for key in ATTRIBUTES},
        "weight_reasons": reasons, "max_price_usd": profile["max_price_usd"],
        "explanation": (
            "Uses grounded FAQ topics; explicit preferences override FAQ hints."
            if faq["status"] == "answered" else
            "FAQ abstained: use general onboarding and user preferences only; no topic inferred."
        ),
    }


def _compare(request, adaptive):
    weights = adaptive["weights"]
    budget = adaptive["max_price_usd"]
    normalized, eligible, excluded = [], [], []
    for product in sorted(request["products"], key=lambda item: item["id"]):
        item = {"id": product["id"], "name": product["name"],
                "attributes": normalize_attributes(product["attributes"])}
        normalized.append(item)
        price = item["attributes"].get("price")
        reason = None
        if budget is not None:
            if price is None:
                reason = "price_unknown_under_budget_constraint"
            elif price > budget:
                reason = "over_budget"
        if reason:
            excluded.append({"product_id": item["id"], "reason": reason})
        else:
            eligible.append(item)
    table, contributions = [], {item["id"]: {} for item in eligible}
    for attribute in ATTRIBUTES:
        values = {item["id"]: item["attributes"].get(attribute) for item in eligible}
        known = [value for value in values.values() if value is not None]
        low, high = (min(known), max(known)) if known else (0, 0)
        direction = "lower" if attribute in ("price", "weight") else "higher"
        table.append({"attribute": attribute, "unit": CANONICAL_UNITS[attribute],
                      "preferred": direction, "weight": weights[attribute], "values": values})
        for product_id, value in values.items():
            if value is None:
                utility = 0.0
            elif low == high:
                utility = 1.0
            else:
                utility = (value - low) / (high - low)
                if direction == "lower":
                    utility = 1.0 - utility
            contributions[product_id][attribute] = utility * weights[attribute]
    ranking = []
    for item in eligible:
        parts = contributions[item["id"]]
        ranking.append({
            "product_id": item["id"], "score": round(sum(parts.values()), 6),
            "contributions": {key: round(value, 6) for key, value in parts.items()},
            "missing_attributes": [key for key in ATTRIBUTES if key not in item["attributes"]],
        })
    ranking.sort(key=lambda item: (-item["score"], item["product_id"]))
    for index, item in enumerate(ranking, 1):
        item["rank"] = index
    return {
        "weights": dict(weights), "max_price_usd": budget,
        "normalized_products": normalized, "side_by_side": table,
        "ranking": ranking, "excluded": excluded,
        "explanation": (
            "Eligible-catalog min-max utility, weighted by onboarding preferences. "
            "Missing values score zero; equal known values score one. "
            "Scores round to six decimals; ties break by product ID. "
            "Budget excludes unknown prices. Scores are relative, not quality guarantees."
        ),
    }


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=True)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValidationError("envelope must contain finite JSON-compatible values") from error


def validate(envelope, expected_state=None):
    """One contract for input and every handoff, including derived-data integrity.

    Small deterministic projections are recomputed to validate persisted stage
    results. Thus structurally valid but ungrounded/tampered data is rejected.
    """
    fields(envelope, ("schema_version", "fixture_label", "status", "request", "stages"), "envelope")
    require(envelope["schema_version"] == VERSION, "unsupported schema_version")
    text(envelope["fixture_label"], "fixture_label", 200)
    require(envelope["fixture_label"].startswith("SYNTHETIC"), "fixture_label must start with SYNTHETIC")
    require(envelope["status"] in STATES, "unsupported envelope status")
    if expected_state is not None:
        require(envelope["status"] == expected_state, f"expected {expected_state} envelope")
    validate_request(envelope["request"])
    depth = STATES.index(envelope["status"])
    stages = envelope["stages"]
    require(type(stages) is dict and set(stages) == set(STAGE_NAMES[:depth]),
            "stages do not match envelope status")
    expected = {}
    if depth >= 1:
        expected["faq"] = _faq(envelope["request"])
    if depth >= 2:
        expected["adaptive"] = _adaptive(envelope["request"], expected["faq"])
    if depth >= 3:
        expected["compare"] = _compare(envelope["request"], expected["adaptive"])
    require(canonical(stages) == canonical(expected), "stage output violates its grounded deterministic contract")
    return envelope


def _advance(envelope, index, builder):
    validate(envelope, STATES[index])
    output = copy.deepcopy(envelope)
    output["stages"][STAGE_NAMES[index]] = builder(output)
    output["status"] = STATES[index + 1]
    return validate(output, STATES[index + 1])


def answer_faq(envelope):
    return _advance(envelope, 0, lambda value: _faq(value["request"]))


def adapt_onboarding(envelope):
    return _advance(envelope, 1, lambda value: _adaptive(value["request"], value["stages"]["faq"]))


def compare_products(envelope):
    return _advance(envelope, 2, lambda value: _compare(value["request"], value["stages"]["adaptive"]))


def run_pipeline(envelope):
    return compare_products(adapt_onboarding(answer_faq(envelope)))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonfinite JSON number: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        path = Path(args[0])
        require(path.stat().st_size <= 2_000_000, "input file exceeds 2 MB")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = run_pipeline(data)
        print(json.dumps(output, allow_nan=False, ensure_ascii=True))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, RecursionError, OverflowError) as error:
        print(json.dumps({"schema_version": VERSION, "status": "error",
                          "error": {"type": "validation_or_file_error", "message": str(error)}},
                         ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
