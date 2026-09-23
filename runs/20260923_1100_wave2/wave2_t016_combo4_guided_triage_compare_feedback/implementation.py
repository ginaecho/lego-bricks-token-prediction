"""Synthetic, deterministic onboarding -> triage -> comparison -> feedback CLI.

Run: python -B implementation.py example_input.json
Only Python's standard library is used. No provider or network integration exists.
"""

import copy
import json
import math
import re
import sys
import unicodedata
from decimal import Decimal
from pathlib import Path


STAGES = ("input", "guided", "triage", "compare", "feedback")
PRIORITIES = ("low", "normal", "high", "urgent")
ATTRIBUTES = ("price_usd", "memory_gb", "weight_kg", "brand")
UNITS = {
    "price_usd": {"usd": 1},
    "memory_gb": {"gb": 1, "tb": 1024, "mb": 1 / 1024},
    "weight_kg": {"kg": 1, "g": 0.001, "lb": 0.45359237},
}
ALIASES = {
    "price": "price_usd", "price_usd": "price_usd",
    "memory": "memory_gb", "ram": "memory_gb", "memory_gb": "memory_gb",
    "weight": "weight_kg", "weight_kg": "weight_kg", "brand": "brand",
}


class ValidationError(ValueError):
    pass


def require(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def obj(value, path, required, optional=()):
    require(isinstance(value, dict), path, "must be an object")
    require(set(required) <= set(value), path, f"required keys: {', '.join(required)}")
    require(set(value) <= set(required) | set(optional), path, "unknown keys")


def text(value, path, max_length=2000):
    require(isinstance(value, str) and bool(value.strip()), path, "must be nonempty text")
    require(len(value) <= max_length, path, f"exceeds {max_length} characters")


def number(value, path, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    require(
        finite
        and (value > 0 if positive else value >= 0),
        path, "must be a finite " + ("positive" if positive else "nonnegative") + " number",
    )


def array(value, path, nonempty=False):
    require(isinstance(value, list), path, "must be an array")
    require(not nonempty or bool(value), path, "must not be empty")


def strings(value, path, nonempty=False):
    array(value, path, nonempty)
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), path, "duplicates are not allowed")


def canonical(value):
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def matches(content, keyword):
    return f" {canonical(keyword)} " in f" {canonical(content)} "


def keywords(value, path):
    strings(value, path, nonempty=True)
    require(all(canonical(item) for item in value), path, "keywords need letters or digits")
    require(len({canonical(item) for item in value}) == len(value), path,
            "normalized keywords must be unique")


def preferences(value, path):
    obj(value, path, (), ("max_price_usd", "min_memory_gb", "max_weight_kg",
                        "preferred_brand", "weights"))
    for key in ("max_price_usd", "min_memory_gb", "max_weight_kg"):
        if key in value:
            number(value[key], f"{path}.{key}")
    if "preferred_brand" in value:
        text(value["preferred_brand"], f"{path}.preferred_brand")
        require(bool(canonical(value["preferred_brand"])), path, "brand must contain text")
    if "weights" in value:
        obj(value["weights"], f"{path}.weights", ATTRIBUTES)
        for key, weight in value["weights"].items():
            number(weight, f"{path}.weights.{key}")
        require(any(value["weights"].values()), path, "at least one weight must be positive")


def normalize_product(product):
    attributes = {}
    for alias, raw in product["attributes"].items():
        require(alias in ALIASES, f"product.{product['id']}", f"unknown attribute {alias}")
        name = ALIASES[alias]
        require(name not in attributes, f"product.{product['id']}", f"duplicate alias for {name}")
        if name == "brand":
            text(raw, "product.brand")
            attributes[name] = canonical(raw)
            require(bool(attributes[name]), "product.brand", "must contain letters or digits")
        else:
            obj(raw, f"product.{name}", ("value", "unit"))
            number(raw["value"], f"product.{name}.value", positive=True)
            text(raw["unit"], f"product.{name}.unit")
            unit = raw["unit"].strip().casefold()
            require(unit in UNITS[name], f"product.{name}.unit", "unsupported unit")
            # Decimal conversion avoids excluding 1400 g at an exact 1.4 kg limit.
            converted = float(Decimal(str(raw["value"])) * Decimal(str(UNITS[name][unit])))
            number(converted, f"product.{name}.normalized", positive=True)
            attributes[name] = converted
    require(set(attributes) == set(ATTRIBUTES), "product.attributes",
            "price, memory, weight and brand are all required")
    return {"id": product["id"], "name": product["name"], "attributes": attributes}


def unique_ids(items, path):
    ids = []
    for item in items:
        require(isinstance(item, dict) and "id" in item, path, "items require id")
        text(item["id"], f"{path}.id", 100)
        ids.append(item["id"])
    require(len(ids) == len(set(ids)), path, "duplicate ids")
    return set(ids)


def validate_request(data):
    obj(data, "input", ("schema_version", "synthetic", "guided", "triage",
                        "products", "feedback", "feedback_themes"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version", "must be integer 1")
    require(data["synthetic"] is True, "synthetic", "reference fixtures must be labeled true")
    guided = data["guided"]
    obj(guided, "guided", ("customer_id", "steps", "completed_steps", "preferences"))
    text(guided["customer_id"], "guided.customer_id", 100)
    preferences(guided["preferences"], "guided.preferences")
    array(guided["steps"], "guided.steps", nonempty=True)
    step_ids = unique_ids(guided["steps"], "guided.steps")
    for step in guided["steps"]:
        obj(step, "guided.step", ("id", "requires", "required"))
        strings(step["requires"], "guided.step.requires")
        require(type(step["required"]) is bool, "guided.step.required", "must be boolean")
        require(set(step["requires"]) <= step_ids, "guided.step.requires", "unknown prerequisite")
        require(step["id"] not in step["requires"], "guided.step.requires", "self dependency")
    remaining = {step["id"]: set(step["requires"]) for step in guided["steps"]}
    visited = set()
    while remaining:
        ready = {key for key, dependencies in remaining.items() if dependencies <= visited}
        require(bool(ready), "guided.steps", "prerequisite cycle")
        visited.update(ready)
        remaining = {key: deps for key, deps in remaining.items() if key not in ready}
    strings(guided["completed_steps"], "guided.completed_steps")
    require(set(guided["completed_steps"]) <= step_ids, "guided.completed_steps", "unknown step")
    finished = set()
    by_id = {step["id"]: step for step in guided["steps"]}
    for step_id in guided["completed_steps"]:
        require(set(by_id[step_id]["requires"]) <= finished, "guided.completed_steps",
                f"prerequisites for {step_id} must be completed first")
        finished.add(step_id)
    triage = data["triage"]
    obj(triage, "triage", ("rules", "default", "tickets"))
    array(triage["rules"], "triage.rules")
    categories = set()
    for route in triage["rules"] + [triage["default"]]:
        is_default = route is triage["default"]
        obj(route, "triage.route", ("category", "owner", "priority", "preferences")
            + (() if is_default else ("keywords",)))
        text(route["category"], "triage.category", 100)
        text(route["owner"], "triage.owner", 100)
        require(isinstance(route["priority"], str) and route["priority"] in PRIORITIES,
                "triage.priority", "unknown priority")
        require(route["category"] not in categories, "triage.category", "duplicate category")
        categories.add(route["category"])
        preferences(route["preferences"], "triage.preferences")
        if not is_default:
            keywords(route["keywords"], "triage.keywords")
    array(triage["tickets"], "triage.tickets")
    ticket_ids = unique_ids(triage["tickets"], "triage.tickets")
    for ticket in triage["tickets"]:
        obj(ticket, "triage.ticket", ("id", "customer_id", "text"))
        text(ticket["text"], "triage.ticket.text")
        require(ticket["customer_id"] == guided["customer_id"], "triage.ticket.customer_id",
                "must match onboarded customer")
    array(data["products"], "products", nonempty=True)
    product_ids = unique_ids(data["products"], "products")
    for product in data["products"]:
        obj(product, "product", ("id", "name", "attributes"))
        text(product["name"], "product.name")
        require(isinstance(product["attributes"], dict), "product.attributes", "must be an object")
        normalize_product(product)
    array(data["feedback"], "feedback")
    unique_ids(data["feedback"], "feedback")
    for feedback in data["feedback"]:
        obj(feedback, "feedback.item", ("id", "ticket_id", "product_id", "text"))
        text(feedback["text"], "feedback.text")
        require(bool(canonical(feedback["text"])), "feedback.text", "must contain letters or digits")
        require(isinstance(feedback["ticket_id"], str) and feedback["ticket_id"] in ticket_ids,
                "feedback.ticket_id", "unknown ticket")
        require(isinstance(feedback["product_id"], str) and feedback["product_id"] in product_ids,
                "feedback.product_id", "unknown product")
    require(isinstance(data["feedback_themes"], dict), "feedback_themes", "must be an object")
    for theme, terms in data["feedback_themes"].items():
        text(theme, "feedback.theme")
        keywords(terms, f"feedback_themes.{theme}")
    return data


def guided_result(request):
    guided = request["guided"]
    completed = set(guided["completed_steps"])
    rows = [
        {
            "id": step["id"], "required": step["required"],
            "status": ("completed" if step["id"] in completed else
                       "ready" if set(step["requires"]) <= completed else "blocked"),
            "missing_prerequisites": sorted(set(step["requires"]) - completed),
        }
        for step in guided["steps"]
    ]
    required = {step["id"] for step in guided["steps"] if step["required"]}
    return {
        "customer_id": guided["customer_id"],
        "preferences": copy.deepcopy(guided["preferences"]),
        "steps": rows,
        "completed_count": len(completed),
        "total_count": len(rows),
        "progress": round(len(completed) / len(rows), 6),
        "required_remaining": sorted(required - completed),
        "ready_for_triage": required <= completed,
    }


def triage_result(request, guided):
    rows = []
    for ticket in request["triage"]["tickets"]:
        candidates = []
        for index, rule in enumerate(request["triage"]["rules"]):
            hits = [word for word in rule["keywords"] if matches(ticket["text"], word)]
            if hits:
                candidates.append((len(hits), PRIORITIES.index(rule["priority"]), -index, rule, hits))
        if candidates:
            _, _, _, route, hits = max(candidates, key=lambda entry: entry[:3])
        else:
            route, hits = request["triage"]["default"], []
        prefs = copy.deepcopy(guided["preferences"])
        prefs.update(copy.deepcopy(route["preferences"]))
        rows.append({
            "id": ticket["id"], "customer_id": guided["customer_id"],
            "category": route["category"], "priority": route["priority"],
            "owner": route["owner"], "matched_keywords": hits,
            "preferences": prefs,
        })
    return {"tickets": rows}


def compare_result(request, triage):
    catalog = sorted((normalize_product(p) for p in request["products"]), key=lambda p: p["id"])
    comparisons = []
    for ticket in triage["tickets"]:
        prefs = ticket["preferences"]
        rows = []
        for product in catalog:
            row = copy.deepcopy(product)
            attributes = row["attributes"]
            reasons = []
            for constraint, attr, minimum in (
                ("max_price_usd", "price_usd", False),
                ("min_memory_gb", "memory_gb", True),
                ("max_weight_kg", "weight_kg", False),
            ):
                if constraint in prefs:
                    accepted = (attributes[attr] >= prefs[constraint] if minimum
                                else attributes[attr] <= prefs[constraint])
                    if not accepted:
                        reasons.append(constraint)
            row.update({"eligible": not reasons, "rejection_reasons": reasons,
                        "score": None, "rank": None})
            rows.append(row)
        eligible = [row for row in rows if row["eligible"]]
        weights = prefs.get("weights", {"price_usd": 1, "memory_gb": 1, "weight_kg": 1, "brand": 1})
        # Scale weights first so finite, very large weights cannot overflow their sum.
        largest = max(weights.values())
        scaled = {key: weight / largest for key, weight in weights.items()}
        total = sum(scaled.values())
        for row in eligible:
            utilities = {}
            for attr in ATTRIBUTES:
                if attr == "brand":
                    utilities[attr] = (
                        1.0 if "preferred_brand" not in prefs
                        or row["attributes"][attr] == canonical(prefs["preferred_brand"]) else 0.0
                    )
                else:
                    values = [item["attributes"][attr] for item in eligible]
                    low, high = min(values), max(values)
                    utility = (1.0 if high == low
                               else (row["attributes"][attr] - low) / (high - low))
                    utilities[attr] = utility if attr == "memory_gb" or high == low else 1 - utility
            row["score"] = round(sum(utilities[attr] * scaled[attr] for attr in ATTRIBUTES) / total, 6)
        ranking = sorted(eligible, key=lambda row: (-row["score"], row["id"]))
        for index, row in enumerate(ranking, 1):
            row["rank"] = index
        comparisons.append({
            "ticket_id": ticket["id"], "category": ticket["category"],
            "priority": ticket["priority"], "owner": ticket["owner"],
            "preferences": copy.deepcopy(prefs), "columns": list(ATTRIBUTES), "products": rows,
            "ranked_product_ids": [row["id"] for row in ranking],
            "recommended_product_id": ranking[0]["id"] if ranking else None,
            "outcome": "ranked" if ranking else "no_eligible_products",
        })
    return {"comparisons": comparisons}


def feedback_result(request, comparison):
    comparisons = {row["ticket_id"]: row for row in comparison["comparisons"]}
    groups = {}
    for item in request["feedback"]:
        context = comparisons[item["ticket_id"]]
        products = {row["id"]: row for row in context["products"]}
        require(item["product_id"] in products, "feedback.product_id",
                "product was not compared for this ticket")
        key = (item["ticket_id"], item["product_id"], canonical(item["text"]))
        if key not in groups:
            product = products[item["product_id"]]
            groups[key] = {
                "id": item["id"], "ticket_id": item["ticket_id"], "product_id": item["product_id"],
                "owner": context["owner"], "category": context["category"],
                "comparison_rank": product["rank"], "comparison_score": product["score"],
                "was_recommended": context["recommended_product_id"] == item["product_id"],
                "source_ids": [], "themes": [], "evidence": [],
            }
        group = groups[key]
        group["source_ids"].append(item["id"])
        # Whole bounded source text is an exact excerpt, retaining accents and punctuation.
        group["evidence"].append({"source_id": item["id"], "excerpt": item["text"]})
    themes = {theme: {"theme": theme, "unique_feedback_count": 0, "evidence": []}
              for theme in request["feedback_themes"]}
    unthemed = 0
    for group in groups.values():
        content = group["evidence"][0]["excerpt"]
        for theme, terms in request["feedback_themes"].items():
            hits = [term for term in terms if matches(content, term)]
            if hits:
                group["themes"].append(theme)
                themes[theme]["unique_feedback_count"] += 1
                for evidence in group["evidence"]:
                    themes[theme]["evidence"].append({
                        **evidence, "feedback_id": group["id"],
                        "ticket_id": group["ticket_id"], "product_id": group["product_id"],
                        "matched_keywords": hits,
                    })
        if not group["themes"]:
            unthemed += 1
    return {
        "raw_count": len(request["feedback"]), "unique_count": len(groups),
        "duplicates_removed": len(request["feedback"]) - len(groups),
        "unthemed_count": unthemed, "groups": list(groups.values()),
        "themes": [themes[key] for key in sorted(themes)],
    }


def validate_state(state, expected_stage=None):
    """One envelope contract; recomputation verifies every deterministic handoff."""
    obj(state, "state", ("schema_version", "status", "stage", "request", "results"))
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "state.schema_version", "must be integer 1")
    require(state["status"] == "ok", "state.status", "must be ok")
    require(isinstance(state["stage"], str) and state["stage"] in STAGES,
            "state.stage", "unknown stage")
    if expected_stage is not None:
        require(state["stage"] == expected_stage, "state.stage", f"expected {expected_stage}")
    validate_request(state["request"])
    level = STAGES.index(state["stage"])
    obj(state["results"], "state.results", STAGES[1:level + 1])
    calculated = {}
    if level >= 1:
        calculated["guided"] = guided_result(state["request"])
    if level >= 2:
        require(calculated["guided"]["ready_for_triage"], "guided", "required onboarding incomplete")
        calculated["triage"] = triage_result(state["request"], calculated["guided"])
    if level >= 3:
        calculated["compare"] = compare_result(state["request"], calculated["triage"])
    if level >= 4:
        calculated["feedback"] = feedback_result(state["request"], calculated["compare"])
    # JSON equality distinguishes true from 1 and rejects malformed/nonfinite outputs.
    try:
        actual = json.dumps(state["results"], sort_keys=True, allow_nan=False)
        expected = json.dumps(calculated, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"state.results: not valid JSON: {exc}") from exc
    require(actual == expected, "state.results", "handoff differs from validated deterministic output")
    return state


def initialize(request):
    validate_request(request)
    return validate_state({"schema_version": 1, "status": "ok", "stage": "input",
                           "request": copy.deepcopy(request), "results": {}})


def advance(state, expected, next_stage, builder):
    validate_state(state, expected)
    output = copy.deepcopy(state)
    output["results"][next_stage] = builder(output["request"], output["results"])
    output["stage"] = next_stage
    return validate_state(output, next_stage)


def guided_setup(state):
    return advance(state, "input", "guided", lambda request, _: guided_result(request))


def triage_tickets(state):
    def build(request, results):
        require(results["guided"]["ready_for_triage"], "guided", "required onboarding incomplete")
        return triage_result(request, results["guided"])
    return advance(state, "guided", "triage", build)


def compare_products(state):
    return advance(state, "triage", "compare",
                   lambda request, results: compare_result(request, results["triage"]))


def analyze_feedback(state):
    return advance(state, "compare", "feedback",
                   lambda request, results: feedback_result(request, results["compare"]))


def run_pipeline(request):
    state = initialize(request)
    for stage in (guided_setup, triage_tickets, compare_products, analyze_feedback):
        state = stage(state)
    return state


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "json", f"duplicate key {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError(f"json: nonfinite constant {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "arguments", "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open(encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=no_duplicate_keys, parse_constant=invalid_constant)
        result = run_pipeline(data)
        encoded = json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
