"""Deterministic synthetic retail pipeline; Python standard library only."""

import hashlib
import json
import math
import re
import sys


VERSION = "1.0"
STAGES = ("compare", "journey", "normal", "adaptive")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(required), label + " has missing or unknown fields")


def integer(value, minimum, label):
    require(type(value) is int and value >= minimum, label + " is invalid")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def normalized(product):
    attrs = product["attributes"]
    return {
        "weight_g": round(attrs["weight"]["value"] *
                          {"g": 1, "kg": 1000}[attrs["weight"]["unit"]], 6),
        "capacity_ml": round(attrs["capacity"]["value"] *
                             {"ml": 1, "l": 1000}[attrs["capacity"]["unit"]], 6),
        "waterproof": attrs["waterproof"],
    }


def plan(initial, steps):
    available = set(initial)
    result = []
    for action, needs, provides, explanation in steps:
        require(set(needs) <= available, "Unsatisfied prerequisite for " + action)
        result.append({"action": action, "requires": needs, "provides": provides,
                       "explanation": explanation, "status": "planned"})
        available.update(provides)
    return result


class Validation:
    """Shared input and stage contract validation; unknown fields fail closed."""

    @staticmethod
    def input(value):
        keys(value, ("schema_version", "synthetic", "catalog", "customer",
                     "basket", "clickstream", "research_query"), "input")
        require(value["schema_version"] == VERSION, "Unsupported schema_version")
        require(value["synthetic"] is True, "Only explicitly synthetic fixtures are supported")
        text(value["research_query"], "research_query")
        require(bool(tokens(value["research_query"])), "research_query needs searchable words")
        catalog = value["catalog"]
        keys(catalog, ("currency", "products"), "catalog")
        require(catalog["currency"] == "USD", "This reference supports USD only")
        require(isinstance(catalog["products"], list) and
                1 <= len(catalog["products"]) <= 100, "Expected 1-100 products")
        skus, sources = set(), set()
        for product in catalog["products"]:
            keys(product, ("sku", "name", "price_cents", "stock", "attributes", "passages"),
                 "product (reviews and endorsements are unsupported)")
            text(product["sku"], "sku")
            require(product["sku"] not in skus, "Duplicate SKU")
            skus.add(product["sku"])
            text(product["name"], "name")
            integer(product["price_cents"], 0, "price_cents")
            integer(product["stock"], 0, "stock")
            attrs = product["attributes"]
            keys(attrs, ("weight", "capacity", "waterproof"), "attributes")
            for field, units in (("weight", ("g", "kg")), ("capacity", ("ml", "l"))):
                measure = attrs[field]
                keys(measure, ("value", "unit"), field)
                require(type(measure["value"]) in (int, float) and
                        math.isfinite(measure["value"]) and
                        0 < measure["value"] <= 1000000, field + " value is invalid")
                require(measure["unit"] in units, field + " unit is unsupported")
            require(type(attrs["waterproof"]) is bool, "waterproof must be boolean")
            require(isinstance(product["passages"], list) and
                    1 <= len(product["passages"]) <= 20, "Expected 1-20 passages")
            for passage in product["passages"]:
                keys(passage, ("source_id", "kind", "text"), "passage")
                text(passage["source_id"], "source_id")
                require(passage["source_id"] not in sources, "Duplicate source_id")
                sources.add(passage["source_id"])
                require(passage["kind"] in ("specification", "care"),
                        "Only specification/care evidence is allowed, not reviews/endorsements")
                text(passage["text"], "passage text")
                require(not tokens(passage["text"]) &
                        {"review", "reviews", "endorsement", "endorsed", "testimonial",
                         "testimonials", "stars"},
                        "Reviews and endorsements are not accepted as evidence")
        customer = value["customer"]
        keys(customer, ("customer_id", "consent", "experience", "preferences"), "customer")
        text(customer["customer_id"], "customer_id")
        keys(customer["consent"], ("personalization",), "consent")
        require(customer["consent"]["personalization"] is True,
                "Explicit GDPR/CCPA-style personalization consent is required")
        require(customer["experience"] in ("novice", "experienced"), "Invalid experience")
        prefs = customer["preferences"]
        keys(prefs, ("budget_cents", "minimum_capacity_ml", "waterproof_required",
                     "priority"), "preferences")
        integer(prefs["budget_cents"], 0, "budget_cents")
        integer(prefs["minimum_capacity_ml"], 0, "minimum_capacity_ml")
        require(type(prefs["waterproof_required"]) is bool, "Invalid waterproof preference")
        require(prefs["priority"] in ("price", "weight", "capacity"), "Invalid priority")
        keys(value["basket"], ("basket_id", "lines"), "basket")
        text(value["basket"]["basket_id"], "basket_id")
        require(isinstance(value["basket"]["lines"], list), "basket lines must be a list")
        catalog_by_sku = {p["sku"]: p for p in catalog["products"]}
        seen = set()
        for line in value["basket"]["lines"]:
            keys(line, ("sku", "quantity"), "basket line")
            text(line["sku"], "basket SKU")
            require(line["sku"] in skus and line["sku"] not in seen, "Invalid/duplicate basket SKU")
            seen.add(line["sku"])
            integer(line["quantity"], 1, "quantity")
            require(line["quantity"] <= catalog_by_sku[line["sku"]]["stock"],
                    "Basket quantity exceeds catalog stock")
        require(isinstance(value["clickstream"], list), "clickstream must be a list")
        previous_sequence = -1
        for event in value["clickstream"]:
            keys(event, ("sequence", "event", "sku"), "clickstream event")
            integer(event["sequence"], 0, "event sequence")
            require(event["sequence"] > previous_sequence, "Clickstream must be ordered")
            previous_sequence = event["sequence"]
            require(event["event"] in ("view", "compare"), "Unsupported clickstream event")
            text(event["sku"], "event SKU")
            require(event["sku"] in skus, "Unknown clickstream SKU")
        return value

    @staticmethod
    def stage(stage, record, value, previous):
        require(stage in STAGES, "Unknown stage")
        index = STAGES.index(stage)
        require((previous is None and index == 0) or
                (isinstance(previous, dict) and index > 0 and
                 previous.get("stage") == STAGES[index - 1]), "Invalid stage order")
        keys(record, ("schema_version", "stage", "status", "input_digest", "data"), "stage")
        require(record["schema_version"] == VERSION and record["stage"] == stage and
                record["status"] == "ok", "Invalid stage envelope")
        require(record["input_digest"] == digest(value if previous is None else previous),
                "Broken handoff digest")
        # Re-derive the bounded deterministic contract rather than trust supplied output.
        expected = BUILDERS[stage](value, previous)
        require(record["data"] == expected, "Stage data violates " + stage + " contract")
        if stage == "normal":
            sources = {p["source_id"]: (product["sku"], p["text"])
                       for product in value["catalog"]["products"]
                       for p in product["passages"]}
            for finding in record["data"]["findings"]:
                citation = finding["citation"]
                sku, original = sources[citation["source_id"]]
                require(sku == record["data"]["sku"] and
                        original[citation["start"]:citation["end"]] == finding["quote"],
                        "Citation is not an exact selected-product source span")
        return record


def compare(value, previous):
    prefs = value["customer"]["preferences"]
    quantities = {line["sku"]: line["quantity"] for line in value["basket"]["lines"]}
    viewed = {event["sku"] for event in value["clickstream"]}
    rows = []
    for product in value["catalog"]["products"]:
        attrs = normalized(product)
        reasons = []
        if product["stock"] <= quantities.get(product["sku"], 0):
            reasons.append("no_unallocated_stock")
        if product["price_cents"] > prefs["budget_cents"]:
            reasons.append("over_budget")
        if attrs["capacity_ml"] < prefs["minimum_capacity_ml"]:
            reasons.append("insufficient_capacity")
        if prefs["waterproof_required"] and not attrs["waterproof"]:
            reasons.append("not_waterproof")
        score = {"price": -product["price_cents"], "weight": -attrs["weight_g"],
                 "capacity": attrs["capacity_ml"]}[prefs["priority"]]
        rows.append({"sku": product["sku"], "name": product["name"],
                     "price_cents": product["price_cents"], "stock": product["stock"],
                     "attributes": attrs, "eligible": not reasons, "exclusions": reasons,
                     "preference_score": score, "previously_viewed": product["sku"] in viewed})
    ranked = sorted((r for r in rows if r["eligible"]),
                    key=lambda r: (-r["preference_score"], not r["previously_viewed"], r["sku"]))
    require(bool(ranked), "No eligible product matches preferences and available stock")
    return {"currency": value["catalog"]["currency"], "priority": prefs["priority"],
            "comparison": sorted(rows, key=lambda r: r["sku"]),
            "ranking": [r["sku"] for r in ranked], "selected_sku": ranked[0]["sku"],
            "ranking_explanation": "Priority score, then consented clickstream interest, then SKU."}


def journey(value, previous):
    selected = previous["data"]["selected_sku"]
    row = next(r for r in previous["data"]["comparison"] if r["sku"] == selected)
    actions = plan(["product_selected"], [
        ("inspect_specs", ["product_selected"], ["specs_inspected"],
         "Inspect catalog evidence before considering an additional unit."),
        ("preview_basket", ["specs_inspected"], ["basket_previewed"],
         "Review catalog price and stock; this plan does not place an order."),
    ])
    return {"sku": selected, "offer": {"price_cents": row["price_cents"], "stock": row["stock"]},
            "actions": actions, "initial_facts": ["product_selected"],
            "research_request": {"sku": selected, "action": actions[0]["action"],
                                 "query": value["research_query"]}}


def normal(value, previous):
    request = previous["data"]["research_request"]
    product = next(p for p in value["catalog"]["products"] if p["sku"] == request["sku"])
    terms = tokens(request["query"])
    matches = [(len(terms & tokens(p["text"])), p) for p in product["passages"]]
    matches = sorted((m for m in matches if m[0] > 0),
                     key=lambda pair: (-pair[0], pair[1]["source_id"]))[:3]
    require(bool(matches), "No matching catalog evidence for selected SKU")
    findings = [{"quote": passage["text"],
                 "citation": {"source_id": passage["source_id"], "sku": product["sku"],
                              "start": 0, "end": len(passage["text"])},
                 "matched_terms": sorted(terms & tokens(passage["text"]))}
                for _, passage in matches]
    return {"sku": request["sku"], "query": request["query"], "findings": findings,
            "offer": previous["data"]["offer"], "journey_actions": previous["data"]["actions"],
            "evidence_label": "Synthetic catalog excerpts; not customer reviews or endorsements."}


def adaptive(value, previous):
    research = previous["data"]
    novice = value["customer"]["experience"] == "novice"
    priority = value["customer"]["preferences"]["priority"]
    first = "guided_evidence_walkthrough" if novice else "quick_evidence_check"
    actions = plan(["findings_ready"], [
        (first, ["findings_ready"], ["findings_acknowledged"],
         ("Explain units and every citation" if novice else "Summarize source checks") +
         "; prioritize " + priority + " based on consented preferences."),
        ("confirm_basket_preview", ["findings_acknowledged"], ["ready_for_customer_decision"],
         "Check quantities against current fixture stock and prices before any purchase."),
    ])
    catalog = {p["sku"]: p for p in value["catalog"]["products"]}
    quantities = {line["sku"]: line["quantity"] for line in value["basket"]["lines"]}
    quantities[research["sku"]] = quantities.get(research["sku"], 0) + 1
    lines = []
    for sku, quantity in sorted(quantities.items()):
        product = catalog[sku]
        require(quantity <= product["stock"], "Preview exceeds stock")
        lines.append({"sku": sku, "quantity": quantity, "unit_price_cents": product["price_cents"],
                      "stock": product["stock"], "line_total_cents": quantity * product["price_cents"]})
    return {"sku": research["sku"], "experience": value["customer"]["experience"],
            "priority": priority, "initial_facts": ["findings_ready"], "steps": actions,
            "evidence": research["findings"], "journey_actions": research["journey_actions"],
            "offer": research["offer"],
            "basket_preview": {"basket_id": value["basket"]["basket_id"],
                               "currency": value["catalog"]["currency"], "lines": lines,
                               "total_cents": sum(line["line_total_cents"] for line in lines),
                               "order_placed": False},
            "explanation": "Planned onboarding only; prerequisites are satisfied in plan order."}


BUILDERS = dict(zip(STAGES, (compare, journey, normal, adaptive)))


def run_pipeline(value):
    Validation.input(value)
    records = []
    previous = None
    for stage in STAGES:
        record = {"schema_version": VERSION, "stage": stage, "status": "ok",
                  "input_digest": digest(value if previous is None else previous),
                  "data": BUILDERS[stage](value, previous)}
        Validation.stage(stage, record, value, previous)
        records.append(record)
        previous = record
    return {"schema_version": VERSION, "status": "ok", "synthetic": True, "stages": records,
            "notice": "Demonstrative retail validation only; no compliance certification."}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=unique_object)
        result = run_pipeline(value)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
