"""Deterministic synthetic retail documents -> insights -> research CLI.

Run: python -B implementation.py example_input.json
Money uses integer minor units. Consent checks are demonstrative controls,
not a GDPR/CCPA compliance certification. Outputs are not public endorsements.
"""

import copy
import datetime
import json
import re
import sys


SCHEMA_VERSION = "1.0"
BASE_FIELDS = {
    "schema_version", "synthetic", "catalog", "customers", "orders",
    "clickstream", "feedback", "sources", "questions",
}
STAGES = ("input", "documents", "insights", "research")
THEMES = {
    "price": ({"price", "expensive", "cost", "affordable"}, "Review pricing clarity."),
    "availability": ({"stock", "unavailable", "sold", "availability"}, "Review inventory availability."),
    "delivery": ({"delivery", "shipping", "late", "arrived"}, "Review delivery communication."),
    "quality": ({"quality", "broken", "durable", "damaged"}, "Review product quality feedback."),
    "usability": ({"easy", "confusing", "checkout", "navigation"}, "Review shopping usability."),
}
STOP_WORDS = {"the", "a", "an", "is", "are", "and", "or", "to", "of", "for", "what",
              "how", "why", "should", "we", "do", "about", "our", "with", "in"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, label, optional=()):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(required) | (set(value) & set(optional)),
            label + " has missing or unknown fields")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be an integer >= " + str(minimum))


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOP_WORDS


def exact_data(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_data(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_data(a, b) for a, b in zip(actual, expected))
    return actual == expected


def indexed(items, label, key="id"):
    require(isinstance(items, list), label + " must be an array")
    result = {}
    for item in items:
        require(isinstance(item, dict), label + " entries must be objects")
        text(item.get(key), label + "." + key)
        require(item[key] not in result, label + " contains duplicate " + key)
        result[item[key]] = item
    return result


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("timestamp must be ISO 8601") from exc
    require(parsed.tzinfo is not None, "timestamp must include a timezone")


def check_offer(offer, catalog, label):
    fields(offer, {"sku", "price_cents", "stock"}, label)
    require(isinstance(offer["sku"], str) and offer["sku"] in catalog, label + " unknown SKU")
    integer(offer["price_cents"], label + ".price_cents")
    integer(offer["stock"], label + ".stock")
    product = catalog[offer["sku"]]
    require(offer["price_cents"] == product["price_cents"], label + " price differs from catalog")
    require(offer["stock"] == product["stock"], label + " stock differs from catalog")


def validate(envelope, expected_stage=None):
    """One shared validator checks input, industry rules, and every stage handoff."""
    require(isinstance(envelope, dict), "payload must be an object")
    stage = envelope.get("stage", "input")
    require(isinstance(stage, str) and stage in STAGES, "unknown stage")
    if expected_stage is not None:
        require(stage == expected_stage, "expected stage " + expected_stage)
    stage_number = STAGES.index(stage)
    extra = set() if stage == "input" else {"stage", *STAGES[1:stage_number + 1]}
    fields(envelope, BASE_FIELDS | extra, "payload")
    require(envelope["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    require(envelope["synthetic"] is True, "only explicitly synthetic fixtures are supported")
    feed = envelope["catalog"]
    fields(feed, {"currency", "products"}, "catalog")
    require(isinstance(feed["currency"], str) and re.fullmatch(r"[A-Z]{3}", feed["currency"]),
            "currency must be three uppercase letters")
    catalog = indexed(feed["products"], "catalog.products", "sku")
    for product in catalog.values():
        fields(product, {"sku", "name", "price_cents", "stock"}, "product")
        text(product["name"], "product.name")
        integer(product["price_cents"], "product.price_cents")
        integer(product["stock"], "product.stock")
    customers = indexed(envelope["customers"], "customers")
    for customer in customers.values():
        fields(customer, {"id", "persona", "consent"}, "customer")
        text(customer["persona"], "customer.persona")
        fields(customer["consent"], {"personalization", "withdrawn"}, "consent")
        require(all(type(v) is bool for v in customer["consent"].values()), "consent values must be booleans")
    orders = indexed(envelope["orders"], "orders")
    for order in orders.values():
        fields(order, {"id", "customer_id", "status", "items"}, "order")
        require(isinstance(order["customer_id"], str) and order["customer_id"] in customers,
                "order references unknown customer")
        require(order["status"] in ("basket", "completed"), "invalid order status")
        require(isinstance(order["items"], list) and len(order["items"]) > 0, "order items cannot be empty")
        seen = set()
        for line in order["items"]:
            fields(line, {"sku", "quantity", "unit_price_cents"}, "order line")
            require(isinstance(line["sku"], str) and line["sku"] in catalog, "order references unknown SKU")
            require(line["sku"] not in seen, "duplicate SKU in order")
            seen.add(line["sku"])
            integer(line["quantity"], "quantity", 1)
            integer(line["unit_price_cents"], "unit_price_cents")
            require(line["unit_price_cents"] == catalog[line["sku"]]["price_cents"],
                    "order price differs from catalog")
            if order["status"] == "basket":
                require(line["quantity"] <= catalog[line["sku"]]["stock"], "basket exceeds stock")
    events = indexed(envelope["clickstream"], "clickstream")
    for event in events.values():
        fields(event, {"id", "customer_id", "sku", "event", "timestamp",
                       "shown_price_cents", "shown_stock"}, "clickstream event")
        require(isinstance(event["customer_id"], str) and event["customer_id"] in customers,
                "event references unknown customer")
        require(event["event"] in ("view", "add_to_basket", "purchase"), "unknown event type")
        timestamp(event["timestamp"])
        check_offer({"sku": event["sku"], "price_cents": event["shown_price_cents"],
                     "stock": event["shown_stock"]}, catalog, "displayed offer")
    feedback = indexed(envelope["feedback"], "feedback")
    for entry in feedback.values():
        fields(entry, {"id", "customer_id", "sku", "order_id", "kind", "text", "synthetic"}, "feedback")
        require(entry["synthetic"] is True, "feedback must be labeled synthetic")
        require(isinstance(entry["customer_id"], str) and entry["customer_id"] in customers,
                "feedback references unknown customer")
        require(isinstance(entry["sku"], str) and entry["sku"] in catalog, "feedback references unknown SKU")
        require(entry["kind"] in ("review", "service"), "unknown feedback kind")
        text(entry["text"], "feedback.text")
        order_id = entry["order_id"]
        require(order_id is None or (isinstance(order_id, str) and order_id in orders),
                "feedback references unknown order")
        if order_id is not None:
            order = orders[order_id]
            require(order["customer_id"] == entry["customer_id"], "feedback order belongs to another customer")
            require(entry["sku"] in {line["sku"] for line in order["items"]}, "feedback SKU absent from order")
        if entry["kind"] == "review":
            require(order_id is not None and orders[order_id]["status"] == "completed",
                    "reviews require a supplied completed purchase; no fabricated reviews")
    sources = indexed(envelope["sources"], "sources")
    for source in sources.values():
        fields(source, {"id", "title", "text", "synthetic", "catalog_claims"}, "source")
        text(source["title"], "source.title")
        text(source["text"], "source.text")
        require(source["synthetic"] is True, "sources must be labeled synthetic")
        require(isinstance(source["catalog_claims"], list), "catalog_claims must be an array")
        for claim in source["catalog_claims"]:
            check_offer(claim, catalog, "source catalog claim")
    questions = indexed(envelope["questions"], "questions")
    for question in questions.values():
        fields(question, {"id", "text"}, "question")
        text(question["text"], "question.text")
    if stage_number >= 1:
        require(exact_data(envelope["documents"], document_data(envelope)), "documents output failed integrity validation")
    if stage_number >= 2:
        require(exact_data(envelope["insights"], insight_data(envelope)), "insights output failed integrity validation")
    if stage_number >= 3:
        require(exact_data(envelope["research"], research_data(envelope)), "research output failed integrity validation")
    return envelope


def document_data(data):
    """Extract typed rows and reshape input without inventing source content."""
    return {
        "offers": [{"sku": p["sku"], "name": p["name"], "price_cents": p["price_cents"],
                    "stock": p["stock"], "currency": data["catalog"]["currency"]}
                   for p in data["catalog"]["products"]],
        "order_rows": [{"order_id": o["id"], "customer_id": o["customer_id"], "status": o["status"],
                        "sku": line["sku"], "quantity": line["quantity"],
                        "unit_price_cents": line["unit_price_cents"],
                        "line_total_cents": line["quantity"] * line["unit_price_cents"]}
                       for o in data["orders"] for line in o["items"]],
        "feedback_rows": [{"feedback_id": f["id"], "customer_id": f["customer_id"],
                           "sku": f["sku"], "kind": f["kind"], "text": f["text"].strip(),
                           "synthetic": True} for f in data["feedback"]],
        "event_rows": copy.deepcopy(data["clickstream"]),
        "source_rows": copy.deepcopy(data["sources"]),
        "question_rows": copy.deepcopy(data["questions"]),
        "checks": {"catalog_consistent": True, "review_provenance_checked": True,
                   "synthetic_only": True},
    }


def insight_data(data):
    docs = data["documents"]
    grouped = {}
    for row in docs["feedback_rows"]:
        matched = [name for name, (words, _) in THEMES.items() if tokens(row["text"]) & words]
        for name in matched or ["other"]:
            grouped.setdefault(name, []).append(row)
    themes = []
    for name, rows in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        themes.append({
            "theme": name, "count": len(rows),
            "feedback_ids": [r["feedback_id"] for r in rows],
            "skus": sorted({r["sku"] for r in rows}),
            "action": THEMES[name][1] if name in THEMES else "Manually review uncategorized feedback.",
        })
    product_metrics = []
    for offer in docs["offers"]:
        sku = offer["sku"]
        events = [e for e in docs["event_rows"] if e["sku"] == sku]
        product_metrics.append({
            "sku": sku, "views": sum(e["event"] == "view" for e in events),
            "basket_adds": sum(e["event"] == "add_to_basket" for e in events),
            "purchase_events": sum(e["event"] == "purchase" for e in events),
            "completed_units": sum(r["quantity"] for r in docs["order_rows"]
                                   if r["sku"] == sku and r["status"] == "completed"),
        })
    personalization = []
    for customer in data["customers"]:
        consent = customer["consent"]
        if not consent["personalization"] or consent["withdrawn"]:
            continue
        seen = {e["sku"] for e in docs["event_rows"] if e["customer_id"] == customer["id"]}
        offers = [copy.deepcopy(offer) for offer in docs["offers"] if offer["sku"] in seen and offer["stock"] > 0]
        if offers:
            personalization.append({"customer_id": customer["id"], "basis": "explicit_active_consent",
                                    "offers": offers})
    return {
        "themes": themes, "product_metrics": product_metrics, "personalization": personalization,
        "feedback_count": len(docs["feedback_rows"]),
        "notice": "Synthetic aggregate themes; not endorsements. Counts may overlap across themes.",
    }


def research_data(data):
    docs = data["documents"]
    insights = data["insights"]
    evidence = []
    for source in docs["source_rows"]:
        evidence.append({"id": "source:" + source["id"], "kind": "supplied_source",
                         "text": source["text"], "title": source["title"], "synthetic": True})
    for theme in insights["themes"]:
        evidence.append({"id": "theme:" + theme["theme"], "kind": "observed_feedback_theme",
                         "text": "{}: {} supplied feedback records.".format(theme["theme"], theme["count"]),
                         "feedback_ids": theme["feedback_ids"], "action": theme["action"], "synthetic": True})
    for offer in docs["offers"]:
        evidence.append({"id": "catalog:" + offer["sku"], "kind": "catalog_fact",
                         "text": "{} {}: price {} {} cents; stock {}.".format(
                             offer["name"], offer["sku"], offer["currency"], offer["price_cents"], offer["stock"]),
                         "offer": copy.deepcopy(offer),
                         "synthetic": True})
    answers = []
    for question in docs["question_rows"]:
        query = tokens(question["text"])
        scored = [(len(query & tokens(e["text"] + " " + e.get("title", ""))), e) for e in evidence]
        selected = [e for score, e in sorted(scored, key=lambda pair: (-pair[0], pair[1]["id"])) if score > 0]
        answers.append({
            "question_id": question["id"], "question": question["text"],
            "status": "evidence_found" if selected else "insufficient_evidence",
            "evidence_ids": [e["id"] for e in selected],
            "findings": [{"evidence_id": e["id"], "text": e["text"]} for e in selected],
            "recommended_actions": [{"action": e["action"], "evidence_id": e["id"]}
                                    for e in selected if "action" in e],
            "limitations": "Lexical relevance is not proof or causation; source claims are unverified. "
                          "All material is synthetic. No generated reviews or endorsements.",
        })
    return {"evidence": evidence, "answers": answers,
            "method": "Deterministic lexical retrieval and extractive findings; no external research."}


def documents(data):
    validate(data, "input")
    output = copy.deepcopy(data)
    output["stage"] = "documents"
    output["documents"] = document_data(output)
    return validate(output, "documents")


def insights(data):
    validate(data, "documents")
    output = copy.deepcopy(data)
    output["stage"] = "insights"
    output["insights"] = insight_data(output)
    return validate(output, "insights")


def research(data):
    validate(data, "insights")
    output = copy.deepcopy(data)
    output["stage"] = "research"
    output["research"] = research_data(output)
    return validate(output, "research")


def run_pipeline(data):
    return {"status": "ok", "data": research(insights(documents(data)))}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("invalid JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
