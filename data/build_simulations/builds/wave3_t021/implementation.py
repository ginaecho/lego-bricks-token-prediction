"""Synthetic retail support -> discovery reference CLI; standard library only."""

import copy
import datetime
from decimal import Decimal, InvalidOperation
import json
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(names.split()), label + " has missing or unknown fields")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    require(len(value) <= 2000, label + " is too long")


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum, label + " out of range")


def product_fact(product, currency):
    return {key: product[key] for key in ("sku", "name", "price", "stock")} | {
        "currency": currency
    }


def support_payload(context):
    catalog = context["catalog"]
    request = context["request"]
    query = request["query"].casefold()
    products = catalog["products"]
    selected = [p for p in products if p["sku"] in request["skus"]]
    categories = sorted({
        p["category"] for p in products
        if p in selected or p["category"].casefold() in query
    })
    facts = [product_fact(p, catalog["currency"]) for p in selected]
    intent = request["intent"]
    if intent in ("price", "stock", "product"):
        if facts:
            answer = " ".join(
                f'{p["sku"]} ({p["name"]}): {p["currency"]} {p["price"]}; '
                f'stock {p["stock"]}.'
                for p in facts
            )
            resolution = "answered"
        else:
            answer = "Please provide a catalog SKU so I can check verified product information."
            resolution = "needs_information"
    elif intent == "order":
        order = context["order"]
        answer = f'Order {order["id"]}: {order["status"]}. No delivery date is available.'
        resolution = "answered"
    elif intent == "returns":
        days = context["policies"]["return_window_days"]
        answer = f"The supplied store policy allows returns within {days} days. Eligibility requires staff review."
        resolution = "answered"
    else:
        answer = (
            "I can help with catalog products, prices, stock, orders and returns. "
            "Reviews and endorsements are not provided or inferred. "
            f'For other questions contact {context["policies"]["support_contact"]}.'
        )
        resolution = "needs_staff"
    consent = context["customer"]["consent"]["personalization"]
    return {
        "answer": answer,
        "resolution": resolution,
        "facts": facts,
        "discovery_context": {
            "personalization_allowed": consent,
            "interest_categories": categories if consent else [],
            "preferred_categories": context["customer"]["preferred_categories"] if consent else [],
            "basket_skus": [i["sku"] for i in context["order"]["items"]] if consent else [],
            "viewed_skus": [
                e["sku"] for e in context["clickstream"] if e["event"] == "view"
            ] if consent else [],
        },
    }


def recommendation_payload(state):
    handoff = state["support"]["discovery_context"]
    catalog = state["context"]["catalog"]
    personalized = handoff["personalization_allowed"]
    candidates = []
    for product in catalog["products"]:
        if product["stock"] == 0:
            continue
        if personalized and product["sku"] in handoff["basket_skus"]:
            continue
        reasons = []
        score = 0
        if personalized:
            if product["category"] in handoff["interest_categories"]:
                score += 4
                reasons.append("Matches the category discussed with support")
            if product["category"] in handoff["preferred_categories"]:
                score += 2
                reasons.append("Matches a consented profile preference")
            if product["sku"] in handoff["viewed_skus"]:
                score += 1
                reasons.append("Previously viewed with personalization consent")
        candidates.append((
            -score, product["sku"],
            product_fact(product, catalog["currency"]) | {
                "score": score,
                "reasons": reasons or ["Available catalog item; no popularity or review claim"],
            },
        ))
    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    limit = state["context"]["request"]["recommendation_limit"]
    return {
        "mode": "personalized" if personalized else "non_personalized",
        "items": [entry[2] for entry in candidates[:limit]],
        "notice": (
            "Personalization uses explicit supplied consent."
            if personalized else
            "Personalization consent absent; profile, basket, support interests and clickstream are not used for ranking."
        ),
    }


def validate_document(document, stage="input"):
    """One validation boundary for input, support handoff and final output."""
    require(stage in ("input", "support", "complete"), "Unknown validation stage")
    if stage != "input":
        names = "schema_version status context support"
        if stage == "complete":
            names += " recommendations"
        fields(document, names, stage)
        require(type(document["schema_version"]) is int and document["schema_version"] == 1,
                "Unsupported output schema version")
        require(document["status"] == "ok", "Invalid success status")
        validate_document(document["context"])
        require(document["support"] == support_payload(document["context"]),
                "Support facts, answer or handoff do not match validated context")
        if stage == "complete":
            require(document["recommendations"] == recommendation_payload(document),
                    "Recommendations violate catalog grounding, consent or ranking")
        return document

    fields(document, "schema_version synthetic catalog customer order clickstream policies request", "input")
    require(type(document["schema_version"]) is int and document["schema_version"] == 1,
            "Unsupported schema version")
    require(document["synthetic"] is True, "This reference requires labeled synthetic fixtures")
    catalog = document["catalog"]
    fields(catalog, "currency products", "catalog")
    require(catalog["currency"] in ("USD", "EUR", "GBP"), "Unsupported currency")
    require(isinstance(catalog["products"], list), "products must be a list")
    require(len(catalog["products"]) <= 10000, "Catalog too large")
    skus = set()
    categories = set()
    for product in catalog["products"]:
        fields(product, "sku name category price stock", "product")
        for key in ("sku", "name", "category"):
            text(product[key], key)
        require(product["sku"] not in skus, "Duplicate SKU")
        skus.add(product["sku"])
        categories.add(product["category"])
        price = product["price"]
        require(isinstance(price, str) and re.fullmatch(r"(0|[1-9][0-9]{0,8})\.[0-9]{2}", price),
                "price must be a nonnegative two-decimal string")
        try:
            require(Decimal(price).is_finite(), "Nonfinite price")
        except InvalidOperation as exc:
            raise ValidationError("Invalid price") from exc
        integer(product["stock"], 0, 1000000, "stock")
    customer = document["customer"]
    fields(customer, "id consent preferred_categories", "customer")
    text(customer["id"], "customer id")
    fields(customer["consent"], "personalization", "consent")
    require(type(customer["consent"]["personalization"]) is bool, "Consent must be explicit boolean")
    require(isinstance(customer["preferred_categories"], list), "preferred_categories must be a list")
    for category in customer["preferred_categories"]:
        require(isinstance(category, str) and category in categories, "Unknown preferred category")
    order = document["order"]
    fields(order, "id status items", "order")
    text(order["id"], "order id")
    require(order["status"] in ("basket", "placed", "shipped", "delivered", "cancelled"),
            "Invalid order status")
    require(isinstance(order["items"], list), "Order items must be a list")
    basket_skus = set()
    for item in order["items"]:
        fields(item, "sku quantity", "order item")
        require(isinstance(item["sku"], str) and item["sku"] in skus, "Unknown basket SKU")
        require(item["sku"] not in basket_skus, "Duplicate basket SKU")
        basket_skus.add(item["sku"])
        integer(item["quantity"], 1, 10000, "quantity")
    events = document["clickstream"]
    require(isinstance(events, list) and len(events) <= 100000, "Invalid clickstream list")
    for event in events:
        fields(event, "customer_id event sku timestamp", "clickstream event")
        require(event["customer_id"] == customer["id"], "Clickstream belongs to another customer")
        require(event["event"] in ("view", "add_to_cart", "purchase"), "Unsupported click event")
        require(isinstance(event["sku"], str) and event["sku"] in skus, "Unknown clickstream SKU")
        text(event["timestamp"], "timestamp")
        try:
            parsed = datetime.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            require(parsed.utcoffset() is not None, "Clickstream timestamp requires timezone")
        except ValueError as exc:
            raise ValidationError("Invalid timezone-aware ISO timestamp") from exc
    policies = document["policies"]
    fields(policies, "return_window_days support_contact", "policies")
    integer(policies["return_window_days"], 0, 365, "return window")
    text(policies["support_contact"], "support contact")
    request = document["request"]
    fields(request, "intent query skus recommendation_limit", "request")
    require(request["intent"] in ("product", "price", "stock", "order", "returns", "general"),
            "Unknown support intent")
    text(request["query"], "query")
    require(isinstance(request["skus"], list), "Request skus must be a list")
    for sku in request["skus"]:
        require(isinstance(sku, str) and sku in skus, "Unknown requested SKU")
    require(len(set(request["skus"])) == len(request["skus"]), "Duplicate requested SKU")
    integer(request["recommendation_limit"], 1, 20, "recommendation limit")
    return document


def support_stage(context):
    validate_document(context)
    state = {
        "schema_version": 1, "status": "ok",
        "context": copy.deepcopy(context), "support": support_payload(context),
    }
    return validate_document(state, "support")


def recommend_stage(support_state):
    validate_document(support_state, "support")
    result = copy.deepcopy(support_state)
    result["recommendations"] = recommendation_payload(support_state)
    return validate_document(result, "complete")


def run_pipeline(document):
    return recommend_stage(support_stage(document))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Nonfinite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = run_pipeline(document)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
