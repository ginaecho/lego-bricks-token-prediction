"""Synthetic retail triage reference; no network or provider integration."""

import json
import sys
from decimal import Decimal


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def integer(value, minimum, path):
    require(type(value) is int and value >= minimum, path + " must be an integer >= " + str(minimum))


def money(value, path):
    require(isinstance(value, str), path + " must be a decimal string")
    parts = value.split(".")
    require(len(parts) == 2 and parts[0].isascii() and parts[0].isdigit()
            and len(parts[1]) == 2 and parts[1].isascii() and parts[1].isdigit(),
            path + " must have two decimal places")
    return Decimal(value)


def validate(data, result=None):
    """One validation boundary used by both input and optional output checks."""
    obj(data, ("schema_version", "synthetic", "catalog", "customer", "order",
               "clickstream", "ticket", "config"), "input")
    require(data["schema_version"] == "1.0", "unsupported schema_version")
    require(data["synthetic"] is True, "fixtures must be labeled synthetic")
    catalog = data["catalog"]
    obj(catalog, ("currency", "products"), "catalog")
    require(catalog["currency"] in ("USD", "EUR", "GBP"), "unsupported currency")
    require(isinstance(catalog["products"], list) and catalog["products"], "catalog products required")
    products = {}
    for product in catalog["products"]:
        obj(product, ("sku", "name", "price", "stock"), "product")
        text(product["sku"], "sku")
        text(product["name"], "product name")
        require(product["sku"] not in products, "duplicate SKU")
        money(product["price"], "catalog price")
        integer(product["stock"], 0, "stock")
        products[product["sku"]] = product
    customer = data["customer"]
    obj(customer, ("id", "persona", "consent"), "customer")
    text(customer["id"], "customer id")
    text(customer["persona"], "synthetic persona")
    obj(customer["consent"], ("gdpr_personalization", "ccpa_personalization"), "consent")
    require(all(type(v) is bool for v in customer["consent"].values()), "consent must be explicit booleans")
    order = data["order"]
    obj(order, ("id", "customer_id", "status", "items"), "order")
    text(order["id"], "order id")
    require(order["customer_id"] == customer["id"], "order/customer mismatch")
    require(order["status"] in ("basket", "placed", "shipped", "delivered", "cancelled"), "invalid order status")
    require(isinstance(order["items"], list) and order["items"], "order items required")
    item_skus = set()
    for item in order["items"]:
        obj(item, ("sku", "quantity", "shown_price", "shown_stock"), "order item")
        text(item["sku"], "item SKU")
        require(item["sku"] in products and item["sku"] not in item_skus, "unknown or duplicate order SKU")
        item_skus.add(item["sku"])
        integer(item["quantity"], 1, "quantity")
        integer(item["shown_stock"], 0, "shown stock")
        require(money(item["shown_price"], "shown price") == money(products[item["sku"]]["price"], "price"),
                "shown price must match catalog")
        require(item["shown_stock"] == products[item["sku"]]["stock"], "shown stock must match catalog")
    require(isinstance(data["clickstream"], list), "clickstream must be an event-log array")
    for event in data["clickstream"]:
        obj(event, ("sequence", "customer_id", "event", "sku"), "clickstream event")
        integer(event["sequence"], 0, "sequence")
        require(event["customer_id"] == customer["id"], "clickstream/customer mismatch")
        require(event["event"] in ("view", "add_to_basket", "remove_from_basket"), "unknown clickstream event")
        text(event["sku"], "event SKU")
        require(event["sku"] in products, "unknown clickstream SKU")
    sequences = [e["sequence"] for e in data["clickstream"]]
    require(sequences == sorted(set(sequences)), "clickstream sequence must strictly increase")
    ticket = data["ticket"]
    obj(ticket, ("id", "subject", "body", "skus", "personalize", "request_endorsement"), "ticket")
    for key in ("id", "subject", "body"):
        text(ticket[key], "ticket " + key)
    require(isinstance(ticket["skus"], list), "ticket skus must be a list")
    for sku in ticket["skus"]:
        text(sku, "ticket SKU")
        require(sku in products, "unknown ticket SKU")
    require(len(ticket["skus"]) == len(set(ticket["skus"])), "duplicate ticket SKU")
    require(type(ticket["personalize"]) is bool and type(ticket["request_endorsement"]) is bool,
            "ticket flags must be booleans")
    require(not ticket["personalize"] or all(customer["consent"].values()),
            "GDPR/CCPA consent required before personalization")
    require(not ticket["request_endorsement"], "fabricated reviews or endorsements are not supported")
    config = data["config"]
    obj(config, ("rules", "fallback", "urgent_keywords"), "config")
    require(isinstance(config["rules"], list), "rules must be a list")
    categories = set()
    for route in config["rules"] + [config["fallback"]]:
        keys = ("category", "priority", "team", "owner")
        if route is not config["fallback"]:
            keys += ("keywords",)
        obj(route, keys, "route")
        for key in ("category", "team", "owner"):
            text(route[key], "route " + key)
        require(route["category"] not in categories, "duplicate routing category")
        categories.add(route["category"])
        require(route["priority"] in ("low", "normal", "high", "urgent"), "invalid priority")
        if "keywords" in route:
            require(isinstance(route["keywords"], list) and route["keywords"], "rule keywords required")
            for keyword in route["keywords"]:
                text(keyword, "keyword")
    require(isinstance(config["urgent_keywords"], list), "urgent_keywords must be a list")
    for keyword in config["urgent_keywords"]:
        text(keyword, "urgent keyword")
    if result is not None:
        obj(result, ("schema_version", "synthetic", "status", "ticket_id", "customer_id", "order_id",
                     "category", "priority", "route", "reason", "personalization",
                     "products", "reviews", "endorsements"), "output")
        require(result["schema_version"] == "1.0" and result["synthetic"] is True
                and result["status"] == "ok", "invalid output envelope")
        require(result["ticket_id"] == ticket["id"] and result["customer_id"] == customer["id"]
                and result["order_id"] == order["id"], "output identity mismatch")
        routes = config["rules"] + [config["fallback"]]
        selected = next((r for r in routes if r["category"] == result["category"]), None)
        require(selected is not None, "output category is not configured")
        require(result["route"] == {"team": selected["team"], "owner": selected["owner"]},
                "output accountability mismatch")
        require(result["priority"] in ("low", "normal", "high", "urgent"), "invalid output priority")
        text(result["reason"], "output reason")
        require(result["reviews"] == [] and result["endorsements"] == [], "generated endorsements forbidden")
        require(result["products"] == [products[s] for s in ticket["skus"]], "output catalog mismatch")
        expected = {"applied": ticket["personalize"], "recent_skus": []}
        if ticket["personalize"]:
            expected["recent_skus"] = list(dict.fromkeys(e["sku"] for e in data["clickstream"]))
        require(result["personalization"] == expected, "output personalization mismatch")
    return products


def triage(data, classifier=None):
    """An injected classifier may return ONLY a configured category string."""
    products = validate(data)
    config = data["config"]
    ticket = data["ticket"]
    content = (ticket["subject"] + " " + ticket["body"]).casefold()
    selected = config["fallback"]
    reason = "fallback"
    for rule in config["rules"]:
        if any(keyword.casefold() in content for keyword in rule["keywords"]):
            selected = rule
            reason = "first_matching_rule"
            break
    if classifier is not None:
        # Pass only ticket text, never profiles or clickstream, to the local callable.
        try:
            category = classifier({"subject": ticket["subject"], "body": ticket["body"]})
        except Exception as exc:
            raise ValidationError("injected classifier failed") from exc
        require(isinstance(category, str), "classifier must return a category string")
        selected = next((r for r in config["rules"] + [config["fallback"]]
                         if r["category"] == category), None)
        require(selected is not None, "classifier category is not configured")
        reason = "injected_classifier"
    urgent = any(k.casefold() in content for k in config["urgent_keywords"])
    result = {
        "schema_version": "1.0", "synthetic": True, "status": "ok",
        "ticket_id": ticket["id"], "customer_id": data["customer"]["id"],
        "order_id": data["order"]["id"], "category": selected["category"],
        "priority": "urgent" if urgent else selected["priority"],
        "route": {"team": selected["team"], "owner": selected["owner"]},
        "reason": reason + ("+urgent_keyword" if urgent else ""),
        "personalization": {"applied": ticket["personalize"], "recent_skus": []},
        "products": [dict(products[s]) for s in ticket["skus"]],
        "reviews": [], "endorsements": [],
    }
    if ticket["personalize"]:
        result["personalization"]["recent_skus"] = list(dict.fromkeys(e["sku"] for e in data["clickstream"]))
    validate(data, result)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object)
        output = triage(data)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
