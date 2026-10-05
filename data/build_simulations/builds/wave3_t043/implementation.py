"""Synthetic retail feedback analysis; standard-library reference, not certification."""
import json
import re
import sys
import unicodedata
from decimal import Decimal


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected.split()), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def integer(value, minimum, path):
    require(type(value) is int and value >= minimum, path + " must be an integer >= " + str(minimum))


def records(value, path):
    require(isinstance(value, list), path + " must be an array")
    return value


def identifier(value, path):
    text(value, path)
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is not None, path + " is invalid")


def money(value, path):
    require(isinstance(value, str) and re.fullmatch(r"(0|[1-9][0-9]*)\.[0-9]{2}", value) is not None,
            path + " must be a nonnegative decimal string with two places")
    return Decimal(value)


def index_rows(rows, key, expected, path):
    indexed = {}
    for row in records(rows, path):
        fields(row, expected, path)
        identifier(row[key], path + "." + key)
        require(row[key] not in indexed, path + " contains a duplicate ID")
        indexed[row[key]] = row
    return indexed


def validate(data):
    """The sole validation boundary for file, API, and optional personalization."""
    fields(data, "schema_version fixture_label catalog customers orders clickstream feedback personalize", "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "unsupported schema_version")
    require(data["fixture_label"] == "SYNTHETIC_FICTITIOUS_RETAIL", "clearly labeled synthetic fixtures required")
    require(type(data["personalize"]) is bool, "personalize must be boolean")
    fields(data["catalog"], "currency products", "catalog")
    require(data["catalog"]["currency"] in ("USD", "EUR", "GBP"), "unsupported catalog currency")
    products = index_rows(data["catalog"]["products"], "sku", "sku name price stock", "products")
    for product in products.values():
        text(product["name"], "product.name")
        money(product["price"], "product.price")
        integer(product["stock"], 0, "product.stock")
    customers = index_rows(data["customers"], "customer_id", "customer_id persona consent", "customers")
    for customer in customers.values():
        text(customer["persona"], "customer.persona")
        fields(customer["consent"], "gdpr_personalization ccpa_personalization", "consent")
        require(all(type(v) is bool for v in customer["consent"].values()), "consent values must be boolean")
        if data["personalize"]:
            require(all(customer["consent"].values()), "GDPR/CCPA consent required before personalization")
    orders = index_rows(data["orders"], "order_id", "order_id customer_id items", "orders")
    for order in orders.values():
        require(order["customer_id"] in customers, "order has unknown customer")
        require(bool(records(order["items"], "order.items")), "order basket must not be empty")
        seen = set()
        for item in order["items"]:
            fields(item, "sku quantity shown_price shown_stock", "basket item")
            identifier(item["sku"], "basket sku")
            require(item["sku"] in products, "basket has unknown SKU")
            require(item["sku"] not in seen, "duplicate basket SKU")
            seen.add(item["sku"])
            integer(item["quantity"], 1, "quantity")
            snapshot(item, products[item["sku"]])
    events = index_rows(data["clickstream"], "event_id",
                        "event_id customer_id order_id sku action shown_price shown_stock", "clickstream")
    for event in events.values():
        require(event["customer_id"] in customers, "event has unknown customer")
        require(event["sku"] in products, "event has unknown SKU")
        require(event["action"] in ("view", "add_to_basket", "purchase"), "invalid clickstream action")
        if event["order_id"] is not None:
            require(event["order_id"] in orders, "event has unknown order")
            order = orders[event["order_id"]]
            require(order["customer_id"] == event["customer_id"], "event order customer mismatch")
            require(event["sku"] in {i["sku"] for i in order["items"]}, "event SKU absent from basket")
        require(event["action"] != "purchase" or event["order_id"] is not None, "purchase requires order")
        snapshot(event, products[event["sku"]])
    feedback = index_rows(data["feedback"], "feedback_id",
                          "feedback_id customer_id order_id sku text provenance", "feedback")
    for item in feedback.values():
        require(item["provenance"] == "synthetic_fixture", "fabricated reviews or endorsements are prohibited; only labeled fixture evidence accepted")
        require(item["customer_id"] in customers, "feedback has unknown customer")
        require(item["order_id"] in orders, "feedback has unknown order")
        require(item["sku"] in products, "feedback has unknown SKU")
        order = orders[item["order_id"]]
        require(order["customer_id"] == item["customer_id"], "feedback order customer mismatch")
        require(item["sku"] in {i["sku"] for i in order["items"]}, "feedback SKU absent from basket")
        text(item["text"], "feedback.text")
        require(len(item["text"]) <= 10000, "feedback text too long")
    return products


def snapshot(row, product):
    money(row["shown_price"], "shown_price")
    integer(row["shown_stock"], 0, "shown_stock")
    require(row["shown_price"] == product["price"], "shown price must match catalog")
    require(row["shown_stock"] == product["stock"], "shown stock must match catalog")


THEMES = {
    "delivery": {"delivery", "shipping", "late", "arrived"},
    "packaging": {"packaging", "box", "wrapped"},
    "price_value": {"price", "expensive", "cheap", "value"},
    "product_quality": {"quality", "broken", "durable", "defective"},
    "availability": {"stock", "available", "unavailable"},
}


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def analyze(data):
    products = validate(data)
    groups = {}
    # Customer/order/SKU scoping avoids treating separate shoppers as duplicates.
    for item in sorted(data["feedback"], key=lambda x: x["feedback_id"]):
        key = (item["customer_id"], item["order_id"], item["sku"], normalize(item["text"]))
        if key not in groups:
            groups[key] = {
                "canonical_feedback_id": item["feedback_id"],
                "source_feedback_ids": [],
                "customer_id": item["customer_id"],
                "order_id": item["order_id"],
                "sku": item["sku"],
                "excerpt": item["text"],
                "provenance": "synthetic_fixture",
            }
        groups[key]["source_feedback_ids"].append(item["feedback_id"])
    evidence = list(groups.values())
    grouped_themes = {}
    for group in evidence:
        words = set(re.findall(r"\w+", normalize(group["excerpt"])))
        matches = [name for name, vocabulary in THEMES.items() if words & vocabulary] or ["other"]
        for name in matches:
            grouped_themes.setdefault(name, []).append(group["canonical_feedback_id"])
    result = {
        "status": "ok",
        "schema_version": 1,
        "fixture_label": data["fixture_label"],
        "notice": "Synthetic customer-insight fixtures only; not real reviews or endorsements.",
        "counts": {"received": len(data["feedback"]), "unique": len(evidence),
                   "duplicates_removed": len(data["feedback"]) - len(evidence)},
        "catalog": {"currency": data["catalog"]["currency"],
                    "products": [dict(products[sku]) for sku in sorted(products)]},
        "evidence": evidence,
        "themes": [{"theme": name, "unique_feedback_count": len(ids), "evidence_ids": ids}
                   for name, ids in sorted(grouped_themes.items())],
        "personalized_insights": [],
    }
    if data["personalize"]:
        for customer in sorted(data["customers"], key=lambda x: x["customer_id"]):
            ids = {g["canonical_feedback_id"] for g in evidence if g["customer_id"] == customer["customer_id"]}
            result["personalized_insights"].append({
                "customer_id": customer["customer_id"],
                "themes": [t["theme"] for t in result["themes"] if ids.intersection(t["evidence_ids"])],
            })
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number is prohibited")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = analyze(data)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
