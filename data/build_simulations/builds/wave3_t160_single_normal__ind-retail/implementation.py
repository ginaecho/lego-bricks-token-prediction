"""Synthetic retail research: deterministic retrieval, never generated endorsements.

Run: python -B implementation.py example_input.json
Prices are fixed-point currency strings. Citation offsets are Unicode code-point
offsets into the emitted source passage; JSON pointers identify catalog records.
Consent checks are demonstrative rules, not GDPR/CCPA certification.
"""

import json
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(value) == set(expected), where + " has missing or unknown fields")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()), where + " must be nonempty text")
    require(len(value) <= 10000, where + " is too long")


def integer(value, minimum, where):
    require(type(value) is int and value >= minimum, where + " must be an integer >= " + str(minimum))


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def cents(price):
    whole, fraction = price.split(".")
    return int(whole) * 100 + int(fraction)


def money(value):
    return f"{value // 100}.{value % 100:02d}"


def validate(data):
    """Single shared input-validation boundary for API and CLI."""
    fields(data, ("schema_version", "synthetic", "catalog", "customer", "basket",
                  "clickstream", "research"), "input")
    require(data["schema_version"] == "1.0", "unsupported schema_version")
    require(data["synthetic"] is True, "fixtures must be explicitly synthetic")
    catalog = data["catalog"]
    require(isinstance(catalog, list) and len(catalog) <= 1000, "catalog must be a list of at most 1000 products")
    by_sku = {}
    for i, product in enumerate(catalog):
        fields(product, ("sku", "name", "description", "price", "currency", "stock"), f"catalog[{i}]")
        for key in ("sku", "name", "description"):
            text(product[key], key)
        require(re.fullmatch(r"[A-Z0-9-]{1,40}", product["sku"]) is not None, "invalid SKU")
        require(product["sku"] not in by_sku, "duplicate SKU")
        require(isinstance(product["price"], str) and
                re.fullmatch(r"(0|[1-9][0-9]{0,8})\.[0-9]{2}", product["price"]) is not None,
                "price must be a nonnegative fixed-point string")
        require(product["currency"] in ("USD", "EUR", "GBP"), "unsupported currency")
        integer(product["stock"], 0, "stock")
        # Only factual, noncommercial-claim prose belongs in this bounded feed.
        prose = product["name"] + " " + product["description"]
        require(not re.search(r"review|endorse|testimonial|rated|rating|stars?|recommend|customer says",
                              prose, re.I),
                "reviews and endorsements are not accepted")
        require(not re.search(r"\d|[$€£]|\b(price|stock|cost|available|inventory|sold out|free)\b",
                              prose, re.I),
                "price and stock claims must use structured catalog fields")
        by_sku[product["sku"]] = product
    customer = data["customer"]
    fields(customer, ("id", "persona", "consent"), "customer")
    text(customer["id"], "customer.id")
    text(customer["persona"], "customer.persona")
    fields(customer["consent"], ("personalization", "ccpa_opt_out"), "consent")
    require(all(type(v) is bool for v in customer["consent"].values()), "consent flags must be booleans")
    research = data["research"]
    fields(research, ("query", "max_findings", "personalize"), "research")
    text(research["query"], "query")
    integer(research["max_findings"], 1, "max_findings")
    require(research["max_findings"] <= 20, "max_findings must be <= 20")
    require(type(research["personalize"]) is bool, "personalize must be boolean")
    if research["personalize"]:
        require(customer["consent"]["personalization"] and not customer["consent"]["ccpa_opt_out"],
                "personalization requires affirmative consent and no CCPA opt-out")
    basket = data["basket"]
    fields(basket, ("id", "items"), "basket")
    text(basket["id"], "basket.id")
    require(isinstance(basket["items"], list), "basket.items must be a list")
    seen = set()
    currencies = set()
    for item in basket["items"]:
        fields(item, ("sku", "quantity"), "basket item")
        text(item["sku"], "basket SKU")
        require(item["sku"] in by_sku, "unknown basket SKU")
        require(item["sku"] not in seen, "duplicate basket SKU")
        seen.add(item["sku"])
        integer(item["quantity"], 1, "quantity")
        require(item["quantity"] <= by_sku[item["sku"]]["stock"], "basket quantity exceeds catalog stock")
        currencies.add(by_sku[item["sku"]]["currency"])
    require(len(currencies) <= 1, "mixed-currency basket is unsupported")
    require(isinstance(data["clickstream"], list) and len(data["clickstream"]) <= 10000,
            "clickstream must be a list of at most 10000 events")
    event_ids = set()
    for event in data["clickstream"]:
        fields(event, ("id", "customer_id", "sku", "event", "timestamp"), "clickstream event")
        for key in ("id", "customer_id", "sku", "event", "timestamp"):
            text(event[key], "event." + key)
        require(event["id"] not in event_ids, "duplicate event id")
        event_ids.add(event["id"])
        require(event["customer_id"] == customer["id"], "event belongs to another customer")
        require(event["sku"] in by_sku, "unknown event SKU")
        require(event["event"] in ("view", "add_to_basket"), "unsupported clickstream event")
        try:
            parsed = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            raise ValidationError("invalid event timestamp") from None
        require(parsed.tzinfo is not None, "event timestamp needs timezone")
    return by_sku


def validate_output(result, data, by_sku):
    """Enforce the same catalog and evidence invariants on emitted results."""
    sources = {s["id"]: s for s in result["sources"]}
    for finding in result["findings"]:
        citation = finding["citation"]
        source = sources[citation["source_id"]]
        require(finding["quote"] == source["text"][citation["start"]:citation["end"]],
                "citation does not match exact source span")
        product = by_sku[finding["sku"]]
        require(all(finding[key] == product[key] for key in ("price", "stock", "currency")),
                "finding differs from catalog")
    require(result["personalized"] == data["research"]["personalize"], "personalization mismatch")
    return result


def run(data):
    by_sku = validate(data)
    research = data["research"]
    terms = tokens(research["query"])
    activity = {}
    if research["personalize"]:
        for event in data["clickstream"]:
            activity[event["sku"]] = activity.get(event["sku"], 0) + 1
    sources = []
    candidates = []
    for i, product in enumerate(data["catalog"]):
        passage = (f'{product["name"]}: {product["description"]} '
                   f'Price: {product["price"]} {product["currency"]}. Stock: {product["stock"]}.')
        source = {"id": f'catalog:{product["sku"]}', "kind": "catalog",
                  "input_pointer": f"/catalog/{i}", "text": passage}
        sources.append(source)
        overlap = sorted(terms & tokens(product["sku"] + " " + passage))
        if overlap:
            candidates.append((-len(overlap), -activity.get(product["sku"], 0),
                               product["sku"], product, source, overlap))
    findings = []
    for _, _, sku, product, source, overlap in sorted(candidates)[:research["max_findings"]]:
        findings.append({
            "sku": sku, "quote": source["text"], "matched_terms": overlap,
            "price": product["price"], "stock": product["stock"], "currency": product["currency"],
            "citation": {"source_id": source["id"], "start": 0, "end": len(source["text"])},
        })
    lines = []
    total = 0
    for item in data["basket"]["items"]:
        product = by_sku[item["sku"]]
        subtotal = cents(product["price"]) * item["quantity"]
        total += subtotal
        lines.append({**item, "unit_price": product["price"], "stock": product["stock"],
                      "currency": product["currency"], "subtotal": money(subtotal)})
    result = {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "query": research["query"], "personalized": research["personalize"],
        "findings": findings, "sources": sources,
        "basket": {"id": data["basket"]["id"], "items": lines, "total": money(total),
                   "currency": lines[0]["currency"] if lines else None},
        "notice": "Synthetic reference only; no compliance certification. No reviews or endorsements.",
        "no_matches": not findings,
    }
    return validate_output(result, data, by_sku)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object)
        output = run(data)
        code = 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        output = {"schema_version": "1.0", "status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
