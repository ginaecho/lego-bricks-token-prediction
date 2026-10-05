"""Synthetic retail evidence synthesis; Python standard library only.

Run: python -B implementation.py example_input.json
Money uses integer cents. Evidence is attributed, never treated as an endorsement.
Consent checks are demonstrative safeguards, not compliance certification.
"""

import json
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def record(value, fields, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(fields), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    require(len(value) <= 2000, path + " is too long")


def sequence(value, path):
    require(isinstance(value, list), path + " must be an array")
    require(len(value) <= 1000, path + " exceeds the reference limit")


def integer(value, minimum, path):
    require(type(value) is int and value >= minimum, path + " must be an integer >= " + str(minimum))


def unique(value, seen, path):
    text(value, path)
    require(value not in seen, path + " must be unique")
    seen.add(value)


TOPICS = {"durability", "packaging", "ease_of_use"}
KINDS = {"specification", "analyst_observation", "return_summary"}


def validate(payload, phase="input", source=None):
    """Single validation boundary shared by the CLI, synthesis, and output."""
    if phase == "output":
        require(source is not None, "output validation requires its source")
        validate(source)
        require(payload == _synthesize(source), "output does not match grounded deterministic synthesis")
        return payload
    require(phase == "input", "unknown validation phase")
    record(payload, {"schema_version", "synthetic", "catalog", "customer", "basket",
                     "clickstream", "documents", "research_questions"}, "input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version must be 1")
    require(payload["synthetic"] is True, "fixture must be explicitly synthetic")
    catalog = payload["catalog"]
    record(catalog, {"currency", "products"}, "catalog")
    require(catalog["currency"] in ("USD", "EUR", "GBP"), "unsupported catalog currency")
    sequence(catalog["products"], "catalog.products")
    require(bool(catalog["products"]), "catalog must contain a product")
    skus = set()
    products = {}
    for product in catalog["products"]:
        record(product, {"sku", "name", "price_cents", "stock"}, "product")
        unique(product["sku"], skus, "product.sku")
        text(product["name"], "product.name")
        integer(product["price_cents"], 0, "product.price_cents")
        integer(product["stock"], 0, "product.stock")
        products[product["sku"]] = product
    customer = payload["customer"]
    record(customer, {"customer_id", "persona", "consent", "personalization_requested"}, "customer")
    text(customer["customer_id"], "customer.customer_id")
    text(customer["persona"], "customer.persona")
    record(customer["consent"], {"gdpr", "ccpa"}, "customer.consent")
    require(all(type(v) is bool for v in customer["consent"].values()), "consent flags must be booleans")
    require(type(customer["personalization_requested"]) is bool, "personalization_requested must be boolean")
    if customer["personalization_requested"]:
        require(all(customer["consent"].values()), "GDPR and CCPA consent required before personalization")
    basket = payload["basket"]
    record(basket, {"order_id", "customer_id", "items"}, "basket")
    text(basket["order_id"], "basket.order_id")
    require(basket["customer_id"] == customer["customer_id"], "basket customer mismatch")
    sequence(basket["items"], "basket.items")
    basket_skus = set()
    for item in basket["items"]:
        record(item, {"sku", "quantity"}, "basket.item")
        unique(item["sku"], basket_skus, "basket.item.sku")
        require(item["sku"] in skus, "unknown basket SKU")
        integer(item["quantity"], 1, "basket.quantity")
        require(item["quantity"] <= products[item["sku"]]["stock"], "basket exceeds catalog stock")
    sequence(payload["clickstream"], "clickstream")
    event_ids = set()
    for event in payload["clickstream"]:
        record(event, {"event_id", "customer_id", "sku", "action", "timestamp"}, "event")
        unique(event["event_id"], event_ids, "event.event_id")
        require(event["customer_id"] == customer["customer_id"], "clickstream customer mismatch")
        text(event["sku"], "event.sku")
        require(event["sku"] in skus, "unknown clickstream SKU")
        require(event["action"] in ("view", "click", "add_to_basket"), "unsupported event action")
        text(event["timestamp"], "event.timestamp")
        try:
            stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            require(stamp.utcoffset() is not None, "event timestamp must include timezone")
        except ValueError as exc:
            raise ValidationError("invalid timezone-aware event timestamp") from exc
    sequence(payload["documents"], "documents")
    document_ids = set()
    claim_ids = set()
    for document in payload["documents"]:
        record(document, {"document_id", "title", "source_id", "synthetic", "claims"}, "document")
        unique(document["document_id"], document_ids, "document.document_id")
        text(document["title"], "document.title")
        text(document["source_id"], "document.source_id")
        require(document["synthetic"] is True, "documents must be synthetic")
        sequence(document["claims"], "document.claims")
        local_keys = set()
        for claim in document["claims"]:
            record(claim, {"claim_id", "sku", "topic", "value", "evidence_kind"}, "claim")
            unique(claim["claim_id"], claim_ids, "claim.claim_id")
            text(claim["sku"], "claim.sku")
            require(claim["sku"] in skus, "unknown evidence SKU")
            text(claim["topic"], "claim.topic")
            require(claim["topic"] in TOPICS, "unsupported topic; prices and stock come only from catalog")
            require(claim["value"] in ("positive", "negative", "uncertain"), "unsupported claim value")
            require(claim["evidence_kind"] in tuple(KINDS),
                    "reviews and endorsements are not accepted evidence")
            key = (claim["sku"], claim["topic"])
            require(key not in local_keys, "duplicate SKU/topic in document")
            local_keys.add(key)
    sequence(payload["research_questions"], "research_questions")
    question_ids = set()
    for question in payload["research_questions"]:
        record(question, {"question_id", "sku", "topic"}, "research_question")
        unique(question["question_id"], question_ids, "question.question_id")
        text(question["sku"], "question.sku")
        text(question["topic"], "question.topic")
        require(question["sku"] in skus and question["topic"] in TOPICS, "unknown question SKU or topic")
    return payload


def _synthesize(data):
    personalized = data["customer"]["personalization_requested"]
    selected = {p["sku"] for p in data["catalog"]["products"]}
    if personalized:
        selected = {e["sku"] for e in data["clickstream"]}
        selected.update(i["sku"] for i in data["basket"]["items"])
    groups = {}
    for document in data["documents"]:
        for claim in document["claims"]:
            if claim["sku"] in selected:
                key = (claim["sku"], claim["topic"])
                groups.setdefault(key, []).append({
                    "document_id": document["document_id"],
                    "source_id": document["source_id"],
                    "claim_id": claim["claim_id"],
                    "value": claim["value"],
                    "evidence_kind": claim["evidence_kind"],
                })
    for question in data["research_questions"]:
        if question["sku"] in selected:
            groups.setdefault((question["sku"], question["topic"]), [])
    findings = []
    unresolved = []
    disagreements = []
    for (sku, topic), citations in sorted(groups.items()):
        citations.sort(key=lambda c: (c["document_id"], c["claim_id"]))
        values = {c["value"] for c in citations}
        independent_sources = len({c["source_id"] for c in citations})
        if not citations:
            state = "no_evidence"
        elif {"positive", "negative"} <= values:
            state = "disputed"
        elif "uncertain" in values:
            state = "inconclusive"
        elif independent_sources < 2:
            state = "single_source"
        else:
            state = "corroborated"
        finding = {"sku": sku, "topic": topic, "state": state,
                   "values": sorted(values), "independent_source_count": independent_sources,
                   "citations": citations}
        findings.append(finding)
        if state == "disputed":
            disagreements.append({"sku": sku, "topic": topic,
                                  "claim_ids": [c["claim_id"] for c in citations],
                                  "resolution": "Unresolved; conflicting evidence is retained without majority voting."})
        if state != "corroborated":
            unresolved.append({"sku": sku, "topic": topic, "reason": state,
                               "question_ids": sorted(q["question_id"] for q in data["research_questions"]
                                                      if (q["sku"], q["topic"]) == (sku, topic)),
                               "next_step": "Obtain additional independent, comparable evidence."})
    products = {p["sku"]: p for p in data["catalog"]["products"]}
    lines = [{"sku": i["sku"], "quantity": i["quantity"],
              "unit_price_cents": products[i["sku"]]["price_cents"],
              "line_total_cents": i["quantity"] * products[i["sku"]]["price_cents"]}
             for i in sorted(data["basket"]["items"], key=lambda i: i["sku"])]
    covered = {f["sku"] for f in findings if f["citations"]}
    return {
        "schema_version": 1, "status": "ok", "synthetic": True,
        "research": {
            "mode": "consented_personalized" if personalized else "generic",
            "selected_skus": sorted(selected),
            "findings": findings, "disagreements": disagreements,
            "unresolved_questions": unresolved,
            "coverage_gaps": sorted(selected - covered),
            "out_of_scope_question_ids": sorted(q["question_id"] for q in data["research_questions"]
                                                if q["sku"] not in selected),
            "summary": {"finding_count": len(findings), "disagreement_count": len(disagreements),
                        "unresolved_count": len(unresolved)},
        },
        "catalog_snapshot": {"currency": data["catalog"]["currency"],
                             "products": [dict(products[sku]) for sku in sorted(selected)]},
        "basket": {"order_id": data["basket"]["order_id"], "lines": lines,
                   "currency": data["catalog"]["currency"],
                   "total_cents": sum(line["line_total_cents"] for line in lines)},
        "limitations": [
            "Synthetic structured evidence only; no live retrieval or provider calls.",
            "Corroboration means agreement across supplied source IDs, not verified truth or source independence.",
            "Consent and catalog checks are demonstrative validation rules, not compliance certification.",
            "No reviews, endorsements, purchases, tax calculation, or stock reservations are generated.",
        ],
    }


def run(payload):
    validate(payload)
    output = _synthesize(payload)
    return validate(output, "output", payload)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=object_pairs, parse_constant=reject_constant)
        output = run(data)
        code = 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        output = {"schema_version": 1, "status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
