"""Synthetic, offline retail discovery reference. Python standard library only."""

import copy
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from decimal import Decimal
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing fields: " + str(set(required) - set(value)))
    require(set(value) <= set(required) | set(optional), "Unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")


def number(value, name, minimum=None):
    require(type(value) in (int, float), name + " must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, name + " must be finite")
    require(minimum is None or value >= minimum, name + " is below minimum")


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("Invalid ISO timestamp") from exc
    require(result.tzinfo is not None, "Timestamp needs timezone")
    return result


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def safe_content(value):
    text(value, "source text")
    require(not re.search(r"\b(review\w*|endorse\w*|testimonial\w*|stars?|rated)\b", value, re.I),
            "Reviews and endorsements are excluded from this synthetic reference")


def allowed_url(url, domains):
    text(url, "URL")
    require(not any(c.isspace() or ord(c) < 32 for c in url) and "\\" not in url,
            "Invalid URL characters")
    try:
        parts = urlsplit(url)
        valid = (parts.scheme == "https" and parts.hostname in domains
                 and parts.username is None and parts.password is None
                 and parts.port in (None, 443) and not parts.fragment)
    except ValueError as exc:
        raise ValidationError("Invalid URL") from exc
    require(valid, "URL is not HTTPS on an exactly allowlisted host")


def validate_input(data):
    fields(data, ("schema_version", "fixture_label", "query", "limit", "as_of", "catalog",
                  "customer", "basket", "events", "personalization", "research"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Unsupported schema_version")
    require(data["fixture_label"] == "SYNTHETIC", "Fixtures must be labeled SYNTHETIC")
    text(data["query"], "query")
    require(type(data["limit"]) is int and 1 <= data["limit"] <= 50, "limit must be 1..50")
    now = timestamp(data["as_of"])
    require(isinstance(data["catalog"], list) and bool(data["catalog"]), "Empty catalog")
    catalog = {}
    for product in data["catalog"]:
        fields(product, ("sku", "name", "description", "category", "price", "currency", "stock"))
        for key in ("sku", "name", "description", "category"):
            text(product[key], key)
        safe_content(product["name"] + " " + product["description"])
        require(product["sku"] not in catalog, "Duplicate catalog SKU")
        number(product["price"], "price", 0)
        require(Decimal(str(product["price"])).as_tuple().exponent >= -2,
                "Prices may have at most two decimal places")
        require(isinstance(product["currency"], str) and
                re.fullmatch(r"[A-Z]{3}", product["currency"]) is not None,
                "Currency needs three uppercase letters")
        require(type(product["stock"]) is int and product["stock"] >= 0, "Invalid stock")
        catalog[product["sku"]] = product
    customer = data["customer"]
    fields(customer, ("id", "interests", "consent"))
    text(customer["id"], "customer.id")
    require(isinstance(customer["interests"], list), "interests must be a list")
    for interest in customer["interests"]:
        text(interest, "interest")
    fields(customer["consent"], ("gdpr", "ccpa"))
    require(all(type(v) is bool for v in customer["consent"].values()), "Consent must be boolean")
    fields(data["personalization"], ("enabled",))
    require(type(data["personalization"]["enabled"]) is bool, "enabled must be boolean")
    require(not data["personalization"]["enabled"] or all(customer["consent"].values()),
            "GDPR/CCPA consent is required before personalization")
    basket = data["basket"]
    fields(basket, ("id", "customer_id", "items"))
    text(basket["id"], "basket.id")
    require(basket["customer_id"] == customer["id"], "Basket belongs to another customer")
    require(isinstance(basket["items"], list), "Basket items must be a list")
    seen = set()
    for item in basket["items"]:
        fields(item, ("sku", "quantity"))
        text(item["sku"], "basket.sku")
        require(item["sku"] in catalog and item["sku"] not in seen, "Invalid/duplicate basket SKU")
        require(type(item["quantity"]) is int and
                1 <= item["quantity"] <= catalog[item["sku"]]["stock"], "Invalid basket quantity")
        seen.add(item["sku"])
    require(isinstance(data["events"], list), "events must be a clickstream list")
    seen = set()
    for event in data["events"]:
        fields(event, ("id", "customer_id", "sku", "type", "timestamp"))
        text(event["id"], "event.id")
        text(event["sku"], "event.sku")
        require(event["id"] not in seen, "Duplicate event")
        seen.add(event["id"])
        require(event["customer_id"] == customer["id"], "Event belongs to another customer")
        require(event["sku"] in catalog, "Unknown event SKU")
        require(event["type"] in ("click", "purchase"), "Unsupported event type")
        require(timestamp(event["timestamp"]) <= now, "Future event")
    research = data["research"]
    fields(research, ("allowlisted_domains", "documents", "max_findings_per_product"))
    require(isinstance(research["allowlisted_domains"], list), "domains must be a list")
    for domain in research["allowlisted_domains"]:
        require(isinstance(domain, str) and
                re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)+", domain)
                is not None, "Invalid allowlisted domain")
    require(type(research["max_findings_per_product"]) is int and
            1 <= research["max_findings_per_product"] <= 5, "Invalid findings limit")
    require(isinstance(research["documents"], list), "documents must be a list")
    seen = set()
    for doc in research["documents"]:
        fields(doc, ("url", "sku", "kind", "title", "text", "captured_at"))
        allowed_url(doc["url"], research["allowlisted_domains"])
        require(doc["url"] not in seen, "Duplicate document URL")
        seen.add(doc["url"])
        text(doc["sku"], "document.sku")
        require(doc["sku"] in catalog, "Unknown document SKU")
        require(doc["kind"] == "product_specification", "Only specifications are ingested")
        safe_content(doc["title"])
        safe_content(doc["text"])
        # Research never supplies commercial fields, even inside quoted evidence.
        require(not re.search(r"[$€£]|\b(price|stock|USD|EUR|GBP|available|availability)\b",
                              doc["title"] + " " + doc["text"], re.I),
                "Price and stock claims belong exclusively to the catalog")
        require(timestamp(doc["captured_at"]) <= now, "Future document capture")
    return catalog


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_output(output, data, expected_stage):
    """One envelope and validation layer is used at every handoff."""
    catalog = validate_input(data)
    fields(output, ("status", "schema_version", "fixture_label", "input_digest", "stage",
                    "query", "customer_id", "basket_id", "personalization_mode", "items"))
    require(output["status"] == "ok" and output["schema_version"] == 1, "Invalid output header")
    require(output["fixture_label"] == "SYNTHETIC", "Missing synthetic output label")
    require(output["stage"] == expected_stage, "Wrong stage handoff")
    require(output["input_digest"] == fingerprint(data), "Handoff input changed")
    require(output["query"] == data["query"] and output["customer_id"] == data["customer"]["id"]
            and output["basket_id"] == data["basket"]["id"], "Handoff context mismatch")
    mode = ("disabled" if not data["personalization"]["enabled"] else
            "history" if data["events"] or data["basket"]["items"] else "cold_start")
    require(output["personalization_mode"] == ("pending" if expected_stage == "semantic" else mode),
            "Invalid personalization mode")
    require(isinstance(output["items"], list) and len(output["items"]) <= data["limit"],
            "Invalid output items")
    docs = {doc["url"]: doc for doc in data["research"]["documents"]}
    seen = set()
    for item in output["items"]:
        fields(item, ("sku", "name", "category", "price", "currency", "stock", "semantic_score",
                      "behavior_score", "score", "findings"))
        text(item["sku"], "output.sku")
        require(item["sku"] in catalog and item["sku"] not in seen, "Invalid output SKU")
        seen.add(item["sku"])
        product = catalog[item["sku"]]
        for key in ("name", "category", "price", "currency", "stock"):
            require(type(item[key]) is type(product[key]) and item[key] == product[key],
                    "Output must match catalog: " + key)
        for key in ("semantic_score", "behavior_score", "score"):
            number(item[key], key, 0)
        require(item["score"] == round(item["semantic_score"] + item["behavior_score"], 6),
                "Score propagation mismatch")
        require(expected_stage != "semantic" or item["behavior_score"] == 0,
                "Semantic stage cannot personalize")
        require(mode != "disabled" or item["behavior_score"] == 0, "Personalization without consent")
        require(isinstance(item["findings"], list) and
                len(item["findings"]) <= data["research"]["max_findings_per_product"],
                "Invalid findings")
        require(expected_stage == "web" or not item["findings"], "Premature research findings")
        source_urls = set()
        for finding in item["findings"]:
            fields(finding, ("url", "title", "quote", "captured_at", "content_sha256", "retrieval_score"))
            text(finding["url"], "finding.url")
            require(finding["url"] in docs and finding["url"] not in source_urls,
                    "Unknown/duplicate finding source")
            source_urls.add(finding["url"])
            doc = docs[finding["url"]]
            require(doc["sku"] == item["sku"], "Finding belongs to another SKU")
            require(finding["title"] == doc["title"] and finding["quote"] == doc["text"] and
                    finding["captured_at"] == doc["captured_at"] and
                    finding["content_sha256"] == hashlib.sha256(doc["text"].encode()).hexdigest(),
                    "Finding provenance mismatch")
            number(finding["retrieval_score"], "retrieval_score", 0)
    require(output["items"] == sorted(output["items"], key=lambda i: (-i["score"], i["sku"])),
            "Output ranking is not deterministic")
    return output


def semantic_search(data, embedding=None):
    catalog = validate_input(data)
    query_tokens = tokens(data["query"])
    require(bool(query_tokens), "Query needs searchable alphanumeric terms")
    products = sorted(catalog.values(), key=lambda product: product["sku"])
    texts = [" ".join(product[key] for key in ("name", "description", "category"))
             for product in products]
    similarities = [0.0] * len(products)
    if embedding is not None:
        require(callable(embedding), "embedding must be callable")
        try:
            vectors = embedding([data["query"]] + texts)
        except Exception as exc:
            raise ValidationError("Injected embedding callable failed") from exc
        require(isinstance(vectors, list) and len(vectors) == len(products) + 1,
                "Embedding row count mismatch")
        width = None
        normalized = []
        for vector in vectors:
            require(isinstance(vector, list) and 0 < len(vector) <= 4096, "Invalid embedding vector")
            width = len(vector) if width is None else width
            require(len(vector) == width, "Embedding dimension mismatch")
            for value in vector:
                number(value, "embedding component")
            scale = max(abs(value) for value in vector)
            require(scale > 0, "Zero embedding vector")
            scaled = [value / scale for value in vector]
            norm = math.sqrt(sum(value * value for value in scaled))
            normalized.append([value / norm for value in scaled])
        similarities = [max(0.0, min(1.0, sum(a * b for a, b in zip(normalized[0], vector))))
                        for vector in normalized[1:]]
    # A deterministic inverted index supports lexical retrieval without dependencies.
    index = {}
    for position, value in enumerate(texts):
        for term in tokens(value):
            index.setdefault(term, set()).add(position)
    lexical = [0.0] * len(products)
    for term in query_tokens:
        for position in index.get(term, ()):
            lexical[position] += 1 / len(query_tokens)
    items = []
    for product, lexical_score, similarity in zip(products, lexical, similarities):
        score = round(lexical_score + similarity, 6)
        if score <= 0:
            continue
        item = {key: product[key] for key in ("sku", "name", "category", "price", "currency", "stock")}
        item.update(semantic_score=score, behavior_score=0.0, score=score, findings=[])
        items.append(item)
    items.sort(key=lambda item: (-item["score"], item["sku"]))
    result = dict(status="ok", schema_version=1, fixture_label="SYNTHETIC",
                  input_digest=fingerprint(data), stage="semantic", query=data["query"],
                  customer_id=data["customer"]["id"], basket_id=data["basket"]["id"],
                  personalization_mode="pending", items=items[:data["limit"]])
    return validate_output(result, data, "semantic")


def personalize(data, previous):
    validate_output(previous, data, "semantic")
    result = copy.deepcopy(previous)
    result["stage"] = "behavior"
    enabled = data["personalization"]["enabled"]
    result["personalization_mode"] = ("disabled" if not enabled else
                                     "history" if data["events"] or data["basket"]["items"]
                                     else "cold_start")
    now = timestamp(data["as_of"])
    catalog = {product["sku"]: product for product in data["catalog"]}
    weights = {}
    if enabled:
        for event in data["events"]:
            age_days = (now - timestamp(event["timestamp"])).total_seconds() / 86400
            weight = (3.0 if event["type"] == "purchase" else 1.0) * 2 ** (-age_days / 30)
            weights[event["sku"]] = weights.get(event["sku"], 0.0) + weight
        for item in data["basket"]["items"]:
            weights[item["sku"]] = weights.get(item["sku"], 0.0) + 0.5 * item["quantity"]
    for item in result["items"]:
        boost = 0.0
        if enabled:
            boost = weights.get(item["sku"], 0.0)
            boost += 0.25 * sum(weight for sku, weight in sorted(weights.items())
                                if sku != item["sku"] and catalog[sku]["category"] == item["category"])
            if item["category"] in data["customer"]["interests"]:
                boost += 0.2
        item["behavior_score"] = round(boost, 6)
        item["score"] = round(item["semantic_score"] + item["behavior_score"], 6)
    result["items"].sort(key=lambda item: (-item["score"], item["sku"]))
    return validate_output(result, data, "behavior")


def research(data, previous):
    validate_output(previous, data, "behavior")
    result = copy.deepcopy(previous)
    result["stage"] = "web"
    terms = tokens(previous["query"])
    for item in result["items"]:
        matches = []
        for doc in data["research"]["documents"]:
            if doc["sku"] != item["sku"]:
                continue
            relevance = len(terms & tokens(doc["title"] + " " + doc["text"])) / len(terms)
            if relevance > 0:
                matches.append(dict(url=doc["url"], title=doc["title"], quote=doc["text"],
                                    captured_at=doc["captured_at"],
                                    content_sha256=hashlib.sha256(doc["text"].encode()).hexdigest(),
                                    retrieval_score=round(relevance, 6)))
        matches.sort(key=lambda finding: (-finding["retrieval_score"], finding["url"]))
        item["findings"] = matches[:data["research"]["max_findings_per_product"]]
    return validate_output(result, data, "web")


def run_pipeline(data, embedding=None):
    validate_input(data)
    return research(data, personalize(data, semantic_search(data, embedding)))


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
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except (ValueError, OSError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
