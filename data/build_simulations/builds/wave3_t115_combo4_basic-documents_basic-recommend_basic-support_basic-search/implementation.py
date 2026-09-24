"""Deterministic, standard-library-only marketplace reference pipeline."""

import copy
import csv
import io
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation


SCHEMA_VERSION = "1.0"
STAGES = ("input", "documents", "recommend", "support", "search")
FIELDS = {"id", "name", "category", "description", "price", "stock", "tags"}
REQUIRED_FIELDS = {"id", "name", "category", "price", "stock"}
SYNONYMS = {
    "trainers": "sneaker", "trainer": "sneaker", "sneakers": "sneaker",
    "shoes": "shoe", "footwear": "shoe", "notebook": "laptop",
    "notebooks": "laptop", "laptops": "laptop", "coat": "jacket",
    "coats": "jacket", "jackets": "jacket", "rucksack": "backpack",
    "rucksacks": "backpack", "backpacks": "backpack",
}
STOP_WORDS = {"a", "an", "the", "for", "and", "or", "of", "to", "i", "want",
              "please", "find", "me", "with", "is", "are", "some"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, where, allowed, required=()):
    require(isinstance(value, dict), f"{where} must be an object")
    require(set(value) <= set(allowed), f"{where} contains unknown fields")
    require(set(required) <= set(value), f"{where} is missing required fields")


def text(value, where, empty=False):
    require(isinstance(value, str), f"{where} must be text")
    require(empty or bool(value.strip()), f"{where} must not be blank")
    return value.strip()


def strings(value, where):
    require(isinstance(value, list), f"{where} must be a list")
    for item in value:
        text(item, where)
    return value


def integer(value, where, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high),
            f"{where} must be an integer from {low}" + (f" to {high}" if high else ""))
    return value


def money(value, where):
    require(type(value) in (int, float, str), f"{where} must be a monetary number")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValidationError(f"{where} must be a monetary number") from None
    require(amount.is_finite() and 0 <= amount <= Decimal("1000000000"),
            f"{where} must be finite and between 0 and 1000000000")
    require(amount == amount.quantize(Decimal("0.01")), f"{where} allows at most two decimal places")
    return float(amount)


def validate_input(data):
    obj(data, "input", {"schema_version", "fixture_label", "documents", "customer",
                       "policies", "support", "search"},
        {"schema_version", "documents", "customer", "policies", "support", "search"})
    require(data["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    if "fixture_label" in data:
        text(data["fixture_label"], "fixture_label")
    require(isinstance(data["documents"], list) and data["documents"],
            "documents must be a nonempty list")
    doc_ids = set()
    for doc in data["documents"]:
        obj(doc, "document", {"id", "format", "content", "field_map"},
            {"id", "format", "content"})
        doc_id = text(doc["id"], "document.id")
        require(doc_id not in doc_ids, "duplicate document id")
        doc_ids.add(doc_id)
        require(doc["format"] in ("json", "csv"), "document format must be json or csv")
        if doc["format"] == "json":
            require(isinstance(doc["content"], list), "json document content must be a list")
        else:
            text(doc["content"], "csv content")
        mapping = doc.get("field_map", {})
        obj(mapping, "field_map", FIELDS)
        for source in mapping.values():
            text(source, "field_map source")
        names = [mapping.get(field, field) for field in FIELDS]
        require(len(set(names)) == len(names), "field_map source names must be unique")
    customer = data["customer"]
    obj(customer, "customer", {"liked_categories", "liked_tags", "purchased_ids",
                              "budget", "limit"})
    for key in ("liked_categories", "liked_tags", "purchased_ids"):
        strings(customer.get(key, []), f"customer.{key}")
    if "budget" in customer:
        money(customer["budget"], "customer.budget")
    integer(customer.get("limit", 3), "customer.limit", 1, 20)
    obj(data["policies"], "policies", {"shipping", "returns"})
    for value in data["policies"].values():
        text(value, "policy")
    obj(data["support"], "support", {"question"}, {"question"})
    text(data["support"]["question"], "support.question")
    obj(data["search"], "search", {"query", "limit", "in_stock_only", "max_price"})
    text(data["search"].get("query", ""), "search.query", empty=True)
    integer(data["search"].get("limit", 5), "search.limit", 1, 50)
    require(type(data["search"].get("in_stock_only", True)) is bool,
            "search.in_stock_only must be boolean")
    if "max_price" in data["search"]:
        money(data["search"]["max_price"], "search.max_price")


def validate(state, expected_stage):
    """One boundary validator for every producer and consumer in the pipeline."""
    require(expected_stage in STAGES, "unknown stage")
    obj(state, "state", {"schema_version", "status", "stage", "request", "catalog",
                        "document_report", "recommendations", "support", "search"},
        {"schema_version", "status", "stage", "request"})
    require(state["schema_version"] == SCHEMA_VERSION, "invalid state schema")
    require(state["status"] == "ok" and state["stage"] == expected_stage,
            f"expected validated {expected_stage} state")
    validate_input(state["request"])
    index = STAGES.index(expected_stage)
    expected_keys = {"schema_version", "status", "stage", "request"}
    if index >= 1:
        expected_keys |= {"catalog", "document_report"}
    if index >= 2:
        expected_keys.add("recommendations")
    if index >= 3:
        expected_keys.add("support")
    if index >= 4:
        expected_keys.add("search")
    require(set(state) == expected_keys, "state fields do not match stage")
    if index == 0:
        return state
    require(isinstance(state["catalog"], list), "catalog must be a list")
    catalog = {}
    document_ids = {d["id"].strip() for d in state["request"]["documents"]}
    for product in state["catalog"]:
        obj(product, "product", FIELDS | {"source"}, FIELDS | {"source"})
        for key in ("id", "name", "category"):
            text(product[key], f"product.{key}")
        text(product["description"], "product.description", empty=True)
        strings(product["tags"], "product.tags")
        require(type(product["price"]) in (int, float), "normalized price must be numeric")
        money(product["price"], "product.price")
        integer(product["stock"], "product.stock")
        require(product["id"] not in catalog, "duplicate product id")
        obj(product["source"], "source", {"document_id", "row"}, {"document_id", "row"})
        require(product["source"]["document_id"] in document_ids, "unknown source document")
        integer(product["source"]["row"], "source.row", 1)
        catalog[product["id"]] = product
    report = state["document_report"]
    obj(report, "document_report", {"documents", "rows", "products"},
        {"documents", "rows", "products"})
    require(report == {"documents": len(state["request"]["documents"]),
                       "rows": len(catalog), "products": len(catalog)}, "invalid document counts")
    if index >= 2:
        _validate_ranked(state["recommendations"], catalog, "recommendations")
        for item in state["recommendations"]:
            require(catalog[item["product_id"]]["stock"] > 0, "recommended item unavailable")
    if index >= 3:
        support = state["support"]
        obj(support, "support output",
            {"question", "answer", "intent", "escalate", "evidence", "product_ids",
             "suggested_query", "recommendation_ids"},
            {"question", "answer", "intent", "escalate", "evidence", "product_ids",
             "suggested_query", "recommendation_ids"})
        require(support["question"] == state["request"]["support"]["question"].strip(),
                "support question changed")
        for key in ("answer", "intent"):
            text(support[key], f"support.{key}")
        text(support["suggested_query"], "support.suggested_query", empty=True)
        require(type(support["escalate"]) is bool, "escalate must be boolean")
        require(support["recommendation_ids"] ==
                [r["product_id"] for r in state["recommendations"]], "lost recommendation handoff")
        strings(support["product_ids"], "support.product_ids")
        require(all(pid in catalog for pid in support["product_ids"]), "unknown support product")
        require(isinstance(support["evidence"], list), "evidence must be a list")
        for evidence in support["evidence"]:
            obj(evidence, "evidence", {"kind", "id", "field", "value"},
                {"kind", "id", "field", "value"})
            if evidence["kind"] == "product":
                require(evidence["id"] in support["product_ids"], "uncited support product")
                require(evidence["field"] in FIELDS, "unknown evidence field")
                require(evidence["value"] == catalog[evidence["id"]][evidence["field"]],
                        "product evidence does not match catalog")
            else:
                require(evidence["kind"] == "policy" and evidence["id"] in
                        state["request"]["policies"], "unknown policy evidence")
                require(evidence["field"] == "text" and evidence["value"] ==
                        state["request"]["policies"][evidence["id"]], "policy evidence mismatch")
    if index >= 4:
        search = state["search"]
        obj(search, "search output", {"query", "query_source", "normalized_terms",
                                     "results", "support_intent", "recommendation_ids"},
            {"query", "query_source", "normalized_terms", "results", "support_intent",
             "recommendation_ids"})
        explicit = state["request"]["search"].get("query", "").strip()
        require(search["query"] == (explicit or state["support"]["suggested_query"]),
                "lost support query handoff")
        require(search["query_source"] == ("request" if explicit else "support"),
                "incorrect query source")
        require(search["support_intent"] == state["support"]["intent"], "lost support intent")
        require(search["recommendation_ids"] == state["support"]["recommendation_ids"],
                "lost search recommendation handoff")
        require(search["normalized_terms"] == tokens(search["query"]), "invalid query terms")
        _validate_ranked(search["results"], catalog, "search.results")
    return state


def _validate_ranked(items, catalog, where):
    require(isinstance(items, list), f"{where} must be a list")
    seen = set()
    previous = math.inf
    for rank, item in enumerate(items, 1):
        obj(item, where, {"rank", "product_id", "score", "reasons"},
            {"rank", "product_id", "score", "reasons"})
        require(type(item["rank"]) is int and item["rank"] == rank, "invalid rank")
        require(isinstance(item["product_id"], str) and item["product_id"] in catalog,
                "unknown ranked product")
        require(item["product_id"] not in seen, "duplicate ranked product")
        seen.add(item["product_id"])
        score = item["score"]
        require(type(score) in (float, int) and math.isfinite(score) and 0 <= score <= previous,
                "invalid ranked score")
        previous = score
        strings(item["reasons"], f"{where}.reasons")
        require(bool(item["reasons"]), "ranking needs reasons")


def start(data):
    state = {"schema_version": SCHEMA_VERSION, "status": "ok", "stage": "input",
             "request": copy.deepcopy(data)}
    return validate(state, "input")


def documents_stage(state):
    validate(state, "input")
    result = copy.deepcopy(state)
    catalog = []
    seen = set()
    for doc in state["request"]["documents"]:
        mapping = doc.get("field_map", {})
        if doc["format"] == "csv":
            try:
                reader = csv.DictReader(io.StringIO(doc["content"]), strict=True)
                headers = reader.fieldnames
                require(headers and len(set(headers)) == len(headers), "CSV headers missing or duplicated")
                require(all(h and h.strip() for h in headers), "CSV header must not be blank")
                require({mapping.get(field, field) for field in REQUIRED_FIELDS} <= set(headers),
                        "CSV headers missing required product columns")
                rows = list(reader)
            except csv.Error as exc:
                raise ValidationError(f"malformed CSV: {exc}") from None
        else:
            rows = doc["content"]
        for number, row in enumerate(rows, 1):
            require(isinstance(row, dict), "document rows must be objects")
            if doc["format"] == "csv":
                require(None not in row and all(v is not None for v in row.values()),
                        "document row has missing values or surplus CSV cells")
            product = {}
            for field in FIELDS:
                source = mapping.get(field, field)
                require(field not in REQUIRED_FIELDS or source in row,
                        f"document {doc['id']} row {number} missing {field}")
                product[field] = row.get(source, [] if field == "tags" else "")
            for field in ("id", "name", "category", "description"):
                product[field] = text(product[field], f"product.{field}", empty=field == "description")
            require(product["id"] not in seen, f"duplicate product id: {product['id']}")
            seen.add(product["id"])
            product["price"] = money(product["price"], "product.price")
            stock = product["stock"]
            if isinstance(stock, str) and re.fullmatch(r"[0-9]+", stock.strip()):
                try:
                    stock = int(stock.strip())
                except ValueError:
                    raise ValidationError("product.stock integer is too large") from None
            product["stock"] = integer(stock, "product.stock")
            tags = product["tags"]
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split("|") if tag.strip()]
            strings(tags, "product.tags")
            product["tags"] = sorted({tag.strip().casefold() for tag in tags})
            product["source"] = {"document_id": doc["id"].strip(), "row": number}
            catalog.append(product)
    result["catalog"] = sorted(catalog, key=lambda p: p["id"])
    result["document_report"] = {"documents": len(state["request"]["documents"]),
                                 "rows": len(catalog), "products": len(catalog)}
    result["stage"] = "documents"
    return validate(result, "documents")


def ranked(candidates, limit):
    candidates.sort(key=lambda row: (-row["score"], row["product_id"]))
    return [dict(row, rank=rank) for rank, row in enumerate(candidates[:limit], 1)]


def recommend_stage(state):
    validate(state, "documents")
    result = copy.deepcopy(state)
    customer = state["request"]["customer"]
    categories = {v.strip().casefold() for v in customer.get("liked_categories", [])}
    tags = {v.strip().casefold() for v in customer.get("liked_tags", [])}
    purchased = {v.strip() for v in customer.get("purchased_ids", [])}
    budget = money(customer["budget"], "budget") if "budget" in customer else math.inf
    candidates = []
    for product in state["catalog"]:
        if product["stock"] == 0 or product["id"] in purchased or product["price"] > budget:
            continue
        reasons = ["available"]
        score = 1
        if product["category"].casefold() in categories:
            score += 5
            reasons.append("preferred category")
        matched = sorted(tags.intersection(product["tags"]))
        score += 2 * len(matched)
        reasons.extend(f"preferred tag: {tag}" for tag in matched)
        if budget != math.inf:
            reasons.append("within budget")
        candidates.append({"product_id": product["id"], "score": score, "reasons": reasons})
    result["recommendations"] = ranked(candidates, customer.get("limit", 3))
    result["stage"] = "recommend"
    return validate(result, "recommend")


def tokens(value):
    words = re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
    return sorted({SYNONYMS.get(word, word) for word in words if word not in STOP_WORDS})


def mentioned(value, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", value.casefold()))


def support_stage(state):
    validate(state, "recommend")
    result = copy.deepcopy(state)
    request = state["request"]
    question = request["support"]["question"].strip()
    query_words = set(tokens(question))
    rec_ids = [r["product_id"] for r in state["recommendations"]]
    catalog = {p["id"]: p for p in state["catalog"]}
    explicit_ids = [p["id"] for p in state["catalog"]
                    if mentioned(question, p["id"]) or mentioned(question, p["name"])]
    product_ids = explicit_ids or rec_ids[:1]
    evidence = []
    answers = []
    intents = []
    for intent, terms in (("shipping", {"shipping", "delivery", "ship"}),
                          ("returns", {"return", "returns", "refund"})):
        if query_words & terms:
            intents.append(intent)
            if intent in request["policies"]:
                value = request["policies"][intent]
                answers.append(value)
                evidence.append({"kind": "policy", "id": intent, "field": "text", "value": value})
            else:
                answers.append(f"No {intent} policy is available; contact support for confirmation.")
    product_question = bool(query_words & {"price", "cost", "stock", "available", "recommend",
                                          "recommendation", "suggest", "buy"})
    if product_question:
        intents.append("product")
        for pid in product_ids:
            product = catalog[pid]
            answers.append(f"{product['name']} ({pid}): price {product['price']:.2f}; "
                           f"stock {product['stock']}.")
            for field in ("name", "price", "stock"):
                evidence.append({"kind": "product", "id": pid, "field": field,
                                 "value": product[field]})
        if not product_ids:
            answers.append("No eligible product is available for a recommendation.")
    unknown = not intents
    if unknown:
        answers.append("I cannot verify an answer from this catalog and these policies. "
                       "Please contact the support team.")
    missing_policy = any(intent not in request["policies"] for intent in intents if intent != "product")
    suggested_query = catalog[product_ids[0]]["category"] if product_ids else ""
    result["support"] = {
        "question": question, "answer": " ".join(answers),
        "intent": "+".join(intents) if intents else "unknown",
        "escalate": unknown or missing_policy or (product_question and not product_ids),
        "evidence": evidence, "product_ids": product_ids, "suggested_query": suggested_query,
        "recommendation_ids": rec_ids,
    }
    result["stage"] = "support"
    return validate(result, "support")


def one_edit(left, right):
    """Bounded typo matching, including a single adjacent transposition."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = [i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
        return len(differences) == 1 or (
            len(differences) == 2 and differences[1] == differences[0] + 1
            and left[differences[0]] == right[differences[1]]
            and left[differences[1]] == right[differences[0]])
    short, long = (left, right) if len(left) < len(right) else (right, left)
    for i in range(len(long)):
        if long[:i] + long[i + 1:] == short:
            return True
    return False


def search_stage(state):
    validate(state, "support")
    result = copy.deepcopy(state)
    options = state["request"]["search"]
    query = options.get("query", "").strip()
    source = "request" if query else "support"
    query = query or state["support"]["suggested_query"]
    terms = tokens(query)
    rec_ids = state["support"]["recommendation_ids"]
    max_price = money(options["max_price"], "max_price") if "max_price" in options else math.inf
    candidates = []
    for product in state["catalog"]:
        if options.get("in_stock_only", True) and not product["stock"]:
            continue
        if product["price"] > max_price:
            continue
        fields = [(tokens(product["id"] + " " + product["name"]), 5),
                  (tokens(product["category"] + " " + " ".join(product["tags"])), 3),
                  (tokens(product["description"]), 1)]
        score = 0
        reasons = []
        for term in terms:
            exact = max((weight for words, weight in fields if term in words), default=0)
            if exact:
                score += exact
                reasons.append(f"matched: {term}")
            elif len(term) >= 5 and any(one_edit(term, word)
                                       for words, _ in fields for word in words if len(word) >= 5):
                score += 1
                reasons.append(f"typo match: {term}")
        # A personalized boost never introduces a lexically unrelated result.
        if terms and not score:
            continue
        if not terms:
            # A nonblank stop-word-only query is not an intentional browse request.
            if query:
                continue
            reasons.append("browse available catalog")
        if product["id"] in rec_ids:
            score += 0.5
            reasons.append("recommended for customer")
        candidates.append({"product_id": product["id"], "score": score, "reasons": reasons})
    result["search"] = {
        "query": query, "query_source": source, "normalized_terms": terms,
        "results": ranked(candidates, options.get("limit", 5)),
        "support_intent": state["support"]["intent"], "recommendation_ids": list(rec_ids),
    }
    result["stage"] = "search"
    return validate(result, "search")


def run_pipeline(data):
    state = start(data)
    for stage in (documents_stage, recommend_stage, support_stage, search_stage):
        state = stage(state)
    return state


def reject_constant(value):
    raise ValidationError(f"non-finite JSON number: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
    except (ValueError, OSError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
