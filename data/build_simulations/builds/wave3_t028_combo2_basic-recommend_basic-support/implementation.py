"""Deterministic synthetic discovery -> grounded support reference pipeline."""
import json
import math
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


class Validator:
    """The same validation primitives protect input and both stage boundaries."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def obj(cls, value, fields, path):
        cls.require(isinstance(value, dict), f"{path} must be an object")
        cls.require(set(value) == set(fields), f"{path} must have fields: {', '.join(fields)}")

    @classmethod
    def text(cls, value, path):
        cls.require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonempty text")

    @classmethod
    def number(cls, value, path, integer=False):
        cls.require(
            type(value) in ((int,) if integer else (int, float))
            and math.isfinite(value) and value >= 0,
            f"{path} must be a finite nonnegative {'integer' if integer else 'number'}",
        )

    @classmethod
    def texts(cls, value, path):
        cls.require(isinstance(value, list), f"{path} must be a list")
        for item in value:
            cls.text(item, path)
        cls.require(len(value) == len(set(value)), f"{path} must be unique")


V = Validator


def validate_input(data):
    V.obj(data, ("schema_version", "fixture_label", "customer", "catalog", "policies", "support_request", "limit"), "input")
    V.require(type(data["schema_version"]) is int and data["schema_version"] == 1, "schema_version must be 1")
    V.require(data["fixture_label"] == "synthetic", "fixture_label must be synthetic")
    customer = data["customer"]
    V.obj(customer, ("preferred_categories", "interests", "budget", "purchased_ids"), "customer")
    for field in ("preferred_categories", "interests", "purchased_ids"):
        V.texts(customer[field], f"customer.{field}")
    if customer["budget"] is not None:
        V.number(customer["budget"], "customer.budget")
    V.number(data["limit"], "limit", integer=True)
    V.require(1 <= data["limit"] <= 20, "limit must be between 1 and 20")
    V.require(isinstance(data["catalog"], list), "catalog must be a list")
    ids = []
    for product in data["catalog"]:
        V.obj(product, ("id", "name", "category", "tags", "price", "currency", "stock", "description"), "product")
        for field in ("id", "name", "category", "description"):
            V.text(product[field], f"product.{field}")
        V.texts(product["tags"], "product.tags")
        V.number(product["price"], "product.price")
        V.number(product["stock"], "product.stock", integer=True)
        V.require(product["currency"] == "USD", "only USD is supported")
        ids.append(product["id"])
    V.require(len(ids) == len(set(ids)), "catalog IDs must be unique")
    # Purchase history may contain items no longer in the current catalog.
    policies = data["policies"]
    V.obj(policies, ("shipping", "returns", "contact"), "policies")
    for field in policies:
        V.text(policies[field], f"policies.{field}")
    request = data["support_request"]
    V.obj(request, ("question", "product_id"), "support_request")
    V.text(request["question"], "support_request.question")
    if request["product_id"] is not None:
        V.text(request["product_id"], "support_request.product_id")
    return data


def _rank(data):
    customer = data["customer"]
    ranked = []
    for product in data["catalog"]:
        if product["stock"] == 0 or product["id"] in customer["purchased_ids"]:
            continue
        if customer["budget"] is not None and product["price"] > customer["budget"]:
            continue
        reasons = []
        score = 0
        if product["category"] in customer["preferred_categories"]:
            score += 4
            reasons.append("preferred_category")
        matching = sorted(set(product["tags"]) & set(customer["interests"]))
        score += 2 * len(matching)
        reasons.extend("interest:" + tag for tag in matching)
        if not reasons:
            reasons.append("eligible_catalog_fallback")
        ranked.append({
            "product_id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "currency": product["currency"],
            "score": score,
            "reasons": reasons,
        })
    ranked.sort(key=lambda row: (-row["score"], row["price"], row["product_id"]))
    return ranked[:data["limit"]]


def validate_discovery(data, discovery):
    V.obj(discovery, ("items", "empty_reason"), "discovery")
    V.require(isinstance(discovery["items"], list), "discovery.items must be a list")
    for row in discovery["items"]:
        V.obj(row, ("product_id", "name", "price", "currency", "score", "reasons"), "recommendation")
        for field in ("product_id", "name", "currency"):
            V.text(row[field], f"recommendation.{field}")
        V.number(row["price"], "recommendation.price")
        V.number(row["score"], "recommendation.score", integer=True)
        V.texts(row["reasons"], "recommendation.reasons")
    V.require(discovery["items"] == _rank(data), "discovery does not match validated catalog and ranking")
    expected_reason = None if discovery["items"] else "no_eligible_products"
    V.require(discovery["empty_reason"] == expected_reason, "invalid discovery.empty_reason")
    return discovery


def recommend(data):
    validate_input(data)
    items = _rank(data)
    return validate_discovery(data, {
        "items": items,
        "empty_reason": None if items else "no_eligible_products",
    })


def _support_facts(data, discovery):
    request = data["support_request"]
    question = request["question"].casefold()
    requested_id = request["product_id"]
    items = discovery["items"]
    selected = next((row for row in items if row["product_id"] == requested_id), None) if requested_id else (items[0] if items else None)
    facts = []
    unresolved = False
    if requested_id is not None and selected is None:
        unresolved = True
    else:
        # Deliberately bounded topic matching: the whole question must consist
        # of supported topics, so a mixed warranty/price question cannot vanish.
        import re
        words = set(re.findall(r"[a-z]+", question))
        filler = {"what", "is", "the", "a", "an", "are", "your", "it", "its", "and", "or",
                  "can", "i", "you", "me", "tell", "about", "please", "this", "product",
                  "how", "much", "does", "cost", "why", "did", "recommend", "recommended",
                  "recommendation", "price", "stock", "available", "availability", "description",
                  "details", "shipping", "delivery", "return", "returns", "policy", "policies"}
        topics = []
        if words & {"shipping", "delivery"}:
            topics.append("shipping")
        if words & {"return", "returns"}:
            topics.append("returns")
        if words & {"price", "cost", "much"}:
            topics.append("price")
        if words & {"stock", "available", "availability"}:
            topics.append("stock")
        if words & {"description", "details"}:
            topics.append("description")
        if words & {"why", "recommend", "recommended", "recommendation"}:
            topics.append("recommendation")
        unresolved = bool(words - filler) or not topics
        catalog = {p["id"]: p for p in data["catalog"]}
        for topic in topics:
            if topic in ("shipping", "returns"):
                facts.append({"source": f"policies.{topic}", "text": data["policies"][topic]})
            elif selected is None:
                unresolved = True
            else:
                product = catalog[selected["product_id"]]
                source = f"catalog[{product['id']}].{topic}"
                if topic == "price":
                    text = f"{selected['name']}: {selected['price']:.2f} {selected['currency']}."
                elif topic == "stock":
                    text = f"{selected['name']}: {product['stock']} units in stock in this snapshot."
                elif topic == "description":
                    text = product["description"]
                else:
                    source = f"discovery[{product['id']}].reasons"
                    text = f"Recommended {selected['name']} because: {', '.join(selected['reasons'])}."
                facts.append({"source": source, "text": text})
    if unresolved:
        facts.append({"source": "policies.contact", "text": data["policies"]["contact"]})
    return selected["product_id"] if selected else None, facts, unresolved


def validate_support(data, discovery, support):
    V.obj(support, ("product_id", "facts", "needs_human", "answer"), "support")
    V.require(type(support["needs_human"]) is bool, "support.needs_human must be boolean")
    V.require(isinstance(support["facts"], list), "support.facts must be a list")
    for fact in support["facts"]:
        V.obj(fact, ("source", "text"), "fact")
        V.text(fact["source"], "fact.source")
        V.text(fact["text"], "fact.text")
    selected, facts, unresolved = _support_facts(data, discovery)
    V.require(support["product_id"] == selected and support["facts"] == facts and support["needs_human"] == unresolved,
              "support must match grounded evidence")
    prefix = "I cannot fully answer this from the supplied information. " if unresolved else ""
    V.require(support["answer"] == prefix + " ".join(f["text"] for f in facts), "support answer must use grounded facts")
    return support


def support_customer(data, discovery):
    validate_input(data)
    validate_discovery(data, discovery)
    selected, facts, unresolved = _support_facts(data, discovery)
    prefix = "I cannot fully answer this from the supplied information. " if unresolved else ""
    return validate_support(data, discovery, {
        "product_id": selected,
        "facts": facts,
        "needs_human": unresolved,
        "answer": prefix + " ".join(f["text"] for f in facts),
    })


def run_pipeline(data):
    validate_input(data)
    discovery = recommend(data)
    support = support_customer(data, discovery)
    return {"status": "ok", "schema_version": 1, "fixture_label": data["fixture_label"],
            "discovery": discovery, "support": support}


def _reject_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        V.require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
