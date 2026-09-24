"""Synthetic retail FAQ reference. Standard library only; no provider calls.

Run: python -B implementation.py example_input.json
Prices are integer minor units. Consent is a demonstrative input assertion,
not a legal consent-management or compliance implementation.
"""

import json
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


class Validator:
    """Shared boundary validation for the request and grounded response."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def fields(cls, obj, names, label):
        cls.require(isinstance(obj, dict), label + " must be an object")
        cls.require(set(obj) == set(names.split()), label + " has missing or unknown fields")

    @classmethod
    def text(cls, value, label):
        cls.require(isinstance(value, str) and bool(value.strip()) and len(value) <= 300,
                    label + " must be nonempty text of at most 300 characters")

    @classmethod
    def number(cls, value, label, minimum=0):
        cls.require(type(value) is int and minimum <= value <= 100000000,
                    label + " must be a bounded integer")

    @classmethod
    def validate_input(cls, data):
        cls.fields(data, "schema_version synthetic catalog customer basket clickstream knowledge_base request", "input")
        cls.require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                    "schema_version must be 1")
        cls.require(data["synthetic"] is True, "fixtures must be explicitly synthetic")
        catalog = data["catalog"]
        cls.fields(catalog, "currency products", "catalog")
        cls.require(catalog["currency"] in ("USD", "EUR", "GBP"), "unsupported currency")
        cls.require(isinstance(catalog["products"], list), "products must be a list")
        products = {}
        for product in catalog["products"]:
            cls.fields(product, "sku name price_minor stock", "product")
            for field in ("sku", "name"):
                cls.text(product[field], field)
            cls.require(bool(re.fullmatch(r"SYN-[A-Z0-9-]+", product["sku"])), "SKU must use synthetic SYN- prefix")
            cls.require(product["sku"] not in products, "duplicate SKU")
            cls.number(product["price_minor"], "price_minor")
            cls.number(product["stock"], "stock")
            products[product["sku"]] = product
        customer = data["customer"]
        cls.fields(customer, "customer_id consent_personalization preferred_sku", "customer")
        cls.text(customer["customer_id"], "customer_id")
        cls.require(type(customer["consent_personalization"]) is bool, "consent must be boolean")
        cls.require(customer["preferred_sku"] is None or
                    isinstance(customer["preferred_sku"], str) and customer["preferred_sku"] in products,
                    "unknown preferred SKU")
        basket = data["basket"]
        cls.fields(basket, "basket_id items", "basket")
        cls.text(basket["basket_id"], "basket_id")
        cls.require(isinstance(basket["items"], list), "basket items must be a list")
        seen = set()
        for item in basket["items"]:
            cls.fields(item, "sku quantity", "basket item")
            cls.require(isinstance(item["sku"], str) and item["sku"] in products, "unknown basket SKU")
            cls.require(item["sku"] not in seen, "duplicate basket SKU")
            seen.add(item["sku"])
            cls.number(item["quantity"], "quantity", 1)
        cls.require(isinstance(data["clickstream"], list), "clickstream must be a list")
        for event in data["clickstream"]:
            cls.fields(event, "timestamp customer_id event sku", "clickstream event")
            cls.require(event["customer_id"] == customer["customer_id"], "clickstream customer mismatch")
            cls.require(event["event"] in ("view", "add_to_basket"), "unsupported clickstream event")
            cls.require(isinstance(event["sku"], str) and event["sku"] in products, "unknown event SKU")
            cls.text(event["timestamp"], "timestamp")
            try:
                stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                cls.require(stamp.tzinfo is not None, "timestamp must include timezone")
            except ValueError as exc:
                raise ValidationError("invalid clickstream timestamp") from exc
        cls.require(isinstance(data["knowledge_base"], list), "knowledge_base must be a list")
        ids, scopes = set(), set()
        for article in data["knowledge_base"]:
            cls.fields(article, "id topic sku days", "knowledge article")
            cls.text(article["id"], "article id")
            cls.require(article["id"] not in ids, "duplicate article id")
            ids.add(article["id"])
            cls.require(article["topic"] in ("returns", "shipping"), "unsupported knowledge topic")
            cls.require(article["sku"] is None or
                        isinstance(article["sku"], str) and article["sku"] in products, "unknown article SKU")
            scope = (article["topic"], article["sku"])
            cls.require(scope not in scopes, "conflicting knowledge scope")
            scopes.add(scope)
            cls.number(article["days"], "days", 1)
        req = data["request"]
        cls.fields(req, "question sku personalize", "request")
        cls.text(req["question"], "question")
        cls.require(req["sku"] is None or isinstance(req["sku"], str) and req["sku"] in products,
                    "unknown requested SKU")
        cls.require(type(req["personalize"]) is bool, "personalize must be boolean")
        return products

    @classmethod
    def validate_output(cls, result, data, products):
        cls.fields(result, "schema_version synthetic status answer reason citations product customer basket", "output")
        cls.require(result["schema_version"] == 1 and result["synthetic"] is True, "invalid output metadata")
        cls.require(result["status"] in ("answered", "abstained"), "invalid response status")
        cls.require(result["basket"] == data["basket"], "basket changed")
        cls.fields(result["customer"], "customer_id personalized", "output customer")
        cls.require(result["customer"]["customer_id"] == data["customer"]["customer_id"], "customer changed")
        cls.require(type(result["customer"]["personalized"]) is bool, "invalid personalization flag")
        cls.require(not result["customer"]["personalized"] or
                    data["request"]["personalize"] and data["customer"]["consent_personalization"],
                    "personalization requires consent")
        product = result["product"]
        if product is not None:
            cls.fields(product, "sku name price_minor stock currency", "output product")
            cls.require(isinstance(product["sku"], str) and product["sku"] in products, "output SKU unknown")
            cls.require(product == dict(products[product["sku"]], currency=data["catalog"]["currency"]),
                        "displayed product must match catalog")
        if result["status"] == "abstained":
            cls.require(result["answer"] is None and result["citations"] == [] and
                        result["reason"] in ("consent_required", "unsupported_question",
                                             "missing_product", "missing_knowledge"),
                        "invalid abstention")
            return
        cls.require(result["reason"] is None, "answered result must not have abstention reason")
        topic = retrieve_topic(data["request"]["question"])
        allowed = []
        if product is not None:
            if topic == "price":
                allowed.append((price_answer(product), ["catalog:" + product["sku"]]))
            if topic == "stock":
                allowed.append((stock_answer(product), ["catalog:" + product["sku"]]))
        sku = product["sku"] if product else None
        article = retrieve_article(data["knowledge_base"], topic, sku)
        if article:
            allowed.append((policy_answer(article), ["kb:" + article["id"]]))
        cls.require((result["answer"], result["citations"]) in allowed,
                    "answer must exactly match grounded templates; reviews and endorsements are unsupported")


PATTERNS = {
    "price": r"(?:price|what is the price|how much does it cost)",
    "stock": r"(?:stock|is it in stock|how many are in stock)",
    "returns": r"(?:returns|what is the return window|how many days do i have to return it)",
    "shipping": r"(?:shipping|how long does shipping take|what is the shipping time)",
}


def retrieve_topic(question):
    """Bounded full-question matching deliberately rejects unsupported clauses."""
    normalized = " ".join(question.lower().strip().rstrip("?.!").split())
    for topic, pattern in PATTERNS.items():
        if re.fullmatch(pattern, normalized):
            return topic
    return None


def retrieve_article(articles, topic, sku):
    eligible = [a for a in articles if a["topic"] == topic and a["sku"] in (None, sku)]
    eligible.sort(key=lambda a: (a["sku"] is None, a["id"]))
    return eligible[0] if eligible else None


def price_answer(product):
    whole, fraction = divmod(product["price_minor"], 100)
    return f'{product["sku"]} costs {product["currency"]} {whole}.{fraction:02d}.'


def stock_answer(product):
    return f'{product["sku"]} has {product["stock"]} units in stock.'


def policy_answer(article):
    if article["topic"] == "returns":
        return f'The return window is {article["days"]} days.'
    return f'The shipping time is {article["days"]} days.'


def answer_request(data):
    products = Validator.validate_input(data)
    req = data["request"]
    result = {
        "schema_version": 1, "synthetic": True, "status": "abstained",
        "answer": None, "reason": None, "citations": [], "product": None,
        "customer": {"customer_id": data["customer"]["customer_id"], "personalized": False},
        "basket": data["basket"],
    }

    def finish(reason=None):
        result["reason"] = reason
        Validator.validate_output(result, data, products)
        return result

    if req["personalize"] and not data["customer"]["consent_personalization"]:
        return finish("consent_required")
    topic = retrieve_topic(req["question"])
    if topic is None:
        return finish("unsupported_question")
    sku = req["sku"]
    if req["personalize"]:
        result["customer"]["personalized"] = True
        if sku is None:
            sku = data["customer"]["preferred_sku"]
            if sku is None and data["clickstream"]:
                latest = max(data["clickstream"], key=lambda e: datetime.fromisoformat(
                    e["timestamp"].replace("Z", "+00:00")))
                sku = latest["sku"]
    if sku:
        result["product"] = dict(products[sku], currency=data["catalog"]["currency"])
    if topic in ("price", "stock"):
        if sku is None:
            return finish("missing_product")
        result["answer"] = price_answer(result["product"]) if topic == "price" else stock_answer(result["product"])
        result["citations"] = ["catalog:" + sku]
    else:
        article = retrieve_article(data["knowledge_base"], topic, sku)
        if article is None:
            return finish("missing_knowledge")
        result["answer"] = policy_answer(article)
        result["citations"] = ["kb:" + article["id"]]
    result["status"] = "answered"
    return finish()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValidationError("duplicate JSON key: " + key)
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = answer_request(data)
        exit_code = 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        result = {"schema_version": 1, "status": "error", "error": str(exc)}
        exit_code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
