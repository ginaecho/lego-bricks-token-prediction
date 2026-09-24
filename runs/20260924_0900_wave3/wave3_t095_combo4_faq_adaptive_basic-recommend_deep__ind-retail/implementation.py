"""Deterministic synthetic retail pipeline; Python standard library only.

Run: python -B implementation.py example_input.json
Prices are integer minor units (cents). Evidence is synthetic technical material,
not customer reviews, endorsements, or independently verified product claims.
"""

import copy
import datetime
import json
import re
import sys
from pathlib import Path


BUILD_ID = "wave3_t095_combo4_faq_adaptive_basic-recommend_deep__ind-retail"
STAGES = ("faq", "adaptive", "basic:recommend", "deep")
ATTRIBUTES = ("material", "durability", "care", "repairability", "fit")
MAX_BYTES = 1_000_000


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing required fields")
    require(set(value) <= set(required) | set(optional), "Unknown fields")


def text(value, label, maximum=4000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            label + " must be a nonempty bounded string")


def integer(value, label, minimum=0, maximum=1_000_000_000):
    require(type(value) is int and minimum <= value <= maximum,
            label + " must be a bounded integer")


def sequence(value, label, maximum=200):
    require(isinstance(value, list) and len(value) <= maximum,
            label + " must be a bounded array")


def unique_strings(value, label, maximum=200):
    sequence(value, label, maximum)
    for item in value:
        text(item, label, 100)
    require(len(set(value)) == len(value), label + " contains duplicates")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def personalization_allowed(request):
    consent = request["customer"]["consent"]
    return consent["gdpr_personalization"] and consent["ccpa_personalization"]


def validate(value, kind="input", request=None, expected_stage=None):
    """The single validation boundary for input and every stage handoff.

    Output contracts are checked against deterministic, grounded derivations,
    including every prior result, rather than trusting provenance labels.
    """
    if kind == "input":
        _validate_input(value)
    elif kind == "handoff":
        require(request is not None, "Handoff validation requires its input")
        _validate_input(request)
        _validate_handoff(value, request, expected_stage)
    else:
        raise ValidationError("Unknown validation kind")
    return value


def _validate_input(request):
    keys(request, ("schema_version", "synthetic", "catalog", "customer", "basket",
                   "clickstream", "question", "knowledge_base", "documents"))
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "Unsupported schema_version")
    require(request["synthetic"] is True, "Only clearly labeled synthetic fixtures are accepted")
    text(request["question"], "question", 1000)
    catalog = request["catalog"]
    keys(catalog, ("currency", "products"))
    require(catalog["currency"] in ("USD", "EUR", "GBP"), "Unsupported currency")
    sequence(catalog["products"], "products")
    products = {}
    for product in catalog["products"]:
        keys(product, ("sku", "name", "category", "price_cents", "stock"))
        for field in ("sku", "name", "category"):
            text(product[field], field, 100)
        require(product["sku"] not in products, "Duplicate catalog SKU")
        integer(product["price_cents"], "price_cents")
        integer(product["stock"], "stock", maximum=1_000_000)
        products[product["sku"]] = product
    customer = request["customer"]
    keys(customer, ("customer_id", "experience", "consent", "preferences"))
    text(customer["customer_id"], "customer_id", 100)
    require(customer["experience"] in ("novice", "intermediate", "expert"),
            "Invalid experience")
    consent = customer["consent"]
    keys(consent, ("gdpr_personalization", "ccpa_personalization"))
    require(all(type(v) is bool for v in consent.values()), "Consent must be explicit booleans")
    preferences = customer["preferences"]
    keys(preferences, ("categories", "max_price_cents"))
    unique_strings(preferences["categories"], "preferred categories", 20)
    if preferences["max_price_cents"] is not None:
        integer(preferences["max_price_cents"], "max_price_cents")
    basket = request["basket"]
    keys(basket, ("order_id", "customer_id", "items"))
    text(basket["order_id"], "order_id", 100)
    require(basket["customer_id"] == customer["customer_id"], "Basket customer mismatch")
    sequence(basket["items"], "basket items")
    seen = set()
    for item in basket["items"]:
        keys(item, ("sku", "quantity"))
        text(item["sku"], "basket SKU", 100)
        require(item["sku"] in products and item["sku"] not in seen,
                "Unknown or duplicate basket SKU")
        seen.add(item["sku"])
        integer(item["quantity"], "quantity", minimum=1, maximum=1_000_000)
        require(item["quantity"] <= products[item["sku"]]["stock"],
                "Basket quantity exceeds catalog stock")
    sequence(request["clickstream"], "clickstream", 1000)
    event_ids = set()
    for event in request["clickstream"]:
        keys(event, ("event_id", "customer_id", "sku", "event", "timestamp"))
        text(event["event_id"], "event_id", 100)
        require(event["event_id"] not in event_ids, "Duplicate clickstream event")
        event_ids.add(event["event_id"])
        require(event["customer_id"] == customer["customer_id"], "Clickstream customer mismatch")
        text(event["sku"], "event SKU", 100)
        require(event["sku"] in products, "Unknown clickstream SKU")
        require(event["event"] in ("view", "add_to_basket", "purchase"), "Unknown event type")
        text(event["timestamp"], "timestamp", 100)
        try:
            stamp = datetime.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            require(stamp.utcoffset() is not None, "Event timestamp needs a timezone")
        except ValueError as exc:
            raise ValidationError("Invalid event timestamp") from exc
    sequence(request["knowledge_base"], "knowledge_base")
    kb_ids = set()
    for article in request["knowledge_base"]:
        keys(article, ("id", "topic", "keywords", "answer", "synthetic"))
        require(article["synthetic"] is True, "KB article must be labeled synthetic")
        text(article["id"], "article id", 100)
        require(article["id"] not in kb_ids, "Duplicate article id")
        kb_ids.add(article["id"])
        text(article["topic"], "topic", 100)
        unique_strings(article["keywords"], "keywords", 30)
        require(bool(article["keywords"]), "Article needs retrieval keywords")
        text(article["answer"], "article answer")
    sequence(request["documents"], "documents")
    doc_ids = set()
    for document in request["documents"]:
        keys(document, ("id", "title", "kind", "synthetic", "text", "claims"))
        require(document["synthetic"] is True, "Evidence must be labeled synthetic")
        text(document["id"], "document id", 100)
        require(document["id"] not in doc_ids, "Duplicate document id")
        doc_ids.add(document["id"])
        text(document["title"], "document title", 200)
        require(document["kind"] in ("specification", "technical_note", "synthetic_test"),
                "Only technical evidence is allowed; no reviews or endorsements")
        text(document["text"], "document text", 20000)
        sequence(document["claims"], "claims", 100)
        claim_ids = set()
        for claim in document["claims"]:
            keys(claim, ("sku", "attribute", "value", "quote"))
            text(claim["sku"], "claim SKU", 100)
            require(claim["sku"] in products, "Unknown evidence SKU")
            require(claim["attribute"] in ATTRIBUTES, "Unsupported technical attribute")
            text(claim["value"], "claim value", 200)
            text(claim["quote"], "evidence quote", 2000)
            require(claim["quote"] in document["text"], "Quote not present in source")
            require(claim["value"] in claim["quote"], "Claim value not present in quote")
            fact = f'{claim["sku"]} {claim["attribute"]}: {claim["value"]}.'
            require(fact in claim["quote"], "Quote must bind the stated SKU, attribute and value")
            marker = (claim["sku"], claim["attribute"], claim["value"])
            require(marker not in claim_ids, "Duplicate claim in document")
            claim_ids.add(marker)


def _seed(request):
    products = {p["sku"]: p for p in request["catalog"]["products"]}
    items = []
    for item in request["basket"]["items"]:
        product = products[item["sku"]]
        items.append({**item, "unit_price_cents": product["price_cents"],
                      "line_total_cents": item["quantity"] * product["price_cents"],
                      "stock": product["stock"]})
    return {
        "schema_version": 1,
        "status": "ok",
        "synthetic": True,
        "customer_id": request["customer"]["customer_id"],
        "personalization_mode": "consented" if personalization_allowed(request) else "generic",
        "basket": {
            "order_id": request["basket"]["order_id"],
            "currency": request["catalog"]["currency"],
            "items": items,
            "total_cents": sum(item["line_total_cents"] for item in items),
        },
        "completed_stages": [],
        "results": {},
    }


def _faq(request, results):
    query = tokens(request["question"])
    candidates = []
    for article in request["knowledge_base"]:
        keyword_tokens = set().union(*(tokens(word) for word in article["keywords"]))
        score = len(query & keyword_tokens)
        if score:
            candidates.append((score, article["id"], article))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return {
            "answer_status": "abstained", "answer": None, "topic": None,
            "citations": [], "question": request["question"],
            "reason": "No unique supported knowledge-base match; contact support.",
        }
    article = candidates[0][2]
    return {
        "answer_status": "answered", "answer": article["answer"], "topic": article["topic"],
        "citations": [{"source_id": article["id"], "quote": article["answer"]}],
        "question": request["question"], "reason": "Extractive synthetic KB answer.",
    }


def _adaptive(request, results):
    faq = results["faq"]
    allowed = personalization_allowed(request)
    customer = request["customer"]
    experience = customer["experience"] if allowed else "unspecified"
    categories = list(customer["preferences"]["categories"]) if allowed else []
    budget = customer["preferences"]["max_price_cents"] if allowed else None
    budget_explanation = (
        f"Confirm your consented budget of {budget} cents before comparing products."
        if budget is not None else "Set a budget before comparing products."
    )
    comparison_explanation = "Compare catalog facts and flag uncertain technical evidence."
    if categories:
        comparison_explanation += " Start with your preferred categories: " + ", ".join(categories) + "."
    support_step = "policy_review" if faq["answer_status"] == "answered" else "support_followup"
    definitions = [
        ("catalog_basics", [], "Learn how catalog price, currency and stock are displayed."),
        ("choose_budget", ["catalog_basics"], budget_explanation),
        (support_step, ["catalog_basics"],
         "Review the grounded answer before selection." if support_step == "policy_review"
         else "Ask support the unanswered question; no policy answer was assumed."),
        ("compare_products", ["choose_budget", support_step],
         comparison_explanation),
        ("review_basket", ["compare_products"],
         "Check basket quantities against catalog stock and prices."),
    ]
    satisfied = set()
    steps = []
    for step_id, prerequisites, explanation in definitions:
        if step_id == "catalog_basics" and experience == "expert":
            status = "satisfied_by_experience"
            satisfied.add(step_id)
        else:
            status = "ready" if set(prerequisites) <= satisfied else "locked"
        steps.append({"id": step_id, "prerequisites": prerequisites,
                      "status": status, "explanation": explanation})
    interests = {}
    if allowed:
        weights = {"view": 1, "add_to_basket": 3, "purchase": 2}
        for event in request["clickstream"]:
            interests[event["sku"]] = interests.get(event["sku"], 0) + weights[event["event"]]
    return {
        "personalization_mode": "consented" if allowed else "generic",
        "experience_used": experience,
        "categories": categories,
        "max_price_cents": budget,
        "interest_scores": interests,
        "steps": steps,
        "support_context": {
            "answer_status": faq["answer_status"], "topic": faq["topic"],
            "source_ids": [citation["source_id"] for citation in faq["citations"]],
            "unresolved_question": faq["question"] if faq["answer_status"] == "abstained" else None,
        },
        "research_attributes": ["material", "durability", "care"],
        "explanation": (
            "Experience, preferences and clickstream used with both explicit consents."
            if allowed else
            "Generic guidance: experience, preferences and clickstream were not used."
        ),
    }


def _recommend(request, results):
    adaptive = results["adaptive"]
    in_basket = {item["sku"] for item in request["basket"]["items"]}
    candidates = []
    for product in request["catalog"]["products"]:
        if product["stock"] == 0 or product["sku"] in in_basket:
            continue
        budget = adaptive["max_price_cents"]
        if budget is not None and product["price_cents"] > budget:
            continue
        score = 0
        reasons = ["In stock; price and availability copied from the synthetic catalog."]
        if product["category"] in adaptive["categories"]:
            score += 10
            reasons.append("Matches a consented category preference.")
        interest = adaptive["interest_scores"].get(product["sku"], 0)
        if interest:
            score += min(interest, 9)
            reasons.append("Matches consented clickstream interest.")
        candidates.append((score, product, reasons))
    candidates.sort(key=lambda item: (-item[0], item[1]["price_cents"], item[1]["sku"]))
    cards = [
        {**product, "currency": request["catalog"]["currency"],
         "score": score, "reasons": reasons}
        for score, product, reasons in candidates[:3]
    ]
    return {
        "mode": adaptive["personalization_mode"],
        "products": cards,
        "selection_status": "recommended" if cards else "no_eligible_products",
        "ranking": "Preference and click score, then price ascending, then SKU.",
        "support_context": copy.deepcopy(adaptive["support_context"]),
        "research_attributes": list(adaptive["research_attributes"]),
        "onboarding_next_steps": [s["id"] for s in adaptive["steps"] if s["status"] == "ready"],
        "notice": "No reviews, endorsements or inferred ratings are generated.",
    }


def _deep(request, results):
    recommendation = results["basic:recommend"]
    findings, disagreements, unresolved = [], [], []
    used_documents = set()
    products = recommendation["products"]
    support = recommendation["support_context"]
    if support["unresolved_question"]:
        unresolved.append({"sku": None, "attribute": "support",
                           "question": support["unresolved_question"],
                           "reason": "The knowledge base explicitly abstained."})
    if not products:
        unresolved.append({"sku": None, "attribute": "selection",
                           "question": "Which product should be researched?",
                           "reason": "No eligible in-stock recommendation."})
    for product in products:
        sku = product["sku"]
        for attribute in recommendation["research_attributes"]:
            groups = {}
            for document in request["documents"]:
                for claim in document["claims"]:
                    if claim["sku"] != sku or claim["attribute"] != attribute:
                        continue
                    normalized = " ".join(claim["value"].casefold().split())
                    groups.setdefault(normalized, {"value": claim["value"], "evidence": []})
                    groups[normalized]["evidence"].append({
                        "document_id": document["id"], "quote": claim["quote"],
                        "kind": document["kind"], "synthetic": True,
                    })
                    used_documents.add(document["id"])
            alternatives = [groups[key] for key in sorted(groups)]
            for alternative in alternatives:
                alternative["evidence"].sort(key=lambda item: (item["document_id"], item["quote"]))
            if not alternatives:
                unresolved.append({
                    "sku": sku, "attribute": attribute,
                    "question": "What is the product's " + attribute + "?",
                    "reason": "No supplied technical evidence; no inference made.",
                })
            elif len(alternatives) == 1:
                alternative = alternatives[0]
                findings.append({
                    "sku": sku, "attribute": attribute, **alternative,
                    "support": "multiple_documents"
                    if len({e["document_id"] for e in alternative["evidence"]}) > 1
                    else "single_document",
                })
            else:
                disagreements.append({
                    "sku": sku, "attribute": attribute, "alternatives": alternatives,
                    "resolution": "Unresolved; conflicting evidence is not majority-voted.",
                })
                unresolved.append({
                    "sku": sku, "attribute": attribute,
                    "question": "Which conflicting " + attribute + " claim is correct?",
                    "reason": "Additional authoritative evidence is required.",
                })
    return {
        "researched_products": [
            {key: product[key] for key in ("sku", "name", "price_cents", "stock", "currency")}
            for product in products
        ],
        "findings": findings,
        "disagreements": disagreements,
        "unresolved_questions": unresolved,
        "document_ids": sorted(used_documents),
        "support_sources": list(support["source_ids"]),
        "limitations": [
            "Synthetic technical fixtures only; not reviews or endorsements.",
            "Agreement across documents does not establish independence or truth.",
            "Lexical disagreement detection cannot reconcile semantically equivalent claims.",
            "Catalog snapshot is authoritative for displayed price and stock; no checkout reservation.",
        ],
    }


COMPUTE = {"faq": _faq, "adaptive": _adaptive, "basic:recommend": _recommend, "deep": _deep}


def _validate_handoff(value, request, expected_stage):
    expected = _seed(request)
    keys(value, tuple(expected))
    sequence(value["completed_stages"], "completed_stages", 4)
    count = len(value["completed_stages"])
    require(value["completed_stages"] == list(STAGES[:count]), "Invalid pipeline stage order")
    require(expected_stage in STAGES or expected_stage is None, "Unknown expected stage")
    if expected_stage is not None:
        require(count > 0 and value["completed_stages"][-1] == expected_stage,
                "Unexpected predecessor stage")
    require(isinstance(value["results"], dict), "results must be an object")
    for stage in STAGES[:count]:
        expected["results"][stage] = COMPUTE[stage](request, expected["results"])
        expected["completed_stages"].append(stage)
    # Canonical JSON comparison distinguishes booleans from integers.
    try:
        observed = json.dumps(value, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Non-JSON handoff") from exc
    require(observed == json.dumps(expected, sort_keys=True, allow_nan=False),
            "Handoff differs from validated grounded derivation")


def _advance(request, previous, stage):
    index = STAGES.index(stage)
    validate(previous, "handoff", request, STAGES[index - 1] if index else None)
    require(previous["completed_stages"] == list(STAGES[:index]), "Invalid predecessor")
    result = copy.deepcopy(previous)
    result["results"][stage] = COMPUTE[stage](request, result["results"])
    result["completed_stages"].append(stage)
    return validate(result, "handoff", request, stage)


def faq(request):
    validate(request)
    return _advance(request, _seed(request), "faq")


def adaptive(previous, request):
    return _advance(request, previous, "adaptive")


def recommend(previous, request):
    return _advance(request, previous, "basic:recommend")


def deep(previous, request):
    return _advance(request, previous, "deep")


def run_pipeline(request):
    first = faq(request)
    second = adaptive(first, request)
    third = recommend(second, request)
    return deep(third, request)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError("Nonfinite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with Path(args[0]).open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "Input exceeds the one-megabyte fixture limit")
        request = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object,
                             parse_constant=_invalid_constant)
        result = run_pipeline(request)
        code = 0
    except (ValueError, OSError, RecursionError, TypeError) as exc:
        result = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
