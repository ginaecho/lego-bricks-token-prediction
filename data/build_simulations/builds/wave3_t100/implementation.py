"""Synthetic, deterministic onboarding -> support -> discovery -> documents CLI.

Run: python -B implementation.py example_input.json
Money is integer cents. No provider, network, persistence, or order submission.
"""

import copy
import json
import re
import sys


STAGES = ("onboarding", "support", "recommendations", "documents")
ROOT_KEYS = {
    "schema_version", "fixture_label", "status", "customer", "question",
    "knowledge", "catalog", "document_text", "stages",
}
STOP_WORDS = {"a", "an", "the", "is", "are", "i", "my", "for", "to", "and",
              "of", "can", "do", "how", "with", "what", "me", "please"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path, allow_empty=False, limit=4000):
    require(isinstance(value, str), path + " must be a string")
    require(len(value) <= limit, path + " is too long")
    require(allow_empty or bool(value.strip()), path + " must not be blank")


def integer(value, path, minimum=0, maximum=1_000_000_000):
    require(type(value) is int and minimum <= value <= maximum,
            path + " must be an integer in range")


def texts(value, path):
    require(isinstance(value, list) and len(value) <= 100,
            path + " must be a list with at most 100 entries")
    for item in value:
        text(item, path, limit=100)
    require(len({item.casefold().strip() for item in value}) == len(value),
            path + " contains duplicate entries")


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold())) - STOP_WORDS


def parse_document(source):
    """Extract a bounded key/value document; reject ambiguous/unknown fields."""
    text(source, "document_text")
    fields = {}
    for line in source.splitlines():
        if not line.strip():
            continue
        require(":" in line, "document_text lines must use Field: value")
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.casefold()
        require(key in {"reference", "quantity", "notes"},
                "document_text contains an unknown field")
        require(key not in fields, "document_text contains a duplicate field")
        fields[key] = value
    require({"reference", "quantity"} <= set(fields),
            "document_text requires Reference and Quantity")
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", fields["reference"]) is not None,
            "document reference must be a safe identifier")
    require(re.fullmatch(r"[0-9]{1,4}", fields["quantity"]) is not None,
            "document quantity must contain decimal digits")
    quantity = int(fields["quantity"])
    integer(quantity, "document quantity", 1, 1000)
    return {"reference": fields["reference"], "quantity": quantity,
            "notes": fields.get("notes", "")}


def onboarding_result(state):
    customer = state["customer"]
    if not customer["email"].strip():
        code, instruction = "add_contact", "Add an email so the team can follow up."
    elif not customer["goals"]:
        code, instruction = "choose_goal", "Choose your first goal to personalize discovery."
    else:
        code = "explore_goal"
        instruction = "Explore products for " + customer["goals"][0] + "."
    return {
        "customer_id": customer["id"],
        "greeting": "Welcome, " + customer["name"].strip() + "!",
        "goals": list(customer["goals"]),
        "next_step": {"code": code, "instruction": instruction},
    }


def support_result(state):
    onboarding = state["stages"]["onboarding"]
    query = tokens(state["question"])
    matches = []
    for article in state["knowledge"]:
        score = len(query & tokens(" ".join(article["keywords"])))
        if score:
            matches.append((score, article["id"], article))
    matches.sort(key=lambda item: (-item[0], item[1]))
    article = matches[0][2] if matches else None
    grounded = article["answer"] if article else (
        "The provided knowledge base does not answer this question. "
        "A human review is needed; no ticket has been submitted."
    )
    return {
        "customer_id": onboarding["customer_id"],
        "onboarding_next_step": copy.deepcopy(onboarding["next_step"]),
        "answer": state["customer"]["name"].strip() + ": " + grounded,
        "resolution": "knowledge_match" if article else "human_review_needed",
        "source_ids": [article["id"]] if article else [],
        "topic_tags": list(article["tags"]) if article else [],
        "contact_available": bool(state["customer"]["email"].strip()),
    }


def recommendation_result(state):
    support = state["stages"]["support"]
    customer = state["customer"]
    interests = tokens(" ".join(customer["interests"]))
    goals = tokens(" ".join(state["stages"]["onboarding"]["goals"]))
    topics = tokens(" ".join(support["topic_tags"]))
    ranked = []
    for product in state["catalog"]:
        if not product["active"] or not product["stock"]:
            continue
        if product["price_cents"] > customer["budget_cents"]:
            continue
        tags = tokens(" ".join(product["tags"]))
        matches = {
            "interests": sorted(tags & interests),
            "goals": sorted(tags & goals),
            "support_topics": sorted(tags & topics),
        }
        score = (3 * len(matches["interests"]) + 2 * len(matches["goals"])
                 + 4 * len(matches["support_topics"]))
        ranked.append({
            "product_id": product["id"], "name": product["name"],
            "price_cents": product["price_cents"], "score": score,
            "matched_terms": matches,
            "reason": "Matched preferences or support topic." if score else
                      "Available within your per-item budget; no preference match.",
        })
    ranked.sort(key=lambda item: (-item["score"], item["price_cents"], item["product_id"]))
    return {
        "customer_id": support["customer_id"],
        "support_resolution": support["resolution"],
        "support_source_ids": list(support["source_ids"]),
        "items": ranked[:3],
        "empty_reason": None if ranked else "No available products within budget.",
    }


def document_result(state):
    recommendations = state["stages"]["recommendations"]
    extracted = parse_document(state["document_text"])
    selected = recommendations["items"][:1]
    lines = []
    inventory_ok = True
    for recommendation in selected:
        product = next(p for p in state["catalog"]
                       if p["id"] == recommendation["product_id"])
        quantity = extracted["quantity"]
        inventory_ok = product["stock"] >= quantity
        lines.append({
            "product_id": product["id"], "description": product["name"],
            "quantity": quantity, "unit_price_cents": recommendation["price_cents"],
            "line_total_cents": recommendation["price_cents"] * quantity,
        })
    total = sum(line["line_total_cents"] for line in lines)
    checks = {
        "has_recommendation": bool(lines), "inventory_sufficient": inventory_ok,
        "within_total_budget": total <= state["customer"]["budget_cents"],
    }
    return {
        "document_type": "synthetic_product_proposal",
        "reference": extracted["reference"], "notes": extracted["notes"],
        "customer_id": recommendations["customer_id"],
        "support_source_ids": list(recommendations["support_source_ids"]),
        "lines": lines, "currency": "USD", "total_cents": total,
        "checks": checks,
        "review_status": "draft_ready" if all(checks.values()) else "needs_review",
        "rendered_text": (
            "SYNTHETIC PROPOSAL " + extracted["reference"] + "\n"
            + "Customer: " + state["customer"]["name"] + "\n"
            + "\n".join(
                f'{line["quantity"]} x {line["description"]}: '
                f'USD {line["line_total_cents"] // 100}.{line["line_total_cents"] % 100:02d}'
                for line in lines
            )
            + f"\nTotal: USD {total // 100}.{total % 100:02d}"
        ),
    }


BUILDERS = (onboarding_result, support_result, recommendation_result, document_result)


def validate(state, expected_count=None):
    """One boundary validator for input, every handoff, and complete output."""
    shape(state, ROOT_KEYS, "envelope")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "schema_version must be 1")
    require(state["fixture_label"] == "synthetic", "fixture_label must be synthetic")
    customer = state["customer"]
    shape(customer, {"id", "name", "email", "goals", "interests", "budget_cents"},
          "customer")
    for field in ("id", "name"):
        text(customer[field], "customer." + field, limit=200)
    text(customer["email"], "customer.email", allow_empty=True, limit=254)
    require(not customer["email"].strip() or
            re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", customer["email"]) is not None,
            "customer.email is invalid")
    for field in ("goals", "interests"):
        texts(customer[field], "customer." + field)
    integer(customer["budget_cents"], "customer.budget_cents")
    text(state["question"], "question")
    for collection in ("knowledge", "catalog"):
        entries = state[collection]
        require(isinstance(entries, list) and len(entries) <= 1000,
                collection + " must be a list with at most 1000 entries")
        ids = set()
        for entry in entries:
            keys = {"id", "keywords", "answer", "tags"} if collection == "knowledge" else {
                "id", "name", "tags", "price_cents", "stock", "active",
            }
            shape(entry, keys, collection + " entry")
            text(entry["id"], collection + ".id", limit=100)
            require(entry["id"] not in ids, collection + " contains duplicate ids")
            ids.add(entry["id"])
            texts(entry["tags"], collection + ".tags")
            if collection == "knowledge":
                texts(entry["keywords"], "knowledge.keywords")
                text(entry["answer"], "knowledge.answer")
            else:
                text(entry["name"], "catalog.name", limit=200)
                integer(entry["price_cents"], "catalog.price_cents")
                integer(entry["stock"], "catalog.stock", maximum=1_000_000)
                require(type(entry["active"]) is bool, "catalog.active must be boolean")
    parse_document(state["document_text"])
    require(isinstance(state["stages"], dict), "stages must be an object")
    count = len(state["stages"])
    require(count <= 4 and set(state["stages"]) == set(STAGES[:count]),
            "stages must be a contiguous pipeline prefix")
    if expected_count is not None:
        require(count == expected_count, "unexpected pipeline stage count")
    require(state["status"] == ("ok" if count == 4 else "ready"), "invalid status")
    # Recompute bounded deterministic results to verify types and provenance,
    # rejecting tampered handoffs rather than trusting derived fields.
    for name, builder in zip(STAGES[:count], BUILDERS):
        expected = json.dumps(builder(state), sort_keys=True, ensure_ascii=True)
        try:
            actual = json.dumps(state["stages"][name], sort_keys=True, ensure_ascii=True,
                                allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValidationError("stage output is not JSON-compatible") from exc
        require(actual == expected, name + " output failed validation")
    return state


def advance(state, stage):
    require(stage in STAGES, "unknown stage")
    index = STAGES.index(stage)
    validate(state, index)
    result = copy.deepcopy(state)
    result["stages"][stage] = BUILDERS[index](result)
    if index == 3:
        result["status"] = "ok"
    return validate(result, index + 1)


def run_pipeline(state):
    validate(state, 0)
    for stage in STAGES:
        state = advance(state, stage)
    return state


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], "r", encoding="utf-8") as handle:
            raw = handle.read(1_000_001)
        require(len(raw) <= 1_000_000, "input exceeds 1,000,000 characters")
        state = json.loads(raw, object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
        result = run_pipeline(state)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
