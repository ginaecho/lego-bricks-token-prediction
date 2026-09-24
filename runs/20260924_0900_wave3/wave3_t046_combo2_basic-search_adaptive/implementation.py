"""Deterministic synthetic marketplace search -> adaptive onboarding.

Run: python -B implementation.py example_input.json
All prices are USD. No network, provider, persistence, or third-party packages.
"""

import json
import math
import re
import sys
import unicodedata


class ValidationError(ValueError):
    pass


ALIASES = {
    "trainers": "sneaker", "trainer": "sneaker", "sneakers": "sneaker",
    "shoes": "shoe", "running": "run", "jogging": "run",
    "wireless": "cordless", "headphones": "headphone",
    "earphones": "headphone", "sofa": "couch", "sofas": "couch",
    "laptops": "laptop", "notebook": "laptop", "bikes": "bicycle",
    "bike": "bicycle", "cycling": "bicycle",
}
STOP_WORDS = {"a", "an", "the", "for", "with", "and", "i", "want", "please", "me"}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path, limit=200):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    require(len(value) <= limit, path + " is too long")


def number(value, path):
    require(type(value) in (int, float), path + " must be a number")
    require(0 <= value <= 1_000_000_000 and math.isfinite(value),
            path + " must be finite and between 0 and 1000000000")


def string_list(value, path, limit=50):
    require(isinstance(value, list) and len(value) <= limit, path + " must be a bounded list")
    for entry in value:
        text(entry, path + "[]")
    require(len(value) == len(set(value)), path + " must contain unique values")


def tokens(value, preserve_original=False):
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    result = []
    for word in words:
        if word not in STOP_WORDS:
            result.append(ALIASES.get(word, word))
            if preserve_original:
                result.append(word)
    return list(dict.fromkeys(result))


def validate_product(product):
    obj(product, {"id", "name", "category", "description", "tags", "price_usd",
                  "requires_setup", "safety_review"}, "product")
    for field in ("id", "name", "category"):
        text(product[field], "product." + field)
    text(product["description"], "product.description", 2000)
    string_list(product["tags"], "product.tags")
    number(product["price_usd"], "product.price_usd")
    for field in ("requires_setup", "safety_review"):
        require(type(product[field]) is bool, "product." + field + " must be boolean")


def validate(kind, value, request=None, search=None):
    """One boundary-validation layer for requests and both pipeline stages."""
    if kind == "request":
        obj(value, {"schema_version", "fixture_label", "query", "filters", "profile",
                    "catalog", "limit"}, "request")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        text(value["fixture_label"], "fixture_label")
        text(value["query"], "query", 500)
        require(bool(tokens(value["query"])), "query needs a searchable word")
        obj(value["filters"], {"category", "max_price_usd"}, "filters")
        if value["filters"]["category"] is not None:
            text(value["filters"]["category"], "filters.category")
        if value["filters"]["max_price_usd"] is not None:
            number(value["filters"]["max_price_usd"], "filters.max_price_usd")
        obj(value["profile"], {"experience", "preferred_format"}, "profile")
        require(value["profile"]["experience"] in ("beginner", "intermediate", "advanced"),
                "profile.experience is unsupported")
        require(value["profile"]["preferred_format"] in ("text", "video", "interactive"),
                "profile.preferred_format is unsupported")
        require(type(value["limit"]) is int and 1 <= value["limit"] <= 20,
                "limit must be an integer from 1 to 20")
        require(isinstance(value["catalog"], list) and len(value["catalog"]) <= 1000,
                "catalog must be a list of at most 1000 products")
        ids = []
        for product in value["catalog"]:
            validate_product(product)
            ids.append(product["id"])
        require(len(ids) == len(set(ids)), "catalog product ids must be unique")
    elif kind == "search":
        validate("request", request)
        obj(value, {"query", "normalized_terms", "matches"}, "search")
        require(value["query"] == request["query"], "search query provenance mismatch")
        require(value["normalized_terms"] == tokens(request["query"]),
                "search normalization mismatch")
        require(isinstance(value["matches"], list) and len(value["matches"]) <= request["limit"],
                "search matches exceed limit")
        catalog = {p["id"]: p for p in request["catalog"]}
        ids = []
        for match in value["matches"]:
            obj(match, {"product", "score", "matched_terms", "explanation"}, "match")
            validate_product(match["product"])
            product = match["product"]
            require(product["id"] in catalog and product == catalog[product["id"]],
                    "search product provenance mismatch")
            require(eligible(product, request["filters"]), "search result violates filters")
            require(type(match["score"]) is int and match["score"] > 0, "invalid match score")
            string_list(match["matched_terms"], "matched_terms")
            expected_score, expected_terms = score_product(product, value["normalized_terms"])
            require(match["score"] == expected_score and match["matched_terms"] == expected_terms,
                    "search relevance evidence mismatch")
            text(match["explanation"], "match.explanation", 2000)
            ids.append(product["id"])
        require(len(ids) == len(set(ids)), "duplicate search results")
        require(value["matches"] == sorted(value["matches"],
                key=lambda m: (-m["score"], m["product"]["id"])), "search results not ranked")
    elif kind == "onboarding":
        validate("search", search, request)
        obj(value, {"status", "selected_product_id", "search_result_ids",
                    "profile", "waived_steps", "steps", "explanation"}, "onboarding")
        ids = [m["product"]["id"] for m in search["matches"]]
        require(value["search_result_ids"] == ids, "onboarding search handoff mismatch")
        require(value["profile"] == request["profile"], "onboarding profile mismatch")
        text(value["explanation"], "onboarding.explanation", 2000)
        require(isinstance(value["steps"], list), "steps must be a list")
        require(isinstance(value["waived_steps"], list), "waived_steps must be a list")
        if not ids:
            require(value["status"] == "no_matches" and value["selected_product_id"] is None
                    and value["steps"] == [] and value["waived_steps"] == [],
                    "empty search cannot produce onboarding steps")
            return value
        require(value["status"] == "ready" and value["selected_product_id"] == ids[0],
                "onboarding must select the top search result")
        product = search["matches"][0]["product"]
        graph = prerequisites(product)
        advanced = request["profile"]["experience"] == "advanced"
        waived = value["waived_steps"]
        require(len(waived) == int(advanced), "invalid experience waiver")
        seen = set()
        for waiver in waived:
            obj(waiver, {"id", "reason"}, "waiver")
            require(waiver["id"] == "basics", "only basics can be waived")
            text(waiver["reason"], "waiver.reason")
            seen.add(waiver["id"])
        for step in value["steps"]:
            obj(step, {"id", "title", "product_id", "format", "prerequisites",
                       "detail", "explanation"}, "step")
            text(step["id"], "step.id")
            require(step["id"] in graph and step["id"] not in seen, "unknown or duplicate step")
            require(step["product_id"] == ids[0], "step product handoff mismatch")
            require(step["format"] == request["profile"]["preferred_format"],
                    "step preference mismatch")
            require(step["prerequisites"] == graph[step["id"]], "step prerequisites mismatch")
            require(set(step["prerequisites"]) <= seen, "unresolved step prerequisites")
            for field in ("title", "detail", "explanation"):
                text(step[field], "step." + field, 2000)
            seen.add(step["id"])
        require(seen == set(graph), "onboarding omitted required steps")
    else:
        raise ValidationError("unknown schema boundary")
    return value


def eligible(product, filters):
    category = filters["category"]
    return ((category is None or product["category"].strip().casefold() == category.strip().casefold())
            and (filters["max_price_usd"] is None
                 or product["price_usd"] <= filters["max_price_usd"]))


def one_edit(a, b):
    """Conservative typo tolerance: one insertion, deletion, or substitution."""
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def score_product(product, query_terms):
    # Retain surface forms so plural normalization does not hide nearby typos.
    fields = [(tokens(product["name"], True), 5), (tokens(" ".join(product["tags"]), True), 4),
              (tokens(product["category"], True), 3), (tokens(product["description"], True), 1)]
    total, matched = 0, []
    for term in query_terms:
        best = 0
        for words, weight in fields:
            if term in words:
                best = max(best, weight * 2)
            elif any(one_edit(term, word) for word in words):
                best = max(best, weight)
        if best:
            total += best
            matched.append(term)
    # Partial matches remain available, but complete intent coverage gets a bonus.
    if matched and len(matched) == len(query_terms):
        total += 10
    return total, matched


def smart_search(request):
    validate("request", request)
    terms = tokens(request["query"])
    matches = []
    for product in request["catalog"]:
        if not eligible(product, request["filters"]):
            continue
        score, matched = score_product(product, terms)
        if score:
            matches.append({
                "product": dict(product, tags=list(product["tags"])),
                "score": score, "matched_terms": matched,
                "explanation": "Matched normalized terms: " + ", ".join(matched)
                    + ". Uses field weights, curated synonyms and one-edit typo tolerance."
            })
    matches.sort(key=lambda match: (-match["score"], match["product"]["id"]))
    return validate("search", {"query": request["query"], "normalized_terms": terms,
                              "matches": matches[:request["limit"]]}, request)


def prerequisites(product):
    graph = {"basics": [], "compare": ["basics"]}
    if product["safety_review"]:
        graph["safety"] = ["compare"]
    if product["requires_setup"]:
        graph["configure"] = ["compare"] + (["safety"] if product["safety_review"] else [])
    graph["ready"] = ["compare"] + [s for s in ("safety", "configure") if s in graph]
    return graph


def adaptive_onboarding(request, search):
    validate("search", search, request)
    profile = request["profile"]
    ids = [match["product"]["id"] for match in search["matches"]]
    result = {"status": "ready" if ids else "no_matches",
              "selected_product_id": ids[0] if ids else None, "search_result_ids": ids,
              "profile": dict(profile), "waived_steps": [], "steps": [],
              "explanation": "Top-ranked matching product guides onboarding." if ids
              else "No products matched. Broaden your query or relax explicit filters."}
    if not ids:
        return validate("onboarding", result, request, search)
    product = search["matches"][0]["product"]
    graph = prerequisites(product)
    if profile["experience"] == "advanced":
        result["waived_steps"] = [{"id": "basics",
                                  "reason": "Advanced experience satisfies introductory knowledge."}]
    titles = {"basics": "Understand the product", "compare": "Check fit and budget",
              "safety": "Review safety guidance", "configure": "Plan setup",
              "ready": "Confirm readiness"}
    actions = {
        "basics": "Review the category and key terms before comparing options.",
        "compare": "Compare the matched features and price with your needs.",
        "safety": "Consult the manufacturer's safety instructions before setup or use.",
        "configure": "Check compatibility and follow the manufacturer's setup instructions.",
        "ready": "Review your choices; this guide does not place an order or confirm safety.",
    }
    depth = {"beginner": "Detailed guidance: take one concept at a time.",
             "intermediate": "Concise guidance: focus on unfamiliar features.",
             "advanced": "Expert checklist: verify requirements and assumptions."}[profile["experience"]]
    format_intro = {"text": "Read this checklist.",
                    "video": "Video-style storyboard (text only; no media is generated).",
                    "interactive": "Self-guided interaction: consider each prompt and check your answer."}
    for step_id, deps in graph.items():
        if step_id == "basics" and profile["experience"] == "advanced":
            continue
        result["steps"].append({
            "id": step_id, "title": titles[step_id], "product_id": product["id"],
            "format": profile["preferred_format"], "prerequisites": deps,
            "detail": f"{format_intro[profile['preferred_format']]} {depth} "
                      f"{product['name']} ({product['price_usd']:.2f} USD): {actions[step_id]}",
            "explanation": f"Adapted for {profile['experience']} experience and "
                           f"{profile['preferred_format']} preference. "
                           + ("Requires: " + ", ".join(deps) + ". Earlier steps or explicit "
                              "experience waivers satisfy these prerequisites."
                              if deps else "Introduces knowledge needed for comparison."),
        })
    return validate("onboarding", result, request, search)


def run_pipeline(request):
    validate("request", request)
    search = smart_search(request)
    onboarding = adaptive_onboarding(request, search)
    return {"schema_version": 1, "status": "ok", "fixture_label": request["fixture_label"],
            "search": search, "onboarding": onboarding}


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


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
        with open(args[0], encoding="utf-8") as source:
            request = json.load(source, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(request)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
