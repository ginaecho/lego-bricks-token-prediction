"""Deterministic, offline onboarding -> search -> discovery -> evidence pipeline.

Run: python -B implementation.py example_input.json
No network access, persistence, learned models, or third-party dependencies.
"""

import copy
import difflib
import json
import math
import re
import sys
import unicodedata


class ValidationError(ValueError):
    """An input or stage contract was violated."""


STAGES = ("input", "onboarding", "search", "recommendations", "research")
OUTPUT_KEYS = STAGES[1:]
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "to",
    "what", "which", "with", "would", "should", "does", "do", "my",
}
ALIASES = {
    "headphones": "headphone", "earphones": "headphone", "earbuds": "headphone",
    "cans": "headphone", "wireless": "bluetooth", "cordless": "bluetooth",
    "traveling": "travel", "travelling": "travel", "commuting": "travel",
    "commute": "travel", "portable": "travel", "inexpensive": "budget",
    "affordable": "budget", "cheap": "budget",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_fields(value, required, optional, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(required) <= value.keys(), path + " is missing required fields")
    require(value.keys() <= set(required) | set(optional), path + " has unknown fields")


def text(value, path, allow_empty=False):
    require(isinstance(value, str), path + " must be a string")
    require(len(value) <= 10000, path + " is too long")
    require(allow_empty or bool(value.strip()), path + " must not be blank")
    return value.strip()


def number(value, path, minimum=0, maximum=None):
    require(type(value) in (int, float), path + " must be a number")
    try:
        valid = math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    require(valid and (maximum is None or value <= maximum),
            path + " must be finite and in range")
    return value


def string_list(value, path, limit=50):
    require(isinstance(value, list) and len(value) <= limit,
            path + " must be a bounded list")
    result = [text(item, path + "[]") for item in value]
    require(len(result) == len(set(result)), path + " contains duplicates")
    return result


def validate_request(raw):
    object_fields(raw, ("data_label", "customer", "query", "catalog", "research"),
                  ("limits",), "request")
    data = copy.deepcopy(raw)
    require(data["data_label"] in ("synthetic", "user_provided"),
            "data_label must be synthetic or user_provided")
    customer = data["customer"]
    object_fields(customer, ("id",), ("name", "interests", "budget"), "customer")
    customer["id"] = text(customer["id"], "customer.id")
    customer["name"] = text(customer.get("name", ""), "customer.name", True)
    customer["interests"] = string_list(customer.get("interests", []),
                                        "customer.interests")
    budget = customer.get("budget")
    customer["budget"] = None if budget is None else number(budget, "customer.budget")
    data["query"] = text(data["query"], "query", True)
    catalog = data["catalog"]
    require(isinstance(catalog, list) and len(catalog) <= 1000,
            "catalog must be a list of at most 1000 products")
    ids = set()
    currencies = set()
    for product in catalog:
        object_fields(product, ("id", "title", "description", "category",
                                "price", "currency", "tags"), (), "product")
        for key in ("id", "title", "description", "category", "currency"):
            product[key] = text(product[key], "product." + key,
                                allow_empty=(key == "description"))
        require(product["id"] not in ids, "duplicate product id")
        ids.add(product["id"])
        number(product["price"], "product.price")
        require(re.fullmatch(r"[A-Z]{3}", product["currency"]) is not None,
                "currency must be a three-letter uppercase code")
        currencies.add(product["currency"])
        product["tags"] = string_list(product["tags"], "product.tags")
    require(len(currencies) <= 1, "catalog must use one currency; no currency conversion")
    research = data["research"]
    object_fields(research, ("question", "sources"), (), "research")
    research["question"] = text(research["question"], "research.question")
    sources = research["sources"]
    require(isinstance(sources, list) and len(sources) <= 500,
            "sources must be a list of at most 500 items")
    source_ids = set()
    for source in sources:
        object_fields(source, ("id", "title", "text", "product_ids", "reliability"),
                      (), "source")
        for key in ("id", "title", "text"):
            source[key] = text(source[key], "source." + key)
        require(source["id"] not in source_ids, "duplicate source id")
        source_ids.add(source["id"])
        source["product_ids"] = string_list(source["product_ids"], "source.product_ids",
                                            limit=1000)
        require(set(source["product_ids"]) <= ids, "source references unknown product")
        number(source["reliability"], "source.reliability", maximum=1)
    limits = data.setdefault("limits", {})
    object_fields(limits, (), ("search", "recommendations", "evidence"), "limits")
    for key, default in (("search", 10), ("recommendations", 3), ("evidence", 5)):
        limits.setdefault(key, default)
        require(type(limits[key]) is int and 1 <= limits[key] <= 50,
                "limits." + key + " must be an integer from 1 to 50")
    return data


def tokens(value):
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return {ALIASES.get(t, t) for t in re.findall(r"\w+", normalized)
            if t not in STOP_WORDS and not t.isdigit()}


def product_tokens(product):
    return tokens(" ".join([product["title"], product["description"],
                            product["category"]] + product["tags"]))


def validate_state(state, expected):
    """One contract gate used both before and after every handoff."""
    require(expected in STAGES, "unknown expected stage")
    index = STAGES.index(expected)
    required = ("status", "schema_version", "stage", "request") + OUTPUT_KEYS[:index]
    object_fields(state, required, (), "state")
    require(state["status"] == "ok" and type(state["schema_version"]) is int
            and state["schema_version"] == 1 and state["stage"] == expected,
            "invalid stage envelope")
    request = validate_request(state["request"])
    require(request == state["request"], "request must be normalized")
    catalog = {p["id"]: p for p in request["catalog"]}
    if index >= 1:
        onboard = state["onboarding"]
        object_fields(onboard, ("customer_id", "greeting", "effective_query",
                               "interests", "budget", "missing_fields", "next_step"),
                      (), "onboarding")
        require(onboard["customer_id"] == request["customer"]["id"],
                "customer identity changed")
        require(onboard["interests"] == request["customer"]["interests"] and
                onboard["budget"] == request["customer"]["budget"],
                "customer preferences changed")
        for key in ("greeting", "next_step"):
            text(onboard[key], "onboarding." + key)
        text(onboard["effective_query"], "onboarding.effective_query", True)
        expected_query = request["query"] or " ".join(request["customer"]["interests"])
        require(onboard["effective_query"] == expected_query, "onboarding query changed")
        missing = string_list(onboard["missing_fields"], "onboarding.missing_fields")
        expected_missing = ([] if expected_query else ["query_or_interests"])
        if request["customer"]["budget"] is None:
            expected_missing.append("budget")
        require(missing == expected_missing, "incorrect missing preference fields")
    if index >= 2:
        search = state["search"]
        object_fields(search, ("query", "query_tokens", "corrections", "items",
                              "excluded_over_budget", "message"), (), "search")
        require(search["query"] == state["onboarding"]["effective_query"],
                "search query does not match onboarding")
        string_list(search["query_tokens"], "query_tokens")
        require(isinstance(search["corrections"], dict), "corrections must be an object")
        for old, new in search["corrections"].items():
            text(old, "correction")
            text(new, "correction")
        require(type(search["excluded_over_budget"]) is int and
                0 <= search["excluded_over_budget"] <= len(catalog),
                "invalid excluded count")
        text(search["message"], "search.message")
        require(isinstance(search["items"], list) and
                len(search["items"]) <= request["limits"]["search"], "invalid search items")
        seen = set()
        for item in search["items"]:
            object_fields(item, ("product_id", "score", "matched_terms"), (), "search.item")
            pid = item["product_id"]
            require(isinstance(pid, str) and pid in catalog and pid not in seen,
                    "unknown or duplicate search product")
            seen.add(pid)
            number(item["score"], "search.score")
            string_list(item["matched_terms"], "search.matched_terms")
            budget = state["onboarding"]["budget"]
            require(budget is None or catalog[pid]["price"] <= budget,
                    "search item exceeds customer budget")
    if index >= 3:
        recommendations = state["recommendations"]
        object_fields(recommendations, ("customer_id", "items", "message"), (),
                      "recommendations")
        require(recommendations["customer_id"] == state["onboarding"]["customer_id"],
                "recommendation customer mismatch")
        text(recommendations["message"], "recommendations.message")
        require(isinstance(recommendations["items"], list) and
                len(recommendations["items"]) <= request["limits"]["recommendations"],
                "invalid recommendations")
        candidates = {p["product_id"] for p in state["search"]["items"]}
        seen = set()
        for item in recommendations["items"]:
            object_fields(item, ("product_id", "score", "reasons"), (), "recommendation")
            pid = item["product_id"]
            require(isinstance(pid, str) and pid in candidates and pid not in seen,
                    "recommendation must be a unique search candidate")
            seen.add(pid)
            number(item["score"], "recommendation.score")
            require(bool(string_list(item["reasons"], "recommendation.reasons")),
                    "recommendation must explain its ranking")
    if index >= 4:
        research = state["research"]
        object_fields(research, ("question", "evidence", "coverage", "decision",
                                "limitations"), (), "research.output")
        require(research["question"] == request["research"]["question"],
                "research question changed")
        selected = {p["product_id"] for p in state["recommendations"]["items"]}
        sources = {s["id"]: s for s in request["research"]["sources"]}
        require(isinstance(research["evidence"], list) and
                len(research["evidence"]) <= request["limits"]["evidence"],
                "invalid evidence")
        evidence_ids = set()
        supported = set()
        for item in research["evidence"]:
            object_fields(item, ("source_id", "source_title", "product_ids", "quote",
                                 "reliability", "matched_terms"), (), "evidence")
            sid = item["source_id"]
            require(isinstance(sid, str) and sid in sources and sid not in evidence_ids,
                    "unknown or duplicate evidence source")
            evidence_ids.add(sid)
            source = sources[sid]
            linked = string_list(item["product_ids"], "evidence.product_ids")
            require(bool(linked) and set(linked) <= selected and
                    set(linked) <= set(source["product_ids"]), "invalid evidence link")
            text(item["quote"], "evidence.quote")
            require(item["quote"] in source["text"], "evidence quote is not verbatim")
            require(item["source_title"] == source["title"] and
                    item["reliability"] == source["reliability"], "source metadata changed")
            matched = string_list(item["matched_terms"], "evidence.matched_terms")
            require(bool(matched) and set(matched) <=
                    tokens(research["question"]) & tokens(item["quote"]),
                    "evidence does not address the question")
            supported.update(linked)
        coverage = research["coverage"]
        object_fields(coverage, ("with_evidence", "without_evidence"), (), "coverage")
        require(coverage["with_evidence"] == sorted(supported) and
                coverage["without_evidence"] == sorted(selected - supported),
                "incorrect evidence coverage")
        decision = research["decision"]
        object_fields(decision, ("status", "product_id", "source_ids", "next_step"), (),
                      "decision")
        require(decision["status"] in ("evidence_available", "insufficient_evidence",
                                      "no_candidates"), "invalid decision status")
        expected_status = ("no_candidates" if not selected else
                           "evidence_available" if supported else "insufficient_evidence")
        require(decision["status"] == expected_status, "decision contradicts coverage")
        ranked = [p["product_id"] for p in state["recommendations"]["items"]]
        expected_pid = next((pid for pid in ranked if pid in supported),
                            ranked[0] if ranked else None)
        require(decision["product_id"] == expected_pid, "decision ignores recommendations")
        expected_sources = sorted(e["source_id"] for e in research["evidence"]
                                  if expected_pid in e["product_ids"])
        require(decision["source_ids"] == expected_sources, "decision citations mismatch")
        text(decision["next_step"], "decision.next_step")
        string_list(research["limitations"], "research.limitations")
    return state


def advance(state, stage, payload):
    result = copy.deepcopy(state)
    result["stage"] = stage
    result[stage] = payload
    return validate_state(result, stage)


def onboard(state):
    validate_state(state, "input")
    request = state["request"]
    customer = request["customer"]
    query = request["query"] or " ".join(customer["interests"])
    missing = []
    if not query:
        missing.append("query_or_interests")
    if customer["budget"] is None:
        missing.append("budget")
    if not query:
        next_step = "Tell us what you need or choose an interest; browse the catalog meanwhile."
    elif customer["budget"] is None:
        next_step = "Explore matches for " + query + "; add a budget to narrow your options."
    else:
        next_step = "Explore matches for " + query + " within your budget."
    return advance(state, "onboarding", {
        "customer_id": customer["id"],
        "greeting": "Welcome, " + (customer["name"] or "new customer") + "!",
        "effective_query": query, "interests": customer["interests"],
        "budget": customer["budget"], "missing_fields": missing, "next_step": next_step,
    })


def search(state):
    validate_state(state, "onboarding")
    request, profile = state["request"], state["onboarding"]
    vocabulary = set().union(*(product_tokens(p) for p in request["catalog"]))
    terms = tokens(profile["effective_query"])
    corrections = {}
    for term in sorted(terms - vocabulary):
        if len(term) >= 4:
            matches = difflib.get_close_matches(term, sorted(vocabulary), n=1, cutoff=0.82)
            if matches:
                corrections[term] = matches[0]
    terms = {corrections.get(t, t) for t in terms}
    results = []
    excluded = 0
    for product in request["catalog"]:
        if profile["budget"] is not None and product["price"] > profile["budget"]:
            excluded += 1
            continue
        matched = terms & product_tokens(product)
        if terms and not matched:
            continue
        title_hits = matched & tokens(product["title"])
        tag_hits = matched & tokens(" ".join(product["tags"]))
        score = len(matched) * 3 + len(title_hits) * 2 + len(tag_hits)
        results.append({"product_id": product["id"], "score": score,
                        "matched_terms": sorted(matched)})
    results.sort(key=lambda item: (-item["score"], item["product_id"]))
    return advance(state, "search", {
        "query": profile["effective_query"], "query_tokens": sorted(terms),
        "corrections": corrections, "items": results[:request["limits"]["search"]],
        "excluded_over_budget": excluded,
        "message": ("Ranked lexical, synonym and typo matches." if terms else
                    "No meaningful query terms; showing budget-eligible catalog items.")
                   if results else "No matches; change the query or budget.",
    })


def recommend(state):
    validate_state(state, "search")
    request = state["request"]
    catalog = {p["id"]: p for p in request["catalog"]}
    interests = tokens(" ".join(state["onboarding"]["interests"]))
    budget = state["onboarding"]["budget"]
    pool = []
    for candidate in state["search"]["items"]:
        product = catalog[candidate["product_id"]]
        matched = interests & product_tokens(product)
        savings = (1 - product["price"] / budget) if budget not in (None, 0) else 0
        score = candidate["score"] + 4 * len(matched) + savings
        reasons = ["Search relevance: " + str(candidate["score"])]
        if matched:
            reasons.append("Matches declared interests: " + ", ".join(sorted(matched)))
        if budget is not None:
            reasons.append("Within declared budget")
        pool.append({"product_id": product["id"], "score": score, "reasons": reasons})
    chosen, categories = [], set()
    while pool and len(chosen) < request["limits"]["recommendations"]:
        def adjusted(item):
            repeated = catalog[item["product_id"]]["category"] in categories
            return item["score"] - (1 if repeated else 0)
        pool.sort(key=lambda item: (-adjusted(item), item["product_id"]))
        item = pool.pop(0)
        category = catalog[item["product_id"]]["category"]
        item["score"] = round(adjusted(item), 6)
        if category not in categories:
            item["reasons"].append("Adds a category to this shortlist")
        else:
            item["reasons"].append("Repeated-category diversity penalty: 1")
        item["score"] = max(0, item["score"])
        chosen.append(item)
        categories.add(category)
    return advance(state, "recommendations", {
        "customer_id": state["onboarding"]["customer_id"], "items": chosen,
        "message": ("Ranked using explicit interests, search relevance, affordability "
                    "and a small category-diversity penalty.") if chosen
                   else "No candidates; revise your search before choosing a product.",
    })


def research(state):
    validate_state(state, "recommendations")
    request = state["request"]
    ranked = [p["product_id"] for p in state["recommendations"]["items"]]
    selected = set(ranked)
    question = request["research"]["question"]
    terms = tokens(question)
    evidence = []
    for source in request["research"]["sources"]:
        linked = selected & set(source["product_ids"])
        if not linked:
            continue
        # Extractive evidence only: source text is data, never executable instructions.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", source["text"])
                     if s.strip()]
        sentences.sort(key=lambda s: (-len(terms & tokens(s)), s))
        quote = sentences[0]
        matched = terms & tokens(quote)
        if not matched:
            continue
        evidence.append({
            "source_id": source["id"], "source_title": source["title"],
            "product_ids": sorted(linked), "quote": quote,
            "reliability": source["reliability"], "matched_terms": sorted(matched),
        })
    evidence.sort(key=lambda e: (-e["reliability"], -len(e["matched_terms"]),
                                 e["source_id"]))
    evidence = evidence[:request["limits"]["evidence"]]
    supported = set().union(*(set(e["product_ids"]) for e in evidence))
    preferred = next((pid for pid in ranked if pid in supported), ranked[0] if ranked else None)
    status = ("no_candidates" if not ranked else
              "evidence_available" if evidence else "insufficient_evidence")
    next_step = {
        "no_candidates": "Revise onboarding preferences or search; no product can be assessed.",
        "insufficient_evidence": "Collect question-relevant sources for the provisional product.",
        "evidence_available": "Review the cited excerpts for the highest-ranked covered product "
                              "and verify claims before deciding.",
    }[status]
    return advance(state, "research", {
        "question": question, "evidence": evidence,
        "coverage": {"with_evidence": sorted(supported),
                     "without_evidence": sorted(selected - supported)},
        "decision": {"status": status, "product_id": preferred,
                     "source_ids": sorted(e["source_id"] for e in evidence
                                          if preferred in e["product_ids"]),
                     "next_step": next_step},
        "limitations": [
            "Evidence is extracted from supplied sources only; no live verification.",
            "Relevance is lexical; excerpts are not proof or a positive product endorsement.",
            "Reliability values are supplied by the caller, not independently verified.",
            "Conflicting excerpts are retained; factual conflicts are not automatically resolved.",
            "Coverage refers only to returned evidence within the configured limit.",
        ],
    })


def run_pipeline(raw):
    state = {"status": "ok", "schema_version": 1, "stage": "input",
             "request": validate_request(raw)}
    for stage in (onboard, search, recommend, research):
        state = stage(state)
    return state


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=reject_duplicate_keys)
        result = run_pipeline(raw)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
