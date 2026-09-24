"""Offline deterministic search-to-research reference pipeline (Python stdlib)."""
import difflib
import json
import math
from pathlib import Path
import re
import sys


class ValidationError(ValueError):
    pass


STOP = frozenset("a an the for with and or to of is are which what best should i".split())
SYNONYMS = {
    "wireless": "cordless", "cordless": "cordless",
    "headphones": "headphone", "headsets": "headphone", "headset": "headphone",
    "earphones": "headphone", "laptops": "laptop", "notebook": "laptop",
    "notebooks": "laptop", "sofa": "couch", "sofas": "couch",
    "running": "run", "jogging": "run", "sneakers": "shoe", "shoes": "shoe",
}


def tokens(text):
    return {SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", text.casefold())
            if t not in STOP}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required fields: " + ", ".join(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "Unknown fields")


def number(value, label):
    require((type(value) is int or (type(value) is float and math.isfinite(value))) and value >= 0,
            label + " must be a finite nonnegative number")


def validate(kind, value, context=None):
    """One validation boundary for requests, search handoffs and final reports."""
    if kind == "input":
        fields(value, ("schema_version", "fixture_label", "query", "question", "products", "sources"),
               ("filters", "limit"))
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        for name in ("fixture_label", "query", "question"):
            text(value[name], name)
        require(isinstance(value["products"], list), "products must be an array")
        require(isinstance(value["sources"], list), "sources must be an array")
        ids = set()
        currencies = set()
        for product in value["products"]:
            fields(product, ("id", "name", "description", "category", "price", "currency", "in_stock"))
            for name in ("id", "name", "description", "category", "currency"):
                text(product[name], "product." + name)
            require(product["id"] not in ids, "Duplicate product ID")
            ids.add(product["id"])
            number(product["price"], "price")
            require(bool(re.fullmatch(r"[A-Z]{3}", product["currency"])), "currency must be three uppercase letters")
            currencies.add(product["currency"])
            require(type(product["in_stock"]) is bool, "in_stock must be boolean")
        require(len(currencies) <= 1, "Mixed currencies are unsupported")
        source_ids = set()
        for source in value["sources"]:
            fields(source, ("id", "product_id", "title", "text", "kind"))
            for name in source:
                text(source[name], "source." + name)
            require(source["id"] not in source_ids, "Duplicate source ID")
            source_ids.add(source["id"])
            require(source["product_id"] in ids, "Unknown source product_id")
            require(source["kind"] in ("specification", "review", "note"), "Unsupported source kind")
        filters = value.get("filters", {})
        fields(filters, (), ("max_price", "category", "in_stock"))
        if "max_price" in filters:
            number(filters["max_price"], "max_price")
        if "category" in filters:
            text(filters["category"], "category")
        if "in_stock" in filters:
            require(type(filters["in_stock"]) is bool, "filter in_stock must be boolean")
        require(type(value.get("limit", 5)) is int and 1 <= value.get("limit", 5) <= 50,
                "limit must be an integer from 1 to 50")
    elif kind == "search":
        require(context is not None, "Search validation needs request context")
        validate("input", context)
        fields(value, ("query", "interpreted_terms", "results"))
        require(value["query"] == context["query"], "Query handoff mismatch")
        require(value["interpreted_terms"] == sorted(tokens(context["query"])), "Term handoff mismatch")
        require(isinstance(value["results"], list), "results must be an array")
        require(len(value["results"]) <= context.get("limit", 5), "Too many search results")
        catalog = {p["id"]: p for p in context["products"]}
        seen = set()
        for row in value["results"]:
            fields(row, ("product", "score", "matches"))
            require(isinstance(row["product"], dict), "product must be an object")
            pid = row["product"].get("id")
            require(isinstance(pid, str) and pid in catalog, "Unknown result product")
            require(row["product"] == catalog[pid], "Product changed in handoff")
            require(pid not in seen, "Duplicate result product")
            seen.add(pid)
            number(row["score"], "score")
            require(0 < row["score"] <= 1, "score must be in (0, 1]")
            require(isinstance(row["matches"], list) and bool(row["matches"]), "Missing match explanation")
            for match in row["matches"]:
                fields(match, ("query_term", "matched_term", "method"))
                require(match["query_term"] in value["interpreted_terms"], "Invalid matched query term")
                require(match["matched_term"] in product_tokens(row["product"]), "Invalid matched product term")
                require(match["method"] in ("exact_or_synonym", "typo"), "Invalid match method")
            require(eligible(row["product"], context.get("filters", {})), "Result violates filters")
    elif kind == "output":
        fields(value, ("schema_version", "status", "fixture_label", "search", "research"))
        require(value["schema_version"] == 1 and value["status"] == "ok", "Invalid output envelope")
        require(value["fixture_label"] == context["fixture_label"], "Fixture label changed")
        validate("search", value["search"], context)
        report = value["research"]
        fields(report, ("question", "evidence", "comparison", "recommendation", "limitations"))
        require(report["question"] == context["question"], "Question handoff mismatch")
        selected = [r["product"]["id"] for r in value["search"]["results"]]
        sources = {s["id"]: s for s in context["sources"]}
        require(isinstance(report["evidence"], list), "evidence must be an array")
        cited = set()
        for evidence in report["evidence"]:
            fields(evidence, ("source_id", "product_id", "title", "kind", "quote", "relevance"))
            sid = evidence["source_id"]
            require(isinstance(sid, str) and sid in sources and sid not in cited, "Invalid or repeated citation")
            cited.add(sid)
            source = sources[sid]
            require(evidence["product_id"] in selected, "Evidence references unselected product")
            for key in ("product_id", "title", "kind"):
                require(evidence[key] == source[key], "Evidence provenance changed")
            require(evidence["quote"] == source["text"], "Evidence must quote source verbatim")
            number(evidence["relevance"], "relevance")
            require(0 < evidence["relevance"] <= 1, "Invalid evidence relevance")
        require(isinstance(report["comparison"], list), "comparison must be an array")
        require([r.get("product_id") for r in report["comparison"]] == selected, "Comparison handoff mismatch")
        for row in report["comparison"]:
            fields(row, ("product_id", "evidence_ids", "evidence_count"))
            expected = [e["source_id"] for e in report["evidence"] if e["product_id"] == row["product_id"]]
            require(row["evidence_ids"] == expected and row["evidence_count"] == len(expected),
                    "Comparison citations mismatch")
        recommendation = report["recommendation"]
        if recommendation is not None:
            fields(recommendation, ("product_id", "basis", "evidence_ids"))
            require(recommendation["product_id"] in selected, "Invalid recommendation product")
            text(recommendation["basis"], "recommendation basis")
            expected = next(r["evidence_ids"] for r in report["comparison"]
                            if r["product_id"] == recommendation["product_id"])
            require(bool(expected) and recommendation["evidence_ids"] == expected, "Unsubstantiated recommendation")
        require(isinstance(report["limitations"], list) and bool(report["limitations"]), "Missing limitations")
        for limitation in report["limitations"]:
            text(limitation, "limitation")
    else:
        raise ValidationError("Unknown validation boundary")
    return value


def product_tokens(product):
    return tokens(" ".join(product[k] for k in ("name", "description", "category")))


def eligible(product, filters):
    return (
        ("max_price" not in filters or product["price"] <= filters["max_price"])
        and ("category" not in filters or product["category"].casefold() == filters["category"].casefold())
        and ("in_stock" not in filters or product["in_stock"] == filters["in_stock"])
    )


def search(request):
    validate("input", request)
    terms = sorted(tokens(request["query"]))
    results = []
    for product in request["products"]:
        if not eligible(product, request.get("filters", {})):
            continue
        vocabulary = sorted(product_tokens(product))
        matches = []
        weight = 0
        for term in terms:
            if term in vocabulary:
                matches.append({"query_term": term, "matched_term": term, "method": "exact_or_synonym"})
                weight += 1
            elif len(term) >= 4:
                candidates = [(difflib.SequenceMatcher(None, term, word).ratio(), word)
                              for word in vocabulary if len(word) >= 4]
                candidates.sort(key=lambda item: (-item[0], item[1]))
                if candidates and candidates[0][0] >= 0.82:
                    matches.append({"query_term": term, "matched_term": candidates[0][1], "method": "typo"})
                    weight += candidates[0][0] * 0.8
        if matches:
            results.append({"product": dict(product), "score": round(weight / len(terms), 6), "matches": matches})
    results.sort(key=lambda row: (-row["score"], row["product"]["price"], row["product"]["id"]))
    return validate("search", {"query": request["query"], "interpreted_terms": terms,
                              "results": results[:request.get("limit", 5)]}, request)


def research(request, search_result):
    validate("search", search_result, request)
    selected = [row["product"]["id"] for row in search_result["results"]]
    question_terms = tokens(request["question"])
    evidence = []
    for source in sorted(request["sources"], key=lambda source: source["id"]):
        overlap = question_terms & tokens(source["title"] + " " + source["text"])
        if source["product_id"] in selected and overlap:
            evidence.append({
                "source_id": source["id"], "product_id": source["product_id"],
                "title": source["title"], "kind": source["kind"], "quote": source["text"],
                "relevance": round(len(overlap) / len(question_terms), 6),
            })
    comparison = []
    for pid in selected:
        ids = [e["source_id"] for e in evidence if e["product_id"] == pid]
        comparison.append({"product_id": pid, "evidence_ids": ids, "evidence_count": len(ids)})
    supported = [row for row in comparison if row["evidence_count"]]
    supported.sort(key=lambda row: (-row["evidence_count"], selected.index(row["product_id"])))
    recommendation = None
    if supported:
        best = supported[0]
        recommendation = {
            "product_id": best["product_id"],
            "basis": "Prioritize for further evaluation: most question-relevant supplied sources; ties use search rank. Not a quality or purchase verdict.",
            "evidence_ids": best["evidence_ids"],
        }
    limitations = [
        "Offline lexical matching and a small synonym dictionary do not provide full semantic understanding.",
        "Sources are supplied, not independently verified; counts do not measure quality or independence.",
        "Quotes may disagree; this pipeline does not resolve contradictions, sentiment, or numerical comparisons.",
    ]
    if not selected:
        limitations.append("No products matched the query and filters.")
    if not evidence:
        limitations.append("Insufficient question-relevant evidence; recommendation withheld.")
    elif any(not row["evidence_count"] for row in comparison):
        limitations.append("Some selected products lack question-relevant evidence.")
    return {"question": request["question"], "evidence": evidence, "comparison": comparison,
            "recommendation": recommendation, "limitations": limitations}


def run_pipeline(request):
    validate("input", request)
    handoff = search(request)
    report = research(request, handoff)
    return validate("output", {
        "schema_version": 1, "status": "ok", "fixture_label": request["fixture_label"],
        "search": handoff, "research": report,
    }, request)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        result = run_pipeline(request)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
