"""Synthetic FAQ -> semantic product search reference.

Run: python -B implementation.py example_input.json
Optional Python injection: run_pipeline(payload, embedder=batch_callable).
The callable receives a list of strings and returns one equal-sized, finite,
nonzero numeric vector per string. No provider or network is used here.
"""

import json
import math
import re
import sys
from collections import defaultdict


class ValidationError(ValueError):
    """Invalid input, injected result, or stage handoff."""


STOPWORDS = frozenset(
    "a an the is are do does can i me my for to of and or with which what "
    "suitable work works please you your in on".split()
)
ALIASES = {
    "boots": "boot", "shoes": "shoe", "hiking": "hike", "hikes": "hike",
    "trail": "hike", "trails": "hike", "waterproofing": "waterproof",
    "returns": "return", "refunding": "refund", "refunds": "refund",
}


def terms(text):
    return sorted({
        ALIASES.get(word, word) for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS
    })


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing fields: " +
            ", ".join(sorted(set(required) - set(value))))
    require(set(value) <= set(required) | set(optional), "Unexpected fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()),
            name + " must be a nonblank string")


def strings(value, name):
    require(isinstance(value, list), name + " must be a list")
    for item in value:
        text(item, name + " item")
    require(len(value) == len(set(value)), name + " must have unique values")


def number(value, name, low=0, high=1):
    require(type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high, name + " must be finite and in range")


def validate(value, stage="input", source=None):
    """Single validation boundary used for input and both pipeline stages."""
    if stage == "input":
        fields(value, ("schema_version", "synthetic_fixture", "request",
                       "knowledge_base", "products"))
        require(type(value["schema_version"]) is int
                and value["schema_version"] == 1, "Unsupported schema_version")
        require(value["synthetic_fixture"] is True,
                "This reference requires clearly labeled synthetic data")
        request = value["request"]
        fields(request, ("query",), ("top_k", "faq_min_score", "search_min_score"))
        text(request["query"], "query")
        require(bool(terms(request["query"])), "query must contain searchable terms")
        k = request.get("top_k", 5)
        require(type(k) is int and 1 <= k <= 100, "top_k must be an integer 1..100")
        for name, default in (("faq_min_score", 0.6), ("search_min_score", 0.01)):
            number(request.get(name, default), name)
        ids = {}
        for collection, keys in (
            ("products", ("id", "title", "description", "tags")),
            ("knowledge_base", ("id", "question", "answer", "keywords", "product_ids")),
        ):
            require(isinstance(value[collection], list), collection + " must be a list")
            ids[collection] = set()
            for item in value[collection]:
                fields(item, keys)
                for name in keys:
                    if name in ("tags", "keywords", "product_ids"):
                        strings(item[name], name)
                    else:
                        text(item[name], name)
                require(item["id"] not in ids[collection], "Duplicate " + collection + " id")
                ids[collection].add(item["id"])
        for entry in value["knowledge_base"]:
            require(set(entry["product_ids"]) <= ids["products"],
                    "FAQ references an unknown product")
    elif stage == "faq":
        require(source is not None, "FAQ validation needs input")
        fields(value, ("status", "answer", "citations", "confidence", "handoff"))
        require(value["status"] in ("answered", "abstained"), "Invalid FAQ status")
        number(value["confidence"], "confidence")
        strings(value["citations"], "citations")
        handoff = value["handoff"]
        fields(handoff, ("query", "grounded_terms", "product_ids"))
        require(handoff["query"] == source["request"]["query"], "Query changed in handoff")
        strings(handoff["grounded_terms"], "grounded_terms")
        strings(handoff["product_ids"], "product_ids")
        if value["status"] == "abstained":
            require(value["answer"] is None and value["citations"] == []
                    and handoff["grounded_terms"] == [] and handoff["product_ids"] == [],
                    "Abstention must not invent evidence")
        else:
            require(len(value["citations"]) == 1, "Exactly one citation is required")
            entry = next((item for item in source["knowledge_base"]
                          if item["id"] == value["citations"][0]), None)
            require(entry is not None, "Unknown citation")
            require(value["answer"] == entry["answer"], "Answer is not grounded")
            require(handoff["product_ids"] == sorted(entry["product_ids"])
                    and handoff["grounded_terms"] == terms(" ".join(entry["keywords"])),
                    "Handoff differs from cited evidence")
            expected = faq_score(source["request"]["query"], entry)
            require(value["confidence"] == expected and expected > 0
                    and expected >= source["request"].get("faq_min_score", 0.6),
                    "Insufficient answer evidence")
    elif stage == "semantic":
        fields(value, ("query", "expanded_terms", "embedding_used", "results"))
        payload, faq = source
        validate(faq, "faq", payload)
        require(value["query"] == faq["handoff"]["query"], "Search query changed")
        expected = sorted(set(terms(faq["handoff"]["query"]))
                          | set(faq["handoff"]["grounded_terms"]))
        require(value["expanded_terms"] == expected, "Search omitted FAQ handoff")
        require(type(value["embedding_used"]) is bool, "Invalid embedding flag")
        require(isinstance(value["results"], list), "results must be a list")
        require(len(value["results"]) <= payload["request"].get("top_k", 5),
                "Too many search results")
        products = {p["id"]: p for p in payload["products"]}
        seen = set()
        for result in value["results"]:
            fields(result, ("product_id", "title", "score", "components", "faq_linked"))
            pid = result["product_id"]
            require(isinstance(pid, str) and pid in products and pid not in seen,
                    "Unknown or duplicate result product")
            seen.add(pid)
            require(result["title"] == products[pid]["title"], "Product title changed")
            number(result["score"], "score")
            require(result["score"] > 0 and result["score"] >=
                    payload["request"].get("search_min_score", 0.01), "Irrelevant result")
            fields(result["components"], ("lexical", "embedding", "faq_link"))
            for component in result["components"].values():
                number(component, "score component")
            require(type(result["faq_linked"]) is bool and
                    result["faq_linked"] == (pid in faq["handoff"]["product_ids"]),
                    "FAQ product link changed")
        require(value["results"] == sorted(value["results"],
                key=lambda r: (-r["score"], r["product_id"])), "Results not ranked")
    else:
        raise ValidationError("Unknown validation stage")
    return value


def faq_score(query, entry):
    query_terms = set(terms(query))
    document_terms = set(terms(entry["question"] + " " + " ".join(entry["keywords"])))
    return round(len(query_terms & document_terms) / len(query_terms), 8)


def answer_faq(payload):
    validate(payload)
    ranked = sorted(((faq_score(payload["request"]["query"], entry), entry)
                     for entry in payload["knowledge_base"]),
                    key=lambda pair: (-pair[0], pair[1]["id"]))
    score, entry = ranked[0] if ranked else (0.0, None)
    accepted = entry is not None and score > 0 and score >= payload["request"].get(
        "faq_min_score", 0.6)
    result = {
        "status": "answered" if accepted else "abstained",
        "answer": entry["answer"] if accepted else None,
        "citations": [entry["id"]] if accepted else [],
        "confidence": score,
        "handoff": {
            "query": payload["request"]["query"],
            "grounded_terms": terms(" ".join(entry["keywords"])) if accepted else [],
            "product_ids": sorted(entry["product_ids"]) if accepted else [],
        },
    }
    return validate(result, "faq", payload)


class SearchIndex:
    """In-memory weighted inverted index, rebuilt for each bounded request."""

    def __init__(self, products):
        self.postings = defaultdict(dict)
        for product in products:
            for value, weight in ((product["title"], 3), (" ".join(product["tags"]), 2),
                                  (product["description"], 1)):
                for token in terms(value):
                    previous = self.postings[token].get(product["id"], 0)
                    self.postings[token][product["id"]] = max(previous, weight)

    def scores(self, query_terms):
        scores = defaultdict(float)
        for token in query_terms:
            for pid, weight in self.postings.get(token, {}).items():
                scores[pid] += weight / (3 * len(query_terms))
        return scores


def embedding_scores(embedder, query, products):
    texts = [query] + [p["title"] + " " + p["description"] + " " + " ".join(p["tags"])
                       for p in products]
    try:
        vectors = embedder(texts)
    except Exception as exc:
        raise ValidationError("Injected embedder failed") from exc
    require(isinstance(vectors, (list, tuple)) and len(vectors) == len(texts),
            "Embedder must return one vector per text")
    normalized = []
    dimension = None
    for vector in vectors:
        require(isinstance(vector, (list, tuple)) and len(vector) > 0,
                "Embedding vectors must be nonempty")
        dimension = len(vector) if dimension is None else dimension
        require(len(vector) == dimension, "Embedding dimensions differ")
        for item in vector:
            require(type(item) in (int, float) and math.isfinite(item),
                    "Embedding coordinates must be finite numbers")
        scale = max(abs(item) for item in vector)
        require(scale > 0, "Embedding vectors must be nonzero")
        scaled = [item / scale for item in vector]
        norm = math.sqrt(sum(item * item for item in scaled))
        normalized.append([item / norm for item in scaled])
    return {
        product["id"]: max(0.0, min(1.0, sum(a * b for a, b in
                                           zip(normalized[0], vector))))
        for product, vector in zip(products, normalized[1:])
    }


def search_products(payload, faq, embedder=None):
    validate(payload)
    validate(faq, "faq", payload)
    handoff = faq["handoff"]
    expanded = sorted(set(terms(handoff["query"])) | set(handoff["grounded_terms"]))
    lexical = SearchIndex(payload["products"]).scores(expanded)
    embedded = embedding_scores(embedder, " ".join(expanded), payload["products"]) \
        if embedder is not None else {}
    results = []
    for product in payload["products"]:
        pid = product["id"]
        linked = pid in handoff["product_ids"]
        lex = min(1.0, lexical.get(pid, 0.0))
        emb = embedded.get(pid, 0.0)
        score = round((0.6 if embedder is not None else 0.9) * lex
                      + (0.3 * emb if embedder is not None else 0) + 0.1 * linked, 8)
        if score > 0 and score >= payload["request"].get("search_min_score", 0.01):
            results.append({
                "product_id": pid, "title": product["title"], "score": score,
                "components": {"lexical": round(lex, 8), "embedding": round(emb, 8),
                               "faq_link": float(linked)},
                "faq_linked": linked,
            })
    results.sort(key=lambda item: (-item["score"], item["product_id"]))
    result = {
        "query": handoff["query"], "expanded_terms": expanded,
        "embedding_used": embedder is not None,
        "results": results[:payload["request"].get("top_k", 5)],
    }
    return validate(result, "semantic", (payload, faq))


def run_pipeline(payload, embedder=None):
    validate(payload)
    faq = answer_faq(payload)
    semantic = search_products(payload, faq, embedder)
    return {"schema_version": 1, "synthetic_fixture": True, "status": "ok",
            "faq": faq, "semantic": semantic}


def reject_constant(value):
    raise ValidationError("Nonfinite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        output = run_pipeline(payload)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error",
                          "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
