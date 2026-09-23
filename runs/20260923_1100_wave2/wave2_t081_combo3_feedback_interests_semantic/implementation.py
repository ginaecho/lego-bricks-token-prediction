"""Synthetic, deterministic feedback -> interests -> semantic discovery reference.

Run: python -B implementation.py example_input.json
Optional library injection: run_pipeline(data, embedder=batch_texts_to_vectors).
No providers are contacted. Unknown schema fields are rejected.
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), path="value"):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(required) <= value.keys(), f"{path} missing required fields")
    require(value.keys() <= set(required) | set(optional), f"{path} has unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonempty text")
    return value


def number(value, path, low=None, high=None):
    require(type(value) in (int, float), f"{path} must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, f"{path} must be finite")
    require(low is None or value >= low, f"{path} is below minimum")
    require(high is None or value <= high, f"{path} is above maximum")
    return value


def strings(value, path):
    require(isinstance(value, list), f"{path} must be a list")
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), f"{path} contains duplicates")


def norm(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def tokens(value):
    return re.findall(r"\w+", value.casefold())


def features(product):
    return sorted({norm(product["category"]), *(norm(t) for t in product["tags"])})


def validate(kind, value, context=None):
    """One validation entry point for input and every stage handoff."""
    if kind == "input":
        fields(value, ("schema_version", "fixture_label", "products", "feedback",
                       "preferences", "search"), path="input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        text(value["fixture_label"], "fixture_label")
        require(isinstance(value["products"], list), "products must be a list")
        product_ids = set()
        for p in value["products"]:
            fields(p, ("id", "name", "description", "category", "tags"), path="product")
            for key in ("id", "name", "description", "category"):
                text(p[key], f"product.{key}")
            strings(p["tags"], "product.tags")
            require(all(norm(t) for t in [p["category"], *p["tags"]]),
                    "product categories/tags must contain word characters")
            require(p["id"] not in product_ids, "duplicate product id")
            product_ids.add(p["id"])
        require(isinstance(value["feedback"], list), "feedback must be a list")
        feedback_ids = set()
        for f in value["feedback"]:
            fields(f, ("id", "customer_id", "product_id", "text", "rating"), path="feedback")
            for key in ("id", "customer_id", "product_id", "text"):
                text(f[key], f"feedback.{key}")
            require(bool(norm(f["text"])), "feedback text must contain word characters")
            number(f["rating"], "feedback.rating", 1, 5)
            require(f["id"] not in feedback_ids, "duplicate feedback id")
            require(f["product_id"] in product_ids, "feedback references unknown product")
            feedback_ids.add(f["id"])
        prefs = value["preferences"]
        fields(prefs, ("interests", "exclude", "limit"), path="preferences")
        require(isinstance(prefs["interests"], dict), "interests must be an object")
        seen = set()
        for term, weight in prefs["interests"].items():
            text(term, "interest")
            require(bool(norm(term)) and norm(term) not in seen, "ambiguous interest")
            seen.add(norm(term))
            number(weight, "interest weight", 0, 10)
        fields(prefs["exclude"], ("product_ids", "categories", "tags"), path="exclude")
        for key, items in prefs["exclude"].items():
            strings(items, f"exclude.{key}")
            if key != "product_ids":
                require(all(norm(item) for item in items), "invalid exclusion term")
        search = value["search"]
        fields(search, ("query", "limit"), path="search")
        require(isinstance(search["query"], str), "search.query must be text")
        for label, limit in (("preferences.limit", prefs["limit"]),
                             ("search.limit", search["limit"])):
            require(type(limit) is int and 1 <= limit <= 100, f"{label} must be integer 1..100")
    elif kind == "feedback":
        fields(value, ("records", "themes", "signals", "duplicates_removed"))
        expected = analyze_feedback(context)
        require(value == expected, "feedback handoff has invalid evidence or aggregation")
    elif kind == "interests":
        fields(value, ("profile", "eligible_product_ids", "recommendations"))
        data, feedback = context
        require(value == rank_interests(data, feedback),
                "interests handoff has invalid ranking, evidence or exclusions")
    elif kind == "search":
        fields(value, ("query", "mode", "index", "results"))
        data, interests = context
        require(value["query"] == data["search"]["query"], "search query mismatch")
        require(value["mode"] in ("lexical", "injected_embedding"), "invalid search mode")
        candidates = [r["product_id"] for r in interests["recommendations"]]
        require(list(value["index"]) == candidates, "search index candidate mismatch")
        require(isinstance(value["results"], list), "search results must be list")
        ids = [r["product_id"] for r in value["results"]]
        require(len(ids) == len(set(ids)) and set(ids) <= set(candidates),
                "search results violate candidate boundary")
        require(len(ids) <= data["search"]["limit"], "too many search results")
        for result in value["results"]:
            fields(result, ("product_id", "score", "lexical_score", "embedding_score",
                            "interest_score", "matched_terms", "explanation"))
            for key in ("score", "lexical_score", "embedding_score", "interest_score"):
                number(result[key], key)
            strings(result["matched_terms"], "matched_terms")
            text(result["explanation"], "explanation")
    else:
        raise ValidationError("unknown validation schema")
    return value


THEMES = {
    "comfort": {"comfortable", "comfort", "soft", "cozy"},
    "durability": {"durable", "durability", "sturdy", "broken", "broke"},
    "sustainability": {"recycled", "sustainable", "sustainability", "reusable"},
    "value": {"value", "affordable", "expensive", "price"},
    "travel": {"travel", "portable", "lightweight"},
}


def analyze_feedback(data):
    """Deduplicate per customer/product/normalized text; preserve all source IDs."""
    groups = {}
    for f in sorted(data["feedback"], key=lambda item: item["id"]):
        key = (f["customer_id"], f["product_id"], norm(f["text"]))
        if key in groups:
            require(groups[key]["rating"] == f["rating"],
                    "duplicate feedback has conflicting ratings")
            groups[key]["source_ids"].append(f["id"])
        else:
            groups[key] = {
                "source_ids": [f["id"]], "customer_id": f["customer_id"],
                "product_id": f["product_id"], "excerpt": f["text"],
                "rating": f["rating"], "sentiment": (f["rating"] - 3) / 2,
                "themes": sorted(t for t, words in THEMES.items()
                                 if set(tokens(f["text"])) & words),
            }
    records = sorted(groups.values(), key=lambda item: item["source_ids"][0])
    products = {p["id"]: p for p in data["products"]}
    themes = {}
    signals = defaultdict(list)
    for record in records:
        for theme in record["themes"]:
            entry = themes.setdefault(theme, {"count": 0, "sentiment_sum": 0, "evidence": []})
            entry["count"] += 1
            entry["sentiment_sum"] += record["sentiment"]
            entry["evidence"].append({"source_ids": record["source_ids"],
                                      "excerpt": record["excerpt"],
                                      "product_id": record["product_id"]})
        for term in sorted(set(features(products[record["product_id"]])) |
                           set(record["themes"])):
            signals[term].append(record)
    return {
        "records": records, "themes": dict(sorted(themes.items())),
        "signals": {term: {
            "weight": sum(r["sentiment"] for r in evidence) / len(evidence),
            "source_ids": sorted({i for r in evidence for i in r["source_ids"]}),
        } for term, evidence in sorted(signals.items())},
        "duplicates_removed": len(data["feedback"]) - len(records),
    }


def rank_interests(data, feedback):
    explicit = {norm(t): w for t, w in data["preferences"]["interests"].items()}
    profile = {}
    for term in sorted(explicit.keys() | feedback["signals"].keys()):
        signal = feedback["signals"].get(term, {"weight": 0, "source_ids": []})
        profile[term] = {"weight": explicit.get(term, 0) + signal["weight"],
                         "explicit_weight": explicit.get(term, 0),
                         "feedback_weight": signal["weight"],
                         "source_ids": signal["source_ids"]}
    exclude = data["preferences"]["exclude"]
    ids = set(exclude["product_ids"])
    cats = {norm(c) for c in exclude["categories"]}
    tags = {norm(t) for t in exclude["tags"]}
    ranked = []
    for p in data["products"]:
        if (p["id"] in ids or norm(p["category"]) in cats or
                {norm(t) for t in p["tags"]} & tags):
            continue
        evidence = [
            {"term": term, **profile[term]} for term in features(p)
            if term in profile and profile[term]["weight"] != 0
        ]
        score = sum(e["weight"] for e in evidence)
        ranked.append({
            "product_id": p["id"], "score": score, "evidence": evidence,
            "explanation": (
                "Matched catalog features: " + "; ".join(
                    f"{e['term']} (explicit={e['explicit_weight']:g}, "
                    f"feedback={e['feedback_weight']:g})" for e in evidence)
                if evidence else "No nonzero preference match; deterministic catalog fallback."
            ),
        })
    ranked.sort(key=lambda r: (-r["score"], r["product_id"]))
    return {"profile": profile, "eligible_product_ids": [r["product_id"] for r in ranked],
            "recommendations": ranked[:data["preferences"]["limit"]]}


def cosine(a, b):
    # Scaling first avoids overflow for valid large finite injected coordinates.
    scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
    if not scale_a or not scale_b:
        return 0.0
    a, b = [x / scale_a for x in a], [x / scale_b for x in b]
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) /
                        (math.sqrt(sum(x*x for x in a)) *
                         math.sqrt(sum(y*y for y in b)))))


def embedding_vectors(embedder, texts):
    require(callable(embedder), "embedder must be callable")
    try:
        vectors = embedder(texts)
    except Exception as exc:
        raise ValidationError("injected embedder failed") from exc
    require(isinstance(vectors, list) and len(vectors) == len(texts),
            "embedder must return one vector per text")
    dimension = None
    for vector in vectors:
        require(isinstance(vector, list) and bool(vector), "embedding must be nonempty list")
        dimension = len(vector) if dimension is None else dimension
        require(len(vector) == dimension, "embedding dimensions differ")
        for coordinate in vector:
            number(coordinate, "embedding coordinate")
    return vectors


def semantic_search(data, interests, embedder=None):
    products = {p["id"]: p for p in data["products"]}
    recommendations = interests["recommendations"]
    index = {}
    documents = []
    for r in recommendations:
        p = products[r["product_id"]]
        document = " ".join([p["name"], p["description"], p["category"], *p["tags"]])
        documents.append(document)
        index[p["id"]] = dict(sorted(Counter(tokens(document)).items()))
    query = data["search"]["query"]
    query_terms = set(tokens(query))
    vectors = None
    if embedder is not None:
        vectors = embedding_vectors(embedder, [query, *documents])
    results = []
    for position, r in enumerate(recommendations):
        counts = index[r["product_id"]]
        matched = sorted(query_terms & counts.keys())
        # IDF rewards rarer query terms; logarithmic TF limits repeated-word dominance.
        lexical = sum(
            (1 + math.log(counts[t])) *
            (1 + math.log((len(index) + 1) /
                          (1 + sum(t in doc for doc in index.values()))))
            for t in matched
        )
        semantic = cosine(vectors[0], vectors[position + 1]) if vectors else 0.0
        score = lexical + semantic
        results.append({
            "product_id": r["product_id"], "score": score,
            "lexical_score": lexical, "embedding_score": semantic,
            "interest_score": r["score"], "matched_terms": matched,
            "explanation": (
                "Query terms found in catalog: " + ", ".join(matched) + ". "
                if matched else "No lexical query match. "
            ) + ("Injected cosine similarity contributes to relevance. " if vectors else "") +
            "Interest score breaks relevance ties; candidate passed all exclusions.",
        })
    results.sort(key=lambda r: (-r["score"], -r["interest_score"], r["product_id"]))
    return {"query": query, "mode": "injected_embedding" if vectors else "lexical",
            "index": index, "results": results[:data["search"]["limit"]]}


def run_pipeline(data, embedder=None):
    validate("input", data)
    feedback = validate("feedback", analyze_feedback(data), data)
    interests = validate("interests", rank_interests(data, feedback), (data, feedback))
    search = validate("search", semantic_search(data, interests, embedder), (data, interests))
    return {"status": "ok", "schema_version": 1, "fixture_label": data["fixture_label"],
            "feedback": feedback, "interests": interests, "search": search}


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
            data = json.load(handle, object_pairs_hook=unique_object)
        output = run_pipeline(data)
        encoded = json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
