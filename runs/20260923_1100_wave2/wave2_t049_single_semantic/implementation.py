"""Deterministic reference product search; Python standard library only.

Embedding injection: search(payload, embedder=callable). The callable receives
[query, product_document, ...] and returns equally sized finite numeric vectors.
No provider is configured or called by the CLI. See build_manifest.json for schema.
"""

import json
import math
import re
import sys
from collections import Counter


class ValidationError(ValueError):
    pass


SYNONYMS = (
    {"shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers"},
    {"sofa", "sofas", "couch", "couches"},
    {"wireless", "cordless"},
    {"headphone", "headphones", "headset", "headsets"},
    {"bike", "bikes", "bicycle", "bicycles"},
    {"laptop", "laptops", "notebook", "notebooks"},
)
EXPANSIONS = {word: group for group in SYNONYMS for word in group}


def tokens(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name, maximum=10000):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    require(len(value) <= maximum, name + " is too long")
    return value


def number(value, name, low, high):
    require(type(value) in (int, float), name + " must be a number")
    require(low <= value <= high, name + " is outside its allowed range")
    return value


def validate_input(payload):
    require(isinstance(payload, dict), "input must be an object")
    require(set(payload) <= {"schema_version", "dataset_label", "query", "products", "options"},
            "unknown input field")
    require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1,
            "schema_version must be 1")
    label = text(payload.get("dataset_label"), "dataset_label", 200)
    query = text(payload.get("query"), "query", 2000)
    products = payload.get("products")
    require(isinstance(products, list) and len(products) <= 10000,
            "products must be a list of at most 10000 products")
    seen = set()
    normalized = []
    for i, product in enumerate(products):
        prefix = "products[" + str(i) + "]"
        require(isinstance(product, dict), prefix + " must be an object")
        require(set(product) <= {"id", "title", "description", "category", "tags"},
                prefix + " contains an unknown field")
        identity = text(product.get("id"), prefix + ".id", 200)
        require(identity not in seen, "duplicate product id: " + identity)
        seen.add(identity)
        title = text(product.get("title"), prefix + ".title", 1000)
        description = product.get("description", "")
        category = product.get("category", "")
        require(isinstance(description, str) and len(description) <= 10000,
                prefix + ".description must be text of at most 10000 characters")
        require(isinstance(category, str) and len(category) <= 1000,
                prefix + ".category must be text of at most 1000 characters")
        tags = product.get("tags", [])
        require(isinstance(tags, list) and len(tags) <= 100, prefix + ".tags must be a list")
        for tag in tags:
            text(tag, prefix + ".tags entry", 200)
        normalized.append({"id": identity, "title": title, "description": description,
                           "category": category, "tags": list(tags)})
    options = payload.get("options", {})
    require(isinstance(options, dict), "options must be an object")
    require(set(options) <= {"limit", "min_score", "embedding_weight"}, "unknown option")
    limit = options.get("limit", 10)
    require(type(limit) is int and 1 <= limit <= 100, "limit must be an integer from 1 to 100")
    minimum = number(options.get("min_score", 0), "min_score", 0, 1)
    weight = number(options.get("embedding_weight", 0.35), "embedding_weight", 0, 1)
    return {"schema_version": 1, "dataset_label": label, "query": query,
            "products": normalized, "options": {"limit": limit, "min_score": minimum,
                                               "embedding_weight": weight}}


def document(product):
    return " ".join([product["title"], product["title"], product["category"],
                     " ".join(product["tags"]), product["description"]])


class SearchIndex:
    """In-memory inverted index with title boosting and BM25 term scoring."""

    def __init__(self, products):
        self.documents = [document(product) for product in products]
        self.lengths = []
        self.postings = {}
        for index, doc in enumerate(self.documents):
            counts = Counter(tokens(doc))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings.setdefault(term, {})[index] = count
        self.average_length = sum(self.lengths) / max(1, len(products)) or 1

    def rank(self, query):
        weights = {}
        for term in set(tokens(query)):
            for expanded in EXPANSIONS.get(term, {term}):
                weights[expanded] = max(weights.get(expanded, 0),
                                        1.0 if expanded == term else 0.65)
        scores = [0.0] * len(self.documents)
        matches = [set() for _ in scores]
        for term, weight in sorted(weights.items()):
            postings = self.postings.get(term, {})
            idf = math.log(1 + (len(scores) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings.items():
                norm = 1.2 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                scores[index] += weight * idf * frequency * 2.2 / (frequency + norm)
                matches[index].add(term)
        return [1 - math.exp(-score) for score in scores], matches


def embedding_scores(embedder, texts):
    require(callable(embedder), "embedder must be callable")
    try:
        vectors = embedder(list(texts))
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(vectors, (list, tuple)) and len(vectors) == len(texts),
            "embedder must return one vector per text")
    dimension = None
    units = []
    for vector in vectors:
        require(isinstance(vector, (list, tuple)) and 0 < len(vector) <= 4096,
                "embedding vectors must have 1 to 4096 dimensions")
        if dimension is None:
            dimension = len(vector)
        require(len(vector) == dimension, "embedding dimensions must agree")
        for component in vector:
            number(component, "embedding component", -1e100, 1e100)
        norm = math.hypot(*vector)
        require(norm > 0, "embedding vectors must be nonzero")
        units.append([component / norm for component in vector])
    return [max(0.0, min(1.0, math.fsum(a * b for a, b in zip(units[0], unit))))
            for unit in units[1:]]


def search(payload, embedder=None):
    data = validate_input(payload)
    index = SearchIndex(data["products"])
    lexical, matches = index.rank(data["query"])
    semantic = [0.0] * len(lexical)
    if embedder is not None:
        semantic = embedding_scores(embedder, [data["query"]] + index.documents)
    weight = data["options"]["embedding_weight"] if embedder is not None else 0
    ranked = []
    for i, product in enumerate(data["products"]):
        score = (1 - weight) * lexical[i] + weight * semantic[i]
        if score <= 0 or score < data["options"]["min_score"]:
            continue
        ranked.append((score, product["id"], {
            "id": product["id"], "title": product["title"],
            "score": round(score, 8), "lexical_score": round(lexical[i], 8),
            "embedding_score": round(semantic[i], 8), "matched_terms": sorted(matches[i]),
        }))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {"status": "ok", "schema_version": 1, "dataset_label": data["dataset_label"],
            "query": data["query"], "indexed_count": len(data["products"]),
            "matched_count": len(ranked), "embedding_used": embedder is not None,
            "results": [item[2] for item in ranked[:data["options"]["limit"]]]}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = search(payload)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
