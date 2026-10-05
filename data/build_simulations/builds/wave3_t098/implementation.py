"""Bounded deterministic product retrieval; Python API may inject batch embeddings.

Run: python -B implementation.py example_input.json
Embedding contract: callable(list[str]) -> list[list[finite number]], one vector
per input, consistent dimension, nonzero norm. No provider is bundled or called.
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name, maximum):
    require(isinstance(value, str) and bool(value.strip()),
            f"{name} must be a nonempty string")
    require(len(value) <= maximum, f"{name} exceeds {maximum} characters")
    return value.strip()


def number(value, name, low, high):
    require(type(value) in (int, float), f"{name} must be a number")
    require(low <= value <= high, f"{name} must be between {low} and {high}")
    return value


def tokenize(value):
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def validate(payload):
    require(isinstance(payload, dict), "input must be an object")
    require(set(payload) <= {"dataset_label", "products", "query", "limit",
                             "semantic_weight"}, "unknown input field")
    label = text(payload.get("dataset_label"), "dataset_label", 200)
    query = text(payload.get("query"), "query", 500)
    require(bool(tokenize(query)), "query must contain at least one word")
    limit = payload.get("limit", 5)
    require(type(limit) is int and 1 <= limit <= 100,
            "limit must be an integer between 1 and 100")
    weight = number(payload.get("semantic_weight", 0), "semantic_weight", 0, 1)
    products = payload.get("products")
    require(isinstance(products, list) and len(products) <= 1000,
            "products must be a list of at most 1000 products")
    cleaned, seen = [], set()
    for index, product in enumerate(products):
        prefix = f"products[{index}]"
        require(isinstance(product, dict), f"{prefix} must be an object")
        require(set(product) <= {"id", "title", "description", "tags"},
                f"{prefix} has unknown fields")
        identifier = text(product.get("id"), f"{prefix}.id", 100)
        require(identifier not in seen, f"duplicate product id: {identifier}")
        seen.add(identifier)
        title = text(product.get("title"), f"{prefix}.title", 300)
        description = product.get("description", "")
        require(isinstance(description, str) and len(description) <= 5000,
                f"{prefix}.description must be a string of at most 5000 characters")
        tags = product.get("tags", [])
        require(isinstance(tags, list) and len(tags) <= 30,
                f"{prefix}.tags must be a list of at most 30 strings")
        tags = [text(tag, f"{prefix}.tags", 100) for tag in tags]
        cleaned.append({"id": identifier, "title": title,
                        "description": description.strip(), "tags": tags})
    return {"dataset_label": label, "query": query, "products": cleaned,
            "limit": limit, "semantic_weight": weight}


class SearchIndex:
    """In-memory inverted index with BM25 (k1=1.2, b=0.75)."""

    def __init__(self, products):
        self.products = sorted(products, key=lambda product: product["id"])
        self.documents = [" ".join([p["title"], p["description"], *p["tags"]])
                          for p in self.products]
        self.lengths = []
        self.postings = defaultdict(dict)
        for index, document in enumerate(self.documents):
            frequencies = Counter(tokenize(document))
            self.lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                self.postings[term][index] = frequency
        self.average_length = sum(self.lengths) / max(1, len(products))

    def lexical_scores(self, query):
        scores = [0.0] * len(self.products)
        matched = [set() for _ in self.products]
        count = len(self.products)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, {})
            inverse_frequency = math.log1p(
                (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings.items():
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * self.lengths[index] / self.average_length)
                scores[index] += inverse_frequency * frequency * 2.2 / denominator
                matched[index].add(term)
        return [score / (1 + score) for score in scores], matched


def embedding_scores(embedder, texts):
    require(callable(embedder), "a callable embedder is required for semantic_weight > 0")
    try:
        vectors = embedder(list(texts))
    except Exception as error:
        raise ValidationError("embedding callable failed") from error
    require(isinstance(vectors, list) and len(vectors) == len(texts),
            "embedder must return one vector per text")
    normalized, dimension = [], None
    for vector in vectors:
        require(isinstance(vector, list) and 1 <= len(vector) <= 4096,
                "embedding vectors must be nonempty lists of at most 4096 numbers")
        if dimension is None:
            dimension = len(vector)
        require(len(vector) == dimension, "embedding dimensions must match")
        for coordinate in vector:
            require(type(coordinate) in (int, float),
                    "embedding coordinates must be finite numbers")
            try:
                finite = math.isfinite(coordinate)
            except OverflowError:
                finite = False
            require(finite, "embedding coordinates must be finite numbers")
        scale = max(abs(coordinate) for coordinate in vector)
        require(scale > 0, "embedding vectors must have nonzero norm")
        # Scale first so finite, very large coordinates cannot overflow the norm.
        scaled = [coordinate / scale for coordinate in vector]
        norm = math.sqrt(sum(coordinate * coordinate for coordinate in scaled))
        normalized.append([coordinate / norm for coordinate in scaled])
    query = normalized[0]
    return [max(0.0, min(1.0, math.fsum(a * b for a, b in zip(query, vector))))
            for vector in normalized[1:]]


def search(payload, embedder=None):
    """Return the shared response envelope, including validation failures."""
    try:
        request = validate(payload)
        weight = request["semantic_weight"]
        if weight:
            require(callable(embedder),
                    "a callable embedder is required for semantic_weight > 0")
        index = SearchIndex(request["products"])
        lexical, matched = index.lexical_scores(request["query"])
        semantic = [0.0] * len(index.products)
        if weight and index.products:
            semantic = embedding_scores(embedder, [request["query"], *index.documents])
        matches = []
        for i, product in enumerate(index.products):
            score = (1 - weight) * lexical[i] + weight * semantic[i]
            if score > 0:
                matches.append({"id": product["id"], "title": product["title"],
                                "score": round(score, 6),
                                "lexical_score": round(lexical[i], 6),
                                "semantic_score": round(semantic[i], 6),
                                "matched_terms": sorted(matched[i])})
        matches.sort(key=lambda match: (-match["score"], match["id"]))
        return {"status": "ok", "dataset_label": request["dataset_label"],
                "query": request["query"], "matches": matches[:request["limit"]],
                "total_matches": len(matches),
                "index": {"product_count": len(index.products),
                          "term_count": len(index.postings)},
                "ranking": {"method": "bm25+cosine" if weight else "bm25",
                            "semantic_weight": weight}}
    except ValidationError as error:
        return {"status": "error", "error": {"code": "validation_error",
                                           "message": str(error)}}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        require(len(arguments) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(arguments[0]).open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        response = search(payload)
    except (OSError, UnicodeError) as error:
        response = {"status": "error", "error": {"code": "file_error",
                                                "message": str(error)}}
    except (ValueError, RecursionError) as error:
        response = {"status": "error", "error": {"code": "validation_error",
                                                "message": str(error)}}
    print(json.dumps(response, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0 if response["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
