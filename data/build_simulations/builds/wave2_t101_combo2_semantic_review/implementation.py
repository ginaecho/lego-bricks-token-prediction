"""Synthetic, deterministic semantic-search -> evidence-review reference pipeline."""

import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    return value


def fields(value, required, optional, path):
    require(isinstance(value, dict), path + " must be an object")
    require(required <= value.keys(), path + " missing required fields")
    require(value.keys() <= required | optional, path + " has unknown fields")


def number(value, path):
    require(type(value) in (int, float), path + " must be numeric")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid, path + " must be finite")
    return value


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def validate_input(data):
    fields(data, {"schema_version", "synthetic", "query", "products", "requirements"},
           {"limit", "min_score"}, "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "synthetic must be true")
    require(bool(tokens(text(data["query"], "query"))), "query needs searchable tokens")
    require(type(data.get("limit", 5)) is int and 1 <= data.get("limit", 5) <= 100,
            "limit must be an integer in [1,100]")
    require(0 <= number(data.get("min_score", 0.1), "min_score") <= 1,
            "min_score must be in [0,1]")
    require(isinstance(data["products"], list), "products must be an array")
    product_ids, document_ids = set(), set()
    for product in data["products"]:
        fields(product, {"id", "title", "description", "documents"}, set(), "product")
        pid = text(product["id"], "product.id")
        require(pid not in product_ids, "duplicate product id")
        product_ids.add(pid)
        text(product["title"], "product.title")
        text(product["description"], "product.description")
        require(isinstance(product["documents"], list), "documents must be an array")
        for document in product["documents"]:
            fields(document, {"id", "text"}, set(), "document")
            did = text(document["id"], "document.id")
            require(did not in document_ids, "duplicate document id")
            document_ids.add(did)
            text(document["text"], "document.text")
    require(isinstance(data["requirements"], list) and data["requirements"],
            "requirements must be a nonempty array")
    requirement_ids = set()
    for requirement in data["requirements"]:
        fields(requirement, {"id", "description", "terms"}, set(), "requirement")
        rid = text(requirement["id"], "requirement.id")
        require(rid not in requirement_ids, "duplicate requirement id")
        requirement_ids.add(rid)
        text(requirement["description"], "requirement.description")
        terms = requirement["terms"]
        require(isinstance(terms, list) and terms, "terms must be a nonempty array")
        normalized = []
        for term in terms:
            parts = tokens(text(term, "requirement.term"))
            require(bool(parts), "requirement term needs searchable tokens")
            normalized.append(tuple(parts))
        require(len(set(normalized)) == len(normalized), "duplicate normalized terms")
    return data


def build_index(data):
    """Index titles/descriptions/documents, retaining document provenance."""
    return [
        {"product_id": p["id"],
         "text": " ".join([p["title"], p["description"]] +
                          [d["text"] for d in p["documents"]]),
         "document_ids": [d["id"] for d in p["documents"]]}
        for p in data["products"]
    ]


def validate_vectors(vectors, count):
    require(isinstance(vectors, list) and len(vectors) == count,
            "embedding output must contain one vector per text")
    width = None
    for vector in vectors:
        require(isinstance(vector, list) and vector, "embedding vector must be a nonempty list")
        width = len(vector) if width is None else width
        require(len(vector) == width, "embedding dimensions must agree")
        for value in vector:
            number(value, "embedding coordinate")
        require(any(v != 0 for v in vector), "embedding vector cannot be zero")
    # Scale before normalization to avoid overflow on finite large coordinates.
    result = []
    for vector in vectors:
        scale = max(abs(v) for v in vector)
        scaled = [v / scale for v in vector]
        norm = math.sqrt(sum(v * v for v in scaled))
        result.append([v / norm for v in scaled])
    return result


def validate_handoff(data, search):
    fields(search, {"method", "hits"}, set(), "search")
    require(search["method"] in ("lexical", "injected_embedding"), "invalid search method")
    require(isinstance(search["hits"], list), "search hits must be an array")
    require(len(search["hits"]) <= data.get("limit", 5), "too many hits")
    products = {p["id"]: p for p in data["products"]}
    seen, order = set(), []
    for rank, hit in enumerate(search["hits"], 1):
        fields(hit, {"rank", "product_id", "score", "document_ids"}, set(), "hit")
        pid = text(hit["product_id"], "hit.product_id")
        require(pid in products and pid not in seen, "unknown or duplicate hit product")
        seen.add(pid)
        require(type(hit["rank"]) is int and hit["rank"] == rank, "invalid hit rank")
        score = number(hit["score"], "hit.score")
        require(0 < score <= 1 and score >= data.get("min_score", 0.1), "invalid hit score")
        require(hit["document_ids"] == [d["id"] for d in products[pid]["documents"]],
                "hit document provenance mismatch")
        order.append((-score, pid))
    require(order == sorted(order), "hits must be ranked by score then product id")
    return search


def semantic_search(data, embedding=None):
    validate_input(data)
    index = build_index(data)
    query = set(tokens(data["query"]))
    method = "lexical"
    if embedding is None:
        # Query-token coverage is the transparent, no-model baseline.
        scores = [len(query & set(tokens(item["text"]))) / len(query) for item in index]
    else:
        require(callable(embedding), "embedding must be callable")
        method = "injected_embedding"
        try:
            vectors = embedding([data["query"]] + [item["text"] for item in index])
        except Exception as exc:
            raise ValidationError("embedding callable failed") from exc
        vectors = validate_vectors(vectors, len(index) + 1)
        scores = [max(0.0, min(1.0, sum(a * b for a, b in zip(vectors[0], v))))
                  for v in vectors[1:]]
    hits = []
    for item, score in zip(index, scores):
        if score > 0 and score >= data.get("min_score", 0.1):
            hits.append({"product_id": item["product_id"], "score": score,
                         "document_ids": item["document_ids"]})
    hits.sort(key=lambda hit: (-hit["score"], hit["product_id"]))
    hits = hits[:data.get("limit", 5)]
    for rank, hit in enumerate(hits, 1):
        hit["rank"] = rank
    return validate_handoff(data, {"method": method, "hits": hits})


def contains_phrase(document, term):
    haystack, needle = tokens(document), tokens(term)
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def review_documents(data, search):
    validate_input(data)
    validate_handoff(data, search)
    products = {p["id"]: p for p in data["products"]}
    checks = []
    for requirement in data["requirements"]:
        evidence, partials, examined = [], [], []
        for hit in search["hits"]:
            for document in products[hit["product_id"]]["documents"]:
                reference = {"product_id": hit["product_id"], "document_id": document["id"]}
                examined.append(reference)
                matched = [term for term in requirement["terms"]
                           if contains_phrase(document["text"], term)]
                missing = [term for term in requirement["terms"] if term not in matched]
                trace = {**reference, "search_rank": hit["rank"], "search_score": hit["score"],
                         "matched_terms": matched, "missing_terms": missing,
                         "excerpt": document["text"]}
                if not missing:
                    evidence.append(trace)
                else:
                    partials.append(trace)
        checks.append({
            "requirement_id": requirement["id"],
            "status": "evidence_found" if evidence else "gap",
            "evidence": evidence, "partial_evidence": partials,
            "examined_documents": examined,
            "gap_reason": None if evidence else (
                "No retrieved documents" if not examined
                else "No single retrieved document contains all required terms"),
        })
    return {"checks": checks,
            "evidence_found": sum(c["status"] == "evidence_found" for c in checks),
            "gaps": sum(c["status"] == "gap" for c in checks),
            "notice": "Synthetic lexical evidence checks only; not certification, "
                      "compliance determination, or verification of factual truth."}


def run_pipeline(data, embedding=None):
    data = validate_input(data)
    search = semantic_search(data, embedding)
    review = review_documents(data, search)
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "query": data["query"], "search": search, "review": review}


def reject_constant(value):
    raise ValidationError("nonstandard JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open(encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
