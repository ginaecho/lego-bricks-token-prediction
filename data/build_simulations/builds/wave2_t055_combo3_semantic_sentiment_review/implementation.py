"""Synthetic, offline reference pipeline. Python standard library only.

The same versioned envelope enters and leaves every stage. Public stages reject
out-of-order or inconsistent envelopes. An optional local embedder accepts one
list of texts (query first), returning equally sized, nonzero numeric vectors.
No provider, model, network, persistent index, or certification is involved.
"""

import copy
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


class ValidationError(ValueError):
    """An invalid request, stage handoff, or injected embedding result."""


STAGES = {"input": (), "searched": ("search",),
          "scored": ("search", "sentiment"),
          "ok": ("search", "sentiment", "review")}
SYNONYMS = {
    "quick": "fast", "rapid": "fast", "speedy": "fast",
    "silent": "quiet", "noiseless": "quiet",
    "shipping": "delivery", "shipment": "delivery",
    "damaged": "broken", "faulty": "broken",
    "cordless": "wireless", "rechargeable": "battery",
}
LEXICON = {
    "good": 1, "great": 2, "love": 2, "excellent": 2,
    "reliable": 1, "safe": 1, "happy": 1,
    "bad": -1, "broken": -2, "awful": -2, "hate": -2,
    "dangerous": -3, "unsafe": -3, "slow": -1,
    "disappointed": -2, "terrible": -2, "fire": -3,
    "smoke": -2, "burn": -3,
}
SEVERITIES = {"low": 10, "medium": 40, "high": 70, "critical": 100}
DISCLAIMER = (
    "Reference-only textual requirement/evidence check; evidence_found means "
    "required phrases were located, not that claims are true or requirements "
    "are satisfied. This is not certification, compliance approval, or legal advice."
)


def require(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def keys(value, names, path):
    require(isinstance(value, dict), path, "must be an object")
    require(set(value) == set(names), path,
            "expected exactly these keys: " + ", ".join(sorted(names)))


def text(value, path, limit=20000):
    require(isinstance(value, str) and bool(value.strip()), path,
            "must be a nonblank string")
    require(len(value) <= limit, path, f"must not exceed {limit} characters")


def number(value, path, minimum, maximum):
    require(type(value) in (int, float), path, "must be a number, not boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and minimum <= value <= maximum, path,
            f"must be finite and between {minimum} and {maximum}")


def identifier(value, path):
    text(value, path, 80)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is not None,
            path, "must be an ASCII identifier")


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def concepts(value):
    return [SYNONYMS.get(token, token) for token in tokens(value)]


def records(value, fields, path):
    require(isinstance(value, list) and len(value) <= 1000, path,
            "must be an array with at most 1000 records")
    seen = set()
    for i, record in enumerate(value):
        location = f"{path}[{i}]"
        keys(record, fields, location)
        identifier(record["id"], location + ".id")
        require(record["id"] not in seen, location + ".id", "duplicate identifier")
        seen.add(record["id"])
    return seen


def validate_data(data):
    keys(data, ("query", "options", "products", "feedback", "requirements",
                "documents"), "data")
    text(data["query"], "data.query", 1000)
    require(bool(tokens(data["query"])), "data.query", "must contain a word or number")
    options = data["options"]
    keys(options, ("top_k", "min_relevance"), "data.options")
    require(type(options["top_k"]) is int and 1 <= options["top_k"] <= 100,
            "data.options.top_k", "must be an integer from 1 to 100")
    number(options["min_relevance"], "data.options.min_relevance", 0, 1)
    product_ids = records(data["products"], ("id", "name", "description"),
                          "data.products")
    for i, product in enumerate(data["products"]):
        text(product["name"], f"data.products[{i}].name", 200)
        text(product["description"], f"data.products[{i}].description")
    fields = {
        "feedback": ("id", "product_id", "text", "severity"),
        "requirements": ("id", "product_id", "description", "required_terms"),
        "documents": ("id", "product_id", "title", "text"),
    }
    for collection, expected in fields.items():
        records(data[collection], expected, "data." + collection)
        for i, record in enumerate(data[collection]):
            path = f"data.{collection}[{i}]"
            identifier(record["product_id"], path + ".product_id")
            require(record["product_id"] in product_ids, path + ".product_id",
                    "unknown product")
            if collection == "feedback":
                text(record["text"], path + ".text")
                require(isinstance(record["severity"], str)
                        and record["severity"] in SEVERITIES,
                        path + ".severity", "expected low, medium, high, or critical")
            elif collection == "documents":
                text(record["text"], path + ".text")
                text(record["title"], path + ".title", 200)
            else:
                text(record["description"], path + ".description")
                terms = record["required_terms"]
                require(isinstance(terms, list) and 1 <= len(terms) <= 50,
                        path + ".required_terms", "expected 1 to 50 phrases")
                normalized = set()
                for j, term in enumerate(terms):
                    text(term, f"{path}.required_terms[{j}]", 200)
                    phrase = tuple(tokens(term))
                    require(bool(phrase), path + ".required_terms", "empty phrase")
                    require(phrase not in normalized, path + ".required_terms",
                            "duplicate normalized phrase")
                    normalized.add(phrase)


class SearchIndex:
    """In-memory concept-expanded TF-IDF vectors and an inverted search index."""

    def __init__(self, products):
        self.products = sorted(products, key=lambda item: item["id"])
        self.counts = {
            product["id"]: Counter(concepts(product["name"] + " " +
                                             product["description"]))
            for product in self.products
        }
        self.postings = {}
        for product_id, counts in self.counts.items():
            for term in counts:
                self.postings.setdefault(term, set()).add(product_id)
        self.idf = {
            term: 1 + math.log((1 + len(products)) / (1 + len(ids)))
            for term, ids in self.postings.items()
        }

    def lexical_scores(self, query):
        query_counts = Counter(concepts(query))
        unseen_idf = 1 + math.log(1 + len(self.products))
        query_vector = {
            term: count * self.idf.get(term, unseen_idf)
            for term, count in query_counts.items()
        }
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        candidates = set()
        for term in query_counts:
            candidates.update(self.postings.get(term, ()))
        scores = {}
        for product_id in sorted(candidates):
            vector = {term: count * self.idf[term]
                      for term, count in self.counts[product_id].items()}
            norm = math.sqrt(sum(value * value for value in vector.values()))
            dot = sum(query_vector.get(term, 0) * value
                      for term, value in vector.items())
            scores[product_id] = round(min(1.0, dot / (query_norm * norm)), 6)
        return scores


def embedding_scores(embedder, query, products):
    require(callable(embedder), "embedder", "must be callable")
    batch = [query] + [p["name"] + " " + p["description"] for p in products]
    try:
        vectors = embedder(batch)
    except Exception as exc:
        raise ValidationError("embedder: callable failed") from exc
    require(isinstance(vectors, (list, tuple)) and len(vectors) == len(batch),
            "embedder", "must return one vector per input text")
    dimension = None
    unit_vectors = []
    for i, vector in enumerate(vectors):
        path = f"embedder[{i}]"
        require(isinstance(vector, (list, tuple)) and 1 <= len(vector) <= 4096,
                path, "expected a nonempty vector with at most 4096 dimensions")
        if dimension is None:
            dimension = len(vector)
        require(len(vector) == dimension, path, "inconsistent vector dimension")
        for value in vector:
            number(value, path, -1e100, 1e100)
        scale = max(abs(value) for value in vector)
        require(scale > 0, path, "zero vectors are not allowed")
        scaled = [value / scale for value in vector]
        norm = math.sqrt(sum(value * value for value in scaled))
        unit_vectors.append([value / norm for value in scaled])
    query_vector = unit_vectors[0]
    return {
        product["id"]: round(max(0.0, min(1.0, sum(
            a * b for a, b in zip(query_vector, vector)))), 6)
        for product, vector in zip(products, unit_vectors[1:])
    }


def make_search(data, embedder=None):
    index = SearchIndex(data["products"])
    lexical = index.lexical_scores(data["query"])
    embedded = (embedding_scores(embedder, data["query"], index.products)
                if embedder is not None else None)
    query_terms = set(concepts(data["query"]))
    hits = []
    for product in index.products:
        product_id = product["id"]
        lex = lexical.get(product_id, 0.0)
        emb = embedded[product_id] if embedded is not None else None
        score = round(lex if emb is None else 0.65 * lex + 0.35 * emb, 6)
        if score <= 0 or score < data["options"]["min_relevance"]:
            continue
        hits.append({
            "product_id": product_id, "rank": 0, "score": score,
            "lexical_score": lex, "embedding_score": emb,
            "matched_concepts": sorted(query_terms & set(index.counts[product_id])),
            "feedback_ids": sorted(f["id"] for f in data["feedback"]
                                   if f["product_id"] == product_id),
        })
    hits.sort(key=lambda hit: (-hit["score"], hit["product_id"]))
    hits = hits[:data["options"]["top_k"]]
    for rank, hit in enumerate(hits, 1):
        hit["rank"] = rank
    return {
        "method": "concept_tfidf" if embedded is None else "hybrid_injected",
        "index_stats": {"product_count": len(index.products),
                        "concept_count": len(index.postings)},
        "hits": hits,
    }


def sentiment_score(value):
    parts = re.findall(r"[^\W_]+|[.!?,;:]", value.casefold(), flags=re.UNICODE)
    cues, history = [], []
    for position, part in enumerate(parts):
        if part in ".!?,;:" or part in ("but", "however"):
            history = []
            continue
        if part in LEXICON:
            negated = sum(word in ("not", "no", "never")
                          for word in history[-3:]) % 2 == 1
            multiplier = -1 if negated else 1
            cues.append({"term": part, "position": position,
                         "base_weight": LEXICON[part], "multiplier": multiplier,
                         "contribution": LEXICON[part] * multiplier})
        history.append(part)
    total = sum(cue["contribution"] for cue in cues)
    denominator = max(1, sum(abs(cue["contribution"]) for cue in cues))
    score = round(total / denominator, 6)
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    return score, label, cues


def priority_label(score):
    return ("critical" if score >= 100 else "high" if score >= 70
            else "medium" if score >= 40 else "low")


def make_sentiment(data, search):
    feedback = {item["id"]: item for item in data["feedback"]}
    items = []
    for hit in search["hits"]:
        for feedback_id in hit["feedback_ids"]:
            record = feedback[feedback_id]
            score, label, cues = sentiment_score(record["text"])
            priority_score = SEVERITIES[record["severity"]] + round(max(0, -score) * 20)
            items.append({
                "feedback_id": feedback_id, "product_id": hit["product_id"],
                "search_rank": hit["rank"], "relevance": hit["score"],
                "sentiment_score": score, "sentiment_label": label,
                "cues": cues, "severity": record["severity"],
                "priority_score": priority_score,
                "priority": priority_label(priority_score),
            })
    items.sort(key=lambda item: (-item["priority_score"], -item["relevance"],
                                 item["feedback_id"]))
    return {"items": items}


def phrase_evidence(term, document):
    words = list(re.finditer(r"[^\W_]+", document["text"], flags=re.UNICODE))
    wanted = tokens(term)
    for i in range(len(words) - len(wanted) + 1):
        window = words[i:i + len(wanted)]
        if [word.group().casefold() for word in window] == wanted:
            start, end = window[0].start(), window[-1].end()
            return {"term": term, "document_id": document["id"], "start": start,
                    "end": end, "quote": document["text"][start:end]}
    return None


def make_review(data, search, sentiment):
    selected = {hit["product_id"] for hit in search["hits"]}
    checks = []
    for requirement in data["requirements"]:
        product_id = requirement["product_id"]
        if product_id not in selected:
            continue
        related = [item for item in sentiment["items"]
                   if item["product_id"] == product_id]
        priority = max((item["priority_score"] for item in related), default=0)
        documents = sorted((doc for doc in data["documents"]
                            if doc["product_id"] == product_id), key=lambda doc: doc["id"])
        evidence, missing = [], []
        for term in requirement["required_terms"]:
            matches = [found for doc in documents
                       if (found := phrase_evidence(term, doc)) is not None]
            if matches:
                evidence.extend(matches)
            else:
                missing.append(term)
        checks.append({
            "requirement_id": requirement["id"], "product_id": product_id,
            "status": "gap" if missing else "evidence_found",
            "evidence": evidence, "missing_terms": missing,
            "feedback_ids": [item["feedback_id"] for item in related],
            "priority_score": priority, "priority": priority_label(priority),
        })
    checks.sort(key=lambda check: (-check["priority_score"], check["requirement_id"]))
    gaps = [
        {key: check[key] for key in ("requirement_id", "product_id", "missing_terms",
                                     "feedback_ids", "priority_score", "priority")}
        for check in checks if check["status"] == "gap"
    ]
    return {"checks": checks, "gaps": gaps,
            "unreviewed_requirement_ids": sorted(
                req["id"] for req in data["requirements"]
                if req["product_id"] not in selected),
            "disclaimer": DISCLAIMER}


def same_json(actual, expected, path):
    # Canonical JSON comparison also distinguishes booleans from numeric values.
    try:
        actual_json = json.dumps(actual, sort_keys=True, allow_nan=False)
        expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValidationError(f"{path}: must be finite JSON data") from exc
    require(actual_json == expected_json, path,
            "does not match the validated upstream data or deterministic rules")


def validate_search(data, search):
    keys(search, ("method", "index_stats", "hits"), "results.search")
    require(isinstance(search["method"], str)
            and search["method"] in ("concept_tfidf", "hybrid_injected"),
            "results.search.method", "unknown method")
    if search["method"] == "concept_tfidf":
        same_json(search, make_search(data), "results.search")
        return
    index = SearchIndex(data["products"])
    lexical = index.lexical_scores(data["query"])
    same_json(search["index_stats"], {"product_count": len(index.products),
                                     "concept_count": len(index.postings)},
              "results.search.index_stats")
    require(isinstance(search["hits"], list)
            and len(search["hits"]) <= data["options"]["top_k"],
            "results.search.hits", "must be an array no longer than top_k")
    seen = set()
    for rank, hit in enumerate(search["hits"], 1):
        path = f"results.search.hits[{rank - 1}]"
        keys(hit, ("product_id", "rank", "score", "lexical_score", "embedding_score",
                   "matched_concepts", "feedback_ids"), path)
        identifier(hit["product_id"], path + ".product_id")
        product_id = hit["product_id"]
        require(product_id in index.counts and product_id not in seen,
                path + ".product_id", "unknown or duplicate product")
        seen.add(product_id)
        require(type(hit["rank"]) is int and hit["rank"] == rank, path, "invalid rank")
        for key in ("score", "lexical_score", "embedding_score"):
            number(hit[key], path + "." + key, 0, 1)
        require(hit["score"] > 0 and hit["score"] >= data["options"]["min_relevance"],
                path + ".score", "below relevance threshold")
        require(hit["lexical_score"] == lexical.get(product_id, 0.0), path,
                "incorrect lexical score")
        require(hit["score"] == round(0.65 * hit["lexical_score"]
                                     + 0.35 * hit["embedding_score"], 6),
                path, "incorrect hybrid score")
        same_json(hit["matched_concepts"],
                  sorted(set(concepts(data["query"])) & set(index.counts[product_id])),
                  path + ".matched_concepts")
        same_json(hit["feedback_ids"], sorted(f["id"] for f in data["feedback"]
                                             if f["product_id"] == product_id),
                  path + ".feedback_ids")
    require(search["hits"] == sorted(search["hits"],
                                    key=lambda hit: (-hit["score"], hit["product_id"])),
            "results.search.hits", "invalid ranking order")


def validate_envelope(envelope, expected_status=None):
    keys(envelope, ("schema_version", "fixture_label", "status", "data", "results"),
         "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "schema_version", "expected integer 1")
    text(envelope["fixture_label"], "fixture_label", 200)
    require(envelope["fixture_label"].startswith("SYNTHETIC:"), "fixture_label",
            "must start with SYNTHETIC:")
    status = envelope["status"]
    require(isinstance(status, str) and status in STAGES, "status", "unknown stage")
    if expected_status is not None:
        require(status == expected_status, "status", f"expected {expected_status}")
    validate_data(envelope["data"])
    keys(envelope["results"], STAGES[status], "results")
    results, data = envelope["results"], envelope["data"]
    if "search" in results:
        validate_search(data, results["search"])
    if "sentiment" in results:
        same_json(results["sentiment"], make_sentiment(data, results["search"]),
                  "results.sentiment")
    if "review" in results:
        same_json(results["review"], make_review(data, results["search"],
                                                results["sentiment"]),
                  "results.review")
    return envelope


def semantic_stage(envelope, embedder=None):
    validate_envelope(envelope, "input")
    output = copy.deepcopy(envelope)
    output["results"]["search"] = make_search(output["data"], embedder)
    output["status"] = "searched"
    return validate_envelope(output, "searched")


def sentiment_stage(envelope):
    validate_envelope(envelope, "searched")
    output = copy.deepcopy(envelope)
    output["results"]["sentiment"] = make_sentiment(
        output["data"], output["results"]["search"])
    output["status"] = "scored"
    return validate_envelope(output, "scored")


def review_stage(envelope):
    validate_envelope(envelope, "scored")
    output = copy.deepcopy(envelope)
    output["results"]["review"] = make_review(
        output["data"], output["results"]["search"], output["results"]["sentiment"])
    output["status"] = "ok"
    return validate_envelope(output, "ok")


def run_pipeline(envelope, embedder=None):
    return review_stage(sentiment_stage(semantic_stage(envelope, embedder)))


def reject_constant(value):
    raise ValidationError(f"JSON: nonfinite constant {value} is not allowed")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON", f"duplicate object key {key!r}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "CLI", "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("rb") as handle:
            raw = handle.read(2_000_001)
        require(len(raw) <= 2_000_000, "file", "maximum input size is 2 MB")
        envelope = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant,
                              object_pairs_hook=unique_object)
        output = run_pipeline(envelope)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": {
            "type": "validation_or_file_error", "message": str(exc)}}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
