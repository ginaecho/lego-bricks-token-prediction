"""Synthetic marketplace reference pipeline; Python standard library only.

Run: python -B implementation.py example_input.json
Public injection seam: run_pipeline(document, embedding=callable).
The callable receives [query, product_text, ...] and returns finite, nonzero,
equal-dimensional numeric vectors in that order. It is never loaded from JSON.
"""

import copy
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone


SCHEMA_VERSION = 1
THEMES = {
    "quality": {"quality", "durable", "sturdy", "broken", "flimsy"},
    "comfort": {"comfort", "comfortable", "uncomfortable", "soft"},
    "value": {"price", "affordable", "expensive", "cheap", "value"},
    "delivery": {"delivery", "shipping", "late", "arrived"},
}
POSITIVE = {"great", "good", "love", "excellent", "durable", "sturdy",
            "comfortable", "soft", "affordable", "fast", "perfect"}
NEGATIVE = {"bad", "hate", "broken", "flimsy", "uncomfortable", "expensive",
            "late", "poor", "terrible"}
SYNONYMS = {"couch": "sofa", "couches": "sofa", "trainers": "sneaker",
            "sneakers": "sneaker", "inexpensive": "affordable"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def tokens(text, expand=False):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return [SYNONYMS.get(word, word) for word in words] if expand else words


def timestamp(value):
    require(isinstance(value, str), "timestamp must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO 8601 timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "timestamps must include a timezone")
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValidationError("timestamp outside supported UTC range") from exc


def number(value, label, minimum=None, maximum=None):
    require(type(value) in (int, float), label + " must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, label + " must be finite")
    require(minimum is None or value >= minimum, label + " below minimum")
    require(maximum is None or value <= maximum, label + " above maximum")


def text(value, label, allow_empty=False):
    require(isinstance(value, str), label + " must be a string")
    require(allow_empty or bool(value.strip()), label + " must not be empty")


def fields(value, required, optional=()):
    require(isinstance(value, dict), "expected object")
    require(set(required) <= value.keys(), "missing fields: " +
            ", ".join(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "unknown fields")


class Validator:
    """The sole validation layer for input and all cross-stage contracts."""

    @staticmethod
    def validate(stage, value, context=None):
        require(isinstance(value, dict), stage + " must be an object")
        if stage == "input":
            Validator.input(value)
        elif stage == "feedback":
            Validator.feedback(value, context)
        elif stage == "behavior":
            Validator.behavior(value, context)
        elif stage == "semantic":
            Validator.semantic(value, context)
        else:
            raise ValidationError("unknown validation stage")
        return value

    @staticmethod
    def input(data):
        fields(data, ("schema_version", "synthetic", "as_of", "products",
                      "feedback", "interactions", "request"), ("config",))
        require(type(data["schema_version"]) is int and
                data["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
        require(data["synthetic"] is True, "reference data must be labeled synthetic")
        as_of = timestamp(data["as_of"])
        for collection in ("products", "feedback", "interactions"):
            require(isinstance(data[collection], list), collection + " must be an array")
            identifiers = set()
            for row in data[collection]:
                require(isinstance(row, dict), "array entries must be objects")
                text(row.get("id"), collection + ".id")
                require(row["id"] not in identifiers, "duplicate " + collection + " id")
                identifiers.add(row["id"])
        product_ids = {p["id"] for p in data["products"]}
        for product in data["products"]:
            fields(product, ("id", "title", "description", "tags"))
            text(product["title"], "title")
            text(product["description"], "description", allow_empty=True)
            require(isinstance(product["tags"], list), "tags must be an array")
            for tag in product["tags"]:
                text(tag, "tag")
        for kind in ("feedback", "interactions"):
            for row in data[kind]:
                extra = "text" if kind == "feedback" else "action"
                fields(row, ("id", "customer_id", "product_id", "timestamp", extra))
                text(row["customer_id"], "customer_id")
                text(row["product_id"], "product_id")
                require(row["product_id"] in product_ids, "unknown product reference")
                require(timestamp(row["timestamp"]) <= as_of, "future event")
                if kind == "feedback":
                    text(row["text"], "feedback.text")
                    require(bool(tokens(row["text"])), "feedback must contain words")
                else:
                    require(row["action"] in ("browse", "purchase"), "unknown action")
        request = data["request"]
        fields(request, ("customer_id", "query", "limit"))
        text(request["customer_id"], "request.customer_id")
        text(request["query"], "request.query", allow_empty=True)
        require(type(request["limit"]) is int and 1 <= request["limit"] <= 100,
                "limit must be an integer in [1, 100]")
        config = data.get("config", {})
        fields(config, (), ("half_life_days", "feedback_weight", "personalization_weight"))
        number(config.get("half_life_days", 30), "half_life_days", 0.001, 36500)
        number(config.get("feedback_weight", 0.5), "feedback_weight", 0, 10)
        number(config.get("personalization_weight", 0.2), "personalization_weight", 0, 1)

    @staticmethod
    def feedback(result, data):
        fields(result, ("input_count", "unique_count", "duplicate_count",
                        "items", "themes", "product_signals"))
        originals = {item["id"]: item for item in data["feedback"]}
        require(result["input_count"] == len(originals), "feedback count mismatch")
        require(result["unique_count"] == len(result["items"]), "unique count mismatch")
        require(result["duplicate_count"] == len(originals) - len(result["items"]),
                "duplicate count mismatch")
        seen = []
        item_ids = set()
        for item in result["items"]:
            fields(item, ("id", "customer_id", "product_id", "source_ids",
                          "excerpt", "sentiment", "themes"))
            require(item["id"] in originals and item["id"] not in item_ids,
                    "invalid deduplicated id")
            item_ids.add(item["id"])
            source = originals[item["id"]]
            require(item["source_ids"] and item["id"] in item["source_ids"],
                    "missing representative source")
            require(item["excerpt"] == source["text"], "excerpt must be verbatim")
            require(item["product_id"] == source["product_id"] and
                    item["customer_id"] == source["customer_id"], "feedback reference mismatch")
            for source_id in item["source_ids"]:
                require(source_id in originals, "unknown supporting feedback")
                other = originals[source_id]
                require(other["product_id"] == item["product_id"] and
                        other["customer_id"] == item["customer_id"] and
                        tokens(other["text"]) == tokens(item["excerpt"]),
                        "invalid duplicate grouping")
            seen.extend(item["source_ids"])
            number(item["sentiment"], "sentiment", -1, 1)
            require(set(item["themes"]) <= set(THEMES) | {"other"}, "unknown theme")
        require(sorted(seen) == sorted(originals), "feedback sources not covered exactly once")
        by_id = {item["id"]: item for item in result["items"]}
        expected_themes = {theme for item in result["items"] for theme in item["themes"]}
        require(set(result["themes"]) == expected_themes, "theme coverage mismatch")
        for theme, evidence in result["themes"].items():
            expected = {item["id"] for item in result["items"] if theme in item["themes"]}
            require(len(evidence) == len(expected) and
                    {item["feedback_id"] for item in evidence} == expected,
                    "theme evidence mismatch")
            for entry in evidence:
                source = by_id[entry["feedback_id"]]
                require(entry["excerpt"] == source["excerpt"] and
                        entry["product_id"] == source["product_id"], "untraceable excerpt")
        require(set(result["product_signals"]) == {p["id"] for p in data["products"]},
                "product signal coverage mismatch")
        for product_id, signal in result["product_signals"].items():
            fields(signal, ("sentiment", "feedback_ids", "themes"))
            number(signal["sentiment"], "product sentiment", -1, 1)
            expected = {item["id"] for item in result["items"]
                        if item["product_id"] == product_id}
            require(len(signal["feedback_ids"]) == len(expected) and
                    set(signal["feedback_ids"]) == expected, "signal evidence mismatch")
            require(set(signal["themes"]) == {theme for item_id in expected
                                             for theme in by_id[item_id]["themes"]},
                    "signal themes mismatch")

    @staticmethod
    def behavior(result, context):
        data, feedback = context
        fields(result, ("cold_start", "ranking", "half_life_days"))
        require(type(result["cold_start"]) is bool, "cold_start must be boolean")
        number(result["half_life_days"], "half_life_days", 0.001, 36500)
        ids = [row["product_id"] for row in result["ranking"]]
        require(sorted(ids) == sorted(p["id"] for p in data["products"]),
                "behavior must rank each product once")
        allowed_interactions = {row["id"]: row for row in data["interactions"]}
        for row in result["ranking"]:
            fields(row, ("product_id", "score", "event_score", "tag_score",
                         "feedback_adjustment", "normalized_score",
                         "interaction_ids", "feedback_ids", "themes"))
            for field in ("score", "event_score", "tag_score", "feedback_adjustment"):
                number(row[field], field)
            number(row["normalized_score"], "normalized_score", 0, 1)
            require(abs(row["score"] - row["event_score"] - row["tag_score"] -
                        row["feedback_adjustment"]) < 1e-8, "behavior score mismatch")
            signal = feedback["product_signals"][row["product_id"]]
            require(row["feedback_ids"] == signal["feedback_ids"] and
                    row["themes"] == signal["themes"], "feedback handoff mismatch")
            for event_id in row["interaction_ids"]:
                require(event_id in allowed_interactions and
                        allowed_interactions[event_id]["product_id"] == row["product_id"],
                        "invalid interaction evidence")
        require(result["ranking"] == sorted(result["ranking"],
                key=lambda row: (-row["score"], row["product_id"])), "unstable behavior order")

    @staticmethod
    def semantic(result, context):
        data, behavior = context
        fields(result, ("query", "embedding_used", "index", "results"))
        require(result["query"] == data["request"]["query"], "query mismatch")
        require(type(result["embedding_used"]) is bool, "embedding_used must be boolean")
        products = {p["id"] for p in data["products"]}
        require(set(result["index"]) == products, "index coverage mismatch")
        for terms in result["index"].values():
            require(isinstance(terms, dict), "index terms must be an object")
            for term, frequency in terms.items():
                text(term, "index term")
                require(type(frequency) is int and frequency > 0, "invalid term frequency")
        behavior_by_id = {row["product_id"]: row for row in behavior["ranking"]}
        require(len(result["results"]) <= data["request"]["limit"], "limit exceeded")
        seen = set()
        for row in result["results"]:
            fields(row, ("product_id", "title", "score", "relevance",
                         "personalization", "lexical_score", "embedding_score",
                         "matched_terms", "feedback_ids", "themes"))
            product_id = row["product_id"]
            require(product_id in products and product_id not in seen, "invalid search result")
            seen.add(product_id)
            for field in ("score", "relevance", "personalization", "embedding_score"):
                number(row[field], field, 0, 1)
            number(row["lexical_score"], "lexical_score", 0)
            previous = behavior_by_id[product_id]
            require(row["personalization"] == previous["normalized_score"] and
                    row["feedback_ids"] == previous["feedback_ids"] and
                    row["themes"] == previous["themes"], "behavior handoff mismatch")
        require(result["results"] == sorted(result["results"],
                key=lambda row: (-row["score"], -row["relevance"], row["product_id"])),
                "unstable search order")


def sentiment(words):
    positive = negative = 0
    for index, word in enumerate(words):
        polarity = int(word in POSITIVE) - int(word in NEGATIVE)
        if index and words[index - 1] in {"not", "never", "no"}:
            polarity *= -1
        positive += polarity > 0
        negative += polarity < 0
    return (positive - negative) / max(1, positive + negative)


def analyze_feedback(data):
    groups = defaultdict(list)
    for row in data["feedback"]:
        key = (row["customer_id"], row["product_id"], tuple(tokens(row["text"])))
        groups[key].append(row)
    items = []
    for group in groups.values():
        group.sort(key=lambda row: (timestamp(row["timestamp"]), row["id"]))
        first = group[0]
        words = tokens(first["text"])
        themes = sorted(theme for theme, keywords in THEMES.items() if set(words) & keywords)
        items.append({
            "id": first["id"], "customer_id": first["customer_id"],
            "product_id": first["product_id"], "source_ids": sorted(row["id"] for row in group),
            "excerpt": first["text"], "sentiment": sentiment(words),
            "themes": themes or ["other"],
        })
    items.sort(key=lambda row: row["id"])
    theme_evidence = defaultdict(list)
    for item in items:
        for theme in item["themes"]:
            theme_evidence[theme].append({"feedback_id": item["id"],
                                          "product_id": item["product_id"],
                                          "excerpt": item["excerpt"]})
    signals = {}
    for product in sorted(data["products"], key=lambda row: row["id"]):
        related = [item for item in items if item["product_id"] == product["id"]]
        signals[product["id"]] = {
            "sentiment": sum(item["sentiment"] for item in related) / max(1, len(related)),
            "feedback_ids": [item["id"] for item in related],
            "themes": sorted({theme for item in related for theme in item["themes"]}),
        }
    return {"input_count": len(data["feedback"]), "unique_count": len(items),
            "duplicate_count": len(data["feedback"]) - len(items), "items": items,
            "themes": dict(sorted(theme_evidence.items())), "product_signals": signals}


def rank_behavior(data, feedback):
    Validator.validate("feedback", feedback, data)
    config = data.get("config", {})
    half_life = config.get("half_life_days", 30)
    now = timestamp(data["as_of"])
    personal = [row for row in data["interactions"]
                if row["customer_id"] == data["request"]["customer_id"]]
    cold_start = not personal
    events = personal if personal else data["interactions"]
    scores = defaultdict(float)
    evidence = defaultdict(list)
    product_tags = {p["id"]: set(tag.casefold().strip() for tag in p["tags"])
                    for p in data["products"]}
    tag_affinity = defaultdict(float)
    for event in sorted(events, key=lambda row: row["id"]):
        age = (now - timestamp(event["timestamp"])).total_seconds() / 86400
        contribution = (3 if event["action"] == "purchase" else 1) * 2 ** (-age / half_life)
        scores[event["product_id"]] += contribution
        evidence[event["product_id"]].append(event["id"])
        for tag in product_tags[event["product_id"]]:
            tag_affinity[tag] += contribution
    ranking = []
    for product in sorted(data["products"], key=lambda row: row["id"]):
        product_id = product["id"]
        tags = product_tags[product_id]
        tag_score = 0.25 * sum(tag_affinity[tag] for tag in sorted(tags)) / max(1, len(tags))
        signal = feedback["product_signals"][product_id]
        adjustment = config.get("feedback_weight", 0.5) * signal["sentiment"]
        ranking.append({
            "product_id": product_id, "event_score": scores[product_id],
            "tag_score": tag_score, "feedback_adjustment": adjustment,
            "score": scores[product_id] + tag_score + adjustment,
            "interaction_ids": evidence[product_id], "feedback_ids": signal["feedback_ids"][:],
            "themes": signal["themes"][:],
        })
    low = min((row["score"] for row in ranking), default=0)
    high = max((row["score"] for row in ranking), default=0)
    for row in ranking:
        row["normalized_score"] = (row["score"] - low) / (high - low) if high > low else 0.5
    ranking.sort(key=lambda row: (-row["score"], row["product_id"]))
    return {"cold_start": cold_start, "half_life_days": half_life, "ranking": ranking}


def validated_embeddings(embedding, texts):
    try:
        vectors = embedding(texts)
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(vectors, (list, tuple)) and len(vectors) == len(texts),
            "embedding count mismatch")
    dimensions = None
    normalized = []
    for vector in vectors:
        require(isinstance(vector, (list, tuple)) and len(vector) > 0, "empty embedding vector")
        dimensions = dimensions if dimensions is not None else len(vector)
        require(len(vector) == dimensions, "embedding dimensions mismatch")
        for value in vector:
            number(value, "embedding component")
        # Scaling first avoids overflow when valid components are near float limits.
        scale = max(abs(value) for value in vector)
        require(scale > 0, "zero embedding vector")
        scaled = [value / scale for value in vector]
        norm = math.sqrt(sum(value * value for value in scaled))
        normalized.append([value / norm for value in scaled])
    return normalized


def semantic_search(data, feedback, behavior, embedding=None):
    Validator.validate("behavior", behavior, (data, feedback))
    products = sorted(data["products"], key=lambda row: row["id"])
    index = {}
    documents = []
    for product in products:
        title = tokens(product["title"], expand=True)
        tags = tokens(" ".join(product["tags"]), expand=True)
        description = tokens(product["description"], expand=True)
        index[product["id"]] = dict(sorted(Counter(title * 3 + tags * 2 + description).items()))
        documents.append(product["title"] + " " + product["description"] + " " +
                         " ".join(product["tags"]))
    query = data["request"]["query"]
    query_terms = set(tokens(query, expand=True))
    browsing = not query.strip()
    lexical = {}
    matches = {}
    for product_id, terms in index.items():
        matched = sorted(query_terms & terms.keys())
        matches[product_id] = matched
        lexical[product_id] = sum(
            (1 + math.log(terms[term])) *
            (1 + math.log((1 + len(products)) /
                          (1 + sum(term in doc for doc in index.values()))))
            for term in matched)
    peak = max(lexical.values(), default=0)
    embedding_scores = {product["id"]: 0.0 for product in products}
    used = embedding is not None and not browsing and bool(products)
    if used:
        require(callable(embedding), "embedding must be callable")
        vectors = validated_embeddings(embedding, [query] + documents)
        for product, vector in zip(products, vectors[1:]):
            cosine = sum(a * b for a, b in zip(vectors[0], vector))
            embedding_scores[product["id"]] = min(1.0, max(0.0, cosine))
    by_id = {row["product_id"]: row for row in behavior["ranking"]}
    weight = data.get("config", {}).get("personalization_weight", 0.2)
    results = []
    for product in products:
        product_id = product["id"]
        lexical_score = lexical[product_id]
        embedding_score = embedding_scores[product_id]
        if not browsing and lexical_score == 0 and embedding_score == 0:
            continue
        relevance = lexical_score / peak if peak else 0.0
        if used:
            relevance = 0.8 * relevance + 0.2 * embedding_score
        previous = by_id[product_id]
        personalization = previous["normalized_score"]
        score = personalization if browsing else (1 - weight) * relevance + weight * personalization
        results.append({
            "product_id": product_id, "title": product["title"], "score": score,
            "relevance": relevance, "personalization": personalization,
            "lexical_score": lexical_score, "embedding_score": embedding_score,
            "matched_terms": matches[product_id], "feedback_ids": previous["feedback_ids"][:],
            "themes": previous["themes"][:],
        })
    results.sort(key=lambda row: (-row["score"], -row["relevance"], row["product_id"]))
    return {"query": query, "embedding_used": used, "index": index,
            "results": results[:data["request"]["limit"]]}


def run_pipeline(document, embedding=None):
    data = copy.deepcopy(document)
    Validator.validate("input", data)
    require(embedding is None or callable(embedding), "embedding must be callable")
    feedback = Validator.validate("feedback", analyze_feedback(data), data)
    behavior = Validator.validate("behavior", rank_behavior(data, feedback), (data, feedback))
    semantic = Validator.validate("semantic", semantic_search(data, feedback, behavior, embedding),
                                  (data, behavior))
    return {"status": "ok", "schema_version": SCHEMA_VERSION, "synthetic": True,
            "feedback": feedback, "behavior": behavior, "semantic": semantic}


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        require(len(arguments) == 1, "usage: python -B implementation.py INPUT.json")
        with open(arguments[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_keys)
        output = run_pipeline(data)
        encoded = json.dumps(output, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
