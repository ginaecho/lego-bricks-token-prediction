"""Deterministic synthetic marketplace pipeline; Python standard library only.

Run: python -B implementation.py example_input.json
Research consumes supplied snapshots, never fetches URLs. run_pipeline optionally
accepts an embedding callable: text -> finite, nonzero, fixed-width numeric vector.
"""

import hashlib
import ipaddress
import json
import math
import re
import sys
import unicodedata
from urllib.parse import urlsplit, urlunsplit


VERSION = "1.0"
STAGES = ("interests", "semantic", "feedback", "web")
ALIASES = {
    "hiking": "outdoor", "trekking": "outdoor", "trail": "outdoor",
    "outdoors": "outdoor", "backpacks": "backpack", "rucksack": "backpack",
    "rucksacks": "backpack", "durability": "durable", "sturdy": "durable",
    "rainproof": "waterproof", "weatherproof": "waterproof",
    "comfortable": "comfort", "cushioned": "comfort",
    "affordable": "price", "expensive": "price", "cost": "price",
}
THEMES = {
    "comfort": {"comfort", "strap", "straps", "padding", "pain"},
    "durability": {"durable", "broken", "broke", "tear", "seam"},
    "price": {"price", "value", "cheap"},
    "weather": {"waterproof", "rain", "wet", "leak"},
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(keys), f"{path} requires exactly: {', '.join(keys)}")


def text(value, path, allow_empty=False):
    require(isinstance(value, str), f"{path} must be a string")
    require((allow_empty or bool(value.strip())) and len(value) <= 10000,
            f"{path} must be nonblank and at most 10000 characters")


def number(value, path, low, high):
    require(type(value) in (int, float) and low <= value <= high
            and math.isfinite(value), f"{path} must be a finite number in [{low}, {high}]")


def integer(value, path, low, high):
    require(type(value) is int and low <= value <= high,
            f"{path} must be an integer in [{low}, {high}]")


def array(value, path, limit=1000):
    require(isinstance(value, list) and len(value) <= limit,
            f"{path} must be an array of at most {limit} entries")


def strings(value, path, unique=True):
    array(value, path)
    for item in value:
        text(item, path)
    require(not unique or len(set(value)) == len(value), f"{path} contains duplicates")


def rows(value, keys, path, id_key=None):
    array(value, path)
    for row in value:
        obj(row, keys, path)
    if id_key:
        identifiers = [row[id_key] for row in value]
        for identifier in identifiers:
            text(identifier, f"{path}.{id_key}")
        require(len(set(identifiers)) == len(identifiers), f"{path} IDs must be unique")


def tokens(value):
    raw = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold())
    return {ALIASES.get(word, word) for word in raw}


def content_key(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def host_valid(host):
    if not isinstance(host, str) or host != host.strip() or len(host) > 253:
        return False
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host):
        return False
    if "." not in host or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                              for label in host.split(".")):
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        return True


def canonical_url(url, allowed_hosts):
    """Exact allowlist; reject credentials, non-HTTPS, ports and ambiguous syntax."""
    if any(ord(char) <= 32 for char in url) or "\\" in url:
        raise ValidationError("invalid_url")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        require(parsed.scheme == "https", "https_required")
        require(parsed.username is None and parsed.password is None, "credentials_forbidden")
        require(parsed.port in (None, 443), "port_forbidden")
        require(host_valid(host), "invalid_host")
        require(host.casefold() in {item.casefold() for item in allowed_hosts}, "host_not_allowlisted")
        return urlunsplit(("https", host.casefold(), parsed.path or "/", parsed.query, ""))
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("invalid_url") from exc


def validate(kind, value, data=None, previous=None):
    """The shared boundary validator for input, every handoff and final output."""
    if kind == "input":
        obj(value, ("schema_version", "fixture_label", "catalog", "preferences",
                    "search", "feedback", "research"), "input")
        require(value["schema_version"] == VERSION, "unsupported schema_version")
        text(value["fixture_label"], "fixture_label")
        require(value["fixture_label"].startswith("SYNTHETIC"), "fixture_label must start with SYNTHETIC")
        rows(value["catalog"], ("id", "title", "description", "tags", "price"), "catalog", "id")
        for product in value["catalog"]:
            text(product["title"], "product.title")
            text(product["description"], "product.description", allow_empty=True)
            strings(product["tags"], "product.tags")
            number(product["price"], "product.price", 0, 1000000000)
        preferences = value["preferences"]
        obj(preferences, ("interests", "excluded_tags", "excluded_ids", "max_price", "max_results"), "preferences")
        for field in ("interests", "excluded_tags", "excluded_ids"):
            strings(preferences[field], f"preferences.{field}")
            require(len({item.casefold() for item in preferences[field]}) == len(preferences[field]),
                    f"preferences.{field} has case-insensitive duplicates")
        if preferences["max_price"] is not None:
            number(preferences["max_price"], "preferences.max_price", 0, 1000000000)
        integer(preferences["max_results"], "preferences.max_results", 1, 1000)
        obj(value["search"], ("query", "top_k", "min_score"), "search")
        text(value["search"]["query"], "search.query")
        require(bool(tokens(value["search"]["query"])), "search.query needs searchable terms")
        integer(value["search"]["top_k"], "search.top_k", 1, 1000)
        number(value["search"]["min_score"], "search.min_score", 0, 1)
        rows(value["feedback"], ("id", "product_id", "text", "rating"), "feedback", "id")
        product_ids = {product["id"] for product in value["catalog"]}
        for item in value["feedback"]:
            text(item["product_id"], "feedback.product_id")
            require(item["product_id"] in product_ids, "feedback references unknown product")
            text(item["text"], "feedback.text")
            require(bool(content_key(item["text"])), "feedback needs searchable text")
            integer(item["rating"], "feedback.rating", 1, 5)
        obj(value["research"], ("allowed_hosts", "pages", "max_findings_per_theme"), "research")
        strings(value["research"]["allowed_hosts"], "research.allowed_hosts")
        require(all(host_valid(host) for host in value["research"]["allowed_hosts"]), "invalid allowlist host")
        require(len({host.casefold() for host in value["research"]["allowed_hosts"]})
                == len(value["research"]["allowed_hosts"]), "duplicate allowlist host")
        integer(value["research"]["max_findings_per_theme"], "research.max_findings_per_theme", 1, 20)
        rows(value["research"]["pages"], ("id", "url", "title", "text"), "research.pages", "id")
        for page in value["research"]["pages"]:
            for field in ("url", "title", "text"):
                text(page[field], f"research.pages.{field}")
        return value

    require(data is not None, "validation requires input context")
    if kind == "output":
        obj(value, ("schema_version", "fixture_label", "status", "stages"), "output")
        require(value["schema_version"] == VERSION and value["status"] == "ok", "invalid output envelope")
        require(value["fixture_label"] == data["fixture_label"], "fixture provenance lost")
        obj(value["stages"], STAGES, "stages")
        prior = None
        for stage in STAGES:
            validate(stage, value["stages"][stage], data, prior)
            prior = value["stages"][stage]
        return value

    catalog = {product["id"]: product for product in data["catalog"]}
    if kind == "interests":
        obj(value, ("stage", "recommendations"), kind)
        rows(value["recommendations"], ("product_id", "score", "evidence", "explanation"), kind, "product_id")
        require(len(value["recommendations"]) <= data["preferences"]["max_results"], "too many recommendations")
        for row in value["recommendations"]:
            require(row["product_id"] in catalog, "unknown recommendation product")
            product = catalog[row["product_id"]]
            require(eligible(product, data["preferences"]), "excluded recommendation")
            number(row["score"], "recommendation.score", 0, 1)
            expected = interest_record(product, data["preferences"])
            require(row == expected, "ungrounded preference recommendation")
        require(value["recommendations"] == rank_interests(data), "incorrect recommendation selection or order")
    elif kind == "semantic":
        obj(value, ("stage", "query", "index", "embedding_used", "matches"), kind)
        require(previous is not None, "semantic requires interests handoff")
        require(value["query"] == data["search"]["query"], "query changed")
        require(type(value["embedding_used"]) is bool, "embedding_used must be boolean")
        allowed = {row["product_id"]: row for row in previous["recommendations"]}
        require(value["index"] == make_index(catalog, allowed), "invalid search index")
        rows(value["matches"], ("product_id", "score", "lexical_score", "embedding_score",
                                "matched_terms", "preference_evidence"), kind, "product_id")
        require(len(value["matches"]) <= data["search"]["top_k"], "too many search matches")
        for row in value["matches"]:
            require(row["product_id"] in allowed, "search escaped recommendations")
            number(row["score"], "match.score", 0, 1)
            number(row["lexical_score"], "match.lexical_score", 0, 1)
            require(row["score"] > 0 and row["score"] >= data["search"]["min_score"], "irrelevant match")
            query_terms = tokens(value["query"])
            matched = sorted(query_terms & product_tokens(catalog[row["product_id"]]))
            require(row["matched_terms"] == matched, "ungrounded matched terms")
            require(row["lexical_score"] == round(len(matched) / len(query_terms), 8), "incorrect lexical score")
            if value["embedding_used"]:
                number(row["embedding_score"], "match.embedding_score", 0, 1)
                expected_score = round(0.7 * row["lexical_score"] + 0.3 * row["embedding_score"], 8)
            else:
                require(row["embedding_score"] is None, "unexpected embedding score")
                expected_score = row["lexical_score"]
            require(row["score"] == expected_score, "incorrect relevance score")
            require(row["preference_evidence"] == allowed[row["product_id"]]["evidence"], "preference provenance lost")
        require(value["matches"] == sorted(value["matches"], key=lambda row: (
            -row["score"], -allowed[row["product_id"]]["score"], row["product_id"])), "invalid search order")
    elif kind == "feedback":
        obj(value, ("stage", "selected_product_ids", "items", "duplicate_count", "themes"), kind)
        require(previous is not None, "feedback requires semantic handoff")
        expected = aggregate_feedback(data, previous)
        require(value == expected, "invalid feedback aggregation or provenance")
    elif kind == "web":
        obj(value, ("stage", "queries", "sources", "rejected_sources", "findings"), kind)
        require(previous is not None, "web requires feedback handoff")
        require(value == retrieve_pages(data, previous), "invalid research retrieval or provenance")
    else:
        raise ValidationError("unknown validation kind")
    require(value["stage"] == kind, "incorrect stage label")
    return value


def eligible(product, preferences):
    tags = {tag.casefold() for tag in product["tags"]}
    return (
        product["id"] not in preferences["excluded_ids"]
        and not tags.intersection(tag.casefold() for tag in preferences["excluded_tags"])
        and (preferences["max_price"] is None or product["price"] <= preferences["max_price"])
    )


def interest_record(product, preferences):
    evidence = [
        {"interest": interest, "matched_tag": tag}
        for interest in preferences["interests"]
        for tag in product["tags"] if interest.casefold() == tag.casefold()
    ]
    matched = {item["interest"].casefold() for item in evidence}
    score = round(len(matched) / len(preferences["interests"]), 8) if preferences["interests"] else 0.0
    explanation = ("Matches catalog tags: " + ", ".join(item["matched_tag"] for item in evidence)
                   if evidence else "No interests supplied; passes explicit exclusions and budget.")
    return {"product_id": product["id"], "score": score, "evidence": evidence, "explanation": explanation}


def rank_interests(data):
    recommendations = []
    for product in data["catalog"]:
        if eligible(product, data["preferences"]):
            row = interest_record(product, data["preferences"])
            if row["evidence"] or not data["preferences"]["interests"]:
                recommendations.append(row)
    recommendations.sort(key=lambda row: (-row["score"], row["product_id"]))
    return recommendations[:data["preferences"]["max_results"]]


def interests_stage(data):
    validate("input", data)
    result = {"stage": "interests", "recommendations": rank_interests(data)}
    return validate("interests", result, data)


def product_tokens(product):
    return tokens(" ".join([product["title"], product["description"], *product["tags"]]))


def make_index(catalog, ids):
    index = {}
    for product_id in sorted(ids):
        for term in sorted(product_tokens(catalog[product_id])):
            index.setdefault(term, []).append(product_id)
    return dict(sorted(index.items()))


def vector(embedder, value, dimension=None):
    try:
        result = embedder(value)
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(result, (list, tuple)) and 0 < len(result) <= 4096, "invalid embedding vector")
    for component in result:
        number(component, "embedding component", -1e100, 1e100)
    require(dimension is None or len(result) == dimension, "embedding dimension mismatch")
    norm = math.hypot(*result)
    require(norm > 0, "zero embedding vector")
    return [component / norm for component in result]


def semantic_stage(data, interests, embedder=None):
    validate("input", data)
    validate("interests", interests, data)
    require(embedder is None or callable(embedder), "embedder must be callable")
    catalog = {product["id"]: product for product in data["catalog"]}
    recommendations = {row["product_id"]: row for row in interests["recommendations"]}
    index = make_index(catalog, recommendations)
    query = data["search"]["query"]
    terms = tokens(query)
    query_vector = vector(embedder, query) if embedder is not None else None
    candidates = set(recommendations) if embedder is not None else {
        product_id for term in terms for product_id in index.get(term, [])
    }
    matches = []
    for product_id in sorted(candidates):
        product = catalog[product_id]
        matched_terms = sorted(terms & product_tokens(product))
        lexical_score = round(len(matched_terms) / len(terms), 8)
        embedding_score = None
        score = lexical_score
        if embedder is not None:
            embedding = vector(embedder, " ".join([product["title"], product["description"], *product["tags"]]),
                               len(query_vector))
            embedding_score = round(max(0.0, min(1.0, sum(a * b for a, b in zip(query_vector, embedding)))), 8)
            score = round(0.7 * lexical_score + 0.3 * embedding_score, 8)
        if score > 0 and score >= data["search"]["min_score"]:
            matches.append({"product_id": product_id, "score": score, "lexical_score": lexical_score,
                            "embedding_score": embedding_score, "matched_terms": matched_terms,
                            "preference_evidence": recommendations[product_id]["evidence"]})
    matches.sort(key=lambda row: (-row["score"], -recommendations[row["product_id"]]["score"], row["product_id"]))
    result = {"stage": "semantic", "query": query, "index": index, "embedding_used": embedder is not None,
              "matches": matches[:data["search"]["top_k"]]}
    return validate("semantic", result, data, interests)


def aggregate_feedback(data, semantic):
    selected = [row["product_id"] for row in semantic["matches"]]
    grouped = {}
    duplicate_count = 0
    for row in data["feedback"]:
        if row["product_id"] not in selected:
            continue
        key = (row["product_id"], content_key(row["text"]))
        if key in grouped:
            grouped[key]["source_feedback_ids"].append(row["id"])
            duplicate_count += 1
        else:
            grouped[key] = {"feedback_id": row["id"], "product_id": row["product_id"], "text": row["text"],
                            "rating": row["rating"], "source_feedback_ids": [row["id"]]}
    items = list(grouped.values())
    themes = {}
    for item in items:
        names = [name for name, terms in THEMES.items() if tokens(item["text"]) & terms] or ["other"]
        for name in names:
            theme = themes.setdefault(name, {"theme": name, "count": 0, "sentiment": {
                "positive": 0, "neutral": 0, "negative": 0}, "support": []})
            sentiment = "positive" if item["rating"] >= 4 else "negative" if item["rating"] <= 2 else "neutral"
            theme["count"] += 1
            theme["sentiment"][sentiment] += 1
            theme["support"].append({"feedback_id": item["feedback_id"], "product_id": item["product_id"],
                                     "quote": item["text"], "source_feedback_ids": item["source_feedback_ids"]})
    return {"stage": "feedback", "selected_product_ids": selected, "items": items,
            "duplicate_count": duplicate_count, "themes": [themes[name] for name in sorted(themes)]}


def feedback_stage(data, semantic, interests):
    validate("input", data)
    validate("interests", interests, data)
    validate("semantic", semantic, data, interests)
    result = aggregate_feedback(data, semantic)
    return validate("feedback", result, data, semantic)


def retrieve_pages(data, feedback):
    research = data["research"]
    sources, rejected, accepted = [], [], {}
    seen = set()
    for page in research["pages"]:
        try:
            url = canonical_url(page["url"], research["allowed_hosts"])
            require(url not in seen, "duplicate_url")
        except ValidationError as exc:
            rejected.append({"source_id": page["id"], "url": page["url"], "reason": str(exc)})
            continue
        seen.add(url)
        accepted[page["id"]] = page
        sources.append({"source_id": page["id"], "url": url, "title": page["title"],
                        "content_sha256": hashlib.sha256(page["text"].encode("utf-8")).hexdigest(),
                        "ingestion": "supplied_offline_snapshot"})
    queries, findings = [], []
    catalog = {product["id"]: product for product in data["catalog"]}
    for theme in feedback["themes"]:
        product_ids = sorted({support["product_id"] for support in theme["support"]})
        feedback_ids = sorted({source_id for support in theme["support"] for source_id in support["source_feedback_ids"]})
        query = theme["theme"] + " " + " ".join(catalog[product_id]["title"] for product_id in product_ids)
        queries.append({"theme": theme["theme"], "query": query,
                        "product_ids": product_ids, "feedback_ids": feedback_ids})
        theme_terms = THEMES.get(theme["theme"], set())
        if not theme_terms:
            continue
        ranked = []
        for source in sources:
            page = accepted[source["source_id"]]
            excerpts = [match.group().strip() for match in re.finditer(r"[^.!?\n]+[.!?]?", page["text"])
                        if match.group().strip()]
            relevant = [(len(tokens(excerpt) & theme_terms), excerpt) for excerpt in excerpts]
            relevant = [item for item in relevant if item[0] > 0]
            if not relevant:
                continue
            # Earliest equally strong sentence wins; every quote is verbatim.
            strength, quote = max(relevant, key=lambda item: item[0])
            score = round(strength / len(theme_terms), 8)
            ranked.append({"theme": theme["theme"], "source_id": source["source_id"],
                           "url": source["url"], "title": source["title"], "quote": quote, "score": score,
                           "product_ids": product_ids, "feedback_ids": feedback_ids,
                           "relationship": "theme_context_not_product_verification"})
        ranked.sort(key=lambda row: (-row["score"], row["source_id"]))
        findings.extend(ranked[:research["max_findings_per_theme"]])
    return {"stage": "web", "queries": queries, "sources": sources, "rejected_sources": rejected,
            "findings": findings}


def web_stage(data, feedback, semantic, interests):
    validate("input", data)
    validate("interests", interests, data)
    validate("semantic", semantic, data, interests)
    validate("feedback", feedback, data, semantic)
    result = retrieve_pages(data, feedback)
    return validate("web", result, data, feedback)


def run_pipeline(data, embedder=None):
    validate("input", data)
    interests = interests_stage(data)
    semantic = semantic_stage(data, interests, embedder)
    feedback = feedback_stage(data, semantic, interests)
    web = web_stage(data, feedback, semantic, interests)
    output = {"schema_version": VERSION, "fixture_label": data["fixture_label"], "status": "ok",
              "stages": dict(zip(STAGES, (interests, semantic, feedback, web)))}
    return validate("output", output, data)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"non-finite JSON number: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
