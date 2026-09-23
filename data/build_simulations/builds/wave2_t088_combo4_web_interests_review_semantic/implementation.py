"""Synthetic, offline research -> recommendation -> review -> search pipeline.

Run: python -B implementation.py example_input.json
Injection interfaces:
  retriever(url: str) -> str: returns a page, never a redirect or response object.
  embedder(text: str) -> sequence[finite float]: same dimension on every call.
No default code performs network access. Keyword evidence is not verification or
certification of a product's real-world properties.
"""

import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from urllib.parse import urlsplit


VERSION = "1.0"
STAGES = ("web", "interests", "review", "semantic")
NOTICE = "Synthetic keyword-evidence review only; not certification or factual verification."


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, context):
    require(isinstance(value, dict), context + " must be an object")
    require(set(value) == set(required), context + " has missing or unknown fields")


def text(value, context, maximum=20000):
    require(isinstance(value, str) and bool(value.strip()), context + " must be nonempty text")
    require(len(value) <= maximum, context + " is too long")
    return value


def strings(value, context, nonempty=False):
    require(isinstance(value, list), context + " must be an array")
    require(not nonempty or bool(value), context + " must not be empty")
    for item in value:
        text(item, context, 200)
    require(len(set(value)) == len(value), context + " has duplicates")


def integer(value, context, minimum=1, maximum=100):
    require(type(value) is int and minimum <= value <= maximum, context + " is out of range")


def number(value, context, minimum=0, maximum=1):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, context + " must be finite and in range")


def tokens(value):
    return re.findall(r"[a-z0-9]+", value.casefold())


def allowed_url(value, hosts):
    text(value, "URL", 2000)
    try:
        parsed = urlsplit(value)
        allowed = (parsed.scheme == "https" and parsed.hostname in hosts
                   and parsed.username is None and parsed.password is None
                   and parsed.port in (None, 443) and not parsed.fragment
                   and not any(c.isspace() or ord(c) < 32 for c in value)
                   and "\\" not in value)
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(allowed, "URL is not an allowlisted HTTPS resource")


def validate_request(data):
    fields(data, ("schema_version", "synthetic", "allowlisted_hosts", "sources",
                  "fixture_pages", "preferences", "requirements", "search"), "request")
    require(data["schema_version"] == VERSION, "Unsupported schema version")
    require(data["synthetic"] is True, "Fixture data must be explicitly synthetic")
    hosts = data["allowlisted_hosts"]
    strings(hosts, "allowlisted_hosts", True)
    for host in hosts:
        require(bool(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host)),
                "Allowlist entries must be lowercase exact hostnames")
    sources = data["sources"]
    require(isinstance(sources, list) and 1 <= len(sources) <= 100, "sources must contain 1..100 items")
    ids, urls, products = set(), set(), {}
    for source in sources:
        fields(source, ("id", "url", "product_id", "title", "tags"), "source")
        for key in ("id", "product_id", "title"):
            text(source[key], key, 200)
        strings(source["tags"], "tags")
        allowed_url(source["url"], hosts)
        require(source["id"] not in ids and source["url"] not in urls, "Duplicate source id or URL")
        ids.add(source["id"])
        urls.add(source["url"])
        metadata = (source["title"], source["tags"])
        require(source["product_id"] not in products or products[source["product_id"]] == metadata,
                "Conflicting product metadata")
        products[source["product_id"]] = metadata
    require(isinstance(data["fixture_pages"], dict), "fixture_pages must be an object")
    for url, page in data["fixture_pages"].items():
        allowed_url(url, hosts)
        require(url in urls, "Fixture page does not correspond to a source")
        text(page, "fixture page")
    prefs = data["preferences"]
    fields(prefs, ("interests", "exclude_tags", "exclude_product_ids", "limit"), "preferences")
    strings(prefs["interests"], "interests")
    strings(prefs["exclude_tags"], "exclude_tags")
    strings(prefs["exclude_product_ids"], "exclude_product_ids")
    for interest in prefs["interests"]:
        require(bool(tokens(interest)), "Interest must contain searchable words")
    integer(prefs["limit"], "recommendation limit")
    requirements = data["requirements"]
    require(isinstance(requirements, list) and len(requirements) <= 100, "Invalid requirements")
    req_ids = set()
    for req in requirements:
        fields(req, ("id", "description", "keywords", "applies_to_tags"), "requirement")
        text(req["id"], "requirement id", 200)
        text(req["description"], "requirement description", 1000)
        strings(req["keywords"], "keywords", True)
        strings(req["applies_to_tags"], "applies_to_tags")
        for keyword in req["keywords"]:
            require(bool(tokens(keyword)), "Keyword must contain searchable words")
        require(req["id"] not in req_ids, "Duplicate requirement id")
        req_ids.add(req["id"])
    search = data["search"]
    fields(search, ("query", "limit", "min_score"), "search")
    text(search["query"], "search query", 1000)
    require(bool(tokens(search["query"])), "Query must contain searchable words")
    integer(search["limit"], "search limit")
    number(search["min_score"], "min_score")
    return data


def keyword_match(phrase, content):
    return set(tokens(phrase)).issubset(set(tokens(content)))


def validate(data, expected_stage=None):
    """One boundary validator for requests and every stage's shared envelope."""
    if expected_stage is None:
        return validate_request(data)
    fields(data, ("schema_version", "synthetic", "status", "stage", "request",
                  "sources", "findings", "recommendations", "reviews", "search"), "envelope")
    require(data["schema_version"] == VERSION and data["synthetic"] is True
            and data["status"] == "ok" and data["stage"] == expected_stage,
            "Invalid envelope identity or stage")
    require(expected_stage in STAGES, "Unknown stage")
    request = validate_request(data["request"])
    for key in ("sources", "findings", "recommendations", "reviews"):
        require(isinstance(data[key], list), key + " must be an array")
    source_specs = {s["id"]: s for s in request["sources"]}
    require(len(data["sources"]) == len(source_specs), "Research sources missing")
    source_map = {}
    for source in data["sources"]:
        fields(source, ("id", "url", "product_id", "title", "tags", "text", "sha256"), "retrieved source")
        require(source["id"] in source_specs and source["id"] not in source_map, "Invalid source identity")
        spec = source_specs[source["id"]]
        require(all(source[k] == v for k, v in spec.items()), "Source metadata changed")
        text(source["text"], "retrieved text")
        require(source["sha256"] == hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
                "Source digest mismatch")
        source_map[source["id"]] = source
    finding_map = {}
    require(len(data["findings"]) == len(source_map), "Findings must cover every source")
    for finding in data["findings"]:
        fields(finding, ("id", "product_id", "source_id", "url", "quote", "span", "sha256"), "finding")
        require(isinstance(finding["id"], str) and finding["id"] not in finding_map, "Invalid finding id")
        require(isinstance(finding["source_id"], str) and finding["source_id"] in source_map,
                "Unknown finding source")
        source = source_map[finding["source_id"]]
        require(finding["id"] == "finding:" + source["id"], "Finding id must identify its source")
        require(finding["product_id"] == source["product_id"] and finding["url"] == source["url"]
                and finding["sha256"] == source["sha256"], "Finding provenance mismatch")
        require(finding["span"] == [0, len(source["text"])] and finding["quote"] == source["text"],
                "Finding quote or span mismatch")
        finding_map[finding["id"]] = finding
    prefs = request["preferences"]
    recommendations = {}
    for rank, rec in enumerate(data["recommendations"], 1):
        fields(rec, ("product_id", "title", "tags", "score", "rank", "finding_ids",
                     "matched_interests", "explanations"), "recommendation")
        require(isinstance(rec["product_id"], str) and rec["product_id"] not in recommendations,
                "Duplicate or invalid recommendation")
        require(rec["product_id"] not in prefs["exclude_product_ids"]
                and not {t.casefold() for t in rec["tags"]}.intersection(
                    t.casefold() for t in prefs["exclude_tags"]), "Excluded product propagated")
        own = [f["id"] for f in data["findings"] if f["product_id"] == rec["product_id"]]
        require(bool(own) and rec["finding_ids"] == own, "Recommendation evidence mismatch")
        source = source_map[finding_map[own[0]]["source_id"]]
        require(rec["title"] == source["title"] and rec["tags"] == source["tags"], "Product metadata changed")
        number(rec["score"], "preference score", 0, 10000)
        require(type(rec["rank"]) is int and rec["rank"] == rank, "Invalid recommendation rank")
        require(isinstance(rec["matched_interests"], list) and isinstance(rec["explanations"], list),
                "Invalid explanations")
        expected = interest_evidence(request, [finding_map[f] for f in own])
        require(rec["explanations"] == expected and rec["matched_interests"] == [e["interest"] for e in expected]
                and rec["score"] == len(expected), "Ungrounded preference explanation")
        recommendations[rec["product_id"]] = rec
    require(len(recommendations) <= prefs["limit"], "Too many recommendations")
    require(data["recommendations"] == sorted(data["recommendations"],
            key=lambda r: (-r["score"], r["product_id"])), "Recommendation order invalid")
    if expected_stage == "web":
        require(not recommendations, "Premature recommendations")
    if STAGES.index(expected_stage) < 2:
        require(not data["reviews"], "Premature reviews")
    else:
        require(len(data["reviews"]) == len(recommendations), "Reviews missing")
        reviewed = set()
        for review in data["reviews"]:
            fields(review, ("product_id", "notice", "checks", "gap_count"), "review")
            pid = review["product_id"]
            require(isinstance(pid, str) and pid in recommendations and pid not in reviewed,
                    "Review product mismatch")
            reviewed.add(pid)
            require(review == make_review(request, recommendations[pid], finding_map),
                    "Review evidence or gaps mismatch")
    if expected_stage != "semantic":
        require(data["search"] is None, "Premature search")
    else:
        validate_search(data)
    return data


def research(request, retriever=None):
    validate(request)
    if retriever is not None:
        require(callable(retriever), "retriever must be callable")
    result = dict(schema_version=VERSION, synthetic=True, status="ok", stage="web",
                  request=copy.deepcopy(request), sources=[], findings=[],
                  recommendations=[], reviews=[], search=None)
    for spec in request["sources"]:
        # Validate before invoking an injected retriever; no live default exists.
        allowed_url(spec["url"], request["allowlisted_hosts"])
        try:
            page = retriever(spec["url"]) if retriever else request["fixture_pages"][spec["url"]]
        except Exception as exc:
            raise ValidationError("Retrieval failed for source " + spec["id"]) from exc
        text(page, "retrieved page")
        digest = hashlib.sha256(page.encode("utf-8")).hexdigest()
        result["sources"].append(dict(copy.deepcopy(spec), text=page, sha256=digest))
        result["findings"].append(dict(id="finding:" + spec["id"], product_id=spec["product_id"],
                                      source_id=spec["id"], url=spec["url"], quote=page,
                                      span=[0, len(page)], sha256=digest))
    return validate(result, "web")


def interest_evidence(request, findings):
    explanations = []
    for interest in request["preferences"]["interests"]:
        matches = [f for f in findings if keyword_match(interest, f["quote"])]
        if matches:
            explanations.append({"interest": interest, "finding_ids": [f["id"] for f in matches],
                                 "reason": "Interest words occur in the cited synthetic source text."})
    return explanations


def recommend(previous):
    validate(previous, "web")
    result = copy.deepcopy(previous)
    prefs = result["request"]["preferences"]
    products = {}
    excluded_tags = {t.casefold() for t in prefs["exclude_tags"]}
    for source in result["sources"]:
        pid = source["product_id"]
        if pid in prefs["exclude_product_ids"] or excluded_tags.intersection(t.casefold() for t in source["tags"]):
            continue
        products[pid] = source
    candidates = []
    for pid, source in products.items():
        findings = [f for f in result["findings"] if f["product_id"] == pid]
        explanations = interest_evidence(result["request"], findings)
        candidates.append(dict(product_id=pid, title=source["title"], tags=source["tags"],
                               score=len(explanations), rank=0, finding_ids=[f["id"] for f in findings],
                               matched_interests=[e["interest"] for e in explanations],
                               explanations=explanations))
    candidates.sort(key=lambda r: (-r["score"], r["product_id"]))
    result["recommendations"] = candidates[:prefs["limit"]]
    for rank, rec in enumerate(result["recommendations"], 1):
        rec["rank"] = rank
    result["stage"] = "interests"
    return validate(result, "interests")


def make_review(request, rec, finding_map):
    checks = []
    findings = [finding_map[fid] for fid in rec["finding_ids"]]
    for req in request["requirements"]:
        applies = not req["applies_to_tags"] or bool(
            {t.casefold() for t in req["applies_to_tags"]}.intersection(t.casefold() for t in rec["tags"]))
        matches = [f["id"] for f in findings
                   if all(keyword_match(k, f["quote"]) for k in req["keywords"])] if applies else []
        missing = [k for k in req["keywords"] if not any(keyword_match(k, f["quote"]) for f in findings)]
        state = "not_applicable" if not applies else ("keyword_evidence_found" if matches else "gap")
        checks.append(dict(requirement_id=req["id"], description=req["description"], status=state,
                           evidence_finding_ids=matches, checked_finding_ids=rec["finding_ids"] if applies else [],
                           missing_keywords=missing if state == "gap" else [],
                           gap_reason=("Required words were not found together in one cited source."
                                       if state == "gap" else None)))
    return dict(product_id=rec["product_id"], notice=NOTICE, checks=checks,
                gap_count=sum(c["status"] == "gap" for c in checks))


def review(previous):
    validate(previous, "interests")
    result = copy.deepcopy(previous)
    finding_map = {f["id"]: f for f in result["findings"]}
    result["reviews"] = [make_review(result["request"], rec, finding_map) for rec in result["recommendations"]]
    result["stage"] = "review"
    return validate(result, "review")


def documents(data):
    finding_map = {f["id"]: f for f in data["findings"]}
    recs = {r["product_id"]: r for r in data["recommendations"]}
    docs = []
    for rev in data["reviews"]:
        rec = recs[rev["product_id"]]
        # Only source-backed content is searchable, not unfulfilled requirement words.
        content = rec["title"] + "\n" + "\n".join(finding_map[f]["quote"] for f in rec["finding_ids"])
        docs.append(dict(product_id=rec["product_id"], text=content, finding_ids=rec["finding_ids"],
                         gap_count=rev["gap_count"], review_status="gaps_present" if rev["gap_count"] else "no_keyword_gaps"))
    return docs


def cosine(left, right):
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if not left_norm or not right_norm:
        return 0.0
    require(math.isfinite(left_norm) and math.isfinite(right_norm), "Vector norm overflow")
    return max(0.0, min(1.0, math.fsum((a / left_norm) * (b / right_norm) for a, b in zip(left, right))))


def embedding(embedder, value, dimension=None):
    try:
        vector = embedder(value)
    except Exception as exc:
        raise ValidationError("Embedding callable failed") from exc
    require(isinstance(vector, (list, tuple)) and 1 <= len(vector) <= 4096,
            "Embedding must be a nonempty bounded numeric sequence")
    for item in vector:
        number(item, "embedding element", -1e100, 1e100)
    require(dimension is None or len(vector) == dimension, "Embedding dimension mismatch")
    return list(vector)


def semantic_search(previous, embedder=None):
    validate(previous, "review")
    result = copy.deepcopy(previous)
    spec = result["request"]["search"]
    docs = documents(result)
    index = {}
    counts = {}
    for doc in docs:
        count = Counter(tokens(doc["text"]))
        counts[doc["product_id"]] = count
        for token in sorted(count):
            index.setdefault(token, []).append(doc["product_id"])
    query_count = Counter(tokens(spec["query"]))
    if embedder is not None:
        require(callable(embedder), "embedder must be callable")
        query_vector = embedding(embedder, spec["query"])
    hits = []
    for doc in docs:
        count = counts[doc["product_id"]]
        if embedder is None:
            vocabulary = sorted(set(query_count) | set(count))
            score = cosine([query_count[t] for t in vocabulary], [count[t] for t in vocabulary])
        else:
            score = cosine(query_vector, embedding(embedder, doc["text"], len(query_vector)))
        if score > 0 and score >= spec["min_score"]:
            hits.append(dict(product_id=doc["product_id"], score=score, finding_ids=doc["finding_ids"],
                             matched_terms=sorted(set(query_count).intersection(count)),
                             gap_count=doc["gap_count"], review_status=doc["review_status"]))
    hits.sort(key=lambda h: (-h["score"], h["product_id"]))
    result["search"] = dict(query=spec["query"], mode="injected_embedding" if embedder else "lexical_cosine",
                            index=index, documents=docs, hits=hits[:spec["limit"]])
    result["stage"] = "semantic"
    return validate(result, "semantic")


def validate_search(data):
    search = data["search"]
    fields(search, ("query", "mode", "index", "documents", "hits"), "search output")
    require(search["mode"] in ("lexical_cosine", "injected_embedding"), "Unknown search mode")
    require(search["query"] == data["request"]["search"]["query"], "Search query changed")
    expected_docs = documents(data)
    require(search["documents"] == expected_docs, "Index documents lost review provenance")
    expected_index = {}
    for doc in expected_docs:
        for token in sorted(set(tokens(doc["text"]))):
            expected_index.setdefault(token, []).append(doc["product_id"])
    require(search["index"] == expected_index, "Search index mismatch")
    docs = {d["product_id"]: d for d in expected_docs}
    require(isinstance(search["hits"], list) and len(search["hits"]) <= data["request"]["search"]["limit"],
            "Invalid search hits")
    seen = set()
    for hit in search["hits"]:
        fields(hit, ("product_id", "score", "finding_ids", "matched_terms", "gap_count", "review_status"), "hit")
        pid = hit["product_id"]
        require(isinstance(pid, str) and pid in docs and pid not in seen, "Invalid hit identity")
        seen.add(pid)
        doc = docs[pid]
        number(hit["score"], "relevance score")
        require(hit["score"] > 0 and hit["score"] >= data["request"]["search"]["min_score"],
                "Hit below score threshold")
        require(all(hit[k] == doc[k] for k in ("finding_ids", "gap_count", "review_status")),
                "Hit lost review/evidence linkage")
        require(hit["matched_terms"] == sorted(set(tokens(search["query"])).intersection(tokens(doc["text"]))),
                "Ungrounded matched terms")
    require(search["hits"] == sorted(search["hits"], key=lambda h: (-h["score"], h["product_id"])),
            "Search ranking is not deterministic")


def run_pipeline(request, retriever=None, embedder=None):
    return semantic_search(review(recommend(research(request, retriever))), embedder)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], "r", encoding="utf-8") as handle:
            request = json.load(handle, object_pairs_hook=unique_object)
        output = run_pipeline(request)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        output = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
