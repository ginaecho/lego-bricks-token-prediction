"""Offline synthetic marketplace pipeline. Python standard library only.

Embedding injection: callable(list[str]) -> list[list[finite float]], one vector
per query/product. No network fetches occur: URL retrieval uses supplied fixtures.
Review reports evidence coverage, never certification or factual verification.
"""

import copy
import datetime as dt
import json
import math
import re
import sys
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            name + " must be a finite number >= " + str(minimum))
    return value


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def timestamp(value):
    text(value, "timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(result.tzinfo is not None, "timestamp requires a timezone")
    return result.astimezone(dt.timezone.utc)


def safe_url(value, hosts):
    text(value, "URL")
    require(not any(c.isspace() or ord(c) < 32 for c in value) and "\\" not in value,
            "URL contains invalid characters")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ValidationError("malformed URL") from exc
    require(parts.scheme == "https" and parts.hostname in hosts
            and parts.username is None and parts.password is None
            and port in (None, 443) and not parts.fragment,
            "URL must use HTTPS on an exact allowlisted host, with no credentials or fragment")
    return value


def objects(value, name):
    require(isinstance(value, list), name + " must be a list")
    require(all(isinstance(item, dict) for item in value), name + " entries must be objects")
    return value


def unique_ids(items, name):
    ids = [text(item.get("id"), name + ".id") for item in items]
    require(len(ids) == len(set(ids)), name + " IDs must be unique")
    return set(ids)


def validate_input(data):
    require(isinstance(data, dict), "input must be an object")
    require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data.get("synthetic") is True, "reference inputs must be labeled synthetic")
    text(data.get("query"), "query")
    require(bool(tokens(data["query"])), "query must contain searchable tokens")
    hosts = data.get("allowlisted_hosts")
    require(isinstance(hosts, list) and bool(hosts), "allowlisted_hosts must be a nonempty list")
    require(all(isinstance(h, str) and re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", h)
                for h in hosts), "hosts must be lowercase bare hostnames")
    require(len(hosts) == len(set(hosts)), "duplicate allowlisted hosts")
    now = timestamp(data.get("now"))
    limit = data.get("limit", 10)
    require(type(limit) is int and 1 <= limit <= 100, "limit must be an integer from 1 to 100")
    require(number(data.get("half_life_days", 30), "half_life_days") > 0,
            "half_life_days must be positive")
    products = objects(data.get("products"), "products")
    ids = unique_ids(products, "products")
    for product in products:
        text(product.get("title"), "product.title")
        text(product.get("description"), "product.description")
        tags = product.get("tags", [])
        require(isinstance(tags, list) and all(isinstance(t, str) and t.strip() for t in tags),
                "tags must contain nonempty strings")
        urls = product.get("source_urls")
        require(isinstance(urls, list), "source_urls must be a list")
        for url in urls:
            safe_url(url, hosts)
        require(len(urls) == len(set(urls)), "duplicate product source URL")
    docs = objects(data.get("documents"), "documents")
    urls = []
    for document in docs:
        urls.append(safe_url(document.get("url"), hosts))
        text(document.get("text"), "document.text")
    require(len(urls) == len(set(urls)), "duplicate document URL")
    requirements = objects(data.get("requirements"), "requirements")
    unique_ids(requirements, "requirements")
    for requirement in requirements:
        text(requirement.get("text"), "requirement.text")
        words = requirement.get("keywords")
        require(isinstance(words, list) and bool(words)
                and all(isinstance(w, str) and bool(tokens(w)) for w in words),
                "requirement keywords must be nonempty searchable strings")
        count = requirement.get("min_sources", 1)
        require(type(count) is int and 1 <= count <= 10, "min_sources must be 1 to 10")
    events = objects(data.get("events", []), "events")
    for event in events:
        require(event.get("product_id") in ids, "event references unknown product")
        require(event.get("kind") in ("browse", "purchase"), "unsupported event kind")
        require(timestamp(event.get("timestamp")) <= now, "event cannot be in the future")
    return data


def envelope(stage, candidates):
    return {"schema_version": 1, "status": "ok", "stage": stage, "candidates": candidates}


def validate_stage(stage, expected, data):
    """Shared stage contract, including evidence provenance and score checks."""
    require(expected in ("semantic", "web", "review", "behavior"), "unknown stage")
    require(isinstance(stage, dict) and type(stage.get("schema_version")) is int
            and stage.get("schema_version") == 1
            and stage.get("status") == "ok" and stage.get("stage") == expected,
            "invalid " + expected + " stage envelope")
    candidates = objects(stage.get("candidates"), "candidates")
    ids = [c.get("product_id") for c in candidates]
    require(all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids)),
            "candidate IDs must be unique strings")
    products = {p["id"]: p for p in data["products"]}
    documents = {d["url"]: d["text"] for d in data["documents"]}
    requirements = {r["id"]: r for r in data["requirements"]}
    for candidate in candidates:
        pid = candidate["product_id"]
        require(pid in products, "unknown candidate")
        number(candidate.get("semantic_score"), "semantic_score")
        require(candidate["semantic_score"] <= 1, "semantic score out of bounds")
        product = products[pid]
        product_tokens = tokens(" ".join([product["title"], product["description"], *product.get("tags", [])]))
        require(candidate.get("matched_tokens") == sorted(tokens(data["query"]) & product_tokens),
                "matched tokens inconsistent with candidate")
        if expected == "semantic":
            continue
        sources = objects(candidate.get("sources"), "sources")
        available = set()
        for source in sources:
            url = source.get("url")
            safe_url(url, data["allowlisted_hosts"])
            require(url in products[pid]["source_urls"] and url in documents,
                    "source is not attributable to candidate")
            require(source.get("text") == documents[url], "source text differs from fixture")
            require(source.get("retrieval") == "synthetic_fixture", "invalid retrieval provenance")
            require(url not in available, "duplicate retrieved source")
            available.add(url)
        expected_sources = [url for url in product["source_urls"] if url in documents]
        missing_urls = [url for url in product["source_urls"] if url not in documents]
        require([s["url"] for s in sources] == expected_sources,
                "retrieved sources must cover available product URLs")
        require(candidate.get("missing_urls") == missing_urls, "invalid missing URL provenance")
        findings = objects(candidate.get("findings"), "findings")
        unique_ids(findings, "findings")
        for finding in findings:
            url = finding.get("url")
            require(url in available, "finding references unavailable source")
            require(finding.get("requirement_id") in requirements, "unknown finding requirement")
            start, end = finding.get("start"), finding.get("end")
            require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(documents[url]), "invalid evidence offsets")
            require(finding.get("quote") == documents[url][start:end], "evidence quote mismatch")
            needed = tokens(" ".join(requirements[finding["requirement_id"]]["keywords"]))
            require(needed <= tokens(finding["quote"]), "finding does not match requirement")
        if expected == "web":
            continue
        checks = objects(candidate.get("checks"), "checks")
        require([c.get("requirement_id") for c in checks] == list(requirements),
                "review must cover each requirement in input order")
        for check in checks:
            req = requirements[check["requirement_id"]]
            matching = [f for f in findings if f["requirement_id"] == req["id"]]
            require(check.get("evidence_ids") == [f["id"] for f in matching],
                    "review evidence references must match findings")
            covered = len({f["url"] for f in matching}) >= req.get("min_sources", 1)
            require(check.get("status") == ("covered" if covered else "gap"),
                    "review status inconsistent with evidence")
            require(check.get("requirement_text") == req["text"], "review requirement text changed")
            expected_gap = None if covered else {
                "reason": "insufficient_keyword_evidence",
                "sources_found": len({f["url"] for f in matching}),
                "sources_required": req.get("min_sources", 1), "missing_urls": missing_urls}
            require("gap" in check and check["gap"] == expected_gap, "invalid review gap")
        coverage = sum(c["status"] == "covered" for c in checks) / len(checks) if checks else 1.0
        require(candidate.get("coverage") == coverage, "invalid review coverage")
        if expected == "behavior":
            number(candidate.get("behavior_score"), "behavior_score")
            number(candidate.get("final_score"), "final_score")
            require(candidate["behavior_score"] <= 1 and candidate["final_score"] <= 1,
                    "personalization score out of bounds")
    return stage


def semantic_search(data, embedding=None):
    query = tokens(data["query"])
    index = {}
    documents = []
    for product in data["products"]:
        document = " ".join([product["title"], product["description"], *product.get("tags", [])])
        documents.append(document)
        for token in tokens(document):
            index.setdefault(token, set()).add(product["id"])
    vectors = None
    if embedding is not None:
        try:
            vectors = embedding([data["query"], *documents])
        except Exception as exc:
            raise ValidationError("embedding callable failed") from exc
        require(isinstance(vectors, list) and len(vectors) == len(documents) + 1,
                "embedding must return one vector per text")
        size = None
        for vector in vectors:
            require(isinstance(vector, list) and bool(vector), "embedding vectors must be nonempty lists")
            size = len(vector) if size is None else size
            require(len(vector) == size, "embedding dimensions differ")
            for value in vector:
                number(value, "embedding value", -float("inf"))
    results = []
    for i, product in enumerate(data["products"]):
        matched = sorted(token for token in query if product["id"] in index.get(token, set()))
        score = len(matched) / len(query)
        if vectors is not None:
            # Normalize using hypot to avoid overflow in the norm of large vectors.
            qnorm, pnorm = math.hypot(*vectors[0]), math.hypot(*vectors[i + 1])
            require(math.isfinite(qnorm) and math.isfinite(pnorm), "embedding norm overflow")
            cosine = (sum((a / qnorm) * (b / pnorm) for a, b in zip(vectors[0], vectors[i + 1]))
                      if qnorm and pnorm else 0)
            score = 0.6 * score + 0.4 * max(0, min(1, cosine))
        if score > 0:
            results.append({"product_id": product["id"], "semantic_score": round(score, 8),
                            "matched_tokens": matched})
    results.sort(key=lambda c: (-c["semantic_score"], c["product_id"]))
    return validate_stage(envelope("semantic", results[:data.get("limit", 10)]), "semantic", data)


def web_research(previous, data):
    validate_stage(previous, "semantic", data)
    products = {p["id"]: p for p in data["products"]}
    docs = {d["url"]: d["text"] for d in data["documents"]}
    candidates = copy.deepcopy(previous["candidates"])
    for candidate in candidates:
        candidate.update(sources=[], findings=[], missing_urls=[])
        for url in products[candidate["product_id"]]["source_urls"]:
            safe_url(url, data["allowlisted_hosts"])
            if url not in docs:
                candidate["missing_urls"].append(url)
                continue
            body = docs[url]
            candidate["sources"].append({"url": url, "text": body, "retrieval": "synthetic_fixture"})
            for req in data["requirements"]:
                needed = tokens(" ".join(req["keywords"]))
                for match in re.finditer(r"[^.!?\n]+(?:[.!?]|$)", body):
                    start = match.start() + len(match.group()) - len(match.group().lstrip())
                    end = match.end()
                    quote = body[start:end]
                    if needed <= tokens(quote):
                        candidate["findings"].append({
                            "id": candidate["product_id"] + ":f" + str(len(candidate["findings"]) + 1),
                            "requirement_id": req["id"], "url": url,
                            "quote": quote, "start": start, "end": end,
                            "claim_type": "keyword_evidence_not_verified_fact"})
                        break
    return validate_stage(envelope("web", candidates), "web", data)


def document_review(previous, data):
    validate_stage(previous, "web", data)
    candidates = copy.deepcopy(previous["candidates"])
    for candidate in candidates:
        checks = []
        for req in data["requirements"]:
            evidence = [f for f in candidate["findings"] if f["requirement_id"] == req["id"]]
            count = len({f["url"] for f in evidence})
            minimum = req.get("min_sources", 1)
            checks.append({"requirement_id": req["id"], "requirement_text": req["text"],
                           "status": "covered" if count >= minimum else "gap",
                           "evidence_ids": [f["id"] for f in evidence],
                           "gap": None if count >= minimum else {
                               "reason": "insufficient_keyword_evidence", "sources_found": count,
                               "sources_required": minimum, "missing_urls": candidate["missing_urls"]}})
        candidate["checks"] = checks
        candidate["coverage"] = sum(c["status"] == "covered" for c in checks) / len(checks) if checks else 1.0
        candidate["review_notice"] = "Keyword evidence coverage only; not certification or factual verification."
    return validate_stage(envelope("review", candidates), "review", data)


def personalize(previous, data):
    validate_stage(previous, "review", data)
    now = timestamp(data["now"])
    events = data.get("events", [])
    scores = {}
    for event in events:
        age = (now - timestamp(event["timestamp"])).total_seconds() / 86400
        weight = (3 if event["kind"] == "purchase" else 1) * 2 ** (-age / data.get("half_life_days", 30))
        pid = event["product_id"]
        scores[pid] = scores.get(pid, 0) + weight
    candidates = copy.deepcopy(previous["candidates"])
    active = any(scores.get(c["product_id"], 0) > 0 for c in candidates)
    for candidate in candidates:
        raw = scores.get(candidate["product_id"], 0)
        affinity = raw / (1 + raw)
        candidate["behavior_score"] = round(affinity, 8)
        candidate["personalization_mode"] = "history" if active else "cold_start"
        if active:
            score = 0.55 * candidate["semantic_score"] + 0.25 * candidate["coverage"] + 0.2 * affinity
        else:
            score = 0.7 * candidate["semantic_score"] + 0.3 * candidate["coverage"]
        candidate["final_score"] = round(score, 8)
    candidates.sort(key=lambda c: (-c["final_score"], c["product_id"]))
    return validate_stage(envelope("behavior", candidates), "behavior", data)


def run_pipeline(data, embedding=None):
    validate_input(data)
    semantic = semantic_search(data, embedding)
    web = web_research(semantic, data)
    review = document_review(web, data)
    behavior = personalize(review, data)
    return {"schema_version": 1, "status": "ok", "synthetic": True,
            "stages": {"semantic": semantic, "web": web, "review": review, "behavior": behavior}}


def reject_constant(value):
    raise ValidationError("nonfinite JSON constant: " + value)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)
        result = run_pipeline(data)
    except (ValidationError, OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
