"""Deterministic synthetic marketplace pipeline; Python standard library only.

Run: python -B implementation.py example_input.json
The envelope is shared by input and every stage: schema_version, fixture_label,
stage, status, data, results. Input results must be empty. See the example and
manifest for the contract. review checks are keyword evidence, NOT certification.
Optional run_pipeline(..., embed=callable) accepts text -> finite nonzero vector.
No provider, network, dynamic import, or model loading is performed.
"""

import copy
import datetime as dt
import json
import math
import re
import sys
from collections import Counter


STAGES = ("input", "behavior", "review", "normal", "semantic")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(value) == set(names.split()), where + " has missing or unknown fields")


def text(value, where, empty=False):
    require(isinstance(value, str) and (empty or bool(value.strip())),
            where + " must be a nonempty string")


def number(value, where, minimum=None):
    require(type(value) in (float, int) and math.isfinite(value),
            where + " must be finite numeric")
    require(minimum is None or value >= minimum, where + " is below minimum")


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.utcoffset() is not None, "timestamp must have a timezone")
        return parsed.astimezone(dt.timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError("invalid timezone-aware ISO timestamp") from exc


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def records(value, where):
    require(isinstance(value, list), where + " must be an array")
    seen = set()
    for item in value:
        require(isinstance(item, dict), where + " entries must be objects")
        text(item.get("id"), where + ".id")
        require(item["id"] not in seen, where + " ids must be unique")
        seen.add(item["id"])
    return seen


def passages(document):
    for match in re.finditer(r"[^.!?\n]+[.!?]?", document["text"]):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - len(raw) + len(raw.rstrip())
        if start < end:
            yield {"document_id": document["id"], "start": start, "end": end,
                   "quote": document["text"][start:end]}


def validate_citation(citation, docs, product_id):
    fields(citation, "document_id start end quote", "citation")
    require(isinstance(citation["document_id"], str), "citation document id must be text")
    require(citation["document_id"] in docs, "citation document does not exist")
    doc = docs[citation["document_id"]]
    require(doc["product_id"] == product_id, "citation product mismatch")
    start, end = citation["start"], citation["end"]
    require(type(start) is int and type(end) is int, "citation offsets must be integers")
    require(0 <= start < end <= len(doc["text"]), "citation offsets out of range")
    require(citation["quote"] == doc["text"][start:end], "citation is not an exact extract")


def validate(envelope, expected=None):
    """Single validation boundary used before and after every stage."""
    fields(envelope, "schema_version fixture_label stage status data results", "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "unsupported schema version")
    text(envelope["fixture_label"], "fixture_label")
    require(envelope["fixture_label"].startswith("SYNTHETIC"), "fixture must be labeled SYNTHETIC")
    require(envelope["stage"] in STAGES, "invalid stage")
    require(expected is None or envelope["stage"] == expected, "unexpected pipeline stage")
    require(envelope["status"] == "ok", "successful envelope must have status ok")
    data = envelope["data"]
    fields(data, "as_of user_id query top_k half_life_days products events documents requirements",
           "data")
    as_of = timestamp(data["as_of"])
    text(data["user_id"], "user_id")
    text(data["query"], "query", empty=True)
    require(type(data["top_k"]) is int and data["top_k"] > 0, "top_k must be a positive integer")
    number(data["half_life_days"], "half_life_days", 0)
    require(data["half_life_days"] > 0, "half_life_days must be positive")
    pids = records(data["products"], "products")
    for product in data["products"]:
        fields(product, "id title description popularity", "product")
        text(product["title"], "title")
        text(product["description"], "description", empty=True)
        number(product["popularity"], "popularity", 0)
    records(data["documents"], "documents")
    docs = {d["id"]: d for d in data["documents"]}
    for doc in docs.values():
        fields(doc, "id product_id text", "document")
        text(doc["product_id"], "document.product_id")
        require(doc["product_id"] in pids, "unknown document product")
        text(doc["text"], "document.text", empty=True)
    rids = records(data["requirements"], "requirements")
    for req in data["requirements"]:
        fields(req, "id description keywords", "requirement")
        text(req["description"], "requirement.description")
        require(isinstance(req["keywords"], list) and bool(req["keywords"]),
                "requirement keywords must be a nonempty array")
        for keyword in req["keywords"]:
            text(keyword, "keyword")
            require(bool(tokens(keyword)), "keyword must contain a searchable token")
    require(isinstance(data["events"], list), "events must be an array")
    for event in data["events"]:
        fields(event, "user_id product_id kind timestamp", "event")
        text(event["user_id"], "event.user_id")
        text(event["product_id"], "event.product_id")
        require(event["product_id"] in pids, "unknown event product")
        require(event["kind"] in ("view", "purchase"), "unsupported event kind")
        require(timestamp(event["timestamp"]) <= as_of, "future events are invalid")
    results = envelope["results"]
    require(isinstance(results, dict), "results must be an object")
    index = STAGES.index(envelope["stage"])
    require(set(results) == set(STAGES[1:index + 1]), "missing or out-of-order stage results")
    selected = []
    if index >= 1:
        behavior = results["behavior"]
        fields(behavior, "cold_start recommendations", "behavior")
        require(type(behavior["cold_start"]) is bool, "cold_start must be boolean")
        require(isinstance(behavior["recommendations"], list), "recommendations must be an array")
        for row in behavior["recommendations"]:
            fields(row, "product_id score", "recommendation")
            text(row["product_id"], "recommendation.product_id")
            require(row["product_id"] in pids and row["product_id"] not in selected,
                    "invalid or duplicate recommendation")
            number(row["score"], "behavior score", 0)
            selected.append(row["product_id"])
        require(len(selected) == min(data["top_k"], len(pids)), "incorrect recommendation count")
    if index >= 2:
        review = results["review"]
        fields(review, "notice products", "review")
        require(review["notice"] == "Keyword evidence review only; no certification or compliance claim.",
                "review notice missing")
        require(isinstance(review["products"], list), "review products must be an array")
        require([p.get("product_id") for p in review["products"] if isinstance(p, dict)] == selected,
                "review must preserve behavioral candidates and order")
        for row in review["products"]:
            fields(row, "product_id document_ids checks gaps", "review product")
            require(row["document_ids"] == [d["id"] for d in data["documents"]
                                          if d["product_id"] == row["product_id"]],
                    "review document provenance mismatch")
            require(isinstance(row["checks"], list), "checks must be an array")
            require([c.get("requirement_id") for c in row["checks"] if isinstance(c, dict)] ==
                    [r["id"] for r in data["requirements"]], "requirement propagation mismatch")
            for check in row["checks"]:
                fields(check, "requirement_id status evidence", "check")
                require(isinstance(check["evidence"], list), "evidence must be an array")
                require(check["status"] == ("evidence_found" if check["evidence"] else "gap"),
                        "invalid evidence status")
                req = next(r for r in data["requirements"] if r["id"] == check["requirement_id"])
                needed = set(tokens(" ".join(req["keywords"])))
                for citation in check["evidence"]:
                    validate_citation(citation, docs, row["product_id"])
                    require(needed <= set(tokens(citation["quote"])), "evidence does not meet keyword check")
            require(row["gaps"] == [c["requirement_id"] for c in row["checks"] if c["status"] == "gap"],
                    "gap trace mismatch")
    if index >= 3:
        normal = results["normal"]
        fields(normal, "products", "normal")
        require(isinstance(normal["products"], list), "normal products must be an array")
        require([p.get("product_id") for p in normal["products"] if isinstance(p, dict)] == selected,
                "normal must preserve reviewed candidates")
        for row, reviewed in zip(normal["products"], results["review"]["products"]):
            fields(row, "product_id gaps findings", "normal product")
            require(row["gaps"] == reviewed["gaps"], "research lost review gaps")
            require(isinstance(row["findings"], list), "findings must be an array")
            for finding in row["findings"]:
                fields(finding, "citation relevance requirement_ids", "finding")
                validate_citation(finding["citation"], docs, row["product_id"])
                require(finding["citation"]["document_id"] in reviewed["document_ids"],
                        "finding not from reviewed documents")
                number(finding["relevance"], "finding relevance", 0)
                require(isinstance(finding["requirement_ids"], list), "requirement_ids must be an array")
                require(all(isinstance(r, str) and r in rids for r in finding["requirement_ids"]),
                        "finding has unknown requirement")
                require(finding["requirement_ids"] ==
                        [c["requirement_id"] for c in reviewed["checks"]
                         if finding["citation"] in c["evidence"]],
                        "finding requirement evidence links mismatch")
    if index >= 4:
        semantic = results["semantic"]
        fields(semantic, "mode index hits", "semantic")
        require(semantic["mode"] in ("lexical", "lexical+embedding"), "invalid search mode")
        require(isinstance(semantic["index"], dict), "search index must be an object")
        for term, posting in semantic["index"].items():
            text(term, "index term")
            require(isinstance(posting, list) and
                    all(isinstance(pid, str) and pid in selected for pid in posting),
                    "index contains unreviewed product")
        require(isinstance(semantic["hits"], list), "hits must be an array")
        hits = semantic["hits"]
        require(len(hits) == len(selected), "search lost candidates")
        seen = set()
        contexts = {p["product_id"]: p for p in results["normal"]["products"]}
        for hit in hits:
            fields(hit, "product_id score lexical_score embedding_score behavior_score gaps citations", "hit")
            pid = hit["product_id"]
            text(pid, "hit.product_id")
            require(pid in selected and pid not in seen, "invalid search product")
            seen.add(pid)
            for key in ("score", "lexical_score", "embedding_score", "behavior_score"):
                number(hit[key], key)
            require(hit["gaps"] == contexts[pid]["gaps"], "search lost gaps")
            require(hit["citations"] == [f["citation"] for f in contexts[pid]["findings"]],
                    "search lost exact research citations")
    return envelope


def advance(envelope, stage, value):
    out = copy.deepcopy(envelope)
    out["stage"] = stage
    out["results"][stage] = value
    return validate(out, stage)


def behavioral(envelope):
    validate(envelope, "input")
    data = envelope["data"]
    events = [e for e in data["events"] if e["user_id"] == data["user_id"]]
    scores = {p["id"]: 0.0 for p in data["products"]}
    now = timestamp(data["as_of"])
    for event in events:
        age = (now - timestamp(event["timestamp"])).total_seconds() / 86400
        weight = 3.0 if event["kind"] == "purchase" else 1.0
        scores[event["product_id"]] += weight * 2 ** (-age / data["half_life_days"])
    popularity = {p["id"]: p["popularity"] for p in data["products"]}
    cold_start = not events
    if cold_start:
        scores = dict(popularity)
    ranked = sorted(scores, key=lambda pid: (-scores[pid], -popularity[pid], pid))
    recommendations = [{"product_id": pid, "score": scores[pid]}
                       for pid in ranked[:data["top_k"]]]
    return advance(envelope, "behavior", {"cold_start": cold_start, "recommendations": recommendations})


def review(envelope):
    validate(envelope, "behavior")
    data = envelope["data"]
    products = []
    for candidate in envelope["results"]["behavior"]["recommendations"]:
        pid = candidate["product_id"]
        documents = [d for d in data["documents"] if d["product_id"] == pid]
        extracted = [c for d in documents for c in passages(d)]
        checks = []
        for req in data["requirements"]:
            needed = set(tokens(" ".join(req["keywords"])))
            evidence = [c for c in extracted if needed <= set(tokens(c["quote"]))]
            checks.append({"requirement_id": req["id"],
                           "status": "evidence_found" if evidence else "gap", "evidence": evidence})
        products.append({"product_id": pid, "document_ids": [d["id"] for d in documents],
                         "checks": checks,
                         "gaps": [c["requirement_id"] for c in checks if c["status"] == "gap"]})
    return advance(envelope, "review",
                   {"notice": "Keyword evidence review only; no certification or compliance claim.",
                    "products": products})


def research(envelope):
    validate(envelope, "review")
    data = envelope["data"]
    docs = {d["id"]: d for d in data["documents"]}
    query_terms = set(tokens(data["query"]))
    products = []
    for row in envelope["results"]["review"]["products"]:
        findings = []
        gap_terms = set(tokens(" ".join(" ".join(r["keywords"]) for r in data["requirements"]
                                       if r["id"] in row["gaps"])))
        for doc_id in row["document_ids"]:
            for citation in passages(docs[doc_id]):
                words = set(tokens(citation["quote"]))
                linked = [c["requirement_id"] for c in row["checks"] if citation in c["evidence"]]
                relevance = len(words & query_terms) + 0.25 * len(words & gap_terms) + 0.5 * len(linked)
                if relevance > 0:
                    findings.append({"citation": citation, "relevance": relevance,
                                     "requirement_ids": linked})
        findings.sort(key=lambda f: (-f["relevance"], f["citation"]["document_id"],
                                    f["citation"]["start"]))
        products.append({"product_id": row["product_id"], "gaps": list(row["gaps"]),
                         "findings": findings[:data["top_k"]]})
    return advance(envelope, "normal", {"products": products})


def embedding_vector(embed, value, dimension=None):
    try:
        vector = embed(value)
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(vector, (list, tuple)) and bool(vector), "embedding must be a nonempty vector")
    for v in vector:
        number(v, "embedding component")
    require(dimension is None or len(vector) == dimension, "embedding dimensions differ")
    scale = max(abs(v) for v in vector)
    require(scale > 0, "embedding vector must be nonzero")
    scaled = [v / scale for v in vector]
    norm = math.sqrt(sum(v * v for v in scaled))
    return [v / norm for v in scaled]


def semantic_search(envelope, embed=None):
    validate(envelope, "normal")
    require(embed is None or callable(embed), "embed must be callable")
    data = envelope["data"]
    products = {p["id"]: p for p in data["products"]}
    scores = {p["product_id"]: p["score"] for p in envelope["results"]["behavior"]["recommendations"]}
    contexts = envelope["results"]["normal"]["products"]
    documents = {}
    counts = {}
    index = {}
    for context in contexts:
        pid = context["product_id"]
        product = products[pid]
        documents[pid] = " ".join([product["title"], product["description"]] +
                                  [f["citation"]["quote"] for f in context["findings"]])
        counts[pid] = Counter(tokens(documents[pid]))
        for term in sorted(counts[pid]):
            index.setdefault(term, []).append(pid)
    qterms = Counter(tokens(data["query"]))
    qvector = embedding_vector(embed, data["query"]) if embed is not None and qterms else None
    hits = []
    for context in contexts:
        pid = context["product_id"]
        lexical = sum(qfreq * (1 + math.log(counts[pid][term])) *
                      (1 + math.log((len(contexts) + 1) / (len(index[term]) + 1)))
                      for term, qfreq in qterms.items() if term in counts[pid])
        cosine = 0.0
        if qvector is not None:
            vector = embedding_vector(embed, documents[pid], len(qvector))
            cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(qvector, vector))))
        hits.append({"product_id": pid, "score": lexical + cosine, "lexical_score": lexical,
                     "embedding_score": cosine, "behavior_score": scores[pid],
                     "gaps": list(context["gaps"]),
                     "citations": [f["citation"] for f in context["findings"]]})
    hits.sort(key=lambda h: (-h["score"], -h["behavior_score"], h["product_id"]))
    return advance(envelope, "semantic", {"mode": "lexical+embedding" if embed is not None else "lexical",
                                         "index": dict(sorted(index.items())), "hits": hits})


def run_pipeline(envelope, embed=None):
    return semantic_search(research(review(behavioral(envelope))), embed=embed)


def reject_constant(value):
    raise ValidationError("nonstandard JSON number: " + value)


def unique_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            envelope = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_keys)
        output = run_pipeline(envelope)
        encoded = json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
