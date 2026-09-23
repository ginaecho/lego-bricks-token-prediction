"""Synthetic research -> product search reference pipeline; standard library only."""

import json
import math
import re
import sys
from collections import Counter


class ValidationError(ValueError):
    """Invalid input, stage handoff, or injected embedding response."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")


def number(value, label, minimum=0, maximum=1):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, label + " is out of range")


def tokens(value):
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def validate(kind, payload, request=None):
    """One validation boundary for requests, research handoffs, and final outputs."""
    require(isinstance(payload, dict), kind + " must be an object")
    if kind == "input":
        require(set(payload) <= {"schema_version", "synthetic", "query", "products",
                                 "sources", "research_limit", "search_limit",
                                 "embedding_weight"}, "unknown input field")
        require(type(payload.get("schema_version")) is int
                and payload["schema_version"] == 1, "schema_version must be 1")
        require(payload.get("synthetic") is True, "synthetic must be true")
        text(payload.get("query"), "query")
        for key, default in (("research_limit", 20), ("search_limit", 10)):
            value = payload.get(key, default)
            require(type(value) is int and 1 <= value <= 100, key + " must be 1..100")
        number(payload.get("embedding_weight", 0.35), "embedding_weight")
        require(isinstance(payload.get("products"), list), "products must be a list")
        require(isinstance(payload.get("sources"), list), "sources must be a list")
        product_ids = set()
        for product in payload["products"]:
            require(isinstance(product, dict), "product must be an object")
            for key in ("product_id", "name", "category"):
                text(product.get(key), key)
            require(product["product_id"] not in product_ids, "duplicate product_id")
            product_ids.add(product["product_id"])
        source_ids = set()
        for source in payload["sources"]:
            require(isinstance(source, dict), "source must be an object")
            for key in ("source_id", "product_id", "title", "text"):
                text(source.get(key), key)
            require(source["source_id"] not in source_ids, "duplicate source_id")
            require(source["product_id"] in product_ids, "source references unknown product")
            source_ids.add(source["source_id"])
        return payload

    require(request is not None, "request context is required")
    validate("input", request)
    if kind == "research":
        require(payload.get("query") == request["query"], "research query mismatch")
        findings = payload.get("findings")
        require(isinstance(findings, list), "findings must be a list")
        require(len(findings) <= request.get("research_limit", 20), "too many findings")
        sources = {source["source_id"]: source for source in request["sources"]}
        seen = set()
        for finding in findings:
            require(isinstance(finding, dict), "finding must be an object")
            text(finding.get("finding_id"), "finding_id")
            require(finding["finding_id"] not in seen, "duplicate finding_id")
            seen.add(finding["finding_id"])
            number(finding.get("score"), "finding score")
            citation = finding.get("citation")
            require(isinstance(citation, dict), "citation must be an object")
            text(citation.get("source_id"), "citation source_id")
            source = sources.get(citation["source_id"])
            require(source is not None, "unknown citation source")
            require(finding.get("product_id") == source["product_id"],
                    "citation product mismatch")
            start, end = citation.get("start"), citation.get("end")
            require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(source["text"]), "invalid citation offsets")
            require(citation.get("quote") == source["text"][start:end],
                    "citation is not an exact source excerpt")
            require(citation.get("title") == source["title"], "citation title mismatch")
            require(finding.get("text") == citation["quote"], "finding is not extractive")
        return payload

    require(kind == "output", "unknown validation kind")
    require(payload.get("schema_version") == 1 and payload.get("synthetic") is True
            and payload.get("status") == "ok", "invalid output envelope")
    research = validate("research", payload.get("research"), request)
    semantic = payload.get("semantic")
    require(isinstance(semantic, dict), "semantic must be an object")
    require(semantic.get("query") == research["query"], "semantic query mismatch")
    indexed_ids = {f["product_id"] for f in research["findings"]}
    require(type(semantic.get("indexed_products")) is int
            and semantic["indexed_products"] == len(indexed_ids), "index size mismatch")
    require(semantic.get("ranking_mode") in ("lexical", "hybrid"), "invalid ranking mode")
    results = semantic.get("results")
    require(isinstance(results, list) and len(results) <= request.get("search_limit", 10),
            "invalid search results")
    products = {p["product_id"]: p for p in request["products"]}
    seen = set()
    for result in results:
        require(isinstance(result, dict), "result must be an object")
        text(result.get("product_id"), "result product_id")
        pid = result["product_id"]
        require(pid in indexed_ids and pid not in seen, "unindexed or duplicate result")
        seen.add(pid)
        require(result.get("name") == products[pid]["name"], "result name mismatch")
        for key in ("score", "lexical_score", "embedding_score"):
            number(result.get(key), key)
        expected = [f for f in research["findings"] if f["product_id"] == pid]
        require(result.get("finding_ids") == [f["finding_id"] for f in expected],
                "result finding provenance mismatch")
        require(result.get("citations") == [f["citation"] for f in expected],
                "result citation provenance mismatch")
    return payload


def passages(source_text):
    """Return sentence/line spans with Python character offsets (end exclusive)."""
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)", source_text):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - len(raw) + len(raw.rstrip())
        if start < end:
            yield start, end, source_text[start:end]


def research_stage(request):
    validate("input", request)
    query_terms = set(tokens(request["query"]))
    findings = []
    for source in request["sources"]:
        for start, end, quote in passages(source["text"]):
            overlap = query_terms.intersection(tokens(quote))
            if not overlap:
                continue
            findings.append({
                "product_id": source["product_id"],
                "text": quote,
                "score": len(overlap) / len(query_terms),
                "citation": {"source_id": source["source_id"], "title": source["title"],
                             "start": start, "end": end, "quote": quote},
            })
    findings.sort(key=lambda f: (-f["score"], f["citation"]["source_id"],
                                 f["citation"]["start"]))
    findings = findings[:request.get("research_limit", 20)]
    for index, finding in enumerate(findings, 1):
        finding["finding_id"] = "finding-" + str(index)
    return validate("research", {"query": request["query"], "findings": findings}, request)


def embedding_scores(embedder, texts):
    try:
        vectors = embedder(list(texts))
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(vectors, (list, tuple)) and len(vectors) == len(texts),
            "embedding response count mismatch")
    dimension = None
    normalized = []
    for vector in vectors:
        require(isinstance(vector, (list, tuple)) and bool(vector),
                "embedding vectors must be nonempty lists")
        if dimension is None:
            dimension = len(vector)
        require(len(vector) == dimension, "embedding dimensions differ")
        for component in vector:
            number(component, "embedding component", -1e100, 1e100)
        norm = math.hypot(*vector)
        normalized.append([value / norm if norm else 0.0 for value in vector])
    query = normalized[0]
    return [max(0.0, min(1.0, sum(a * b for a, b in zip(query, vector))))
            for vector in normalized[1:]]


def semantic_stage(request, research, embedder=None):
    validate("research", research, request)
    require(embedder is None or callable(embedder), "embedder must be callable")
    products = {p["product_id"]: p for p in request["products"]}
    grouped = {}
    for finding in research["findings"]:
        grouped.setdefault(finding["product_id"], []).append(finding)
    ids = sorted(grouped)
    documents = [" ".join([products[pid]["name"], products[pid]["category"]]
                          + [f["text"] for f in grouped[pid]]) for pid in ids]
    counts = [Counter(tokens(document)) for document in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) if lengths else 1
    query_terms = set(tokens(research["query"]))
    # An inverted index supplies document frequency and sparse candidate postings.
    index = {}
    for position, count in enumerate(counts):
        for term, frequency in count.items():
            index.setdefault(term, {})[position] = frequency
    lexical = [0.0] * len(ids)
    for term in sorted(query_terms):
        posting = index.get(term, {})
        idf = math.log(1 + (len(ids) - len(posting) + 0.5) / (len(posting) + 0.5))
        for position, frequency in posting.items():
            denominator = frequency + 1.2 * (0.25 + 0.75 * lengths[position] / average)
            lexical[position] += idf * frequency * 2.2 / denominator
    lexical = [score / (1 + score) for score in lexical]
    embedded = (embedding_scores(embedder, [research["query"]] + documents)
                if embedder is not None and documents else [0.0] * len(ids))
    weight = request.get("embedding_weight", 0.35) if embedder is not None else 0
    results = []
    for position, pid in enumerate(ids):
        results.append({
            "product_id": pid, "name": products[pid]["name"],
            "score": (1 - weight) * lexical[position] + weight * embedded[position],
            "lexical_score": lexical[position], "embedding_score": embedded[position],
            "finding_ids": [f["finding_id"] for f in grouped[pid]],
            "citations": [f["citation"] for f in grouped[pid]],
        })
    results.sort(key=lambda result: (-result["score"], result["product_id"]))
    return {"query": research["query"], "indexed_products": len(ids),
            "ranking_mode": "hybrid" if embedder is not None else "lexical",
            "results": results[:request.get("search_limit", 10)]}


def run_pipeline(request, embedder=None):
    validate("input", request)
    research = research_stage(request)
    result = {
        "schema_version": 1, "synthetic": True, "status": "ok", "research": research,
        "semantic": semantic_stage(request, research, embedder),
    }
    return validate("output", result, request)


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as handle:
            request = json.load(handle, parse_constant=reject_constant)
        result = run_pipeline(request)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
