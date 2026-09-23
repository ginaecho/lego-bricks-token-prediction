"""Synthetic, deterministic review -> evidence retrieval -> feedback pipeline.

Optional embedder contract: callable(list[str]) -> list[list[finite number]].
It is injected by Python callers only; the CLI never loads or calls a provider.
"""

import copy
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


def tokens(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def text(value, path):
    require(isinstance(value, str) and bool(tokens(value)), path + " must contain words")
    require(len(value) <= 20000, path + " exceeds 20000 characters")


def fields(value, names, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(names.split()), path + " has missing or unknown fields")


def records(value, names, path):
    require(isinstance(value, list) and len(value) <= 500, path + " must be a list of at most 500")
    ids = set()
    for item in value:
        fields(item, names, path + " item")
        text(item["id"], path + ".id")
        require(item["id"] not in ids, path + " has duplicate ids")
        ids.add(item["id"])
    return ids


def term_list(value, path):
    require(isinstance(value, list) and 0 < len(value) <= 50, path + " must have 1..50 terms")
    for term in value:
        text(term, path)
    normalized = [tuple(tokens(term)) for term in value]
    require(len(set(normalized)) == len(normalized), path + " has duplicate normalized terms")


def validate_input(data):
    fields(data, "schema_version synthetic documents requirements query feedback themes", "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "unsupported schema_version")
    require(data["synthetic"] is True, "this reference implementation requires synthetic=true")
    doc_ids = records(data["documents"], "id title text", "documents")
    for doc in data["documents"]:
        text(doc["title"], "document title")
        text(doc["text"], "document text")
    records(data["requirements"], "id description terms", "requirements")
    for req in data["requirements"]:
        text(req["description"], "requirement description")
        term_list(req["terms"], "requirement terms")
    fields(data["query"], "text limit", "query")
    text(data["query"]["text"], "query.text")
    limit = data["query"]["limit"]
    require(type(limit) is int and 1 <= limit <= 50, "query.limit must be an integer in 1..50")
    records(data["feedback"], "id document_id text", "feedback")
    for feedback in data["feedback"]:
        require(isinstance(feedback["document_id"], str) and feedback["document_id"] in doc_ids,
                "feedback references an unknown document")
        text(feedback["text"], "feedback text")
    records(data["themes"], "id terms", "themes")
    require(all(theme["id"] != "other" for theme in data["themes"]), "theme id 'other' is reserved")
    for theme in data["themes"]:
        term_list(theme["terms"], "theme terms")
    return data


def find_phrase(source, phrase):
    spans = list(re.finditer(r"\w+", source, flags=re.UNICODE))
    wanted = tokens(phrase)
    words = [match.group().casefold() for match in spans]
    for index in range(len(words) - len(wanted) + 1):
        if words[index:index + len(wanted)] == wanted:
            return spans[index].start(), spans[index + len(wanted) - 1].end()
    return None


def compute_review(data):
    checks = []
    for ri, req in enumerate(data["requirements"]):
        evidence = []
        present = set()
        for di, doc in enumerate(data["documents"]):
            for ti, term in enumerate(req["terms"]):
                span = find_phrase(doc["text"], term)
                if span is None:
                    continue
                present.add(term)
                start, end = max(0, span[0] - 80), min(len(doc["text"]), span[1] + 80)
                evidence.append({
                    "id": f"r{ri}d{di}t{ti}", "document_id": doc["id"],
                    "requirement_id": req["id"], "term": term,
                    "excerpt": doc["text"][start:end], "start": start, "end": end,
                })
        missing = [term for term in req["terms"] if term not in present]
        checks.append({
            "requirement_id": req["id"],
            "status": "supported" if not missing else ("partial" if evidence else "gap"),
            "missing_terms": missing, "evidence": evidence,
        })
    return {
        "notice": "Term-based evidence check only; not certification or a compliance determination.",
        "checks": checks,
    }


def make_index(envelope):
    titles = {doc["id"]: doc["title"] for doc in envelope["input"]["documents"]}
    return [
        {**evidence, "search_text": titles[evidence["document_id"]] + " " + evidence["excerpt"]}
        for check in envelope["review"]["checks"] for evidence in check["evidence"]
    ]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(envelope, stage):
    """One validation boundary used before and after every stage."""
    stage_fields = {
        "review": "schema_version status stage input review",
        "semantic": "schema_version status stage input review semantic",
        "feedback": "schema_version status stage input review semantic feedback",
    }
    require(stage in stage_fields, "unknown stage")
    fields(envelope, stage_fields[stage], "pipeline envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "invalid envelope version")
    require(envelope["status"] == "ok" and envelope["stage"] == stage, "invalid envelope status or stage")
    validate_input(envelope["input"])
    require(envelope["review"] == compute_review(envelope["input"]), "review evidence or gaps were altered")
    if stage == "review":
        return envelope
    search = envelope["semantic"]
    fields(search, "mode index hits", "semantic")
    require(search["mode"] in ("lexical", "hybrid"), "unknown retrieval mode")
    require(search["index"] == make_index(envelope), "index does not preserve review evidence")
    require(isinstance(search["hits"], list), "hits must be a list")
    require(len(search["hits"]) <= envelope["input"]["query"]["limit"], "too many search hits")
    index = {entry["id"]: entry for entry in search["index"]}
    seen = set()
    order = []
    for hit in search["hits"]:
        fields(hit, "evidence_id document_id requirement_id score", "hit")
        eid = hit["evidence_id"]
        require(isinstance(eid, str) and eid in index and eid not in seen, "invalid or repeated hit")
        seen.add(eid)
        require(hit["document_id"] == index[eid]["document_id"] and
                hit["requirement_id"] == index[eid]["requirement_id"], "hit provenance mismatch")
        require(finite_number(hit["score"]) and 0 < hit["score"] <= 1, "invalid hit score")
        order.append((-hit["score"], eid))
    require(order == sorted(order), "hits are not deterministically ranked")
    if stage == "feedback":
        require(envelope["feedback"] == compute_feedback(envelope), "feedback provenance or themes were altered")
    return envelope


def review_stage(data):
    validate_input(data)
    result = {
        "schema_version": 1, "status": "ok", "stage": "review",
        "input": copy.deepcopy(data), "review": compute_review(data),
    }
    return validate(result, "review")


def embedding_vectors(embedder, texts):
    try:
        vectors = embedder(texts)
        require(isinstance(vectors, list) and len(vectors) == len(texts), "embedding batch size mismatch")
        size = None
        normalized = []
        for vector in vectors:
            require(isinstance(vector, list) and 0 < len(vector) <= 4096, "invalid embedding vector")
            require(all(finite_number(value) for value in vector), "embedding values must be finite numbers")
            if size is None:
                size = len(vector)
            require(len(vector) == size, "embedding dimensions differ")
            scale = max(abs(value) for value in vector)
            require(scale > 0, "zero embedding vector")
            scaled = [value / scale for value in vector]
            norm = math.sqrt(sum(value * value for value in scaled))
            normalized.append([value / norm for value in scaled])
        return normalized
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("embedding callable failed: " + type(exc).__name__) from exc


def semantic_stage(previous, embedder=None):
    validate(previous, "review")
    result = copy.deepcopy(previous)
    index = make_index(result)
    query = result["input"]["query"]
    words = set(tokens(query["text"]))
    vectors = None
    if embedder is not None:
        require(callable(embedder), "embedder must be callable")
        if index:
            vectors = embedding_vectors(embedder, [query["text"]] + [entry["search_text"] for entry in index])
    hits = []
    for i, entry in enumerate(index):
        score = len(words.intersection(tokens(entry["search_text"]))) / len(words)
        if vectors is not None:
            cosine = max(0.0, min(1.0, sum(a * b for a, b in zip(vectors[0], vectors[i + 1]))))
            score = (score + cosine) / 2
        score = round(score, 8)
        if score > 0:
            hits.append({
                "evidence_id": entry["id"], "document_id": entry["document_id"],
                "requirement_id": entry["requirement_id"], "score": score,
            })
    hits.sort(key=lambda hit: (-hit["score"], hit["evidence_id"]))
    result["stage"] = "semantic"
    result["semantic"] = {
        "mode": "hybrid" if embedder is not None else "lexical",
        "index": index, "hits": hits[:query["limit"]],
    }
    return validate(result, "semantic")


def compute_feedback(envelope):
    data = envelope["input"]
    hits = envelope["semantic"]["hits"]
    relevant_docs = {hit["document_id"] for hit in hits}
    groups = {}
    excluded = []
    for item in data["feedback"]:
        if item["document_id"] not in relevant_docs:
            excluded.append(item["id"])
            continue
        key = tuple(tokens(item["text"]))
        groups.setdefault(key, []).append(item)
    analyzed = []
    buckets = {theme["id"]: [] for theme in data["themes"]}
    buckets["other"] = []
    for items in groups.values():
        first = items[0]
        group_id = first["id"]
        matched = [
            theme["id"] for theme in data["themes"]
            if any(find_phrase(first["text"], term) is not None for term in theme["terms"])
        ] or ["other"]
        documents = sorted({item["document_id"] for item in items})
        supporting = []
        theme_terms = {theme["id"]: theme["terms"] for theme in data["themes"]}
        for item in items:
            for theme_id in matched:
                spans = [find_phrase(item["text"], term) for term in theme_terms.get(theme_id, [])]
                span = next((span for span in spans if span is not None), None)
                start = max(0, span[0] - 80) if span else 0
                end = min(len(item["text"]), span[1] + 80) if span else min(240, len(item["text"]))
                supporting.append({
                    "feedback_id": item["id"], "document_id": item["document_id"],
                    "theme_id": theme_id, "excerpt": item["text"][start:end],
                    "start": start, "end": end,
                })
        analyzed.append({
            "id": group_id, "source_ids": [item["id"] for item in items],
            "document_ids": documents,
            "evidence_ids": [hit["evidence_id"] for hit in hits if hit["document_id"] in documents],
            "themes": matched, "supporting_excerpts": supporting,
        })
        for theme_id in matched:
            buckets[theme_id].append(group_id)
    return {
        "scope": "Only feedback linked to documents in the semantic search hits.",
        "groups": analyzed,
        "themes": [{"id": key, "unique_feedback_count": len(ids), "group_ids": ids}
                   for key, ids in buckets.items() if ids],
        "excluded_source_ids": excluded,
        "duplicate_count": sum(len(group["source_ids"]) - 1 for group in analyzed),
    }


def feedback_stage(previous):
    validate(previous, "semantic")
    result = copy.deepcopy(previous)
    result["stage"] = "feedback"
    result["feedback"] = compute_feedback(result)
    return validate(result, "feedback")


def run_pipeline(data, embedder=None):
    return feedback_stage(semantic_stage(review_stage(data), embedder))


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        source = Path(argv[0]).read_text(encoding="utf-8")
        data = json.loads(source, object_pairs_hook=reject_duplicates,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValidationError("nonfinite JSON number")))
        output = run_pipeline(data)
    except (ValidationError, OSError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
