"""Synthetic, deterministic feedback -> search -> research reference pipeline.

Run: python -B implementation.py example_input.json
Library: run_pipeline(payload, embedding=callable). The optional embedding callable
receives one string and returns a finite, nonzero numeric vector. No provider,
network, model, or external dependency is used by this implementation.
"""

import copy
import json
import math
import re
import sys
from collections import Counter


VERSION = "1.0"
STAGES = ("feedback_analysis", "semantic_search", "deep_research")
THEMES = {
    "delivery": {"delivery", "shipping", "late", "arrived", "delay"},
    "price": {"price", "cost", "expensive", "cheap", "value", "affordable"},
    "quality": {"quality", "broken", "durable", "sturdy", "fragile"},
    "usability": {"easy", "difficult", "setup", "instructions", "usable"},
}
STOP = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "i", "in", "is", "it", "of", "on", "or", "the",
        "this", "to", "was", "with"}


class ValidationError(ValueError):
    """Invalid input, stage handoff, or injected component output."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def same_artifact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_artifact(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_artifact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required object fields")
    require(value.keys() <= set(required) | set(optional), "Unknown object fields")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()),
            label + " must be a nonblank string")
    require(len(value) <= 20000, label + " exceeds 20000 characters")
    return value


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def terms(value):
    return set(tokens(value)) - STOP


def normalized(value):
    return " ".join(tokens(value))


def validate_request(request):
    keys(request, ("schema_version", "synthetic", "query", "feedback", "documents"),
         ("limit",))
    require(request["schema_version"] == VERSION, "Unsupported schema_version")
    require(request["synthetic"] is True, "Fixture input must be labeled synthetic")
    text(request["query"], "query")
    require(bool(terms(request["query"])), "query must contain searchable terms")
    limit = request.get("limit", 5)
    require(type(limit) is int and 1 <= limit <= 20, "limit must be 1..20")
    for kind in ("feedback", "documents"):
        rows = request[kind]
        require(isinstance(rows, list) and 1 <= len(rows) <= 1000,
                kind + " must have 1..1000 records")
        seen = set()
        for row in rows:
            if kind == "feedback":
                keys(row, ("id", "text"))
            else:
                keys(row, ("id", "title", "text"), ("claims",))
                text(row["title"], "document title")
            identifier = text(row["id"], kind + " id")
            require(identifier not in seen, "Duplicate " + kind + " id")
            seen.add(identifier)
            text(row["text"], kind + " text")
            require(bool(tokens(row["text"])), kind + " text needs words")
            if kind == "documents":
                claims = row.get("claims", [])
                require(isinstance(claims, list) and len(claims) <= 100,
                        "claims must be an array of at most 100 items")
                for claim in claims:
                    keys(claim, ("topic", "position", "excerpt"))
                    text(claim["topic"], "claim topic")
                    require(bool(normalized(claim["topic"])), "claim topic needs words")
                    require(claim["position"] in ("support", "oppose", "uncertain"),
                            "Invalid claim position")
                    text(claim["excerpt"], "claim excerpt")
                    require(claim["excerpt"] in row["text"],
                            "Claim excerpt must occur verbatim in its document")


def analyze(request):
    groups = {}
    for row in request["feedback"]:
        key = normalized(row["text"])
        if key not in groups:
            groups[key] = {
                "group_id": "feedback-" + str(len(groups) + 1),
                "canonical_text": row["text"],
                "source_ids": [],
                "excerpts": [],
            }
        groups[key]["source_ids"].append(row["id"])
        groups[key]["excerpts"].append({"source_id": row["id"], "text": row["text"]})
    grouped = list(groups.values())
    themed = {}
    for group in grouped:
        found = [name for name, words in THEMES.items()
                 if words & terms(group["canonical_text"])] or ["other"]
        for name in found:
            item = themed.setdefault(name, {
                "theme_id": name, "unique_feedback_count": 0,
                "raw_feedback_count": 0, "group_ids": [], "supporting_excerpts": [],
            })
            item["unique_feedback_count"] += 1
            item["raw_feedback_count"] += len(group["source_ids"])
            item["group_ids"].append(group["group_id"])
            item["supporting_excerpts"].extend(copy.deepcopy(group["excerpts"]))
    return {
        "raw_count": len(request["feedback"]),
        "unique_count": len(grouped),
        "duplicate_count": len(request["feedback"]) - len(grouped),
        "groups": grouped,
        "themes": [themed[key] for key in sorted(themed)],
    }


def search_terms(request, feedback):
    return {
        "query_terms": sorted(terms(request["query"])),
        "feedback_terms": sorted(set().union(
            *(terms(group["canonical_text"]) for group in feedback["groups"]))),
    }


def search(request, feedback, semantic_scores):
    effective = search_terms(request, feedback)
    query_terms = set(effective["query_terms"])
    feedback_terms = set(effective["feedback_terms"])
    index = {}
    for document in request["documents"]:
        for term, count in Counter(tokens(document["title"] + " " + document["text"])).items():
            index.setdefault(term, []).append({
                "document_id": document["id"], "term_frequency": count,
            })
    index = {term: sorted(postings, key=lambda p: p["document_id"])
             for term, postings in sorted(index.items())}
    matched_by_document = {doc["id"]: set() for doc in request["documents"]}
    for term in query_terms | feedback_terms:
        for posting in index.get(term, []):
            matched_by_document[posting["document_id"]].add(term)
    candidates = []
    for document in request["documents"]:
        doc_terms = terms(document["title"] + " " + document["text"])
        matches = matched_by_document[document["id"]]
        query_matches = sorted(query_terms & matches)
        feedback_matches = sorted(feedback_terms & matches)
        lexical = (0.75 * len(query_matches) / len(query_terms)
                   + 0.25 * len(feedback_matches) / max(1, len(feedback_terms)))
        semantic = None if semantic_scores is None else semantic_scores[document["id"]]
        score = lexical if semantic is None else 0.8 * lexical + 0.2 * max(0, semantic)
        score = round(score, 6)
        if score <= 0:
            continue
        candidates.append({
            "document_id": document["id"], "score": score,
            "lexical_score": round(lexical, 6), "embedding_similarity": semantic,
            "matched_query_terms": query_matches,
            "matched_feedback_terms": feedback_matches,
            "feedback_theme_ids": sorted(
                theme["theme_id"] for theme in feedback["themes"]
                if THEMES.get(theme["theme_id"], set()) & doc_terms),
        })
    candidates.sort(key=lambda item: (-item["score"], item["document_id"]))
    results = candidates[:request.get("limit", 5)]
    for rank, result in enumerate(results, 1):
        result["rank"] = rank
    return {
        "method": "lexical+injected-cosine" if semantic_scores is not None else "lexical",
        "effective_query": effective,
        "index": index,
        "semantic_scores": semantic_scores,
        "candidate_count": len(candidates),
        "results": results,
    }


def synthesize(request, search_output):
    documents = {row["id"]: row for row in request["documents"]}
    evidence, grouped = [], {}
    for result in search_output["results"]:
        document = documents[result["document_id"]]
        claims = document.get("claims", [])
        if not claims:
            excerpt = re.split(r"(?<=[.!?])\s+", document["text"], maxsplit=1)[0][:240]
            claims = [{"topic": "document overview", "position": "observation",
                       "excerpt": excerpt}]
        for number, claim in enumerate(claims, 1):
            evidence_id = "evidence-" + str(len(evidence) + 1)
            topic = normalized(claim["topic"])
            item = {
                "evidence_id": evidence_id, "document_id": document["id"],
                "claim_number": number if document.get("claims") else None,
                "topic": topic, "position": claim["position"], "excerpt": claim["excerpt"],
                "search_rank": result["rank"],
                "feedback_theme_ids": list(result["feedback_theme_ids"]),
            }
            evidence.append(item)
            grouped.setdefault(topic, []).append(item)
    findings, disagreements, unresolved = [], [], []
    for topic, items in sorted(grouped.items()):
        positions = sorted({item["position"] for item in items})
        sources = sorted({item["document_id"] for item in items})
        ids = [item["evidence_id"] for item in items]
        disputed = "support" in positions and "oppose" in positions
        assessment = ("disputed" if disputed else
                      "support-only" if "support" in positions else
                      "opposition-only" if "oppose" in positions else "descriptive")
        findings.append({"topic": topic, "assessment": assessment,
                         "positions": positions, "document_ids": sources, "evidence_ids": ids})
        if disputed:
            disagreements.append({
                "topic": topic,
                "supporting_evidence_ids": [i["evidence_id"] for i in items
                                            if i["position"] == "support"],
                "opposing_evidence_ids": [i["evidence_id"] for i in items
                                         if i["position"] == "oppose"],
                "cross_document": any(
                    a["document_id"] != b["document_id"]
                    for a in items if a["position"] == "support"
                    for b in items if b["position"] == "oppose"),
            })
            unresolved.append({"code": "conflicting_evidence", "topic": topic,
                               "question": "What conditions explain the conflicting positions?",
                               "evidence_ids": ids})
        if len(sources) == 1:
            unresolved.append({"code": "single_source", "topic": topic,
                               "question": "Can an independent document corroborate this topic?",
                               "evidence_ids": ids})
        if "uncertain" in positions:
            unresolved.append({"code": "explicit_uncertainty", "topic": topic,
                               "question": "What additional evidence resolves the stated uncertainty?",
                               "evidence_ids": ids})
        if "observation" in positions:
            unresolved.append({"code": "unstructured_evidence", "topic": topic,
                               "question": "Which explicit claims can these excerpts substantiate?",
                               "evidence_ids": ids})
    if not evidence:
        unresolved.append({"code": "no_evidence", "topic": None,
                           "question": "Which documents address the query or feedback?",
                           "evidence_ids": []})
    if search_output["candidate_count"] > len(search_output["results"]):
        unresolved.append({"code": "retrieval_truncated", "topic": None,
                           "question": "Would additional ranked documents change the synthesis?",
                           "evidence_ids": []})
    return {
        "summary": {
            "retrieved_document_count": len(search_output["results"]),
            "evidence_count": len(evidence), "topic_count": len(findings),
            "disputed_topic_count": len(disagreements),
            "interpretation": "Positions are synthetic source annotations, not verified facts.",
        },
        "evidence": evidence, "findings": findings,
        "disagreements": disagreements, "unresolved_questions": unresolved,
    }


def validate_state(state, completed):
    """One validation boundary for initial input and every cumulative handoff.

    Rebuilding deterministic artifacts validates their entire shape, ranking,
    exact excerpts, and all cross-stage references instead of only checking IDs.
    Injected similarity values are checked separately; no callable is rerun.
    """
    require(type(completed) is int and 0 <= completed <= 3, "Invalid stage number")
    keys(state, ("schema_version", "status", "synthetic", "request") + STAGES[:completed])
    require(state["schema_version"] == VERSION and state["status"] == "ok"
            and state["synthetic"] is True, "Invalid pipeline envelope")
    validate_request(state["request"])
    request = state["request"]
    if completed >= 1:
        require(same_artifact(state["feedback_analysis"], analyze(request)),
                "Invalid feedback output or source provenance")
    if completed >= 2:
        result = state["semantic_search"]
        require(isinstance(result, dict) and "semantic_scores" in result,
                "Invalid search output")
        scores = result["semantic_scores"]
        if scores is not None:
            require(isinstance(scores, dict)
                    and set(scores) == {doc["id"] for doc in request["documents"]},
                    "Semantic scores must cover exactly the input documents")
            for value in scores.values():
                require(type(value) in (int, float) and math.isfinite(value)
                        and -1 <= value <= 1, "Invalid semantic similarity")
        require(same_artifact(result, search(request, state["feedback_analysis"], scores)),
                "Invalid search ranking or feedback handoff")
    if completed >= 3:
        require(same_artifact(state["deep_research"], synthesize(request, state["semantic_search"])),
                "Invalid research output or evidence provenance")
    return state


def initialize(request):
    validate_request(request)
    return validate_state({"schema_version": VERSION, "status": "ok", "synthetic": True,
                           "request": copy.deepcopy(request)}, 0)


def feedback_stage(state):
    validate_state(state, 0)
    output = copy.deepcopy(state)
    output["feedback_analysis"] = analyze(output["request"])
    return validate_state(output, 1)


def unit_vector(value):
    require(isinstance(value, (list, tuple)) and 1 <= len(value) <= 4096,
            "Embedding must be a vector with 1..4096 components")
    require(all(type(component) in (int, float) for component in value),
            "Embedding components must be numeric, not booleans")
    try:
        converted = [float(component) for component in value]
    except (ValueError, OverflowError) as exc:
        raise ValidationError("Embedding component cannot be represented") from exc
    require(all(math.isfinite(component) for component in converted),
            "Embedding components must be finite")
    scale = max(abs(component) for component in converted)
    require(scale > 0, "Embedding must be nonzero")
    scaled = [component / scale for component in converted]
    norm = math.sqrt(sum(component * component for component in scaled))
    return [component / norm for component in scaled]


def semantic_stage(state, embedding=None):
    validate_state(state, 1)
    output = copy.deepcopy(state)
    request = output["request"]
    scores = None
    if embedding is not None:
        require(callable(embedding), "embedding must be callable")
        effective = search_terms(request, output["feedback_analysis"])
        query = " ".join(effective["query_terms"]) + "\nFeedback: " + " ".join(
            effective["feedback_terms"])
        try:
            vector = unit_vector(embedding(query))
            scores = {}
            for document in request["documents"]:
                candidate = unit_vector(embedding(document["title"] + "\n" + document["text"]))
                require(len(candidate) == len(vector), "Embedding dimension mismatch")
                cosine = sum(a * b for a, b in zip(vector, candidate))
                scores[document["id"]] = round(max(-1.0, min(1.0, cosine)), 8)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError("Injected embedding callable failed") from exc
    output["semantic_search"] = search(request, output["feedback_analysis"], scores)
    return validate_state(output, 2)


def research_stage(state):
    validate_state(state, 2)
    output = copy.deepcopy(state)
    output["deep_research"] = synthesize(output["request"], output["semantic_search"])
    return validate_state(output, 3)


def run_pipeline(request, embedding=None):
    return research_stage(semantic_stage(feedback_stage(initialize(request)), embedding))


def unique_object(pairs):
    output = {}
    for key, value in pairs:
        require(key not in output, "Duplicate JSON object key: " + key)
        output[key] = value
    return output


def reject_constant(value):
    raise ValidationError("Non-finite JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            request = json.load(handle, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        output = run_pipeline(request)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        output = {"schema_version": VERSION, "status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
