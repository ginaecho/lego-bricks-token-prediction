"""Synthetic, deterministic research -> document automation -> discovery CLI.

Run: python -B implementation.py example_input.json
Research identifies lexical evidence, not verified truth or causal conclusions.
All stages share a versioned state and validate every handoff. No providers used.
"""

import copy
import json
import math
import re
import sys
from pathlib import Path


BUILD_ID = "wave3_t068_combo3_basic-research_basic-documents_basic-recommend"
STAGES = ("input", "research", "documents", "recommend")
STOP_WORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or that the "
    "this to was what which with would".split()
)


class ValidationError(ValueError):
    """A readable, deterministic schema or handoff failure."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(expected), f"{path} must have keys: {', '.join(expected)}")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 20000,
            f"{path} must be nonempty text of at most 20000 characters")


def number(value, path, low=0, high=None):
    require(type(value) in (int, float), f"{path} must be a number, not a boolean")
    require(math.isfinite(value) and value >= low and (high is None or value <= high),
            f"{path} is outside its finite allowed range")


def array(value, path, nonempty=False):
    require(isinstance(value, list) and len(value) <= 1000 and (value or not nonempty),
            f"{path} must be a list of {'1' if nonempty else '0'}..1000 items")


def strings(value, path):
    array(value, path)
    for item in value:
        text(item, path)
    require(len({s.casefold().strip() for s in value}) == len(value),
            f"{path} contains duplicate values")


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)) - STOP_WORDS


def validate_request(request):
    keys(request, ("questions", "sources", "document_options", "profile", "products"), "data.request")
    shapes = {
        "questions": ("id", "text"),
        "sources": ("id", "title", "text", "reliability"),
        "products": ("id", "name", "description", "tags", "price"),
    }
    for field, shape in shapes.items():
        items = request[field]
        array(items, field, nonempty=field == "questions")
        ids = set()
        for index, item in enumerate(items):
            path = f"{field}[{index}]"
            keys(item, shape, path)
            for name in shape:
                if name not in ("reliability", "tags", "price"):
                    text(item[name], f"{path}.{name}")
            require(item["id"] not in ids, f"{field} contains duplicate id")
            ids.add(item["id"])
            if field == "questions":
                require(bool(tokens(item["text"])), f"{path}.text has no searchable terms")
            if field == "sources":
                number(item["reliability"], f"{path}.reliability", high=1)
            if field == "products":
                strings(item["tags"], f"{path}.tags")
                number(item["price"], f"{path}.price", high=1_000_000_000)
    options = request["document_options"]
    keys(options, ("minimum_reliability",), "document_options")
    number(options["minimum_reliability"], "minimum_reliability", high=1)
    profile = request["profile"]
    keys(profile, ("interests", "excluded_tags", "budget", "limit"), "profile")
    strings(profile["interests"], "profile.interests")
    strings(profile["excluded_tags"], "profile.excluded_tags")
    number(profile["budget"], "profile.budget", high=1_000_000_000)
    require(type(profile["limit"]) is int and 1 <= profile["limit"] <= 100,
            "profile.limit must be an integer from 1 to 100")


def _research(data):
    request = data["request"]
    evidence = []
    unanswered = []
    for question in request["questions"]:
        terms = tokens(question["text"])
        matches = []
        for source in request["sources"]:
            matched = sorted(terms & tokens(source["title"] + " " + source["text"]))
            if matched:
                matches.append({
                    "question_id": question["id"],
                    "source_id": source["id"],
                    "source_title": source["title"],
                    "excerpt": source["text"],
                    "matched_terms": matched,
                    "reliability": source["reliability"],
                    "relevance": round(len(matched) / len(terms), 6),
                    "strength": round(len(matched) / len(terms) * source["reliability"], 6),
                })
        matches.sort(key=lambda item: (-item["strength"], item["source_id"]))
        evidence.extend(matches)
        if not matches:
            unanswered.append(question["id"])
    return {
        "evidence": evidence,
        "unanswered_question_ids": unanswered,
        "method": "casefolded exact token overlap; source reliability is supplied, not verified",
    }


def _documents(data):
    minimum = data["request"]["document_options"]["minimum_reliability"]
    rows = []
    dropped = []
    for evidence in data["research"]["evidence"]:
        if evidence["reliability"] < minimum:
            dropped.append({"question_id": evidence["question_id"], "source_id": evidence["source_id"],
                            "reason": "below_minimum_reliability"})
            continue
        rows.append({
            "question_id": evidence["question_id"], "source_id": evidence["source_id"],
            "text": " ".join(evidence["excerpt"].split()),
            "matched_terms": evidence["matched_terms"],
            "reliability": evidence["reliability"], "strength": evidence["strength"],
        })
    summary = []
    for question in data["request"]["questions"]:
        related = [row for row in rows if row["question_id"] == question["id"]]
        summary.append({
            "question_id": question["id"], "question": question["text"],
            "evidence_count": len(related),
            "best_evidence_strength": max((row["strength"] for row in related), default=0),
            "status": "evidence_available" if related else "no_usable_evidence",
        })
    weights = {}
    for row in rows:
        for term in row["matched_terms"]:
            weights[term] = max(weights.get(term, 0), row["strength"])
    return {
        "rows": rows, "question_summary": summary, "dropped_evidence": dropped,
        "discovery_terms": [{"term": term, "weight": weights[term]} for term in sorted(weights)
                            if weights[term] > 0],
        "checks": {"accepted_rows": len(rows), "rejected_rows": len(dropped),
                   "questions_without_usable_evidence": sum(not item["evidence_count"] for item in summary)},
    }


def _recommend(data):
    profile = data["request"]["profile"]
    interests = tokens(" ".join(profile["interests"]))
    excluded_tags = {tag.casefold().strip() for tag in profile["excluded_tags"]}
    weights = {item["term"]: item["weight"] for item in data["documents"]["discovery_terms"]}
    ranked = []
    excluded = []
    for product in data["request"]["products"]:
        reasons = []
        if product["price"] > profile["budget"]:
            reasons.append("over_budget")
        if excluded_tags & {tag.casefold().strip() for tag in product["tags"]}:
            reasons.append("excluded_tag")
        if reasons:
            excluded.append({"product_id": product["id"], "reasons": reasons})
            continue
        product_terms = tokens(" ".join([product["name"], product["description"], *product["tags"]]))
        personal_matches = sorted(interests & product_terms)
        research_matches = sorted(set(weights) & product_terms)
        personal_score = 2 * len(personal_matches)
        research_score = round(sum(weights[term] for term in research_matches), 6)
        supporting = sorted({
            (row["question_id"], row["source_id"])
            for row in data["documents"]["rows"]
            if row["strength"] > 0 and set(row["matched_terms"]) & set(research_matches)
        })
        ranked.append({
            "product_id": product["id"], "name": product["name"], "price": product["price"],
            "score": round(personal_score + research_score, 6),
            "score_components": {"profile": personal_score, "research": research_score},
            "matched_interests": personal_matches, "matched_research_terms": research_matches,
            "supporting_evidence": [{"question_id": qid, "source_id": sid} for qid, sid in supporting],
            "reason": "matched_preferences_or_evidence" if personal_score + research_score else "budget_eligible_fallback",
        })
    ranked.sort(key=lambda item: (-item["score"], item["price"], item["product_id"]))
    excluded.sort(key=lambda item: item["product_id"])
    return {
        "items": ranked[:profile["limit"]], "eligible_count": len(ranked), "excluded": excluded,
        "ranking_policy": "2 per matched profile token + sum of evidence term weights; ties: price, id",
    }


BUILDERS = {"research": _research, "documents": _documents, "recommend": _recommend}


def validate_state(state, expected_stage=None):
    """One schema gate, including canonical output/provenance checks at each stage."""
    keys(state, ("schema_version", "synthetic", "status", "stage", "data"), "state")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "schema_version must be 1")
    require(state["synthetic"] is True, "synthetic must be true for these fixtures")
    require(state["status"] == "ok", "state.status must be ok")
    require(isinstance(state["stage"], str) and state["stage"] in STAGES, "unknown stage")
    if expected_stage is not None:
        require(state["stage"] == expected_stage, f"expected {expected_stage} stage")
    position = STAGES.index(state["stage"])
    keys(state["data"], ("request", *STAGES[1:position + 1]), "state.data")
    validate_request(state["data"]["request"])
    for stage in STAGES[1:position + 1]:
        expected = BUILDERS[stage](state["data"])
        # JSON comparison also distinguishes booleans from numeric output fields.
        require(json.dumps(state["data"][stage], sort_keys=True, allow_nan=False) ==
                json.dumps(expected, sort_keys=True, allow_nan=False),
                f"{stage} output failed deterministic schema/provenance validation")
    return state


def advance(state, stage):
    require(stage in BUILDERS, "unknown destination stage")
    validate_state(state, STAGES[STAGES.index(stage) - 1])
    result = copy.deepcopy(state)
    result["data"][stage] = BUILDERS[stage](result["data"])
    result["stage"] = stage
    return validate_state(result, stage)


def research(state):
    return advance(state, "research")


def documents(state):
    return advance(state, "documents")


def recommend(state):
    return advance(state, "recommend")


def run_pipeline(state):
    return recommend(documents(research(state)))


def reject_constant(value):
    raise ValidationError(f"nonfinite JSON number: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("r", encoding="utf-8") as stream:
            state = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(state)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
