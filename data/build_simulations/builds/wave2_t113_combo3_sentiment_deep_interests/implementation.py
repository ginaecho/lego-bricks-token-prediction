"""Deterministic synthetic customer-insights -> research -> discovery CLI.

The manifest documents the version-1 contract. No network, model, or third-party
dependency is used. Evidence quotes are verbatim, not generated summaries.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


SEVERITY = {"low": 10, "medium": 30, "high": 60, "critical": 90}
LEXICON = {
    "good": 1, "great": 2, "excellent": 3, "love": 2, "helpful": 1,
    "fast": 1, "happy": 1, "bad": -1, "terrible": -3, "hate": -2,
    "broken": -2, "slow": -1, "unsafe": -3, "disappointed": -2,
}
NEGATORS = {"not", "never", "no"}
STANCE = ("support", "refute", "uncertain")
MATCH = {"token": str, "weight": int, "negated": bool, "contribution": int}
ISSUE = {
    "id": str, "topic": str, "severity": tuple(SEVERITY),
    "score": int, "label": ("positive", "neutral", "negative"),
    "matches": [MATCH], "priority": int,
}
EVIDENCE = {"document_id": str, "stance": STANCE, "quote": str}
FINDING = {
    "claim": str, "status": ("supported", "refuted", "contested", "uncertain"),
    "evidence": [EVIDENCE],
}
TOPIC = {
    "topic": str, "issue_ids": [str], "priority": int,
    "negative_issue_count": int, "document_ids": [str],
    "findings": [FINDING], "disagreements": [str], "unresolved_questions": [str],
}
RECOMMENDATION = {
    "item_id": str, "title": str, "score": int, "interest_score": int,
    "research_score": int, "matched_interests": [str], "matched_topics": [str],
    "issue_ids": [str], "document_ids": [str], "explanation": str,
    "cautions": [str],
}
SCHEMAS = {
    "input": {
        "schema_version": int, "fixture_label": str,
        "feedback": [{"id": str, "topic": str, "text": str,
                      "severity": tuple(SEVERITY)}],
        "documents": [{
            "id": str, "title": str, "text": str,
            "evidence": [{"topic": str, "claim": str, "stance": STANCE, "quote": str}],
        }],
        "preferences": {
            "interests": [{"tag": str, "weight": int}],
            "excluded_tags": [str], "excluded_item_ids": [str], "limit": int,
        },
        "catalog": [{"id": str, "title": str, "tags": [str], "topics": [str]}],
    },
    "sentiment": {"issues": [ISSUE]},
    "research": {"topics": [TOPIC]},
    "interests": {"ranked": [RECOMMENDATION], "excluded_item_ids": [str]},
}
SCHEMAS["result"] = {
    "status": ("ok",), "schema_version": int, "fixture_label": str,
    "sentiment": SCHEMAS["sentiment"], "research": SCHEMAS["research"],
    "recommendations": SCHEMAS["interests"],
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def _shape(value, spec, path):
    if isinstance(spec, dict):
        require(type(value) is dict, f"{path}: expected object")
        require(set(value) == set(spec), f"{path}: missing or unknown fields")
        for key, child in spec.items():
            _shape(value[key], child, f"{path}.{key}")
    elif isinstance(spec, list):
        require(type(value) is list, f"{path}: expected array")
        require(len(value) <= 1000, f"{path}: at most 1000 entries")
        for index, item in enumerate(value):
            _shape(item, spec[0], f"{path}[{index}]")
    elif isinstance(spec, tuple):
        require(type(value) is str and value in spec, f"{path}: invalid choice")
    else:
        require(type(value) is spec, f"{path}: expected {spec.__name__}")
        if spec is str:
            require(bool(value.strip()) and len(value) <= 100000,
                    f"{path}: expected nonblank string of at most 100000 characters")


def _unique(values, path):
    require(len(values) == len(set(values)), f"{path}: duplicate values")


def validate(name, value, source=None, previous=None):
    """One validation layer for input, output, and both stage boundaries.

    Intermediate contracts are checked against deterministic derivations, so a
    well-shaped but forged priority, citation, or exclusion cannot cross a seam.
    """
    require(name in SCHEMAS, "unknown schema")
    _shape(value, SCHEMAS[name], name)
    if name == "input":
        require(value["schema_version"] == 1, "unsupported schema_version")
        for group in ("feedback", "documents", "catalog"):
            _unique([item["id"] for item in value[group]], group)
        prefs = value["preferences"]
        _unique([i["tag"] for i in prefs["interests"]], "preferences.interests")
        _unique(prefs["excluded_tags"], "preferences.excluded_tags")
        _unique(prefs["excluded_item_ids"], "preferences.excluded_item_ids")
        require(0 <= prefs["limit"] <= 100, "preferences.limit: expected 0..100")
        for interest in prefs["interests"]:
            require(1 <= interest["weight"] <= 5, "interest weight: expected 1..5")
        for item in value["catalog"]:
            _unique(item["tags"], f"catalog.{item['id']}.tags")
            _unique(item["topics"], f"catalog.{item['id']}.topics")
        for document in value["documents"]:
            _unique([(e["topic"], e["claim"], e["stance"], e["quote"])
                     for e in document["evidence"]], f"documents.{document['id']}.evidence")
            for evidence in document["evidence"]:
                require(evidence["quote"] in document["text"],
                        f"documents.{document['id']}: evidence quote not found in text")
        return value
    require(source is not None, f"{name}: source input is required")
    validate("input", source)
    if name == "sentiment":
        expected = _score(source)
    elif name == "research":
        require(previous is not None, "research: sentiment handoff required")
        validate("sentiment", previous, source)
        expected = _synthesize(source, previous)
    elif name == "interests":
        require(previous is not None, "interests: research handoff required")
        validate("research", previous, source, _score(source))
        expected = _rank(source, previous)
    else:
        require(value["schema_version"] == source["schema_version"]
                and value["fixture_label"] == source["fixture_label"],
                "result: metadata differs from input")
        validate("sentiment", value["sentiment"], source)
        validate("research", value["research"], source, value["sentiment"])
        validate("interests", value["recommendations"], source, value["research"])
        return value
    require(value == expected, f"{name}: derived values or provenance are inconsistent")
    return value


def _score(source):
    issues = []
    for feedback in source["feedback"]:
        matches, recent = [], []
        for token in re.findall(r"[a-z]+|[.!?;,:]", feedback["text"].lower()):
            if token in ".!?;,:":
                recent = []
                continue
            if token in LEXICON:
                negated = sum(word in NEGATORS for word in recent[-3:]) % 2 == 1
                weight = LEXICON[token]
                matches.append({"token": token, "weight": weight, "negated": negated,
                                "contribution": -weight if negated else weight})
            recent.append(token)
        score = max(-10, min(10, sum(match["contribution"] for match in matches)))
        issues.append({
            "id": feedback["id"], "topic": feedback["topic"],
            "severity": feedback["severity"], "score": score,
            "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
            "matches": matches, "priority": SEVERITY[feedback["severity"]] + max(0, -score),
        })
    return {"issues": sorted(issues, key=lambda item: (-item["priority"], item["id"]))}


def sentiment_stage(source):
    validate("input", source)
    return validate("sentiment", _score(source), source)


def _finding_status(evidence):
    stances = {entry["stance"] for entry in evidence}
    if "support" in stances and "refute" in stances:
        return "contested"
    if "uncertain" in stances:
        return "uncertain"
    return "supported" if "support" in stances else "refuted"


def _synthesize(source, sentiment):
    groups = {}
    for issue in sentiment["issues"]:
        groups.setdefault(issue["topic"], []).append(issue)
    topics = []
    for topic, issues in groups.items():
        claims = {}
        for document in source["documents"]:
            for evidence in document["evidence"]:
                if evidence["topic"] == topic:
                    claims.setdefault(evidence["claim"], []).append({
                        "document_id": document["id"], "stance": evidence["stance"],
                        "quote": evidence["quote"],
                    })
        findings, disagreements, questions = [], [], []
        documents = set()
        for claim in sorted(claims):
            evidence = sorted(claims[claim],
                              key=lambda e: (e["document_id"], e["stance"], e["quote"]))
            status = _finding_status(evidence)
            documents.update(e["document_id"] for e in evidence)
            findings.append({"claim": claim, "status": status, "evidence": evidence})
            if status == "contested":
                disagreements.append(claim)
                questions.append(f"Which evidence resolves the disagreement about: {claim}?")
            if any(e["stance"] == "uncertain" for e in evidence):
                questions.append(f"What additional evidence would clarify: {claim}?")
            if len({e["document_id"] for e in evidence}) == 1:
                questions.append(f"Can an independent document corroborate: {claim}?")
        if not findings:
            questions.append(f"What evidence addresses customer issues about {topic}?")
        topics.append({
            "topic": topic, "issue_ids": [issue["id"] for issue in issues],
            "priority": max(issue["priority"] for issue in issues),
            "negative_issue_count": sum(issue["label"] == "negative" for issue in issues),
            "document_ids": sorted(documents), "findings": findings,
            "disagreements": disagreements, "unresolved_questions": questions,
        })
    return {"topics": sorted(topics, key=lambda item: (-item["priority"], item["topic"]))}


def research_stage(source, sentiment):
    validate("sentiment", sentiment, source)
    return validate("research", _synthesize(source, sentiment), source, sentiment)


def _rank(source, research):
    prefs = source["preferences"]
    interests = {interest["tag"]: interest["weight"] for interest in prefs["interests"]}
    excluded_tags, excluded_ids = set(prefs["excluded_tags"]), set(prefs["excluded_item_ids"])
    grounded = {topic["topic"]: topic for topic in research["topics"] if topic["findings"]}
    ranked, excluded = [], []
    for item in source["catalog"]:
        if item["id"] in excluded_ids or excluded_tags.intersection(item["tags"]):
            excluded.append(item["id"])
            continue
        tags = sorted(set(item["tags"]).intersection(interests))
        topics = sorted(set(item["topics"]).intersection(grounded))
        if not tags or not topics:
            continue
        interest_score = 100 * sum(interests[tag] for tag in tags)
        # The bounded research bonus cannot outweigh one unit of stated interest.
        research_score = max(grounded[topic]["priority"] for topic in topics) // 2
        issue_ids = sorted({i for topic in topics for i in grounded[topic]["issue_ids"]})
        document_ids = sorted({d for topic in topics for d in grounded[topic]["document_ids"]})
        cautions = sorted({q for topic in topics for q in grounded[topic]["unresolved_questions"]})
        for topic in topics:
            for finding in grounded[topic]["findings"]:
                if finding["status"] == "refuted":
                    cautions.append(f"Refuted claim on {topic}: {finding['claim']}")
        ranked.append({
            "item_id": item["id"], "title": item["title"],
            "score": interest_score + research_score,
            "interest_score": interest_score, "research_score": research_score,
            "matched_interests": tags, "matched_topics": topics,
            "issue_ids": issue_ids, "document_ids": document_ids,
            "explanation": (
                f"Matches stated interests {', '.join(tags)} (+{interest_score}); "
                f"covers researched topics {', '.join(topics)} (+{research_score} priority). "
                f"Customer issues: {', '.join(issue_ids)}. "
                f"Evidence documents: {', '.join(document_ids)}. "
                "This is a topical discovery match, not proof of item effectiveness."
            ),
            "cautions": sorted(set(cautions)),
        })
    ranked.sort(key=lambda item: (-item["score"], item["item_id"]))
    return {"ranked": ranked[:prefs["limit"]], "excluded_item_ids": sorted(excluded)}


def interests_stage(source, research):
    validate("research", research, source, _score(validate("input", source)))
    return validate("interests", _rank(source, research), source, research)


def run_pipeline(source):
    validate("input", source)
    sentiment = sentiment_stage(source)
    research = research_stage(source, sentiment)
    recommendations = interests_stage(source, research)
    return validate("result", {
        "status": "ok", "schema_version": 1, "fixture_label": source["fixture_label"],
        "sentiment": sentiment, "research": research, "recommendations": recommendations,
    }, source)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError(f"non-finite JSON number: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 2_000_000, "input file exceeds 2000000 bytes")
        source = json.loads(path.read_text(encoding="utf-8-sig"),
                            object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
        result = run_pipeline(source)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
