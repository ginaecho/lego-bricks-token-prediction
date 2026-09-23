"""Synthetic customer-insights/discovery reference pipeline, Python standard library."""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    """A document violates the shared pipeline contract."""


LEXICON = {
    "love": 2, "excellent": 2, "great": 2, "good": 1, "happy": 1,
    "helpful": 1, "bad": -1, "disappointed": -1, "poor": -1,
    "broken": -2, "hate": -2, "unsafe": -3,
}
NEGATORS = {"not", "never", "no"}
SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(expected), f"{path} must have exactly {sorted(expected)}")


def text(value, path, limit=200):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            f"{path} must be a nonblank string of at most {limit} characters")


def integer(value, lower, upper, path):
    require(type(value) is int and lower <= value <= upper,
            f"{path} must be an integer in [{lower}, {upper}]")


def tag(value, path):
    require(isinstance(value, str) and
            re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", value) is not None,
            f"{path} must be a lowercase tag")


def strings(value, path, tags=False):
    require(isinstance(value, list), f"{path} must be an array")
    for entry in value:
        if tags:
            tag(entry, path)
        else:
            text(entry, path, 80)
    require(len(value) == len(set(value)), f"{path} cannot contain duplicates")


def validate_input(value):
    keys(value, {"schema_version", "synthetic", "customer", "feedback", "catalog",
                 "top_k"}, "input")
    integer(value["schema_version"], 1, 1, "schema_version")
    require(value["synthetic"] is True, "synthetic must be true")
    integer(value["top_k"], 1, 20, "top_k")
    customer = value["customer"]
    keys(customer, {"id", "interests", "excluded_tags", "excluded_item_ids"}, "customer")
    text(customer["id"], "customer.id", 80)
    require(isinstance(customer["interests"], dict), "interests must be an object")
    for interest, weight in customer["interests"].items():
        tag(interest, "interest")
        integer(weight, 1, 5, "interest weight")
    strings(customer["excluded_tags"], "excluded_tags", tags=True)
    strings(customer["excluded_item_ids"], "excluded_item_ids")
    for collection, fields in (
        ("feedback", {"id", "text", "severity", "tags"}),
        ("catalog", {"id", "title", "tags"}),
    ):
        require(isinstance(value[collection], list), f"{collection} must be an array")
        ids = []
        for entry in value[collection]:
            keys(entry, fields, collection)
            text(entry["id"], f"{collection}.id", 80)
            strings(entry["tags"], f"{collection}.tags", tags=True)
            ids.append(entry["id"])
            if collection == "feedback":
                text(entry["text"], "feedback.text", 4000)
                require(isinstance(entry["severity"], str) and
                        entry["severity"] in SEVERITY, "unknown feedback severity")
            else:
                text(entry["title"], "catalog.title")
        require(len(ids) == len(set(ids)), f"duplicate {collection} ids")


def analyze_feedback(feedback):
    # Punctuation forms a negation boundary; only an immediately preceding
    # negator reverses a lexicon word. This is deliberately not semantic NLP.
    tokens = re.findall(r"[^\W\d_]+|[.!?,;:]", feedback["text"].lower())
    evidence = []
    for index, token in enumerate(tokens):
        if token not in LEXICON:
            continue
        negated = index > 0 and tokens[index - 1] in NEGATORS
        contribution = LEXICON[token] * (-1 if negated else 1)
        evidence.append({
            "token_index": index, "token": token,
            "lexicon_value": LEXICON[token], "negated": negated,
            "contribution": contribution,
        })
    score = sum(entry["contribution"] for entry in evidence)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {
        "feedback_id": feedback["id"], "severity": feedback["severity"],
        "tags": list(feedback["tags"]), "sentiment_score": score,
        "sentiment": label, "evidence": evidence,
        "priority_score": SEVERITY[feedback["severity"]] * 100 +
                          min(max(-score, 0), 9) * 5,
    }


def build_insights(source):
    issues = [analyze_feedback(feedback) for feedback in source["feedback"]]
    issues.sort(key=lambda issue: (-issue["priority_score"], issue["feedback_id"]))
    concerns = []
    for issue in issues:
        if issue["sentiment_score"] < 0:
            for interest in sorted(issue["tags"]):
                concerns.append({
                    "tag": interest, "feedback_id": issue["feedback_id"],
                    "severity": issue["severity"],
                    "penalty": SEVERITY[issue["severity"]] * 3,
                })
    return {"issues": issues, "concerns": concerns}


def rank_items(source, insights):
    customer = source["customer"]
    recommendations, exclusions = [], []
    for item in source["catalog"]:
        blocked_tags = sorted(set(item["tags"]) & set(customer["excluded_tags"]))
        blocked_id = item["id"] in customer["excluded_item_ids"]
        if blocked_id or blocked_tags:
            exclusions.append({
                "item_id": item["id"], "excluded_by_id": blocked_id,
                "excluded_tags": blocked_tags,
            })
            continue
        preferences = [
            {"tag": interest, "weight": customer["interests"][interest],
             "contribution": customer["interests"][interest] * 10}
            for interest in sorted(item["tags"]) if interest in customer["interests"]
        ]
        if not preferences:
            continue
        concerns = [copy.deepcopy(concern) for concern in insights["concerns"]
                    if concern["tag"] in item["tags"]]
        base_score = sum(entry["contribution"] for entry in preferences)
        penalty = sum(entry["penalty"] for entry in concerns)
        recommendations.append({
            "item_id": item["id"], "title": item["title"],
            "score": base_score - penalty,
            "explanation": {
                "preference_matches": preferences,
                "feedback_concerns": concerns,
                "preference_score": base_score, "concern_penalty": penalty,
            },
        })
    recommendations.sort(key=lambda item: (-item["score"], item["item_id"]))
    exclusions.sort(key=lambda item: item["item_id"])
    return {"recommendations": recommendations[:source["top_k"]],
            "exclusions": exclusions}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=True)


def validate(document, phase="input"):
    """Shared validation for input, the intermediate envelope, and final output.

    Derived fields are checked against the deterministic source rules, including
    evidence, exclusions and rankings, rather than trusting their shape alone.
    """
    require(phase in {"input", "sentiment", "complete"}, "unknown validation phase")
    if phase == "input":
        validate_input(document)
        return document
    fields = {"schema_version", "status", "input", "insights"}
    if phase == "complete":
        fields.add("discovery")
    keys(document, fields, phase)
    integer(document["schema_version"], 1, 1, "schema_version")
    require(document["status"] == ("ok" if phase == "complete" else "sentiment_ready"),
            "invalid pipeline status")
    validate_input(document["input"])
    try:
        require(canonical(document["insights"]) ==
                canonical(build_insights(document["input"])),
                "insights do not match validated feedback")
        if phase == "complete":
            require(canonical(document["discovery"]) ==
                    canonical(rank_items(document["input"], document["insights"])),
                    "discovery does not match validated insights and preferences")
    except (TypeError, ValueError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError("derived output is not valid JSON") from error
    return document


def sentiment_stage(source):
    validate(source)
    output = {
        "schema_version": 1, "status": "sentiment_ready",
        "input": copy.deepcopy(source), "insights": build_insights(source),
    }
    return validate(output, "sentiment")


def interests_stage(previous):
    validate(previous, "sentiment")
    output = copy.deepcopy(previous)
    output["status"] = "ok"
    output["discovery"] = rank_items(output["input"], output["insights"])
    return validate(output, "complete")


def run_pipeline(source):
    return interests_stage(sentiment_stage(source))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            source = json.load(stream, object_pairs_hook=unique_object,
                               parse_constant=reject_constant)
        result = run_pipeline(source)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
