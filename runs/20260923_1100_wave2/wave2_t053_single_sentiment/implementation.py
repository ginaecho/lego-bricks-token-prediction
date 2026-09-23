"""Deterministic, English-only customer sentiment and severity prioritization."""

import json
import re
import sys
from pathlib import Path


LEXICON = {
    "amazing": 3, "excellent": 3, "love": 3, "great": 2, "good": 1,
    "helpful": 2, "happy": 2, "fast": 1, "resolved": 2,
    "bad": -1, "broken": -2, "terrible": -3, "hate": -3,
    "slow": -1, "frustrated": -2, "unusable": -3, "failed": -2,
    "awful": -3, "disappointed": -2,
}
SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't"}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.!?,;:]")


class ValidationError(ValueError):
    pass


def exact_fields(value, fields, location):
    if not isinstance(value, dict):
        raise ValidationError(f"{location} must be an object")
    if set(value) != set(fields):
        raise ValidationError(f"{location} requires exactly: {', '.join(fields)}")


def validate(payload):
    """Single validation boundary shared by the Python API and CLI."""
    exact_fields(payload, ("schema_version", "dataset_label", "issues"), "input")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    label = payload["dataset_label"]
    if not isinstance(label, str) or not label.strip() or len(label) > 200:
        raise ValidationError("dataset_label must be a nonblank string of at most 200 characters")
    if not isinstance(payload["issues"], list) or len(payload["issues"]) > 1000:
        raise ValidationError("issues must be a list with at most 1000 entries")
    seen = set()
    for index, issue in enumerate(payload["issues"]):
        location = f"issues[{index}]"
        exact_fields(issue, ("id", "text", "severity", "affected_customers"), location)
        identity = issue["id"]
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 100:
            raise ValidationError(f"{location}.id must be a nonblank string of at most 100 characters")
        if identity != identity.strip() or identity in seen:
            raise ValidationError(f"{location}.id must be unique and have no surrounding whitespace")
        seen.add(identity)
        text = issue["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 10000:
            raise ValidationError(f"{location}.text must be nonblank and at most 10000 characters")
        if not isinstance(issue["severity"], str) or issue["severity"] not in SEVERITY:
            raise ValidationError(f"{location}.severity must be low, medium, high, or critical")
        affected = issue["affected_customers"]
        if type(affected) is not int or not 1 <= affected <= 1000000:
            raise ValidationError(f"{location}.affected_customers must be an integer from 1 to 1000000")
    return payload


def sentiment(text):
    words = TOKEN_PATTERN.findall(text.lower().replace("\u2019", "'"))
    history = []
    evidence = []
    for index, word in enumerate(words):
        if word in ".!?,;:":
            history = []
            continue
        if word in LEXICON:
            base = LEXICON[word]
            negated = sum(item in NEGATORS for item in history[-3:]) % 2 == 1
            evidence.append({
                "token_index": index, "token": word, "base_weight": base,
                "negated": negated, "contribution": -base if negated else base,
            })
        history.append(word)
    raw = sum(item["contribution"] for item in evidence)
    denominator = sum(abs(item["base_weight"]) for item in evidence)
    score = round(raw / denominator, 6) if denominator else 0.0
    label = "positive" if score >= 0.2 else "negative" if score <= -0.2 else "neutral"
    return {
        "label": label, "score": score, "raw_score": raw,
        "normalization_denominator": denominator, "evidence": evidence,
        "matched_terms": len(evidence),
    }


def analyze(payload):
    validate(payload)
    results = []
    for issue in payload["issues"]:
        feeling = sentiment(issue["text"])
        components = {
            "severity": SEVERITY[issue["severity"]] * 100,
            "negative_sentiment": round(max(0, -feeling["score"]) * 40, 6),
            "affected_customers": min(issue["affected_customers"], 20),
        }
        results.append({
            **issue, "sentiment": feeling,
            "priority": {
                "score": round(sum(components.values()), 6),
                "components": components,
                "severity_band": issue["severity"],
            },
        })
    results.sort(key=lambda item: (-item["priority"]["score"], item["id"]))
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for rank, issue in enumerate(results, 1):
        issue["priority"]["rank"] = rank
        counts[issue["sentiment"]["label"]] += 1
    return {
        "status": "ok", "schema_version": 1, "dataset_label": payload["dataset_label"],
        "issues": results, "summary": {"issue_count": len(results), "sentiment_counts": counts},
        "method": {
            "name": "english_lexicon_v1",
            "lexicon": LEXICON,
            "negation": "Odd count of negators in preceding three words, reset by .!?,;:",
            "negators": sorted(NEGATORS),
            "sentiment": "sum(contributions) / sum(abs(base_weights)); zero when no matches",
            "labels": "positive >= 0.2; negative <= -0.2; otherwise neutral",
            "priority": "severity_index * 100 + max(0, -sentiment_score) * 40 + min(affected_customers, 20)",
            "ties": "ascending case-sensitive issue id",
            "limitations": "English lexical heuristic; no sarcasm, semantic context, or severity inference",
        },
    }


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        content = Path(args[0]).read_text(encoding="utf-8")
        payload = json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = analyze(payload)
        code = 0
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        result = {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
