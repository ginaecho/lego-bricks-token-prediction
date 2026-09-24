"""Deterministic reference implementation using synthetic customer feedback."""

import json
import re
import sys


LEXICON = {
    "excellent": 3, "love": 3, "great": 2, "helpful": 2,
    "good": 1, "happy": 2, "thanks": 1,
    "bad": -1, "broken": -2, "slow": -1, "hate": -3,
    "terrible": -3, "awful": -3, "unusable": -3, "frustrating": -2,
}
NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't"}
INTENSIFIERS = {"very": 2, "extremely": 2, "really": 2}
SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class ValidationError(ValueError):
    pass


def validate(payload):
    """The single shared validation boundary for API and CLI inputs."""
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "synthetic", "issues"}:
        raise ValidationError("Expected schema_version, synthetic and issues only")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    if payload["synthetic"] is not True:
        raise ValidationError("synthetic must be true for this fixture-only reference")
    issues = payload["issues"]
    if not isinstance(issues, list) or len(issues) > 1000:
        raise ValidationError("issues must be a list containing at most 1000 entries")
    seen = set()
    for index, issue in enumerate(issues):
        prefix = f"issues[{index}]"
        if not isinstance(issue, dict) or set(issue) != {"id", "text", "severity"}:
            raise ValidationError(f"{prefix} requires id, text and severity only")
        identifier = issue["id"]
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 100:
            raise ValidationError(f"{prefix}.id must be a nonblank string of at most 100 characters")
        if identifier != identifier.strip() or identifier in seen:
            raise ValidationError(f"{prefix}.id must be unique with no surrounding whitespace")
        seen.add(identifier)
        if not isinstance(issue["text"], str) or len(issue["text"]) > 10000:
            raise ValidationError(f"{prefix}.text must be a string of at most 10000 characters")
        if not isinstance(issue["severity"], str) or issue["severity"] not in SEVERITY:
            raise ValidationError(f"{prefix}.severity must be low, medium, high or critical")
    return payload


def score_text(text):
    # Punctuation terminates the short negation/intensifier context.
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[^\w\s]", text.lower().replace("’", "'"))
    context = []
    evidence = []
    for index, token in enumerate(tokens):
        if not token[0].isalpha():
            context = []
            continue
        if token in LEXICON:
            negated = sum(word in NEGATIONS for word in context[-3:]) % 2 == 1
            multiplier = INTENSIFIERS.get(context[-1], 1) if context else 1
            base = LEXICON[token]
            contribution = base * multiplier * (-1 if negated else 1)
            evidence.append({
                "token": token, "token_index": index, "base_weight": base,
                "negated": negated, "multiplier": multiplier,
                "contribution": contribution,
            })
        context.append(token)
    raw = sum(item["contribution"] for item in evidence)
    label = "positive" if raw > 0 else "negative" if raw < 0 else "neutral"
    return {"label": label, "raw_score": raw, "evidence": evidence}


def analyze(payload):
    payload = validate(payload)
    rows = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for issue in payload["issues"]:
        sentiment = score_text(issue["text"])
        counts[sentiment["label"]] += 1
        urgency = min(99, max(0, -sentiment["raw_score"]))
        severity_rank = SEVERITY[issue["severity"]]
        rows.append({
            **issue, "sentiment": sentiment,
            "priority": {
                "score": severity_rank * 100 + urgency,
                "severity_rank": severity_rank, "negative_urgency": urgency,
                "reason": "severity_rank * 100 + min(99, max(0, -raw_score))",
            },
        })
    rows.sort(key=lambda row: (-row["priority"]["score"], row["id"]))
    for rank, row in enumerate(rows, 1):
        row["priority"]["rank"] = rank
    return {
        "status": "ok", "schema_version": 1, "synthetic": True,
        "issues": rows, "summary": {"total": len(rows), "sentiment_counts": counts},
        "method": {
            "name": "english_lexicon_v1", "lexicon": LEXICON.copy(),
            "negation": "Odd negator count in previous three word tokens; punctuation resets context",
            "intensifiers": INTENSIFIERS.copy(),
            "tie_break": "Ascending case-sensitive issue id",
        },
    }


def reject_constant(value):
    raise ValidationError(f"Nonstandard JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = analyze(payload)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
