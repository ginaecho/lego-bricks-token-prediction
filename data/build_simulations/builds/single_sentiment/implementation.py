"""Offline sentiment/issue analysis. CLI: python implementation.py input.json."""

import argparse
import json
import math
import re
import sys


DEFAULT_LEXICON = {
    "good": 0.7, "great": 1.0, "helpful": 0.7, "love": 1.0,
    "bad": -0.7, "broken": -1.0, "slow": -0.6, "hate": -1.0,
}
DEFAULT_ISSUES = {
    "performance": ["slow", "latency"],
    "reliability": ["broken", "crash", "error"],
    "support": ["support", "refund"],
}
DEFAULT_NEGATIONS = ["not", "never", "no", "isn't", "don't", "wasn't"]
TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
BOUNDARY = re.compile(r"[.!?;\n]")


class InputError(ValueError):
    """Invalid input or classifier result."""


def number(value, where, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not low <= value <= high or not math.isfinite(value)):
        raise InputError(f"{where} must be a finite number in [{low}, {high}]")
    return float(value)


def word(value, where):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise InputError(f"{where} must be a single word")
    return value.casefold().replace("’", "'")


def fields(value, allowed, where):
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    if set(value) - allowed:
        raise InputError(f"{where} contains unknown fields")


def validate(payload):
    fields(payload, {"feedback", "config"}, "input")
    feedback = payload.get("feedback")
    if not isinstance(feedback, list):
        raise InputError("feedback must be an array")
    config = payload.get("config", {})
    fields(config, {"lexicon", "issues", "negations", "negation_window"}, "config")
    lexicon = config.get("lexicon", DEFAULT_LEXICON)
    if not isinstance(lexicon, dict):
        raise InputError("lexicon must be an object")
    normalized = {}
    for term, score in lexicon.items():
        term = word(term, "lexicon key")
        if term in normalized:
            raise InputError("duplicate normalized lexicon term")
        normalized[term] = number(score, "lexicon score", -1, 1)
    issues = config.get("issues", DEFAULT_ISSUES)
    if not isinstance(issues, dict):
        raise InputError("issues must be an object")
    parsed_issues = {}
    for name, keywords in issues.items():
        if not isinstance(name, str) or not name.strip():
            raise InputError("issue names must be nonblank strings")
        if not isinstance(keywords, list) or not keywords:
            raise InputError("issue keywords must be a nonempty array")
        parsed = []
        for keyword in keywords:
            if not isinstance(keyword, str) or not keyword.strip():
                raise InputError("issue keywords must be nonblank strings")
            parts = keyword.split()
            parsed.append((keyword, [word(part, "keyword term") for part in parts]))
        parsed_issues[name] = parsed
    negations = config.get("negations", DEFAULT_NEGATIONS)
    if not isinstance(negations, list):
        raise InputError("negations must be an array")
    negations = {word(term, "negation") for term in negations}
    window = config.get("negation_window", 3)
    if isinstance(window, bool) or not isinstance(window, int) or not 0 <= window <= 100:
        raise InputError("negation_window must be an integer in [0, 100]")
    seen = set()
    clean = []
    for row in feedback:
        fields(row, {"id", "text", "severity", "urgency"}, "feedback entry")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise InputError("feedback IDs must be unique nonblank strings")
        seen.add(identifier)
        if not isinstance(row.get("text"), str):
            raise InputError("feedback text must be a string")
        clean.append({
            "id": identifier, "text": row["text"],
            "severity": number(row.get("severity", 1), "severity", 0, 5),
            "urgency": number(row.get("urgency", 1), "urgency", 0, 5),
        })
    return clean, normalized, parsed_issues, negations, window


def excerpt(text, start, end):
    left = 0
    right = len(text)
    for match in BOUNDARY.finditer(text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.end()
            break
    return {"text": text[left:right], "start": left, "end": right}


def classify(callback, rows):
    try:
        results = callback([{"id": row["id"], "text": row["text"]} for row in rows])
    except Exception as exc:
        raise InputError("classifier callback failed") from exc
    if not isinstance(results, list):
        raise InputError("classifier must return an array")
    expected = {row["id"] for row in rows}
    validated = {}
    for result in results:
        fields(result, {"id", "label", "confidence"}, "classifier result")
        identifier = result.get("id")
        if not isinstance(identifier, str) or identifier not in expected or identifier in validated:
            raise InputError("classifier IDs must match original IDs exactly once")
        label = result.get("label")
        if not isinstance(label, str) or label not in {"positive", "neutral", "negative"}:
            raise InputError("classifier label must be positive, neutral, or negative")
        confidence = number(result.get("confidence"), "classifier confidence", 0, 1)
        validated[identifier] = {"label": label, "confidence": confidence}
    if set(validated) != expected:
        raise InputError("classifier must return every original feedback ID")
    return validated


def analyze(payload, classifier=None):
    """Analyze JSON-compatible input; optional batch classifier only adds annotations."""
    rows, lexicon, issue_config, negations, window = validate(payload)
    annotations = classify(classifier, rows) if classifier is not None else {}
    results = []
    issue_rows = {name: [] for name in issue_config}
    for row in rows:
        text = row["text"]
        tokens = list(TOKEN.finditer(text))
        words = [word(token.group(), "token") for token in tokens]
        evidence = []
        for index, term in enumerate(words):
            if term not in lexicon:
                continue
            preceding = []
            for previous in range(index - 1, max(-1, index - window - 1), -1):
                if BOUNDARY.search(text[tokens[previous].end():tokens[index].start()]):
                    break
                if words[previous] in negations:
                    preceding.append(words[previous])
            negated = len(preceding) % 2 == 1
            base = lexicon[term]
            evidence.append({
                "term": term, "start": tokens[index].start(), "end": tokens[index].end(),
                "base_score": base, "negations": preceding, "negated": negated,
                "effective_score": -base if negated else base,
                "excerpt": excerpt(text, tokens[index].start(), tokens[index].end()),
            })
        total = sum(item["effective_score"] for item in evidence)
        score = total / len(evidence) if evidence else 0.0
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        result = {
            **row, "status": "empty_text" if not text.strip() else "analyzed",
            "sentiment": {"label": label, "score": score, "sum": total,
                          "matched_terms": len(evidence), "evidence": evidence},
        }
        if classifier is not None:
            result["classifier"] = annotations[row["id"]]
        results.append(result)
        for name, keywords in issue_config.items():
            matches = []
            for keyword, parts in keywords:
                for index in range(len(tokens) - len(parts) + 1):
                    last = index + len(parts) - 1
                    if words[index:index + len(parts)] != parts:
                        continue
                    if any(not text[tokens[i].end():tokens[i + 1].start()].isspace()
                           for i in range(index, last)):
                        continue
                    matches.append({
                        "keyword": keyword, "start": tokens[index].start(),
                        "end": tokens[last].end(),
                        "excerpt": excerpt(text, tokens[index].start(), tokens[last].end()),
                    })
            if matches:
                issue_rows[name].append({
                    "feedback_id": row["id"], "severity": row["severity"],
                    "urgency": row["urgency"], "negative_intensity": max(0.0, -score),
                    "matches": matches,
                })
    issues = []
    for name, sources in issue_rows.items():
        if not sources:
            continue
        frequency = len(sources)
        severity = sum(source["severity"] for source in sources) / frequency
        urgency = sum(source["urgency"] for source in sources) / frequency
        negative = sum(source["negative_intensity"] for source in sources) / frequency
        issues.append({
            "issue": name, "frequency": frequency, "mean_severity": severity,
            "mean_urgency": urgency, "mean_negative_intensity": negative,
            "priority_score": frequency * (severity + urgency) * (1 + negative),
            "sources": sources,
        })
    issues.sort(key=lambda issue: (-issue["priority_score"], issue["issue"]))
    return {
        "status": "ok" if rows else "empty",
        "feedback_count": len(rows), "feedback": results, "prioritized_issues": issues,
        "method": {
            "sentiment": "sum(effective lexicon scores) / matched term count; zero if none",
            "priority": "frequency * (mean_severity + mean_urgency) * (1 + mean_negative_intensity)",
            "frequency": "unique matching feedback IDs",
            "ties": "issue name ascending, case-sensitive; feedback preserves input order",
            "classifier": "annotations only; never changes lexicon scores or issue priorities",
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise InputError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="UTF-8 JSON input file")
    parser.add_argument("--output", help="optional UTF-8 JSON output file (default stdout)")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as source:
            payload = json.load(source, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = json.dumps(analyze(payload), ensure_ascii=True, allow_nan=False, indent=2) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as destination:
                destination.write(output)
        else:
            sys.stdout.write(output)
    except (InputError, OSError, UnicodeError, ValueError, OverflowError) as exc:
        sys.stderr.write(json.dumps({"status": "error", "error": str(exc)}) + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
