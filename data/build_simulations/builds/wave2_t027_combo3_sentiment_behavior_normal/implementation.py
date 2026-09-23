"""Deterministic synthetic customer-insight -> discovery -> research pipeline.

Run: python -B implementation.py example_input.json
All timestamps are timezone-aware ISO 8601. Source citation offsets are Python
Unicode character offsets, end-exclusive, into the unmodified source text.
"""

import copy
import datetime as dt
import json
import math
import re
import sys


SCHEMA_VERSION = 1
LEXICON = {
    "love": 2, "great": 2, "excellent": 3, "good": 1, "helpful": 1,
    "bad": -1, "broken": -3, "poor": -2, "hate": -2, "slow": -1,
    "unsafe": -4, "dangerous": -4,
}
SEVERITY = {"low": 10, "medium": 35, "high": 65, "critical": 90}
NEGATIONS = {"not", "never", "no"}
STOPWORDS = {"the", "a", "an", "is", "and", "to", "of", "for", "i", "it"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(names.split()), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def number(value, path, minimum=0, maximum=None):
    require(type(value) in (int, float), path + " must be numeric")
    require(type(value) is int or math.isfinite(value), path + " must be finite numeric")
    require(value >= minimum and (maximum is None or value <= maximum), path + " out of range")


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "timestamp must include timezone")
    return parsed


def tokens(value):
    return re.findall(r"[a-z]+", value.lower())


def validate_input(data):
    fields(data, "schema_version synthetic as_of customer_id feedback events catalog sources", "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "unsupported schema_version")
    require(data["synthetic"] is True, "this reference accepts labeled synthetic data only")
    text(data["customer_id"], "customer_id")
    now = timestamp(data["as_of"])
    layouts = {
        "feedback": "id customer_id category text severity",
        "events": "id customer_id product_id kind at",
        "catalog": "id name category description popularity",
        "sources": "id title text",
    }
    for collection, layout in layouts.items():
        require(isinstance(data[collection], list), collection + " must be a list")
        seen = set()
        for item in data[collection]:
            fields(item, layout, collection)
            for key in layout.split():
                if key != "popularity":
                    text(item[key], collection + "." + key)
            require(item["id"] not in seen, collection + " has duplicate id")
            seen.add(item["id"])
    products = {p["id"]: p for p in data["catalog"]}
    categories = {p["category"] for p in products.values()}
    for product in products.values():
        number(product["popularity"], "popularity", 0, 1)
    for feedback in data["feedback"]:
        require(feedback["customer_id"] == data["customer_id"], "feedback belongs to another customer")
        require(feedback["category"] in categories, "unknown feedback category")
        require(feedback["severity"] in SEVERITY, "unknown severity")
    for event in data["events"]:
        require(event["customer_id"] == data["customer_id"], "event belongs to another customer")
        require(event["product_id"] in products, "unknown event product")
        require(event["kind"] in ("browse", "purchase"), "unknown event kind")
        require(timestamp(event["at"]) <= now, "future event")
    return data


def validate(data, phase="input"):
    """Single validation gateway, used at every pipeline boundary."""
    if phase == "input":
        return validate_input(data)
    require(phase in ("sentiment", "behavior", "research"), "unknown pipeline phase")
    keys = "schema_version status synthetic input sentiment"
    if phase in ("behavior", "research"):
        keys += " behavior"
    if phase == "research":
        keys += " research"
    fields(data, keys, "pipeline")
    require(data["schema_version"] == 1 and type(data["schema_version"]) is int,
            "unsupported pipeline schema")
    require(data["status"] == "ok" and data["synthetic"] is True, "invalid pipeline status")
    original = validate_input(data["input"])
    # Recompute deterministic contracts to reject stale, forged, or inconsistent
    # stage handoffs, not merely well-shaped but semantically invalid objects.
    require(data["sentiment"] == score_sentiment(original), "invalid sentiment handoff")
    if phase in ("behavior", "research"):
        require(data["behavior"] == rank_behavior(original, data["sentiment"]),
                "invalid behavior handoff")
    if phase == "research":
        require(data["research"] == retrieve(original, data["sentiment"], data["behavior"]),
                "invalid research handoff")
    return data


def score_sentiment(data):
    issues = []
    for feedback in data["feedback"]:
        words = tokens(feedback["text"])
        evidence = []
        for index, word in enumerate(words):
            if word not in LEXICON:
                continue
            # Scope is exactly the previous three word tokens; odd negations flip.
            negated = sum(w in NEGATIONS for w in words[max(0, index - 3):index]) % 2 == 1
            contribution = LEXICON[word] * (-1 if negated else 1)
            evidence.append({"token": word, "token_index": index,
                             "lexicon_weight": LEXICON[word], "negated": negated,
                             "contribution": contribution})
        score = sum(item["contribution"] for item in evidence)
        priority = min(100, SEVERITY[feedback["severity"]] + min(10, max(0, -score) * 2))
        issues.append({
            "feedback_id": feedback["id"], "category": feedback["category"],
            "severity": feedback["severity"], "score": score,
            "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "priority": priority, "evidence": evidence,
        })
    issues.sort(key=lambda item: (-item["priority"], item["feedback_id"]))
    return {"issues": issues,
            "policy": "severity base 10/35/65/90 plus min(10, 2*negative magnitude); capped 100"}


def rank_behavior(data, sentiment):
    products = {p["id"]: p for p in data["catalog"]}
    now = timestamp(data["as_of"])
    signals = []
    direct = {key: 0.0 for key in products}
    category_totals = {}
    for event in sorted(data["events"], key=lambda event: event["id"]):
        days = (now - timestamp(event["at"])).total_seconds() / 86400
        weight = (3 if event["kind"] == "purchase" else 1) / (1 + days / 30)
        category = products[event["product_id"]]["category"]
        direct[event["product_id"]] += weight
        category_totals[category] = category_totals.get(category, 0.0) + weight
        signals.append({"event_id": event["id"], "product_id": event["product_id"],
                        "category": category, "age_days": round(days, 8),
                        "weight": round(weight, 8)})
    concerns = {}
    for issue in sentiment["issues"]:
        # Positive comments do not become discovery concerns unless severity is high.
        if issue["score"] < 0 or issue["severity"] in ("high", "critical"):
            concerns.setdefault(issue["category"], []).append(issue)
    rankings = []
    for product in products.values():
        relevant = concerns.get(product["category"], [])
        concern = max((i["priority"] for i in relevant), default=0) / 100
        components = {
            "direct": round(direct[product["id"]], 8),
            "category_affinity": round(category_totals.get(product["category"], 0.0) * 0.25, 8),
            "issue_attention": concern,
            "popularity": round(product["popularity"] * 0.1, 8),
        }
        rankings.append({"product_id": product["id"], "category": product["category"],
                         "score": round(sum(components.values()), 8), "components": components,
                         "issue_ids": sorted(i["feedback_id"] for i in relevant)})
    rankings.sort(key=lambda item: (-item["score"], item["product_id"]))
    return {"cold_start": not bool(data["events"]), "signals": signals,
            "rankings": rankings, "selected_product_ids": [r["product_id"] for r in rankings[:3]],
            "policy": "browse=1 purchase=3; decay=1/(1+age_days/30); category=0.25; attention<=1"}


def passages(source):
    # Blank-line passage boundaries preserve exact original offsets and punctuation.
    content = source["text"]
    boundaries = list(re.finditer(r"\r?\n[ \t]*\r?\n", content))
    begin = 0
    for boundary in boundaries + [None]:
        end = boundary.start() if boundary else len(content)
        segment = content[begin:end]
        trimmed = segment.strip()
        if trimmed:
            start = begin + len(segment) - len(segment.lstrip())
            yield start, start + len(trimmed), trimmed
        begin = boundary.end() if boundary else len(content)


def retrieve(data, sentiment, behavior):
    products = {p["id"]: p for p in data["catalog"]}
    feedback = {f["id"]: f for f in data["feedback"]}
    ranks = {r["product_id"]: r for r in behavior["rankings"]}
    findings = []
    for product_id in behavior["selected_product_ids"]:
        product = products[product_id]
        issue_ids = ranks[product_id]["issue_ids"]
        query = " ".join([product["name"], product["category"], product["description"]] +
                         [feedback[key]["text"] for key in issue_ids])
        query_terms = set(tokens(query)) - STOPWORDS
        candidates = []
        for source in data["sources"]:
            for start, end, passage in passages(source):
                overlap = sorted(query_terms & (set(tokens(passage)) - STOPWORDS))
                if overlap:
                    candidates.append({
                        "finding": passage, "matched_terms": overlap, "retrieval_score": len(overlap),
                        "citation": {"source_id": source["id"], "title": source["title"],
                                     "start": start, "end": end, "quote": passage},
                    })
        candidates.sort(key=lambda c: (-c["retrieval_score"], c["citation"]["source_id"],
                                       c["citation"]["start"]))
        findings.append({"product_id": product_id, "behavior_score": ranks[product_id]["score"],
                         "issue_ids": issue_ids, "query_terms": sorted(query_terms),
                         "status": "found" if candidates else "no_evidence",
                         "findings": candidates[:2]})
    return {"results": findings, "policy": "unique lexical overlap; top 2 passages per selected product",
            "limitations": "Extractive synthetic evidence only; retrieval does not verify source truth."}


def sentiment_stage(raw):
    validate(raw)
    result = {"schema_version": 1, "status": "ok", "synthetic": True,
              "input": copy.deepcopy(raw), "sentiment": score_sentiment(raw)}
    return validate(result, "sentiment")


def behavior_stage(previous):
    validate(previous, "sentiment")
    result = copy.deepcopy(previous)
    result["behavior"] = rank_behavior(result["input"], result["sentiment"])
    return validate(result, "behavior")


def research_stage(previous):
    validate(previous, "behavior")
    result = copy.deepcopy(previous)
    result["research"] = retrieve(result["input"], result["sentiment"], result["behavior"])
    return validate(result, "research")


def run_pipeline(data):
    return research_stage(behavior_stage(sentiment_stage(data)))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 ValidationError("non-finite JSON constant: " + value)))
        output = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
