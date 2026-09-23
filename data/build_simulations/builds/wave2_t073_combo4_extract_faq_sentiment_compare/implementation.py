"""Synthetic, deterministic extraction -> FAQ -> sentiment -> comparison reference.

Run: python -B implementation.py example_input.json
Regex schemas are trusted configuration, not an untrusted regex execution service.
All offsets are zero-based, half-open Python Unicode character offsets.
"""

import copy
import json
import math
import re
import sys


VERSION = "1.0"
STAGES = ("input", "extract", "faq", "sentiment", "compare")
UNITS = {
    "price": {"usd": 1, "cents": 0.01},
    "battery": {"hours": 1, "minutes": 1 / 60},
    "warranty": {"months": 1, "years": 12},
}
CANONICAL = {"price": "USD", "battery": "hours", "warranty": "months"}
STOP = {"a", "an", "the", "is", "are", "i", "my", "to", "and", "of",
        "it", "can", "do", "how", "what", "for", "with", "this", "please"}
LEXICON = {"love": 2, "great": 2, "good": 1, "happy": 2,
           "bad": -1, "broken": -2, "angry": -2, "unsafe": -3,
           "danger": -3, "disappointed": -2, "refund": -1}
SEVERITY = {"critical": {"smoke", "fire", "injury", "unsafe", "danger"},
            "high": {"broken", "unusable", "fraud"}}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "Unexpected or missing object keys")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold())) - STOP


def validate_request(request):
    keys(request, ("schema_version", "fixture_label", "document", "fields",
                   "question_field", "knowledge_base", "retrieval_threshold",
                   "products", "preferences"))
    require(request["schema_version"] == VERSION, "Unsupported schema version")
    require(text(request["fixture_label"]), "fixture_label must be nonempty")
    require(isinstance(request["document"], str), "document must be a string")
    fields = request["fields"]
    require(isinstance(fields, list) and fields, "fields must be a nonempty list")
    names = set()
    for field in fields:
        keys(field, ("name", "pattern", "type", "required"))
        require(text(field["name"]) and field["name"] not in names,
                "Field names must be unique nonempty strings")
        names.add(field["name"])
        require(field["type"] in ("string", "integer", "number"), "Invalid field type")
        require(type(field["required"]) is bool, "required must be boolean")
        require(text(field["pattern"]), "pattern must be nonempty")
        try:
            re.compile(field["pattern"], re.MULTILINE)
        except re.error as exc:
            raise ValidationError("Invalid extraction pattern") from exc
    require(isinstance(request["question_field"], str)
            and request["question_field"] in names, "Unknown question_field")
    require(next(f for f in fields if f["name"] == request["question_field"])["type"]
            == "string", "question_field must have string type")
    threshold = request["retrieval_threshold"]
    require(number(threshold) and 0 < threshold <= 1, "Invalid retrieval threshold")
    require(isinstance(request["knowledge_base"], list), "knowledge_base must be a list")
    ids = set()
    for article in request["knowledge_base"]:
        keys(article, ("id", "question", "answer"))
        require(all(text(article[k]) for k in article), "Invalid knowledge-base article")
        require(article["id"] not in ids, "Duplicate article id")
        ids.add(article["id"])
    require(isinstance(request["products"], list), "products must be a list")
    ids = set()
    for product in request["products"]:
        keys(product, ("id", "name", "attributes"))
        require(text(product["id"]) and text(product["name"]), "Invalid product identity")
        require(product["id"] not in ids, "Duplicate product id")
        ids.add(product["id"])
        require(isinstance(product["attributes"], dict), "attributes must be an object")
        require(set(product["attributes"]) <= set(UNITS), "Unknown product attribute")
        for attribute, quantity in product["attributes"].items():
            keys(quantity, ("value", "unit"))
            require(number(quantity["value"]) and quantity["value"] >= 0,
                    "Attribute values must be nonnegative finite numbers")
            require(isinstance(quantity["unit"], str)
                    and quantity["unit"].casefold() in UNITS[attribute],
                    "Unsupported attribute unit")
            normalized = quantity["value"] * UNITS[attribute][quantity["unit"].casefold()]
            require(number(normalized), "Normalized value is not finite")
    require(isinstance(request["preferences"], list) and request["preferences"],
            "preferences must be nonempty")
    seen = set()
    for preference in request["preferences"]:
        keys(preference, ("attribute", "direction", "weight"))
        require(isinstance(preference["attribute"], str)
                and preference["attribute"] in UNITS
                and preference["attribute"] not in seen, "Invalid preference attribute")
        seen.add(preference["attribute"])
        require(preference["direction"] in ("min", "max"), "Invalid preference direction")
        require(number(preference["weight"]) and 0 < preference["weight"] <= 100,
                "Preference weight must be in (0, 100]")


def validate(envelope, expected=None):
    """The single validation boundary used by every producer and consumer."""
    keys(envelope, ("schema_version", "status", "stage", "request", "data"))
    require(envelope["schema_version"] == VERSION and envelope["status"] == "ok",
            "Invalid envelope")
    require(envelope["stage"] in STAGES, "Unknown stage")
    require(expected is None or envelope["stage"] == expected, "Wrong predecessor stage")
    request = envelope["request"]
    validate_request(request)
    index = STAGES.index(envelope["stage"])
    data = envelope["data"]
    keys(data, STAGES[1:index + 1])
    if index >= 1:
        extraction = data["extract"]
        keys(extraction, ("fields", "missing_fields", "missing_required", "invalid_fields"))
        require(isinstance(extraction["fields"], dict), "Invalid extracted fields")
        definitions = {field["name"]: field for field in request["fields"]}
        require(set(extraction["fields"]) <= set(definitions), "Unknown extracted field")
        for name, field in extraction["fields"].items():
            keys(field, ("value", "source_text", "span"))
            span = field["span"]
            require(isinstance(span, list) and len(span) == 2
                    and all(type(n) is int for n in span)
                    and 0 <= span[0] < span[1] <= len(request["document"]),
                    "Invalid source span")
            require(field["source_text"] == request["document"][span[0]:span[1]],
                    "Source span does not match document")
            kind = definitions[name]["type"]
            require((kind == "string" and text(field["value"]))
                    or (kind == "integer" and type(field["value"]) is int)
                    or (kind == "number" and number(field["value"])), "Invalid extracted value")
            try:
                expected_value = convert(field["source_text"], kind)
            except ValueError as exc:
                raise ValidationError("Invalid source value") from exc
            require(field["value"] == expected_value, "Extracted value differs from source")
        missing = [name for name in definitions if name not in extraction["fields"]]
        require(extraction["missing_fields"] == missing, "Incorrect missing field report")
        require(extraction["missing_required"] ==
                [name for name in missing if definitions[name]["required"]],
                "Incorrect required field report")
        require(isinstance(extraction["invalid_fields"], list), "Invalid conversion report")
        for failure in extraction["invalid_fields"]:
            keys(failure, ("field", "reason"))
            require(failure["field"] in missing and text(failure["reason"]),
                    "Invalid field failure")
    if index >= 2:
        faq = data["faq"]
        keys(faq, ("query", "status", "answer", "citation", "confidence", "reason"))
        question = data["extract"]["fields"].get(request["question_field"], {})
        require(faq["query"] == question.get("value", ""), "FAQ query lost extraction provenance")
        require(number(faq["confidence"]) and 0 <= faq["confidence"] <= 1,
                "Invalid retrieval confidence")
        if faq["status"] == "answered":
            article = next((a for a in request["knowledge_base"] if a["id"] == faq["citation"]), None)
            require(article is not None and faq["answer"] == article["answer"],
                    "Ungrounded answer")
            require(faq["confidence"] >= request["retrieval_threshold"]
                    and faq["reason"] is None, "Invalid answered state")
        else:
            require(faq["status"] == "abstained" and faq["answer"] is None
                    and faq["citation"] is None and text(faq["reason"]), "Invalid abstention")
    if index >= 3:
        sentiment = data["sentiment"]
        keys(sentiment, ("source_text", "faq_status", "score", "label", "contributions",
                         "severity", "severity_terms", "priority", "priority_reasons"))
        require(sentiment == analyze_sentiment(data["faq"]), "Invalid sentiment handoff")
    if index >= 4:
        require(data["compare"] == compare_products(request, data["sentiment"]),
                "Invalid comparison handoff")
    return envelope


def convert(source, kind):
    if kind == "integer":
        require(re.fullmatch(r"[+-]?\d+", source) is not None, "Not an integer")
        return int(source)
    if kind == "number":
        result = float(source)
        require(number(result), "Not a finite number")
        return result
    return source


def advance(previous, predecessor, stage, output):
    validate(previous, predecessor)
    result = copy.deepcopy(previous)
    result["stage"] = stage
    result["data"][stage] = output
    return validate(result, stage)


def extract(previous):
    validate(previous, "input")
    request = previous["request"]
    result = {"fields": {}, "missing_fields": [], "missing_required": [], "invalid_fields": []}
    for field in request["fields"]:
        match = re.search(field["pattern"], request["document"], re.MULTILINE)
        if match:
            group = "value" if "value" in match.re.groupindex else 0
            source = match.group(group)
            if source and source.strip():
                start, end = match.span(group)
                start += len(source) - len(source.lstrip())
                end -= len(source) - len(source.rstrip())
                source = request["document"][start:end]
                try:
                    result["fields"][field["name"]] = {
                        "value": convert(source, field["type"]),
                        "source_text": source, "span": [start, end]}
                except (ValueError, OverflowError):
                    result["invalid_fields"].append(
                        {"field": field["name"], "reason": "Type conversion failed"})
        if field["name"] not in result["fields"]:
            result["missing_fields"].append(field["name"])
            if field["required"]:
                result["missing_required"].append(field["name"])
    return advance(previous, "input", "extract", result)


def faq(previous):
    validate(previous, "extract")
    request = previous["request"]
    query = previous["data"]["extract"]["fields"].get(request["question_field"], {}).get("value", "")
    query_tokens = tokens(query)
    candidates = []
    for article in request["knowledge_base"]:
        score = len(query_tokens & tokens(article["question"])) / len(query_tokens) if query_tokens else 0
        candidates.append((score, article["id"], article))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best = candidates[0] if candidates else None
    result = {"query": query, "status": "abstained", "answer": None, "citation": None,
              "confidence": best[0] if best else 0, "reason": "No sufficiently relevant article"}
    if not query_tokens:
        result["reason"] = "Missing or non-informative extracted question"
    elif best and best[0] >= request["retrieval_threshold"]:
        result.update(status="answered", answer=best[2]["answer"], citation=best[1], reason=None)
    return advance(previous, "extract", "faq", result)


def analyze_sentiment(answer):
    words = re.findall(r"[a-z0-9]+", answer["query"].casefold())
    contributions = []
    for index, word in enumerate(words):
        if word in LEXICON:
            negated = index > 0 and words[index - 1] in ("not", "never", "no")
            contributions.append({"token": word, "token_index": index,
                                  "negated": negated,
                                  "score": -LEXICON[word] if negated else LEXICON[word]})
    score = sum(item["score"] for item in contributions)
    severity, matches = "normal", []
    for level, triggers in SEVERITY.items():
        found = sorted(set(words) & triggers)
        if found:
            severity, matches = level, found
            break
    priority = 100 if severity == "critical" else 70 if severity == "high" else 30
    reasons = ["severity:" + severity]
    if score < 0:
        priority += 10
        reasons.append("negative_sentiment:+10")
    if answer["status"] == "abstained":
        priority += 10
        reasons.append("unanswered:+10")
    return {"source_text": answer["query"], "faq_status": answer["status"], "score": score,
            "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
            "contributions": contributions, "severity": severity, "severity_terms": matches,
            "priority": min(priority, 100), "priority_reasons": reasons}


def sentiment(previous):
    validate(previous, "faq")
    return advance(previous, "faq", "sentiment", analyze_sentiment(previous["data"]["faq"]))


def compare_products(request, insight):
    preferences = copy.deepcopy(request["preferences"])
    adjustments = []
    if insight["priority"] >= 70:
        warranty = next((p for p in preferences if p["attribute"] == "warranty"), None)
        if warranty is None:
            preferences.append({"attribute": "warranty", "direction": "max", "weight": 2})
            adjustments.append("High-priority issue adds warranty preference (weight 2)")
        elif warranty["direction"] == "max":
            warranty["weight"] += 2
            adjustments.append("High-priority issue increases warranty weight by 2")
        else:
            adjustments.append("Explicit minimize-warranty preference preserved")
    rows = []
    for product in request["products"]:
        attributes = {}
        for name in UNITS:
            raw = product["attributes"].get(name)
            attributes[name] = (raw["value"] * UNITS[name][raw["unit"].casefold()]
                                if raw is not None else None)
        rows.append({"id": product["id"], "name": product["name"], "attributes": attributes})
    ranges = {}
    for preference in preferences:
        name = preference["attribute"]
        values = [row["attributes"][name] for row in rows if row["attributes"][name] is not None]
        ranges[name] = (min(values), max(values)) if values else (None, None)
    ranking = []
    total_weight = sum(p["weight"] for p in preferences)
    for row in rows:
        contributions, weighted = [], 0
        for preference in preferences:
            name = preference["attribute"]
            value = row["attributes"][name]
            low, high = ranges[name]
            utility = 0
            if value is not None:
                utility = 1 if low == high else (value - low) / (high - low)
                if low != high and preference["direction"] == "min":
                    utility = 1 - utility
            weighted += utility * preference["weight"]
            contributions.append({"attribute": name, "utility": utility,
                                  "weight": preference["weight"], "missing": value is None})
        ranking.append({"id": row["id"], "score": round(weighted / total_weight, 6),
                        "contributions": contributions})
    ranking.sort(key=lambda row: (-row["score"], row["id"]))
    return {"issue_priority": insight["priority"], "issue_severity": insight["severity"],
            "faq_status": insight["faq_status"], "units": CANONICAL.copy(),
            "side_by_side": rows, "effective_preferences": preferences,
            "adjustments": adjustments, "ranking": ranking,
            "recommended_product_id": ranking[0]["id"] if ranking else None}


def compare(previous):
    validate(previous, "sentiment")
    return advance(previous, "sentiment", "compare",
                   compare_products(previous["request"], previous["data"]["sentiment"]))


def run_pipeline(request):
    current = validate({"schema_version": VERSION, "status": "ok", "stage": "input",
                        "request": copy.deepcopy(request), "data": {}})
    for stage in (extract, faq, sentiment, compare):
        current = stage(current)
    return current


def reject_constant(value):
    raise ValidationError("Non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            request = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(request)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
