"""Synthetic reference pipeline; Python standard library, deterministic, offline.

Run: python -B implementation.py example_input.json
Field labels match whole line prefixes, case-insensitively. Spans use Python
Unicode character offsets [start, end), not encoded byte offsets. All timestamps
must include a timezone. No clock, random state, or provider is consulted.
"""

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


class ValidationError(ValueError):
    pass


def obj(**properties):
    return ("object", properties)


def array(schema):
    return ("array", schema)


def enum(*values):
    return ("enum", values)


def nullable(schema):
    return ("nullable", schema)


SPEC = obj(label="text", type=enum("string", "number"), required="boolean")
ITEM = obj(id="text", title="text", category="text")
EVENT = obj(customer_id="text", item_id="text",
            kind=enum("browse", "purchase"), at="timestamp")
ARTICLE = obj(id="text", category="text", question="text", answer="text")
INPUT = obj(
    schema_version=enum(1), fixture_label="text", document="string",
    fields=("map", SPEC), as_of="timestamp", items=array(ITEM),
    events=array(EVENT), knowledge_base=array(ARTICLE),
    config=obj(half_life_days="positive", top_k="positive_integer",
               faq_min_score="fraction"),
)
FIELD = obj(value=("union", ("text", "number")), raw="text",
            span=array("integer"))
EXTRACTION = obj(fields=("map", nullable(FIELD)),
                 missing_fields=array("text"), missing_required=array("text"))
CONTEXT = obj(customer_id=nullable("text"), category=nullable("text"),
              question=nullable("text"))
RANKED = obj(id="text", title="text", category="text",
             score="nonnegative", history_score="nonnegative",
             interest_score="nonnegative")
BEHAVIOR = obj(context=CONTEXT, cold_start="boolean",
               matched_events="integer", recommendations=array(RANKED))
FAQ = obj(status=enum("answered", "abstained"), question=nullable("text"),
          category=nullable("text"), answer=nullable("text"),
          citation=nullable(obj(id="text", quote="text")),
          retrieval_score="fraction", reason=nullable("text"))
OUTPUT = obj(schema_version=enum(1), status=enum("ok"),
             extraction=EXTRACTION, behavior=BEHAVIOR, faq=FAQ)
SCHEMAS = {"input": INPUT, "extraction": EXTRACTION,
           "behavior": BEHAVIOR, "faq": FAQ, "output": OUTPUT}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Invalid ISO 8601 timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "Timestamp must include a timezone")
    return parsed


def check(value, schema, path):
    """Single strict, recursive schema engine for all inputs and handoffs."""
    if isinstance(schema, tuple):
        kind, child = schema
        if kind == "nullable":
            if value is not None:
                check(value, child, path)
        elif kind == "union":
            for option in child:
                try:
                    check(value, option, path)
                    return
                except ValidationError:
                    pass
            raise ValidationError(path + ": no matching type")
        elif kind == "enum":
            require(any(type(value) is type(v) and value == v for v in child),
                    path + ": invalid enum value")
        elif kind in ("object", "map"):
            require(type(value) is dict, path + ": expected object")
            if kind == "object":
                require(value.keys() == child.keys(),
                        path + ": missing or unknown keys")
            for key, entry in value.items():
                require(type(key) is str and bool(key.strip()),
                        path + ": invalid property name")
                check(entry, child[key] if kind == "object" else child,
                      path + "." + key)
        elif kind == "array":
            require(type(value) is list, path + ": expected array")
            for index, entry in enumerate(value):
                check(entry, child, f"{path}[{index}]")
        return
    if schema in ("string", "text", "timestamp"):
        require(type(value) is str, path + ": expected string")
        if schema == "text":
            require(bool(value.strip()), path + ": expected nonblank string")
        if schema == "timestamp":
            timestamp(value)
    elif schema == "boolean":
        require(type(value) is bool, path + ": expected boolean")
    elif schema in ("integer", "positive_integer"):
        require(type(value) is int, path + ": expected integer")
        require(value >= (1 if schema == "positive_integer" else 0),
                path + ": integer out of range")
    else:
        require(type(value) in (int, float), path + ": expected number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        require(finite, path + ": expected finite number")
        if schema == "positive":
            require(value > 0, path + ": expected positive number")
        elif schema == "nonnegative":
            require(value >= 0, path + ": expected nonnegative number")
        elif schema == "fraction":
            require(0 <= value <= 1, path + ": expected number in [0, 1]")


def validate(value, stage, source=None):
    check(value, SCHEMAS[stage], stage)
    if stage == "input":
        fields = value["fields"]
        for name in ("customer_id", "category", "question"):
            require(name in fields and fields[name]["type"] == "string",
                    "fields must declare string " + name)
        labels = [s["label"].casefold() for s in fields.values()]
        require(len(set(labels)) == len(labels), "Field labels must be unique")
        require(all("\n" not in s["label"] and "\r" not in s["label"]
                    and s["label"] == s["label"].strip()
                    for s in fields.values()), "Invalid field label")
        for collection in ("items", "knowledge_base"):
            ids = [entry["id"] for entry in value[collection]]
            require(len(ids) == len(set(ids)), collection + ": duplicate IDs")
        ids = {item["id"] for item in value["items"]}
        now = timestamp(value["as_of"])
        for event in value["events"]:
            require(event["item_id"] in ids, "Event references unknown item")
            require(timestamp(event["at"]) <= now, "Future events are invalid")
    elif stage == "extraction":
        require(source is not None, "Extraction validation needs input")
        require(value["fields"].keys() == source["fields"].keys(),
                "Extracted field set differs from input schema")
        missing = []
        for name, field in value["fields"].items():
            if field is None:
                missing.append(name)
                continue
            spec = source["fields"][name]
            check(field["value"], "text" if spec["type"] == "string"
                  else "number", "extraction." + name)
            span = field["span"]
            require(len(span) == 2 and 0 <= span[0] < span[1]
                    <= len(source["document"]), "Invalid source span")
            require(source["document"][span[0]:span[1]] == field["raw"],
                    "Source span does not match raw text")
            expected = (field["raw"] if spec["type"] == "string"
                        else parse_number(field["raw"]))
            require(field["value"] == expected, "Extracted value mismatch")
        require(value["missing_fields"] == sorted(missing),
                "Missing-field report is inconsistent")
        require(value["missing_required"] == sorted(
            name for name in missing if source["fields"][name]["required"]),
            "Required-field report is inconsistent")
    elif stage == "behavior":
        require(value["cold_start"] == (value["matched_events"] == 0),
                "Cold-start flag is inconsistent")
        rows = value["recommendations"]
        require(len({r["id"] for r in rows}) == len(rows),
                "Duplicate recommendation")
        require(rows == sorted(rows, key=lambda r: (-r["score"], r["id"])),
                "Recommendations must be ranked deterministically")
        for row in rows:
            require(math.isclose(row["score"], row["history_score"]
                                 + row["interest_score"]),
                    "Ranking score components do not sum")
    elif stage == "faq":
        if value["status"] == "answered":
            require(value["answer"] is not None and value["citation"] is not None
                    and value["reason"] is None, "Invalid answered state")
            require(source is not None, "FAQ validation needs input")
            article = next((a for a in source["knowledge_base"]
                            if a["id"] == value["citation"]["id"]), None)
            require(article is not None
                    and article["answer"] == value["answer"]
                    == value["citation"]["quote"]
                    and article["category"] == value["category"],
                    "FAQ answer is not grounded in its source")
        else:
            require(value["answer"] is None and value["citation"] is None
                    and value["reason"] is not None, "Invalid abstention state")
    return value


def parse_number(raw):
    require(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw) is not None,
            "Numeric field must contain a plain decimal")
    number = float(raw)
    require(math.isfinite(number), "Numeric field is not finite")
    return number


def extract(data):
    fields = {}
    for name, spec in data["fields"].items():
        pattern = (r"^[ \t]*" + re.escape(spec["label"])
                   + r"[ \t]*:[ \t]*(?P<value>[^\r\n]*)")
        matches = list(re.finditer(pattern, data["document"],
                                   re.MULTILINE | re.IGNORECASE))
        require(len(matches) <= 1, "Ambiguous repeated field: " + name)
        if not matches or not matches[0].group("value").strip():
            fields[name] = None
            continue
        match = matches[0]
        original = match.group("value")
        raw = original.strip()
        start = match.start("value") + len(original) - len(original.lstrip())
        fields[name] = {
            "value": raw if spec["type"] == "string" else parse_number(raw),
            "raw": raw, "span": [start, start + len(raw)],
        }
    missing = sorted(name for name, value in fields.items() if value is None)
    return validate({
        "fields": fields, "missing_fields": missing,
        "missing_required": [n for n in missing if data["fields"][n]["required"]],
    }, "extraction", data)


def personalize(data, extracted):
    validate(extracted, "extraction", data)
    context = {name: (extracted["fields"][name]["value"]
                      if extracted["fields"][name] is not None else None)
               for name in ("customer_id", "category", "question")}
    events = [e for e in data["events"]
              if e["customer_id"] == context["customer_id"]]
    scores = {item["id"]: 0.0 for item in data["items"]}
    now = timestamp(data["as_of"])
    for event in events:
        age = (now - timestamp(event["at"])).total_seconds() / 86400
        decay = 0.5 ** (age / data["config"]["half_life_days"])
        scores[event["item_id"]] += (3 if event["kind"] == "purchase" else 1) * decay
    ranked = []
    for item in data["items"]:
        interest = 0.5 if item["category"] == context["category"] else 0.0
        ranked.append(dict(item, score=scores[item["id"]] + interest,
                           history_score=scores[item["id"]],
                           interest_score=interest))
    ranked.sort(key=lambda row: (-row["score"], row["id"]))
    return validate({
        "context": context, "cold_start": not events, "matched_events": len(events),
        "recommendations": ranked[:data["config"]["top_k"]],
    }, "behavior")


STOPWORDS = frozenset("a an the is are what how can do i my for of to in and".split())


def tokens(text):
    return set(re.findall(r"\w+", text.casefold(), re.UNICODE)) - STOPWORDS


def answer_faq(data, behavior):
    validate(behavior, "behavior")
    question = behavior["context"]["question"]
    recommendations = behavior["recommendations"]
    category = recommendations[0]["category"] if recommendations else None
    result = {"status": "abstained", "question": question, "category": category,
              "answer": None, "citation": None, "retrieval_score": 0.0,
              "reason": None}
    if question is None:
        result["reason"] = "missing_question"
    elif category is None:
        result["reason"] = "no_recommendations"
    else:
        query = tokens(question)
        candidates = []
        for article in data["knowledge_base"]:
            if article["category"] == category:
                score = len(query & tokens(article["question"])) / len(query) if query else 0
                candidates.append((score, article["id"], article))
        candidates.sort(key=lambda entry: (-entry[0], entry[1]))
        if candidates:
            score, _, article = candidates[0]
            result["retrieval_score"] = score
        if (not candidates or candidates[0][0] <= 0
                or candidates[0][0] < data["config"]["faq_min_score"]):
            result["reason"] = "insufficient_evidence"
        else:
            result.update(status="answered", answer=article["answer"], reason=None,
                          citation={"id": article["id"], "quote": article["answer"]})
    return validate(result, "faq", data)


def run_pipeline(data):
    validate(data, "input")
    extracted = extract(data)
    behavior = personalize(data, extracted)
    faq = answer_faq(data, behavior)
    return validate({"schema_version": 1, "status": "ok", "extraction": extracted,
                     "behavior": behavior, "faq": faq}, "output")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON property: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
