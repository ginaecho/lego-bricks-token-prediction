"""Synthetic, deterministic FAQ -> personalized discovery reference pipeline.

Run: python -B implementation.py example_input.json
Optional answerer: faq_stage(request, answerer) or run(request, answerer).
The injected callable receives a deep copy of retrieved evidence and must return
one complete evidence passage verbatim. Unsupported answers fail validation.
"""

import copy
import datetime as dt
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


DEFAULTS = {
    "half_life_days": 30.0,
    "top_k": 5,
    "min_match": 0.3,
    "faq_boost": 2.0,
}
STOPWORDS = frozenset("a an the is are do does i my can how what to of for and".split())


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing fields: " + str(set(required) - set(value)))
    require(set(value) <= set(required) | set(optional), "Unknown fields")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonblank text")


def number(value, label, minimum=0, maximum=None):
    require(type(value) in (int, float), label + " must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= minimum, label + " must be finite and >= " + str(minimum))
    if maximum is not None:
        require(value <= maximum, label + " exceeds maximum")


def timestamp(value):
    text(value, "timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.utcoffset() is not None, "Timestamp must include a timezone")
        return parsed.astimezone(dt.timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError("Invalid timezone-aware timestamp") from exc


def tokens(value):
    return set(re.findall(r"\w+", value.casefold())) - STOPWORDS


def validate(value, phase="input"):
    """Shared boundary validator for requests, FAQ handoffs and final outputs."""
    require(phase in ("input", "faq", "complete"), "Unknown validation phase")
    if phase == "input":
        fields(value, ("schema_version", "fixture_label", "query", "as_of",
                       "knowledge_base", "products", "events"), ("settings",))
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        text(value["fixture_label"], "fixture_label")
        require(value["fixture_label"].lower().startswith("synthetic"),
                "Fixtures must be explicitly labeled synthetic")
        text(value["query"], "query")
        now = timestamp(value["as_of"])
        for name in ("knowledge_base", "products", "events"):
            require(isinstance(value[name], list), name + " must be an array")
        product_ids = set()
        for product in value["products"]:
            fields(product, ("id", "name", "popularity"))
            text(product["id"], "product id")
            text(product["name"], "product name")
            require(product["id"] not in product_ids, "Duplicate product id")
            product_ids.add(product["id"])
            number(product["popularity"], "popularity", maximum=1)
        kb_ids = set()
        for entry in value["knowledge_base"]:
            fields(entry, ("id", "title", "text", "product_ids"))
            for key in ("id", "title", "text"):
                text(entry[key], "knowledge " + key)
            require(entry["id"] not in kb_ids, "Duplicate knowledge id")
            kb_ids.add(entry["id"])
            require(isinstance(entry["product_ids"], list), "product_ids must be an array")
            seen = set()
            for product_id in entry["product_ids"]:
                text(product_id, "product reference")
                require(product_id in product_ids, "Unknown knowledge product reference")
                require(product_id not in seen, "Duplicate knowledge product reference")
                seen.add(product_id)
        event_ids = set()
        for event in value["events"]:
            fields(event, ("id", "product_id", "type", "timestamp"))
            text(event["id"], "event id")
            text(event["product_id"], "event product id")
            require(event["id"] not in event_ids, "Duplicate event id")
            event_ids.add(event["id"])
            require(event["product_id"] in product_ids, "Unknown event product")
            require(event["type"] in ("view", "purchase"), "Unsupported event type")
            require(timestamp(event["timestamp"]) <= now, "Future events are not allowed")
        settings = value.get("settings", {})
        fields(settings, (), DEFAULTS)
        merged = dict(DEFAULTS, **settings)
        number(merged["half_life_days"], "half_life_days", minimum=0.000001, maximum=36500)
        require(type(merged["top_k"]) is int and 1 <= merged["top_k"] <= 100,
                "top_k must be an integer from 1 to 100")
        number(merged["min_match"], "min_match", minimum=0.000001, maximum=1)
        number(merged["faq_boost"], "faq_boost", maximum=100)
        return value

    required = ("schema_version", "fixture_label", "status", "stage", "request", "faq")
    fields(value, required + (("behavior",) if phase == "complete" else ()))
    validate(value["request"])
    request = value["request"]
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Invalid output schema version")
    require(value["fixture_label"] == request["fixture_label"], "Fixture label changed")
    require(value["status"] == "ok" and value["stage"] == phase, "Invalid envelope stage")
    faq = value["faq"]
    fields(faq, ("status", "answer", "evidence", "product_ids", "reason"))
    require(faq["status"] in ("answered", "abstained"), "Invalid FAQ status")
    require(isinstance(faq["evidence"], list), "Evidence must be an array")
    require(isinstance(faq["product_ids"], list), "FAQ product_ids must be an array")
    expected_evidence = retrieve(request)
    require(faq["evidence"] == expected_evidence, "Evidence differs from validated retrieval")
    expected_ids = sorted({pid for entry in expected_evidence for pid in entry["product_ids"]})
    require(faq["product_ids"] == expected_ids, "Invalid FAQ product propagation")
    if expected_evidence:
        require(faq["status"] == "answered" and faq["reason"] is None,
                "Evidence requires an answered FAQ")
        require(isinstance(faq["answer"], str) and
                faq["answer"] in [entry["text"] for entry in expected_evidence],
                "Answer must be a verbatim grounded evidence passage")
    else:
        require(faq["status"] == "abstained" and faq["answer"] is None and
                faq["reason"] == "No sufficiently matching knowledge-base evidence.",
                "Missing evidence requires explicit abstention")
    if phase == "complete":
        require(value["behavior"] == rank(request, faq), "Invalid behavioral ranking or handoff")
    return value


def settings(request):
    return dict(DEFAULTS, **request.get("settings", {}))


def retrieve(request):
    query_tokens = tokens(request["query"])
    evidence = []
    if not query_tokens:
        return evidence
    for entry in request["knowledge_base"]:
        score = len(query_tokens & tokens(entry["title"] + " " + entry["text"])) / len(query_tokens)
        if score >= settings(request)["min_match"]:
            evidence.append({
                "kb_id": entry["id"], "text": entry["text"], "score": score,
                "product_ids": sorted(entry["product_ids"]),
            })
    return sorted(evidence, key=lambda item: (-item["score"], item["kb_id"]))[:3]


def faq_stage(request, answerer=None):
    validate(request)
    request = copy.deepcopy(request)
    evidence = retrieve(request)
    answer = None
    if evidence:
        if answerer is None:
            answer = evidence[0]["text"]
        else:
            try:
                answer = answerer(copy.deepcopy(evidence))
            except Exception as exc:
                raise ValidationError("Injected answerer failed") from exc
    result = {
        "schema_version": 1,
        "fixture_label": request["fixture_label"],
        "status": "ok",
        "stage": "faq",
        "request": request,
        "faq": {
            "status": "answered" if evidence else "abstained",
            "answer": answer,
            "evidence": evidence,
            "product_ids": sorted({pid for item in evidence for pid in item["product_ids"]}),
            "reason": None if evidence else "No sufficiently matching knowledge-base evidence.",
        },
    }
    return validate(result, "faq")


def rank(request, faq):
    config = settings(request)
    now = timestamp(request["as_of"])
    histories = {product["id"]: [] for product in request["products"]}
    for event in sorted(request["events"], key=lambda entry: entry["id"]):
        age_days = (now - timestamp(event["timestamp"])).total_seconds() / 86400
        weight = 3.0 if event["type"] == "purchase" else 1.0
        histories[event["product_id"]].append(weight * 2.0 ** (-age_days / config["half_life_days"]))
    cold_start = not request["events"]
    ranked = []
    for product in request["products"]:
        components = {
            "history_score": math.fsum(histories[product["id"]]),
            "faq_score": config["faq_boost"] if product["id"] in faq["product_ids"] else 0.0,
            "popularity_score": product["popularity"] * (1.0 if cold_start else 0.01),
        }
        ranked.append({
            "product_id": product["id"], "name": product["name"],
            "score": math.fsum(components.values()), "components": components,
        })
    ranked.sort(key=lambda entry: (-entry["score"], entry["product_id"]))
    return {
        "mode": "cold_start" if cold_start else "personalized",
        "faq_status": faq["status"],
        "faq_product_ids": list(faq["product_ids"]),
        "recommendations": ranked[:config["top_k"]],
    }


def behavior_stage(handoff):
    validate(handoff, "faq")
    result = copy.deepcopy(handoff)
    result["behavior"] = rank(result["request"], result["faq"])
    result["stage"] = "complete"
    return validate(result, "complete")


def run(request, answerer=None):
    return behavior_stage(faq_stage(request, answerer))


def reject_constant(value):
    raise ValidationError("Non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            request = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run(request)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
