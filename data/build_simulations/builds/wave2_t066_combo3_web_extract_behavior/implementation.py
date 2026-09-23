"""Synthetic, offline web -> extraction -> personalization reference pipeline."""

import hashlib
import json
import math
import re
import sys
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite numeric")
    return value


def timestamp(value):
    text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid ISO timestamp") from exc
    require(result.tzinfo is not None, "timestamps must include a timezone")
    return result


def canonical_url(value, hosts):
    text(value, "URL")
    require(not any(c.isspace() or ord(c) < 32 for c in value), "URL contains whitespace")
    try:
        parsed = urlsplit(value)
        require(parsed.scheme == "https", "only HTTPS URLs are accepted")
        require(parsed.hostname in hosts, "URL host is not allowlisted")
        require(parsed.username is None and parsed.password is None, "URL credentials forbidden")
        require(parsed.port in (None, 443), "nonstandard port forbidden")
        require(not parsed.fragment and "\\" not in value, "ambiguous URL forbidden")
        return urlunsplit(("https", parsed.hostname, parsed.path or "/", parsed.query, ""))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def digest(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def convert(raw, kind):
    if kind == "string":
        return text(raw, "field value")
    if kind == "integer":
        require(re.fullmatch(r"[+-]?\d+", raw) is not None, "invalid integer field")
        return int(raw)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError("invalid numeric field") from exc
    return number(value, "field value")


def validate(stage, data):
    """One validation boundary used by all stages and by the public entry point."""
    require(isinstance(data, dict), stage + " must be an object")
    if stage == "input":
        require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
                "schema_version must be integer 1")
        require(data.get("synthetic") is True, "fixtures must be labeled synthetic")
        hosts = data.get("allowlist")
        require(isinstance(hosts, list) and hosts, "allowlist must be a nonempty list")
        for host in hosts:
            require(isinstance(host, str) and
                    re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) is not None,
                    "allowlist must contain lowercase exact hostnames")
        urls = data.get("urls")
        require(isinstance(urls, list) and 0 < len(urls) <= 100, "provide 1..100 URLs")
        canonical = [canonical_url(url, hosts) for url in urls]
        require(len(set(canonical)) == len(canonical), "duplicate requested URL")
        fixtures = data.get("fixtures")
        require(isinstance(fixtures, dict) and len(fixtures) <= 200, "fixtures must be an object")
        keys = set()
        for url, fixture in fixtures.items():
            key = canonical_url(url, hosts)
            require(key not in keys, "duplicate canonical fixture URL")
            keys.add(key)
            require(isinstance(fixture, dict), "fixture must be an object")
            require(("text" in fixture) != ("redirect" in fixture), "fixture needs text XOR redirect")
            if "redirect" in fixture:
                canonical_url(fixture["redirect"], hosts)
            else:
                text(fixture["text"], "fixture text")
                require(len(fixture["text"]) <= 100000, "fixture text too large")
        schema = data.get("fields")
        require(isinstance(schema, list) and 4 <= len(schema) <= 30, "fields must define 4..30 fields")
        names, labels = set(), set()
        for field in schema:
            require(isinstance(field, dict), "field must be an object")
            name, label = text(field.get("name"), "field name"), text(field.get("label"), "label")
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, "invalid field name")
            require(name not in ("source", "score", "explanation"), "reserved field name")
            require("\n" not in label and "\r" not in label and label == label.strip(),
                    "label must be one trimmed line")
            require(name not in names and label not in labels, "duplicate field name or label")
            names.add(name)
            labels.add(label)
            require(field.get("type") in ("string", "number", "integer"), "unsupported field type")
            require(type(field.get("required")) is bool, "required must be boolean")
        by_name = {field["name"]: field for field in schema}
        for name, kind in (("id", "string"), ("title", "string"),
                           ("category", "string"), ("price", "number")):
            require(name in by_name and by_name[name]["type"] == kind and by_name[name]["required"],
                    "core fields id/title/category/price must have required canonical types")
        now = timestamp(data.get("now"))
        ranking = data.get("ranking", {})
        require(isinstance(ranking, dict), "ranking must be an object")
        require(number(ranking.get("half_life_days", 30), "half_life_days") > 0,
                "half_life_days must be positive")
        limit = ranking.get("limit", 10)
        require(type(limit) is int and 1 <= limit <= 100, "limit must be 1..100")
        history = data.get("history", [])
        require(isinstance(history, list) and len(history) <= 10000, "history must be a bounded list")
        for event in history:
            require(isinstance(event, dict), "history event must be an object")
            require(event.get("action") in ("view", "purchase"), "unsupported action")
            text(event.get("product_id"), "product_id")
            require(timestamp(event.get("at")) <= now, "future history is invalid")
    elif stage == "research":
        validate("input", data["input"])
        hosts = data["input"]["allowlist"]
        documents = data["documents"]
        require(len(documents) == len(data["input"]["urls"]), "document count mismatch")
        for requested, document in zip(data["input"]["urls"], documents):
            require(document["requested_url"] == canonical_url(requested, hosts), "request mismatch")
            require(document["sha256"] == digest(document["text"]), "document digest mismatch")
            chain = document["retrieval_chain"]
            require(chain and chain[0] == document["requested_url"] and
                    chain[-1] == document["url"], "invalid retrieval chain")
            for url in chain:
                require(canonical_url(url, hosts) == url, "noncanonical provenance URL")
    elif stage == "extraction":
        validate("research", data)
        require(len(data["records"]) == len(data["documents"]), "record count mismatch")
        definitions = data["input"]["fields"]
        for document, record in zip(data["documents"], data["records"]):
            require(record["source"] == {"url": document["url"], "sha256": document["sha256"]},
                    "record provenance mismatch")
            require(set(record["fields"]) == {f["name"] for f in definitions}, "field set mismatch")
            missing = []
            for definition in definitions:
                field = record["fields"][definition["name"]]
                if field["status"] == "missing":
                    require(field["value"] is None and field["span"] is None, "invalid missing field")
                    missing.append(definition["name"])
                else:
                    require(field["status"] == "present", "invalid field status")
                    start, end = field["span"]
                    require(type(start) is int and type(end) is int and
                            0 <= start < end <= len(document["text"]), "invalid source span")
                    require(convert(document["text"][start:end], definition["type"]) == field["value"],
                            "source span does not support value")
            require(record["missing_fields"] == missing, "missing fields mismatch")
            required = [f["name"] for f in definitions if f["required"] and f["name"] in missing]
            require(record["missing_required"] == required, "missing required mismatch")
    elif stage == "behavior":
        validate("extraction", data)
        eligible = {r["fields"]["id"]["value"]: r for r in data["records"] if not r["missing_required"]}
        seen = set()
        for recommendation in data["behavior"]["recommendations"]:
            key = recommendation["id"]
            require(key in eligible and key not in seen, "recommendation is not a unique extracted item")
            seen.add(key)
            require(recommendation["source"] == eligible[key]["source"], "recommendation provenance mismatch")
            require(number(recommendation["score"], "score") >= 0, "negative ranking score")
    else:
        raise ValidationError("unknown validation stage")
    return data


def research(payload):
    validate("input", payload)
    hosts = payload["allowlist"]
    fixtures = {canonical_url(url, hosts): fixture for url, fixture in payload["fixtures"].items()}
    documents = []
    for requested in payload["urls"]:
        current = canonical_url(requested, hosts)
        chain = []
        while True:
            require(current not in chain, "redirect cycle")
            require(len(chain) < 10, "redirect chain exceeds 10 URLs")
            chain.append(current)
            require(current in fixtures, "no offline fixture for " + current)
            fixture = fixtures[current]
            if "redirect" not in fixture:
                body = fixture["text"]
                break
            current = canonical_url(fixture["redirect"], hosts)
        documents.append({"requested_url": chain[0], "url": current, "retrieval_chain": chain,
                          "text": body, "sha256": digest(body), "retrieved_at": payload["now"],
                          "transport": "synthetic_fixture"})
    return validate("research", {"input": payload, "documents": documents})


def extract(bundle):
    validate("research", bundle)
    records = []
    for document in bundle["documents"]:
        fields, missing, missing_required = {}, [], []
        for definition in bundle["input"]["fields"]:
            pattern = r"^" + re.escape(definition["label"]) + r":[ \t]*([^\r\n]*)"
            matches = list(re.finditer(pattern, document["text"], re.MULTILINE))
            require(len(matches) <= 1, "ambiguous duplicate label: " + definition["label"])
            raw = matches[0].group(1).strip() if matches else ""
            if not raw:
                missing.append(definition["name"])
                if definition["required"]:
                    missing_required.append(definition["name"])
                fields[definition["name"]] = {"status": "missing", "value": None, "span": None}
                continue
            match = matches[0]
            start = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip())
            value = convert(raw, definition["type"])
            if definition["name"] == "price":
                require(value >= 0, "price must not be negative")
            fields[definition["name"]] = {"status": "present", "value": value,
                                           "span": [start, start + len(raw)]}
        records.append({"source": {"url": document["url"], "sha256": document["sha256"]},
                        "fields": fields, "missing_fields": missing, "missing_required": missing_required})
    return validate("extraction", {**bundle, "records": records})


def personalize(bundle):
    validate("extraction", bundle)
    payload = bundle["input"]
    candidates, excluded = {}, []
    for record in bundle["records"]:
        if record["missing_required"]:
            excluded.append({"source": record["source"], "missing_required": record["missing_required"]})
            continue
        values = {key: field["value"] for key, field in record["fields"].items()}
        require(values["id"] not in candidates, "duplicate extracted product id")
        candidates[values["id"]] = (values, record["source"])
    config = payload.get("ranking", {})
    half_life = config.get("half_life_days", 30)
    now = timestamp(payload["now"])
    direct, category, used = {}, {}, 0
    history = payload.get("history", [])
    for event in history:
        key = event["product_id"]
        if key not in candidates:
            continue
        age = (now - timestamp(event["at"])).total_seconds() / 86400
        weight = (3 if event["action"] == "purchase" else 1) * 2 ** (-age / half_life)
        group = candidates[key][0]["category"]
        direct[key] = direct.get(key, 0) + weight
        category[group] = category.get(group, 0) + weight
        used += 1
    recommendations = []
    for key, (values, source) in candidates.items():
        item_weight, group_weight = direct.get(key, 0), category.get(values["category"], 0)
        recommendations.append({**values, "source": source, "score": item_weight + group_weight,
                                "explanation": {"direct_weight": item_weight,
                                                "category_weight": group_weight}})
    recommendations.sort(key=lambda item: (-item["score"], item["id"]))
    behavior = {"cold_start": not used, "events_used": used, "events_ignored": len(history) - used,
                "excluded": excluded, "recommendations": recommendations[:config.get("limit", 10)]}
    return validate("behavior", {**bundle, "behavior": behavior})


def run_pipeline(payload):
    bundle = personalize(extract(research(payload)))
    return {"status": "ok", "schema_version": 1, "synthetic": True,
            "research": {"documents": bundle["documents"]},
            "extraction": {"records": bundle["records"]}, "behavior": bundle["behavior"]}


def reject_constant(value):
    raise ValidationError("nonstandard JSON numeric constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant)
        result = run_pipeline(payload)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError,
            KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
