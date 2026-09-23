"""Synthetic, offline extract -> research -> compare reference pipeline.

Usage: python -B implementation.py example_input.json
Only supplied page fixtures are retrieved; this module never opens a network.
Spans are zero-based, half-open Python character offsets in original text.
"""

import json
import math
import re
import sys
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def url_host(url):
    text(url, "URL")
    require(not any(c.isspace() for c in url), "URL contains whitespace")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(parts.scheme == "https" and bool(parts.hostname), "URL must be absolute HTTPS")
    require(parts.username is None and parts.password is None, "URL credentials prohibited")
    require(port in (None, 443) and not parts.fragment, "URL port or fragment prohibited")
    return parts.hostname.lower()


def parse_value(raw, field):
    if field["type"] == "string":
        return raw, None
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([^\d\s].*)?", raw)
    require(match is not None, "Invalid numeric value for " + field["name"])
    value = float(match.group(1))
    unit = (match.group(2) or field["unit"]).strip().casefold()
    require(number(value), "Nonfinite number")
    require(unit in field["unit_factors"], "Unknown unit for " + field["name"])
    require(number(value * field["unit_factors"][unit]), "Normalized number overflow")
    return value, unit


def extract_fields(content, schema, source):
    fields = {}
    for field in schema:
        pattern = r"^[ \t]*" + re.escape(field["label"]) + r"[ \t]*:[ \t]*(?P<raw>[^\r\n]*)"
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match is None or not match["raw"].strip():
            continue
        raw = match["raw"].strip()
        start = match.start("raw") + len(match["raw"]) - len(match["raw"].lstrip())
        value, unit = parse_value(raw, field)
        fields[field["name"]] = {
            "value": value, "unit": unit, "raw": raw,
            "span": [start, start + len(raw)], "source": dict(source),
        }
    return fields


def validate_evidence(evidence, field, config, kind, source_id):
    require(isinstance(evidence, dict), "Evidence must be an object")
    require(set(evidence) == {"value", "unit", "raw", "span", "source"}, "Evidence shape invalid")
    source = evidence["source"]
    require(isinstance(source, dict), "Evidence source invalid")
    require(source.get("kind") == kind and source.get("id") == source_id, "Evidence source mismatch")
    if kind == "document":
        originals = {d["product_id"]: d["text"] for d in config["documents"]}
        require(set(source) == {"kind", "id"}, "Document source shape invalid")
    else:
        originals = {p["url"]: p["text"] for p in config["pages"]}
        pages = {p["url"]: p for p in config["pages"]}
        require(source_id in pages, "Unknown page evidence")
        require(source.get("url") == source_id and source.get("title") == pages[source_id]["title"],
                "Page provenance mismatch")
        require(url_host(source_id) in config["allowed_hosts"], "Nonallowlisted evidence")
    require(source_id in originals, "Unknown evidence source")
    span = evidence["span"]
    require(isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span),
            "Span must contain two integers")
    require(0 <= span[0] < span[1] <= len(originals[source_id]), "Span out of bounds")
    require(originals[source_id][span[0]:span[1]] == evidence["raw"], "Span does not match source")
    value, unit = parse_value(evidence["raw"], field)
    require(evidence["value"] == value and evidence["unit"] == unit, "Evidence value mismatch")


def validate(stage, payload, config=None):
    """One validation boundary shared by input and every stage handoff."""
    require(isinstance(payload, dict), stage + " must be an object")
    if stage == "input":
        require(set(payload) == {"schema_version", "synthetic", "fields", "documents",
                                 "allowed_hosts", "pages", "preferences"}, "Input keys invalid")
        require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
                "Unsupported schema_version")
        require(payload["synthetic"] is True, "Fixture must be labeled synthetic")
        fields = payload["fields"]
        require(isinstance(fields, list) and fields, "fields must be a nonempty list")
        names, labels = set(), set()
        for field in fields:
            require(isinstance(field, dict), "Field definition must be an object")
            require(field.get("type") in ("string", "number"), "Unsupported field type")
            expected = {"name", "label", "type", "required"}
            if field["type"] == "number":
                expected |= {"unit", "unit_factors"}
            require(set(field) == expected, "Field keys invalid")
            text(field["name"], "Field name")
            text(field["label"], "Field label")
            require(re.fullmatch(r"[a-z][a-z0-9_]*", field["name"]) is not None, "Invalid field name")
            require(not any(c in field["label"] for c in "\r\n:"), "Invalid field label")
            require(type(field["required"]) is bool, "required must be boolean")
            require(field["name"] not in names and field["label"].casefold() not in labels,
                    "Duplicate field name or label")
            names.add(field["name"])
            labels.add(field["label"].casefold())
            if field["type"] == "number":
                text(field["unit"], "Canonical unit")
                factors = field["unit_factors"]
                require(isinstance(factors, dict) and factors, "unit_factors invalid")
                for unit, factor in factors.items():
                    text(unit, "Unit")
                    require(unit == unit.strip().casefold() and number(factor) and factor > 0,
                            "Unit factors must have normalized names and positive finite values")
                require(factors.get(field["unit"]) == 1, "Canonical unit must have factor 1")
        hosts = payload["allowed_hosts"]
        require(isinstance(hosts, list) and hosts, "allowed_hosts must be nonempty")
        require(all(isinstance(h, str) and re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", h)
                    for h in hosts), "Hosts must be exact lowercase hostnames")
        require(len(set(hosts)) == len(hosts), "Duplicate allowed host")
        documents = payload["documents"]
        require(isinstance(documents, list) and documents, "documents must be nonempty")
        ids = set()
        for document in documents:
            require(isinstance(document, dict) and set(document) ==
                    {"product_id", "text", "research_urls"}, "Document shape invalid")
            text(document["product_id"], "product_id")
            require(document["product_id"] not in ids, "Duplicate product_id")
            ids.add(document["product_id"])
            require(isinstance(document["text"], str), "Document text must be a string")
            require(isinstance(document["research_urls"], list), "research_urls must be a list")
            for url in document["research_urls"]:
                url_host(url)
            require(len(set(document["research_urls"])) == len(document["research_urls"]),
                    "Duplicate research URL")
        require(isinstance(payload["pages"], list), "pages must be a list")
        urls = set()
        for page in payload["pages"]:
            require(isinstance(page, dict) and set(page) == {"url", "title", "text"},
                    "Page fixture shape invalid")
            url_host(page["url"])
            text(page["title"], "Page title")
            require(isinstance(page["text"], str), "Page text must be a string")
            require(page["url"] not in urls, "Duplicate fixture URL")
            urls.add(page["url"])
        preferences = payload["preferences"]
        require(isinstance(preferences, list) and preferences, "preferences must be nonempty")
        seen = set()
        definitions = {f["name"]: f for f in fields}
        for pref in preferences:
            require(isinstance(pref, dict), "Preference must be an object")
            direction = pref.get("direction")
            require(direction in ("min", "max", "target"), "Invalid preference direction")
            keys = {"field", "direction", "weight"} | ({"target"} if direction == "target" else set())
            require(set(pref) == keys, "Preference keys invalid")
            require(isinstance(pref["field"], str) and pref["field"] in names, "Unknown preference field")
            require(pref["field"] not in seen, "Duplicate preference field")
            seen.add(pref["field"])
            require(number(pref["weight"]) and pref["weight"] > 0, "Weight must be positive and finite")
            if direction == "target":
                require(definitions[pref["field"]]["type"] == "string", "target requires a string field")
                text(pref["target"], "Preference target")
            else:
                require(definitions[pref["field"]]["type"] == "number", "min/max requires a numeric field")
        require(number(sum(p["weight"] for p in preferences)), "Weight sum overflow")
        return payload

    require(config is not None, "Stage validation needs the shared configuration")
    require(payload.get("schema_version") == 1, "Stage schema version invalid")
    definitions = {f["name"]: f for f in config["fields"]}
    documents = {d["product_id"]: d for d in config["documents"]}
    products = payload.get("products")
    require(isinstance(products, list), "Stage products must be a list")
    require(all(isinstance(p, dict) for p in products), "Stage product must be an object")
    require([p.get("product_id") for p in products] == list(documents), "Stage product IDs mismatch")
    for product in products:
        pid = product["product_id"]
        if stage in ("extract", "web"):
            fields = product.get("fields" if stage == "extract" else "document_fields")
            require(isinstance(fields, dict) and set(fields) <= set(definitions), "Unknown stage field")
            for name, evidence in fields.items():
                validate_evidence(evidence, definitions[name], config, "document", pid)
            require(fields == extract_fields(documents[pid]["text"], config["fields"],
                                             {"kind": "document", "id": pid}),
                    "Document extraction incomplete or altered")
            if stage == "extract":
                require(product.get("research_urls") == documents[pid]["research_urls"], "URL handoff altered")
                missing = [f["name"] for f in config["fields"] if f["name"] not in fields]
                require(product.get("missing_fields") == missing, "Missing fields report invalid")
                require(product.get("missing_required") ==
                        [n for n in missing if definitions[n]["required"]], "Missing required report invalid")
            else:
                retrievals = product.get("retrievals")
                require(isinstance(retrievals, list), "Retrievals invalid")
                require(all(isinstance(r, dict) for r in retrievals), "Retrieval entry invalid")
                require([r.get("url") for r in retrievals] == documents[pid]["research_urls"],
                        "Retrieval URL handoff mismatch")
                fixtures = {p["url"]: p for p in config["pages"]}
                expected_findings = {name: [] for name in definitions}
                for retrieval in retrievals:
                    url = retrieval["url"]
                    expected = ("blocked" if url_host(url) not in config["allowed_hosts"]
                                else "retrieved" if url in fixtures else "unavailable")
                    require(retrieval.get("status") == expected, "Retrieval status mismatch")
                    if expected == "retrieved":
                        page = fixtures[url]
                        found = extract_fields(page["text"], config["fields"],
                                               {"kind": "page", "id": url, "url": url, "title": page["title"]})
                        for name, evidence in found.items():
                            expected_findings[name].append(evidence)
                require(product.get("findings") == expected_findings, "Research findings altered or incomplete")
                for name, findings in product["findings"].items():
                    for evidence in findings:
                        validate_evidence(evidence, definitions[name], config, "page", evidence["source"]["id"])
        elif stage == "compare":
            attributes = product.get("attributes")
            require(isinstance(attributes, dict) and set(attributes) == set(definitions),
                    "Comparison attributes incomplete")
            require(number(product.get("score")) and 0 <= product["score"] <= 1, "Invalid score")
            for name, attribute in attributes.items():
                require(isinstance(attribute, dict), "Attribute invalid")
                value = attribute.get("value")
                if value is not None:
                    require(number(value) if definitions[name]["type"] == "number" else isinstance(value, str),
                            "Normalized attribute type invalid")
                    evidence = attribute.get("evidence")
                    require(isinstance(evidence, dict) and isinstance(evidence.get("source"), dict),
                            "Comparison provenance missing")
                    source = evidence["source"]
                    require(source.get("kind") in ("document", "page"), "Comparison source kind invalid")
                    if source["kind"] == "document":
                        require(source.get("id") == pid, "Cross-product document evidence")
                    else:
                        require(source.get("id") in documents[pid]["research_urls"], "Cross-product page evidence")
                    validate_evidence(evidence, definitions[name], config, source["kind"], source["id"])
                    require(value == normalize(evidence, definitions[name]), "Normalization mismatch")
            require(product.get("missing_required") ==
                    [f["name"] for f in config["fields"] if f["required"] and
                     attributes[f["name"]]["value"] is None], "Comparison missing report invalid")
        else:
            raise ValidationError("Unknown validation stage")
    if stage == "compare":
        expected_order = sorted(products, key=lambda p: (-p["score"], p["product_id"]))
        require(payload.get("ranking") == [{"rank": i + 1, "product_id": p["product_id"], "score": p["score"]}
                                           for i, p in enumerate(expected_order)], "Ranking mismatch")
    return payload


def extract(config):
    validate("input", config)
    products = []
    for document in config["documents"]:
        fields = extract_fields(document["text"], config["fields"],
                                {"kind": "document", "id": document["product_id"]})
        missing = [f["name"] for f in config["fields"] if f["name"] not in fields]
        products.append({
            "product_id": document["product_id"], "fields": fields,
            "research_urls": list(document["research_urls"]), "missing_fields": missing,
            "missing_required": [f["name"] for f in config["fields"] if f["required"] and f["name"] in missing],
        })
    return validate("extract", {"schema_version": 1, "products": products}, config)


def research(extraction, config):
    validate("extract", extraction, config)
    pages = {p["url"]: p for p in config["pages"]}
    products = []
    for product in extraction["products"]:
        findings = {f["name"]: [] for f in config["fields"]}
        retrievals = []
        for url in product["research_urls"]:
            status = ("blocked" if url_host(url) not in config["allowed_hosts"]
                      else "retrieved" if url in pages else "unavailable")
            retrievals.append({"url": url, "status": status})
            if status == "retrieved":
                page = pages[url]
                evidence = extract_fields(page["text"], config["fields"],
                                          {"kind": "page", "id": url, "url": url, "title": page["title"]})
                for name, finding in evidence.items():
                    findings[name].append(finding)
        products.append({"product_id": product["product_id"], "document_fields": product["fields"],
                         "findings": findings, "retrievals": retrievals})
    return validate("web", {"schema_version": 1, "products": products}, config)


def normalize(evidence, field):
    if field["type"] == "number":
        return evidence["value"] * field["unit_factors"][evidence["unit"]]
    return " ".join(evidence["value"].split()).casefold()


def compare(research_output, config):
    validate("web", research_output, config)
    products = []
    for product in research_output["products"]:
        attributes = {}
        for field in config["fields"]:
            name = field["name"]
            candidates = ([product["document_fields"][name]] if name in product["document_fields"] else [])
            candidates += product["findings"][name]
            selected = candidates[0] if candidates else None
            value = normalize(selected, field) if selected else None
            attributes[name] = {
                "value": value, "unit": field.get("unit"), "evidence": selected,
                "alternatives": candidates[1:],
                "conflict": any(normalize(e, field) != value for e in candidates[1:]),
            }
        products.append({"product_id": product["product_id"], "attributes": attributes,
                         "missing_required": [f["name"] for f in config["fields"] if f["required"] and
                                              attributes[f["name"]]["value"] is None],
                         "score": 0.0, "score_components": {}})
    total = sum(p["weight"] for p in config["preferences"])
    for pref in config["preferences"]:
        name = pref["field"]
        values = [p["attributes"][name]["value"] for p in products if p["attributes"][name]["value"] is not None]
        for product in products:
            value = product["attributes"][name]["value"]
            if value is None:
                score = 0.0
            elif pref["direction"] == "target":
                score = float(value == " ".join(pref["target"].split()).casefold())
            elif max(values) == min(values):
                score = 1.0
            else:
                # Scale first to avoid overflow when the range spans large signed numbers.
                scale = max(abs(min(values)), abs(max(values)))
                low, high, current = min(values) / scale, max(values) / scale, value / scale
                score = (current - low) / (high - low)
                if pref["direction"] == "min":
                    score = 1.0 - score
            product["score_components"][name] = score
            product["score"] += score * (pref["weight"] / total)
    for product in products:
        product["score"] = round(min(1.0, max(0.0, product["score"])), 12)
    ranking = [{"rank": i + 1, "product_id": p["product_id"], "score": p["score"]}
               for i, p in enumerate(sorted(products, key=lambda p: (-p["score"], p["product_id"])))]
    return validate("compare", {"schema_version": 1, "products": products, "ranking": ranking}, config)


def run_pipeline(config):
    validate("input", config)
    extraction = extract(config)
    findings = research(extraction, config)
    comparison = compare(findings, config)
    return {"status": "ok", "schema_version": 1, "synthetic": True,
            "extraction": extraction, "research": findings, "comparison": comparison}


def reject_constant(value):
    raise ValidationError("Nonfinite JSON constant: " + value)


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
            config = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(config)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
