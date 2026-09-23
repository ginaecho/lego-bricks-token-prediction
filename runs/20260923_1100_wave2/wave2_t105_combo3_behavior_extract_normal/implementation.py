"""Deterministic synthetic discovery -> extraction -> research reference CLI.

Offsets are zero-based, half-open Python Unicode character offsets in document
text. Regex schemas are trusted local configuration, not untrusted remote code.
"""

import datetime as dt
import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


class Validator:
    """One validation boundary shared by the input and all stage handoffs."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def obj(cls, value, keys, label):
        cls.require(isinstance(value, dict), label + " must be an object")
        cls.require(set(value) == set(keys), label + " has missing/unknown keys")

    @classmethod
    def text(cls, value, label, allow_empty=False):
        cls.require(isinstance(value, str), label + " must be a string")
        cls.require(allow_empty or bool(value.strip()), label + " must not be blank")

    @classmethod
    def number(cls, value, label, minimum=0, positive=False):
        cls.require(type(value) in (int, float), label + " must be numeric")
        cls.require(abs(value) <= 1e12 and math.isfinite(value),
                    label + " must be finite and bounded")
        cls.require(value > minimum if positive else value >= minimum,
                    label + " is out of range")

    @classmethod
    def count(cls, value, label, maximum):
        cls.require(type(value) is int and 1 <= value <= maximum,
                    label + " must be a positive bounded integer")

    @classmethod
    def timestamp(cls, value):
        cls.text(value, "timestamp")
        try:
            result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("invalid ISO timestamp") from exc
        cls.require(result.tzinfo is not None, "timestamp requires a timezone")
        return result

    @classmethod
    def array(cls, value, label, maximum, nonempty=False):
        cls.require(isinstance(value, list), label + " must be an array")
        cls.require(len(value) <= maximum and (not nonempty or len(value) > 0),
                    label + " has an invalid length")

    @classmethod
    def input(cls, data):
        cls.obj(data, ("schema_version", "synthetic", "as_of", "documents",
                       "events", "behavior", "extraction_schema", "research"), "input")
        cls.require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                    "unsupported schema_version")
        cls.require(data["synthetic"] is True, "fixture must be explicitly synthetic")
        now = cls.timestamp(data["as_of"])
        cls.array(data["documents"], "documents", 1000)
        documents = {}
        for doc in data["documents"]:
            cls.obj(doc, ("id", "category", "text", "popularity"), "document")
            for key in ("id", "category", "text"):
                cls.text(doc[key], "document." + key, allow_empty=key == "text")
            cls.require(len(doc["text"]) <= 50000, "document text is too long")
            cls.number(doc["popularity"], "popularity")
            cls.require(doc["id"] not in documents, "duplicate document id")
            documents[doc["id"]] = doc
        cls.obj(data["behavior"], ("top_k", "half_life_days", "purchase_weight"), "behavior")
        cls.count(data["behavior"]["top_k"], "top_k", 1000)
        cls.number(data["behavior"]["half_life_days"], "half_life_days",
                   minimum=0.000001, positive=True)
        cls.number(data["behavior"]["purchase_weight"], "purchase_weight", minimum=1)
        cls.array(data["events"], "events", 10000)
        for event in data["events"]:
            cls.obj(event, ("document_id", "kind", "at"), "event")
            cls.text(event["document_id"], "event.document_id")
            cls.require(event["document_id"] in documents, "event references unknown document")
            cls.require(event["kind"] in ("browse", "purchase"), "invalid event kind")
            cls.require(cls.timestamp(event["at"]) <= now, "future events are not allowed")
        cls.array(data["extraction_schema"], "extraction_schema", 64, nonempty=True)
        names = set()
        for field in data["extraction_schema"]:
            cls.obj(field, ("name", "pattern", "type", "required"), "field schema")
            cls.text(field["name"], "field name")
            cls.require(field["name"] not in names, "duplicate field name")
            names.add(field["name"])
            cls.text(field["pattern"], "pattern")
            cls.require(len(field["pattern"]) <= 512, "pattern is too long")
            cls.require(field["type"] in ("string", "integer", "date"), "unsupported field type")
            cls.require(type(field["required"]) is bool, "required must be boolean")
            try:
                pattern = re.compile(field["pattern"], re.MULTILINE)
            except re.error as exc:
                raise ValidationError("invalid extraction pattern") from exc
            cls.require("value" in pattern.groupindex, "pattern requires a named value group")
        cls.obj(data["research"], ("queries", "max_findings"), "research")
        cls.array(data["research"]["queries"], "queries", 20, nonempty=True)
        for query in data["research"]["queries"]:
            cls.text(query, "query")
            cls.require(len(query) <= 500, "query is too long")
        cls.count(data["research"]["max_findings"], "max_findings", 50)
        return data

    @classmethod
    def span(cls, span, document):
        cls.obj(span, ("document_id", "start", "end", "quote"), "source span")
        cls.require(span["document_id"] == document["id"], "source document mismatch")
        start, end = span["start"], span["end"]
        cls.require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(document["text"]), "invalid source offsets")
        cls.require(span["quote"] == document["text"][start:end], "source quote mismatch")

    @classmethod
    def behavior(cls, output, data):
        cls.obj(output, ("cold_start", "ranking"), "behavior output")
        cls.require(type(output["cold_start"]) is bool, "cold_start must be boolean")
        cls.array(output["ranking"], "ranking", data["behavior"]["top_k"])
        documents = {doc["id"]: doc for doc in data["documents"]}
        cls.require(len(output["ranking"]) == min(len(documents), data["behavior"]["top_k"]),
                    "ranking length mismatch")
        ids = set()
        for item in output["ranking"]:
            cls.obj(item, ("document_id", "score", "reason"), "ranked item")
            cls.text(item["document_id"], "ranked document id")
            cls.require(item["document_id"] in documents and item["document_id"] not in ids,
                        "invalid or duplicate ranked document")
            ids.add(item["document_id"])
            cls.require(type(item["score"]) in (int, float)
                        and math.isfinite(item["score"]) and item["score"] >= 0,
                        "invalid ranking score")
            cls.require(item["reason"] == ("popularity" if output["cold_start"]
                                          else "recency_weighted_affinity"), "invalid rank reason")
        cls.require(output["ranking"] == sorted(output["ranking"],
                    key=lambda item: (-item["score"], item["document_id"])),
                    "ranking is not sorted")
        return output

    @classmethod
    def extraction(cls, output, ranking, data):
        cls.behavior(ranking, data)
        cls.obj(output, ("documents",), "extraction output")
        cls.array(output["documents"], "extracted documents", data["behavior"]["top_k"])
        cls.require([item.get("document_id") for item in output["documents"]
                     if isinstance(item, dict)] ==
                    [item["document_id"] for item in ranking["ranking"]],
                    "extraction must preserve ranked documents and order")
        docs = {doc["id"]: doc for doc in data["documents"]}
        schema = {field["name"]: field for field in data["extraction_schema"]}
        for item in output["documents"]:
            cls.obj(item, ("document_id", "fields", "missing_fields", "complete"), "extracted document")
            cls.require(isinstance(item["fields"], dict), "fields must be an object")
            cls.array(item["missing_fields"], "missing_fields", len(schema))
            missing = {}
            for entry in item["missing_fields"]:
                cls.obj(entry, ("name", "reason", "required"), "missing field")
                cls.text(entry["name"], "missing name")
                cls.require(entry["name"] in schema and entry["name"] not in missing,
                            "invalid missing field")
                cls.require(entry["reason"] in ("not_found", "empty_value", "invalid_value"),
                            "invalid missing reason")
                cls.require(entry["required"] is schema[entry["name"]]["required"],
                            "missing required flag mismatch")
                missing[entry["name"]] = entry
            cls.require(set(item["fields"]).isdisjoint(missing)
                        and set(item["fields"]) | set(missing) == set(schema),
                        "fields must be either extracted or missing")
            cls.require(type(item["complete"]) is bool and item["complete"] ==
                        (not any(entry["required"] for entry in missing.values())),
                        "complete flag mismatch")
            for name, field in item["fields"].items():
                cls.obj(field, ("value", "source"), "extracted field")
                cls.span(field["source"], docs[item["document_id"]])
                try:
                    expected = convert(field["source"]["quote"], schema[name]["type"])
                except ValueError as exc:
                    raise ValidationError("invalid extracted value") from exc
                cls.require(type(field["value"]) is type(expected) and field["value"] == expected,
                            "extracted value disagrees with source")
        return output

    @classmethod
    def research(cls, output, extracted, ranking, data):
        cls.extraction(extracted, ranking, data)
        cls.obj(output, ("queries",), "research output")
        cls.array(output["queries"], "research queries", 20)
        cls.require(len(output["queries"]) == len(data["research"]["queries"]), "query count mismatch")
        docs = {doc["id"]: doc for doc in data["documents"]}
        eligible = {item["document_id"]: item for item in extracted["documents"]}
        for result, query in zip(output["queries"], data["research"]["queries"]):
            cls.obj(result, ("query", "findings"), "query result")
            cls.require(result["query"] == query, "query mismatch")
            cls.array(result["findings"], "findings", data["research"]["max_findings"])
            seen = set()
            for finding in result["findings"]:
                cls.obj(finding, ("text", "score", "citation", "evidence_fields"), "finding")
                citation = finding["citation"]
                cls.require(isinstance(citation, dict), "citation must be an object")
                cls.text(citation.get("document_id"), "citation document id")
                doc_id = citation["document_id"]
                cls.require(doc_id in eligible, "finding bypassed extraction")
                cls.span(citation, docs[doc_id])
                cls.require(finding["text"] == citation["quote"], "finding is not extractive")
                cls.number(finding["score"], "finding score", positive=True)
                cls.array(finding["evidence_fields"], "evidence_fields", 64, nonempty=True)
                expected = evidence_fields(eligible[doc_id], citation["start"], citation["end"])
                cls.require(finding["evidence_fields"] == expected, "evidence linkage mismatch")
                key = (doc_id, citation["start"], citation["end"])
                cls.require(key not in seen, "duplicate finding")
                seen.add(key)
        return output


def convert(text, kind):
    if kind == "integer":
        if not re.fullmatch(r"[+-]?[0-9]+", text):
            raise ValueError("not an integer")
        return int(text)
    if kind == "date":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
            raise ValueError("not an ISO date")
        return dt.date.fromisoformat(text).isoformat()
    return text


def personalize(data):
    Validator.input(data)
    docs = {doc["id"]: doc for doc in data["documents"]}
    direct, categories = {}, {}
    now = Validator.timestamp(data["as_of"])
    for event in data["events"]:
        age = (now - Validator.timestamp(event["at"])).total_seconds() / 86400
        weight = (data["behavior"]["purchase_weight"] if event["kind"] == "purchase" else 1)
        weight *= 2 ** (-age / data["behavior"]["half_life_days"])
        doc_id = event["document_id"]
        category = docs[doc_id]["category"]
        direct[doc_id] = direct.get(doc_id, 0) + weight
        categories[category] = categories.get(category, 0) + weight
    cold = not any(value > 0 for value in direct.values())
    ranking = []
    for doc in data["documents"]:
        score = doc["popularity"] if cold else (
            direct.get(doc["id"], 0) + 0.25 * categories.get(doc["category"], 0))
        ranking.append({"document_id": doc["id"], "score": score,
                        "reason": "popularity" if cold else "recency_weighted_affinity"})
    ranking.sort(key=lambda item: (-item["score"], item["document_id"]))
    return Validator.behavior({"cold_start": cold,
                               "ranking": ranking[:data["behavior"]["top_k"]]}, data)


def source(doc, start, end):
    return {"document_id": doc["id"], "start": start, "end": end,
            "quote": doc["text"][start:end]}


def extract(data, ranking):
    Validator.behavior(ranking, data)
    docs = {doc["id"]: doc for doc in data["documents"]}
    output = []
    for ranked in ranking["ranking"]:
        doc = docs[ranked["document_id"]]
        fields, missing = {}, []
        for schema in data["extraction_schema"]:
            match = re.search(schema["pattern"], doc["text"], re.MULTILINE)
            reason = "not_found"
            if match and match.group("value") is not None:
                raw = match.group("value")
                quote = raw.strip()
                reason = "empty_value"
                if quote:
                    start = match.start("value") + len(raw) - len(raw.lstrip())
                    try:
                        value = convert(quote, schema["type"])
                    except ValueError:
                        reason = "invalid_value"
                    else:
                        fields[schema["name"]] = {"value": value,
                                                 "source": source(doc, start, start + len(quote))}
                        continue
            missing.append({"name": schema["name"], "reason": reason,
                            "required": schema["required"]})
        output.append({"document_id": doc["id"], "fields": fields,
                       "missing_fields": missing,
                       "complete": not any(field["required"] for field in missing)})
    return Validator.extraction({"documents": output}, ranking, data)


def tokens(text):
    return set(re.findall(r"\w+", text.casefold(), re.UNICODE))


def passages(text):
    """Use nonempty lines as bounded passages; never alter citation text."""
    for match in re.finditer(r"[^\r\n]+", text):
        raw = match.group()
        if raw.strip():
            start = match.start() + len(raw) - len(raw.lstrip())
            yield start, start + len(raw.strip())


def evidence_fields(extracted, start, end):
    return sorted(name for name, field in extracted["fields"].items()
                  if start <= field["source"]["start"] < field["source"]["end"] <= end)


def research(data, ranking, extracted):
    Validator.extraction(extracted, ranking, data)
    docs = {doc["id"]: doc for doc in data["documents"]}
    results = []
    for query in data["research"]["queries"]:
        terms = tokens(query)
        findings = []
        for rank, item in enumerate(extracted["documents"]):
            doc = docs[item["document_id"]]
            for start, end in passages(doc["text"]):
                evidence = evidence_fields(item, start, end)
                overlap = terms & tokens(doc["text"][start:end])
                if not evidence or not overlap:
                    continue
                score = len(overlap) / len(terms)
                findings.append((score, rank, start, {
                    "text": doc["text"][start:end], "score": score,
                    "citation": source(doc, start, end), "evidence_fields": evidence}))
        findings.sort(key=lambda item: (-item[0], item[1], item[2]))
        results.append({"query": query, "findings": [
            item[3] for item in findings[:data["research"]["max_findings"]]]})
    return Validator.research({"queries": results}, extracted, ranking, data)


def run_pipeline(data):
    Validator.input(data)
    ranking = personalize(data)
    extracted = extract(data, ranking)
    findings = research(data, ranking, extracted)
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "behavior": ranking, "extraction": extracted, "research": findings}


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        Validator.require(len(args) == 1, "usage: implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8-sig"),
                          parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
