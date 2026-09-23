"""Deterministic synthetic marketplace discovery -> extraction -> research CLI."""

import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def keys(obj, required, optional=()):
    require(isinstance(obj, dict), "Expected object")
    require(set(required) <= set(obj), "Missing keys: " + str(set(required) - set(obj)))
    require(set(obj) <= set(required) | set(optional), "Unknown keys: " + str(set(obj) - set(required) - set(optional)))


def citation(document, start, end):
    return {"document_id": document["id"], "start": start, "end": end,
            "quote": document["text"][start:end]}


def validate_citation(cite, documents):
    keys(cite, ("document_id", "start", "end", "quote"))
    require(cite["document_id"] in documents, "Unknown citation document")
    start, end = cite["start"], cite["end"]
    require(type(start) is int and type(end) is int, "Citation offsets must be integers")
    source = documents[cite["document_id"]]["text"]
    require(0 <= start < end <= len(source), "Citation span out of bounds")
    require(cite["quote"] == source[start:end], "Citation does not match source")


def validate(value, stage="input"):
    """One boundary validator shared by every stage; offsets are Unicode code points."""
    if stage == "input":
        keys(value, ("schema_version", "synthetic", "items", "documents", "events",
                     "fields", "query", "config"))
        require(value["schema_version"] == "1.0", "Unsupported schema version")
        require(value["synthetic"] is True, "Fixtures must be labeled synthetic")
        for name in ("items", "documents", "events", "fields"):
            require(isinstance(value[name], list), name + " must be an array")
        require(text(value["query"]), "Query must be nonempty")
        config = value["config"]
        keys(config, ("half_life_days", "max_items", "max_findings"))
        require(number(config["half_life_days"]) and config["half_life_days"] > 0,
                "half_life_days must be positive and finite")
        for name in ("max_items", "max_findings"):
            require(type(config[name]) is int and config[name] > 0, name + " must be a positive integer")
        docs = {}
        for doc in value["documents"]:
            keys(doc, ("id", "title", "text"))
            require(text(doc["id"]) and doc["id"] not in docs, "Duplicate/invalid document id")
            require(text(doc["title"]) and isinstance(doc["text"], str), "Invalid document content")
            docs[doc["id"]] = doc
        items = {}
        for item in value["items"]:
            keys(item, ("id", "categories", "popularity", "document_ids"))
            require(text(item["id"]) and item["id"] not in items, "Duplicate/invalid item id")
            for name in ("categories", "document_ids"):
                require(isinstance(item[name], list) and all(text(x) for x in item[name]),
                        name + " must contain strings")
                require(len(set(item[name])) == len(item[name]), "Duplicate " + name)
            require(number(item["popularity"]) and 0 <= item["popularity"] <= 1,
                    "Popularity must be in [0, 1]")
            require(all(d in docs for d in item["document_ids"]), "Unknown item document")
            items[item["id"]] = item
        for event in value["events"]:
            keys(event, ("item_id", "kind", "age_days"))
            require(text(event["item_id"]) and event["item_id"] in items, "Unknown event item")
            require(event["kind"] in ("browse", "purchase"), "Unknown event kind")
            require(number(event["age_days"]) and event["age_days"] >= 0, "Invalid event age")
        names = set()
        for field in value["fields"]:
            keys(field, ("name", "pattern", "type", "required"))
            require(text(field["name"]) and field["name"] not in names, "Duplicate/invalid field name")
            names.add(field["name"])
            require(field["type"] in ("string", "number") and type(field["required"]) is bool,
                    "Invalid field type/required flag")
            require(text(field["pattern"]), "Invalid field pattern")
            try:
                pattern = re.compile(field["pattern"])
            except re.error as exc:
                raise ValidationError("Invalid field regex: " + str(exc)) from exc
            require(pattern.groups == 1, "Field regex must have exactly one capture group")
        return value

    require(stage in ("behavior", "extract", "normal"), "Unknown validation stage")
    required = ["schema_version", "status", "input", "behavior"]
    if stage in ("extract", "normal"):
        required.append("extraction")
    if stage == "normal":
        required.append("research")
    keys(value, required)
    require(value["schema_version"] == "1.0" and value["status"] == "ok", "Invalid envelope")
    data = validate(value["input"])
    docs = {d["id"]: d for d in data["documents"]}
    items = {i["id"]: i for i in data["items"]}
    behavior = value["behavior"]
    keys(behavior, ("mode", "ranked_items", "document_ids"))
    require(behavior["mode"] in ("personalized", "cold_start"), "Invalid ranking mode")
    require(isinstance(behavior["ranked_items"], list), "Invalid ranking")
    seen, doc_ids, last = set(), [], None
    require(len(behavior["ranked_items"]) == min(len(items), data["config"]["max_items"]),
            "Incorrect ranked item count")
    for row in behavior["ranked_items"]:
        keys(row, ("item_id", "score"))
        require(text(row["item_id"]) and row["item_id"] in items and row["item_id"] not in seen,
                "Invalid ranked item")
        require(number(row["score"]) and row["score"] >= 0, "Invalid ranking score")
        order = (-row["score"], row["item_id"])
        require(last is None or last <= order, "Ranking must be sorted")
        last = order
        seen.add(row["item_id"])
        for doc_id in items[row["item_id"]]["document_ids"]:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
    require(behavior["document_ids"] == doc_ids, "Ranking document handoff mismatch")
    if stage == "behavior":
        return value
    extraction = value["extraction"]
    keys(extraction, ("records",))
    require(isinstance(extraction["records"], list), "Invalid extraction records")
    require([r.get("document_id") for r in extraction["records"] if isinstance(r, dict)] == doc_ids,
            "Extraction document handoff mismatch")
    definitions = {f["name"]: f for f in data["fields"]}
    for record in extraction["records"]:
        keys(record, ("document_id", "fields", "missing_fields", "missing_required_fields", "errors"))
        require(isinstance(record["fields"], dict) and set(record["fields"]) <= set(definitions),
                "Invalid extracted fields")
        for name, extracted in record["fields"].items():
            keys(extracted, ("value", "citation"))
            spec = definitions[name]
            require(number(extracted["value"]) if spec["type"] == "number" else text(extracted["value"]),
                    "Extracted field type mismatch")
            validate_citation(extracted["citation"], docs)
            require(extracted["citation"]["document_id"] == record["document_id"], "Field source mismatch")
            match = re.search(spec["pattern"], docs[record["document_id"]]["text"])
            require(match is not None and match.span(1) ==
                    (extracted["citation"]["start"], extracted["citation"]["end"]), "Field span mismatch")
            expected = float(match.group(1)) if spec["type"] == "number" else match.group(1)
            require(extracted["value"] == expected, "Field value mismatch")
        missing = [f["name"] for f in data["fields"] if f["name"] not in record["fields"]]
        require(record["missing_fields"] == missing, "Missing field report mismatch")
        require(record["missing_required_fields"] ==
                [n for n in missing if definitions[n]["required"]], "Missing required report mismatch")
        require(isinstance(record["errors"], list), "Invalid extraction errors")
        for error in record["errors"]:
            keys(error, ("field", "reason"))
            require(error["field"] in missing and text(error["reason"]), "Invalid field error")
    if stage == "extract":
        return value
    research = value["research"]
    keys(research, ("query", "findings", "no_results"))
    require(research["query"] == data["query"], "Research query mismatch")
    require(isinstance(research["findings"], list) and
            len(research["findings"]) <= data["config"]["max_findings"], "Invalid findings")
    require(type(research["no_results"]) is bool and
            research["no_results"] == (not research["findings"]), "Invalid no-results flag")
    records = {r["document_id"]: r for r in extraction["records"]}
    for finding in research["findings"]:
        keys(finding, ("text", "score", "citation", "extracted_fields"))
        validate_citation(finding["citation"], docs)
        require(finding["citation"]["document_id"] in records, "Research escaped extraction scope")
        require(finding["text"] == finding["citation"]["quote"], "Finding must be extractive")
        require(number(finding["score"]) and 0 < finding["score"] <= 1, "Invalid retrieval score")
        cite = finding["citation"]
        expected = [name for name, field in records[cite["document_id"]]["fields"].items()
                    if cite["start"] <= field["citation"]["start"] and field["citation"]["end"] <= cite["end"]]
        require(finding["extracted_fields"] == expected, "Research field handoff mismatch")
    return value


def personalize(data):
    validate(data)
    items = {item["id"]: item for item in data["items"]}
    preferences, direct = {}, {}
    for event in data["events"]:
        weight = (3 if event["kind"] == "purchase" else 1) * (
            2 ** (-event["age_days"] / data["config"]["half_life_days"]))
        direct[event["item_id"]] = direct.get(event["item_id"], 0) + weight
        categories = items[event["item_id"]]["categories"]
        for category in categories:
            preferences[category] = preferences.get(category, 0) + weight / len(categories)
    active = any(weight > 0 for weight in direct.values())
    ranked = []
    for item in data["items"]:
        score = (sum(preferences.get(c, 0) for c in item["categories"]) +
                 0.25 * direct.get(item["id"], 0)) if active else item["popularity"]
        ranked.append({"item_id": item["id"], "score": score})
    ranked.sort(key=lambda row: (-row["score"], row["item_id"]))
    ranked = ranked[:data["config"]["max_items"]]
    doc_ids = list(dict.fromkeys(d for row in ranked for d in items[row["item_id"]]["document_ids"]))
    return validate({"schema_version": "1.0", "status": "ok", "input": data,
                     "behavior": {"mode": "personalized" if active else "cold_start",
                                  "ranked_items": ranked, "document_ids": doc_ids}}, "behavior")


def extract(previous):
    validate(previous, "behavior")
    data = previous["input"]
    docs = {doc["id"]: doc for doc in data["documents"]}
    records = []
    for doc_id in previous["behavior"]["document_ids"]:
        doc = docs[doc_id]
        fields, missing, errors = {}, [], []
        for spec in data["fields"]:
            match = re.search(spec["pattern"], doc["text"])
            raw = match.group(1) if match else None
            if raw is None or not raw.strip():
                missing.append(spec["name"])
                continue
            try:
                result = float(raw) if spec["type"] == "number" else raw
                if spec["type"] == "number" and not number(result):
                    raise ValueError("nonfinite")
            except ValueError:
                missing.append(spec["name"])
                errors.append({"field": spec["name"], "reason": "not a finite number"})
                continue
            fields[spec["name"]] = {"value": result, "citation": citation(doc, *match.span(1))}
        records.append({"document_id": doc_id, "fields": fields, "missing_fields": missing,
                        "missing_required_fields": [s["name"] for s in data["fields"]
                                                    if s["required"] and s["name"] in missing],
                        "errors": errors})
    result = dict(previous, extraction={"records": records})
    return validate(result, "extract")


def tokens(value):
    return set(re.findall(r"\w+", value.casefold()))


def research(previous):
    validate(previous, "extract")
    data = previous["input"]
    docs = {doc["id"]: doc for doc in data["documents"]}
    query = tokens(data["query"])
    findings = []
    for record in previous["extraction"]["records"]:
        doc = docs[record["document_id"]]
        # Sentence-like passages retain exact offsets, including punctuation.
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", doc["text"]):
            raw = match.group()
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.end() - len(raw) + len(raw.rstrip())
            passage = doc["text"][start:end]
            overlap = len(query & tokens(passage))
            if not overlap:
                continue
            linked = [name for name, field in record["fields"].items()
                      if start <= field["citation"]["start"] and field["citation"]["end"] <= end]
            findings.append({"text": passage, "score": overlap / len(query),
                             "citation": citation(doc, start, end), "extracted_fields": linked})
    findings.sort(key=lambda f: (-f["score"], f["citation"]["document_id"], f["citation"]["start"]))
    findings = findings[:data["config"]["max_findings"]]
    result = dict(previous, research={"query": data["query"], "findings": findings,
                                    "no_results": not findings})
    return validate(result, "normal")


def run(data):
    return research(extract(personalize(data)))


def reject_constant(value):
    raise ValidationError("Nonfinite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run(data)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
