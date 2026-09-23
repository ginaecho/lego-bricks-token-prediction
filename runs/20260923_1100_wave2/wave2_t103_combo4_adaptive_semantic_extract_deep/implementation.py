"""Deterministic synthetic reference pipeline. No network or third-party dependencies."""
import datetime
import json
import math
from pathlib import Path
import re
import sys


SCHEMA_VERSION = "1.0"
EXPERIENCES = ("novice", "intermediate", "expert")
PREREQUISITES = ("search_basics", "source_verification")
TYPES = ("string", "number", "boolean", "date")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_keys(value, required, optional=(), path="input"):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(required) <= set(value), f"{path} missing keys: {sorted(set(required) - set(value))}")
    require(set(value) <= set(required) | set(optional), f"{path} contains unknown keys")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonempty text")


def strings(value, path):
    require(isinstance(value, list), f"{path} must be a list")
    for entry in value:
        text(entry, path)
    require(len(value) == len(set(value)), f"{path} contains duplicates")


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def validate_input(data):
    object_keys(data, ("schema_version", "synthetic", "profile", "query", "documents", "fields"), ("options",))
    require(data["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    require(data["synthetic"] is True, "reference input must be labeled synthetic")
    text(data["query"], "query")
    require(bool(tokens(data["query"])), "query must contain searchable tokens")
    profile = data["profile"]
    object_keys(profile, ("experience", "focus_terms", "completed_prerequisites", "explanation_detail"), path="profile")
    require(profile["experience"] in EXPERIENCES, "invalid experience")
    strings(profile["focus_terms"], "focus_terms")
    strings(profile["completed_prerequisites"], "completed_prerequisites")
    require(set(profile["completed_prerequisites"]) <= set(PREREQUISITES), "unknown prerequisite")
    require(profile["explanation_detail"] in ("brief", "detailed"), "invalid explanation_detail")
    require(isinstance(data["documents"], list), "documents must be a list")
    ids = []
    for doc in data["documents"]:
        object_keys(doc, ("id", "title", "text"), path="document")
        for key in ("id", "title", "text"):
            text(doc[key], f"document.{key}")
        ids.append(doc["id"])
    require(len(ids) == len(set(ids)), "duplicate document id")
    require(isinstance(data["fields"], list) and bool(data["fields"]), "fields must be a nonempty list")
    names = []
    used_aliases = set()
    for field in data["fields"]:
        object_keys(field, ("name", "aliases", "type", "required"), path="field")
        text(field["name"], "field.name")
        require(bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field["name"])), "invalid field name")
        strings(field["aliases"], "field.aliases")
        require(bool(field["aliases"]), "field aliases cannot be empty")
        require(field["type"] in TYPES, "invalid field type")
        require(type(field["required"]) is bool, "field.required must be boolean")
        for alias in field["aliases"]:
            require(alias == alias.strip() and "\n" not in alias and "\r" not in alias and ":" not in alias,
                    "aliases must be single-line labels without colons")
            require(alias.casefold() not in used_aliases, "ambiguous duplicate alias")
            used_aliases.add(alias.casefold())
        names.append(field["name"])
    require(len(names) == len(set(names)), "duplicate field name")
    options = data.get("options", {})
    object_keys(options, (), ("top_k", "min_score"), path="options")
    require(type(options.get("top_k", 5)) is int and 1 <= options.get("top_k", 5) <= 100, "top_k must be 1..100")
    minimum = options.get("min_score", 0.01)
    require(number(minimum) and 0 <= minimum <= 1, "min_score must be finite and in [0,1]")
    return data


def validate_stage(name, stage, data, previous=None):
    """One shared boundary validator: shape, provenance, and handoff invariants."""
    require(isinstance(stage, dict), f"{name} output must be an object")
    docs = {doc["id"]: doc for doc in data["documents"]}
    fields = {field["name"]: field for field in data["fields"]}
    if name == "adaptive":
        object_keys(stage, ("query", "focus_terms", "required_prerequisites", "missing_prerequisites",
                            "ready", "steps", "explanation"), path=name)
        require(stage["query"] == data["query"], "onboarding changed query")
        require(stage["focus_terms"] == data["profile"]["focus_terms"], "onboarding changed preferences")
        expected = list(PREREQUISITES if data["profile"]["experience"] == "novice" else PREREQUISITES[1:])
        require(stage["required_prerequisites"] == expected, "invalid prerequisite plan")
        missing = [item for item in expected if item not in data["profile"]["completed_prerequisites"]]
        require(stage["missing_prerequisites"] == missing, "incorrect prerequisite state")
        require(type(stage["ready"]) is bool and stage["ready"] == (not missing), "incorrect readiness")
        require(stage["steps"] == expected + ["semantic", "extract", "deep"], "invalid step ordering")
        text(stage["explanation"], "explanation")
    elif name == "semantic":
        object_keys(stage, ("query", "focus_terms", "mode", "index", "results"), path=name)
        require(previous["ready"], "search requires completed onboarding prerequisites")
        require(stage["query"] == previous["query"] and stage["focus_terms"] == previous["focus_terms"],
                "search lost onboarding context")
        require(stage["mode"] in ("lexical", "hybrid"), "invalid search mode")
        require(isinstance(stage["index"], dict), "invalid index")
        for token, postings in stage["index"].items():
            text(token, "index token")
            strings(postings, "postings")
            require(set(postings) <= set(docs), "index references unknown document")
        require(isinstance(stage["results"], list), "results must be a list")
        seen = set()
        ordering = []
        require(len(stage["results"]) <= data.get("options", {}).get("top_k", 5), "too many results")
        for result in stage["results"]:
            object_keys(result, ("document_id", "score", "matched_terms"), path="result")
            doc_id = result["document_id"]
            require(isinstance(doc_id, str) and doc_id in docs and doc_id not in seen, "invalid search document")
            require(number(result["score"]) and 0 <= result["score"] <= 1, "invalid relevance score")
            require(result["score"] >= data.get("options", {}).get("min_score", 0.01), "score below threshold")
            strings(result["matched_terms"], "matched_terms")
            seen.add(doc_id)
            ordering.append((-result["score"], doc_id))
        require(ordering == sorted(ordering), "search results are not ranked")
    elif name == "extract":
        object_keys(stage, ("documents",), path=name)
        require(isinstance(stage["documents"], list), "extracted documents must be a list")
        require(all(isinstance(doc, dict) for doc in stage["documents"]), "extracted document must be an object")
        require([doc.get("document_id") for doc in stage["documents"] if isinstance(doc, dict)] ==
                [result["document_id"] for result in previous["results"]], "extraction lost search ordering")
        for document, result in zip(stage["documents"], previous["results"]):
            object_keys(document, ("document_id", "relevance", "fields", "missing_fields", "invalid_fields"), path=name)
            require(document["relevance"] == result["score"], "extraction lost relevance")
            require(isinstance(document["fields"], dict) and set(document["fields"]) == set(fields), "field schema mismatch")
            source = docs[document["document_id"]]["text"]
            for field_name, entries in document["fields"].items():
                require(isinstance(entries, list), "field values must be a list")
                for entry in entries:
                    object_keys(entry, ("value", "raw", "span"), path="evidence")
                    validate_span(entry, source)
                    converted = convert(entry["raw"], fields[field_name]["type"])
                    require(type(entry["value"]) is type(converted) and entry["value"] == converted,
                            "typed evidence mismatch")
            expected_missing = [key for key, field in fields.items() if field["required"] and not document["fields"][key]]
            require(document["missing_fields"] == expected_missing, "incorrect missing fields")
            require(isinstance(document["invalid_fields"], list), "invalid_fields must be a list")
            for invalid in document["invalid_fields"]:
                object_keys(invalid, ("field", "raw", "span", "reason"), path="invalid field")
                require(invalid["field"] in fields, "unknown invalid field")
                validate_span(invalid, source)
                text(invalid["reason"], "invalid reason")
    elif name == "deep":
        object_keys(stage, ("documents_considered", "findings", "disagreements", "unresolved_questions", "summary"), path=name)
        require(stage == synthesize(previous, data), "research must be grounded in extracted evidence")
    else:
        raise ValidationError("unknown stage")
    return stage


def validate_span(entry, source):
    span = entry["span"]
    require(isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span), "invalid span")
    start, end = span
    require(0 <= start <= end <= len(source) and source[start:end] == entry["raw"], "source span mismatch")


def adaptive(data):
    validate_input(data)
    profile = data["profile"]
    required = list(PREREQUISITES if profile["experience"] == "novice" else PREREQUISITES[1:])
    missing = [item for item in required if item not in profile["completed_prerequisites"]]
    explanation = "Verify sources before synthesis; preferences refine relevance, not evidence."
    if profile["explanation_detail"] == "detailed":
        explanation += (" Novices first learn search basics. Source verification remains mandatory for all levels."
                        " Extraction preserves original character spans; research reports conflicting values without choosing a winner.")
    stage = {"query": data["query"], "focus_terms": list(profile["focus_terms"]),
             "required_prerequisites": required, "missing_prerequisites": missing,
             "ready": not missing, "steps": required + ["semantic", "extract", "deep"],
             "explanation": explanation}
    return validate_stage("adaptive", stage, data)


def embedding_scores(embedder, query, documents):
    try:
        vectors = embedder([query] + [doc["title"] + "\n" + doc["text"] for doc in documents])
        require(isinstance(vectors, (list, tuple)) and len(vectors) == len(documents) + 1,
                "embedding count mismatch")
        dimension = None
        unit_vectors = []
        for vector in vectors:
            require(isinstance(vector, (list, tuple)) and bool(vector), "embedding vector must be nonempty")
            require(all(number(value) for value in vector), "embedding values must be finite numbers")
            dimension = len(vector) if dimension is None else dimension
            require(len(vector) == dimension, "embedding dimension mismatch")
            scale = max(abs(value) for value in vector)
            require(scale > 0, "embedding vector cannot be zero")
            scaled = [value / scale for value in vector]
            norm = math.sqrt(sum(value * value for value in scaled))
            unit_vectors.append([value / norm for value in scaled])
        return [max(0.0, min(1.0, sum(a * b for a, b in zip(unit_vectors[0], vector))))
                for vector in unit_vectors[1:]]
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc


def semantic(data, onboarding, embedder=None):
    validate_input(data)
    validate_stage("adaptive", onboarding, data)
    require(onboarding["ready"], "onboarding prerequisites unmet: " + ", ".join(onboarding["missing_prerequisites"]))
    weights = {term: 2.0 for term in tokens(onboarding["query"])}
    for term in tokens(" ".join(onboarding["focus_terms"])):
        weights.setdefault(term, 1.0)
    documents = data["documents"]
    index = {}
    for doc in documents:
        for term in sorted(set(tokens(doc["title"] + "\n" + doc["text"]))):
            index.setdefault(term, []).append(doc["id"])
    index = {term: sorted(ids) for term, ids in sorted(index.items())}
    weighted = {term: weight * (1 + math.log((len(documents) + 1) / (len(index.get(term, [])) + 1)))
                for term, weight in weights.items()}
    denominator = sum(weighted.values())
    similarities = embedding_scores(embedder, onboarding["query"] + " " + " ".join(onboarding["focus_terms"]), documents) if embedder is not None else None
    results = []
    minimum = data.get("options", {}).get("min_score", 0.01)
    for i, doc in enumerate(documents):
        matched = sorted(term for term in weights if doc["id"] in index.get(term, []))
        lexical = sum(weighted[term] for term in matched) / denominator
        score = lexical if similarities is None else 0.7 * lexical + 0.3 * similarities[i]
        score = round(min(1.0, score), 12)
        if score >= minimum:
            results.append({"document_id": doc["id"], "score": score, "matched_terms": matched})
    results.sort(key=lambda result: (-result["score"], result["document_id"]))
    stage = {"query": onboarding["query"], "focus_terms": list(onboarding["focus_terms"]),
             "mode": "lexical" if embedder is None else "hybrid", "index": index,
             "results": results[:data.get("options", {}).get("top_k", 5)]}
    return validate_stage("semantic", stage, data, onboarding)


def convert(raw, kind):
    if kind == "string":
        require(bool(raw), "empty string")
        return raw
    if kind == "number":
        require(bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw)), "invalid number")
        value = float(raw)
        require(math.isfinite(value), "nonfinite number")
        return value
    if kind == "boolean":
        require(raw.casefold() in ("true", "false", "yes", "no"), "invalid boolean")
        return raw.casefold() in ("true", "yes")
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw)), "invalid ISO date")
    try:
        return datetime.date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValidationError("invalid calendar date") from exc


def extract(data, search, onboarding):
    validate_input(data)
    validate_stage("adaptive", onboarding, data)
    validate_stage("semantic", search, data, onboarding)
    docs = {doc["id"]: doc for doc in data["documents"]}
    output = []
    for result in search["results"]:
        source = docs[result["document_id"]]["text"]
        values = {}
        invalid = []
        for field in data["fields"]:
            aliases = "|".join(re.escape(alias) for alias in sorted(field["aliases"], key=len, reverse=True))
            pattern = re.compile(r"^[ \t]*(?:" + aliases + r")[ \t]*:[ \t]*(?P<value>[^\r\n]*)", re.MULTILINE | re.IGNORECASE)
            entries = []
            for match in pattern.finditer(source):
                untrimmed = match.group("value")
                raw = untrimmed.strip()
                start = match.start("value") + len(untrimmed) - len(untrimmed.lstrip())
                span = [start, start + len(raw)]
                try:
                    value = convert(raw, field["type"])
                    entries.append({"value": value, "raw": raw, "span": span})
                except ValidationError as exc:
                    invalid.append({"field": field["name"], "raw": raw, "span": span, "reason": str(exc)})
            values[field["name"]] = entries
        missing = [field["name"] for field in data["fields"] if field["required"] and not values[field["name"]]]
        output.append({"document_id": result["document_id"], "relevance": result["score"], "fields": values,
                       "missing_fields": missing, "invalid_fields": invalid})
    return validate_stage("extract", {"documents": output}, data, search)


def synthesize(extracted, data):
    findings = []
    disagreements = []
    questions = []
    if not extracted["documents"]:
        questions.append("No documents met the search threshold; broaden the query or supply evidence.")
    for field in data["fields"]:
        name = field["name"]
        groups = {}
        for document in extracted["documents"]:
            for entry in document["fields"][name]:
                key = json.dumps(entry["value"], sort_keys=True, ensure_ascii=False)
                group = groups.setdefault(key, {"value": entry["value"], "evidence": [], "supporting_documents": []})
                group["evidence"].append({"document_id": document["document_id"], "span": entry["span"], "raw": entry["raw"]})
                if document["document_id"] not in group["supporting_documents"]:
                    group["supporting_documents"].append(document["document_id"])
            if name in document["missing_fields"]:
                questions.append(f"What is {name} in document {document['document_id']}?")
            for invalid in document["invalid_fields"]:
                if invalid["field"] == name:
                    questions.append(f"How should invalid {name} in document {document['document_id']} at {invalid['span']} be corrected?")
        variants = [groups[key] for key in sorted(groups)]
        state = "unresolved" if not variants else "disputed" if len(variants) > 1 else "supported"
        findings.append({"field": name, "state": state, "variants": variants})
        if len(variants) > 1:
            disagreements.append({"field": name, "variants": variants})
            questions.append(f"Which reported {name} value applies? Sources disagree; no winner was inferred.")
        elif not variants:
            questions.append(f"No usable evidence for {name}; what source can supply it?")
    return {"documents_considered": [doc["document_id"] for doc in extracted["documents"]],
            "findings": findings, "disagreements": disagreements,
            "unresolved_questions": list(dict.fromkeys(questions)),
            "summary": f"Synthesized {len(extracted['documents'])} synthetic documents; {len(disagreements)} disputed fields. Support is not proof of truth."}


def deep(data, extracted, search, onboarding):
    validate_input(data)
    validate_stage("adaptive", onboarding, data)
    validate_stage("semantic", search, data, onboarding)
    validate_stage("extract", extracted, data, search)
    return validate_stage("deep", synthesize(extracted, data), data, extracted)


def run_pipeline(data, embedder=None):
    validate_input(data)
    onboarding = adaptive(data)
    search = semantic(data, onboarding, embedder)
    extracted = extract(data, search, onboarding)
    research = deep(data, extracted, search, onboarding)
    return {"schema_version": SCHEMA_VERSION, "status": "ok", "synthetic": True,
            "stages": {"adaptive": onboarding, "semantic": search, "extract": extracted, "deep": research}}


def reject_constant(value):
    raise ValidationError("nonfinite JSON constant: " + value)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(data)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
