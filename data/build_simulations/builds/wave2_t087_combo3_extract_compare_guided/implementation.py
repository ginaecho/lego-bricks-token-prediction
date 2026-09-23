"""Synthetic, deterministic document-to-product-to-onboarding reference CLI."""
import json
import math
import re
import sys
from pathlib import Path


SCHEMA_VERSION = "1.0"
FIELD_SCHEMA = {
    "name": {"required": True, "kind": "text", "unit": None},
    "price": {"required": True, "kind": "money", "unit": "USD"},
    "storage": {"required": True, "kind": "capacity", "unit": "GB"},
    "warranty": {"required": False, "kind": "duration", "unit": "months"},
}
STEPS = ("purchase", "configure", "transfer")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_keys(value, allowed, required, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) <= set(allowed), path + " contains unknown keys")
    require(set(required) <= set(value), path + " is missing required keys")


def number(value, path, minimum=0):
    require(type(value) in (int, float), path + " must be a number")
    require(math.isfinite(value) and value >= minimum,
            path + " must be finite and >= " + str(minimum))


def validate(value, phase="input"):
    """One shared validation boundary for the request and every stage handoff."""
    if phase == "input":
        object_keys(value, ("schema_version", "synthetic", "documents", "preferences",
                            "onboarding"),
                    ("schema_version", "synthetic", "documents", "preferences",
                     "onboarding"), "input")
        require(value["schema_version"] == SCHEMA_VERSION, "Unsupported schema_version")
        require(value["synthetic"] is True, "Fixture must be labeled synthetic")
        docs = value["documents"]
        require(isinstance(docs, list) and 1 <= len(docs) <= 100,
                "documents must contain 1..100 entries")
        ids = set()
        for doc in docs:
            object_keys(doc, ("id", "text"), ("id", "text"), "document")
            require(isinstance(doc["id"], str) and doc["id"].strip(),
                    "document.id must be nonempty text")
            require(doc["id"] not in ids, "Duplicate document.id")
            ids.add(doc["id"])
            require(isinstance(doc["text"], str) and len(doc["text"]) <= 100000,
                    "document.text must be text of at most 100000 characters")
        prefs = value["preferences"]
        object_keys(prefs, ("weights", "max_price", "min_storage"),
                    ("weights",), "preferences")
        object_keys(prefs["weights"], ("price", "storage", "warranty"),
                    ("price", "storage"), "weights")
        for field, weight in prefs["weights"].items():
            number(weight, "weight." + field)
            require(weight <= 1, "Weights must be <= 1")
        require(sum(prefs["weights"].values()) > 0, "At least one weight must be positive")
        for field in ("max_price", "min_storage"):
            if field in prefs:
                number(prefs[field], field)
        setup = value["onboarding"]
        object_keys(setup, ("completed", "answers", "transfer_gb"),
                    ("completed", "answers", "transfer_gb"), "onboarding")
        number(setup["transfer_gb"], "transfer_gb")
        completed = setup["completed"]
        require(isinstance(completed, list), "completed must be an array")
        require(all(isinstance(s, str) and s in STEPS for s in completed),
                "Unknown completed step")
        require(len(set(completed)) == len(completed), "Duplicate completed step")
        answers = setup["answers"]
        object_keys(answers, ("purchase_confirmed", "device_name", "transfer_confirmed"),
                    (), "answers")
        for key in ("purchase_confirmed", "transfer_confirmed"):
            if key in answers:
                require(type(answers[key]) is bool, key + " must be boolean")
        if "device_name" in answers:
            require(isinstance(answers["device_name"], str)
                    and 1 <= len(answers["device_name"].strip()) <= 80,
                    "device_name must contain 1..80 non-whitespace characters")
        return value
    require(phase in ("extracted", "compared", "guided"), "Unknown validation phase")
    object_keys(value, ("schema_version", "synthetic", "status", "request",
                        "extraction", "comparison", "guided"),
                ("schema_version", "synthetic", "status", "request", "extraction"),
                "pipeline")
    validate(value["request"])
    require(value["schema_version"] == SCHEMA_VERSION and value["synthetic"] is True,
            "Invalid pipeline metadata")
    require(value["status"] == "ok", "Invalid pipeline status")
    docs = value["request"]["documents"]
    rows = value["extraction"]
    require(isinstance(rows, list) and len(rows) == len(docs), "Extraction row mismatch")
    for row, doc in zip(rows, docs):
        require(row["id"] == doc["id"], "Extraction identity mismatch")
        require(set(row["fields"]) == set(FIELD_SCHEMA), "Extraction schema mismatch")
        missing = []
        for field, spec in FIELD_SCHEMA.items():
            record = row["fields"][field]
            if record is None:
                missing.append(field)
                continue
            require(record["unit"] == spec["unit"], "Unit mismatch")
            start, end = record["span"]
            require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(doc["text"]), "Invalid source span")
            require(doc["text"][start:end] == record["raw"], "Source span mismatch")
            require(normalize(record["raw"], spec) == record["value"],
                    "Normalized value mismatch")
        require(row["missing_fields"] == missing, "Missing field mismatch")
        eligible = not any(FIELD_SCHEMA[f]["required"] for f in missing)
        require(row["complete"] == eligible, "Completeness mismatch")
    if phase in ("compared", "guided"):
        require("comparison" in value, "Missing comparison")
        expected = comparison_data(value)
        require(value["comparison"] == expected, "Comparison does not match extraction")
    if phase == "guided":
        require("guided" in value, "Missing guided output")
        require(value["guided"] == guided_data(value),
                "Guided output does not match comparison")
    return value


def normalize(raw, spec):
    kind = spec["kind"]
    if kind == "text":
        require(bool(raw.strip()), "Empty text field")
        return raw.strip()
    patterns = {
        "money": r"(?:USD\s*|\$\s*)?(\d+(?:\.\d+)?)\s*(USD)?",
        "capacity": r"(\d+(?:\.\d+)?)\s*(GB|TB)",
        "duration": r"(\d+(?:\.\d+)?)\s*(months?|years?)",
    }
    match = re.fullmatch(patterns[kind], raw, re.IGNORECASE)
    require(match is not None, "Invalid " + kind + " value: " + raw)
    amount = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit == "tb":
        amount *= 1000
    elif unit.startswith("year"):
        amount *= 12
    number(amount, kind)
    require(amount <= 1e12, kind + " exceeds reference limit")
    return amount


def extract(request):
    validate(request)
    result = {"schema_version": SCHEMA_VERSION, "synthetic": True, "status": "ok",
              "request": request, "extraction": []}
    for doc in request["documents"]:
        fields = {}
        for field, spec in FIELD_SCHEMA.items():
            pattern = r"^[ \t]*" + field + r"[ \t]*:[ \t]*([^\r\n]*)"
            matches = list(re.finditer(pattern, doc["text"], re.IGNORECASE | re.MULTILINE))
            require(len(matches) <= 1, "Duplicate field " + field + " in " + doc["id"])
            fields[field] = None
            if matches and matches[0].group(1).strip():
                match = matches[0]
                raw = match.group(1).strip()
                start = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip())
                fields[field] = {"raw": raw, "span": [start, start + len(raw)],
                                 "value": normalize(raw, spec), "unit": spec["unit"]}
        missing = [key for key in FIELD_SCHEMA if fields[key] is None]
        result["extraction"].append({
            "id": doc["id"], "fields": fields, "missing_fields": missing,
            "complete": not any(FIELD_SCHEMA[key]["required"] for key in missing),
        })
    return validate(result, "extracted")


def comparison_data(pipeline):
    prefs = pipeline["request"]["preferences"]
    columns, excluded = [], []
    for row in pipeline["extraction"]:
        attrs = {key: record["value"] if record else None
                 for key, record in row["fields"].items()}
        reasons = []
        if not row["complete"]:
            reasons.append("missing_required_fields")
        else:
            if attrs["price"] > prefs.get("max_price", float("inf")):
                reasons.append("over_budget")
            if attrs["storage"] < prefs.get("min_storage", 0):
                reasons.append("insufficient_storage")
        columns.append({"id": row["id"], "attributes": attrs,
                        "eligible": not reasons})
        if reasons:
            excluded.append({"id": row["id"], "reasons": reasons})
    candidates = [col for col in columns if col["eligible"]]
    ranking = []
    weights = prefs["weights"]
    for candidate in candidates:
        utilities = {}
        for field in weights:
            values = [c["attributes"][field] or 0 for c in candidates]
            low, high = min(values), max(values)
            amount = candidate["attributes"][field] or 0
            utility = 1.0 if high == low else (amount - low) / (high - low)
            if field == "price" and high != low:
                utility = 1 - utility
            utilities[field] = utility
        score = sum(weights[f] * utilities[f] for f in weights) / sum(weights.values())
        ranking.append({"id": candidate["id"], "score": round(score, 8),
                        "utilities": utilities})
    ranking.sort(key=lambda row: (-row["score"], row["id"]))
    return {
        "units": {key: spec["unit"] for key, spec in FIELD_SCHEMA.items()},
        "side_by_side": columns, "excluded": excluded, "ranking": ranking,
        "selected_id": ranking[0]["id"] if ranking else None,
    }


def compare(pipeline):
    validate(pipeline, "extracted")
    result = dict(pipeline, comparison=comparison_data(pipeline))
    return validate(result, "compared")


def guided_data(pipeline):
    selected = pipeline["comparison"]["selected_id"]
    setup = pipeline["request"]["onboarding"]
    completed, answers = set(setup["completed"]), setup["answers"]
    if selected is None:
        require(not completed, "Cannot complete steps without a selected product")
        return {"selected_id": None, "status": "blocked", "reason": "no_eligible_product",
                "steps": [], "next_step": None, "progress": 0.0}
    product = next(c for c in pipeline["comparison"]["side_by_side"]
                   if c["id"] == selected)
    capacity_ok = product["attributes"]["storage"] >= setup["transfer_gb"]
    evidence = {
        "purchase": answers.get("purchase_confirmed") is True,
        "configure": bool(answers.get("device_name", "").strip()),
        "transfer": answers.get("transfer_confirmed") is True,
    }
    rows = []
    for index, step in enumerate(STEPS):
        prerequisites = list(STEPS[:index])
        blockers = [s for s in prerequisites if s not in completed]
        if step == "transfer" and not capacity_ok:
            blockers.append("insufficient_selected_storage")
        if step in completed:
            require(not blockers, "Unmet prerequisites for " + step)
            require(evidence[step], "Missing completion evidence for " + step)
        rows.append({"id": step, "prerequisites": prerequisites,
                     "status": "completed" if step in completed else
                     ("blocked" if blockers else "ready"),
                     "blockers": blockers,
                     "evidence_valid": evidence[step]})
    ready = next((row["id"] for row in rows if row["status"] == "ready"), None)
    return {"selected_id": selected,
            "status": "complete" if len(completed) == len(STEPS) else
            ("in_progress" if ready else "blocked"),
            "steps": rows, "next_step": ready,
            "progress": round(len(completed) / len(STEPS), 6)}


def guide(pipeline):
    validate(pipeline, "compared")
    return validate(dict(pipeline, guided=guided_data(pipeline)), "guided")


def run(request):
    return guide(compare(extract(request)))


def reject_constant(text):
    raise ValidationError("Non-finite JSON constant: " + text)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        request = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                             parse_constant=reject_constant, object_pairs_hook=unique_keys)
        output = run(request)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
