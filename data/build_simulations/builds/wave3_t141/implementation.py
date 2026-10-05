"""Synthetic healthcare sentiment triage, not clinical decision support.

Only controlled, identifier-free fixture narratives are accepted. This bounded
validation demonstrates de-identification safeguards; it is not certification.
"""
import copy
import json
import re
import sys
from pathlib import Path


BUILD_ID = "wave3_t141_single_sentiment__ind-healthcare"
LEXICON = {"happy": 2, "helpful": 2, "clear": 1, "satisfied": 2,
           "delay": -2, "delayed": -2, "denied": -3, "confusing": -1,
           "frustrated": -2, "poor": -2, "resolved": 2}
NEGATORS = {"not", "no", "never"}
VOCABULARY = set("""
synthetic patient clinical note prior authorization request care service
staff communication process appointment result results diagnosis lab review
human needs pending approved submitted routine urgent administrative
is was are were the a an and but with for of to very reports reported
feels feeling has had experience support response received waiting
not no never
""".split()) | set(LEXICON)
IMPACT = {"low": 0, "medium": 10, "high": 20, "critical": 30}
KINDS = {"Patient", "DocumentReference", "Claim"}
LABS = {"hemoglobin": ("g/dL", 0, 30), "glucose": ("mg/dL", 0, 1000)}
DIAGNOSES = {"synthetic-diabetes", "synthetic-anemia", "synthetic-hypertension"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact(obj, fields, location):
    require(isinstance(obj, dict) and set(obj) == set(fields),
            location + ": unexpected or missing fields (identifiers prohibited)")


def narrative(text):
    require(isinstance(text, str) and 0 < len(text) <= 2000,
            "text must be a nonempty controlled synthetic narrative")
    require(re.fullmatch(r"[A-Za-z ,.!?;:\-]+", text) is not None,
            "text contains prohibited identifier-like characters")
    words = re.findall(r"[a-z]+", text.lower())
    require(bool(words) and all(word in VOCABULARY for word in words),
            "text contains unsupported vocabulary; use identifier-free fixture terms")
    return words


def validate(data):
    exact(data, {"schema_version", "synthetic", "records"}, "input")
    require(data["schema_version"] == "1.0", "unsupported schema_version")
    require(data["synthetic"] is True, "only clearly labeled synthetic data accepted")
    records = data["records"]
    require(isinstance(records, list) and 0 < len(records) <= 1000,
            "records must contain 1 to 1000 entries")
    ids = set()
    for record in records:
        require(isinstance(record, dict), "record must be an object")
        kind = record.get("resourceType")
        require(isinstance(kind, str) and kind in KINDS, "unsupported resourceType")
        fields = {"id", "resourceType", "synthetic", "version", "impact", "text"}
        fields |= {"diagnoses", "labs"} if kind == "Patient" else {"patientRef"}
        if kind == "Claim":
            fields |= {"use", "status"}
        exact(record, fields, "record")
        rid = record["id"]
        require(isinstance(rid, str) and re.fullmatch(r"syn-[a-z]{1,16}-[0-9]{1,4}", rid),
                "id must be a synthetic local reference")
        require(rid not in ids, "duplicate record id")
        ids.add(rid)
        require(record["synthetic"] is True, "record must be synthetic")
        require(type(record["version"]) is int and record["version"] >= 1,
                "version must be a positive integer")
        require(isinstance(record["impact"], str) and record["impact"] in IMPACT,
                "impact must be low, medium, high, or critical")
        narrative(record["text"])
        if kind == "Patient":
            diagnoses = record["diagnoses"]
            require(isinstance(diagnoses, list) and all(
                isinstance(d, str) and d in DIAGNOSES for d in diagnoses),
                "only approved synthetic diagnoses accepted")
            require(isinstance(record["labs"], list), "labs must be an array")
            for lab in record["labs"]:
                exact(lab, {"code", "value", "unit"}, "lab")
                code = lab["code"]
                require(isinstance(code, str) and code in LABS, "unsupported synthetic lab")
                unit, low, high = LABS[code]
                require(lab["unit"] == unit and type(lab["value"]) in (int, float)
                        and low <= lab["value"] <= high,
                        "invalid synthetic lab value or unit")
        else:
            require(isinstance(record["patientRef"], str), "patientRef must be a string")
        if kind == "Claim":
            require(record["use"] == "preauthorization", "Claim must be preauthorization")
            require(isinstance(record["status"], str) and
                    record["status"] in {"pending", "approved", "denied"},
                    "invalid authorization workflow status")
    patients = {r["id"] for r in records if r["resourceType"] == "Patient"}
    for record in records:
        if record["resourceType"] != "Patient":
            require(record["patientRef"] in patients, "patientRef must resolve to a Patient")
    return data


def score(text):
    """A negator affects the next three words, stopping at punctuation or 'but'."""
    tokens = re.findall(r"[a-z]+|[,.!?;:\-]", text.lower())
    evidence = []
    remaining = 0
    word_index = -1
    for token in tokens:
        if not token.isalpha() or token == "but":
            remaining = 0
            continue
        word_index += 1
        if token in NEGATORS:
            remaining = 3
            continue
        if token in LEXICON:
            base = LEXICON[token]
            evidence.append({"term": token, "word_index": word_index, "base": base,
                             "negated": remaining > 0,
                             "contribution": -base if remaining else base})
        remaining = max(0, remaining - 1)
    total = sum(item["contribution"] for item in evidence)
    return {"score": total, "label": "positive" if total > 0 else
            "negative" if total < 0 else "neutral", "evidence": evidence}


def run(data):
    validate(data)
    output, audit = [], []
    for index, original in enumerate(data["records"], 1):
        record = copy.deepcopy(original)
        sentiment = score(record["text"])
        negative_weight = min(9, max(0, -sentiment["score"]))
        analysis = {
            "sentiment": sentiment,
            "priority_score": IMPACT[record["impact"]] + negative_weight,
            "priority_components": {"human_supplied_impact": IMPACT[record["impact"]],
                                    "negative_sentiment": negative_weight},
            "human_review_required": True,
            "clinical_decision": None,
            "recommended_action": "human_administrative_review",
        }
        record["analysis"] = analysis
        record["version"] += 1
        output.append(record)
        audit.append({"event_id": f"event-{index:04d}", "record_id": record["id"],
                      "actor": "deterministic-synthetic-triage",
                      "operation": "annotate", "reason": "sentiment and administrative prioritization",
                      "changes": [
                          {"path": "version", "before": original["version"], "after": record["version"]},
                          {"path": "analysis", "before": None, "after": copy.deepcopy(analysis)}
                      ]})
    queue = sorted(output, key=lambda r: (-r["analysis"]["priority_score"], r["id"]))
    return {"status": "ok", "schema_version": "1.0", "synthetic": True,
            "human_review_required": True, "records": output,
            "review_queue": [r["id"] for r in queue], "audit_trail": audit,
            "notice": "Synthetic demonstration only; not HIPAA certification or clinical advice."}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValidationError("nonfinite JSON number")))
        result = run(data)
    except (OSError, UnicodeError, ValueError, RecursionError):
        # Do not echo paths, narratives, or identifiers from rejected input.
        print(json.dumps({"status": "error", "message": "Input file or schema validation failed."}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
