"""Deterministic document/evidence review; lexical support is not certification."""

import json
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


DISCLAIMER = (
    "Lexical evidence screening only. Results are not certification, compliance "
    "approval, or verification that supplied evidence is authentic or sufficient."
)


def fail(path, message):
    raise ValidationError(f"{path}: {message}")


def fields(value, expected, path):
    if not isinstance(value, dict):
        fail(path, "must be an object")
    if set(value) != set(expected):
        fail(path, "expected exactly these fields: " + ", ".join(sorted(expected)))


def text(value, path, limit=20000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        fail(path, f"must be a nonblank string of at most {limit} characters")


def array(value, path, maximum, minimum=0):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        fail(path, f"must be an array with {minimum} to {maximum} entries")


def validate(payload):
    """One validation boundary shared by the Python API and CLI."""
    fields(payload, {"schema_version", "dataset_label", "document",
                     "requirements", "evidence"}, "$")
    if payload["schema_version"] != "1.0":
        fail("$.schema_version", "must be '1.0'")
    text(payload["dataset_label"], "$.dataset_label", 200)
    fields(payload["document"], {"id", "title"}, "$.document")
    for key in ("id", "title"):
        text(payload["document"][key], "$.document." + key, 200)
    array(payload["requirements"], "$.requirements", 100, 1)
    array(payload["evidence"], "$.evidence", 1000)
    requirement_ids = set()
    for index, item in enumerate(payload["requirements"]):
        path = f"$.requirements[{index}]"
        fields(item, {"id", "description", "required_terms", "min_evidence"}, path)
        text(item["id"], path + ".id", 200)
        if item["id"] in requirement_ids:
            fail(path + ".id", "duplicate requirement ID")
        requirement_ids.add(item["id"])
        text(item["description"], path + ".description")
        array(item["required_terms"], path + ".required_terms", 50, 1)
        seen_terms = set()
        for term in item["required_terms"]:
            text(term, path + ".required_terms[]", 200)
            normalized = term.strip().casefold()
            if normalized in seen_terms:
                fail(path + ".required_terms", "duplicate normalized term")
            seen_terms.add(normalized)
        if type(item["min_evidence"]) is not int or not 1 <= item["min_evidence"] <= 1000:
            fail(path + ".min_evidence", "must be an integer from 1 to 1000")
    evidence_ids = set()
    for index, item in enumerate(payload["evidence"]):
        path = f"$.evidence[{index}]"
        fields(item, {"id", "requirement_id", "source", "locator", "text"}, path)
        for key in ("id", "requirement_id", "source", "locator", "text"):
            text(item[key], path + "." + key, 20000 if key == "text" else 200)
        if item["id"] in evidence_ids:
            fail(path + ".id", "duplicate evidence ID")
        evidence_ids.add(item["id"])
        if item["requirement_id"] not in requirement_ids:
            fail(path + ".requirement_id", "unknown requirement ID")
    return payload


def review(payload):
    validate(payload)
    grouped = {requirement["id"]: [] for requirement in payload["requirements"]}
    for item in payload["evidence"]:
        grouped[item["requirement_id"]].append(item)
    results = []
    gaps = []
    for requirement in payload["requirements"]:
        traces = []
        for evidence in grouped[requirement["id"]]:
            normalized = evidence["text"].casefold()
            missing = [
                term for term in requirement["required_terms"]
                if term.strip().casefold() not in normalized
            ]
            traces.append({
                "evidence_id": evidence["id"],
                "source": evidence["source"],
                "locator": evidence["locator"],
                "qualifies": not missing,
                "missing_terms": missing,
            })
        qualifying = sum(trace["qualifies"] for trace in traces)
        shortfall = max(0, requirement["min_evidence"] - qualifying)
        result = {
            "requirement_id": requirement["id"],
            "description": requirement["description"],
            "status": "gap" if shortfall else "lexically_supported",
            "required_terms": list(requirement["required_terms"]),
            "min_evidence": requirement["min_evidence"],
            "qualifying_evidence_count": qualifying,
            "evidence_traces": traces,
        }
        results.append(result)
        if shortfall:
            gaps.append({
                "gap_id": "gap:" + requirement["id"],
                "requirement_id": requirement["id"],
                "reason": "missing_evidence" if not traces else "insufficient_qualifying_evidence",
                "additional_qualifying_evidence_needed": shortfall,
                "reviewed_evidence_ids": [trace["evidence_id"] for trace in traces],
                "action": "Supply distinct evidence entries containing every required term; "
                          "have a human assess meaning, authenticity, and adequacy.",
            })
    return {
        "schema_version": "1.0",
        "status": "ok",
        "stage": "document_review",
        "dataset_label": payload["dataset_label"],
        "document": dict(payload["document"]),
        "disclaimer": DISCLAIMER,
        "summary": {
            "requirements": len(results),
            "lexically_supported": len(results) - len(gaps),
            "gaps": len(gaps),
        },
        "requirement_results": results,
        "gaps": gaps,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("$", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    fail("$", "nonfinite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            fail("$", "usage: python -B implementation.py INPUT.json")
        raw = Path(args[0]).read_bytes()
        if len(raw) > 5_000_000:
            fail("$", "input file exceeds 5,000,000 bytes")
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        result = review(payload)
    except (OSError, ValueError, RecursionError) as exc:
        result = {"schema_version": "1.0", "status": "error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
