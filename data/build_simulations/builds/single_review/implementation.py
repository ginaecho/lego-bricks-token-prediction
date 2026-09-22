"""Deterministic document-evidence triage; Python standard library only.

Input: requirements (id, description, priority 1..5, evidence_rules) and
documents (id, text, optional title). Lower priority numbers rank first.
Rules use case-sensitive literal strings: all_of must all match, any_of
must have at least one match if present, and contradicts flags conflicts.
At least one positive rule is required. No semantic or legal inference occurs.
Citation offsets are zero-based, end-exclusive Python Unicode character indices.

analyze(data, reviewer=None) optionally calls reviewer with an isolated copy of
the requirements, documents, and findings. It must return the complete findings
list. Reordering is allowed; changing IDs, statuses, or evidence is not. This is
a validation hook, not permission to invent evidence or override rule outcomes.
"""

import copy
import json
import sys
from collections import Counter


STATUSES = ("supported", "partial", "missing", "conflicting")
RULES = ("all_of", "any_of", "contradicts")
SCOPE = (
    "Document-evidence triage only, not legal advice or compliance certification. "
    "Absent evidence is not proof of failure. Supported means only that declared "
    "literal evidence rules matched supplied documents; authenticity, adequacy, "
    "applicability, and real-world compliance are not established."
)


class ValidationError(ValueError):
    """Invalid input or invalid reviewer response."""


def _object(value, allowed, required, location):
    if not isinstance(value, dict):
        raise ValidationError(f"{location} must be an object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise ValidationError(f"{location} has unknown or missing fields")


def _text(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{location} must be a nonblank string")


def _validate(data):
    _object(data, ("label", "requirements", "documents"),
            ("requirements", "documents"), "input")
    if "label" in data:
        _text(data["label"], "label")
    for kind in ("requirements", "documents"):
        if not isinstance(data[kind], list):
            raise ValidationError(f"{kind} must be a list")
        seen = set()
        for item in data[kind]:
            required = (("id", "description", "priority", "evidence_rules")
                        if kind == "requirements" else ("id", "text"))
            allowed = required if kind == "requirements" else (*required, "title")
            _object(item, allowed, required, kind)
            _text(item["id"], f"{kind}.id")
            if item["id"] in seen:
                raise ValidationError(f"duplicate {kind} id: {item['id']}")
            seen.add(item["id"])
            if kind == "documents":
                if not isinstance(item["text"], str):
                    raise ValidationError("document text must be a string")
                if "title" in item:
                    _text(item["title"], "document title")
                continue
            _text(item["description"], "requirement description")
            if type(item["priority"]) is not int or not 1 <= item["priority"] <= 5:
                raise ValidationError("priority must be an integer from 1 to 5")
            rules = item["evidence_rules"]
            _object(rules, RULES, (), "evidence_rules")
            positive = set()
            for name, terms in rules.items():
                if not isinstance(terms, list):
                    raise ValidationError(f"{name} must be a list")
                for term in terms:
                    _text(term, f"{name} term")
                if len(set(terms)) != len(terms):
                    raise ValidationError(f"duplicate terms in {name}")
                if name != "contradicts":
                    positive.update(terms)
            if not positive:
                raise ValidationError("at least one all_of or any_of term is required")
            if positive.intersection(rules.get("contradicts", [])):
                raise ValidationError("a term cannot both support and contradict")
            if set(rules.get("all_of", [])).intersection(rules.get("any_of", [])):
                raise ValidationError("duplicate positive terms across rule groups")


def _evaluate(requirement, documents):
    rules = requirement["evidence_rules"]
    citations = []
    hits = {name: set() for name in RULES}
    for name in RULES:
        for term in rules.get(name, []):
            for document in documents:
                start = document["text"].find(term)
                while start >= 0:
                    citations.append({
                        "document_id": document["id"],
                        "start": start,
                        "end": start + len(term),
                        "quote": document["text"][start:start + len(term)],
                        "rule": name,
                        "term": term,
                    })
                    hits[name].add(term)
                    start = document["text"].find(term, start + 1)
    all_met = len(hits["all_of"]) == len(rules.get("all_of", []))
    any_met = not rules.get("any_of") or bool(hits["any_of"])
    if hits["contradicts"]:
        status = "conflicting"
    elif all_met and any_met:
        status = "supported"
    elif hits["all_of"] or hits["any_of"]:
        status = "partial"
    else:
        status = "missing"
    return {
        "requirement_id": requirement["id"],
        "status": status,
        "citations": citations,
    }


def _citation_key(citation):
    return tuple(citation[field] for field in
                 ("document_id", "start", "end", "quote", "rule", "term"))


def _validate_review(candidate, expected, documents):
    if not isinstance(candidate, list):
        raise ValidationError("reviewer must return a findings list")
    expected_by_id = {item["requirement_id"]: item for item in expected}
    documents_by_id = {item["id"]: item["text"] for item in documents}
    seen = set()
    for finding in candidate:
        _object(finding, ("requirement_id", "status", "citations"),
                ("requirement_id", "status", "citations"), "review finding")
        rid = finding["requirement_id"]
        if not isinstance(rid, str) or rid not in expected_by_id or rid in seen:
            raise ValidationError("reviewer returned unknown or duplicate requirement ID")
        seen.add(rid)
        if finding["status"] not in STATUSES:
            raise ValidationError("reviewer returned invalid finding status")
        if not isinstance(finding["citations"], list):
            raise ValidationError("reviewer citations must be a list")
        for citation in finding["citations"]:
            fields = ("document_id", "start", "end", "quote", "rule", "term")
            _object(citation, fields, fields, "review citation")
            doc_id = citation["document_id"]
            if not isinstance(doc_id, str) or doc_id not in documents_by_id:
                raise ValidationError("reviewer citation has unknown document ID")
            start, end = citation["start"], citation["end"]
            text = documents_by_id[doc_id]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text)
                    or not isinstance(citation["quote"], str)
                    or text[start:end] != citation["quote"]):
                raise ValidationError("reviewer citation has invalid span or quote")
            if (citation["rule"] not in RULES
                    or not isinstance(citation["term"], str)
                    or citation["term"] != citation["quote"]):
                raise ValidationError("reviewer citation has invalid evidence rule")
        original = expected_by_id[rid]
        if finding["status"] != original["status"]:
            raise ValidationError("reviewer status disagrees with declared evidence rules")
        if (Counter(map(_citation_key, finding["citations"]))
                != Counter(map(_citation_key, original["citations"]))):
            raise ValidationError("reviewer changed, omitted, or duplicated evidence")
    if seen != set(expected_by_id):
        raise ValidationError("reviewer omitted requirements")


def analyze(data, reviewer=None):
    """Return a reproducible report; raise ValidationError on invalid data."""
    data = copy.deepcopy(data)
    _validate(data)
    findings = [_evaluate(req, data["documents"]) for req in data["requirements"]]
    reviewer_used = reviewer is not None
    if reviewer_used:
        if not callable(reviewer):
            raise ValidationError("reviewer must be callable")
        bundle = copy.deepcopy({
            "requirements": data["requirements"],
            "documents": data["documents"],
            "findings": findings,
        })
        try:
            candidate = reviewer(bundle)
        except Exception as exc:
            raise ValidationError("reviewer callback failed") from exc
        _validate_review(candidate, findings, data["documents"])
    gaps = []
    for requirement, finding in zip(data["requirements"], findings):
        if finding["status"] == "supported":
            continue
        rules = requirement["evidence_rules"]
        matched = {(c["rule"], c["term"]) for c in finding["citations"]}
        missing_all = [term for term in rules.get("all_of", [])
                       if ("all_of", term) not in matched]
        missing_any = (list(rules.get("any_of", []))
                       if not any(name == "any_of" for name, _ in matched) else [])
        conflicting = finding["status"] == "conflicting"
        gaps.append({
            "gap_id": f"gap:{requirement['id']}",
            "requirement_id": requirement["id"],
            "description": requirement["description"],
            "priority": requirement["priority"],
            "finding_status": finding["status"],
            "action": ("Resolve cited contradictory evidence with an accountable reviewer."
                       if conflicting else
                       "Request relevant evidence; current documents do not establish "
                       "all declared evidence rules. This is not proof of failure."),
            "unmatched_all_of": missing_all,
            "unmatched_any_of_options": missing_any,
            "citations": copy.deepcopy(finding["citations"]),
        })
    rank = {"conflicting": 0, "missing": 1, "partial": 2}
    gaps.sort(key=lambda gap: (gap["priority"], rank[gap["finding_status"]],
                              gap["requirement_id"]))
    counts = {status: sum(f["status"] == status for f in findings) for status in STATUSES}
    return {
        "label": data.get("label", ""),
        "scope": SCOPE,
        "citation_convention": "Zero-based, end-exclusive Unicode character offsets.",
        "findings": findings,
        "gaps": gaps,
        "summary": {
            "requirements_reviewed": len(findings),
            "documents_supplied": len(data["documents"]),
            "counts": counts,
            "prioritized_gap_ids": [gap["gap_id"] for gap in gaps],
            "human_review_recommended": bool(gaps),
            "injected_reviewer_validated": reviewer_used,
            "note": ("No requirements were declared; no assessment was performed."
                     if not findings else
                     "Resolve prioritized evidence gaps; no certification is issued."),
        },
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=_unique_object,
                             parse_constant=_invalid_constant)
        report = analyze(data)
    except (ValueError, OSError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
