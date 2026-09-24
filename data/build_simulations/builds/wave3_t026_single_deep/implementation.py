"""Offline, deterministic evidence synthesis. All example data is synthetic.

Run: python -B implementation.py example_input.json
Library: run(payload, summarizer=None). An optional offline callable receives a
deep copy of each question result and returns summary plus exact evidence IDs.
Its response is structurally validated, not semantically fact-checked.
"""

import copy
import json
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(type(value) is dict, f"{path}: expected object")
    require(set(value) == set(expected), f"{path}: expected keys {sorted(expected)}")


def text(value, path):
    require(type(value) is str and bool(value.strip()), f"{path}: expected nonempty text")
    require(len(value) <= 100000, f"{path}: text exceeds 100000 characters")


def array(value, path, limit=1000):
    require(type(value) is list, f"{path}: expected array")
    require(len(value) <= limit, f"{path}: exceeds {limit} entries")


def unique_id(value, seen, path):
    text(value, path)
    require(value == value.strip(), f"{path}: surrounding whitespace prohibited")
    require(value not in seen, f"{path}: duplicate ID {value}")
    seen.add(value)


def validate(payload):
    """The single input validation boundary, used by both API and CLI."""
    fields(payload, {"schema_version", "synthetic", "questions", "documents"}, "input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version: expected integer 1")
    require(payload["synthetic"] is True, "synthetic: must be true for this reference build")
    array(payload["questions"], "questions", 100)
    require(bool(payload["questions"]), "questions: at least one required")
    question_ids = set()
    for q in payload["questions"]:
        fields(q, {"id", "text"}, "question")
        unique_id(q["id"], question_ids, "question.id")
        text(q["text"], "question.text")
    array(payload["documents"], "documents", 100)
    document_ids, evidence_ids = set(), set()
    evidence_count = 0
    for doc in payload["documents"]:
        fields(doc, {"id", "title", "content", "evidence"}, "document")
        unique_id(doc["id"], document_ids, "document.id")
        text(doc["title"], "document.title")
        text(doc["content"], "document.content")
        array(doc["evidence"], "document.evidence")
        evidence_count += len(doc["evidence"])
        require(evidence_count <= 10000, "input: exceeds 10000 evidence entries")
        for item in doc["evidence"]:
            fields(item, {"id", "question_id", "stance", "excerpt"}, "evidence")
            unique_id(item["id"], evidence_ids, "evidence.id")
            text(item["question_id"], "evidence.question_id")
            require(item["question_id"] in question_ids, "evidence: unknown question_id")
            text(item["stance"], "evidence.stance")
            require(item["stance"] in {"supports", "contradicts", "uncertain"},
                    "evidence.stance: expected supports, contradicts or uncertain")
            text(item["excerpt"], "evidence.excerpt")
            require(item["excerpt"] in doc["content"],
                    "evidence.excerpt: must occur verbatim in document.content")
    return payload


def synthesize(payload, summarizer=None):
    validate(payload)
    require(summarizer is None or callable(summarizer), "summarizer: must be callable")
    results = []
    unresolved = []
    for question in payload["questions"]:
        evidence = [
            dict(item, document_id=doc["id"], document_title=doc["title"])
            for doc in payload["documents"]
            for item in doc["evidence"]
            if item["question_id"] == question["id"]
        ]
        evidence.sort(key=lambda item: (item["document_id"], item["id"]))
        groups = {
            stance: sorted({item["document_id"] for item in evidence if item["stance"] == stance})
            for stance in ("supports", "contradicts", "uncertain")
        }
        supporting, contradicting = groups["supports"], groups["contradicts"]
        if supporting and contradicting:
            finding = "disputed"
        elif supporting:
            finding = "support_only"
        elif contradicting:
            finding = "contradiction_only"
        elif evidence:
            finding = "uncertain_only"
        else:
            finding = "no_evidence"
        reasons = []
        if finding == "disputed":
            reasons.append("Opposing evidence needs reconciliation; document counts do not decide truth.")
        if finding == "no_evidence":
            reasons.append("No supplied evidence addresses this question.")
        if groups["uncertain"]:
            reasons.append("At least one source explicitly leaves this question uncertain.")
        source_count = len({item["document_id"] for item in evidence})
        if source_count == 1:
            reasons.append("Only one document addresses this question; seek independent corroboration.")
        disagreements = []
        if finding == "disputed":
            disagreements.append({
                "supporting_evidence_ids": [e["id"] for e in evidence if e["stance"] == "supports"],
                "contradicting_evidence_ids": [e["id"] for e in evidence if e["stance"] == "contradicts"],
                "internally_conflicted_document_ids": sorted(set(supporting) & set(contradicting)),
                "cross_document": any(a != b for a in supporting for b in contradicting),
            })
        result = {
            "question_id": question["id"],
            "question": question["text"],
            "finding": finding,
            "summary": (
                f"{finding}: {len(supporting)} supporting, {len(contradicting)} contradicting, "
                f"{len(groups['uncertain'])} uncertain documents. "
                "These are supplied annotations, not independently verified conclusions."
            ),
            "document_ids_by_stance": groups,
            "distinct_document_count": source_count,
            "evidence": evidence,
            "disagreements": disagreements,
            "unresolved_questions": [
                {"question": question["text"], "reason": reason} for reason in reasons
            ],
        }
        if summarizer is not None:
            response = summarizer(copy.deepcopy(result))
            fields(response, {"summary", "evidence_ids"}, "summarizer response")
            text(response["summary"], "summarizer.summary")
            array(response["evidence_ids"], "summarizer.evidence_ids", 10000)
            seen = set()
            for identifier in response["evidence_ids"]:
                unique_id(identifier, seen, "summarizer.evidence_id")
            require(seen == {e["id"] for e in evidence},
                    "summarizer: evidence_ids must cite all and only this question's evidence")
            result["optional_summary"] = dict(copy.deepcopy(response), verification="structure_only")
        results.append(result)
        unresolved.extend(dict(item, question_id=question["id"])
                          for item in result["unresolved_questions"])
    return {
        "status": "ok",
        "schema_version": 1,
        "synthetic": True,
        "question_results": results,
        "unresolved_questions": unresolved,
        "source_inventory": [
            {"document_id": doc["id"], "title": doc["title"], "evidence_count": len(doc["evidence"])}
            for doc in sorted(payload["documents"], key=lambda doc: doc["id"])
        ],
        "limitations": [
            "Evidence relevance and stance are supplied by the caller, not inferred.",
            "Verbatim excerpt validation establishes provenance, not truth or source independence.",
            "No live research, causal inference, confidence scoring, or majority-vote resolution.",
            "An empty unresolved list means no rule-triggered gaps, not a complete research answer.",
        ],
    }


def run(payload, summarizer=None):
    """Return the shared success/error envelope without mutating input."""
    try:
        return synthesize(payload, summarizer)
    except Exception as exc:
        return {"status": "error", "schema_version": 1,
                "error": {"type": type(exc).__name__, "message": str(exc)}}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate key {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"JSON: non-finite value {value} prohibited")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=reject_duplicate_keys,
                                parse_constant=reject_constant)
        output = run(payload)
    except Exception as exc:
        output = {"status": "error", "schema_version": 1,
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0 if output["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
