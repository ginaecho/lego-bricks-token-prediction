"""Synthetic, deterministic review -> FAQ reference pipeline (Python standard library).

Run: python -B implementation.py example_input.json
Review checks literal, case-insensitive phrases in evidence explicitly linked to
each requirement. A match is evidence coverage, never a certification decision.
FAQ retrieves only validated, supported review evidence; it never invents answers.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def array(value, path):
    require(isinstance(value, list), path + " must be an array")


def unique(values, path):
    require(len(values) == len(set(values)), path + " contains duplicate values")


def rows(value, fields, path):
    array(value, path)
    for i, row in enumerate(value):
        obj(row, fields, f"{path}[{i}]")
        text(row["id"], path + ".id")
    unique([row["id"] for row in value], path + ".id")


def validate_input(data):
    obj(data, ["schema_version", "synthetic", "requirements", "evidence", "questions"], "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    require(data["synthetic"] is True, "this reference accepts labeled synthetic fixtures only")
    rows(data["evidence"], ["id", "source", "locator", "text"], "evidence")
    for item in data["evidence"]:
        for key in ("source", "locator", "text"):
            text(item[key], "evidence." + key)
    evidence_ids = {item["id"] for item in data["evidence"]}
    rows(data["requirements"], ["id", "description", "required_phrases", "evidence_ids"], "requirements")
    for item in data["requirements"]:
        text(item["description"], "requirement.description")
        for key in ("required_phrases", "evidence_ids"):
            array(item[key], "requirement." + key)
            for entry in item[key]:
                text(entry, "requirement." + key)
            unique(item[key], "requirement." + key)
        require(bool(item["required_phrases"]), "required_phrases must not be empty")
        unique([p.casefold() for p in item["required_phrases"]], "normalized required_phrases")
        require(set(item["evidence_ids"]) <= evidence_ids, "unknown requirement evidence reference")
    rows(data["questions"], ["id", "text", "requirement_ids"], "questions")
    requirement_ids = {item["id"] for item in data["requirements"]}
    for item in data["questions"]:
        text(item["text"], "question.text")
        array(item["requirement_ids"], "question.requirement_ids")
        for reference in item["requirement_ids"]:
            text(reference, "question.requirement_ids")
        unique(item["requirement_ids"], "question.requirement_ids")
        require(bool(item["requirement_ids"]), "each question must explicitly scope at least one requirement")
        require(set(item["requirement_ids"]) <= requirement_ids, "unknown question requirement reference")
    return data


def contains_phrase(phrase, content):
    return re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)",
                     content.casefold()) is not None


def _review(data):
    evidence = {item["id"]: item for item in data["evidence"]}
    findings = []
    for requirement in data["requirements"]:
        citations = []
        missing = []
        for phrase in requirement["required_phrases"]:
            matches = [evidence[eid] for eid in requirement["evidence_ids"]
                       if contains_phrase(phrase, evidence[eid]["text"])]
            if not matches:
                missing.append(phrase)
            for match in matches:
                citations.append({
                    "phrase": phrase, "evidence_id": match["id"],
                    "source": match["source"], "locator": match["locator"],
                    "quote": match["text"],
                })
        findings.append({
            "requirement_id": requirement["id"],
            "status": "gap" if missing else "supported",
            "missing_phrases": missing,
            "citations": citations,
        })
    return {
        "schema_version": 1,
        "notice": "Literal evidence coverage only; not certification or a compliance determination.",
        "findings": findings,
    }


def validate_review(data, review):
    validate_input(data)
    obj(review, ["schema_version", "notice", "findings"], "review")
    require(type(review["schema_version"]) is int, "review.schema_version must be integer")
    # Canonical reconstruction validates both schema and every provenance link.
    require(review == _review(data), "review failed schema, coverage, or provenance validation")
    return review


def review_documents(data):
    validate_input(data)
    return validate_review(data, _review(data))


STOP_WORDS = {"a", "an", "the", "is", "are", "what", "how", "do", "does", "can",
              "i", "we", "you", "to", "of", "for", "and", "in", "on", "it", "our"}


def tokens(value):
    return set(re.findall(r"\w+", value.casefold())) - STOP_WORDS


def _faq(data, review):
    findings = {row["requirement_id"]: row for row in review["findings"]}
    answers = []
    for question in data["questions"]:
        selected = [findings[rid] for rid in question["requirement_ids"]]
        gaps = [row["requirement_id"] for row in selected if row["status"] == "gap"]
        candidates = {}
        query = tokens(question["text"])
        for finding in selected:
            if finding["status"] != "supported":
                continue
            for citation in finding["citations"]:
                eid = citation["evidence_id"]
                score = len(query & tokens(citation["quote"]))
                if score:
                    candidates[eid] = (score, {
                        "evidence_id": eid, "source": citation["source"],
                        "locator": citation["locator"], "quote": citation["quote"],
                    })
        ranked = sorted(candidates.values(), key=lambda item: (-item[0], item[1]["evidence_id"]))
        citations = [item[1] for item in ranked]
        if gaps:
            reason = "review_gap"
        elif not citations:
            reason = "no_relevant_evidence"
        else:
            reason = None
        answers.append({
            "question_id": question["id"],
            "requirement_ids": list(question["requirement_ids"]),
            "status": "abstained" if reason else "answered",
            "reason": reason,
            "gap_requirement_ids": gaps,
            "answer": None if reason else "\n".join(item["quote"] for item in citations),
            "citations": [] if reason else citations,
        })
    return {"schema_version": 1, "answers": answers}


def validate_faq(data, review, faq):
    validate_review(data, review)
    obj(faq, ["schema_version", "answers"], "faq")
    require(type(faq["schema_version"]) is int, "faq.schema_version must be integer")
    require(faq == _faq(data, review), "FAQ failed schema, grounding, or gap propagation validation")
    return faq


def answer_questions(data, review):
    validate_review(data, review)
    return validate_faq(data, review, _faq(data, review))


def run_pipeline(data):
    validate_input(data)
    review = review_documents(data)
    faq = answer_questions(data, review)
    result = {"schema_version": 1, "synthetic": True, "status": "ok",
              "review": review, "faq": faq}
    validate_output(data, result)
    return result


def validate_output(data, result):
    obj(result, ["schema_version", "synthetic", "status", "review", "faq"], "output")
    require(type(result["schema_version"]) is int and result["schema_version"] == 1,
            "output schema_version must be integer 1")
    require(result["synthetic"] is True and result["status"] == "ok", "invalid output metadata")
    validate_faq(data, result["review"], result["faq"])
    return result


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON value: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py input.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
