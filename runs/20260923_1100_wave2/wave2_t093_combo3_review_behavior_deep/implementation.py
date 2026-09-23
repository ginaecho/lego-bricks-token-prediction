"""Synthetic, offline reference pipeline. Run: python -B implementation.py INPUT.json.

All stages share an envelope and the validate() boundary. Evidence checking verifies
quoted text and references, not truth, compliance, or certification. Behavioral
signals prioritize reading; they never change evidence polarity or settle disputes.
Only Python's standard library is used; there is no provider integration.
"""

import copy
import datetime as dt
import json
import math
import sys


BUILD_ID = "wave2_t093_combo3_review_behavior_deep"
STAGES = ("input", "review", "behavior", "deep")
HALF_LIFE_DAYS = 30.0
EVENT_WEIGHTS = {"browse": 1.0, "purchase": 3.0}


class ValidationError(ValueError):
    """Invalid input or an untrusted stage handoff."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonblank text")


def sequence(value, path):
    require(isinstance(value, list), path + " must be an array")


def timestamp(value, path):
    text(value, path)
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.tzinfo is not None, path + " must include a timezone")
        return result.astimezone(dt.timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(path + " must be a timezone-aware ISO 8601 timestamp within UTC range") from exc


def unique_id(value, seen, path):
    text(value, path)
    require(value not in seen, path + " must be unique")
    seen.add(value)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("payload must contain JSON-compatible finite values") from exc


def validate_data(data):
    fields(data, ("as_of", "requirements", "documents", "events"), "data")
    as_of = timestamp(data["as_of"], "as_of")
    for name in ("requirements", "documents", "events"):
        sequence(data[name], name)
    require(bool(data["requirements"]), "at least one requirement is required")
    requirement_ids = set()
    for requirement in data["requirements"]:
        fields(requirement, ("id", "question"), "requirement")
        unique_id(requirement["id"], requirement_ids, "requirement.id")
        text(requirement["question"], "requirement.question")
    document_ids, evidence_ids = set(), set()
    for document in data["documents"]:
        fields(document, ("id", "title", "text", "evidence"), "document")
        unique_id(document["id"], document_ids, "document.id")
        text(document["title"], "document.title")
        text(document["text"], "document.text")
        sequence(document["evidence"], "document.evidence")
        for evidence in document["evidence"]:
            fields(evidence, ("id", "requirement_id", "stance", "quote"), "evidence")
            unique_id(evidence["id"], evidence_ids, "evidence.id")
            text(evidence["requirement_id"], "evidence.requirement_id")
            require(evidence["requirement_id"] in requirement_ids,
                    "evidence references an unknown requirement")
            require(evidence["stance"] in ("supports", "contradicts"),
                    "evidence stance must be supports or contradicts")
            text(evidence["quote"], "evidence.quote")
            require(evidence["quote"] in document["text"],
                    "evidence quote must be an exact substring of its source document")
    event_ids = set()
    for event in data["events"]:
        fields(event, ("id", "document_id", "kind", "at"), "event")
        unique_id(event["id"], event_ids, "event.id")
        text(event["document_id"], "event.document_id")
        require(event["document_id"] in document_ids, "event references an unknown document")
        require(isinstance(event["kind"], str) and event["kind"] in EVENT_WEIGHTS,
                "event.kind must be browse or purchase")
        require(timestamp(event["at"], "event.at") <= as_of, "future events are not allowed")


def review_result(data):
    checked, gaps = [], []
    document_checks = []
    for document in sorted(data["documents"], key=lambda item: item["id"]):
        ids = []
        for evidence in sorted(document["evidence"], key=lambda item: item["id"]):
            checked.append({
                **evidence, "document_id": document["id"],
                "source_title": document["title"],
                "quote_start": document["text"].index(evidence["quote"]),
                "quote_end": document["text"].index(evidence["quote"]) + len(evidence["quote"]),
            })
            ids.append(evidence["id"])
        document_checks.append({"document_id": document["id"], "evidence_ids": ids})
    for requirement in sorted(data["requirements"], key=lambda item: item["id"]):
        evidence = [entry for entry in checked if entry["requirement_id"] == requirement["id"]]
        supporting = [entry["id"] for entry in evidence if entry["stance"] == "supports"]
        contradicting = [entry["id"] for entry in evidence if entry["stance"] == "contradicts"]
        if not supporting:
            gaps.append({
                "requirement_id": requirement["id"], "reason": "no_supporting_evidence",
                "evidence_ids": contradicting,
            })
        if contradicting:
            gaps.append({
                "requirement_id": requirement["id"], "reason": "contradictory_evidence",
                "evidence_ids": [entry["id"] for entry in evidence],
            })
    return {
        "document_checks": document_checks, "checked_evidence": checked, "gaps": gaps,
        "notice": "Evidence presence and exact quotation checks only; not certification or verification of truth.",
    }


def behavior_result(data, review):
    as_of = timestamp(data["as_of"], "as_of")
    ranking = []
    for checked in review["document_checks"]:
        document_id = checked["document_id"]
        evidence = [item for item in review["checked_evidence"]
                    if item["id"] in checked["evidence_ids"]]
        # Coverage is polarity-neutral: disagreement must remain discoverable.
        coverage = len({item["requirement_id"] for item in evidence})
        prior = coverage / len(data["requirements"])
        signals = []
        for event in sorted(data["events"], key=lambda item: item["id"]):
            if event["document_id"] != document_id:
                continue
            age = (as_of - timestamp(event["at"], "event.at")).total_seconds() / 86400.0
            contribution = EVENT_WEIGHTS[event["kind"]] * math.pow(0.5, age / HALF_LIFE_DAYS)
            signals.append({"event_id": event["id"], "kind": event["kind"],
                            "age_days": age, "contribution": contribution})
        activity = math.fsum(signal["contribution"] for signal in signals)
        ranking.append({
            "document_id": document_id, "score": prior + activity,
            "coverage_prior": prior, "activity_score": activity,
            "cold_start": not signals, "signals": signals,
            "evidence_ids": list(checked["evidence_ids"]),
        })
    ranking.sort(key=lambda item: (-item["score"], item["document_id"]))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    return {
        "half_life_days": HALF_LIFE_DAYS, "event_weights": dict(EVENT_WEIGHTS),
        "cold_start": not data["events"], "ranking": ranking,
        "review_gaps": copy.deepcopy(review["gaps"]),
        "notice": "Ranking prioritizes reading, not evidentiary truth; all reviewed documents are retained.",
    }


def deep_result(data, review, behavior):
    by_id = {item["id"]: item for item in review["checked_evidence"]}
    ordered_evidence = []
    for ranked in behavior["ranking"]:
        for evidence_id in ranked["evidence_ids"]:
            item = by_id[evidence_id]
            ordered_evidence.append({
                **item, "reading_rank": ranked["rank"],
                "personalization_score": ranked["score"],
            })
    findings, disagreements, unresolved = [], [], []
    for requirement in sorted(data["requirements"], key=lambda item: item["id"]):
        evidence = [item for item in ordered_evidence
                    if item["requirement_id"] == requirement["id"]]
        supporting = [item for item in evidence if item["stance"] == "supports"]
        contradicting = [item for item in evidence if item["stance"] == "contradicts"]
        if supporting and contradicting:
            conclusion = "conflicting_evidence"
        elif supporting:
            conclusion = "supporting_evidence_only"
        elif contradicting:
            conclusion = "contradicting_evidence_only"
        else:
            conclusion = "no_evidence"
        finding = {
            "requirement_id": requirement["id"], "question": requirement["question"],
            "conclusion": conclusion, "citations": evidence,
            "source_document_ids": list(dict.fromkeys(item["document_id"] for item in evidence)),
            "review_gaps": [copy.deepcopy(gap) for gap in behavior["review_gaps"]
                            if gap["requirement_id"] == requirement["id"]],
        }
        findings.append(finding)
        if supporting and contradicting:
            disagreements.append({
                "requirement_id": requirement["id"],
                "supporting_evidence_ids": [item["id"] for item in supporting],
                "contradicting_evidence_ids": [item["id"] for item in contradicting],
                "question": "What explains the conflicting sources for: " + requirement["question"],
            })
        if not supporting or contradicting:
            unresolved.append({
                "requirement_id": requirement["id"], "reason": conclusion,
                "question": requirement["question"],
                "evidence_ids": [item["id"] for item in evidence],
            })
    return {
        "reading_order": [item["document_id"] for item in behavior["ranking"]],
        "findings": findings, "disagreements": disagreements,
        "unresolved_questions": unresolved,
        "notice": "Synthetic, extractive synthesis; source agreement is not proof or certification.",
    }


def validate(payload, expected_stage=None):
    """Validate input and exact derived contracts, including all handoff references.

    Recomputing small deterministic stages is intentional: mutated outputs cannot
    inject a citation, hide a gap, alter ranking, or bypass prior validation.
    """
    fields(payload, ("schema_version", "synthetic", "status", "stage", "data",
                     "review", "behavior", "deep"), "envelope")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version must be integer 1")
    require(payload["synthetic"] is True, "this reference accepts labeled synthetic data only")
    require(payload["status"] == "ok", "pipeline envelope status must be ok")
    require(payload["stage"] in STAGES, "unknown pipeline stage")
    require(expected_stage is None or payload["stage"] == expected_stage,
            "stage must be " + str(expected_stage))
    canonical(payload)
    validate_data(payload["data"])
    index = STAGES.index(payload["stage"])
    expected = {}
    if index >= 1:
        expected["review"] = review_result(payload["data"])
    if index >= 2:
        expected["behavior"] = behavior_result(payload["data"], expected["review"])
    if index >= 3:
        expected["deep"] = deep_result(payload["data"], expected["review"], expected["behavior"])
    for name in STAGES[1:]:
        require(canonical(payload[name]) == canonical(expected.get(name)),
                name + " output violates the validated stage contract")
    return payload


def advance(payload, source, target, derive):
    validate(payload, source)
    output = copy.deepcopy(payload)
    output[target] = derive(output)
    output["stage"] = target
    return validate(output, target)


def document_review(payload):
    return advance(payload, "input", "review", lambda item: review_result(item["data"]))


def personalize(payload):
    return advance(payload, "review", "behavior",
                   lambda item: behavior_result(item["data"], item["review"]))


def deep_research(payload):
    return advance(payload, "behavior", "deep",
                   lambda item: deep_result(item["data"], item["review"], item["behavior"]))


def run_pipeline(payload):
    return deep_research(personalize(document_review(payload)))


def reject_constant(value):
    raise ValidationError("nonfinite JSON number is not allowed: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        output = run_pipeline(payload)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": {
            "type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
