"""Synthetic, deterministic review -> research -> journey reference CLI.

Usage: python -B implementation.py example_input.json
No provider, certification decision, persistence, or external dependencies.
Document IDs count as sources; source independence is not inferred.
"""

import json
import sys
from pathlib import Path


VERSION = "1.0"
STANCES = {"supports", "contradicts", "unknown"}


class ValidationError(ValueError):
    """Invalid input or invalid stage handoff."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(expected.split()), path + " has missing or unknown fields")


def text(value, path):
    require(type(value) is str and bool(value.strip()), path + " must be nonempty text")


def array(value, path):
    require(type(value) is list, path + " must be an array")


def identifiers(items, path):
    array(items, path)
    for item in items:
        text(item, path)
    require(len(set(items)) == len(items), path + " must not contain duplicates")


def unique_objects(items, expected, path):
    array(items, path)
    ids = set()
    for item in items:
        fields(item, expected, path)
        text(item["id"], path + ".id")
        require(item["id"] not in ids, path + " contains duplicate id")
        ids.add(item["id"])
    return ids


def validate_input(value):
    fields(value, "schema_version synthetic requirements documents actions completed_actions", "input")
    require(value["schema_version"] == VERSION, "unsupported schema_version")
    require(value["synthetic"] is True, "fixtures must explicitly be synthetic")
    requirements = value["requirements"]
    rids = unique_objects(requirements, "id text min_sources", "requirements")
    require(bool(requirements), "at least one requirement is required")
    for row in requirements:
        text(row["text"], "requirement.text")
        require(type(row["min_sources"]) is int and row["min_sources"] >= 1,
                "min_sources must be a positive integer")
    unique_objects(value["documents"], "id title content evidence", "documents")
    evidence_ids = set()
    for document in value["documents"]:
        text(document["title"], "document.title")
        text(document["content"], "document.content")
        local_ids = unique_objects(document["evidence"],
                                   "id requirement_id stance quote", "evidence")
        require(not evidence_ids.intersection(local_ids), "evidence ids must be globally unique")
        evidence_ids.update(local_ids)
        for evidence in document["evidence"]:
            text(evidence["requirement_id"], "evidence.requirement_id")
            text(evidence["stance"], "evidence.stance")
            require(evidence["requirement_id"] in rids, "unknown evidence requirement")
            require(evidence["stance"] in STANCES, "invalid evidence stance")
            text(evidence["quote"], "evidence.quote")
            require(evidence["quote"] in document["content"],
                    "evidence quote must occur verbatim in document content")
    aids = unique_objects(value["actions"], "id title requirement_ids prerequisites", "actions")
    for action in value["actions"]:
        text(action["title"], "action.title")
        identifiers(action["requirement_ids"], "action.requirement_ids")
        identifiers(action["prerequisites"], "action.prerequisites")
        require(set(action["requirement_ids"]) <= rids, "unknown action requirement")
        require(set(action["prerequisites"]) <= aids, "unknown action prerequisite")
    identifiers(value["completed_actions"], "completed_actions")
    completed = set(value["completed_actions"])
    require(completed <= aids, "unknown completed action")
    # Iterative topological check avoids recursion limits on prerequisite chains.
    pending = {a["id"]: set(a["prerequisites"]) for a in value["actions"]}
    visited = set()
    while pending:
        ready = {aid for aid, deps in pending.items() if deps <= visited}
        require(bool(ready), "action prerequisite cycle")
        visited.update(ready)
        for aid in ready:
            del pending[aid]
    for action in value["actions"]:
        if action["id"] in completed:
            require(set(action["prerequisites"]) <= completed,
                    "completed actions must include their prerequisites")


def envelope(stage, **data):
    return dict(schema_version=VERSION, synthetic=True, stage=stage, **data)


def compute_review(data):
    rows = []
    gaps = []
    for requirement in sorted(data["requirements"], key=lambda r: r["id"]):
        evidence = [
            dict(id=e["id"], document_id=d["id"], stance=e["stance"], quote=e["quote"])
            for d in data["documents"] for e in d["evidence"]
            if e["requirement_id"] == requirement["id"]
        ]
        evidence.sort(key=lambda e: e["id"])
        support = sorted({e["document_id"] for e in evidence if e["stance"] == "supports"})
        contrary = sorted({e["document_id"] for e in evidence if e["stance"] == "contradicts"})
        if support and contrary:
            state, reason = "conflict", "Supporting and contradicting evidence coexist."
        elif contrary:
            state, reason = "contradicted", "Contradicting evidence has no supporting evidence."
        elif len(support) >= requirement["min_sources"]:
            state, reason = "satisfied", "Configured supporting-document threshold met."
        elif support:
            state, reason = "insufficient", "Supporting-document threshold is not met."
        elif evidence:
            state, reason = "unknown", "Only non-conclusive evidence is available."
        else:
            state, reason = "missing", "No evidence is available."
        row = dict(requirement_id=requirement["id"], text=requirement["text"],
                   min_sources=requirement["min_sources"], status=state,
                   reason=reason, supporting_documents=support,
                   contradicting_documents=contrary, evidence=evidence)
        rows.append(row)
        if state != "satisfied":
            gaps.append(dict(requirement_id=requirement["id"], reason=reason,
                             evidence_ids=[e["id"] for e in evidence]))
    return envelope("review", requirements=rows, gaps=gaps,
                    notice="Evidence checking only; not certification or a compliance determination.")


def compute_deep(review):
    findings, disagreements, questions = [], [], []
    mapping = dict(satisfied="supported", conflict="contested", contradicted="adverse",
                   missing="incomplete", insufficient="incomplete", unknown="incomplete")
    all_sources = set()
    for row in review["requirements"]:
        sources = sorted({e["document_id"] for e in row["evidence"]})
        all_sources.update(sources)
        citations = [dict(e) for e in row["evidence"]]
        findings.append(dict(
            requirement_id=row["requirement_id"], review_status=row["status"],
            conclusion=mapping[row["status"]],
            synthesis=f'{row["text"]}: {row["reason"]}',
            source_ids=sources, citations=citations))
        if row["status"] == "conflict":
            disagreements.append(dict(
                requirement_id=row["requirement_id"],
                supporting_evidence_ids=[e["id"] for e in citations if e["stance"] == "supports"],
                contradicting_evidence_ids=[e["id"] for e in citations if e["stance"] == "contradicts"]))
        if row["status"] != "satisfied":
            questions.append(dict(
                id="q:" + row["requirement_id"], requirement_id=row["requirement_id"],
                question="What additional or reconciled evidence resolves: " + row["text"] + "?",
                reason=row["reason"], evidence_ids=[e["id"] for e in citations]))
    return envelope("deep", findings=findings, disagreements=disagreements,
                    unresolved_questions=questions, source_count=len(all_sources),
                    limitation="Document IDs are source proxies, not proof of independent corroboration.")


def compute_journey(deep, data):
    weights = dict(satisfied=1, conflict=5, contradicted=5, insufficient=4, unknown=3, missing=4)
    findings = {f["requirement_id"]: f for f in deep["findings"]}
    questions = {q["requirement_id"]: q["id"] for q in deep["unresolved_questions"]}
    completed = set(data["completed_actions"])
    actions = {a["id"]: a for a in data["actions"] if a["id"] not in completed}

    def score(action):
        return sum(weights[findings[r]["review_status"]] for r in action["requirement_ids"])

    def annotate(action, done):
        rids = sorted(action["requirement_ids"])
        return dict(
            action_id=action["id"], title=action["title"], score=score(action),
            requirement_ids=rids,
            question_ids=[questions[r] for r in rids if r in questions],
            evidence_ids=sorted({e["id"] for r in rids for e in findings[r]["citations"]}),
            prerequisites=sorted(action["prerequisites"]),
            prerequisites_satisfied=all(p in done for p in action["prerequisites"]),
            rationale=("Prioritize unresolved evidence; scores use validated research statuses."
                       if any(r in questions for r in rids)
                       else "Maintenance or prerequisite-enabling action."))

    ready = sorted((a for a in actions.values() if set(a["prerequisites"]) <= completed),
                   key=lambda a: (-score(a), a["id"]))
    pairs = []
    for first in ready:
        after_first = completed | {first["id"]}
        for second in actions.values():
            if second["id"] != first["id"] and set(second["prerequisites"]) <= after_first:
                pairs.append((first, second))
    pairs.sort(key=lambda pair: (
        -(score(pair[0]) + score(pair[1])), -score(pair[0]), pair[0]["id"], pair[1]["id"]))
    if pairs:
        first, second = pairs[0]
        steps = [annotate(first, completed), annotate(second, completed | {first["id"]})]
        state, reason = "ready", "Two distinct steps validated in order; step two assumes step one completes."
    else:
        steps = []
        state, reason = "blocked", "Fewer than two remaining actions can form an executable sequence."
    return envelope("journey", status=state, reason=reason,
                    next_actions=[annotate(a, completed) for a in ready],
                    two_step_journey=steps,
                    unaddressed_requirement_ids=sorted(
                        set(questions) - {r for a in actions.values() for r in a["requirement_ids"]}))


def validate(kind, value, context=None):
    """Single shared boundary validator; deterministic outputs are checked exactly.

    Exact comparison also rejects missing/extra fields, fabricated citations,
    altered statuses, unsupported conclusions, and invalid journey ordering.
    """
    if kind == "input":
        validate_input(value)
        return value
    if kind == "review":
        expected = compute_review(context)
    elif kind == "deep":
        expected = compute_deep(context)
    elif kind == "journey":
        expected = compute_journey(context["deep"], context["input"])
    else:
        raise ValidationError("unknown schema kind: " + str(kind))
    try:
        actual_json = json.dumps(value, sort_keys=True, allow_nan=False)
        expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("non-JSON stage output") from exc
    require(actual_json == expected_json, "invalid " + kind + " handoff")
    return value


def review_stage(data):
    validate("input", data)
    return validate("review", compute_review(data), data)


def deep_stage(review, data):
    validate("input", data)
    validate("review", review, data)
    return validate("deep", compute_deep(review), review)


def journey_stage(deep, review, data):
    validate("input", data)
    validate("review", review, data)
    validate("deep", deep, review)
    return validate("journey", compute_journey(deep, data), {"deep": deep, "input": data})


def run_pipeline(data):
    review = review_stage(data)
    deep = deep_stage(review, data)
    journey = journey_stage(deep, review, data)
    return dict(schema_version=VERSION, synthetic=True, status="ok",
                review=review, deep=deep, journey=journey)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=reject_duplicate_keys,
                             parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps(dict(status="error", error=str(exc)), ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
