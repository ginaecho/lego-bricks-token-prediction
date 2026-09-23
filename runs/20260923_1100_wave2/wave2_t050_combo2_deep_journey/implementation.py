"""Synthetic, deterministic evidence-to-journey reference pipeline (stdlib only).

Run: python -B implementation.py example_input.json
Evidence counts represent document coverage, not scientific confidence or truth.
"""

import itertools
import json
import sys


class ValidationError(ValueError):
    """A shared input, stage-output, or handoff validation error."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(expected), f"{path}: expected fields {sorted(expected)}")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")


def records(value, expected, path):
    require(isinstance(value, list), f"{path}: expected array")
    ids = set()
    for index, record in enumerate(value):
        fields(record, expected, f"{path}[{index}]")
        text(record["id"], f"{path}[{index}].id")
        require(record["id"] not in ids, f"{path}: duplicate id {record['id']}")
        ids.add(record["id"])
    return ids


def references(value, allowed, path):
    require(isinstance(value, list), f"{path}: expected array")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), f"{path}: duplicate references")
    require(set(value) <= allowed, f"{path}: unknown reference")


def validate(value, stage, context=None):
    """One validation boundary used for input and every stage handoff.

    Output validation reconstructs deterministic contracts from validated context,
    rejecting missing, invented, reordered, or altered evidence and journey steps.
    """
    if stage == "input":
        fields(value, {"schema_version", "synthetic", "questions", "documents",
                       "actions", "user"}, "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "input: schema_version must be integer 1")
        require(value["synthetic"] is True, "input: this reference requires synthetic fixtures")
        questions = records(value["questions"], {"id", "text"}, "questions")
        require(bool(questions), "questions: at least one question required")
        for question in value["questions"]:
            text(question["text"], "question.text")
        records(value["documents"], {"id", "title", "claims"}, "documents")
        for document in value["documents"]:
            text(document["title"], "document.title")
            require(isinstance(document["claims"], list), "document.claims: expected array")
            seen = set()
            for claim in document["claims"]:
                fields(claim, {"question_id", "stance", "quote"}, "claim")
                text(claim["question_id"], "claim.question_id")
                require(claim["question_id"] in questions, "claim: unknown question")
                require(isinstance(claim["stance"], str) and
                        claim["stance"] in {"support", "oppose", "uncertain"},
                        "claim: invalid stance")
                text(claim["quote"], "claim.quote")
                signature = (claim["question_id"], claim["stance"], claim["quote"])
                require(signature not in seen, "document: duplicate claim")
                seen.add(signature)
        actions = records(value["actions"], {"id", "title", "kind", "question_ids",
                                             "prerequisites"}, "actions")
        graph = {}
        for action in value["actions"]:
            text(action["title"], "action.title")
            require(isinstance(action["kind"], str) and action["kind"] in {"research", "apply"},
                    "action: kind must be research or apply")
            references(action["question_ids"], questions, "action.question_ids")
            require(bool(action["question_ids"]), "action: at least one question required")
            references(action["prerequisites"], actions, "action.prerequisites")
            graph[action["id"]] = set(action["prerequisites"])
        remaining = dict(graph)
        while remaining:
            ready = {key for key, dependencies in remaining.items() if not dependencies}
            require(bool(ready), "actions: prerequisite cycle")
            remaining = {key: dependencies - ready for key, dependencies in remaining.items()
                         if key not in ready}
        fields(value["user"], {"completed_actions", "interests"}, "user")
        references(value["user"]["completed_actions"], actions, "user.completed_actions")
        references(value["user"]["interests"], questions, "user.interests")
        completed = set(value["user"]["completed_actions"])
        require(all(graph[action] <= completed for action in completed),
                "user: completed actions must include their prerequisites")
    elif stage == "research":
        validate(context, "input")
        fields(value, {"schema_version", "findings", "disagreements",
                       "unresolved_questions"}, "research")
        require(_same_json(value, _synthesize(context)),
                "research: evidence or synthesis does not match validated source documents")
    elif stage == "journey":
        require(isinstance(context, tuple) and len(context) == 2,
                "journey: expected input and research context")
        source, research = context
        validate(research, "research", source)
        fields(value, {"schema_version", "status", "steps", "next_actions",
                       "blocked_actions", "unresolved_questions", "reason"}, "journey")
        require(_same_json(value, _recommend(source, research)),
                "journey: invalid prerequisite order, evidence handoff, or recommendation")
    elif stage == "output":
        fields(value, {"schema_version", "status", "synthetic", "research", "journey"}, "output")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "output: invalid schema version")
        require(value["status"] == "ok" and value["synthetic"] is True,
                "output: invalid envelope")
        validate(value["journey"], "journey", (context, value["research"]))
    else:
        raise ValidationError(f"Unknown validation stage: {stage}")
    return value


def _same_json(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _synthesize(source):
    findings, disagreements, unresolved = [], [], []
    for question in sorted(source["questions"], key=lambda item: item["id"]):
        evidence = []
        for document in source["documents"]:
            for claim in document["claims"]:
                if claim["question_id"] == question["id"]:
                    evidence.append({"document_id": document["id"], "stance": claim["stance"],
                                     "quote": claim["quote"]})
        evidence.sort(key=lambda item: (item["document_id"], item["stance"], item["quote"]))
        supporting = sorted({item["document_id"] for item in evidence
                             if item["stance"] == "support"})
        opposing = sorted({item["document_id"] for item in evidence if item["stance"] == "oppose"})
        uncertain = any(item["stance"] == "uncertain" for item in evidence)
        if supporting and opposing:
            assessment = "contested"
        elif uncertain or not evidence:
            assessment = "unresolved"
        elif supporting:
            assessment = "supported"
        else:
            assessment = "opposed"
        finding = {
            "question_id": question["id"], "question": question["text"],
            "assessment": assessment, "evidence": evidence,
            "supporting_documents": supporting, "opposing_documents": opposing,
            "document_count": len({item["document_id"] for item in evidence}),
        }
        findings.append(finding)
        if assessment == "contested":
            disagreements.append({"question_id": question["id"],
                                  "supporting_documents": supporting,
                                  "opposing_documents": opposing})
        if assessment in {"contested", "unresolved"}:
            reason = ("conflicting evidence" if assessment == "contested" else
                      "uncertain evidence" if uncertain else "no evidence")
            unresolved.append({"question_id": question["id"], "question": question["text"],
                               "reason": reason})
    return {"schema_version": 1, "findings": findings, "disagreements": disagreements,
            "unresolved_questions": unresolved}


def deep_research(source):
    validate(source, "input")
    return validate(_synthesize(source), "research", source)


def _recommend(source, research):
    # Only validated findings supply evidence to personalization, never raw claims.
    findings = {item["question_id"]: item for item in research["findings"]}
    completed = set(source["user"]["completed_actions"])
    interests = set(source["user"]["interests"])
    actions = sorted(source["actions"], key=lambda item: item["id"])
    eligible, blocked = [], []
    for action in actions:
        if action["id"] in completed:
            continue
        unsafe = [qid for qid in action["question_ids"]
                  if action["kind"] == "apply" and findings[qid]["assessment"] != "supported"]
        missing = sorted(set(action["prerequisites"]) - completed)
        if not unsafe:
            eligible.append(action)
        if unsafe or missing:
            blocked.append({"action_id": action["id"], "missing_prerequisites": missing,
                            "unsupported_questions": sorted(unsafe)})

    def score(action):
        addressed = [findings[qid] for qid in action["question_ids"]]
        return (10 * len(interests.intersection(action["question_ids"])) +
                3 * sum(item["assessment"] in {"contested", "unresolved"} for item in addressed) +
                sum(item["document_count"] for item in addressed))

    def step(action, number, before):
        addressed = [findings[qid] for qid in sorted(action["question_ids"])]
        return {
            "step": number, "action_id": action["id"], "title": action["title"],
            "kind": action["kind"], "score": score(action),
            "prerequisites_satisfied": sorted(set(action["prerequisites"]).intersection(before)),
            "rationale": [{"question_id": item["question_id"], "assessment": item["assessment"],
                           "evidence": item["evidence"]} for item in addressed],
        }

    next_actions = sorted(
        [action for action in eligible if set(action["prerequisites"]) <= completed],
        key=lambda action: (-score(action), action["id"]))
    pairs = [(first, second) for first, second in itertools.permutations(eligible, 2)
             if set(first["prerequisites"]) <= completed and
             set(second["prerequisites"]) <= completed | {first["id"]}]
    # Maximize combined relevance; on ties prefer an actual dependency transition.
    pairs.sort(key=lambda pair: (-(score(pair[0]) + score(pair[1])),
                                -(pair[0]["id"] in pair[1]["prerequisites"]),
                                -score(pair[0]), pair[0]["id"], pair[1]["id"]))
    steps = []
    if pairs:
        first, second = pairs[0]
        steps = [step(first, 1, completed), step(second, 2, completed | {first["id"]})]
    return {
        "schema_version": 1, "status": "ready" if steps else "blocked", "steps": steps,
        "next_actions": [action["id"] for action in next_actions],
        "blocked_actions": blocked,
        "unresolved_questions": research["unresolved_questions"],
        "reason": ("Two distinct actions satisfy evidence and prerequisite constraints."
                   if steps else "No valid two-step journey; available next actions are listed."),
    }


def recommend_journey(source, research):
    validate(research, "research", source)
    return validate(_recommend(source, research), "journey", (source, research))


def run_pipeline(source):
    research = deep_research(source)
    journey = recommend_journey(source, research)
    return validate({"schema_version": 1, "status": "ok", "synthetic": True,
                     "research": research, "journey": journey}, "output", source)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate key {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError(f"JSON: non-finite constant {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            source = json.load(handle, object_pairs_hook=_unique_object,
                               parse_constant=_reject_constant)
        result = run_pipeline(source)
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
