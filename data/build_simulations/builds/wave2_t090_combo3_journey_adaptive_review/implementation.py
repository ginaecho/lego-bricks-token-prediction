"""Synthetic, deterministic journey -> onboarding -> document-review reference CLI."""

import copy
import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


STAGES = ("input", "journey", "adaptive", "review")
BASE_FIELDS = {"schema_version", "fixture_label", "profile", "actions",
               "requirements", "documents"}
DISCLAIMER = (
    "Synthetic reference review: lexical evidence checks only; "
    "not certification, legal advice, or verification of document authenticity."
)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(keys), f"{path}: unexpected or missing fields")


def text(value, path, maximum=200):
    require(isinstance(value, str) and bool(value.strip())
            and len(value) <= maximum, f"{path}: expected nonempty bounded string")


def items(value, path, maximum=100):
    require(isinstance(value, list) and len(value) <= maximum,
            f"{path}: expected list of at most {maximum} items")


def strings(value, path, allowed=None, nonempty=False):
    items(value, path)
    for entry in value:
        text(entry, path)
    require(len(set(value)) == len(value), f"{path}: duplicate entries")
    require(not nonempty or bool(value), f"{path}: must not be empty")
    if allowed is not None:
        require(set(value) <= set(allowed), f"{path}: unknown reference or choice")


def indexed(records, keys, path):
    items(records, path)
    result = {}
    for record in records:
        obj(record, keys, path)
        text(record["id"], f"{path}.id")
        require(record["id"] not in result, f"{path}: duplicate id")
        result[record["id"]] = record
    return result


def _validate_input(data, through):
    extra = set(STAGES[1:STAGES.index(through) + 1])
    if through != "input":
        extra.add("status")
    obj(data, BASE_FIELDS | extra, "envelope")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version: expected integer 1")
    text(data["fixture_label"], "fixture_label")
    require(data["fixture_label"].startswith("SYNTHETIC:"),
            "fixture_label: must start with SYNTHETIC:")
    profile = data["profile"]
    obj(profile, {"experience", "interests", "preferred_formats", "completed_actions"},
        "profile")
    require(profile["experience"] in ("beginner", "experienced"),
            "profile.experience: unsupported value")
    strings(profile["interests"], "profile.interests")
    strings(profile["preferred_formats"], "profile.preferred_formats",
            {"text", "video", "exercise"}, nonempty=True)
    actions = indexed(data["actions"],
                      {"id", "title", "topic", "prerequisites", "requirement_ids"},
                      "actions")
    requirements = indexed(data["requirements"],
                           {"id", "description", "required_terms"}, "requirements")
    documents = indexed(data["documents"],
                        {"id", "locator", "requirement_ids", "text"}, "documents")
    for action in actions.values():
        text(action["title"], "action.title")
        text(action["topic"], "action.topic")
        strings(action["prerequisites"], "action.prerequisites", actions)
        strings(action["requirement_ids"], "action.requirement_ids", requirements)
    for requirement in requirements.values():
        text(requirement["description"], "requirement.description", 2000)
        terms = requirement["required_terms"]
        strings(terms, "requirement.required_terms", nonempty=True)
        require(all(re.fullmatch(r"\w+", term) for term in terms),
                "required_terms: each term must be one word")
        require(len({term.casefold() for term in terms}) == len(terms),
                "required_terms: case-insensitive duplicates")
    for document in documents.values():
        text(document["locator"], "document.locator", 1000)
        text(document["text"], "document.text", 100000)
        strings(document["requirement_ids"], "document.requirement_ids", requirements)

    active, visited = set(), set()

    def visit(action_id):
        require(action_id not in active, "actions: prerequisite cycle")
        if action_id in visited:
            return
        active.add(action_id)
        for prerequisite in actions[action_id]["prerequisites"]:
            visit(prerequisite)
        active.remove(action_id)
        visited.add(action_id)

    for action_id in actions:
        visit(action_id)
    strings(profile["completed_actions"], "profile.completed_actions", actions)
    completed = set(profile["completed_actions"])
    require(all(set(actions[action_id]["prerequisites"]) <= completed
                for action_id in completed),
            "completed_actions: missing completed prerequisite")


def _journey(data):
    completed = set(data["profile"]["completed_actions"])
    interests = set(data["profile"]["interests"])
    remaining = [a for a in data["actions"] if a["id"] not in completed]

    def rank(action):
        return (action["topic"] not in interests, action["id"])

    available = sorted(
        [a for a in remaining if set(a["prerequisites"]) <= completed], key=rank)
    pairs = []
    for first in available:
        for second in remaining:
            if first["id"] != second["id"] and set(second["prerequisites"]) <= (
                    completed | {first["id"]}):
                pairs.append((first, second))
    pairs.sort(key=lambda pair: (
        -sum(a["topic"] in interests for a in pair),
        pair[0]["id"] not in pair[1]["prerequisites"],
        rank(pair[0]), rank(pair[1])))
    steps = []
    if pairs:
        for number, action in enumerate(pairs[0], 1):
            steps.append({
                "step_id": f"journey-{number}",
                "action_id": action["id"],
                "prerequisites": sorted(action["prerequisites"]),
                "requirement_ids": sorted(action["requirement_ids"]),
                "reason": ("Matches a stated interest." if action["topic"] in interests
                           else "Prerequisite-safe discovery outside stated interests."),
            })
    return {
        "state": "ready" if steps else "no_two_step_path",
        "next_actions": [a["id"] for a in available],
        "steps": steps,
        "completion_policy": "Planning does not mark any action as completed.",
    }


def _adaptive(data):
    profile = data["profile"]
    completed = set(profile["completed_actions"])
    beginner = profile["experience"] == "beginner"
    steps = []
    for journey_step in data["journey"]["steps"]:
        prerequisites = journey_step["prerequisites"]
        pending = sorted(set(prerequisites) - completed)
        steps.append({
            "step_id": journey_step["step_id"].replace("journey", "onboarding"),
            "source_journey_step_id": journey_step["step_id"],
            "action_id": journey_step["action_id"],
            "requirement_ids": list(journey_step["requirement_ids"]),
            "prerequisites": list(prerequisites),
            "requires_completion": pending,
            "readiness": "conditional" if pending else "ready_now",
            "format": profile["preferred_formats"][0],
            "depth": "guided" if beginner else "concise",
            "instructions": (["Read the explanation", "Practice with synthetic data",
                              "Confirm completion"] if beginner else
                             ["Check the summary", "Confirm completion"]),
            "explanation": (
                f"{profile['experience']} experience selects "
                f"{'guided' if beginner else 'concise'} instructions; "
                f"{profile['preferred_formats'][0]} is the first preferred format. "
                + ("Finish prerequisites before starting." if pending
                   else "All prerequisites are already completed.")
            ),
        })
    return {
        "state": "ready" if steps else "blocked_no_journey",
        "steps": steps,
        "review_requirement_ids": sorted(
            {rid for step in steps for rid in step["requirement_ids"]}),
    }


def _review(data):
    requirements = {r["id"]: r for r in data["requirements"]}
    findings, gaps = [], []
    for rid in data["adaptive"]["review_requirement_ids"]:
        requirement = requirements[rid]
        sources = [s for s in data["adaptive"]["steps"] if rid in s["requirement_ids"]]
        observations = []
        for document in sorted(data["documents"], key=lambda d: d["id"]):
            if rid not in document["requirement_ids"]:
                continue
            tokens = set(re.findall(r"\w+", document["text"].casefold()))
            missing = sorted(t for t in requirement["required_terms"]
                             if t.casefold() not in tokens)
            observations.append({
                "document_id": document["id"], "locator": document["locator"],
                "missing_terms": missing,
                "result": "insufficient" if missing else "lexical_match",
            })
        covered = any(o["result"] == "lexical_match" for o in observations)
        trace = {
            "requirement_id": rid,
            "action_ids": sorted({s["action_id"] for s in sources}),
            "onboarding_step_ids": [s["step_id"] for s in sources],
            "journey_step_ids": [s["source_journey_step_id"] for s in sources],
        }
        findings.append({
            **trace, "description": requirement["description"],
            "result": "evidence_found" if covered else "gap",
            "observations": observations,
        })
        if not covered:
            gaps.append({
                **trace,
                "reason": "insufficient_terms" if observations else "missing_evidence",
                "evidence_document_ids": [o["document_id"] for o in observations],
                "requested_terms": sorted(requirement["required_terms"]),
            })
    return {
        "state": ("not_applicable" if not findings else
                  "gaps_found" if gaps else "evidence_found"),
        "findings": findings, "gaps": gaps, "disclaimer": DISCLAIMER,
    }


DERIVERS = {"journey": _journey, "adaptive": _adaptive, "review": _review}


def validate(data, through="input"):
    """One validation layer for inputs and every stage boundary.

    Re-derivation checks both the shape and provenance of bounded deterministic
    handoffs, rejecting forged, stale, or reordered stage outputs.
    """
    require(through in STAGES, "unknown validation stage")
    _validate_input(data, through)
    if through != "input":
        require(data["status"] == "ok", "status: expected ok")
    for stage in STAGES[1:STAGES.index(through) + 1]:
        expected = DERIVERS[stage](data)
        require(data[stage] == expected, f"{stage}: invalid or stale handoff")
    return data


def advance(data, stage):
    require(stage in DERIVERS, "unknown pipeline stage")
    previous = STAGES[STAGES.index(stage) - 1]
    validate(data, previous)
    result = copy.deepcopy(data)
    result["status"] = "ok"
    result[stage] = DERIVERS[stage](result)
    validate(result, stage)
    return result


def run_pipeline(data):
    result = data
    for stage in STAGES[1:]:
        result = advance(result, stage)
    return result


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
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
