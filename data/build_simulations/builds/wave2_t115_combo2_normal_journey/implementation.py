"""Deterministic synthetic research -> prerequisite-aware discovery reference CLI."""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, location):
    require(isinstance(value, dict), f"{location} must be an object")
    require(set(value) == set(names.split()), f"{location} has invalid fields")


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), f"{location} must be nonempty text")


def strings(value, location):
    require(isinstance(value, list), f"{location} must be a list")
    for item in value:
        text(item, location)
    require(len(set(value)) == len(value), f"{location} contains duplicates")


def tokens(value):
    return set(re.findall(r"\w+", value.casefold()))


def _research(data):
    query = tokens(data["query"])
    findings = []
    for source in data["sources"]:
        for passage in source["passages"]:
            score = len(query & tokens(passage["text"]))
            if score:
                findings.append({
                    "id": f"finding-{len(findings) + 1}",
                    "quote": passage["text"],
                    "citation": {
                        "source_id": source["id"],
                        "passage_id": passage["id"],
                        "start": 0,
                        "end": len(passage["text"]),
                    },
                    "score": score,
                    "action_ids": list(passage["action_ids"]),
                })
    findings.sort(key=lambda f: (-f["score"], f["citation"]["source_id"],
                                 f["citation"]["passage_id"]))
    return {"query": data["query"], "findings": findings[:data["limit"]]}


def _journey(data, research):
    skills = set(data["profile"]["skills"])
    interests = set(data["profile"]["interests"])
    evidence = {}
    scores = {}
    for finding in research["findings"]:
        for action_id in finding["action_ids"]:
            evidence.setdefault(action_id, []).append(finding["id"])
            scores[action_id] = scores.get(action_id, 0) + finding["score"]
    supported = [action for action in data["actions"] if action["id"] in evidence]

    def rank(action):
        return scores[action["id"]] + len(interests & set(action["tags"]))

    supported.sort(key=lambda action: (-rank(action), action["id"]))

    def ready(action, known):
        return set(action["requires"]) <= known

    def recommendation(action):
        return {"action_id": action["id"], "title": action["title"],
                "finding_ids": list(evidence[action["id"]]), "score": rank(action)}

    next_actions = [recommendation(action) for action in supported if ready(action, skills)]
    pairs = []
    for first in supported:
        if not ready(first, skills):
            continue
        for second in supported:
            if second["id"] != first["id"] and ready(second, skills | set(first["provides"])):
                pairs.append((first, second))
    pairs.sort(key=lambda pair: (-(rank(pair[0]) + rank(pair[1])),
                                -rank(pair[0]), pair[0]["id"], pair[1]["id"]))
    steps = []
    if pairs:
        known = set(skills)
        for number, action in enumerate(pairs[0], 1):
            before = sorted(known)
            known.update(action["provides"])
            steps.append(dict(recommendation(action), step=number,
                              skills_before=before, skills_after=sorted(known)))
    blocked = [{"action_id": action["id"],
                "missing_prerequisites": sorted(set(action["requires"]) - skills)}
               for action in supported if not ready(action, skills)]
    status = "ready" if steps else ("insufficient_evidence" if not supported
                                    else "no_valid_two_step")
    return {"status": status, "next_actions": next_actions,
            "blocked_now": blocked, "steps": steps}


def validate(value, kind, context=None):
    """One validation entry point for input, research handoff, and final output."""
    if kind == "input":
        fields(value, "schema_version synthetic query limit profile sources actions", "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(value["synthetic"] is True, "fixtures must be labeled synthetic")
        text(value["query"], "query")
        require(bool(tokens(value["query"])), "query must contain a word")
        require(type(value["limit"]) is int and 1 <= value["limit"] <= 50,
                "limit must be an integer from 1 to 50")
        fields(value["profile"], "skills interests", "profile")
        strings(value["profile"]["skills"], "profile.skills")
        strings(value["profile"]["interests"], "profile.interests")
        require(isinstance(value["actions"], list), "actions must be a list")
        action_ids = set()
        for action in value["actions"]:
            fields(action, "id title requires provides tags", "action")
            text(action["id"], "action.id")
            text(action["title"], "action.title")
            require(action["id"] not in action_ids, "duplicate action id")
            action_ids.add(action["id"])
            for key in ("requires", "provides", "tags"):
                strings(action[key], f"action.{key}")
        require(isinstance(value["sources"], list), "sources must be a list")
        source_ids = set()
        for source in value["sources"]:
            fields(source, "id title passages", "source")
            text(source["id"], "source.id")
            text(source["title"], "source.title")
            require(source["id"] not in source_ids, "duplicate source id")
            source_ids.add(source["id"])
            require(isinstance(source["passages"], list), "passages must be a list")
            passage_ids = set()
            for passage in source["passages"]:
                fields(passage, "id text action_ids", "passage")
                text(passage["id"], "passage.id")
                text(passage["text"], "passage.text")
                require(passage["id"] not in passage_ids, "duplicate passage id within source")
                passage_ids.add(passage["id"])
                strings(passage["action_ids"], "passage.action_ids")
                require(set(passage["action_ids"]) <= action_ids, "unknown action reference")
    elif kind == "research":
        validate(context, "input")
        fields(value, "query findings", "research")
        require(value == _research(context), "research differs from verified source extraction")
    elif kind == "output":
        validate(context, "input")
        fields(value, "schema_version synthetic status research journey", "output")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "invalid output version")
        require(value["synthetic"] is True and value["status"] == "ok", "invalid output status")
        validate(value["research"], "research", context)
        fields(value["journey"], "status next_actions blocked_now steps", "journey")
        require(value["journey"] == _journey(context, value["research"]),
                "journey differs from validated evidence and prerequisite transitions")
    else:
        raise ValidationError("unknown schema kind")
    return value


def research_stage(data):
    validate(data, "input")
    return validate(_research(data), "research", data)


def journey_stage(data, research):
    validate(research, "research", data)
    result = {"schema_version": 1, "synthetic": True, "status": "ok",
              "research": research, "journey": _journey(data, research)}
    return validate(result, "output", data)


def run_pipeline(data):
    return journey_stage(data, research_stage(data))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
