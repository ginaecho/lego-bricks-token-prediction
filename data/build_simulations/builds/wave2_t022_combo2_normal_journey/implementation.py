"""Synthetic, deterministic research -> prerequisite-aware discovery reference.

Run: python -B implementation.py example_input.json
Only JSON is written to stdout. No network, third-party packages, or providers.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


STOPWORDS = frozenset("a an and are as at be by for from in is it of on or the to with".split())


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, name):
    require(isinstance(value, dict), name + " must be an object")
    require(set(value) == set(expected.split()), name + " has missing or unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonblank text")


def strings(value, name, nonempty=False):
    require(isinstance(value, list), name + " must be an array")
    for item in value:
        text(item, name + " item")
    require(len(set(value)) == len(value), name + " must contain unique strings")
    require(not nonempty or bool(value), name + " must not be empty")


def terms(value):
    return set(re.findall(r"\w+", value.casefold())) - STOPWORDS


def sentence_spans(value):
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", value):
        start, end = match.span()
        while start < end and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end


def validate(kind, value, document=None, research=None):
    """The shared validation boundary for input and both stage handoffs."""
    if kind == "input":
        fields(value, "schema_version synthetic_data query profile passages actions", "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(value["synthetic_data"] is True, "synthetic_data must be true")
        text(value["query"], "query")
        require(bool(terms(value["query"])), "query must have informative terms")
        fields(value["profile"], "skills goals", "profile")
        strings(value["profile"]["skills"], "skills")
        strings(value["profile"]["goals"], "goals", True)
        for collection in ("passages", "actions"):
            require(isinstance(value[collection], list), collection + " must be an array")
        passage_ids = set()
        for passage in value["passages"]:
            fields(passage, "id title source_uri text", "passage")
            for key in passage:
                text(passage[key], "passage." + key)
            require(passage["id"] not in passage_ids, "duplicate passage id")
            passage_ids.add(passage["id"])
        action_ids = set()
        for action in value["actions"]:
            fields(action, "id label requires grants evidence_passage_ids", "action")
            text(action["id"], "action.id")
            text(action["label"], "action.label")
            require(action["id"] not in action_ids, "duplicate action id")
            action_ids.add(action["id"])
            strings(action["requires"], "requires")
            strings(action["grants"], "grants", True)
            strings(action["evidence_passage_ids"], "evidence_passage_ids", True)
            require(set(action["evidence_passage_ids"]) <= passage_ids,
                    "action references unknown passage")
        return value

    require(document is not None, "validation requires original input")
    validate("input", document)
    passages = {p["id"]: p for p in document["passages"]}
    if kind == "research":
        fields(value, "query findings", "research")
        require(value["query"] == document["query"], "research query changed")
        require(isinstance(value["findings"], list), "findings must be an array")
        seen = set()
        previous = None
        for index, finding in enumerate(value["findings"], 1):
            fields(finding, "id score citation", "finding")
            require(finding["id"] == "finding-" + str(index), "invalid finding id")
            citation = finding["citation"]
            fields(citation, "passage_id title source_uri start end quote", "citation")
            text(citation["passage_id"], "citation.passage_id")
            require(citation["passage_id"] in passages, "unknown cited passage")
            require(citation["passage_id"] not in seen, "duplicate cited passage")
            seen.add(citation["passage_id"])
            passage = passages[citation["passage_id"]]
            start, end = citation["start"], citation["end"]
            require(type(start) is int and type(end) is int, "offsets must be integers")
            require(0 <= start < end <= len(passage["text"]), "invalid citation offsets")
            require(citation["quote"] == passage["text"][start:end], "citation quote mismatch")
            require(citation["title"] == passage["title"] and
                    citation["source_uri"] == passage["source_uri"], "citation source mismatch")
            score = len(terms(document["query"]) & terms(citation["quote"]))
            require(type(finding["score"]) is int and finding["score"] == score and score > 0,
                    "invalid relevance score")
            order = (-score, citation["passage_id"])
            require(previous is None or previous <= order, "findings are not ranked")
            previous = order
        return value

    if kind == "journey":
        require(research is not None, "journey requires research")
        validate("research", research, document)
        fields(value, "status reason recommendations steps final_skills goals_met", "journey")
        require(value["status"] in ("ok", "no_evidence", "no_journey"), "invalid journey status")
        text(value["reason"], "reason")
        require(isinstance(value["recommendations"], list), "recommendations must be an array")
        require(isinstance(value["steps"], list), "steps must be an array")
        actions = {a["id"]: a for a in document["actions"]}
        initial = set(document["profile"]["skills"])
        eligible = supported_actions(document, research)
        eligible_ids = {a["id"] for a in eligible}
        refs = finding_refs(research)

        def check_action(item, expected_fields, known):
            fields(item, expected_fields, "journey action")
            text(item["action_id"], "action_id")
            require(item["action_id"] in eligible_ids, "action lacks retrieved evidence")
            action = actions[item["action_id"]]
            require(item["label"] == action["label"], "action label changed")
            require(set(action["requires"]) <= known, "unsatisfied prerequisites")
            require(bool(set(action["grants"]) - known), "action makes no progress")
            require(item["finding_ids"] == [refs[p] for p in action["evidence_passage_ids"]],
                    "action evidence changed")
            return action

        recommended = set()
        for recommendation in value["recommendations"]:
            action = check_action(recommendation, "action_id label finding_ids", initial)
            require(action["id"] not in recommended, "duplicate recommendation")
            recommended.add(action["id"])
        expected = {a["id"] for a in eligible if set(a["requires"]) <= initial
                    and set(a["grants"]) - initial}
        require(recommended == expected, "recommendations omit actionable supported actions")
        known = initial.copy()
        used = set()
        for index, step in enumerate(value["steps"], 1):
            action = check_action(step, "position action_id label finding_ids skills_before skills_after", known)
            require(type(step["position"]) is int and step["position"] == index, "invalid step position")
            require(action["id"] not in used, "repeated journey action")
            used.add(action["id"])
            require(step["skills_before"] == sorted(known), "invalid skills_before")
            known.update(action["grants"])
            require(step["skills_after"] == sorted(known), "invalid skills_after")
        goals_met = set(document["profile"]["goals"]) <= known
        require(value["final_skills"] == sorted(known), "invalid final skills")
        require(type(value["goals_met"]) is bool and value["goals_met"] == goals_met,
                "invalid goals_met")
        if value["status"] == "ok":
            require(len(value["steps"]) == 2 and goals_met, "journey must have two steps and meet goals")
            require(bool(set(document["profile"]["goals"]) - initial), "goals already met")
        else:
            require(not value["steps"], "unsuccessful journey must not have partial steps")
            require((value["status"] == "no_evidence") == (not research["findings"]),
                    "status does not match evidence")
        return value
    raise ValidationError("unknown schema kind")


def run_research(document):
    validate("input", document)
    query_terms = terms(document["query"])
    findings = []
    for passage in document["passages"]:
        candidates = [(len(query_terms & terms(passage["text"][start:end])), start, end)
                      for start, end in sentence_spans(passage["text"])]
        if not candidates:
            continue
        score, start, end = min(candidates, key=lambda x: (-x[0], x[1]))
        if score:
            findings.append({"score": score, "citation": {
                "passage_id": passage["id"], "title": passage["title"],
                "source_uri": passage["source_uri"], "start": start, "end": end,
                "quote": passage["text"][start:end]}})
    findings.sort(key=lambda f: (-f["score"], f["citation"]["passage_id"]))
    for index, finding in enumerate(findings, 1):
        finding["id"] = "finding-" + str(index)
    return validate("research", {"query": document["query"], "findings": findings}, document)


def finding_refs(research):
    return {f["citation"]["passage_id"]: f["id"] for f in research["findings"]}


def supported_actions(document, research):
    retrieved = set(finding_refs(research))
    return [a for a in document["actions"] if set(a["evidence_passage_ids"]) <= retrieved]


def run_journey(document, research):
    validate("research", research, document)
    known = set(document["profile"]["skills"])
    goals = set(document["profile"]["goals"])
    refs = finding_refs(research)
    actions = supported_actions(document, research)
    first_actions = [a for a in actions if set(a["requires"]) <= known and set(a["grants"]) - known]
    first_actions.sort(key=lambda a: (-len(set(a["grants"]) & (goals - known)), a["id"]))

    def reference(action):
        return {"action_id": action["id"], "label": action["label"],
                "finding_ids": [refs[p] for p in action["evidence_passage_ids"]]}

    result = {"status": "no_journey", "reason": "no_valid_two_step_journey",
              "recommendations": [reference(a) for a in first_actions], "steps": [],
              "final_skills": sorted(known), "goals_met": goals <= known}
    if not research["findings"]:
        result.update(status="no_evidence", reason="no_relevant_passages")
    elif goals <= known:
        result["reason"] = "goals_already_met"
    else:
        pairs = []
        for first in first_actions:
            after_first = known | set(first["grants"])
            for second in actions:
                final = after_first | set(second["grants"])
                if (second["id"] != first["id"] and set(second["requires"]) <= after_first
                        and set(second["grants"]) - after_first and goals <= final):
                    pairs.append((first, second, final))
        if pairs:
            first, second, final = min(pairs, key=lambda pair: (
                -len(pair[2] - known), pair[0]["id"], pair[1]["id"]))
            for position, action in enumerate((first, second), 1):
                step = reference(action)
                step.update(position=position, skills_before=sorted(known))
                known.update(action["grants"])
                step["skills_after"] = sorted(known)
                result["steps"].append(step)
            result.update(status="ok", reason="validated_two_step_journey",
                          final_skills=sorted(final), goals_met=True)
    return validate("journey", result, document, research)


def pipeline(document):
    validate("input", document)
    research = run_research(document)
    journey = run_journey(document, research)
    return {"schema_version": 1, "synthetic_data": True, "status": journey["status"],
            "research": research, "journey": journey}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py input.json")
        with Path(argv[0]).open(encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = pipeline(document)
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
