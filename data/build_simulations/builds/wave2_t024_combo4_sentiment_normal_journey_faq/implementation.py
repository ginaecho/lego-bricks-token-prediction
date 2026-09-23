"""Deterministic synthetic customer-insight pipeline; Python standard library only.

Usage: python -B implementation.py example_input.json
All four stages share one validated envelope. Stage outputs are canonical,
reproducible values; validation also rejects altered evidence or handoffs.
No provider, learning model, or network is used.
"""

import copy
import json
import re
import sys


VERSION = "1.0"
STAGES = ("sentiment", "normal", "journey", "faq")
SEVERITY = {"low": 0, "normal": 1, "high": 2, "critical": 3}
LEXICON = {
    "good": 1, "great": 2, "love": 2, "helpful": 1, "resolved": 1,
    "bad": -1, "broken": -2, "fail": -2, "failed": -2, "slow": -1,
    "terrible": -2, "blocked": -2, "frustrated": -1,
}
NEGATORS = {"not", "no", "never", "can't", "won't", "isn't", "don't"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "my", "of", "on", "or", "please",
    "the", "this", "to", "was", "we", "what", "with", "you", "your",
} | NEGATORS
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.!?;,]", re.IGNORECASE)
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class ValidationError(ValueError):
    """A shared input, stage, or evidence invariant was violated."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(type(value) is dict, f"{path} must be an object")
    require(set(value) == set(expected), f"{path} has missing or unknown fields")


def text(value, path, limit=10000):
    require(type(value) is str and bool(value.strip()) and len(value) <= limit,
            f"{path} must be nonempty text of at most {limit} characters")


def identifier(value, path):
    require(type(value) is str and bool(IDENTIFIER.fullmatch(value)),
            f"{path} must be an ASCII identifier")


def sequence(value, path, limit=200):
    require(type(value) is list and len(value) <= limit,
            f"{path} must be an array of at most {limit} items")


def identifiers(value, path):
    sequence(value, path)
    for item in value:
        identifier(item, path)
    require(len(set(value)) == len(value), f"{path} contains duplicate identifiers")


def records(value, keys, path, limit=200):
    sequence(value, path, limit)
    seen = set()
    for record in value:
        fields(record, keys, path)
        identifier(record["id"], f"{path}.id")
        require(record["id"] not in seen, f"{path} contains duplicate ids")
        seen.add(record["id"])


def _validate_input(data):
    fields(data, ("schema_version", "synthetic", "fixture_label", "feedback",
                  "research_passages", "actions", "completed_action_ids",
                  "knowledge_base", "retrieval_limit"), "input")
    require(data["schema_version"] == VERSION, "unsupported schema_version")
    require(data["synthetic"] is True, "synthetic must be true")
    text(data["fixture_label"], "fixture_label", 200)
    require(type(data["retrieval_limit"]) is int
            and 1 <= data["retrieval_limit"] <= 5, "retrieval_limit must be 1..5")
    records(data["feedback"], ("id", "text", "severity"), "feedback")
    for item in data["feedback"]:
        text(item["text"], "feedback.text")
        require(type(item["severity"]) is str and item["severity"] in SEVERITY,
                "feedback.severity must be low, normal, high, or critical")
    records(data["research_passages"], ("id", "source", "text"), "research_passages")
    records(data["knowledge_base"], ("id", "source", "text", "action_ids"),
            "knowledge_base")
    for item in data["research_passages"] + data["knowledge_base"]:
        text(item["source"], "document.source", 500)
        text(item["text"], "document.text")
    records(data["actions"], ("id", "title", "prerequisites", "passage_ids",
                              "faq_question"), "actions", 50)
    action_ids = {item["id"] for item in data["actions"]}
    passage_ids = {item["id"] for item in data["research_passages"]}
    for item in data["actions"]:
        text(item["title"], "action.title", 500)
        text(item["faq_question"], "action.faq_question", 1000)
        identifiers(item["prerequisites"], "action.prerequisites")
        identifiers(item["passage_ids"], "action.passage_ids")
        require(set(item["prerequisites"]) <= action_ids, "unknown prerequisite")
        require(set(item["passage_ids"]) <= passage_ids, "unknown research passage")
    identifiers(data["completed_action_ids"], "completed_action_ids")
    require(set(data["completed_action_ids"]) <= action_ids, "unknown completed action")
    for item in data["knowledge_base"]:
        identifiers(item["action_ids"], "knowledge_base.action_ids")
        require(set(item["action_ids"]) <= action_ids, "unknown knowledge-base action")
    graph = {item["id"]: item["prerequisites"] for item in data["actions"]}
    visiting, visited = set(), set()

    def visit(node):
        require(node not in visiting, "cyclic action prerequisites")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def terms(value):
    return {match.group().lower() for match in WORD.finditer(value)} - STOPWORDS


def sentences(value):
    """Return sentence-like extracts and exact Python Unicode character offsets."""
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", value):
        raw = match.group()
        quote = raw.strip()
        if quote:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield start, start + len(quote), quote


def citation(document, start, end):
    return {"document_id": document["id"], "source": document["source"],
            "start": start, "end": end, "quote": document["text"][start:end]}


def _sentiment(data, previous):
    issues = []
    for feedback in data["feedback"]:
        history, evidence = [], []
        for match in TOKEN.finditer(feedback["text"]):
            term = match.group().lower()
            if term in ".!?;,":
                history = []
                continue
            if term in LEXICON:
                negated = sum(word in NEGATORS for word in history[-3:]) % 2 == 1
                weight = LEXICON[term]
                evidence.append({
                    "term": term, "start": match.start(), "end": match.end(),
                    "base_weight": weight, "negated": negated,
                    "contribution": -weight if negated else weight,
                })
            history.append(term)
        score = sum(item["contribution"] for item in evidence)
        issues.append({
            "feedback_id": feedback["id"], "text": feedback["text"],
            "severity": feedback["severity"], "severity_rank": SEVERITY[feedback["severity"]],
            "score": score, "label": "negative" if score < 0 else
            ("positive" if score > 0 else "neutral"), "evidence": evidence,
        })
    issues.sort(key=lambda item: (-item["severity_rank"], item["score"],
                                 item["feedback_id"]))
    for rank, issue in enumerate(issues, 1):
        issue["priority_rank"] = rank
    return {
        "method": "lexicon; odd negators in preceding 3 words; punctuation resets scope",
        "priority_rule": "severity descending, sentiment score ascending, feedback id ascending",
        "issues": issues,
    }


def _normal(data, previous):
    issues = previous["issues"]
    focus = None
    query = set()
    if issues:
        issue = issues[0]
        focus = {key: issue[key] for key in
                 ("feedback_id", "severity", "score", "label", "priority_rank")}
        query = terms(issue["text"])
    candidates = []
    for document in data["research_passages"]:
        overlap = query & terms(document["text"])
        if not overlap:
            continue
        extracts = [(len(query & terms(quote)), start, end)
                    for start, end, quote in sentences(document["text"])]
        if not extracts:
            continue
        score, start, end = min(extracts, key=lambda item: (-item[0], item[1]))
        if score:
            candidates.append((len(overlap), document["id"], document, start, end))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    findings = []
    for score, _, document, start, end in candidates[:data["retrieval_limit"]]:
        evidence = citation(document, start, end)
        findings.append({
            "finding_id": "finding:" + document["id"], "retrieval_score": score,
            "matched_terms": sorted(query & terms(evidence["quote"])),
            "text": evidence["quote"], "citation": evidence,
        })
    return {"status": "found" if findings else "no_findings", "focus": focus,
            "query_terms": sorted(query), "findings": findings}


def _journey(data, previous):
    completed = set(data["completed_action_ids"])
    findings = previous["findings"]
    supported = {finding["citation"]["document_id"]: finding for finding in findings}
    candidates = sorted(
        (action for action in data["actions"] if action["id"] not in completed
         and set(action["passage_ids"]) & supported.keys()),
        key=lambda action: action["id"],
    )
    available = [action for action in candidates
                 if set(action["prerequisites"]) <= completed]
    plans = [(first, second) for first in available for second in candidates
             if first["id"] != second["id"]
             and set(second["prerequisites"]) <= completed | {first["id"]}]
    # Enumerate complete pairs rather than greedily taking a dead-end first action.
    plans.sort(key=lambda plan: tuple(action["id"] for action in plan))
    steps = []
    if plans:
        satisfied = set(completed)
        for index, action in enumerate(plans[0], 1):
            evidence = [supported[pid] for pid in sorted(action["passage_ids"])
                        if pid in supported]
            steps.append({
                "step": index, "action_id": action["id"], "title": action["title"],
                "prerequisites": sorted(action["prerequisites"]),
                "satisfied_prerequisites": sorted(set(action["prerequisites"]) & satisfied),
                "finding_ids": [item["finding_id"] for item in evidence],
                "citations": [copy.deepcopy(item["citation"]) for item in evidence],
            })
            satisfied.add(action["id"])
    return {
        "status": "ready" if steps else "abstained", "focus": copy.deepcopy(previous["focus"]),
        "next_action_ids": [action["id"] for action in available],
        "validated_two_step": bool(steps), "steps": steps,
        "reason": None if steps else ("no_research_evidence" if not findings
                                     else "no_valid_two_step_journey"),
    }


def _faq(data, previous):
    answers = []
    actions = {action["id"]: action for action in data["actions"]}
    for step in previous["steps"]:
        action = actions[step["action_id"]]
        query = terms(action["faq_question"])
        candidates = []
        for document in data["knowledge_base"]:
            if action["id"] not in document["action_ids"]:
                continue
            for start, end, quote in sentences(document["text"]):
                score = len(query & terms(quote))
                if score:
                    candidates.append((score, document["id"], start, end, document))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        evidence = []
        if candidates:
            _, _, start, end, document = candidates[0]
            evidence = [citation(document, start, end)]
        answers.append({
            "action_id": action["id"], "question": action["faq_question"],
            "status": "answered" if evidence else "abstained",
            "answer": evidence[0]["quote"] if evidence else None,
            "citations": evidence,
            "reason": None if evidence else "no_action_scoped_matching_evidence",
        })
    answered = sum(answer["status"] == "answered" for answer in answers)
    status = ("answered" if answers and answered == len(answers) else
              ("partial" if answered else "abstained"))
    return {
        "status": status, "focus": copy.deepcopy(previous["focus"]),
        "journey_action_ids": [step["action_id"] for step in previous["steps"]],
        "answers": answers,
        "reason": ("no_valid_journey" if not answers else
                   ("insufficient_knowledge_base_evidence" if answered < len(answers) else None)),
    }


BUILDERS = dict(zip(STAGES, (_sentiment, _normal, _journey, _faq)))


def validate(value, kind="input"):
    """The shared boundary validator, including exact canonical stage evidence.

    Recalculation validates types, order, grounding, priorities, prerequisites,
    and provenance together. JSON encoding prevents bool/int equality loopholes.
    """
    if kind == "input":
        _validate_input(value)
        return
    require(kind == "state", "unknown validation kind")
    fields(value, ("schema_version", "status", "input", "stages"), "state")
    require(value["schema_version"] == VERSION and value["status"] == "ok",
            "invalid state header")
    _validate_input(value["input"])
    require(type(value["stages"]) is dict, "state.stages must be an object")
    present = value["stages"]
    require(len(present) <= len(STAGES)
            and set(present) == set(STAGES[:len(present)]),
            "stages must form an unbroken pipeline prefix")
    previous = None
    for name in STAGES[:len(present)]:
        expected = BUILDERS[name](value["input"], previous)
        try:
            actual_json = json.dumps(present[name], sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError(f"invalid {name} output") from error
        require(actual_json == json.dumps(expected, sort_keys=True, allow_nan=False),
                f"{name} output violates canonical evidence or handoff schema")
        previous = present[name]


def run_stage(state, name):
    validate(state, "state")
    count = len(state["stages"])
    require(count < len(STAGES) and name == STAGES[count], "stage executed out of order")
    updated = copy.deepcopy(state)
    previous = updated["stages"][STAGES[count - 1]] if count else None
    updated["stages"][name] = BUILDERS[name](updated["input"], previous)
    validate(updated, "state")
    return updated


def run_pipeline(data):
    validate(data)
    state = {"schema_version": VERSION, "status": "ok",
             "input": copy.deepcopy(data), "stages": {}}
    for name in STAGES:
        state = run_stage(state, name)
    return state


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError(f"nonfinite JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], "r", encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        result = run_pipeline(data)
    except (ValidationError, OSError, ValueError, UnicodeError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
