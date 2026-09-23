"""Deterministic synthetic customer-insight pipeline; Python standard library only.

Usage: python -B implementation.py example_input.json
The manifest describes the shared schema, ranking rules, and bounded scope.
"""

import copy
import itertools
import json
import re
import sys


STAGES = ("sentiment", "deep", "journey", "extract")
SEVERITY = {"low": 1, "medium": 3, "high": 6, "critical": 10}
LEXICON = {
    "good": 1, "great": 2, "excellent": 3, "love": 2, "helpful": 1,
    "bad": -1, "broken": -2, "terrible": -3, "hate": -2,
    "slow": -1, "failed": -2, "unsafe": -3,
}
NEGATORS = {"not", "never", "no"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown keys")


def text(value, path, limit=10000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            path + " must be a nonblank bounded string")


def sequence(value, path, limit=50):
    require(isinstance(value, list) and len(value) <= limit,
            path + " must be a bounded array")


def unique(values, path):
    require(len(values) == len(set(values)), path + " contains duplicates")


def validate_input(request):
    keys(request, ("schema_version", "synthetic", "feedback", "documents",
                   "actions", "profile", "extraction_schema"), "input")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "schema_version must be 1")
    require(request["synthetic"] is True, "synthetic must be true for this reference")
    for name in ("feedback", "documents", "actions", "extraction_schema"):
        sequence(request[name], name)
    for item in request["feedback"]:
        keys(item, ("id", "topic", "text", "severity"), "feedback item")
        for name in ("id", "topic", "text", "severity"):
            text(item[name], "feedback." + name)
        require(item["severity"] in SEVERITY, "unknown severity")
    unique([item["id"] for item in request["feedback"]], "feedback ids")
    for doc in request["documents"]:
        keys(doc, ("id", "text", "claims"), "document")
        text(doc["id"], "document.id")
        text(doc["text"], "document.text")
        sequence(doc["claims"], "claims")
        for claim in doc["claims"]:
            keys(claim, ("topic", "stance", "quote"), "claim")
            for name in ("topic", "stance", "quote"):
                text(claim[name], "claim." + name)
            require(claim["stance"] in ("supports", "opposes", "uncertain"),
                    "unknown claim stance")
            require(claim["quote"] in doc["text"], "claim quote is not in source")
    unique([doc["id"] for doc in request["documents"]], "document ids")
    for action in request["actions"]:
        keys(action, ("id", "title", "topic", "description", "prerequisites"), "action")
        for name in ("id", "title", "topic", "description"):
            text(action[name], "action." + name)
        sequence(action["prerequisites"], "prerequisites")
        for prerequisite in action["prerequisites"]:
            text(prerequisite, "prerequisite")
        unique(action["prerequisites"], "prerequisites")
    ids = [action["id"] for action in request["actions"]]
    unique(ids, "action ids")
    known = set(ids)
    for action in request["actions"]:
        require(set(action["prerequisites"]) <= known, "unknown prerequisite")
        require(action["id"] not in action["prerequisites"], "self prerequisite")
    keys(request["profile"], ("completed",), "profile")
    sequence(request["profile"]["completed"], "completed")
    for completed in request["profile"]["completed"]:
        text(completed, "completed id")
    unique(request["profile"]["completed"], "completed")
    require(set(request["profile"]["completed"]) <= known, "unknown completed action")
    for field in request["extraction_schema"]:
        keys(field, ("name", "source", "label", "type", "required"), "extraction field")
        for name in ("name", "source", "label", "type"):
            text(field[name], "extraction." + name, 100)
        require(field["source"] in ("title", "description"), "unknown extraction source")
        require(field["type"] in ("string", "integer"), "unknown extraction type")
        require(type(field["required"]) is bool, "required must be boolean")
        require(not any(char in field["label"] for char in "\r\n:"),
                "extraction labels cannot contain colons or newlines")
    unique([field["name"] for field in request["extraction_schema"]], "field names")


def _sentiment(envelope):
    scored = []
    topics = {}
    for item in envelope["input"]["feedback"]:
        tokens = list(re.finditer(r"[A-Za-z]+", item["text"]))
        hits = []
        for index, token in enumerate(tokens):
            word = token.group().lower()
            if word not in LEXICON:
                continue
            negated = index > 0 and tokens[index - 1].group().lower() in NEGATORS
            weight = LEXICON[word] * (-1 if negated else 1)
            hits.append({"token": token.group(), "span": list(token.span()),
                         "base_weight": LEXICON[word], "negated": negated,
                         "weight": weight})
        score = sum(hit["weight"] for hit in hits)
        priority = SEVERITY[item["severity"]] * 10 + max(0, -score)
        result = {"feedback_id": item["id"], "topic": item["topic"],
                  "severity": item["severity"], "score": score,
                  "sentiment": "negative" if score < 0 else "positive" if score > 0 else "neutral",
                  "priority": priority, "contributions": hits}
        scored.append(result)
        issue = topics.setdefault(item["topic"], {
            "topic": item["topic"], "priority": 0, "feedback_ids": [],
            "highest_severity": "low",
        })
        issue["priority"] = max(issue["priority"], priority)
        issue["feedback_ids"].append(item["id"])
        if SEVERITY[item["severity"]] > SEVERITY[issue["highest_severity"]]:
            issue["highest_severity"] = item["severity"]
    return {
        "rule": "priority = severity_weight * 10 + max(0, -sentiment_score); topic uses max",
        "feedback": scored,
        "issues": sorted(topics.values(), key=lambda item: (-item["priority"], item["topic"])),
    }


def _deep(envelope):
    reports = []
    for issue in envelope["stages"]["sentiment"]["issues"]:
        evidence = []
        for doc in envelope["input"]["documents"]:
            for claim in doc["claims"]:
                if claim["topic"] != issue["topic"]:
                    continue
                start = doc["text"].index(claim["quote"])
                record = {
                    "document_id": doc["id"], "stance": claim["stance"],
                    "quote": claim["quote"], "span": [start, start + len(claim["quote"])],
                }
                if record not in evidence:
                    evidence.append(record)
        stances = {record["stance"] for record in evidence}
        disagreement = {"supports", "opposes"} <= stances
        counts = {stance: sum(record["stance"] == stance for record in evidence)
                  for stance in ("supports", "opposes", "uncertain")}
        questions = []
        if not evidence:
            questions.append("What evidence addresses this issue?")
        if disagreement:
            questions.append("What explains the conflicting evidence?")
        if "uncertain" in stances:
            questions.append("What would resolve the uncertain evidence?")
        conclusion = ("no_evidence" if not evidence else "disputed" if disagreement
                      else "inconclusive" if "uncertain" in stances
                      else "supported" if "supports" in stances else "opposed")
        reports.append({
            **copy.deepcopy(issue), "evidence": evidence, "stance_counts": counts,
            "document_count": len({record["document_id"] for record in evidence}),
            "disagreement": disagreement, "conclusion": conclusion,
            "summary": (f"{issue['topic']}: {conclusion}; "
                        f"{counts['supports']} supporting, {counts['opposes']} opposing, "
                        f"{counts['uncertain']} uncertain evidence items."),
            "unresolved_questions": questions,
        })
    return {"topics": reports, "method": "Attributed evidence; no majority-vote truth inference"}


def _journey(envelope):
    request = envelope["input"]
    completed = set(request["profile"]["completed"])
    reports = {report["topic"]: report for report in envelope["stages"]["deep"]["topics"]}
    actions = [action for action in request["actions"] if action["id"] not in completed]

    def valid(plan):
        seen = completed.copy()
        for action in plan:
            if not set(action["prerequisites"]) <= seen:
                return False
            seen.add(action["id"])
        return True

    def score(plan):
        # Count a topic once: repeating an issue must not inflate the journey score.
        return sum(reports[topic]["priority"] for topic in {a["topic"] for a in plan}
                   if topic in reports)

    def relevant(plan):
        return all(action["topic"] in reports or
                   any(action["id"] in later["prerequisites"] for later in plan[index + 1:])
                   for index, action in enumerate(plan))

    pairs = [pair for pair in itertools.permutations(actions, 2)
             if valid(pair) and relevant(pair) and score(pair) > 0]
    singles = [(action,) for action in actions
               if valid((action,)) and score((action,)) > 0]
    candidates = pairs or singles
    plan = min(candidates, key=lambda p: (-score(p), tuple(a["id"] for a in p))) if candidates else ()
    seen = completed.copy()
    steps = []
    for number, action in enumerate(plan, 1):
        report = reports.get(action["topic"])
        steps.append({
            "step": number, **copy.deepcopy(action),
            "prerequisites_satisfied": sorted(set(action["prerequisites"]) & seen),
            "priority": report["priority"] if report else 0,
            "research_conclusion": report["conclusion"] if report else "prerequisite_only",
            "unresolved_questions": copy.deepcopy(report["unresolved_questions"]) if report else [],
            "evidence_document_ids": sorted({e["document_id"] for e in report["evidence"]}) if report else [],
        })
        seen.add(action["id"])
    eligible = sorted(action["id"] for action in actions
                      if set(action["prerequisites"]) <= completed)
    return {
        "status": "complete" if len(plan) == 2 else "blocked",
        "steps": steps, "eligible_now": eligible,
        "plan_priority": score(plan),
        "blocking_reason": None if len(plan) == 2 else
        "No prerequisite-valid, relevant two-step journey exists; no steps were invented.",
    }


def _extract(envelope):
    fields = []
    missing = []
    required_missing = []
    steps = envelope["stages"]["journey"]["steps"]
    for spec in envelope["input"]["extraction_schema"]:
        candidates = []
        for step in steps:
            source = step[spec["source"]]
            # Labels are literal. Capture a single line, preserving exact code-point offsets.
            pattern = r"(?m)^[ \t]*" + re.escape(spec["label"]) + r"[ \t]*:[ \t]*([^\r\n]*)"
            for match in re.finditer(pattern, source):
                raw = match.group(1)
                value = raw.strip()
                if not value:
                    continue
                start = match.start(1) + len(raw) - len(raw.lstrip())
                candidates.append({
                    "raw": value, "source": {"action_id": step["id"],
                    "field": spec["source"], "span": [start, start + len(value)],
                    "text": source},
                })
        selected = candidates[0] if candidates else None
        reason = None if selected else "not_found"
        value = selected["raw"] if selected else None
        if selected and spec["type"] == "integer":
            if re.fullmatch(r"[+-]?[0-9]+", value) and len(value) <= 100:
                value = int(value)
            else:
                value = None
                reason = "invalid_integer"
        if reason:
            missing.append(spec["name"])
            if spec["required"]:
                required_missing.append(spec["name"])
        fields.append({
            "name": spec["name"], "type": spec["type"], "required": spec["required"],
            "value": value, "source": selected["source"] if selected else None,
            "missing_reason": reason, "match_count": len(candidates),
        })
    return {
        "fields": fields, "missing_fields": missing, "required_missing": required_missing,
        "complete": not required_missing,
        "selection_rule": "First nonblank occurrence in journey order, then source order",
        "journey_status": envelope["stages"]["journey"]["status"],
    }


BUILDERS = dict(zip(STAGES, (_sentiment, _deep, _journey, _extract)))


def validate_envelope(envelope, expected_stage_count=None):
    """One shared boundary validator verifies structure AND deterministic semantics."""
    keys(envelope, ("schema_version", "status", "input", "stages"), "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "invalid envelope version")
    require(envelope["status"] == "ok", "invalid success status")
    validate_input(envelope["input"])
    require(isinstance(envelope["stages"], dict), "stages must be an object")
    count = len(envelope["stages"])
    require(count <= len(STAGES) and set(envelope["stages"]) == set(STAGES[:count]),
            "stages must be a contiguous pipeline prefix")
    if expected_stage_count is not None:
        require(count == expected_stage_count, "unexpected stage boundary")
    for stage in STAGES[:count]:
        actual = json.dumps(envelope["stages"][stage], sort_keys=True, allow_nan=False)
        expected = json.dumps(BUILDERS[stage](envelope), sort_keys=True, allow_nan=False)
        require(actual == expected, stage + " output failed semantic validation")
    return envelope


def advance(envelope, stage):
    require(isinstance(stage, str) and stage in STAGES, "unknown stage")
    validate_envelope(envelope, STAGES.index(stage))
    result = copy.deepcopy(envelope)
    result["stages"][stage] = BUILDERS[stage](result)
    return validate_envelope(result, STAGES.index(stage) + 1)


def run_pipeline(request):
    validate_input(request)
    envelope = {"schema_version": 1, "status": "ok",
                "input": copy.deepcopy(request), "stages": {}}
    for stage in STAGES:
        envelope = advance(envelope, stage)
    return envelope


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], "r", encoding="utf-8") as handle:
            payload = handle.read(2_000_001)
        require(len(payload) <= 2_000_000, "input exceeds 2,000,000 characters")
        request = json.loads(payload, object_pairs_hook=_object, parse_constant=_invalid_constant)
        result = run_pipeline(request)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
