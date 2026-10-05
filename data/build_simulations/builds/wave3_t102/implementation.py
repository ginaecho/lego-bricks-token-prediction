"""Deterministic synthetic evidence-to-customer-service reference pipeline."""
import json
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def strings(value):
    return isinstance(value, list) and all(text(item) for item in value)


def unique(values, label):
    require(len(values) == len(set(values)), label + " must be unique")


def validate(data, stage="input", context=None):
    """Shared structural and semantic validation for all pipeline boundaries."""
    require(isinstance(data, dict), stage + " must be an object")
    if stage == "input":
        require(data.get("schema_version") == "1.0", "schema_version must be 1.0")
        require(data.get("synthetic") is True, "fixtures must be labeled synthetic")
        docs = data.get("documents")
        require(isinstance(docs, list) and bool(docs), "documents must be nonempty")
        doc_ids = []
        for doc in docs:
            require(isinstance(doc, dict), "document must be an object")
            require(text(doc.get("id")) and text(doc.get("text")), "invalid document")
            doc_ids.append(doc["id"])
            claims = doc.get("claims")
            require(isinstance(claims, list), "claims must be a list")
            for claim in claims:
                require(isinstance(claim, dict), "claim must be an object")
                require(all(text(claim.get(k)) for k in ("topic", "value", "quote")),
                        "claim topic, value and quote are required strings")
                require(claim["value"] in claim["quote"], "value must occur in quote")
                start = claim.get("start")
                if start is None:
                    require(doc["text"].count(claim["quote"]) == 1,
                            "quote must occur exactly once or specify start")
                else:
                    require(type(start) is int and start >= 0, "invalid claim start")
                    require(doc["text"][start:start + len(claim["quote"])] == claim["quote"],
                            "claim span does not match document")
        unique(doc_ids, "document ids")
        require(strings(data.get("research_questions")), "invalid research questions")
        request = data.get("request")
        require(isinstance(request, dict) and text(request.get("query")),
                "request query required")
        require(strings(request.get("topics")) and bool(request["topics"]),
                "request topics must be nonempty strings")
        unique(request["topics"], "request topics")
        require(type(request.get("team_online")) is bool, "team_online must be boolean")
        require(strings(data.get("completed_actions")), "invalid completed_actions")
        unique(data["completed_actions"], "completed actions")
        actions = data.get("actions")
        require(isinstance(actions, list), "actions must be a list")
        ids = []
        for action in actions:
            require(isinstance(action, dict), "action must be an object")
            require(text(action.get("id")) and text(action.get("title")), "invalid action")
            require(strings(action.get("requires_topics")), "invalid required topics")
            require(strings(action.get("prerequisites")), "invalid prerequisites")
            unique(action["prerequisites"], "prerequisites")
            ids.append(action["id"])
        unique(ids, "action ids")
        require(set(data["completed_actions"]) <= set(ids), "unknown completed action")
        graph = {a["id"]: a["prerequisites"] for a in actions}
        for key, deps in graph.items():
            require(set(deps) <= set(ids), "unknown prerequisite")
            require(key not in deps, "self prerequisite")
        visiting, visited = set(), set()

        def visit(key):
            require(key not in visiting, "cyclic prerequisites")
            if key in visited:
                return
            visiting.add(key)
            for dep in graph[key]:
                visit(dep)
            visiting.remove(key)
            visited.add(key)

        for key in ids:
            visit(key)
        fields = data.get("extraction_schema")
        require(isinstance(fields, list) and bool(fields), "extraction schema required")
        names = []
        for field in fields:
            require(isinstance(field, dict), "field must be an object")
            require(text(field.get("name")) and text(field.get("topic")), "invalid field")
            require(field.get("type") in ("string", "integer", "boolean"), "unsupported type")
            require(type(field.get("required")) is bool, "required must be boolean")
            if "enum" in field:
                require(isinstance(field["enum"], list) and bool(field["enum"]),
                        "enum must be nonempty")
                target = {"string": str, "integer": int, "boolean": bool}[field["type"]]
                require(all(type(v) is target for v in field["enum"]), "enum type mismatch")
            names.append(field["name"])
        unique(names, "field names")
        return data
    require(context is not None, "validation context required")
    require(data.get("schema_version") == "1.0" and data.get("stage") == stage,
            "invalid stage envelope")
    # Recompute the deterministic contract, rejecting fabricated or altered handoffs.
    builders = {"deep": _deep, "journey": _journey, "support": _support, "extract": _extract}
    require(stage in builders, "unknown stage")
    expected = builders[stage](*context)
    require(json.dumps(data, sort_keys=True) == json.dumps(expected, sort_keys=True),
            stage + " violates validated handoff contract")
    return data


def envelope(stage, **payload):
    return {"schema_version": "1.0", "stage": stage, **payload}


def _deep(source):
    topics = {}
    for doc in source["documents"]:
        for claim in doc["claims"]:
            start = claim.get("start")
            if start is None:
                start = doc["text"].find(claim["quote"])
            item = {"document_id": doc["id"], "start": start,
                    "end": start + len(claim["quote"]), "quote": claim["quote"],
                    "value": claim["value"]}
            bucket = topics.setdefault(claim["topic"], [])
            if item not in bucket:
                bucket.append(item)
    findings = []
    questions = list(source["research_questions"])
    for topic, evidence in sorted(topics.items()):
        values = sorted({e["value"] for e in evidence})
        agreed = len(values) == 1
        findings.append({"topic": topic, "status": "agreed" if agreed else "disputed",
                         "value": values[0] if agreed else None,
                         "alternatives": values, "evidence": evidence,
                         "document_count": len({e["document_id"] for e in evidence})})
        if not agreed:
            questions.append("Which documented value is current for " + topic + "?")
    for topic in source["request"]["topics"]:
        if topic not in topics:
            questions.append("No evidence is available for " + topic + ".")
    return envelope("deep", findings=findings,
                    unresolved_questions=list(dict.fromkeys(questions)))


def _journey(source, deep):
    agreed = {f["topic"]: f for f in deep["findings"] if f["status"] == "agreed"}
    completed = set(source["completed_actions"])
    requested = set(source["request"]["topics"])
    candidates = [a for a in source["actions"] if a["id"] not in completed]
    candidates.sort(key=lambda a: (-len(set(a["requires_topics"]) & requested), a["id"]))

    def eligible(action, done):
        return (set(action["requires_topics"]) <= agreed.keys()
                and set(action["prerequisites"]) <= done)

    steps = []
    for first in candidates:
        if not eligible(first, completed):
            continue
        second = next((a for a in candidates if a["id"] != first["id"]
                       and eligible(a, completed | {first["id"]})), None)
        if second:
            for action in (first, second):
                steps.append({**action, "evidence": {
                    topic: agreed[topic]["evidence"] for topic in action["requires_topics"]}})
            break
    blocked = []
    for action in candidates:
        missing = sorted(set(action["requires_topics"]) - agreed.keys())
        prerequisites = sorted(set(action["prerequisites"]) - completed)
        if missing or prerequisites:
            blocked.append({"action_id": action["id"], "unresolved_topics": missing,
                            "pending_prerequisites": prerequisites})
    return envelope("journey", status="ready" if steps else "blocked", steps=steps,
                    blocked_actions=blocked, research=deep,
                    reason=None if steps else "No feasible two-step journey from current prerequisites.")


def _support(source, journey):
    findings = {f["topic"]: f for f in journey["research"]["findings"]}
    topics = list(source["request"]["topics"])
    for step in journey["steps"]:
        topics.extend(step["requires_topics"])
    answer = "Synthetic reference support response.\n"
    facts, unresolved = [], []
    for topic in dict.fromkeys(topics):
        finding = findings.get(topic)
        if finding is None or finding["status"] != "agreed":
            reason = "documents conflict" if finding else "there is no documented evidence"
            answer += topic + ": I cannot confirm this because " + reason + ".\n"
            unresolved.append(topic)
            continue
        answer += topic + ": "
        start = len(answer)
        answer += finding["value"]
        facts.append({"topic": topic, "value": finding["value"], "start": start,
                      "end": len(answer), "citations": finding["evidence"]})
        answer += ".\n"
    if journey["status"] == "ready":
        answer += "Recommended next actions (not executed):\n"
        for number, step in enumerate(journey["steps"], 1):
            answer += str(number) + ". " + step["title"] + "\n"
    else:
        answer += journey["reason"] + "\n"
    online = source["request"]["team_online"]
    answer += ("The team is online; contact support for unresolved questions."
               if online else
               "The team is offline. Save your question for support; no ticket has been created "
               "and no response time is promised.")
    return envelope("support", answer=answer, facts=facts, unresolved_topics=unresolved,
                    team_online=online, escalation_needed=bool(unresolved),
                    query=source["request"]["query"], journey=journey)


def convert(value, kind):
    if kind == "string":
        return value
    if kind == "integer":
        stripped = value.strip()
        require(bool(stripped) and stripped.lstrip("+-").isascii()
                and stripped.lstrip("+-").isdigit(), "not an integer")
        return int(stripped)
    require(value.casefold() in ("true", "false"), "not a boolean")
    return value.casefold() == "true"


def _extract(source, support):
    facts = {f["topic"]: f for f in support["facts"]}
    fields, missing = {}, []
    for spec in source["extraction_schema"]:
        fact = facts.get(spec["topic"])
        reason = None
        value = None
        if fact is None:
            reason = "No agreed, support-grounded value for topic."
        else:
            try:
                value = convert(fact["value"], spec["type"])
                require("enum" not in spec or value in spec["enum"], "outside enum")
            except (ValueError, OverflowError) as exc:
                reason = "Type or enum validation failed: " + str(exc)
        if reason:
            missing.append({"name": spec["name"], "required": spec["required"], "reason": reason})
            fields[spec["name"]] = {"value": None, "source_span": None, "citations": []}
        else:
            fields[spec["name"]] = {
                "value": value,
                "source_span": {"source": "support.answer", "start": fact["start"],
                                "end": fact["end"], "text": fact["value"]},
                "citations": fact["citations"]}
    return envelope("extract", fields=fields, missing_fields=missing,
                    status="incomplete" if any(f["required"] for f in missing) else "complete")


def run_pipeline(source):
    validate(source)
    deep = validate(_deep(source), "deep", (source,))
    journey = validate(_journey(source, deep), "journey", (source, deep))
    support = validate(_support(source, journey), "support", (source, journey))
    extraction = validate(_extract(source, support), "extract", (source, support))
    return {"schema_version": "1.0", "synthetic": True, "status": "ok",
            "deep": deep, "journey": journey, "support": support, "extract": extraction}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            source = json.load(handle)
        output = run_pipeline(source)
        code = 0
    except (ValidationError, OSError, ValueError, TypeError, RecursionError) as exc:
        output = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
