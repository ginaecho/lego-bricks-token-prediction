"""Synthetic, deterministic research -> triage -> onboarding -> extraction.

Run: python -B implementation.py example_input.json
Offsets are zero-based, half-open Python Unicode character offsets. Extraction
patterns are trusted configuration, not an untrusted regex execution service.
All stages use the same validated envelope and retain their upstream evidence.
"""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    """A user-readable contract violation."""


PHASES = ("research", "triage", "adaptive", "extract")
EXPERIENCES = ("beginner", "intermediate", "advanced")
MODES = ("text", "video", "interactive")
PRIORITIES = ("urgent", "high", "normal", "low")


def check(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, path):
    check(isinstance(value, dict), path + " must be an object")
    return value


def text(value, path, limit=20000):
    check(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    check(len(value) <= limit, path + " exceeds length limit")
    return value


def items(value, path, nonempty=False, limit=100):
    check(isinstance(value, list), path + " must be an array")
    check(len(value) <= limit and (not nonempty or bool(value)), path + " has invalid size")
    return value


def strings(value, path, nonempty=False):
    values = items(value, path, nonempty)
    for entry in values:
        text(entry, path)
    check(len(values) == len(set(values)), path + " contains duplicates")
    return values


def identified(value, path, nonempty=False):
    values = items(value, path, nonempty)
    ids = set()
    for entry in values:
        obj(entry, path)
        identifier = text(entry.get("id"), path + ".id", 100)
        check(identifier not in ids, path + " contains duplicate id " + identifier)
        ids.add(identifier)
    return values, ids


def words(value):
    return set(re.findall(r"\w+", value.casefold(), re.UNICODE))


def validate_input(data):
    obj(data, "input")
    check(type(data.get("schema_version")) is int and data["schema_version"] == 1,
          "schema_version must be 1")
    check(data.get("synthetic") is True, "synthetic must be true for this reference implementation")
    query = text(data.get("query"), "query", 500)
    check(bool(words(query)), "query must contain a searchable word")
    docs, doc_ids = identified(data.get("documents"), "documents", True)
    for doc in docs:
        text(doc.get("text"), "document.text")
    tickets, _ = identified(data.get("tickets"), "tickets", True)
    for ticket in tickets:
        text(ticket.get("title"), "ticket.title", 500)
        text(ticket.get("description"), "ticket.description")
        refs = strings(ticket.get("document_ids"), "ticket.document_ids")
        check(set(refs) <= doc_ids, "ticket references unknown document")
    profile = obj(data.get("profile"), "profile")
    check(profile.get("experience") in EXPERIENCES, "invalid profile experience")
    check(profile.get("preference") in MODES, "invalid profile preference")
    policy = obj(data.get("policy"), "policy")
    rules, categories = identified(policy.get("categories"), "policy.categories", True)
    for rule in rules:
        strings(rule.get("keywords"), "category.keywords", True)
        text(rule.get("queue"), "category.queue", 100)
        text(rule.get("owner"), "category.owner", 100)
    check(policy.get("default_category") in categories, "invalid default category")
    priorities = items(policy.get("priorities"), "policy.priorities")
    seen = set()
    for rule in priorities:
        obj(rule, "priority rule")
        priority = rule.get("priority")
        check(priority in PRIORITIES and priority not in seen, "invalid or duplicate priority")
        seen.add(priority)
        strings(rule.get("keywords"), "priority.keywords", True)
    check(policy.get("default_priority") in PRIORITIES, "invalid default priority")
    limit = policy.get("research_limit")
    check(type(limit) is int and 1 <= limit <= 100, "research_limit must be 1..100")
    steps, step_ids = identified(data.get("onboarding_steps"), "onboarding_steps")
    for step in steps:
        text(step.get("title"), "step.title", 500)
        targets = strings(step.get("categories"), "step.categories", True)
        check(set(targets) <= categories | {"*"}, "step references unknown category")
        audiences = strings(step.get("audiences"), "step.audiences", True)
        check(set(audiences) <= set(EXPERIENCES), "invalid step audience")
        modes = strings(step.get("modes"), "step.modes", True)
        check(set(modes) <= set(MODES), "invalid step mode")
        prereqs = strings(step.get("prerequisites"), "step.prerequisites")
        check(set(prereqs) <= step_ids, "unknown prerequisite")
        refs = strings(step.get("document_ids"), "step.document_ids")
        check(set(refs) <= doc_ids, "step references unknown document")
    by_id = {step["id"]: step for step in steps}
    visiting, visited = set(), set()

    def visit(identifier):
        check(identifier not in visiting, "onboarding prerequisites contain a cycle")
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in by_id[identifier]["prerequisites"]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in by_id:
        visit(identifier)
    fields = items(data.get("extraction_schema"), "extraction_schema", True)
    names = set()
    for field in fields:
        obj(field, "field")
        name = text(field.get("name"), "field.name", 100)
        check(name not in names, "duplicate extraction field")
        names.add(name)
        check(type(field.get("required")) is bool, "field.required must be boolean")
        pattern = text(field.get("pattern"), "field.pattern", 500)
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            raise ValidationError("invalid extraction regex: " + str(exc)) from exc
        check("value" in regex.groupindex, "field pattern requires a named value group")
    return data


def citation(doc_id, start, end, content):
    return {"document_id": doc_id, "start": start, "end": end, "text": content[start:end]}


def validate_citation(source, documents):
    obj(source, "source")
    check(source.get("document_id") in documents, "source references unknown document")
    start, end = source.get("start"), source.get("end")
    content = documents[source["document_id"]]
    check(type(start) is int and type(end) is int and 0 <= start < end <= len(content),
          "invalid source span")
    check(source.get("text") == content[start:end], "source text does not match span")


def validate(state, expected=None):
    """The single validation boundary used before and after every stage."""
    obj(state, "envelope")
    check(state.get("schema_version") == 1 and type(state.get("schema_version")) is int,
          "invalid envelope schema_version")
    check(state.get("status") == "ok" and state.get("synthetic") is True, "invalid envelope status")
    data = validate_input(state.get("input"))
    stages = obj(state.get("stages"), "stages")
    count = len(stages)
    check(count <= len(PHASES) and set(stages) == set(PHASES[:count]),
          "stages must be a contiguous pipeline prefix")
    if expected is not None:
        check(count == expected, "unexpected pipeline stage")
    docs = {doc["id"]: doc["text"] for doc in data["documents"]}
    tickets = {ticket["id"]: ticket for ticket in data["tickets"]}
    categories = {rule["id"]: rule for rule in data["policy"]["categories"]}
    findings = {}
    if count >= 1:
        research_output = obj(stages["research"], "research")
        passages, _ = identified(research_output.get("findings"), "research.findings")
        check(len(passages) <= data["policy"]["research_limit"], "too many findings")
        for passage in passages:
            validate_citation(passage.get("source"), docs)
            check(passage.get("finding") == passage["source"]["text"], "finding is not extractive")
            score = passage.get("score")
            actual_score = len(words(data["query"]) & words(passage["finding"]))
            check(type(score) is int and score == actual_score and score > 0, "invalid research score")
            findings[passage["id"]] = passage
        check(research_output.get("query") == data["query"], "research query changed")
    routes = {}
    if count >= 2:
        triage_output = obj(stages["triage"], "triage")
        records = items(triage_output.get("tickets"), "triage.tickets")
        for record in records:
            obj(record, "triage ticket")
            tid = record.get("ticket_id")
            check(tid in tickets and tid not in routes, "invalid triage ticket id")
            category = record.get("category")
            check(category in categories, "unknown triage category")
            check(record.get("priority") in PRIORITIES, "invalid triage priority")
            check(record.get("queue") == categories[category]["queue"] and
                  record.get("owner") == categories[category]["owner"], "unaccountable routing")
            check(record.get("document_ids") == tickets[tid]["document_ids"], "triage document drift")
            evidence = strings(record.get("finding_ids"), "triage.finding_ids")
            expected_evidence = [fid for fid, finding in findings.items()
                                 if finding["source"]["document_id"] in record["document_ids"]]
            check(evidence == expected_evidence, "triage evidence drift")
            text(record.get("explanation"), "triage.explanation")
            routes[tid] = record
        check(set(routes) == set(tickets), "triage must cover every ticket")
    assignments = {}
    if count >= 3:
        adaptive_output = obj(stages["adaptive"], "adaptive")
        records = items(adaptive_output.get("assignments"), "adaptive.assignments")
        steps = {step["id"]: step for step in data["onboarding_steps"]}
        for record in records:
            obj(record, "assignment")
            tid = record.get("ticket_id")
            check(tid in routes and tid not in assignments, "invalid adaptive ticket id")
            for key in ("category", "priority", "queue", "owner", "finding_ids"):
                check(record.get(key) == routes[tid][key], "adaptive routing/evidence drift")
            selected = items(record.get("steps"), "assignment.steps")
            seen = set()
            required_docs = list(routes[tid]["document_ids"])
            for selection in selected:
                obj(selection, "selected step")
                sid = selection.get("step_id")
                check(sid in steps and sid not in seen, "invalid selected step")
                step = steps[sid]
                check(set(step["prerequisites"]) <= seen, "prerequisite not scheduled before step")
                check(selection.get("mode") in step["modes"], "unsupported selected mode")
                text(selection.get("explanation"), "step.explanation")
                seen.add(sid)
                for doc in step["document_ids"]:
                    if doc not in required_docs:
                        required_docs.append(doc)
            check(record.get("document_ids") == required_docs, "adaptive document drift")
            assignments[tid] = record
        check(set(assignments) == set(tickets), "adaptive must cover every ticket")
    if count >= 4:
        extraction = obj(stages["extract"], "extract")
        records = items(extraction.get("records"), "extract.records")
        seen = set()
        schema = {field["name"]: field for field in data["extraction_schema"]}
        for record in records:
            obj(record, "extraction record")
            tid = record.get("ticket_id")
            check(tid in assignments and tid not in seen, "invalid extraction ticket id")
            seen.add(tid)
            assignment = assignments[tid]
            for key in ("category", "priority", "queue", "owner", "finding_ids", "document_ids"):
                check(record.get(key) == assignment[key], "extraction handoff drift")
            check(record.get("step_ids") == [step["step_id"] for step in assignment["steps"]],
                  "extraction step drift")
            fields = obj(record.get("fields"), "extraction.fields")
            check(set(fields) == set(schema), "extraction fields do not match schema")
            missing = []
            for name, value in fields.items():
                if value is None:
                    missing.append(name)
                    continue
                obj(value, "extracted field")
                validate_citation(value.get("source"), docs)
                check(value["source"]["document_id"] in assignment["document_ids"],
                      "extraction source outside assigned documents")
                check(value.get("value") == value["source"]["text"], "extracted value/span mismatch")
            check(record.get("missing_fields") == missing, "incorrect missing fields")
            required = [name for name in missing if schema[name]["required"]]
            check(record.get("missing_required_fields") == required, "incorrect missing required fields")
            check(record.get("complete") is (not required), "incorrect completion status")
        check(seen == set(tickets), "extraction must cover every ticket")
    return state


def advance(state, name, output):
    result = copy.deepcopy(state)
    result["stages"][name] = output
    return validate(result)


def research(state):
    validate(state, 0)
    data = state["input"]
    query_words = words(data["query"])
    ranked = []
    for doc_index, doc in enumerate(data["documents"]):
        for match in re.finditer(r"[^.!?\n]+[.!?]*", doc["text"]):
            raw = match.group()
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.end() - (len(raw) - len(raw.rstrip()))
            passage = doc["text"][start:end]
            score = len(query_words & words(passage))
            if score:
                ranked.append((-score, doc_index, start, end, doc))
    ranked.sort(key=lambda row: row[:3])
    findings = []
    for index, (negative_score, _, start, end, doc) in enumerate(
            ranked[:data["policy"]["research_limit"]], 1):
        source = citation(doc["id"], start, end, doc["text"])
        findings.append({"id": "finding-" + str(index), "finding": source["text"],
                         "score": -negative_score, "source": source})
    return advance(state, "research", {"query": data["query"], "findings": findings})


def keyword_matches(keywords, content):
    return [keyword for keyword in keywords
            if re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", content, re.IGNORECASE)]


def triage(state):
    validate(state, 1)
    data = state["input"]
    policy = data["policy"]
    records = []
    for ticket in data["tickets"]:
        evidence = [finding for finding in state["stages"]["research"]["findings"]
                    if finding["source"]["document_id"] in ticket["document_ids"]]
        content = "\n".join([ticket["title"], ticket["description"]] +
                            [finding["finding"] for finding in evidence])
        ranked = [(len(keyword_matches(rule["keywords"], content)), -index, rule)
                  for index, rule in enumerate(policy["categories"])]
        score, _, chosen = max(ranked, key=lambda row: row[:2])
        if not score:
            chosen = next(rule for rule in policy["categories"]
                          if rule["id"] == policy["default_category"])
        priority = policy["default_priority"]
        priority_reason = "default priority"
        # Priority rules are ordered; the first matching rule wins.
        for rule in policy["priorities"]:
            matches = keyword_matches(rule["keywords"], content)
            if matches:
                priority = rule["priority"]
                priority_reason = "priority keyword: " + matches[0]
                break
        records.append({
            "ticket_id": ticket["id"], "category": chosen["id"], "priority": priority,
            "queue": chosen["queue"], "owner": chosen["owner"],
            "document_ids": list(ticket["document_ids"]),
            "finding_ids": [finding["id"] for finding in evidence],
            "explanation": ("category keyword score " + str(score) if score else "default category") +
                           "; " + priority_reason + "; ties use category configuration order",
        })
    return advance(state, "triage", {"tickets": records})


def adaptive(state):
    validate(state, 2)
    data = state["input"]
    profile = data["profile"]
    steps = {step["id"]: step for step in data["onboarding_steps"]}
    assignments = []
    for route in state["stages"]["triage"]["tickets"]:
        targets = [step["id"] for step in steps.values()
                   if (route["category"] in step["categories"] or "*" in step["categories"])
                   and profile["experience"] in step["audiences"]]
        selections, visited = [], set()

        def include(sid, dependency_of=None):
            if sid in visited:
                return
            step = steps[sid]
            for prerequisite in step["prerequisites"]:
                include(prerequisite, sid)
            preferred = profile["preference"] in step["modes"]
            mode = profile["preference"] if preferred else step["modes"][0]
            reason = ("prerequisite for " + dependency_of if dependency_of else
                      "matches category " + route["category"] + " and experience " + profile["experience"])
            reason += "; preferred mode" if preferred else "; preference unavailable, first supported mode"
            selections.append({"step_id": sid, "title": step["title"], "mode": mode,
                               "explanation": reason})
            visited.add(sid)

        for target in targets:
            include(target)
        record = {key: copy.deepcopy(route[key]) for key in
                  ("ticket_id", "category", "priority", "queue", "owner", "finding_ids", "document_ids")}
        record["steps"] = selections
        for selection in selections:
            for doc_id in steps[selection["step_id"]]["document_ids"]:
                if doc_id not in record["document_ids"]:
                    record["document_ids"].append(doc_id)
        assignments.append(record)
    return advance(state, "adaptive", {"assignments": assignments})


def extract(state):
    validate(state, 3)
    data = state["input"]
    docs = {doc["id"]: doc["text"] for doc in data["documents"]}
    records = []
    for assignment in state["stages"]["adaptive"]["assignments"]:
        record = {key: copy.deepcopy(assignment[key]) for key in
                  ("ticket_id", "category", "priority", "queue", "owner", "finding_ids", "document_ids")}
        record["step_ids"] = [step["step_id"] for step in assignment["steps"]]
        fields, missing, required = {}, [], []
        for field in data["extraction_schema"]:
            regex = re.compile(field["pattern"], re.IGNORECASE | re.MULTILINE)
            value = None
            for doc_id in assignment["document_ids"]:
                for match in regex.finditer(docs[doc_id]):
                    start, end = match.span("value")
                    if start >= 0 and end > start:
                        source = citation(doc_id, start, end, docs[doc_id])
                        value = {"value": source["text"], "source": source}
                        break
                if value is not None:
                    break
            fields[field["name"]] = value
            if value is None:
                missing.append(field["name"])
                if field["required"]:
                    required.append(field["name"])
        record.update(fields=fields, missing_fields=missing,
                      missing_required_fields=required, complete=not required)
        records.append(record)
    return advance(state, "extract", {"records": records})


def run_pipeline(data):
    state = {"schema_version": 1, "status": "ok", "synthetic": True,
             "input": copy.deepcopy(data), "stages": {}}
    validate(state, 0)
    for stage in (research, triage, adaptive, extract):
        state = stage(state)
    return state


def reject_constant(value):
    raise ValidationError("nonstandard JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        check(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], "r", encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
