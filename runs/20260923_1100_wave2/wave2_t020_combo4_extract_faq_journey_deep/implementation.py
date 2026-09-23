"""Deterministic synthetic document -> FAQ -> journey -> research reference.

Run: python -B implementation.py example_input.json
All source offsets are half-open Python Unicode character offsets, not bytes.
Retrieval uses token-set Jaccard similarity (threshold 0.2); answers are quotes.
Actions have monotonic prerequisites; higher priority wins, then lexical ID.
No models, dependencies, network, persistent state, or external services are used.
"""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown keys")


def text(value, path, allow_empty=False):
    require(isinstance(value, str) and (allow_empty or bool(value.strip())),
            path + " must be a nonempty string")


def sequence(value, path):
    require(isinstance(value, list), path + " must be an array")


def strings(value, path):
    sequence(value, path)
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), path + " contains duplicates")


def indexed(items, path):
    sequence(items, path)
    result = {}
    for item in items:
        require(isinstance(item, dict) and "id" in item, path + " requires IDs")
        text(item["id"], path + ".id")
        require(item["id"] not in result, path + " has duplicate IDs")
        result[item["id"]] = item
    return result


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def convert(raw, kind):
    if kind == "string":
        return raw
    if kind == "integer":
        require(re.fullmatch(r"[+-]?[0-9]+", raw) is not None, "invalid integer")
        try:
            return int(raw)
        except ValueError as exc:
            raise ValidationError("invalid integer") from exc
    require(raw.casefold() in {"true", "false", "yes", "no"}, "invalid boolean")
    return raw.casefold() in {"true", "yes"}


def token_set(value):
    stop = {"a", "an", "the", "is", "are", "what", "how", "do", "does",
            "i", "my", "to", "for", "of", "can", "and", "in"}
    return set(re.findall(r"\w+", value.casefold())) - stop


def score(question, candidate):
    left, right = token_set(question), token_set(candidate)
    return len(left & right) / len(left | right) if left | right else 0.0


class Validator:
    """One validation boundary for input, every handoff, and final output."""

    stages = ("extract", "faq", "journey", "deep")

    def __init__(self, request):
        self.request = request
        self.input()

    def source(self, value):
        obj(value, ("document_id", "start", "end", "text"), "source")
        require(value["document_id"] in self.documents, "unknown source document")
        document = self.documents[value["document_id"]]["text"]
        start, end = value["start"], value["end"]
        require(type(start) is int and type(end) is int and
                0 <= start < end <= len(document), "invalid source span")
        require(value["text"] == document[start:end], "source text mismatch")

    def reference(self, value):
        obj(value, ("document_id", "start", "end"), "reference")
        require(isinstance(value["document_id"], str), "document_id must be a string")
        require(value["document_id"] in self.documents, "unknown reference document")
        document = self.documents[value["document_id"]]["text"]
        start, end = value["start"], value["end"]
        require(type(start) is int and type(end) is int and
                0 <= start < end <= len(document), "invalid reference span")

    def quoted(self, ref):
        return dict(ref, text=self.documents[ref["document_id"]]["text"]
                    [ref["start"]:ref["end"]])

    def input(self):
        r = self.request
        obj(r, ("schema_version", "synthetic", "documents", "extraction",
                "questions", "knowledge_base", "actions", "completed_actions",
                "research_questions", "claims"), "input")
        require(type(r["schema_version"]) is int and r["schema_version"] == 1,
                "unsupported schema_version")
        require(r["synthetic"] is True, "fixtures must be explicitly synthetic")
        self.documents = indexed(r["documents"], "documents")
        for document in r["documents"]:
            obj(document, ("id", "text"), "document")
            text(document["text"], "document.text", allow_empty=True)
        obj(r["extraction"], ("document_ids", "fields"), "extraction")
        strings(r["extraction"]["document_ids"], "extraction.document_ids")
        require(set(r["extraction"]["document_ids"]) <= self.documents.keys(),
                "unknown extraction document")
        self.fields = indexed(r["extraction"]["fields"], "fields")
        for field in self.fields.values():
            obj(field, ("id", "pattern", "type", "required"), "field")
            require(field["type"] in ("string", "integer", "boolean"), "unknown field type")
            require(type(field["required"]) is bool, "required must be boolean")
            text(field["pattern"], "field.pattern")
            try:
                pattern = re.compile(field["pattern"])
            except re.error as exc:
                raise ValidationError("invalid extraction regex: " + str(exc)) from exc
            require(pattern.groups == 1, "field pattern must have exactly one capture")
        self.questions = indexed(r["questions"], "questions")
        for question in self.questions.values():
            obj(question, ("id", "text", "required_fields"), "question")
            text(question["text"], "question.text")
            strings(question["required_fields"], "required_fields")
            require(set(question["required_fields"]) <= self.fields.keys(),
                    "unknown required field")
        self.kb = indexed(r["knowledge_base"], "knowledge_base")
        for entry in self.kb.values():
            obj(entry, ("id", "question", "source", "requires"), "knowledge_base entry")
            text(entry["question"], "knowledge_base.question")
            self.reference(entry["source"])
            require(isinstance(entry["requires"], dict), "requires must be an object")
            require(entry["requires"].keys() <= self.fields.keys(), "unknown KB field")
            for name, value in entry["requires"].items():
                expected = {"string": str, "integer": int, "boolean": bool}[self.fields[name]["type"]]
                require(type(value) is expected, "KB requirement type mismatch")
        require(isinstance(r["research_questions"], dict), "research_questions must be an object")
        for topic, question in r["research_questions"].items():
            text(topic, "research topic")
            text(question, "research question")
        self.actions = indexed(r["actions"], "actions")
        for action in self.actions.values():
            obj(action, ("id", "title", "priority", "requires_fields", "requires_answers",
                         "prerequisites", "research_topics"), "action")
            text(action["title"], "action.title")
            require(type(action["priority"]) is int, "priority must be integer")
            for key in ("requires_fields", "requires_answers", "prerequisites", "research_topics"):
                strings(action[key], "action." + key)
            require(set(action["requires_fields"]) <= self.fields.keys(), "unknown action field")
            require(set(action["requires_answers"]) <= self.questions.keys(), "unknown action answer")
            require(set(action["prerequisites"]) <= self.actions.keys(), "unknown prerequisite")
            require(set(action["research_topics"]) <= r["research_questions"].keys(),
                    "unknown research topic")
        # Kahn-style closure also rejects self-dependencies, without recursion limits.
        visited = set()
        while True:
            ready = {a["id"] for a in self.actions.values()
                     if set(a["prerequisites"]) <= visited}
            if ready <= visited:
                break
            visited |= ready
        require(visited == self.actions.keys(), "action prerequisite cycle")
        strings(r["completed_actions"], "completed_actions")
        require(set(r["completed_actions"]) <= self.actions.keys(), "unknown completed action")
        for action_id in r["completed_actions"]:
            require(set(self.actions[action_id]["prerequisites"]) <= set(r["completed_actions"]),
                    "completed actions lack prerequisite closure")
        self.claims = indexed(r["claims"], "claims")
        for claim in self.claims.values():
            obj(claim, ("id", "topic", "stance", "source"), "claim")
            text(claim["topic"], "claim.topic")
            require(claim["topic"] in r["research_questions"], "unknown claim topic")
            require(claim["stance"] in ("supports", "opposes"), "invalid claim stance")
            self.reference(claim["source"])

    def state(self, value, through):
        require(through in self.stages, "unknown stage")
        keys = self.stages[:self.stages.index(through) + 1]
        obj(value, ("schema_version", "synthetic", "status") + keys, "pipeline state")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "invalid state version")
        require(value["synthetic"] is True and value["status"] == "ok", "invalid state header")
        extraction = value["extract"]
        obj(extraction, ("values", "spans", "missing_fields", "required_missing"), "extract")
        obj(extraction["values"], self.fields, "extract.values")
        obj(extraction["spans"], self.fields, "extract.spans")
        missing = []
        for name, field in self.fields.items():
            field_value, span = extraction["values"][name], extraction["spans"][name]
            if field_value is None:
                require(span is None, "missing value cannot have a span")
                missing.append(name)
            else:
                expected = {"string": str, "integer": int, "boolean": bool}[field["type"]]
                require(type(field_value) is expected, "extracted value type mismatch")
                self.source(span)
                require(span["document_id"] in self.request["extraction"]["document_ids"],
                        "field came from an unselected document")
                require(same(convert(span["text"], field["type"]), field_value),
                        "field value does not match its source")
                document = self.documents[span["document_id"]]["text"]
                matching = False
                for match in re.finditer(field["pattern"], document):
                    if match.group(1) is None:
                        continue
                    raw = match.group(1)
                    start = match.start(1) + len(raw) - len(raw.lstrip())
                    if start == span["start"] and start + len(raw.strip()) == span["end"]:
                        matching = True
                        break
                require(matching, "source does not match field schema")
        require(extraction["missing_fields"] == missing, "incorrect missing_fields")
        require(extraction["required_missing"] ==
                [name for name in missing if self.fields[name]["required"]],
                "incorrect required_missing")
        require(same(extraction, extract_fields(self)),
                "extraction does not match the current input")
        if "faq" not in keys:
            return value
        obj(value["faq"], ("answers",), "faq")
        answers = indexed(value["faq"]["answers"], "answers")
        require(list(answers) == list(self.questions), "answer IDs/order mismatch")
        for answer in answers.values():
            obj(answer, ("id", "status", "answer", "source", "kb_id", "score",
                         "reason", "missing_fields"), "answer")
            require(same(answer, answer_question(self, extraction, self.questions[answer["id"]])),
                    "answer is not grounded in eligible retrieved evidence")
        if "journey" not in keys:
            return value
        journey = value["journey"]
        obj(journey, ("status", "completed_actions", "next_actions", "steps", "blocked"), "journey")
        require(same(journey, plan_journey(self, extraction, answers)), "invalid journey plan")
        if "deep" in keys:
            obj(value["deep"], ("topics", "synthesis", "disagreements", "unresolved_questions"), "deep")
            require(same(value["deep"], synthesize(self, journey)), "ungrounded research output")
        return value


def answer_question(validator, extraction, question):
    values = extraction["values"]
    missing = [name for name in question["required_fields"] if values[name] is None]
    answer = {"id": question["id"], "status": "abstained", "answer": None,
              "source": None, "kb_id": None, "score": 0.0,
              "reason": "missing_required_fields" if missing else "no_relevant_evidence",
              "missing_fields": missing}
    if missing:
        return answer
    candidates = []
    for entry in validator.kb.values():
        if all(same(values[name], expected) for name, expected in entry["requires"].items()):
            relevance = score(question["text"], entry["question"])
            if relevance >= 0.2:
                candidates.append((relevance, entry["id"], entry))
    if candidates:
        relevance, _, entry = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
        source = validator.quoted(entry["source"])
        answer.update(status="answered", answer=source["text"], source=source,
                      kb_id=entry["id"], score=round(relevance, 6), reason=None)
    return answer


def action_blockers(action, values, answers, completed):
    return {
        "missing_fields": [name for name in action["requires_fields"] if values[name] is None],
        "unanswered_questions": [name for name in action["requires_answers"]
                                 if answers[name]["status"] != "answered"],
        "unmet_prerequisites": [name for name in action["prerequisites"] if name not in completed],
    }


def plan_journey(validator, extraction, answers):
    completed = set(validator.request["completed_actions"])
    ordered = sorted(validator.actions.values(), key=lambda a: (-a["priority"], a["id"]))
    values = extraction["values"]

    def eligible(done):
        return [a for a in ordered if a["id"] not in done and
                not any(action_blockers(a, values, answers, done).values())]

    next_actions = [a["id"] for a in eligible(completed)]
    blocked = []
    for action in ordered:
        if action["id"] not in completed:
            reasons = action_blockers(action, values, answers, completed)
            if any(reasons.values()):
                blocked.append(dict(action_id=action["id"], **reasons))
    steps = []
    for number in (1, 2):
        available = eligible(completed)
        if not available:
            break
        action = available[0]
        steps.append({"step": number, "action_id": action["id"], "title": action["title"],
                      "prerequisites": list(action["prerequisites"]),
                      "research_topics": list(action["research_topics"])})
        completed.add(action["id"])
    return {"status": "ready" if len(steps) == 2 else "partial" if steps else "blocked",
            "completed_actions": list(validator.request["completed_actions"]),
            "next_actions": next_actions, "steps": steps, "blocked": blocked}


def synthesize(validator, journey):
    topics = list(dict.fromkeys(topic for step in journey["steps"] for topic in step["research_topics"]))
    synthesis, disagreements, unresolved = [], [], []
    for topic in topics:
        claims = sorted((c for c in validator.claims.values() if c["topic"] == topic),
                        key=lambda c: c["id"])
        evidence = [{"claim_id": c["id"], "stance": c["stance"],
                     "source": validator.quoted(c["source"])} for c in claims]
        documents = sorted({c["source"]["document_id"] for c in claims})
        support = [c["id"] for c in claims if c["stance"] == "supports"]
        opposition = [c["id"] for c in claims if c["stance"] == "opposes"]
        assessment = ("disputed" if support and opposition else "supported" if support
                      else "opposed" if opposition else "insufficient_evidence")
        synthesis.append({"topic": topic, "question": validator.request["research_questions"][topic],
                          "assessment": assessment, "document_ids": documents, "evidence": evidence})
        reasons = []
        if not claims:
            reasons.append("no_evidence")
        elif len(documents) < 2:
            reasons.append("single_document_only")
        if support and opposition:
            disagreements.append({"topic": topic, "supporting_claims": support, "opposing_claims": opposition})
            reasons.append("conflicting_evidence")
        if reasons:
            unresolved.append({"topic": topic, "question": validator.request["research_questions"][topic],
                               "reasons": reasons})
    if not topics:
        unresolved.append({"topic": None, "question": "Which research topic should follow an eligible action?",
                           "reasons": ["no_journey_research_topics"]})
    return {"topics": topics, "synthesis": synthesis, "disagreements": disagreements,
            "unresolved_questions": unresolved}


def extract_fields(v):
    values, spans = {}, {}
    for name, field in v.fields.items():
        values[name], spans[name] = None, None
        found = False
        for document_id in v.request["extraction"]["document_ids"]:
            document = v.documents[document_id]["text"]
            for match in re.finditer(field["pattern"], document):
                if match.group(1) is None or not match.group(1).strip():
                    continue
                raw = match.group(1)
                start = match.start(1) + len(raw) - len(raw.lstrip())
                try:
                    value = convert(raw.strip(), field["type"])
                except ValidationError:
                    continue
                values[name] = value
                spans[name] = {"document_id": document_id, "start": start,
                               "end": start + len(raw.strip()), "text": raw.strip()}
                found = True
                break
            if found:
                break
    missing = [name for name, value in values.items() if value is None]
    return {"values": values, "spans": spans, "missing_fields": missing,
            "required_missing": [name for name in missing if v.fields[name]["required"]]}


def extract(request):
    v = Validator(request)
    state = {"schema_version": 1, "synthetic": True, "status": "ok",
             "extract": extract_fields(v)}
    return v.state(state, "extract")


def faq(request, previous):
    v = Validator(request)
    v.state(previous, "extract")
    state = copy.deepcopy(previous)
    state["faq"] = {"answers": [answer_question(v, state["extract"], question)
                                for question in v.questions.values()]}
    return v.state(state, "faq")


def journey(request, previous):
    v = Validator(request)
    v.state(previous, "faq")
    state = copy.deepcopy(previous)
    state["journey"] = plan_journey(v, state["extract"], indexed(state["faq"]["answers"], "answers"))
    return v.state(state, "journey")


def deep(request, previous):
    v = Validator(request)
    v.state(previous, "journey")
    state = copy.deepcopy(previous)
    state["deep"] = synthesize(v, state["journey"])
    return v.state(state, "deep")


def run_pipeline(request):
    return deep(request, journey(request, faq(request, extract(request))))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            request = json.load(handle, object_pairs_hook=unique_object, parse_constant=invalid_constant)
        result = run_pipeline(request)
        output = json.dumps(result, ensure_ascii=True, allow_nan=False)
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError,
            KeyError, RecursionError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
