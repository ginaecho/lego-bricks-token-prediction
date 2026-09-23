"""Synthetic, deterministic journey -> research -> feedback -> FAQ reference CLI.

Only Python's standard library is used. All stages exchange versioned envelopes;
Validator checks input, prerequisites, source offsets, and handoff consistency.
"""

import json
import re
import sys
from pathlib import Path


VERSION = 1
STAGES = ("journey", "normal", "feedback", "faq")
STOP_WORDS = frozenset(
    "a an and are as at be by can do for from how i in is it of on or "
    "our the this to was we what when where which with you your".split()
)


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP_WORDS


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def indexed(records):
    return {record["id"]: record for record in records}


def sentence_spans(text):
    for match in re.finditer(r"[^.!?]+[.!?]?", text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end


def citation(collection, record, start, end):
    return {
        "collection": collection,
        "id": record["id"],
        "source": record.get("source", record["id"]),
        "start": start,
        "end": end,
        "quote": record["text"][start:end],
    }


def envelope(stage, data):
    return {"schema_version": VERSION, "status": "ok", "stage": stage, "data": data}


def faq_status(answers):
    if not answers:
        return "no_feedback"
    answered = sum(answer["status"] == "answered" for answer in answers)
    return "answered" if answered == len(answers) else "partial" if answered else "abstained"


def eligible(actions, completed):
    return [
        action for action in actions
        if action["id"] not in completed
        and set(action["prerequisites"]).issubset(completed)
    ]


def ranked_actions(actions, goals):
    goal_tokens = tokens(" ".join(goals))
    return sorted(
        actions,
        key=lambda action: (
            -len(tokens(" ".join(action["tags"])) & goal_tokens), action["id"]
        ),
    )


def retrieval(query, records, minimum=1, required_tokens=None):
    """Return the best qualifying sentence per source, in stable score/id order."""
    query_tokens = tokens(query)
    results = []
    for record in records:
        candidates = []
        for start, end in sentence_spans(record["text"]):
            words = tokens(record["text"][start:end])
            score = len(words & query_tokens)
            if score >= minimum and (
                required_tokens is None or words & required_tokens
            ):
                candidates.append((score, start, end))
        if candidates:
            score, start, end = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            results.append((score, record, start, end))
    return sorted(results, key=lambda item: (-item[0], item[1]["id"]))


class Validator:
    """One validation layer used before and after every stage."""

    def __init__(self, payload):
        self.input(payload)
        self.payload = payload
        self.actions = indexed(payload["actions"])
        self.collections = {
            name: indexed(payload[name])
            for name in ("passages", "feedback", "knowledge_base")
        }
        self.themes = indexed(payload["themes"])

    @staticmethod
    def object(value, keys, label):
        require(isinstance(value, dict), f"{label} must be an object")
        require(set(value) == set(keys), f"{label} has missing or unknown fields")

    @staticmethod
    def text(value, label):
        require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
        require(len(value) <= 20000, f"{label} exceeds the text limit")

    @classmethod
    def strings(cls, values, label, nonempty=False):
        require(isinstance(values, list), f"{label} must be a list")
        require(len(values) <= 200, f"{label} exceeds the list limit")
        require(not nonempty or bool(values), f"{label} must not be empty")
        for value in values:
            cls.text(value, label)
        require(len(values) == len(set(values)), f"{label} has duplicates")

    @classmethod
    def records(cls, records, keys, label):
        require(isinstance(records, list) and len(records) <= 200,
                f"{label} must be a list with at most 200 records")
        ids = []
        for record in records:
            cls.object(record, keys, label)
            cls.text(record["id"], f"{label}.id")
            ids.append(record["id"])
        require(len(ids) == len(set(ids)), f"{label} has duplicate ids")

    @classmethod
    def input(cls, value):
        cls.object(value, (
            "schema_version", "synthetic", "profile", "actions", "passages",
            "feedback", "themes", "knowledge_base",
        ), "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == VERSION,
                "unsupported schema_version")
        require(value["synthetic"] is True, "fixture must be explicitly synthetic")
        cls.object(value["profile"], ("completed", "goals"), "profile")
        cls.strings(value["profile"]["completed"], "completed")
        cls.strings(value["profile"]["goals"], "goals", nonempty=True)
        require(bool(tokens(" ".join(value["profile"]["goals"]))), "goals need searchable words")
        cls.records(value["actions"], ("id", "title", "prerequisites", "tags", "research_query"), "actions")
        action_ids = {action["id"] for action in value["actions"]}
        known = action_ids | set(value["profile"]["completed"])
        for action in value["actions"]:
            cls.text(action["title"], "action.title")
            cls.text(action["research_query"], "action.research_query")
            require(bool(tokens(action["research_query"])), "research_query needs searchable words")
            cls.strings(action["prerequisites"], "prerequisites")
            cls.strings(action["tags"], "tags", nonempty=True)
            require(set(action["prerequisites"]) <= known, "unknown prerequisite")
            require(action["id"] not in action["prerequisites"], "self prerequisite")
        for name in ("passages", "knowledge_base"):
            cls.records(value[name], ("id", "source", "text"), name)
            for record in value[name]:
                cls.text(record["source"], f"{name}.source")
                cls.text(record["text"], f"{name}.text")
        cls.records(value["feedback"], ("id", "text", "passage_ids"), "feedback")
        passage_ids = {record["id"] for record in value["passages"]}
        for record in value["feedback"]:
            cls.text(record["text"], "feedback.text")
            require(bool(normalized(record["text"])), "feedback needs words")
            cls.strings(record["passage_ids"], "passage_ids")
            require(set(record["passage_ids"]) <= passage_ids, "unknown feedback passage")
        cls.records(value["themes"], ("id", "keywords", "question"), "themes")
        for theme in value["themes"]:
            cls.strings(theme["keywords"], "keywords", nonempty=True)
            cls.text(theme["question"], "question")
            require(bool(tokens(" ".join(theme["keywords"]))), "theme needs searchable keywords")
            require(bool(tokens(theme["question"])), "question needs searchable words")

    def cite(self, value, collection):
        self.object(value, ("collection", "id", "source", "start", "end", "quote"), "citation")
        require(value["collection"] == collection, "citation collection mismatch")
        self.text(value["id"], "citation.id")
        require(value["id"] in self.collections[collection], "unknown citation id")
        record = self.collections[collection][value["id"]]
        start, end = value["start"], value["end"]
        require(type(start) is int and type(end) is int, "citation offsets must be integers")
        require(0 <= start < end <= len(record["text"]), "citation offsets out of range")
        require(value["source"] == record.get("source", record["id"]), "citation source mismatch")
        require(value["quote"] == record["text"][start:end], "citation quote mismatch")

    def stage(self, value, expected, previous=None):
        self.object(value, ("schema_version", "status", "stage", "data"), "stage")
        require(type(value["schema_version"]) is int and value["schema_version"] == VERSION,
                "stage schema version mismatch")
        require(value["stage"] == expected and value["status"] == "ok", "stage envelope mismatch")
        require(expected in STAGES, "unknown stage")
        position = STAGES.index(expected)
        if position:
            require(isinstance(previous, dict), "validated previous stage required")
            require(previous.get("stage") == STAGES[position - 1]
                    and previous.get("status") == "ok"
                    and previous.get("schema_version") == VERSION,
                    "incorrect handoff stage")
        else:
            require(previous is None, "journey cannot have a previous stage")
        getattr(self, f"validate_{expected}")(value["data"], previous)
        return value

    def validate_journey(self, data, previous):
        self.object(data, ("next_actions", "steps"), "journey.data")
        completed = set(self.payload["profile"]["completed"])
        expected_next = [
            action["id"] for action in ranked_actions(
                eligible(self.payload["actions"], completed), self.payload["profile"]["goals"]
            )
        ]
        require(data["next_actions"] == expected_next, "invalid next actions")
        self.records(data["steps"], ("id", "position", "title", "research_query"), "journey.steps")
        require(len(data["steps"]) == 2, "journey must have exactly two steps")
        for position, step in enumerate(data["steps"], 1):
            require(type(step["position"]) is int and step["position"] == position, "invalid step order")
            require(step["id"] in self.actions, "unknown journey action")
            action = self.actions[step["id"]]
            require(action["id"] not in completed, "journey repeats a completed action")
            require(set(action["prerequisites"]) <= completed, "journey prerequisite not met")
            require(step["title"] == action["title"]
                    and step["research_query"] == action["research_query"], "journey metadata mismatch")
            completed.add(action["id"])

    def validate_normal(self, data, previous):
        self.object(data, ("findings",), "normal.data")
        findings = data["findings"]
        require(isinstance(findings, list) and len(findings) == 2, "two research findings required")
        for finding, step in zip(findings, previous["data"]["steps"]):
            self.object(finding, ("action_id", "query", "status", "text", "citations"), "finding")
            require(finding["action_id"] == step["id"]
                    and finding["query"] == step["research_query"], "journey/research handoff mismatch")
            require(isinstance(finding["citations"], list), "finding citations must be a list")
            matches = retrieval(finding["query"], self.payload["passages"])
            if matches:
                _, record, start, end = matches[0]
                expected = citation("passages", record, start, end)
                require(finding["status"] == "found"
                        and finding["citations"] == [expected]
                        and finding["text"] == expected["quote"], "research must use best exact extract")
                self.cite(finding["citations"][0], "passages")
            else:
                require(finding["status"] == "no_evidence" and finding["text"] == ""
                        and finding["citations"] == [], "unsupported research must abstain")

    def validate_feedback(self, data, previous):
        self.object(data, (
            "research_passage_ids", "groups", "themes", "excluded_feedback_ids",
            "unclassified_group_ids",
        ), "feedback.data")
        cited_ids = sorted({
            cite["id"] for finding in previous["data"]["findings"]
            for cite in finding["citations"]
        })
        require(data["research_passage_ids"] == cited_ids, "research/feedback handoff mismatch")
        selected = sorted(
            (record for record in self.payload["feedback"]
             if set(record["passage_ids"]) & set(cited_ids)), key=lambda record: record["id"]
        )
        expected_groups = {}
        for record in selected:
            expected_groups.setdefault(normalized(record["text"]), []).append(record)
        groups = data["groups"]
        self.records(groups, ("id", "member_ids", "passage_ids", "excerpt"), "feedback.groups")
        require(len(groups) == len(expected_groups), "incorrect deduplication group count")
        expected_members = list(expected_groups.values())
        for group, members in zip(groups, expected_members):
            representative = members[0]
            require(group["id"] == representative["id"]
                    and group["member_ids"] == [item["id"] for item in members],
                    "deduplication membership mismatch")
            links = sorted({link for item in members for link in item["passage_ids"]} & set(cited_ids))
            require(group["passage_ids"] == links, "feedback provenance mismatch")
            require(group["excerpt"] == citation("feedback", representative, 0, len(representative["text"])),
                    "feedback excerpt must be exact and complete")
            self.cite(group["excerpt"], "feedback")
        excluded = sorted(record["id"] for record in self.payload["feedback"] if record not in selected)
        require(data["excluded_feedback_ids"] == excluded, "excluded feedback mismatch")
        self.records(data["themes"], ("id", "question", "keywords", "group_ids", "unique_count", "excerpts"),
                     "feedback.themes")
        expected_themes = []
        classified = set()
        for theme in sorted(self.payload["themes"], key=lambda item: item["id"]):
            support = [
                group for group in groups
                if tokens(group["excerpt"]["quote"]) & tokens(" ".join(theme["keywords"]))
            ]
            if support:
                expected_themes.append({
                    "id": theme["id"], "question": theme["question"], "keywords": theme["keywords"],
                    "group_ids": [group["id"] for group in support],
                    "unique_count": len(support), "excerpts": [group["excerpt"] for group in support],
                })
                classified.update(group["id"] for group in support)
        require(data["themes"] == expected_themes, "theme support mismatch")
        require(data["unclassified_group_ids"] == [group["id"] for group in groups if group["id"] not in classified],
                "unclassified groups mismatch")

    def validate_faq(self, data, previous):
        self.object(data, ("status", "answers"), "faq.data")
        themes = previous["data"]["themes"]
        answers = data["answers"]
        require(isinstance(answers, list) and len(answers) == len(themes), "FAQ theme count mismatch")
        for answer, theme in zip(answers, themes):
            self.object(answer, (
                "theme_id", "question", "supporting_feedback_ids", "status", "answer", "citations", "reason",
            ), "faq.answer")
            require(answer["theme_id"] == theme["id"] and answer["question"] == theme["question"]
                    and answer["supporting_feedback_ids"] == theme["group_ids"],
                    "feedback/FAQ handoff mismatch")
            candidates = retrieval(
                theme["question"] + " " + " ".join(theme["keywords"]),
                self.payload["knowledge_base"], minimum=2,
                required_tokens=tokens(" ".join(theme["keywords"])),
            )
            if candidates:
                _, record, start, end = candidates[0]
                expected = citation("knowledge_base", record, start, end)
                require(answer["status"] == "answered" and answer["answer"] == expected["quote"]
                        and answer["citations"] == [expected] and answer["reason"] is None,
                        "FAQ answer is not grounded in the strongest source")
                self.cite(answer["citations"][0], "knowledge_base")
            else:
                require(answer["status"] == "abstained" and answer["answer"] is None
                        and answer["citations"] == []
                        and answer["reason"] == "No sufficiently relevant knowledge-base sentence.",
                        "FAQ must explicitly abstain")
        require(data["status"] == faq_status(answers), "FAQ status mismatch")


def journey_stage(validator):
    payload = validator.payload
    completed = set(payload["profile"]["completed"])
    available = ranked_actions(eligible(payload["actions"], completed), payload["profile"]["goals"])
    chosen = None
    for first in available:
        second = ranked_actions(
            eligible(payload["actions"], completed | {first["id"]}), payload["profile"]["goals"]
        )
        if second:
            chosen = (first, second[0])
            break
    require(chosen is not None, "no prerequisite-valid two-step journey exists")
    data = {
        "next_actions": [action["id"] for action in available],
        "steps": [
            {"id": action["id"], "position": position,
             "title": action["title"], "research_query": action["research_query"]}
            for position, action in enumerate(chosen, 1)
        ],
    }
    return validator.stage(envelope("journey", data), "journey")


def normal_stage(validator, journey):
    validator.stage(journey, "journey")
    findings = []
    for step in journey["data"]["steps"]:
        matches = retrieval(step["research_query"], validator.payload["passages"])
        cites = []
        if matches:
            _, record, start, end = matches[0]
            cites = [citation("passages", record, start, end)]
        findings.append({
            "action_id": step["id"], "query": step["research_query"],
            "status": "found" if cites else "no_evidence",
            "text": cites[0]["quote"] if cites else "", "citations": cites,
        })
    return validator.stage(envelope("normal", {"findings": findings}), "normal", journey)


def feedback_stage(validator, normal, journey):
    validator.stage(journey, "journey")
    validator.stage(normal, "normal", journey)
    passage_ids = sorted({
        cite["id"] for finding in normal["data"]["findings"] for cite in finding["citations"]
    })
    deduplicated = {}
    excluded = []
    for record in sorted(validator.payload["feedback"], key=lambda item: item["id"]):
        links = set(record["passage_ids"]) & set(passage_ids)
        if not links:
            excluded.append(record["id"])
            continue
        key = normalized(record["text"])
        if key not in deduplicated:
            deduplicated[key] = {
                "id": record["id"], "member_ids": [], "passage_ids": [],
                "excerpt": citation("feedback", record, 0, len(record["text"])),
            }
        group = deduplicated[key]
        group["member_ids"].append(record["id"])
        group["passage_ids"] = sorted(set(group["passage_ids"]) | links)
    groups = list(deduplicated.values())
    themes = []
    classified = set()
    for theme in sorted(validator.payload["themes"], key=lambda item: item["id"]):
        support = [
            group for group in groups
            if tokens(group["excerpt"]["quote"]) & tokens(" ".join(theme["keywords"]))
        ]
        if support:
            themes.append({
                "id": theme["id"], "question": theme["question"], "keywords": theme["keywords"],
                "group_ids": [group["id"] for group in support], "unique_count": len(support),
                "excerpts": [group["excerpt"] for group in support],
            })
            classified.update(group["id"] for group in support)
    data = {
        "research_passage_ids": passage_ids, "groups": groups, "themes": themes,
        "excluded_feedback_ids": excluded,
        "unclassified_group_ids": [group["id"] for group in groups if group["id"] not in classified],
    }
    return validator.stage(envelope("feedback", data), "feedback", normal)


def faq_stage(validator, feedback, normal, journey):
    validator.stage(journey, "journey")
    validator.stage(normal, "normal", journey)
    validator.stage(feedback, "feedback", normal)
    answers = []
    for theme in feedback["data"]["themes"]:
        matches = retrieval(
            theme["question"] + " " + " ".join(theme["keywords"]),
            validator.payload["knowledge_base"], minimum=2,
            required_tokens=tokens(" ".join(theme["keywords"])),
        )
        cites = []
        if matches:
            _, record, start, end = matches[0]
            cites = [citation("knowledge_base", record, start, end)]
        answers.append({
            "theme_id": theme["id"], "question": theme["question"],
            "supporting_feedback_ids": theme["group_ids"],
            "status": "answered" if cites else "abstained",
            "answer": cites[0]["quote"] if cites else None, "citations": cites,
            "reason": None if cites else "No sufficiently relevant knowledge-base sentence.",
        })
    return validator.stage(envelope("faq", {
        "status": faq_status(answers), "answers": answers,
    }), "faq", feedback)


def run_pipeline(payload):
    validator = Validator(payload)
    journey = journey_stage(validator)
    normal = normal_stage(validator, journey)
    feedback = feedback_stage(validator, normal, journey)
    faq = faq_stage(validator, feedback, normal, journey)
    return {
        "schema_version": VERSION, "status": "ok", "synthetic": True,
        "stages": [journey, normal, feedback, faq],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError(f"nonstandard JSON number: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        text = Path(argv[0]).read_text(encoding="utf-8-sig")
        payload = json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
        result = run_pipeline(payload)
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
