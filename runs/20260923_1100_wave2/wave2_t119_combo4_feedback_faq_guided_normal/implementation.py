"""Synthetic, deterministic customer-insight pipeline; Python standard library only."""

import json
import re
import sys


BUILD_ID = "wave2_t119_combo4_feedback_faq_guided_normal"
STAGES = ("feedback", "faq", "guided", "normal")
STOP = set("a an the is are to of for and or in on how do i my with can "
           "what should please setup set up use your".split())


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unknown fields")


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), location + " must be nonempty text")


def strings(value, location):
    require(isinstance(value, list), location + " must be an array")
    for item in value:
        text(item, location)
    require(len(value) == len(set(value)), location + " contains duplicates")


def records(value, fields, location):
    require(isinstance(value, list), location + " must be an array")
    seen = set()
    for row in value:
        keys(row, fields, location)
        text(row["id"], location + ".id")
        require(row["id"] not in seen, location + " has duplicate ids")
        seen.add(row["id"])


def tokens(value):
    return set(re.findall(r"\w+", value.casefold())) - STOP


def validate_input(value):
    keys(value, ("schema_version", "synthetic", "feedback", "themes", "knowledge",
                 "steps", "completed_steps", "research_sources"), "input")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "schema_version must be 1")
    require(value["synthetic"] is True, "fixtures must be labeled synthetic")
    records(value["feedback"], ("id", "text"), "feedback")
    records(value["themes"], ("id", "keywords", "question"), "themes")
    for theme in value["themes"]:
        strings(theme["keywords"], "theme.keywords")
        require(bool(theme["keywords"]) and all(tokens(k) for k in theme["keywords"]),
                "keywords must contain searchable words")
        text(theme["question"], "theme.question")
    theme_ids = {t["id"] for t in value["themes"]}
    for field in ("feedback", "knowledge", "research_sources"):
        if field != "feedback":
            records(value[field], ("id", "title", "text"), field)
        for row in value[field]:
            text(row["text"], field + ".text")
            if field != "feedback":
                text(row["title"], field + ".title")
    records(value["steps"], ("id", "title", "topic", "requires", "instructions"), "steps")
    steps = {s["id"]: s for s in value["steps"]}
    for step in steps.values():
        for field in ("title", "topic", "instructions"):
            text(step[field], "step." + field)
        require(step["topic"] in theme_ids, "unknown step topic")
        strings(step["requires"], "step.requires")
        require(set(step["requires"]) <= steps.keys(), "unknown prerequisite")
    # Kahn's algorithm validates the graph without recursion-depth sensitivity.
    visited = set()
    while len(visited) < len(steps):
        ready = {sid for sid, step in steps.items()
                 if sid not in visited and set(step["requires"]) <= visited}
        require(bool(ready), "prerequisite cycle")
        visited.update(ready)
    strings(value["completed_steps"], "completed_steps")
    require(set(value["completed_steps"]) <= steps.keys(), "unknown completed step")
    return value


def cite(source, start=0, end=None):
    end = len(source["text"]) if end is None else end
    return {"source_id": source["id"], "start": start, "end": end,
            "excerpt": source["text"][start:end]}


def validate_citation(citation, sources):
    keys(citation, ("source_id", "start", "end", "excerpt"), "citation")
    text(citation["source_id"], "citation.source_id")
    require(citation["source_id"] in sources, "unknown citation source")
    start, end = citation["start"], citation["end"]
    require(type(start) is int and type(end) is int, "citation offsets must be integers")
    original = sources[citation["source_id"]]["text"]
    require(0 <= start < end <= len(original), "citation offsets out of range")
    require(citation["excerpt"] == original[start:end], "citation excerpt mismatch")


def passages(source):
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", source["text"]):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - len(raw) + len(raw.rstrip())
        if start < end:
            yield cite(source, start, end)


def retrieve(query, sources, keywords=None):
    query_tokens = tokens(query)
    candidates = []
    for source in sources:
        for citation in passages(source):
            passage_tokens = tokens(citation["excerpt"])
            if keywords and not any(tokens(k) <= passage_tokens for k in keywords):
                continue
            score = len(query_tokens & passage_tokens)
            if score:
                candidates.append((-score, citation["source_id"], citation["start"], citation))
    return min(candidates, key=lambda c: c[:3])[3] if candidates else None


def packet(stage, data):
    index = STAGES.index(stage)
    return {"schema_version": 1, "stage": stage, "status": "ok",
            "input_stage": STAGES[index - 1] if index else "input", "data": data}


def validate_packet(value, stage, original, previous=None):
    keys(value, ("schema_version", "stage", "status", "input_stage", "data"), "stage")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "invalid stage schema")
    require(value["stage"] == stage and value["status"] == "ok", "invalid stage envelope")
    index = STAGES.index(stage)
    require(value["input_stage"] == (STAGES[index - 1] if index else "input"),
            "invalid stage lineage")
    require(index == 0 or (isinstance(previous, dict)
                          and previous.get("stage") == STAGES[index - 1]),
            "missing predecessor")
    data = value["data"]
    if stage == "feedback":
        keys(data, ("groups", "themes", "unmatched_feedback_ids"), "feedback output")
        records(data["groups"], ("id", "feedback_ids", "evidence"), "groups")
        sources = {f["id"]: f for f in original["feedback"]}
        covered = []
        for group in data["groups"]:
            strings(group["feedback_ids"], "group.feedback_ids")
            require(bool(group["feedback_ids"]), "empty feedback group")
            require(group["id"] == group["feedback_ids"][0], "invalid canonical feedback")
            require(set(group["feedback_ids"]) <= sources.keys(), "unknown feedback")
            validate_citation(group["evidence"], sources)
            require(group["evidence"] == cite(sources[group["id"]]), "invalid group evidence")
            canonical = normalize(sources[group["id"]]["text"])
            require(all(normalize(sources[i]["text"]) == canonical
                        for i in group["feedback_ids"]), "nonduplicate feedback group")
            covered.extend(group["feedback_ids"])
        require(sorted(covered) == sorted(sources), "feedback must be covered exactly once")
        records(data["themes"], ("id", "question", "keywords", "group_ids",
                                 "feedback_ids", "evidence"), "detected themes")
        definitions = {t["id"]: t for t in original["themes"]}
        groups = {g["id"]: g for g in data["groups"]}
        for theme in data["themes"]:
            require(theme["id"] in definitions, "unknown detected theme")
            definition = definitions[theme["id"]]
            require(theme["question"] == definition["question"]
                    and theme["keywords"] == definition["keywords"], "changed theme definition")
            strings(theme["group_ids"], "theme.group_ids")
            require(bool(theme["group_ids"]) and set(theme["group_ids"]) <= groups.keys(),
                    "invalid theme groups")
            require(theme["feedback_ids"] == [fid for gid in theme["group_ids"]
                                             for fid in groups[gid]["feedback_ids"]],
                    "theme feedback lineage mismatch")
            require(theme["evidence"] == [groups[gid]["evidence"] for gid in theme["group_ids"]],
                    "theme evidence mismatch")
        strings(data["unmatched_feedback_ids"], "unmatched_feedback_ids")
        matched = {fid for t in data["themes"] for fid in t["feedback_ids"]}
        require(set(data["unmatched_feedback_ids"]) == sources.keys() - matched,
                "unmatched feedback mismatch")
    elif stage == "faq":
        keys(data, ("answers",), "faq output")
        records(data["answers"], ("id", "question", "keywords", "feedback_ids",
                                  "feedback_evidence", "status", "answer", "citations"), "answers")
        themes = previous["data"]["themes"]
        require([a["id"] for a in data["answers"]] == [t["id"] for t in themes],
                "FAQ must cover all detected themes")
        sources = {s["id"]: s for s in original["knowledge"]}
        for answer, theme in zip(data["answers"], themes):
            for field in ("question", "keywords", "feedback_ids"):
                require(answer[field] == theme[field], "FAQ lineage mismatch")
            require(answer["feedback_evidence"] == theme["evidence"], "lost feedback evidence")
            validate_extraction(answer, "answer", sources)
            expected = retrieve(theme["question"] + " " + " ".join(theme["keywords"]),
                                original["knowledge"], theme["keywords"])
            require(answer["citations"] == ([expected] if expected else []),
                    "FAQ retrieval mismatch")
    elif stage == "guided":
        keys(data, ("steps", "progress"), "guided output")
        records(data["steps"], ("id", "title", "topic", "requires", "instructions", "status",
                                "feedback_ids", "faq_citations", "research_query"), "guided steps")
        definitions = {s["id"]: s for s in original["steps"]}
        answers = {a["id"]: a for a in previous["data"]["answers"]}
        require({s["id"] for s in data["steps"]} == selected_steps(original, answers),
                "guided selection mismatch")
        complete = set(original["completed_steps"])
        for step in data["steps"]:
            definition = definitions[step["id"]]
            for field in ("title", "topic", "requires", "instructions"):
                require(step[field] == definition[field], "changed onboarding definition")
            answer = answers.get(step["topic"])
            supported = answer is not None and answer["status"] == "answered"
            require(step["status"] == step_status(step, complete, supported),
                    "invalid onboarding status")
            require(step["feedback_ids"] == (answer["feedback_ids"] if answer else [])
                    and step["faq_citations"] == (answer["citations"] if answer else []),
                    "guided lineage mismatch")
            require(step["research_query"] == research_query(step, answer), "changed research query")
        count = sum(s["status"] == "completed" for s in data["steps"])
        require(data["progress"] == {"completed": count, "total": len(data["steps"])},
                "invalid progress")
    else:
        keys(data, ("findings",), "research output")
        records(data["findings"], ("id", "feedback_ids", "faq_citations", "status",
                                   "finding", "citations"), "findings")
        actionable = [s for s in previous["data"]["steps"] if s["status"] in ("ready", "completed")]
        require([r["id"] for r in data["findings"]] == [s["id"] for s in actionable],
                "research selection mismatch")
        sources = {s["id"]: s for s in original["research_sources"]}
        for result, step in zip(data["findings"], actionable):
            require(result["feedback_ids"] == step["feedback_ids"]
                    and result["faq_citations"] == step["faq_citations"], "research lineage mismatch")
            validate_extraction(result, "finding", sources)
            expected = retrieve(step["research_query"], original["research_sources"])
            require(result["citations"] == ([expected] if expected else []),
                    "research retrieval mismatch")
    return value


def validate_extraction(row, field, sources):
    require(row["status"] in ("answered", "abstained"), "invalid extraction status")
    require(isinstance(row["citations"], list), "citations must be an array")
    if row["status"] == "abstained":
        require(row[field] is None and row["citations"] == [], "abstention must not invent content")
    else:
        require(len(row["citations"]) == 1, "extractive answers need one citation")
        validate_citation(row["citations"][0], sources)
        require(row[field] == row["citations"][0]["excerpt"], "answer must be exact extraction")


def normalize(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def feedback_stage(original):
    grouped = {}
    for row in original["feedback"]:
        # Punctuation-only messages stay distinct rather than collapsing to an empty key.
        key = normalize(row["text"]) or row["text"].strip()
        if key not in grouped:
            grouped[key] = {"id": row["id"], "feedback_ids": [], "evidence": cite(row)}
        grouped[key]["feedback_ids"].append(row["id"])
    groups = list(grouped.values())
    themes = []
    for definition in original["themes"]:
        matching = [g for g in groups if any(tokens(k) <= tokens(g["evidence"]["excerpt"])
                                            for k in definition["keywords"])]
        if matching:
            themes.append({**definition, "group_ids": [g["id"] for g in matching],
                           "feedback_ids": [i for g in matching for i in g["feedback_ids"]],
                           "evidence": [g["evidence"] for g in matching]})
    matched = {i for t in themes for i in t["feedback_ids"]}
    return packet("feedback", {"groups": groups, "themes": themes,
                              "unmatched_feedback_ids": [f["id"] for f in original["feedback"]
                                                         if f["id"] not in matched]})


def faq_stage(original, previous):
    validate_packet(previous, "feedback", original)
    answers = []
    for theme in previous["data"]["themes"]:
        evidence = retrieve(theme["question"] + " " + " ".join(theme["keywords"]),
                            original["knowledge"], theme["keywords"])
        answers.append({"id": theme["id"], "question": theme["question"],
                        "keywords": theme["keywords"], "feedback_ids": theme["feedback_ids"],
                        "feedback_evidence": theme["evidence"],
                        "status": "answered" if evidence else "abstained",
                        "answer": evidence["excerpt"] if evidence else None,
                        "citations": [evidence] if evidence else []})
    return packet("faq", {"answers": answers})


def selected_steps(original, answers):
    selected = {s["id"] for s in original["steps"] if s["topic"] in answers}
    while True:
        expanded = selected | {r for s in original["steps"] if s["id"] in selected for r in s["requires"]}
        if expanded == selected:
            return selected
        selected = expanded


def step_status(step, completed, supported):
    prerequisites_met = set(step["requires"]) <= completed
    if step["id"] in completed:
        require(prerequisites_met and supported, "completed step lacks prerequisites or grounded FAQ")
        return "completed"
    if not prerequisites_met:
        return "blocked"
    return "ready" if supported else "needs_answer"


def research_query(step, answer):
    return " ".join([step["title"], step["instructions"],
                     " ".join(answer["keywords"]) if answer else "",
                     answer["answer"] or "" if answer else ""]).strip()


def guided_stage(original, previous, feedback):
    validate_packet(previous, "faq", original, feedback)
    answers = {a["id"]: a for a in previous["data"]["answers"]}
    selected = selected_steps(original, answers)
    completed = set(original["completed_steps"])
    require(completed <= selected, "completed step is outside the active onboarding plan")
    rows = []
    for step in original["steps"]:
        if step["id"] not in selected:
            continue
        answer = answers.get(step["topic"])
        supported = answer is not None and answer["status"] == "answered"
        rows.append({**step, "status": step_status(step, completed, supported),
                     "feedback_ids": answer["feedback_ids"] if answer else [],
                     "faq_citations": answer["citations"] if answer else [],
                     "research_query": research_query(step, answer)})
    return packet("guided", {"steps": rows, "progress": {"completed": len(completed), "total": len(rows)}})


def normal_stage(original, previous, faq):
    validate_packet(previous, "guided", original, faq)
    findings = []
    for step in previous["data"]["steps"]:
        if step["status"] not in ("ready", "completed"):
            continue
        evidence = retrieve(step["research_query"], original["research_sources"])
        findings.append({"id": step["id"], "feedback_ids": step["feedback_ids"],
                         "faq_citations": step["faq_citations"],
                         "status": "answered" if evidence else "abstained",
                         "finding": evidence["excerpt"] if evidence else None,
                         "citations": [evidence] if evidence else []})
    return packet("normal", {"findings": findings})


def run(value):
    validate_input(value)
    feedback = validate_packet(feedback_stage(value), "feedback", value)
    faq = validate_packet(faq_stage(value, feedback), "faq", value, feedback)
    guided = validate_packet(guided_stage(value, faq, feedback), "guided", value, faq)
    normal = validate_packet(normal_stage(value, guided, faq), "normal", value, guided)
    return {"schema_version": 1, "synthetic": True, "status": "ok", "build_id": BUILD_ID,
            "stages": [feedback, faq, guided, normal]}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=unique_object,
                              parse_constant=lambda x: (_ for _ in ()).throw(
                                  ValidationError("nonfinite JSON constant: " + x)))
        output = run(value)
    except (OSError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
