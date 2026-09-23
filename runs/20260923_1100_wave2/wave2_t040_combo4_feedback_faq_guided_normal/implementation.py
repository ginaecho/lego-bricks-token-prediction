"""Synthetic, deterministic feedback -> FAQ -> guided setup -> research pipeline.

Run: python -B implementation.py example_input.json
All stages use the same record envelope and validation layer. Answers and
findings are verbatim excerpts, not generated claims or entailment judgments.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, where):
    require(isinstance(value, dict), f"{where}: expected object")
    require(set(value) == set(expected), f"{where}: invalid fields")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()),
            f"{where}: expected nonempty string")


def strings(value, where, nonempty=False):
    require(isinstance(value, list), f"{where}: expected array")
    if nonempty:
        require(bool(value), f"{where}: must not be empty")
    for item in value:
        text(item, where)
    require(len(value) == len(set(value)), f"{where}: duplicate values")


def objects(value, expected, where):
    require(isinstance(value, list), f"{where}: expected array")
    ids = set()
    for item in value:
        fields(item, expected, where)
        text(item["id"], f"{where}.id")
        require(item["id"] not in ids, f"{where}: duplicate id")
        ids.add(item["id"])
    return ids


def validate_input(data):
    fields(data, ["schema_version", "fixture_label", "feedback", "themes",
                  "knowledge_base", "setup_steps", "completed_step_ids",
                  "research_sources"], "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    text(data["fixture_label"], "fixture_label")
    objects(data["feedback"], ["id", "text"], "feedback")
    for row in data["feedback"]:
        text(row["text"], "feedback.text")
    theme_ids = objects(data["themes"], ["id", "label", "keywords", "question"],
                        "themes")
    for theme in data["themes"]:
        text(theme["label"], "theme.label")
        text(theme["question"], "theme.question")
        strings(theme["keywords"], "theme.keywords", nonempty=True)
        require(all(tokens(word) for word in theme["keywords"]),
                "theme keywords need meaningful tokens")
    for collection in ("knowledge_base", "research_sources"):
        objects(data[collection], ["id", "title", "passages"], collection)
        for source in data[collection]:
            text(source["title"], f"{collection}.title")
            objects(source["passages"], ["id", "text"], f"{collection}.passages")
            for passage in source["passages"]:
                text(passage["text"], "passage.text")
    step_ids = objects(data["setup_steps"],
                       ["id", "instruction", "theme_ids", "prerequisites",
                        "research_query"], "setup_steps")
    for step in data["setup_steps"]:
        text(step["instruction"], "step.instruction")
        text(step["research_query"], "step.research_query")
        strings(step["theme_ids"], "step.theme_ids", nonempty=True)
        strings(step["prerequisites"], "step.prerequisites")
        require(set(step["theme_ids"]) <= theme_ids, "unknown step theme")
        require(set(step["prerequisites"]) <= step_ids, "unknown prerequisite")
    topological_steps(data["setup_steps"])
    strings(data["completed_step_ids"], "completed_step_ids")
    require(set(data["completed_step_ids"]) <= step_ids, "unknown completed step")
    completed = set(data["completed_step_ids"])
    for step in data["setup_steps"]:
        if step["id"] in completed:
            require(set(step["prerequisites"]) <= completed,
                    "completed step has incomplete prerequisite")
    return data


STOPWORDS = set("a an the and or to for of in on is it how do does i my we "
                "with can should please this that need help".split())


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold())) - STOPWORDS


def topological_steps(steps):
    ordered, done = [], set()
    pending = list(steps)
    while pending:
        available = [s for s in pending if set(s["prerequisites"]) <= done]
        require(bool(available), "setup prerequisites contain a cycle")
        for step in available:
            ordered.append(step)
            done.add(step["id"])
            pending.remove(step)
    return ordered


def citation(collection, source_id, passage_id, quote):
    return {"collection": collection, "source_id": source_id,
            "passage_id": passage_id, "start": 0, "end": len(quote),
            "quote": quote}


def source_index(data):
    result = {("feedback", f["id"], f["id"]): f["text"]
              for f in data["feedback"]}
    for collection in ("knowledge_base", "research_sources"):
        for source in data[collection]:
            for passage in source["passages"]:
                result[(collection, source["id"], passage["id"])] = passage["text"]
    return result


def validate_citation(value, data, collection):
    fields(value, ["collection", "source_id", "passage_id", "start", "end",
                   "quote"], "citation")
    require(value["collection"] == collection, "wrong citation collection")
    for key in ("collection", "source_id", "passage_id", "quote"):
        text(value[key], f"citation.{key}")
    key = (value["collection"], value["source_id"], value["passage_id"])
    index = source_index(data)
    require(key in index, "unknown citation source")
    start, end = value["start"], value["end"]
    require(type(start) is int and type(end) is int, "citation offsets must be integers")
    require(0 <= start < end <= len(index[key]), "citation offsets out of bounds")
    require(index[key][start:end] == value["quote"], "citation quote mismatch")


STATES = {"feedback": {"analyzed"}, "faq": {"answered", "abstained"},
          "guided": {"ready", "blocked", "completed"},
          "normal": {"found", "abstained"}}
COLLECTIONS = {"feedback": "feedback", "faq": "knowledge_base",
               "guided": "knowledge_base", "normal": "research_sources"}
DETAILS = {
    "feedback": {"feedback_ids", "theme_ids"},
    "faq": {"question", "reason"},
    "guided": {"prerequisites", "research_query", "blocking_reasons"},
    "normal": {"query", "setup_state", "reason"},
}


def record(identity, state, content, citations, upstream_ids, details):
    return {"id": identity, "state": state, "text": content,
            "citations": citations, "upstream_ids": upstream_ids,
            "details": details}


def envelope(stage, records):
    return {"schema_version": 1, "stage": stage, "status": "ok",
            "records": records}


def setup_reasons(step, answers, completed):
    reasons = []
    for theme in step["theme_ids"]:
        if theme not in answers:
            reasons.append(f"no_feedback_evidence:{theme}")
        elif answers[theme]["state"] != "answered":
            reasons.append(f"faq_abstained:{theme}")
    for dependency in step["prerequisites"]:
        if dependency not in completed:
            reasons.append(f"incomplete_prerequisite:{dependency}")
    return reasons


def validate_stage(output, stage, data, previous=None):
    fields(output, ["schema_version", "stage", "status", "records"], "stage")
    require(type(output["schema_version"]) is int and output["schema_version"] == 1,
            "invalid stage schema version")
    require(output["stage"] == stage and output["status"] == "ok", "invalid stage")
    objects(output["records"], ["id", "state", "text", "citations",
                               "upstream_ids", "details"], "stage.records")
    allowed = ({f["id"] for f in data["feedback"]} if previous is None else
               {r["id"] for r in previous["records"]})
    prior = {} if previous is None else {r["id"]: r for r in previous["records"]}
    for row in output["records"]:
        text(row["state"], "record state")
        require(row["state"] in STATES[stage], "invalid record state")
        require(isinstance(row["text"], str), "record text must be string")
        strings(row["upstream_ids"], "upstream_ids")
        require(set(row["upstream_ids"]) <= allowed, "unknown upstream id")
        require(isinstance(row["citations"], list), "citations must be array")
        for ref in row["citations"]:
            validate_citation(ref, data, COLLECTIONS[stage])
        fields(row["details"], DETAILS[stage], "details")
        if row["state"] in {"answered", "found", "analyzed"}:
            require(bool(row["citations"]), "supported result needs citations")
            require(row["text"] == row["citations"][0]["quote"],
                    "extractive result must equal cited excerpt")
        if row["state"] == "abstained":
            require(row["text"] == "" and row["citations"] == [],
                    "abstention cannot contain an answer")
            text(row["details"]["reason"], "abstention reason")
        if stage == "feedback":
            strings(row["details"]["feedback_ids"], "feedback_ids", nonempty=True)
            require(row["upstream_ids"] == row["details"]["feedback_ids"],
                    "feedback provenance mismatch")
            strings(row["details"]["theme_ids"], "theme_ids")
            require(set(row["details"]["theme_ids"]) <=
                    {t["id"] for t in data["themes"]}, "unknown output theme")
            require(row["id"] == row["upstream_ids"][0], "invalid feedback identity")
            original = {f["id"]: f["text"] for f in data["feedback"]}
            normalized = " ".join(row["text"].casefold().split())
            require(all(" ".join(original[i].casefold().split()) == normalized
                        for i in row["upstream_ids"]), "invalid deduplication group")
        elif stage == "faq":
            themes = {t["id"]: t for t in data["themes"]}
            require(row["id"] in themes, "unknown FAQ theme")
            require(row["details"]["question"] == themes[row["id"]]["question"],
                    "FAQ question mismatch")
            require(isinstance(row["details"]["reason"], str), "invalid FAQ reason")
            expected = [r["id"] for r in previous["records"]
                        if row["id"] in r["details"]["theme_ids"]]
            require(bool(expected) and row["upstream_ids"] == expected,
                    "FAQ feedback support mismatch")
        elif stage == "guided":
            steps = {s["id"]: s for s in data["setup_steps"]}
            require(row["id"] in steps, "unknown guided step")
            step = steps[row["id"]]
            completed = set(data["completed_step_ids"])
            reasons = setup_reasons(step, prior, completed)
            require(not (row["id"] in completed and reasons),
                    "unsupported completed step")
            expected_state = "blocked" if reasons else (
                "completed" if row["id"] in completed else "ready")
            require(row["state"] == expected_state, "guided state mismatch")
            require(row["text"] == step["instruction"], "guided instruction mismatch")
            require(row["details"] == {
                "prerequisites": step["prerequisites"],
                "research_query": step["research_query"],
                "blocking_reasons": reasons}, "guided details mismatch")
            require(row["upstream_ids"] == [
                t for t in step["theme_ids"] if t in prior], "guided FAQ links mismatch")
            expected_refs = []
            for identity in row["upstream_ids"]:
                for ref in prior[identity]["citations"]:
                    if ref not in expected_refs:
                        expected_refs.append(ref)
            require(row["citations"] == expected_refs, "guided citation propagation mismatch")
        elif stage == "normal":
            require(row["id"] in prior and row["upstream_ids"] == [row["id"]],
                    "research setup link mismatch")
            step = prior[row["id"]]
            require(row["details"]["query"] == step["details"]["research_query"]
                    and row["details"]["setup_state"] == step["state"],
                    "research setup propagation mismatch")
            require(isinstance(row["details"]["reason"], str), "invalid research reason")
            if step["state"] == "blocked":
                require(row["state"] == "abstained"
                        and row["details"]["reason"] == "setup_blocked",
                        "blocked setup cannot produce findings")
    return output


def analyze_feedback(data):
    groups = {}
    for item in data["feedback"]:
        key = " ".join(item["text"].casefold().split())
        groups.setdefault(key, []).append(item)
    rows = []
    for group in groups.values():
        first = group[0]
        words = tokens(first["text"])
        themes = [theme["id"] for theme in data["themes"]
                  if any(tokens(k) <= words for k in theme["keywords"])]
        ids = [f["id"] for f in group]
        rows.append(record(first["id"], "analyzed", first["text"],
                           [citation("feedback", first["id"], first["id"],
                                     first["text"])], ids,
                           {"feedback_ids": ids, "theme_ids": themes}))
    return validate_stage(envelope("feedback", rows), "feedback", data)


def retrieve(query, sources, collection):
    words = tokens(query)
    ranked = []
    for source in sources:
        for passage in source["passages"]:
            score = len(words & tokens(passage["text"]))
            if score:
                ranked.append((-score, source["id"], passage["id"], passage["text"]))
    if not ranked:
        return None
    _, source_id, passage_id, quote = sorted(ranked)[0]
    return citation(collection, source_id, passage_id, quote)


def answer_faq(data, feedback):
    validate_stage(feedback, "feedback", data)
    rows = []
    for theme in data["themes"]:
        support = [r for r in feedback["records"]
                   if theme["id"] in r["details"]["theme_ids"]]
        if not support:
            continue
        query = " ".join([theme["question"], *theme["keywords"],
                          *(r["text"] for r in support)])
        ref = retrieve(query, data["knowledge_base"], "knowledge_base")
        rows.append(record(theme["id"], "answered" if ref else "abstained",
                           ref["quote"] if ref else "", [ref] if ref else [],
                           [r["id"] for r in support],
                           {"question": theme["question"],
                            "reason": "" if ref else "no_matching_knowledge_passage"}))
    return validate_stage(envelope("faq", rows), "faq", data, feedback)


def guide_setup(data, faq, feedback):
    validate_stage(faq, "faq", data, feedback)
    answers = {r["id"]: r for r in faq["records"]}
    completed = set(data["completed_step_ids"])
    rows = []
    for step in topological_steps(data["setup_steps"]):
        reasons = setup_reasons(step, answers, completed)
        require(not (step["id"] in completed and reasons),
                f"completed step lacks validated support: {step['id']}")
        state = "blocked" if reasons else (
            "completed" if step["id"] in completed else "ready")
        refs = []
        for theme in step["theme_ids"]:
            for ref in answers.get(theme, {}).get("citations", []):
                if ref not in refs:
                    refs.append(ref)
        rows.append(record(step["id"], state, step["instruction"], refs,
                           [t for t in step["theme_ids"] if t in answers],
                           {"prerequisites": step["prerequisites"],
                            "research_query": step["research_query"],
                            "blocking_reasons": reasons}))
    result = validate_stage(envelope("guided", rows), "guided", data, faq)
    total = len(rows)
    result_progress = {"total": total, "completed": len(completed),
                       "ready": sum(r["state"] == "ready" for r in rows),
                       "blocked": sum(r["state"] == "blocked" for r in rows),
                       "fraction_complete": len(completed) / total if total else 1.0}
    return result, result_progress


def research(data, guided, faq):
    validate_stage(guided, "guided", data, faq)
    rows = []
    for step in guided["records"]:
        query = step["details"]["research_query"]
        ref = None if step["state"] == "blocked" else retrieve(
            query, data["research_sources"], "research_sources")
        reason = "" if ref else ("setup_blocked" if step["state"] == "blocked"
                                 else "no_matching_research_passage")
        rows.append(record(step["id"], "found" if ref else "abstained",
                           ref["quote"] if ref else "", [ref] if ref else [],
                           [step["id"]], {"query": query,
                                         "setup_state": step["state"],
                                         "reason": reason}))
    return validate_stage(envelope("normal", rows), "normal", data, guided)


def run_pipeline(data):
    validate_input(data)
    feedback = analyze_feedback(data)
    faq = answer_faq(data, feedback)
    guided, progress = guide_setup(data, faq, feedback)
    normal = research(data, guided, faq)
    return {"schema_version": 1, "status": "ok",
            "fixture_label": data["fixture_label"],
            "stages": [feedback, faq, guided, normal], "progress": progress}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        output = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError,
            RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error",
                          "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
