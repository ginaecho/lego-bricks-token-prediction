"""Synthetic reference pipeline. Run: python -B implementation.py example_input.json.

Research accepts explicit question-to-source quotations rather than inferring truth.
Setup completion gates feedback eligibility. Review checks literal document terms
and the upstream evidence, never certifies accuracy or regulatory compliance.
Only Python's standard library is used; no model/provider is required.
"""

import copy
import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(keys), f"{path}: expected keys {sorted(keys)}")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonblank string")


def strings(value, path, allowed=None):
    require(isinstance(value, list), f"{path}: expected array")
    for item in value:
        text(item, path)
        if allowed is not None:
            require(item in allowed, f"{path}: unknown reference {item}")
    require(len(set(value)) == len(value), f"{path}: duplicate entries")


def records(value, keys, path, nonempty=False):
    require(isinstance(value, list), f"{path}: expected array")
    require(not nonempty or bool(value), f"{path}: must not be empty")
    ids = set()
    for index, item in enumerate(value):
        obj(item, keys, f"{path}[{index}]")
        text(item["id"], f"{path}[{index}].id")
        require(item["id"] not in ids, f"{path}: duplicate id {item['id']}")
        ids.add(item["id"])
    return ids


def validate_input(data):
    """One strict validation layer shared by the CLI and all stage boundaries."""
    obj(data, {"schema_version", "synthetic", "research", "guided", "feedback", "review"}, "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version: expected integer 1")
    require(data["synthetic"] is True, "synthetic: fixtures must be explicitly labeled true")
    research = data["research"]
    obj(research, {"questions", "sources"}, "research")
    qids = records(research["questions"], {"id", "question"}, "questions", True)
    for question in research["questions"]:
        text(question["question"], "question")
    sids = records(research["sources"], {"id", "title", "text", "evidence"}, "sources")
    for source in research["sources"]:
        text(source["title"], "source.title")
        text(source["text"], "source.text")
        require(isinstance(source["evidence"], list), "source.evidence: expected array")
        seen = set()
        for evidence in source["evidence"]:
            obj(evidence, {"question_id", "quote", "stance"}, "evidence")
            text(evidence["question_id"], "evidence.question_id")
            require(evidence["question_id"] in qids, "evidence: unknown question")
            text(evidence["quote"], "evidence.quote")
            require(evidence["quote"] in source["text"], "evidence: quote not found verbatim")
            require(evidence["stance"] in ("supports", "opposes", "context"),
                    "evidence: invalid stance")
            key = (evidence["question_id"], evidence["quote"], evidence["stance"])
            require(key not in seen, "evidence: duplicate source annotation")
            seen.add(key)
    guided = data["guided"]
    obj(guided, {"fields", "steps"}, "guided")
    require(isinstance(guided["fields"], dict), "guided.fields: expected object")
    for key, value in guided["fields"].items():
        text(key, "field name")
        require(isinstance(value, str), "guided.fields: values must be strings")
    stepids = records(guided["steps"], {"id", "title", "depends_on", "question_ids", "required_fields"},
                      "steps")
    for step in guided["steps"]:
        text(step["title"], "step.title")
        strings(step["depends_on"], "step.depends_on", stepids)
        strings(step["question_ids"], "step.question_ids", qids)
        strings(step["required_fields"], "step.required_fields")
        require(step["id"] not in step["depends_on"], "step: self dependency")
    # Stable Kahn traversal validates cycles without a recursion-depth limit.
    remaining = {s["id"]: set(s["depends_on"]) for s in guided["steps"]}
    complete = set()
    while remaining:
        ready = [key for key, deps in remaining.items() if deps <= complete]
        require(bool(ready), "steps: dependency cycle")
        for key in ready:
            complete.add(key)
            del remaining[key]
    feedback = data["feedback"]
    obj(feedback, {"themes", "items"}, "feedback")
    themeids = records(feedback["themes"], {"id", "label", "keywords"}, "themes")
    require("unclassified" not in themeids, "themes: unclassified is reserved")
    for theme in feedback["themes"]:
        text(theme["label"], "theme.label")
        strings(theme["keywords"], "theme.keywords")
        require(bool(theme["keywords"]), "theme.keywords: must not be empty")
    records(feedback["items"], {"id", "step_id", "text"}, "feedback.items")
    for item in feedback["items"]:
        text(item["step_id"], "feedback.step_id")
        require(item["step_id"] in stepids, "feedback: unknown step")
        text(item["text"], "feedback.text")
    review = data["review"]
    obj(review, {"document", "requirements"}, "review")
    obj(review["document"], {"id", "text"}, "document")
    text(review["document"]["id"], "document.id")
    require(isinstance(review["document"]["text"], str), "document.text: expected string")
    records(review["requirements"], {"id", "description", "question_ids", "step_ids",
                                    "theme_ids", "required_terms"}, "requirements", True)
    for requirement in review["requirements"]:
        text(requirement["description"], "requirement.description")
        strings(requirement["question_ids"], "requirement.question_ids", qids)
        strings(requirement["step_ids"], "requirement.step_ids", stepids)
        strings(requirement["theme_ids"], "requirement.theme_ids", themeids | {"unclassified"})
        strings(requirement["required_terms"], "requirement.required_terms")
        require(any(requirement[k] for k in ("question_ids", "step_ids", "theme_ids", "required_terms")),
                "requirement: at least one check is required")
    return data


STAGES = ("research", "guided", "feedback", "review")


def validate_state(state, expected_stage):
    """Validate the entire cumulative handoff, including deterministic provenance.

    Recomputing bounded stage results makes modified, stale, or fabricated intermediate
    evidence invalid, not merely JSON-shaped. There are at most four stages.
    """
    require(expected_stage in STAGES, "unknown stage")
    obj(state, {"schema_version", "synthetic", "stage", "input", "results"}, "handoff")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "handoff: bad schema version")
    require(state["synthetic"] is True, "handoff: synthetic label missing")
    require(state["stage"] == expected_stage, "handoff: incorrect stage order")
    validate_input(state["input"])
    expected = {}
    for stage in STAGES[:STAGES.index(expected_stage) + 1]:
        expected[stage] = COMPUTERS[stage](state["input"], expected)
    require(isinstance(state["results"], dict), "handoff.results: expected object")
    # JSON comparison also distinguishes bool from int, unlike ordinary Python equality.
    require(json.dumps(state["results"], sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False),
            "handoff: results disagree with validated input or provenance")
    return state


def compute_research(data, previous):
    findings = []
    for question in data["research"]["questions"]:
        evidence = []
        for source in data["research"]["sources"]:
            for annotation in source["evidence"]:
                if annotation["question_id"] == question["id"]:
                    start = source["text"].find(annotation["quote"])
                    evidence.append({
                        "source_id": source["id"], "source_title": source["title"],
                        "quote": annotation["quote"], "start": start,
                        "end": start + len(annotation["quote"]), "stance": annotation["stance"],
                    })
        stances = {e["stance"] for e in evidence}
        status = ("conflicted" if {"supports", "opposes"} <= stances else
                  "answered" if stances & {"supports", "opposes"} else "unanswered")
        findings.append({
            "id": question["id"], "question": question["question"], "status": status,
            "evidence": evidence,
            "next_action": {"answered": "Use cited evidence, subject to source verification.",
                            "conflicted": "Resolve conflicting evidence before proceeding.",
                            "unanswered": "Collect direct supporting or opposing evidence."}[status],
        })
    return {"findings": findings, "limitations": "Source annotations are supplied, not independently verified."}


def compute_guided(data, previous):
    findings = {f["id"]: f for f in previous["research"]["findings"]}
    pending = list(data["guided"]["steps"])
    results = {}
    execution_order = []
    while pending:
        for step in list(pending):
            if not all(dep in results for dep in step["depends_on"]):
                continue
            reasons = []
            for dep in step["depends_on"]:
                if results[dep]["status"] != "completed":
                    reasons.append({"kind": "dependency", "id": dep})
            for qid in step["question_ids"]:
                if findings[qid]["status"] != "answered":
                    reasons.append({"kind": "research", "id": qid,
                                    "status": findings[qid]["status"]})
            for field in step["required_fields"]:
                if not data["guided"]["fields"].get(field, "").strip():
                    reasons.append({"kind": "field", "id": field})
            results[step["id"]] = {
                "id": step["id"], "title": step["title"],
                "status": "blocked" if reasons else "completed", "reasons": reasons,
                "research_refs": step["question_ids"], "dependency_refs": step["depends_on"],
            }
            execution_order.append(step["id"])
            pending.remove(step)
    completed = sum(s["status"] == "completed" for s in results.values())
    total = len(results)
    return {
        "steps": [results[s["id"]] for s in data["guided"]["steps"]],
        "execution_order": execution_order,
        "progress": {"completed": completed, "total": total,
                     "percent": round(100 * completed / total, 2) if total else 100.0},
    }


def normalize_feedback(value):
    # Preserve punctuation: "can't" and "cant" must not silently become identical.
    return " ".join(value.casefold().split())


def compute_feedback(data, previous):
    steps = {s["id"]: s for s in previous["guided"]["steps"]}
    grouped = {}
    deferred = []
    for item in data["feedback"]["items"]:
        if steps[item["step_id"]]["status"] != "completed":
            deferred.append({"item_id": item["id"], "step_id": item["step_id"],
                             "reason": "onboarding_step_blocked"})
            continue
        key = normalize_feedback(item["text"])
        if key not in grouped:
            grouped[key] = {"id": item["id"], "normalized_text": key, "support": []}
        grouped[key]["support"].append({
            "item_id": item["id"], "step_id": item["step_id"],
            "quote": item["text"], "start": 0, "end": len(item["text"]),
        })
    groups = list(grouped.values())
    themes = []
    classified = set()
    for theme in data["feedback"]["themes"]:
        matches = [g for g in groups if any(
            re.search(r"(?<!\w)" + re.escape(normalize_feedback(keyword)) + r"(?!\w)",
                      g["normalized_text"]) is not None for keyword in theme["keywords"])]
        classified.update(g["id"] for g in matches)
        themes.append({"id": theme["id"], "label": theme["label"],
                       "group_ids": [g["id"] for g in matches], "unique_count": len(matches)})
    unmatched = [g["id"] for g in groups if g["id"] not in classified]
    themes.append({"id": "unclassified", "label": "Unclassified",
                   "group_ids": unmatched, "unique_count": len(unmatched)})
    eligible = sum(len(g["support"]) for g in groups)
    return {"groups": groups, "themes": themes, "deferred": deferred,
            "counts": {"received": len(data["feedback"]["items"]), "eligible": eligible,
                       "unique": len(groups), "duplicates": eligible - len(groups),
                       "deferred": len(deferred)}}


def compute_review(data, previous):
    findings = {f["id"]: f for f in previous["research"]["findings"]}
    steps = {s["id"]: s for s in previous["guided"]["steps"]}
    themes = {t["id"]: t for t in previous["feedback"]["themes"]}
    groups = {g["id"]: g for g in previous["feedback"]["groups"]}
    document = data["review"]["document"]
    checks = []
    for requirement in data["review"]["requirements"]:
        gaps, evidence = [], []
        for qid in requirement["question_ids"]:
            finding = findings[qid]
            if finding["status"] != "answered":
                gaps.append({"kind": "research", "id": qid, "reason": finding["status"]})
            else:
                evidence.append({"kind": "research", "id": qid, "support": finding["evidence"]})
        for sid in requirement["step_ids"]:
            if steps[sid]["status"] != "completed":
                gaps.append({"kind": "onboarding", "id": sid, "reason": "blocked"})
            else:
                evidence.append({"kind": "onboarding", "id": sid})
        for tid in requirement["theme_ids"]:
            theme = themes[tid]
            if not theme["group_ids"]:
                gaps.append({"kind": "feedback", "id": tid, "reason": "no_eligible_evidence"})
            else:
                evidence.append({"kind": "feedback", "id": tid,
                                 "support": [s for gid in theme["group_ids"]
                                             for s in groups[gid]["support"]]})
        for term in requirement["required_terms"]:
            match = re.search(re.escape(term), document["text"], re.IGNORECASE)
            if match is None:
                gaps.append({"kind": "document", "id": document["id"],
                             "term": term, "reason": "literal_term_missing"})
            else:
                evidence.append({"kind": "document", "id": document["id"], "term": term,
                                 "quote": match.group(), "start": match.start(), "end": match.end()})
        checks.append({"id": requirement["id"], "description": requirement["description"],
                       "status": "gap" if gaps else "met", "gaps": gaps, "evidence": evidence})
    return {"document_id": document["id"], "requirements": checks,
            "summary": {"met": sum(c["status"] == "met" for c in checks),
                        "gaps": sum(c["status"] == "gap" for c in checks)},
            "disclaimer": "Evidence and literal-term checks only; not certification, legal advice, or a truth assessment."}


COMPUTERS = {"research": compute_research, "guided": compute_guided,
             "feedback": compute_feedback, "review": compute_review}


def research_stage(data):
    validate_input(data)
    state = {"schema_version": 1, "synthetic": True, "stage": "research",
             "input": copy.deepcopy(data), "results": {"research": compute_research(data, {})}}
    return validate_state(state, "research")


def advance(state, previous_stage, next_stage):
    validate_state(state, previous_stage)
    require(STAGES.index(next_stage) == STAGES.index(previous_stage) + 1, "invalid stage transition")
    result = copy.deepcopy(state)
    result["stage"] = next_stage
    result["results"][next_stage] = COMPUTERS[next_stage](result["input"], result["results"])
    return validate_state(result, next_stage)


def guided_stage(state):
    return advance(state, "research", "guided")


def feedback_stage(state):
    return advance(state, "guided", "feedback")


def review_stage(state):
    return advance(state, "feedback", "review")


def run_pipeline(data):
    state = review_stage(feedback_stage(guided_stage(research_stage(data))))
    return {"status": "ok", **state}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON: duplicate key {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"JSON: non-finite number {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
