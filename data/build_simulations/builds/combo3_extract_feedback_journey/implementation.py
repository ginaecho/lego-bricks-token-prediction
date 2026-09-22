"""Synthetic, deterministic three-stage build. Python standard library only."""

import copy
import json
import sys


FIELDS = ("profile.role", "profile.goal", "requirements.channel", "requirements.budget")


class InputError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InputError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "Missing or unknown object keys")


def unique_list(value, label):
    require(isinstance(value, list), label + " must be a list")
    require(all(text(item) for item in value), label + " needs nonempty strings")
    require(len(value) == len(set(value)), "Duplicate " + label)


def validate(data):
    keys(data, ("sources", "claims", "feedback", "actions"))
    indexes = {}
    for collection in ("sources", "claims", "feedback", "actions"):
        rows = data[collection]
        require(isinstance(rows, list), collection + " must be a list")
        index = {}
        for row in rows:
            require(isinstance(row, dict) and text(row.get("id")),
                    collection + " needs nonempty IDs")
            require(row["id"] not in index, "Duplicate ID in " + collection)
            index[row["id"]] = row
        indexes[collection] = index

    for source in data["sources"]:
        keys(source, ("id", "text"))
        require(isinstance(source["text"], str), "Source text must be a string")

    def evidence(spans):
        require(isinstance(spans, list), "Evidence must be a list")
        seen = set()
        for span in spans:
            keys(span, ("source_id", "start", "end"))
            sid, start, end = span["source_id"], span["start"], span["end"]
            require(text(sid) and sid in indexes["sources"], "Unknown source ID")
            require(type(start) is int and type(end) is int, "Offsets must be integers")
            require(0 <= start < end <= len(indexes["sources"][sid]["text"]),
                    "Invalid source span")
            key = (sid, start, end)
            require(key not in seen, "Duplicate evidence span")
            seen.add(key)

    for claim in data["claims"]:
        keys(claim, ("id", "field", "value", "evidence"))
        require(text(claim["field"]) and claim["field"] in FIELDS, "Unknown field")
        require(text(claim["value"]), "Claim value must be a nonempty string")
        evidence(claim["evidence"])
        # Extraction is deliberately literal, not an inferred semantic classifier.
        require(not claim["evidence"] or any(
            claim["value"] in indexes["sources"][s["source_id"]]["text"][s["start"]:s["end"]]
            for s in claim["evidence"]), "Claim value not present in cited evidence")

    for feedback in data["feedback"]:
        keys(feedback, ("id", "theme", "stance", "applies_to", "evidence"))
        require(text(feedback["theme"]), "Theme must be nonempty")
        require(isinstance(feedback["stance"], str) and
                feedback["stance"] in ("support", "oppose", "neutral"), "Invalid stance")
        require(isinstance(feedback["applies_to"], dict), "applies_to must be an object")
        require(all(field in FIELDS and text(value)
                    for field, value in feedback["applies_to"].items()),
                "Invalid feedback context")
        evidence(feedback["evidence"])

    for action in data["actions"]:
        keys(action, ("id", "title", "requires_fields", "prerequisites",
                      "theme_weights", "priority", "evidence"))
        require(text(action["title"]), "Action title must be nonempty")
        unique_list(action["requires_fields"], "required fields")
        require(all(field in FIELDS for field in action["requires_fields"]),
                "Unknown required field")
        unique_list(action["prerequisites"], "prerequisites")
        require(all(dep in indexes["actions"] for dep in action["prerequisites"]),
                "Unknown prerequisite")
        require(type(action["priority"]) is int, "Priority must be an integer")
        require(isinstance(action["theme_weights"], dict) and
                all(text(theme) and type(weight) is int
                    for theme, weight in action["theme_weights"].items()),
                "Invalid theme weights")
        evidence(action["evidence"])

    # Iterative topological check also handles graphs deeper than Python's stack.
    pending = {a["id"]: set(a["prerequisites"]) for a in data["actions"]}
    completed = set()
    while pending:
        ready = {aid for aid, deps in pending.items() if deps <= completed}
        require(bool(ready), "Cyclic prerequisites")
        completed.update(ready)
        for aid in ready:
            del pending[aid]
    return indexes["sources"]


def grounded(spans, sources):
    return [dict(span, quote=sources[span["source_id"]]["text"][span["start"]:span["end"]])
            for span in spans]


def deduplicate(spans):
    return list({(s["source_id"], s["start"], s["end"]): s for s in spans}.values())


def extract(data, sources):
    result = {}
    for field in FIELDS:
        candidates = {}
        unsupported = []
        for claim in data["claims"]:
            if claim["field"] != field:
                continue
            if not claim["evidence"]:
                unsupported.append(claim["id"])
                continue
            item = candidates.setdefault(claim["value"], {"value": claim["value"],
                                                          "claim_ids": [], "evidence": []})
            item["claim_ids"].append(claim["id"])
            item["evidence"].extend(grounded(claim["evidence"], sources))
        values = list(candidates.values())
        for item in values:
            item["evidence"] = deduplicate(item["evidence"])
        result[field] = {
            "status": "resolved" if len(values) == 1 else "contradictory" if values else "missing",
            "value": values[0]["value"] if len(values) == 1 else None,
            "candidates": values,
            "unsupported_claim_ids": unsupported,
        }
    return result


def analyze(data, sources, extraction):
    themes = {}
    for feedback in data["feedback"]:
        theme = themes.setdefault(feedback["theme"], {
            "support": [], "oppose": [], "neutral": [], "inactive": [],
            "disagreement": False, "net_support": 0,
        })
        reasons = []
        context_evidence = []
        if not feedback["evidence"]:
            reasons.append("absent_evidence")
        for field, expected in feedback["applies_to"].items():
            actual = extraction[field]
            if actual["status"] != "resolved":
                reasons.append(field + ":" + actual["status"])
            elif actual["value"] != expected:
                reasons.append(field + ":context_mismatch")
            else:
                context_evidence.extend(actual["candidates"][0]["evidence"])
        record = {
            "feedback_id": feedback["id"], "stance": feedback["stance"],
            "applies_to": feedback["applies_to"],
            "evidence": grounded(feedback["evidence"], sources),
            "context_evidence": deduplicate(context_evidence),
        }
        if reasons:
            theme["inactive"].append(dict(record, reasons=reasons))
        else:
            theme[feedback["stance"]].append(record)
    for theme in themes.values():
        theme["disagreement"] = bool(theme["support"] and theme["oppose"])
        theme["net_support"] = len(theme["support"]) - len(theme["oppose"])
    return themes


def recommend(data, sources, extraction, feedback):
    actions = {a["id"]: a for a in data["actions"]}
    scores, evidence, impacts, intrinsic_blocks = {}, {}, {}, {}
    for aid, action in actions.items():
        blocks = [field + ":" + extraction[field]["status"]
                  for field in action["requires_fields"]
                  if extraction[field]["status"] != "resolved"]
        if not action["evidence"]:
            blocks.append("absent_action_evidence")
        intrinsic_blocks[aid] = blocks
        scores[aid] = action["priority"]
        evidence[aid] = grounded(action["evidence"], sources)
        impacts[aid] = []
        for field in action["requires_fields"]:
            if extraction[field]["status"] == "resolved":
                evidence[aid].extend(extraction[field]["candidates"][0]["evidence"])
        for name, weight in action["theme_weights"].items():
            theme = feedback.get(name)
            if theme is None or weight == 0:
                continue
            records = theme["support"] + theme["oppose"]
            if not records:
                continue
            delta = weight * theme["net_support"]
            scores[aid] += delta
            impacts[aid].append({"theme": name, "delta": delta,
                                "disagreement": theme["disagreement"],
                                "feedback_ids": [r["feedback_id"] for r in records]})
            for record in records:
                evidence[aid].extend(record["evidence"] + record["context_evidence"])

    steps, completed = [], set()
    for number in (1, 2):
        eligible = [aid for aid, action in actions.items()
                    if aid not in completed and not intrinsic_blocks[aid]
                    and set(action["prerequisites"]) <= completed]
        if not eligible:
            break
        aid = min(eligible, key=lambda item: (-scores[item], item))
        action = actions[aid]
        # Earlier steps are proposed prerequisites, never asserted completed work.
        prerequisites = [step for step in steps if step["action_id"] in action["prerequisites"]]
        spans = deduplicate(evidence[aid] + [
            span for step in prerequisites for span in step["evidence"]])
        steps.append({
            "step": number, "action_id": aid, "title": action["title"],
            "score": scores[aid], "prerequisites": action["prerequisites"],
            "conditional_on_completion": [s["action_id"] for s in prerequisites],
            "required_context": {field: extraction[field]["value"]
                                 for field in action["requires_fields"]},
            "feedback_impacts": impacts[aid], "evidence": spans,
            "source_ids": sorted({span["source_id"] for span in spans}),
        })
        completed.add(aid)
    blocked = {}
    for aid, action in actions.items():
        if aid in completed:
            continue
        reasons = intrinsic_blocks[aid] + [
            "prerequisite_not_planned:" + dep for dep in action["prerequisites"]
            if dep not in completed]
        blocked[aid] = reasons or ["two_step_limit"]
    return {"status": "ready" if len(steps) == 2 else "partial" if steps else "no_eligible_action",
            "steps": steps, "not_selected": blocked}


def run_pipeline(data, synthesizer=None):
    sources = validate(data)
    extraction = extract(data, sources)
    feedback = analyze(data, sources, extraction)
    journey = recommend(data, sources, extraction, feedback)
    result = {"extraction": extraction, "feedback": feedback, "journey": journey}
    if synthesizer is not None:
        # A callback cannot mutate decisions, reorder actions, or replace grounding.
        proposed = synthesizer(copy.deepcopy(result))
        keys(proposed, ("steps",))
        require(isinstance(proposed["steps"], list) and
                len(proposed["steps"]) == len(journey["steps"]), "Callback changed plan length")
        for supplied, actual in zip(proposed["steps"], journey["steps"]):
            keys(supplied, ("action_id", "source_ids", "explanation"))
            require(supplied["action_id"] == actual["action_id"], "Callback changed action ID")
            require(supplied["source_ids"] == actual["source_ids"], "Callback changed grounding")
            require(text(supplied["explanation"]), "Callback explanation must be nonempty")
            actual["supplemental_explanation"] = {
                "text": supplied["explanation"], "semantic_verification": "not_performed"}
    return result


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise InputError("Nonstandard JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=reject_duplicate_keys,
                             parse_constant=reject_constant)
        result = run_pipeline(data)
    except (InputError, OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        print(json.dumps({"error": {"type": "invalid_input", "message": str(error)}}))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
