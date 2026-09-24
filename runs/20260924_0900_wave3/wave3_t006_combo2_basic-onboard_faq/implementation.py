"""Deterministic onboarding -> grounded FAQ reference; all examples are synthetic."""
import json
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(keys), label + " has missing or unknown fields")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    require(len(value) <= 10000, label + " is too long")


def tokens(value):
    stop = {"a", "an", "the", "how", "do", "i", "to", "my", "is", "and", "can", "what"}
    return set(re.findall(r"[a-z0-9]+", value.lower())) - stop


def validate(state, phase):
    """Single validation boundary used for input and every integrated stage."""
    require(phase in {"input", "onboard", "faq"}, "unknown validation phase")
    keys = ["schema_version", "fixture_label", "customer", "workflow", "knowledge_base"]
    if phase != "input":
        keys += ["onboarding"]
    if phase == "faq":
        keys += ["support", "status"]
    obj(state, keys, "state")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "schema_version must be 1")
    require(state["fixture_label"] == "synthetic", "fixture_label must be synthetic")
    customer = state["customer"]
    obj(customer, ["id", "name", "goal", "completed_steps", "question"], "customer")
    for key in ("id", "name", "goal"):
        text(customer[key], "customer." + key)
    require(isinstance(customer["question"], str) and len(customer["question"]) <= 10000,
            "question must be text of at most 10000 characters")
    completed = customer["completed_steps"]
    require(isinstance(completed, list), "completed_steps must be a list")
    for step in completed:
        text(step, "completed step")
    require(len(set(completed)) == len(completed), "duplicate completed steps")
    workflow = state["workflow"]
    obj(workflow, ["goal", "steps", "completion_question"], "workflow")
    text(workflow["goal"], "workflow.goal")
    text(workflow["completion_question"], "completion_question")
    require(customer["goal"] == workflow["goal"], "customer goal does not match workflow")
    require(isinstance(workflow["steps"], list) and workflow["steps"], "steps must be a nonempty list")
    ids = []
    for step in workflow["steps"]:
        obj(step, ["id", "instruction", "help_question"], "step")
        for key in step:
            text(step[key], "step." + key)
        ids.append(step["id"])
    require(len(set(ids)) == len(ids), "duplicate workflow step ids")
    require(set(completed) <= set(ids), "unknown completed step")
    kb = state["knowledge_base"]
    require(isinstance(kb, list), "knowledge_base must be a list")
    article_ids = []
    for article in kb:
        obj(article, ["id", "goal", "step_id", "question", "answer"], "article")
        for key in ("id", "goal", "question", "answer"):
            text(article[key], "article." + key)
        require(article["step_id"] is None or isinstance(article["step_id"], str),
                "article.step_id must be text or null")
        require(article["goal"] == workflow["goal"], "article has unknown goal")
        require(article["step_id"] is None or article["step_id"] in ids,
                "article has unknown step")
        article_ids.append(article["id"])
    require(len(set(article_ids)) == len(article_ids), "duplicate article ids")
    if phase in {"onboard", "faq"}:
        obj(state["onboarding"], ["customer_id", "goal", "completed", "next_step",
                                  "message", "support_question"], "onboarding")
        require(state["onboarding"] == expected_onboarding(state),
                "onboarding handoff does not match validated customer/workflow")
    if phase == "faq":
        require(state["status"] == "ok", "invalid success status")
        support = state["support"]
        obj(support, ["customer_id", "goal", "step_id", "question", "status",
                      "answer", "citations", "reason"], "support")
        expected = answer_faq(state)
        require(support == expected, "FAQ output is not grounded in the validated handoff")
    return state


def expected_onboarding(state):
    customer = state["customer"]
    workflow = state["workflow"]
    step = next((s for s in workflow["steps"]
                 if s["id"] not in customer["completed_steps"]), None)
    question = customer["question"].strip() or (
        step["help_question"] if step else workflow["completion_question"])
    return {
        "customer_id": customer["id"], "goal": customer["goal"],
        "completed": step is None, "next_step": dict(step) if step else None,
        "message": (f"{customer['name']}, next for {customer['goal']}: {step['instruction']}"
                    if step else f"{customer['name']}, onboarding for {customer['goal']} is complete."),
        "support_question": question,
    }


def onboard(state):
    validate(state, "input")
    result = dict(state, onboarding=expected_onboarding(state))
    return validate(result, "onboard")


def answer_faq(state):
    handoff = state["onboarding"]
    step_id = handoff["next_step"]["id"] if handoff["next_step"] else None
    query = tokens(handoff["support_question"])
    candidates = []
    for article in state["knowledge_base"]:
        if article["goal"] != handoff["goal"] or article["step_id"] not in (None, step_id):
            continue
        terms = tokens(article["question"])
        # Conservative lexical retrieval: require majority coverage on both sides.
        overlap = len(query & terms)
        if query and terms and overlap / len(query) >= .6 and overlap / len(terms) >= .6:
            score = overlap / len(query | terms)
            candidates.append((score, article))
    candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
    ambiguous = len(candidates) > 1 and candidates[0][0] == candidates[1][0]
    chosen = candidates[0][1] if candidates and not ambiguous else None
    return {
        "customer_id": handoff["customer_id"], "goal": handoff["goal"],
        "step_id": step_id, "question": handoff["support_question"],
        "status": "answered" if chosen else "abstained",
        "answer": chosen["answer"] if chosen else None,
        "citations": [chosen["id"]] if chosen else [],
        "reason": None if chosen else ("ambiguous_matches" if ambiguous else "insufficient_evidence"),
    }


def faq(state):
    validate(state, "onboard")
    result = dict(state, support=answer_faq(state), status="ok")
    return validate(result, "faq")


def run(payload):
    return faq(onboard(payload))


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonfinite JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=reject_duplicates,
                                parse_constant=reject_constant)
        output = run(payload)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
