"""Deterministic onboarding-to-support reference pipeline; Python standard library."""

import json
import re
import sys


class ValidationError(ValueError):
    pass


STEPS = (
    ("verify_email", "Verify your email", "Open your welcome email and verify your address."),
    ("set_goal", "Set your workspace goal", "Save your main goal in workspace settings."),
    ("invite_team", "Invite your team", "Invite a teammate from the Members page."),
    ("start_project", "Start your first project", "Create a project aligned with your goal."),
)
STEP_IDS = {step[0] for step in STEPS}
STOP_WORDS = {"a", "an", "and", "are", "can", "do", "for", "how", "i", "in",
              "is", "it", "my", "of", "on", "please", "the", "to", "with", "you", "your"}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    require(len(value) <= 10000, path + " is too long")


def fields(value, required, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(required), path + " has missing or unknown fields")


def strings(value, path):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path + " item")
    require(len(value) == len(set(value)), path + " must not contain duplicates")


def validate(kind, value):
    """One validation boundary used for request, intermediate, and final schemas."""
    if kind == "input":
        fields(value, ("schema_version", "fixture_label", "customer", "support", "knowledge"), kind)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be integer 1")
        text(value["fixture_label"], "fixture_label")
        customer = value["customer"]
        fields(customer, ("id", "name", "role", "goal", "completed_steps"), "customer")
        for key in ("id", "name", "goal"):
            text(customer[key], "customer." + key)
        require(customer["role"] in ("individual", "team_admin"), "unsupported customer role")
        strings(customer["completed_steps"], "completed_steps")
        require(set(customer["completed_steps"]) <= STEP_IDS, "unknown completed step")
        fields(value["support"], ("question", "team_online"), "support")
        text(value["support"]["question"], "support.question")
        require(type(value["support"]["team_online"]) is bool, "team_online must be boolean")
        require(isinstance(value["knowledge"], list), "knowledge must be an array")
        ids = []
        for article in value["knowledge"]:
            fields(article, ("id", "title", "body", "keywords", "step_ids"), "article")
            for key in ("id", "title", "body"):
                text(article[key], "article." + key)
            strings(article["keywords"], "article.keywords")
            strings(article["step_ids"], "article.step_ids")
            require(set(article["step_ids"]) <= STEP_IDS, "unknown article step")
            ids.append(article["id"])
        require(len(ids) == len(set(ids)), "duplicate article ids")
    elif kind == "onboarding":
        fields(value, ("customer_id", "goal", "status", "next_step", "remaining_steps"), kind)
        text(value["customer_id"], "customer_id")
        text(value["goal"], "goal")
        strings(value["remaining_steps"], "remaining_steps")
        require(set(value["remaining_steps"]) <= STEP_IDS, "unknown remaining step")
        require(value["status"] in ("in_progress", "complete"), "invalid onboarding status")
        if value["status"] == "complete":
            require(value["next_step"] is None and not value["remaining_steps"],
                    "complete onboarding cannot have remaining steps")
        else:
            step = value["next_step"]
            fields(step, ("id", "title", "instruction"), "next_step")
            require(bool(value["remaining_steps"]) and step["id"] == value["remaining_steps"][0],
                    "next step must be first remaining step")
            text(step["title"], "next_step.title")
            text(step["instruction"], "next_step.instruction")
    elif kind == "support":
        fields(value, ("customer_id", "onboarding_step_id", "status", "answer", "citations",
                       "next_step", "handoff"), kind)
        text(value["customer_id"], "customer_id")
        require(value["onboarding_step_id"] is None or value["onboarding_step_id"] in STEP_IDS,
                "invalid support onboarding step")
        require(value["status"] in ("answered", "needs_human"), "invalid support status")
        text(value["answer"], "answer")
        strings(value["citations"], "citations")
        require(bool(value["citations"]) == (value["status"] == "answered"),
                "only grounded answers can have citations")
        require(value["next_step"] is None or isinstance(value["next_step"], dict),
                "invalid next_step")
        if value["status"] == "answered":
            require(value["handoff"] is None, "answered support cannot require handoff")
        else:
            fields(value["handoff"], ("route", "summary"), "handoff")
            require(value["handoff"]["route"] in ("live_team", "offline_queue"), "invalid route")
            text(value["handoff"]["summary"], "handoff.summary")
    elif kind == "output":
        fields(value, ("schema_version", "fixture_label", "status", "onboarding", "support"), kind)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "invalid output schema version")
        text(value["fixture_label"], "fixture_label")
        require(value["status"] == "ok", "invalid output status")
        validate("onboarding", value["onboarding"])
        validate("support", value["support"])
        onboarding, support = value["onboarding"], value["support"]
        require(onboarding["customer_id"] == support["customer_id"], "customer handoff mismatch")
        require(onboarding["next_step"] == support["next_step"], "next-step handoff mismatch")
        expected_id = onboarding["next_step"]["id"] if onboarding["next_step"] else None
        require(support["onboarding_step_id"] == expected_id, "step-id handoff mismatch")
    else:
        raise ValidationError("unknown schema kind")
    return value


def onboard(request):
    validate("input", request)
    customer = request["customer"]
    remaining = [
        step for step in STEPS
        if step[0] not in customer["completed_steps"]
        and (step[0] != "invite_team" or customer["role"] == "team_admin")
    ]
    next_step = None
    if remaining:
        identifier, title, instruction = remaining[0]
        next_step = {
            "id": identifier, "title": title,
            "instruction": f'{customer["name"]}, {instruction} Your goal: {customer["goal"]}',
        }
    return validate("onboarding", {
        "customer_id": customer["id"], "goal": customer["goal"],
        "status": "in_progress" if remaining else "complete",
        "next_step": next_step, "remaining_steps": [step[0] for step in remaining],
    })


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold())) - STOP_WORDS


def answer_support(request, onboarding):
    validate("input", request)
    validate("onboarding", onboarding)
    # Reject valid-looking but stale or foreign handoffs, not just malformed objects.
    require(onboarding == onboard(request), "onboarding handoff does not match request")
    question = request["support"]["question"]
    question_tokens = tokens(question)
    next_step = onboarding["next_step"]
    step_id = next_step["id"] if next_step else None
    candidates = []
    for article in request["knowledge"]:
        overlap = question_tokens & tokens(article["title"] + " " + " ".join(article["keywords"]))
        if overlap:
            candidates.append((len(overlap), int(step_id in article["step_ids"]), article))
    # Input order is the final stable tie-break; onboarding relevance never creates a match.
    chosen = max(candidates, key=lambda candidate: candidate[:2])[2] if candidates else None
    result = {
        "customer_id": onboarding["customer_id"],
        "onboarding_step_id": step_id,
        "status": "answered" if chosen else "needs_human",
        "answer": chosen["body"] if chosen else
                  "I do not have a grounded answer in the provided knowledge. Please contact support.",
        "citations": [chosen["id"]] if chosen else [],
        "next_step": dict(next_step) if next_step else None,
        "handoff": None,
    }
    if chosen is None:
        result["handoff"] = {
            "route": "live_team" if request["support"]["team_online"] else "offline_queue",
            "summary": f'Customer {onboarding["customer_id"]}; goal: {onboarding["goal"]}; '
                       f'next step: {step_id or "complete"}; question: {question}',
        }
    return validate("support", result)


def run_pipeline(request):
    validate("input", request)
    onboarding = onboard(request)
    support = answer_support(request, onboarding)
    return validate("output", {
        "schema_version": 1, "fixture_label": request["fixture_label"], "status": "ok",
        "onboarding": onboarding, "support": support,
    })


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            request = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(request)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
