"""Synthetic public-sector reference pipeline; Python standard library only.

Run: python -B implementation.py example_input.json
Each stage accepts and returns the same validated envelope. No eligibility,
legal, or compliance determination is made. No network or model is used.
"""

import copy
import json
import re
import sys
from pathlib import Path


STAGES = ("input", "research", "recommend", "insights", "support")
PII_FIELDS = ("name", "email", "phone", "address")
THEMES = {
    "accessibility": ({"access", "accessible", "keyboard", "screen", "disability"},
                      "Offer an accessible form or help by phone."),
    "clarity": ({"confusing", "unclear", "complicated", "understand"},
                "Explain each form step in plain language."),
    "delays": ({"wait", "waiting", "delay", "delayed", "slow"},
               "Ask the service team for a case update."),
    "cost": ({"fee", "cost", "expensive"}, "Explain any fees before applying."),
    "positive": ({"helpful", "easy", "clear"}, "Keep the parts people find easy."),
    "other": (set(), "Review this feedback with the service team."),
}
STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "for", "help", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "please", "the", "to", "with", "you",
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(required), label + " has missing or unknown fields")


def text(value, label, maximum=4000):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    require(len(value) <= maximum, label + " is too long")
    require(not any(ord(c) < 32 and c not in "\n\t\r" for c in value),
            label + " contains control characters")


def string_list(value, label, maximum=30):
    require(isinstance(value, list) and len(value) <= maximum, label + " must be a bounded list")
    for item in value:
        text(item, label, 140)
    require(len(set(value)) == len(value), label + " contains duplicates")


def identifier(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", value),
            label + " must be a non-personal identifier")


def words(value):
    return set(re.findall(r"[a-z]+", value.lower())) - STOPWORDS


def validate_context(context):
    keys(context, ("government_form", "policies", "feedback", "support_question"), "context")
    form = context["government_form"]
    keys(form, ("citizen", "service_request", "benefits_application", "preferences"), "government_form")
    keys(form["citizen"], PII_FIELDS, "citizen")
    for field in PII_FIELDS:
        text(form["citizen"][field], "citizen." + field, 200)
        require(len(form["citizen"][field].strip()) >= 4, "PII fields need at least four characters")
    request = form["service_request"]
    keys(request, ("case_number", "question", "category"), "service_request")
    text(request["case_number"], "case_number", 80)
    require(len(request["case_number"]) >= 4, "case_number is too short")
    text(request["question"], "question", 1000)
    text(request["category"], "category", 100)
    application = form["benefits_application"]
    keys(application, ("application_id", "program_interests", "status"), "benefits_application")
    text(application["application_id"], "application_id", 80)
    require(len(application["application_id"]) >= 4, "application_id is too short")
    string_list(application["program_interests"], "program_interests")
    require(application["status"] in ("not_started", "draft", "submitted"),
            "status must be not_started, draft, or submitted")
    preferences = form["preferences"]
    keys(preferences, ("topics", "language", "accessible_format", "max_results"), "preferences")
    string_list(preferences["topics"], "topics")
    require(preferences["language"] == "en", "this reference supports English only")
    require(type(preferences["accessible_format"]) is bool, "accessible_format must be boolean")
    require(type(preferences["max_results"]) is int and 1 <= preferences["max_results"] <= 5,
            "max_results must be an integer from 1 to 5")
    text(context["support_question"], "support_question", 1000)
    require(isinstance(context["policies"], list) and len(context["policies"]) <= 30,
            "policies must be a list with at most 30 items")
    policy_ids, service_ids = set(), set()
    for policy in context["policies"]:
        keys(policy, ("policy_id", "title", "text", "services"), "policy")
        identifier(policy["policy_id"], "policy_id")
        require(policy["policy_id"] not in policy_ids, "duplicate policy_id")
        policy_ids.add(policy["policy_id"])
        text(policy["title"], "policy title", 150)
        text(policy["text"], "policy text", 12000)
        require(isinstance(policy["services"], list) and len(policy["services"]) <= 20,
                "services must be a list with at most 20 items")
        for service in policy["services"]:
            keys(service, ("service_id", "name", "description", "topics",
                           "application_steps", "eligibility_note"), "service")
            identifier(service["service_id"], "service_id")
            require(service["service_id"] not in service_ids, "duplicate service_id")
            service_ids.add(service["service_id"])
            text(service["name"], "service name", 100)
            text(service["description"], "service description", 400)
            text(service["eligibility_note"], "eligibility_note", 300)
            string_list(service["topics"], "service topics")
            string_list(service["application_steps"], "application_steps", 5)
            require(bool(service["application_steps"]), "application_steps cannot be empty")
            for step in service["application_steps"]:
                require(len(step.split()) <= 24, "application steps must use short, plain-language text")
                require(step in policy["text"], "each application step must appear in policy text")
    require(isinstance(context["feedback"], list) and len(context["feedback"]) <= 200,
            "feedback must be a list with at most 200 items")
    feedback_ids = set()
    for feedback in context["feedback"]:
        keys(feedback, ("feedback_id", "service_id", "rating", "text"), "feedback")
        identifier(feedback["feedback_id"], "feedback_id")
        require(feedback["feedback_id"] not in feedback_ids, "duplicate feedback_id")
        feedback_ids.add(feedback["feedback_id"])
        require(feedback["service_id"] in service_ids, "feedback references an unknown service")
        require(type(feedback["rating"]) is int and 1 <= feedback["rating"] <= 5,
                "rating must be an integer from 1 to 5")
        text(feedback["text"], "feedback text", 1000)


def _research(context, results):
    form = context["government_form"]
    terms = words(form["service_request"]["question"])
    terms |= words(" ".join(form["preferences"]["topics"]))
    terms |= words(" ".join(form["benefits_application"]["program_interests"]))
    evidence, catalog = [], []
    for policy in sorted(context["policies"], key=lambda p: p["policy_id"]):
        searchable = " ".join([policy["title"], policy["text"]] +
                              [" ".join(s["topics"]) for s in policy["services"]])
        matched = sorted(terms & words(searchable))
        if not matched:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", policy["text"])
        excerpt = sorted(enumerate(sentences),
                         key=lambda pair: (-len(terms & words(pair[1])), pair[0]))[0][1]
        evidence.append({"policy_id": policy["policy_id"], "excerpt": excerpt,
                         "matched_terms": matched})
        for service in policy["services"]:
            catalog.append(dict(copy.deepcopy(service), policy_id=policy["policy_id"]))
    return {
        "question": form["service_request"]["question"],
        "evidence": evidence,
        "catalog": catalog,
        "limitations": ["Only supplied synthetic policy text was checked.",
                        "A service team must confirm current rules and eligibility."],
        "unanswered": [] if evidence else ["No supplied policy matches the question or interests."],
    }


def _recommend(context, results):
    research = results["research"]
    form = context["government_form"]
    question_terms = words(research["question"])
    interest_terms = words(" ".join(form["preferences"]["topics"] +
                                   form["benefits_application"]["program_interests"]))
    recommendations = []
    for service in research["catalog"]:
        service_terms = words(" ".join(service["topics"]) + " " +
                              service["name"] + " " + service["description"])
        interest_matches = sorted(interest_terms & service_terms)
        question_matches = sorted(question_terms & service_terms)
        score = 3 * len(interest_matches) + len(question_matches)
        if not score:
            continue
        recommendations.append({
            "service_id": service["service_id"], "name": service["name"],
            "policy_id": service["policy_id"], "score": score,
            "interest_matches": interest_matches, "question_matches": question_matches,
            "reason": "Matched stated interests and request words; this is not an eligibility decision.",
        })
    recommendations.sort(key=lambda item: (-item["score"], item["service_id"]))
    return {
        "items": recommendations[:form["preferences"]["max_results"]],
        "ranking_rule": "3 points per interest word and 1 per request word; ties use service ID.",
        "excluded_factors": ["name", "email", "phone", "address", "case_number", "application_id"],
        "eligibility_decision": "not_made",
    }


def _insights(context, results):
    selected = {item["service_id"] for item in results["recommend"]["items"]}
    feedback = [item for item in context["feedback"] if item["service_id"] in selected]
    themes = {}
    for item in feedback:
        tokens = words(item["text"])
        labels = [label for label, (terms, _) in THEMES.items() if tokens & terms] or ["other"]
        for label in labels:
            themes.setdefault(label, []).append(item)
    summaries = []
    for label, items in sorted(themes.items()):
        summaries.append({
            "theme": label, "count": len(items),
            "feedback_ids": sorted(item["feedback_id"] for item in items),
            "service_ids": sorted({item["service_id"] for item in items}),
            "mean_rating": round(sum(item["rating"] for item in items) / len(items), 2),
            "action": THEMES[label][1],
        })
    return {
        "selected_service_ids": sorted(selected), "feedback_count": len(feedback),
        "themes": summaries,
        "limitations": "Small synthetic samples are not population estimates. Themes may overlap.",
    }


def _support(context, results):
    recommendations = results["recommend"]["items"]
    insight = results["insights"]
    question_terms = words(context["support_question"])
    escalation = bool(question_terms & {
        "appeal", "denied", "rejected", "emergency", "urgent", "eligible", "eligibility",
        "approved", "approve", "guarantee",
    })
    selected = []
    for item in recommendations:
        service = next(s for s in results["research"]["catalog"]
                       if s["service_id"] == item["service_id"])
        score = len(question_terms & words(" ".join(service["topics"]) + " " +
                                          service["name"] + " " + service["description"] +
                                          " " + " ".join(service["application_steps"])))
        if score:
            selected.append((score, item, service))
    selected.sort(key=lambda entry: (-entry[0], entry[1]["service_id"]))
    if not selected:
        return {
            "answer": "I cannot answer this from the supplied policy. Ask the service team for help.",
            "steps": [], "citations": [], "service_id": None,
            "handoff_required": True, "reason": "No recommended service grounds this question.",
            "insight_actions": [], "format": "plain_language_numbered_steps",
            "eligibility_decision": "not_made",
        }
    _, item, service = selected[0]
    actions = [theme["action"] for theme in insight["themes"]
               if service["service_id"] in theme["service_ids"] and theme["theme"] != "positive"]
    answer = "Here are the next steps from the supplied policy. The service team must confirm your eligibility."
    if escalation:
        answer = "A service team member must review this question. I cannot decide eligibility or handle an appeal."
    if question_terms & {"urgent", "emergency"}:
        answer += " If someone is in immediate danger, contact local emergency services."
    if context["government_form"]["preferences"]["accessible_format"]:
        actions.append("Ask for an accessible format if you need one.")
    return {
        "answer": answer,
        "steps": [{"number": i + 1, "text": step}
                  for i, step in enumerate(service["application_steps"])],
        "citations": [{"policy_id": item["policy_id"], "section": "supplied policy text"}],
        "service_id": item["service_id"], "handoff_required": escalation,
        "reason": "Human review is needed." if escalation else "Steps are quoted from the cited policy.",
        "insight_actions": actions, "format": "plain_language_numbered_steps",
        "eligibility_decision": "not_made",
    }


COMPUTE = {"research": _research, "recommend": _recommend,
           "insights": _insights, "support": _support}
EXPLANATIONS = {
    "research": "Match question and interest words to supplied policies.",
    "recommend": "Rank evidence-backed services using stated interests and request words.",
    "insights": "Group feedback only for recommended services using visible keyword rules.",
    "support": "Use cited service steps and feedback actions; defer decisions to a human.",
}


def audit_entry(stage, results):
    return {
        "stage": stage, "consumed_stage": STAGES[STAGES.index(stage) - 1],
        "rule": EXPLANATIONS[stage],
        "source_policy_ids": sorted(e["policy_id"] for e in results["research"]["evidence"]),
    }


def sensitive_values(context):
    form = context["government_form"]
    return list(form["citizen"].values()) + [
        form["service_request"]["case_number"],
        form["benefits_application"]["application_id"],
    ]


def sanitize(context):
    result = copy.deepcopy(context)
    values = sorted(set(sensitive_values(context)), key=len, reverse=True)

    def scrub(value):
        if isinstance(value, str):
            for secret in values:
                value = re.sub(re.escape(secret), "[REDACTED]", value, flags=re.IGNORECASE)
            value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", "[REDACTED]", value)
            value = re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{6,}\d)(?!\w)", "[REDACTED]", value)
            return value
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    result = scrub(result)
    form = result["government_form"]
    form["citizen"] = {field: "[REDACTED]" for field in PII_FIELDS}
    form["service_request"]["case_number"] = "[REDACTED]"
    form["benefits_application"]["application_id"] = "[REDACTED]"
    return result


def validate_state(state, expected_stage=None):
    """Shared structural, privacy, provenance, and deterministic-result validation."""
    keys(state, ("schema_version", "synthetic", "status", "stage", "context", "results", "audit"),
         "envelope")
    require(state["schema_version"] == "1.0", "unsupported schema_version")
    require(state["synthetic"] is True, "only clearly labeled synthetic data is accepted")
    require(state["status"] == "ok", "status must be ok")
    require(state["stage"] in STAGES, "invalid stage")
    if expected_stage is not None:
        require(state["stage"] == expected_stage, "unexpected pipeline stage")
    validate_context(state["context"])
    stages = STAGES[1:STAGES.index(state["stage"]) + 1]
    keys(state["results"], stages, "results")
    require(isinstance(state["audit"], list), "audit must be a list")
    if stages:
        require(all(value == "[REDACTED]" for value in sensitive_values(state["context"])),
                "output contains unredacted citizen identifiers")
        require(sanitize(state["context"]) == state["context"], "output contains unredacted contact data")
    verified = {}
    audit = []
    for stage in stages:
        expected = COMPUTE[stage](state["context"], verified)
        require(state["results"][stage] == expected, stage + " result or provenance is invalid")
        verified[stage] = expected
        audit.append(audit_entry(stage, verified))
    require(state["audit"] == audit, "audit trail is invalid")
    return state


def advance(state, target):
    require(target in COMPUTE, "unknown pipeline stage")
    previous = STAGES[STAGES.index(target) - 1]
    validate_state(state, previous)
    output = copy.deepcopy(state)
    if target == "research":
        output["context"] = sanitize(output["context"])
        validate_context(output["context"])
    output["results"][target] = COMPUTE[target](output["context"], output["results"])
    output["stage"] = target
    output["audit"].append(audit_entry(target, output["results"]))
    return validate_state(output, target)


def research(state):
    return advance(state, "research")


def recommend(state):
    return advance(state, "recommend")


def insights(state):
    return advance(state, "insights")


def support(state):
    return advance(state, "support")


def run_pipeline(state):
    validate_state(state, "input")
    for stage in STAGES[1:]:
        state = advance(state, stage)
    return state


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 1_000_000, "input file exceeds 1 MB")
        state = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(state)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        # Do not echo exceptions, paths, or citizen data to logs or error output.
        print(json.dumps({"status": "error", "error": "Invalid input, file, or pipeline state."}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
