"""Synthetic, deterministic public-service pipeline. No eligibility decisions.

Run: python -B implementation.py example_input.json
Use "-" to read government form JSON from stdin. Only the standard library is used.
The privacy checks are demonstrations, not legal or accessibility certification.
"""

import copy
import json
import re
import sys


VERSION = "1.0"
STAGES = ("onboard", "support", "research", "recommend")
MAX_BYTES = 100_000
STOP_WORDS = {
    "the", "a", "an", "for", "of", "to", "i", "is", "are", "what", "should",
    "my", "and", "or", "do", "does", "with", "how", "can", "please",
}
PII_PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b\d{3}[- ]\d{2}[- ]\d{4}\b|"
    r"\b(?:\+?\d{1,2}[- .])?\d{3}[- .]\d{3}[- .]\d{4}\b"
)


class ValidationError(ValueError):
    """A safe, non-PII validation message."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            label + " has missing or unknown fields.")


def text(value, label, maximum=500):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            label + " must be nonempty bounded text.")
    require(not any(ord(c) < 32 and c not in "\n\t" for c in value),
            label + " contains unsupported control characters.")


def identifier(value, prefix):
    require(isinstance(value, str) and
            re.fullmatch(re.escape(prefix) + r"[A-Z0-9-]{1,40}", value) is not None,
            "Use clearly synthetic identifiers.")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def public_text(value, private_values=()):
    for item in strings(value):
        require(not PII_PATTERN.search(item),
                "Public text must not contain direct contact or identity details.")
        folded = item.casefold()
        require(not any(secret.casefold() in folded for secret in private_values),
                "Public text must not repeat citizen identity fields.")


def validate_policies(policies):
    require(isinstance(policies, list) and 1 <= len(policies) <= 20,
            "Provide between 1 and 20 policy documents.")
    seen_policies, seen_programs = set(), set()
    for policy in policies:
        keys(policy, ("policy_id", "title", "service", "text", "program"), "Policy")
        identifier(policy["policy_id"], "SYN-POL-")
        require(policy["policy_id"] not in seen_policies, "Policy IDs must be unique.")
        seen_policies.add(policy["policy_id"])
        for field in ("title", "service"):
            text(policy[field], "Policy " + field, 100)
        text(policy["text"], "Policy text", 8000)
        program = policy["program"]
        keys(program, ("program_id", "title", "next_step"), "Program")
        identifier(program["program_id"], "SYN-PROG-")
        require(program["program_id"] not in seen_programs, "Program IDs must be unique.")
        seen_programs.add(program["program_id"])
        text(program["title"], "Program title", 100)
        text(program["next_step"], "Program next step", 160)


def validate_context(context):
    keys(context, ("service_request", "benefits_application", "preferences",
                   "policy_documents"), "Context")
    request = context["service_request"]
    keys(request, ("case_number", "service", "question"), "Service request")
    identifier(request["case_number"], "SYN-")
    text(request["service"], "Service", 100)
    text(request["question"], "Question", 500)
    application = context["benefits_application"]
    keys(application, ("application_id", "status"), "Benefits application")
    identifier(application["application_id"], "SYN-APP-")
    require(application["status"] in ("draft", "submitted", "needs_information"),
            "Unsupported application status.")
    preferences = context["preferences"]
    keys(preferences, ("preferred_language", "accessibility_needs"), "Preferences")
    require(preferences["preferred_language"] == "en",
            "This reference supports English only; seek language assistance.")
    needs = preferences["accessibility_needs"]
    require(isinstance(needs, list) and len(needs) <= 2 and
            all(isinstance(x, str) and x in ("plain_text", "screen_reader") for x in needs)
            and len(set(needs)) == len(needs), "Unsupported accessibility preferences.")
    validate_policies(context["policy_documents"])
    public_text(context)


def validate(value, expected_stage="input"):
    """One boundary validator for raw forms and every pipeline handoff."""
    if expected_stage == "input":
        keys(value, ("schema_version", "synthetic", "citizen", "privacy",
                     "service_request", "benefits_application", "policy_documents"), "Form")
        require(value["schema_version"] == VERSION and value["synthetic"] is True,
                "Only version 1.0 clearly labeled synthetic forms are supported.")
        keys(value["privacy"], ("processing_consent",), "Privacy")
        require(value["privacy"]["processing_consent"] is True,
                "Demonstration processing consent is required.")
        citizen = value["citizen"]
        keys(citizen, ("name", "email", "address", "preferred_language",
                       "accessibility_needs"), "Citizen")
        for field in ("name", "email", "address"):
            text(citizen[field], "Citizen identity field", 200)
            require(len(citizen[field].strip()) >= 3, "Identity fixture is too short.")
        context = {
            "service_request": value["service_request"],
            "benefits_application": value["benefits_application"],
            "preferences": {key: citizen[key] for key in
                            ("preferred_language", "accessibility_needs")},
            "policy_documents": value["policy_documents"],
        }
        validate_context(context)
        public_text(context, [citizen[key] for key in ("name", "email", "address")])
        return value

    require(expected_stage in STAGES, "Unknown stage.")
    keys(value, ("schema_version", "synthetic", "status", "stage", "context",
                 "results", "audit", "safeguards"), "Envelope")
    require(value["schema_version"] == VERSION and value["synthetic"] is True
            and value["status"] == "ok" and value["stage"] == expected_stage,
            "Unexpected pipeline stage or schema.")
    validate_context(value["context"])
    completed = STAGES[:STAGES.index(expected_stage) + 1]
    keys(value["results"], completed, "Stage results")
    require(isinstance(value["audit"], list) and
            value["audit"] == [{"stage": stage, "rule": "deterministic-" + stage + "-v1"}
                               for stage in completed], "Invalid audit trail.")
    require(value["safeguards"] == safeguards(), "Missing public-sector safeguards.")
    public_text(value)
    results = value["results"]
    keys(results["onboard"], ("next_step", "reason"), "Onboarding result")
    require(results["onboard"] == onboarding_result(value["context"]),
            "Onboarding does not match application status.")
    if "support" in completed:
        support_result = results["support"]
        keys(support_result, ("answer", "citations", "unanswered_question",
                             "handoff_next_step"), "Support result")
        require(support_result["handoff_next_step"] == results["onboard"]["next_step"],
                "Support lost the onboarding handoff.")
        evidence = support_result["citations"]
        validate_evidence(evidence, value["context"])
        expected_answer = " ".join(row["quote"] for row in evidence) if evidence else (
            "The supplied policies do not answer this question. Ask a service officer.")
        require(support_result["answer"] == expected_answer, "Support answer is not grounded.")
        require(support_result["unanswered_question"] ==
                (None if evidence else value["context"]["service_request"]["question"]),
                "Invalid unresolved question.")
    if "research" in completed:
        research_result = results["research"]
        keys(research_result, ("evidence", "finding", "limitations", "next_step"), "Research result")
        require(research_result["evidence"] == results["support"]["citations"],
                "Research evidence must come from the validated support handoff.")
        require(research_result == research_result_for(results["support"]),
                "Research summary is inconsistent with its evidence.")
    if "recommend" in completed:
        require(results["recommend"] == recommendation_result(value["context"], results["research"]),
                "Recommendations must follow the research evidence.")
    return value


def safeguards():
    return {
        "privacy": "Citizen name, email, and address are not retained in outputs.",
        "review": "Keep the source IDs and rule trail for open-records review.",
        "accessibility": "Plain-text English; no color-only or image-only instructions.",
        "decision": "Suggestions only. A service officer must decide benefit eligibility.",
        "notice": "Synthetic demonstration; not a compliance certification.",
    }


def tokens(value):
    return set(re.findall(r"[a-z]+", value.lower())) - STOP_WORDS


def validate_evidence(evidence, context):
    require(isinstance(evidence, list) and len(evidence) <= 3, "Invalid evidence list.")
    policies = {p["policy_id"]: p for p in context["policy_documents"]}
    seen = set()
    for row in evidence:
        keys(row, ("policy_id", "quote"), "Evidence")
        require(isinstance(row["policy_id"], str) and row["policy_id"] in policies,
                "Evidence must cite a supplied policy.")
        text(row["quote"], "Evidence quote", 240)
        policy = policies[row["policy_id"]]
        require(row["quote"] in policy["text"], "Evidence quote is not in its policy.")
        require(policy["service"].casefold() == context["service_request"]["service"].casefold(),
                "Evidence is for a different service.")
        key = (row["policy_id"], row["quote"])
        require(key not in seen, "Duplicate evidence.")
        seen.add(key)


def onboarding_result(context):
    status = context["benefits_application"]["status"]
    steps = {
        "draft": "Review the form and gather the documents listed in the policy.",
        "submitted": "Check your application status before sending another form.",
        "needs_information": "Read the request for information and prepare the missing items.",
    }
    return {"next_step": steps[status],
            "reason": "Your benefits application status is " + status.replace("_", " ") + "."}


def onboard(form):
    validate(form)
    citizen = form["citizen"]
    context = copy.deepcopy({
        "service_request": form["service_request"],
        "benefits_application": form["benefits_application"],
        "preferences": {key: citizen[key] for key in ("preferred_language", "accessibility_needs")},
        "policy_documents": form["policy_documents"],
    })
    state = {
        "schema_version": VERSION, "synthetic": True, "status": "ok", "stage": "onboard",
        "context": context, "results": {"onboard": onboarding_result(context)},
        "audit": [{"stage": "onboard", "rule": "deterministic-onboard-v1"}],
        "safeguards": safeguards(),
    }
    return validate(state, "onboard")


def advance(previous, previous_stage, stage, result):
    validate(previous, previous_stage)
    state = copy.deepcopy(previous)
    state["stage"] = stage
    state["results"][stage] = result
    state["audit"].append({"stage": stage, "rule": "deterministic-" + stage + "-v1"})
    return validate(state, stage)


def support(previous):
    validate(previous, "onboard")
    context = previous["context"]
    request = context["service_request"]
    query = tokens(request["question"])
    matches = []
    for policy in context["policy_documents"]:
        if policy["service"].casefold() != request["service"].casefold():
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", policy["text"]):
            sentence = sentence.strip()
            score = len(query & tokens(sentence))
            if score and 0 < len(sentence) <= 240:
                matches.append((-score, policy["policy_id"], sentence))
    citations = []
    for _, policy_id, quote in sorted(set(matches))[:3]:
        citations.append({"policy_id": policy_id, "quote": quote})
    result = {
        "answer": " ".join(row["quote"] for row in citations) if citations else (
            "The supplied policies do not answer this question. Ask a service officer."),
        "citations": citations,
        "unanswered_question": None if citations else request["question"],
        "handoff_next_step": previous["results"]["onboard"]["next_step"],
    }
    return advance(previous, "onboard", "support", result)


def research_result_for(support_result):
    evidence = copy.deepcopy(support_result["citations"])
    return {
        "evidence": evidence,
        "finding": ("Review these policy excerpts with a service officer." if evidence else
                    "There is not enough supplied evidence to answer the question."),
        "limitations": [
            "Keyword matches may not answer every part of the question.",
            "Policies may be incomplete, outdated, or in conflict.",
            "These excerpts do not establish benefit eligibility.",
        ],
        "next_step": (support_result["handoff_next_step"] if evidence else
                      "Ask a service officer for the missing policy information."),
    }


def research(previous):
    validate(previous, "support")
    return advance(previous, "support", "research",
                   research_result_for(previous["results"]["support"]))


def recommendation_result(context, research_result):
    evidence_ids = {row["policy_id"] for row in research_result["evidence"]}
    items = []
    for policy in sorted(context["policy_documents"], key=lambda p: p["policy_id"]):
        if policy["policy_id"] not in evidence_ids:
            continue
        items.append({
            "program_id": policy["program"]["program_id"],
            "title": policy["program"]["title"],
            "next_step": policy["program"]["next_step"],
            "source_ids": [policy["policy_id"]],
            "reason": "This service matches your request and has cited policy evidence.",
        })
    return {
        "items": items,
        "next_step": research_result["next_step"],
        "review_required": True,
        "fallback": None if items else "Ask a service officer to help find the right service.",
    }


def recommend(previous):
    validate(previous, "research")
    return advance(previous, "research", "recommend",
                   recommendation_result(previous["context"], previous["results"]["research"]))


def run_pipeline(form):
    return recommend(research(support(onboard(form))))


def reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON field.")
        value[key] = item
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Provide one government form JSON file, or '-'.")
        if argv[0] == "-":
            raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        else:
            with open(argv[0], "rb") as handle:
                raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "Input exceeds the size limit.")
        form = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
        output = run_pipeline(form)
    except ValidationError as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 2
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        print(json.dumps({"status": "error", "message": "Could not read a valid government form."}))
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
