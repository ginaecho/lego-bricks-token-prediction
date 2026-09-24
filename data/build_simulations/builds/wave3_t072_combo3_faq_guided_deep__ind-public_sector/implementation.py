"""Synthetic public-sector reference pipeline. Python standard library only."""

import copy
import json
import re
import sys
from pathlib import Path


BUILD_ID = "wave3_t072_combo3_faq_guided_deep__ind-public_sector"
RULES = ("income_limit", "residency_months", "required_documents")
DOCUMENTS = {"identity", "residency", "income"}
LABELS = {
    "income_limit": "Monthly income limit",
    "residency_months": "Months of residency needed",
    "required_documents": "Documents needed",
}
QUESTIONS = {
    "what are the eligibility rules?": ("income_limit", "residency_months"),
    "what documents do i need?": ("required_documents",),
}
PII = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{3}-\d{2}-\d{4}\b"
    r"|\b(?:\+?\d{1,2}[- .])?\d{3}[- .]\d{3}[- .]\d{4}\b"
)


class ValidationError(ValueError):
    """A public-safe validation failure; raw input is never included."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected):
    require(type(value) is dict and set(value) == set(expected),
            "Object fields do not match the shared schema.")


def text(value, limit=2000):
    require(type(value) is str and 0 < len(value.strip()) <= limit,
            "A text field is empty, too long, or invalid.")
    require(not any(ord(char) < 32 and char not in "\n\t" for char in value),
            "Control characters are not allowed.")


def public_text(value):
    text(value)
    require(not PII.search(value), "Public text must not contain personal contact data.")


def integer(value, maximum):
    require(type(value) is int and 0 <= value <= maximum,
            "A numeric field is outside its allowed range.")


def identifier(value, pattern):
    require(type(value) is str and re.fullmatch(pattern, value) is not None,
            "An identifier is invalid.")


def document_list(value):
    require(type(value) is list and all(type(x) is str for x in value),
            "Documents must be a list of supported document names.")
    require(len(value) == len(set(value)) and set(value) <= DOCUMENTS,
            "Documents contain duplicates or unsupported names.")


def parse_policy(policy):
    """Parse the small, documented regulation-text grammar; never infer a rule."""
    facts = {}
    for line in policy["text"].splitlines():
        if ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        if key not in RULES:
            continue
        require(key not in facts, "A policy repeats a rule.")
        if key == "required_documents":
            value = [part.strip() for part in raw.split(",")]
            document_list(value)
            require(bool(value), "A document rule cannot be empty.")
            value = sorted(value)
        else:
            require(re.fullmatch(r"\d{1,7}", raw) is not None,
                    "A policy rule must use a whole number.")
            value = int(raw)
            integer(value, 1000000 if key == "income_limit" else 1200)
        facts[key] = {"value": value, "quote": line.strip()}
    return facts


def validate_context(context):
    fields(context, ("case_number", "persona_id", "program", "question",
                     "monthly_income", "residency_months", "documents", "policies"))
    identifier(context["case_number"], r"SYN-CASE-\d{4}")
    identifier(context["persona_id"], r"SYN-PERSON-\d{3}")
    identifier(context["program"], r"synthetic-[a-z-]{1,40}")
    public_text(context["question"])
    integer(context["monthly_income"], 1000000)
    integer(context["residency_months"], 1200)
    document_list(context["documents"])
    policies = context["policies"]
    require(type(policies) is list and len(policies) <= 30,
            "Policies must be a bounded list.")
    seen = set()
    for policy in policies:
        fields(policy, ("id", "program", "title", "text", "synthetic"))
        identifier(policy["id"], r"SYN-POL-[A-Z0-9-]{1,30}")
        require(policy["id"] not in seen, "Policy IDs must be unique.")
        seen.add(policy["id"])
        identifier(policy["program"], r"synthetic-[a-z-]{1,40}")
        require(policy["synthetic"] is True, "Only synthetic policies are accepted.")
        public_text(policy["title"])
        public_text(policy["text"])
        parse_policy(policy)


def input_context(value):
    fields(value, ("schema_version", "synthetic", "service_request",
                   "benefits_application", "policy_documents"))
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Only schema version 1 is supported.")
    require(value["synthetic"] is True, "Only clearly labeled synthetic fixtures are accepted.")
    request = value["service_request"]
    application = value["benefits_application"]
    fields(request, ("case_number", "question", "citizen"))
    citizen = request["citizen"]
    fields(citizen, ("persona_id", "display_name", "address", "synthetic"))
    require(citizen["synthetic"] is True, "Only synthetic citizens are accepted.")
    text(citizen["display_name"], 120)
    text(citizen["address"], 200)
    require(citizen["display_name"].startswith("Synthetic "),
            "The fabricated citizen name must start with Synthetic.")
    require(citizen["address"].endswith(", Fictional District, ZZ 00000"),
            "Use the non-traceable synthetic address format.")
    fields(application, ("case_number", "program", "monthly_income",
                         "residency_months", "documents"))
    require(request["case_number"] == application["case_number"],
            "The service request and application must use the same case.")
    context = {
        "case_number": request["case_number"],
        "persona_id": citizen["persona_id"],
        "question": request["question"],
        **{key: application[key] for key in (
            "program", "monthly_income", "residency_months", "documents")},
        "policies": value["policy_documents"],
    }
    validate_context(context)
    # Even fabricated identity fields must not leak into public text or callbacks.
    public_parts = [context["question"]]
    for policy in context["policies"]:
        public_parts.extend((policy["title"], policy["text"]))
    for part in public_parts:
        for private in (citizen["display_name"], citizen["address"]):
            require(private.casefold() not in part.casefold(),
                    "Citizen identity data must not be copied into public text.")
    return copy.deepcopy(context)


def evidence_for(context):
    evidence = []
    for policy in sorted(context["policies"], key=lambda p: p["id"]):
        if policy["program"] != context["program"]:
            continue
        for rule, fact in sorted(parse_policy(policy).items()):
            evidence.append({
                "id": policy["id"] + ":" + rule,
                "policy_id": policy["id"],
                "rule": rule,
                "value": fact["value"],
                "quote": fact["quote"],
            })
    return evidence


def groups(evidence):
    result = {}
    for rule in RULES:
        sources = [item for item in evidence if item["rule"] == rule]
        variants = {}
        for source in sources:
            key = json.dumps(source["value"], sort_keys=True)
            variants.setdefault(key, {"value": source["value"], "evidence_ids": []})
            variants[key]["evidence_ids"].append(source["id"])
        result[rule] = list(variants.values())
    return result


def faq_data(context, evidence):
    requested = QUESTIONS.get(context["question"].strip().casefold())
    catalog = groups(evidence)
    missing = [] if requested is None else [
        rule for rule in requested if len(catalog[rule]) != 1]
    if requested is None or missing:
        reason = ("This question is outside the supported policy topics."
                  if requested is None else
                  "The policies are missing a rule or do not agree.")
        return {
            "status": "abstained", "answer": "I cannot give a supported answer. " + reason,
            "requested_rules": list(requested or ()), "facts": {},
            "evidence_ids": [], "reason": reason,
        }
    facts = {rule: catalog[rule][0]["value"] for rule in requested}
    statements = []
    for rule, value in facts.items():
        rendered = ", ".join(value) if type(value) is list else str(value)
        statements.append(LABELS[rule] + ": " + rendered + ".")
    return {
        "status": "answered", "answer": " ".join(statements),
        "requested_rules": list(requested), "facts": facts,
        "evidence_ids": [
            item["id"] for item in evidence if item["rule"] in requested],
        "reason": "Each stated rule is supported by the cited synthetic policies.",
    }


def guided_data(context, faq, evidence):
    catalog = groups(evidence)
    policy_ready = faq["status"] == "answered" and all(
        len(catalog[rule]) == 1 for rule in RULES)
    required = catalog["required_documents"][0]["value"] if policy_ready else []
    missing = sorted(set(required) - set(context["documents"]))
    steps = [
        {"id": "policy", "prerequisites": [],
         "status": "complete" if policy_ready else "blocked",
         "message": "Policy rules are ready." if policy_ready else
                    "Ask a case worker to check missing or conflicting policy rules."},
        {"id": "documents", "prerequisites": ["policy"],
         "status": ("complete" if not missing else "pending") if policy_ready else "blocked",
         "message": ("Documents are ready." if not missing else
                     "Provide these documents: " + ", ".join(missing) + ".")
                    if policy_ready else "First resolve the policy questions."},
        {"id": "review", "prerequisites": ["documents"],
         "status": "pending" if policy_ready and not missing else "blocked",
         "message": "A case worker must review the application. This tool cannot approve it."},
    ]
    checks = []
    if policy_ready:
        for rule, field in (("income_limit", "monthly_income"),
                            ("residency_months", "residency_months")):
            limit = catalog[rule][0]["value"]
            meets = context[field] <= limit if rule == "income_limit" else context[field] >= limit
            checks.append({
                "rule": rule, "observed": context[field], "threshold": limit,
                "result": "meets_stated_rule" if meets else "needs_case_worker_review",
                "evidence_ids": catalog[rule][0]["evidence_ids"],
            })
    return {
        "status": "ready_for_review" if policy_ready and not missing else
                  ("in_progress" if policy_ready else "blocked"),
        "source_faq_status": faq["status"], "steps": steps, "missing_documents": missing,
        "checks": checks, "completed_steps": sum(s["status"] == "complete" for s in steps),
        "total_steps": len(steps), "decision": "not_made",
    }


def deep_data(context, guided, evidence):
    catalog = groups(evidence)
    agreements = []
    disagreements = []
    unresolved = []
    for rule, variants in catalog.items():
        if not variants:
            unresolved.append("Which policy defines " + LABELS[rule].lower() + "?")
        elif len(variants) > 1:
            disagreements.append({"rule": rule, "alternatives": variants})
            unresolved.append("Which " + LABELS[rule].lower() + " rule should a case worker use?")
        else:
            agreements.append({"rule": rule, **variants[0]})
    if guided["missing_documents"]:
        unresolved.append("Can the citizen provide the missing documents?")
    if guided["source_faq_status"] == "abstained":
        unresolved.append("Can a case worker answer the original service question?")
    if any(check["result"] == "needs_case_worker_review" for check in guided["checks"]):
        unresolved.append("Are exceptions or other benefits available for this application?")
    unresolved.append("What is the case worker's final decision?")
    count = len({item["policy_id"] for item in evidence})
    return {
        "status": "needs_review", "source_guided_status": guided["status"],
        "documents_used": count, "agreements": agreements, "disagreements": disagreements,
        "unresolved_questions": unresolved,
        "summary": "Compared " + str(count) + " synthetic policy documents. "
                   + str(len(disagreements)) + " rules have conflicting evidence. "
                   "A case worker must resolve open questions.",
        "limitations": ["Only supplied synthetic policies were used.",
                       "Repeated policy text is not independent proof.",
                       "No benefit approval or denial was made."],
    }


def expected_data(context, stage):
    evidence = evidence_for(context)
    data = {"faq": faq_data(context, evidence)}
    if stage in ("guided", "deep"):
        data["guided"] = guided_data(context, data["faq"], evidence)
    if stage == "deep":
        data["deep"] = deep_data(context, data["guided"], evidence)
    return evidence, data


def envelope(context, stage, data):
    evidence = evidence_for(context)
    return {
        "schema_version": 1, "synthetic": True, "stage": stage, "status": "ok",
        "context": copy.deepcopy(context), "data": copy.deepcopy(data), "evidence": evidence,
        "explanation": [
            "Synthetic demonstration only. This is not a legal or benefits decision.",
            "Citizen names and addresses are removed before any stage runs.",
            "Rule checks include policy citations for open-records review.",
        ],
        "unresolved_questions": data.get("deep", {}).get("unresolved_questions", []),
    }


def validate(value, kind):
    """One shared boundary validator for input and every cumulative stage output."""
    if kind == "input":
        return input_context(value)
    require(kind in ("faq", "guided", "deep"), "Unknown validation boundary.")
    fields(value, ("schema_version", "synthetic", "stage", "status", "context",
                   "data", "evidence", "explanation", "unresolved_questions"))
    validate_context(value["context"])
    evidence, data = expected_data(value["context"], kind)
    expected = envelope(value["context"], kind, data)
    # Canonical JSON comparison distinguishes booleans from integer counters.
    try:
        same = json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        same = False
    require(same, "Stage output is not grounded in the validated shared context.")
    return copy.deepcopy(value)


def faq_stage(raw_input, answer_callable=None):
    context = validate(raw_input, "input")
    evidence, data = expected_data(context, "faq")
    if answer_callable is not None:
        require(callable(answer_callable), "The answer hook must be callable.")
        safe_request = {"question": context["question"], "evidence": evidence,
                        "expected_answer": data["faq"]["answer"],
                        "expected_evidence_ids": data["faq"]["evidence_ids"]}
        try:
            result = answer_callable(copy.deepcopy(safe_request))
        except Exception:
            raise ValidationError("The optional answer hook failed.") from None
        fields(result, ("answer", "evidence_ids"))
        require(result["answer"] == data["faq"]["answer"] and
                result["evidence_ids"] == data["faq"]["evidence_ids"],
                "The optional answer hook returned an unsupported answer.")
    return validate(envelope(context, "faq", data), "faq")


def guided_stage(previous):
    previous = validate(previous, "faq")
    data = previous["data"]
    data["guided"] = guided_data(previous["context"], data["faq"], previous["evidence"])
    return validate(envelope(previous["context"], "guided", data), "guided")


def deep_stage(previous):
    previous = validate(previous, "guided")
    data = previous["data"]
    data["deep"] = deep_data(previous["context"], data["guided"], previous["evidence"])
    return validate(envelope(previous["context"], "deep", data), "deep")


def run_pipeline(raw_input, answer_callable=None):
    return deep_stage(guided_stage(faq_stage(raw_input, answer_callable)))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object fields are not allowed.")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 1000000, "The input file is too large.")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                         parse_constant=lambda _: (_ for _ in ()).throw(
                             ValidationError("Non-finite JSON numbers are not allowed.")))
        output = run_pipeline(raw)
    except ValidationError as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 2
    except (OSError, UnicodeError, ValueError, RecursionError):
        print(json.dumps({"status": "error", "message": "Cannot read a valid input JSON file."}))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
