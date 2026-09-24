"""Synthetic-only, FHIR-style administrative assistance. Not clinical software.

Run: python -B implementation.py example_input.json
The same envelope and validator govern input and every stage handoff. Identifiers
are synthetic aliases, free text uses a bounded vocabulary, and clinical notes
use a canonical template. These are demonstrative de-identification controls,
not a HIPAA certification or a sanitizer for arbitrary real patient records.
"""

import copy
import difflib
import json
import math
import re
import sys


PHASES = ("onboarding", "support", "search", "recommend")
TOPICS = ("authorization", "documents", "portal", "billing", "records")
FORMATS = ("text", "audio", "video")
DIAGNOSES = {"E11.9": "Type 2 diabetes mellitus", "I10": "Hypertension",
             "J45.909": "Asthma", "Z00.00": "Routine examination"}
LABS = {"4548-4": ("Hemoglobin A1c", "%"),
        "8480-6": ("Systolic blood pressure", "mm[Hg]"),
        "2093-3": ("Total cholesterol", "mg/dL")}
TAG = {"tag": [{"system": "urn:synthetic-fixture", "code": "synthetic"}]}
DOCUMENTS = ("clinical-note", "insurance-summary", "referral")
ALIASES = {
    "insurance": "authorization", "approval": "authorization",
    "preauthorization": "authorization", "precertification": "authorization",
    "forms": "documents", "paperwork": "documents", "checklist": "documents",
    "document": "documents", "upload": "documents",
    "login": "portal", "signin": "portal", "account": "portal",
    "price": "billing", "cost": "billing", "costs": "billing",
    "invoice": "billing", "payment": "billing",
    "chart": "records", "charts": "records", "note": "records",
    "notes": "records", "lab": "records", "labs": "records",
}
STOPWORDS = set("a an the i my me we our you your do does can could how what where "
                "when is are to for of with and or please need want find get about "
                "prior track tracking help see use access status online have in on "
                "this that it be should give tell would like explain".split())
CLINICAL = set("dose dosage insulin medication medicine prescribe prescription "
               "diagnose diagnosis treatment treat stop start increase decrease "
               "symptom symptoms pain chest emergency urgent".split())
VOCABULARY = (set(TOPICS) | set(ALIASES) | STOPWORDS | CLINICAL |
              set("guide quickstart accessible audio text video support human "
                  "navigator referral coverage benefits submit process review "
                  "unicorn mystery astronaut".split()))
KB = {
    "authorization": ("kb-authorization-v1",
                      "Prior authorization is reviewed by a payer and a human care team. "
                      "Gather the requested documents and ask the team to check submission "
                      "status. This tool cannot approve coverage or submit requests."),
    "documents": ("kb-documents-v1",
                  "Use the document checklist to identify missing materials. A human "
                  "care team must review clinical notes and referrals before submission."),
    "portal": ("kb-portal-v1",
               "Use the portal quickstart for account navigation. Contact the care "
               "team for access problems; never share passwords or patient identifiers here."),
    "billing": ("kb-billing-v1",
                "The billing guide explains administrative questions to ask your payer. "
                "Costs and coverage require confirmation by a human representative."),
    "records": ("kb-records-v1",
                "The records guide explains how to prepare a records-access request. "
                "A human care team must verify identity outside this synthetic tool."),
    "clinical": ("kb-safety-v1",
                 "I cannot diagnose, interpret lab values, change medication, or recommend "
                 "treatment. Contact a qualified clinician. For a possible emergency, "
                 "contact local emergency services."),
    "unknown": ("kb-fallback-v1",
                "The available administrative knowledge does not answer this question. "
                "A human care navigator must review it; no answer has been invented."),
}
CATALOG = (
    {"id": "resource-auth-checklist", "title": "Authorization document checklist",
     "topics": ["authorization", "documents"], "format": "text",
     "languages": ["en", "es"], "accessible": True},
    {"id": "resource-auth-audio", "title": "Audio authorization navigation guide",
     "topics": ["authorization"], "format": "audio",
     "languages": ["en"], "accessible": True},
    {"id": "resource-portal", "title": "Portal quickstart video",
     "topics": ["portal"], "format": "video",
     "languages": ["en", "es"], "accessible": False},
    {"id": "resource-portal-text", "title": "Accessible portal quickstart",
     "topics": ["portal"], "format": "text",
     "languages": ["en", "es"], "accessible": True},
    {"id": "resource-records", "title": "Clinical records preparation guide",
     "topics": ["records", "documents"], "format": "text",
     "languages": ["en", "es"], "accessible": True},
    {"id": "resource-billing", "title": "Billing and coverage questions",
     "topics": ["billing"], "format": "text",
     "languages": ["en"], "accessible": True},
)
BY_ID = {item["id"]: item for item in CATALOG}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            label + " has missing or unsupported fields")


def choice(value, allowed, label):
    require(isinstance(value, str) and value in allowed, label + " is unsupported")


def selection(value, allowed, label, maximum=10):
    require(isinstance(value, list) and len(value) <= maximum, label + " must be a bounded list")
    for item in value:
        choice(item, allowed, label)
    require(len(value) == len(set(value)), label + " contains duplicates")


def canonical_note(patient):
    diagnoses = ", ".join(item["code"] for item in patient["diagnoses"]) or "none"
    labs = ", ".join(item["code"] + "=" + str(item["value"]) + " " + item["unit"]
                     for item in patient["labs"]) or "none"
    return ("Synthetic clinical note. Diagnosis codes: " + diagnoses + ". Labs: " + labs +
            ". Administrative context only; clinician review required.")


def tokens(text):
    """Reject identifiers instead of attempting an unsafe arbitrary-text PHI scrub."""
    require(isinstance(text, str) and len(text) <= 500, "Request text must be at most 500 characters")
    require(re.fullmatch(r"[A-Za-z\s?.,!'-]*", text) is not None,
            "Request text must contain only synthetic administrative vocabulary")
    result = []
    for token in re.findall(r"[a-z]+", text.lower()):
        if token not in VOCABULARY:
            matches = difflib.get_close_matches(token, sorted(VOCABULARY), n=1, cutoff=0.84)
            require(bool(matches), "Request contains unsupported or potentially identifying text")
            token = matches[0]
        result.append(token)
    return result


def search_terms(text):
    return sorted({ALIASES.get(token, token) for token in tokens(text)
                   if token not in STOPWORDS})


def onboarding_payload(state):
    records, customer = state["records"], state["customer"]
    authorization = records["prior_authorization_request"]
    missing = sorted(set(authorization["required_documents"]) -
                     set(authorization["provided_documents"]))
    audience = "caregiver" if records["patient"]["age_band"] == "child" else "customer"
    if customer["experience"] == "new":
        step = "Review the " + customer["goal"] + " orientation as the " + audience + "."
    else:
        step = "Continue the " + customer["goal"] + " workflow as the " + audience + "."
    if customer["goal"] in ("authorization", "documents") and missing:
        step += " Prepare " + missing[0] + " for human review."
    else:
        step += " Ask a human navigator to verify the next administrative step."
    return {"consumed_stage": "input", "patient_reference": "Patient/" + records["patient"]["id"],
            "authorization_reference": "Claim/" + authorization["id"],
            "goal": customer["goal"], "next_step": step, "missing_documents": missing,
            "preferred_format": customer["preferred_format"],
            "human_review_required": True}


def support_payload(state):
    onboarding = state["stages"]["onboarding"]
    question = tokens(state["request"]["question"])
    query = tokens(state["request"]["search_query"])
    if (set(question) | set(query)) & CLINICAL:
        intent = "clinical"
    else:
        relevant = search_terms(state["request"]["question"])
        intent = next((topic for topic in TOPICS if topic in relevant), None)
        if intent is None:
            intent = "unknown" if relevant else onboarding["goal"]
    source_id, answer = KB[intent]
    if intent == "clinical":
        terms = ["records"]
    else:
        terms = search_terms(state["request"]["search_query"])
        if not terms:
            terms = search_terms(state["request"]["question"]) or [onboarding["goal"]]
    return {"consumed_stage": "onboarding", "onboarding_next_step": onboarding["next_step"],
            "intent": intent, "answer": answer, "source_id": source_id,
            "search_terms": terms, "escalated": intent in ("clinical", "unknown"),
            "availability": "offline-knowledge", "answer_language": "en",
            "human_review_required": True}


def search_payload(state):
    support, customer = state["stages"]["support"], state["customer"]
    terms = set(support["search_terms"])
    results = []
    for item in CATALOG:
        if customer["language"] not in item["languages"]:
            continue
        if customer["accessibility"] == "accessible" and not item["accessible"]:
            continue
        matched = sorted(terms & set(item["topics"]))
        if not matched:
            continue
        score = 10 * len(matched) + (2 if support["intent"] in item["topics"] else 0)
        results.append({"resource_id": item["id"], "title": item["title"], "score": score,
                        "matched_terms": matched, "kind": "administrative-resource",
                        "human_review_required": True})
    results.sort(key=lambda item: (-item["score"], item["resource_id"]))
    return {"consumed_stage": "support", "support_source_id": support["source_id"],
            "query_terms": support["search_terms"], "results": results,
            "no_results": not results, "human_review_required": True}


def recommendation_payload(state):
    search, customer = state["stages"]["search"], state["customer"]
    recommendations = []
    for result in search["results"]:
        item = BY_ID[result["resource_id"]]
        reasons = ["Matches the validated support-to-search handoff."]
        score = result["score"]
        if item["format"] == customer["preferred_format"]:
            score += 5
            reasons.append("Matches preferred " + item["format"] + " format.")
        interests = sorted(set(item["topics"]) & set(customer["interests"]))
        if interests:
            score += 3 * len(interests)
            reasons.append("Matches stated interests: " + ", ".join(interests) + ".")
        recommendations.append({"resource_id": item["id"], "title": item["title"],
                                "score": score, "reasons": reasons,
                                "human_review_required": True})
    recommendations.sort(key=lambda item: (-item["score"], item["resource_id"]))
    recommendations = recommendations[:state["request"]["limit"]]
    next_step = ("Review these administrative resources with a human care navigator."
                 if recommendations else
                 "Ask a human care navigator to locate a suitable administrative resource.")
    return {"consumed_stage": "search",
            "search_result_ids": [item["resource_id"] for item in search["results"]],
            "recommendations": recommendations, "next_step": next_step,
            "clinical_decision": "not-made", "human_review_required": True}


BUILDERS = dict(zip(PHASES, (onboarding_payload, support_payload,
                           search_payload, recommendation_payload)))


def mutations(state, phase):
    if phase == "onboarding":
        return [("workflow_status", "onboarding-ready")]
    if phase == "support":
        status = "support-escalated" if state["stages"]["support"]["escalated"] else "support-ready"
        return [("workflow_status", status)]
    if phase == "search":
        ids = [item["resource_id"] for item in state["stages"]["search"]["results"]]
        return [("suggested_resource_ids", ids),
                ("workflow_status", "resources-found" if ids else "resources-unavailable")]
    return [("workflow_status", "awaiting-human-review")]


def audit_event(sequence, phase, record_id, field, before, after):
    return {"sequence": sequence, "stage": phase, "record_reference": "Claim/" + record_id,
            "field": field, "before": copy.deepcopy(before), "after": copy.deepcopy(after),
            "actor": "administrative-assistant", "human_review_required": True}


def strict_equal(left, right):
    """JSON equality must not equate True with 1 or silently accept NaN."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def validate_envelope(state, expected_phase=None):
    keys(state, ("schema_version", "status", "synthetic", "human_review_required",
                 "records", "customer", "request", "stages", "audit"), "Envelope")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "Unsupported schema version")
    require(state["status"] == "ok" and state["synthetic"] is True and
            state["human_review_required"] is True, "Synthetic and human-review flags are mandatory")
    records = state["records"]
    keys(records, ("patient", "clinical_note", "prior_authorization_request"), "Records")
    patient = records["patient"]
    keys(patient, ("resourceType", "id", "meta", "age_band", "diagnoses", "labs",
                   "human_review_required"), "Patient")
    note = records["clinical_note"]
    keys(note, ("resourceType", "id", "meta", "subject", "text", "human_review_required"),
         "Clinical note")
    authorization = records["prior_authorization_request"]
    keys(authorization, ("resourceType", "id", "meta", "patient", "purpose", "status",
                         "service_code", "required_documents", "provided_documents",
                         "workflow_status", "suggested_resource_ids", "human_review_required"),
         "Prior authorization request")
    for record, resource_type, prefix in (
            (patient, "Patient", "patient"), (note, "DocumentReference", "note"),
            (authorization, "Claim", "authorization")):
        require(record["resourceType"] == resource_type, "Unexpected resourceType")
        require(isinstance(record["id"], str) and
                re.fullmatch("synthetic-" + prefix + r"-[0-9]{3}", record["id"]) is not None,
                "Only explicitly synthetic record aliases are permitted")
        require(record["meta"] == TAG and record["human_review_required"] is True,
                "Every record needs synthetic tagging and human review")
    choice(patient["age_band"], ("child", "adult", "older-adult"), "Age band")
    require(isinstance(patient["diagnoses"], list) and len(patient["diagnoses"]) <= 4,
            "Diagnoses must be a bounded list")
    diagnosis_codes = []
    for diagnosis in patient["diagnoses"]:
        keys(diagnosis, ("code", "display"), "Diagnosis")
        choice(diagnosis["code"], DIAGNOSES, "Diagnosis code")
        require(diagnosis["display"] == DIAGNOSES[diagnosis["code"]], "Diagnosis display mismatch")
        diagnosis_codes.append(diagnosis["code"])
    require(len(set(diagnosis_codes)) == len(diagnosis_codes), "Duplicate diagnosis")
    require(isinstance(patient["labs"], list) and len(patient["labs"]) <= 3,
            "Labs must be a bounded list")
    lab_codes = []
    for lab in patient["labs"]:
        keys(lab, ("code", "display", "value", "unit"), "Lab")
        choice(lab["code"], LABS, "Lab code")
        display, unit = LABS[lab["code"]]
        require(lab["display"] == display and lab["unit"] == unit, "Lab description mismatch")
        require(type(lab["value"]) in (int, float) and math.isfinite(lab["value"]) and
                0 <= lab["value"] <= 10000, "Lab must contain a bounded finite numeric fixture value")
        lab_codes.append(lab["code"])
    require(len(set(lab_codes)) == len(lab_codes), "Duplicate lab")
    reference = {"reference": "Patient/" + patient["id"]}
    require(note["subject"] == reference and authorization["patient"] == reference,
            "Patient references must match")
    require(note["text"] == canonical_note(patient), "Clinical note must use the synthetic-only template")
    require(authorization["purpose"] == "preauthorization" and authorization["status"] == "draft",
            "Autonomous authorization submission or approval is forbidden")
    choice(authorization["service_code"], ("specialist-visit", "diagnostic-test", "care-coordination"),
           "Service code")
    selection(authorization["required_documents"], DOCUMENTS, "Required documents")
    selection(authorization["provided_documents"], authorization["required_documents"], "Provided documents")
    selection(authorization["suggested_resource_ids"], BY_ID, "Suggested resources")
    customer = state["customer"]
    keys(customer, ("experience", "goal", "preferred_format", "language",
                    "accessibility", "interests"), "Customer")
    choice(customer["experience"], ("new", "returning"), "Experience")
    choice(customer["goal"], TOPICS, "Goal")
    choice(customer["preferred_format"], FORMATS, "Preferred format")
    choice(customer["language"], ("en", "es"), "Language")
    choice(customer["accessibility"], ("standard", "accessible"), "Accessibility")
    selection(customer["interests"], TOPICS, "Interests")
    request = state["request"]
    keys(request, ("question", "search_query", "limit"), "Request")
    tokens(request["question"])
    tokens(request["search_query"])
    require(type(request["limit"]) is int and 1 <= request["limit"] <= 5, "Limit must be 1 through 5")
    stages = state["stages"]
    require(isinstance(stages, dict), "Stages must be an object")
    count = len(stages)
    require(count <= len(PHASES) and set(stages) == set(PHASES[:count]),
            "Stages must form an uninterrupted pipeline prefix")
    actual_phase = PHASES[count - 1] if count else "input"
    require(expected_phase is None or expected_phase == actual_phase, "Unexpected pipeline phase")
    require(isinstance(state["audit"], list), "Audit must be a list")
    expected_audit = []
    shadow = {"workflow_status": "new", "suggested_resource_ids": []}
    for phase in PHASES[:count]:
        expected = BUILDERS[phase](state)
        require(strict_equal(stages[phase], expected), "Invalid or ungrounded " + phase + " handoff")
        for field, after in mutations(state, phase):
            before = shadow[field]
            if before != after:
                expected_audit.append(audit_event(len(expected_audit) + 1, phase,
                                                  authorization["id"], field, before, after))
                shadow[field] = copy.deepcopy(after)
    require(strict_equal(state["audit"], expected_audit), "Audit trail is missing or inconsistent")
    require(all(strict_equal(authorization[field], value) for field, value in shadow.items()),
            "Record changes must match the complete audit trail")
    return state


def advance(state, phase, responder=None):
    require(phase in PHASES, "Unknown pipeline phase")
    index = PHASES.index(phase)
    validate_envelope(state, "input" if index == 0 else PHASES[index - 1])
    result = copy.deepcopy(state)
    payload = BUILDERS[phase](result)
    if phase == "support" and responder is not None:
        evidence = {"intent": payload["intent"], "answer": payload["answer"],
                    "source_id": payload["source_id"]}
        try:
            proposal = responder(copy.deepcopy(evidence))
        except Exception as error:
            raise ValidationError("Injected responder failed") from error
        keys(proposal, ("answer", "source_id"), "Injected response")
        require(proposal == {"answer": evidence["answer"], "source_id": evidence["source_id"]},
                "Injected response must match the approved grounded answer")
        payload.update(proposal)
    result["stages"][phase] = payload
    authorization = result["records"]["prior_authorization_request"]
    for field, after in mutations(result, phase):
        before = authorization[field]
        if before != after:
            result["audit"].append(audit_event(len(result["audit"]) + 1, phase,
                                               authorization["id"], field, before, after))
            authorization[field] = copy.deepcopy(after)
    return validate_envelope(result, phase)


def run_pipeline(data, responder=None):
    validate_envelope(data, "input")
    result = data
    for phase in PHASES:
        result = advance(result, phase, responder)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON number")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], "r", encoding="utf-8") as handle:
            text = handle.read(131073)
        require(len(text) <= 131072, "Input exceeds the reference implementation size limit")
        data = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = run_pipeline(data)
        exit_code = 0
    except (OSError, UnicodeError):
        output = {"status": "error", "error": "Input file could not be read as UTF-8 JSON"}
        exit_code = 2
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Do not echo rejected text, paths or identifiers into error messages.
        output = {"status": "error", "error": "Input failed JSON, schema, privacy or workflow validation"}
        exit_code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
