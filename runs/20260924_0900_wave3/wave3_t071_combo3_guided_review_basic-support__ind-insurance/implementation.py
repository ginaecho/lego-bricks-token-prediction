"""Deterministic synthetic insurance onboarding -> review -> support reference CLI."""
import copy
import datetime as dt
import json
import math
import re
import sys
import xml.etree.ElementTree as ET


class ValidationError(ValueError):
    pass


STEPS = ("confirm_synthetic_data", "confirm_policy", "confirm_claim", "confirm_disclosures")
REQUIREMENTS = {
    "policy_schedule": "Policy schedule identifying the policy",
    "claim_form": "Claim form identifying the claim",
    "loss_evidence": "Loss evidence identifying the claim",
    "underwriting_disclosure": "Disclosure identifying every underwriting factor",
}
DISCLAIMER = "Synthetic demonstration only; no compliance certification or claim approval."


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), path="input"):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(required) <= value.keys(), f"{path} missing fields: {sorted(set(required) - value.keys())}")
    require(value.keys() <= set(required) | set(optional),
            f"{path} contains unsupported fields; minimize policyholder data")


def text(value, path, limit=2000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            f"{path} must be nonempty text, at most {limit} characters")
    # Demonstrative guard only: arbitrary text cannot be proven anonymous.
    require(not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{3}-\d{2}-\d{4}\b", value),
            f"{path} contains disallowed personal contact/identifier data")
    return value.strip()


def identifier(value, path):
    text(value, path, 64)
    require(re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None, f"{path} invalid identifier")
    return value


def number(value, path, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"{path} must be a finite number >= {minimum}")
    return value


def date(value, path):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
            f"{path} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise ValidationError(f"{path} invalid date") from None


def parse_submission(raw):
    fields(raw, ("format", "content"), path="submission")
    require(raw["format"] in ("acord_json", "acord_xml"), "unsupported submission format")
    if raw["format"] == "acord_json":
        fields(raw["content"], ("ACORD",), path="submission.content")
        return copy.deepcopy(raw["content"]["ACORD"])
    xml = text(raw["content"], "submission XML", 20000)
    require("<!DOCTYPE" not in xml.upper() and "<!ENTITY" not in xml.upper(), "DTD/entities forbidden")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise ValidationError("malformed ACORD-style XML") from None
    require(root.tag == "ACORD" and not root.attrib, "expected ACORD root without attributes")
    expected = {"SubmissionId", "PolicyId", "Factors"}
    require(len(root) == 3 and {node.tag for node in root} == expected, "invalid ACORD XML fields")
    for node in root:
        require(not node.attrib, "unexpected XML attributes")
        if node.tag != "Factors":
            require(len(node) == 0, "unexpected nested XML")
    factors = []
    for node in root.find("Factors"):
        require(node.tag == "Factor" and len(node) == 0 and
                set(node.attrib) == {"name", "value", "disclosed"}, "invalid XML factor")
        require(node.attrib["disclosed"] in ("true", "false"), "invalid disclosed boolean")
        factors.append({"name": node.attrib["name"], "value": node.attrib["value"],
                        "disclosed": node.attrib["disclosed"] == "true"})
    return {"submission_id": root.findtext("SubmissionId"), "policy_id": root.findtext("PolicyId"),
            "factors": factors}


def parse_claim_form(raw):
    text(raw, "claim_form", 8000)
    result = {}
    for line in raw.splitlines():
        require(":" in line, "claim-form lines require key: value")
        key, value = line.split(":", 1)
        key = key.strip()
        require(key not in result, "duplicate claim-form field")
        result[key] = value.strip()
    fields(result, ("claim_id", "policy_id", "loss_date", "loss_amount", "description"),
           path="claim_form")
    try:
        result["loss_amount"] = float(result["loss_amount"])
    except ValueError:
        raise ValidationError("claim_form loss_amount must be numeric") from None
    return result


def validate_case(raw):
    """The sole external-input boundary, shared by every pipeline stage."""
    fields(raw, ("schema_version", "synthetic", "as_of", "policy", "claim_form", "submission",
                 "evidence", "onboarding", "support"))
    require(raw["schema_version"] == "1.0", "unsupported schema_version")
    require(raw["synthetic"] is True, "only clearly labeled synthetic data accepted")
    as_of = date(raw["as_of"], "as_of")
    p = copy.deepcopy(raw["policy"])
    fields(p, ("policy_id", "policyholder_ref", "effective_date", "expiry_date",
               "coverage_limit", "currency", "asset"), path="policy")
    identifier(p["policy_id"], "policy_id")
    identifier(p["policyholder_ref"], "policyholder_ref")
    require(p["policyholder_ref"].startswith("SYN-"), "policyholder reference must be synthetic")
    start, end = date(p["effective_date"], "effective_date"), date(p["expiry_date"], "expiry_date")
    require(start <= end, "policy date range reversed")
    number(p["coverage_limit"], "coverage_limit", 0.01)
    require(p["currency"] in ("USD", "EUR", "GBP"), "unsupported currency")
    fields(p["asset"], ("type", "reference"), path="asset")
    require(p["asset"]["type"] in ("vehicle", "property"), "unsupported asset type")
    asset = text(p["asset"]["reference"], "asset reference", 120)
    if p["asset"]["type"] == "vehicle":
        require(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", asset) is not None, "VIN must be 17 characters")
    else:
        require(asset.startswith("SYNTHETIC "), "property address must be labeled SYNTHETIC")
    c = parse_claim_form(raw["claim_form"])
    identifier(c["claim_id"], "claim_id")
    require(c["policy_id"] == p["policy_id"], "claim policy_id mismatch")
    require(date(c["loss_date"], "loss_date") <= as_of, "loss cannot be after as_of")
    number(c["loss_amount"], "loss_amount", 0.01)
    text(c["description"], "claim description")
    u = parse_submission(raw["submission"])
    fields(u, ("submission_id", "policy_id", "factors"), path="underwriting")
    identifier(u["submission_id"], "submission_id")
    require(u["policy_id"] == p["policy_id"], "submission policy_id mismatch")
    require(isinstance(u["factors"], list) and 0 < len(u["factors"]) <= 20, "factors must be a nonempty list")
    seen = set()
    for f in u["factors"]:
        fields(f, ("name", "value", "disclosed"), path="factor")
        require(isinstance(f["name"], str) and f["name"] in
                ("vehicle_age", "roof_age", "use_type", "deductible", "prior_claims"),
                "unsupported underwriting factor; protected attributes are not accepted")
        require(f["name"] not in seen, "duplicate underwriting factor")
        seen.add(f["name"])
        require(f["disclosed"] is True, "all underwriting factors must be disclosed, not hidden")
        if isinstance(f["value"], str):
            text(f["value"], "factor value", 80)
        else:
            number(f["value"], "factor value")
    require(isinstance(raw["evidence"], list) and len(raw["evidence"]) <= 100, "evidence must be a bounded list")
    evidence = copy.deepcopy(raw["evidence"])
    seen = set()
    for e in evidence:
        fields(e, ("evidence_id", "kind", "entity_id", "text"), path="evidence")
        identifier(e["evidence_id"], "evidence_id")
        require(e["evidence_id"] not in seen, "duplicate evidence_id")
        seen.add(e["evidence_id"])
        require(isinstance(e["kind"], str) and e["kind"] in REQUIREMENTS, "unsupported evidence kind")
        identifier(e["entity_id"], "evidence entity_id")
        text(e["text"], "evidence text", 8000)
    fields(raw["onboarding"], ("completed_steps",), path="onboarding")
    completed = raw["onboarding"]["completed_steps"]
    require(isinstance(completed, list) and completed == list(STEPS[:len(completed)]),
            "completed_steps must be an ordered prerequisite prefix without duplicates")
    fields(raw["support"], ("question", "team_online"), path="support")
    text(raw["support"]["question"], "support question", 1000)
    require(type(raw["support"]["team_online"]) is bool, "team_online must be boolean")
    return {"schema_version": "1.0", "synthetic": True, "as_of": raw["as_of"],
            "case_id": c["claim_id"], "policy": p, "claim": c, "underwriting": u,
            "evidence": evidence, "completed_steps": list(completed),
            "support_request": copy.deepcopy(raw["support"])}


def validate_handoff(result, expected_stage):
    require(isinstance(result, dict) and result.get("stage") == expected_stage
            and result.get("validated") is True and isinstance(result.get("case"), dict),
            f"invalid {expected_stage} handoff")
    require(result["case"].get("schema_version") == "1.0", "invalid handoff schema")
    require(result["case_id"] == result["case"]["case_id"], "handoff case_id mismatch")
    require(result["status"] == "complete", f"{expected_stage} must complete before proceeding")


def guided(case):
    done = case["completed_steps"]
    steps = [{"id": s, "prerequisites": list(STEPS[:i]), "state": "complete" if s in done else
              ("ready" if i == len(done) else "blocked")} for i, s in enumerate(STEPS)]
    return {"stage": "guided", "validated": True, "case_id": case["case_id"], "case": case,
            "status": "complete" if len(done) == len(STEPS) else "incomplete",
            "progress": {"completed": len(done), "total": len(STEPS)}, "steps": steps}


def review(onboarding):
    validate_handoff(onboarding, "guided")
    case = onboarding["case"]
    checks = []
    for kind, description in REQUIREMENTS.items():
        entity = (case["policy"]["policy_id"] if kind == "policy_schedule" else
                  case["underwriting"]["submission_id"] if kind == "underwriting_disclosure"
                  else case["case_id"])
        matches = [e for e in case["evidence"] if e["kind"] == kind and e["entity_id"] == entity]
        if kind == "underwriting_disclosure":
            matches = [e for e in matches if all(f["name"] in e["text"]
                       for f in case["underwriting"]["factors"])]
        # The parsed, validated claim form itself is an evidence source.
        sources = ["input.claim_form"] if kind == "claim_form" else []
        sources += ["evidence." + e["evidence_id"] for e in matches]
        checks.append({"requirement": kind, "description": description, "entity_id": entity,
                       "status": "present" if sources else "gap", "sources": sources,
                       "reason": "Evidence reference present; authenticity not verified." if sources
                       else "No matching evidence for this requirement and entity."})
    gaps = [c for c in checks if c["status"] == "gap"]
    reasons = [f"Missing {g['requirement']} for {g['entity_id']}." for g in gaps]
    p, c = case["policy"], case["claim"]
    if not p["effective_date"] <= c["loss_date"] <= p["expiry_date"]:
        reasons.append("Reported loss date is outside the policy period; human coverage review required.")
    if c["loss_amount"] > p["coverage_limit"]:
        reasons.append("Reported loss exceeds stated limit; human review of coverage and terms required.")
    if not reasons:
        reasons = ["Required evidence references are present; a human must assess authenticity, terms and merits."]
    decision = {"action": "request_information" if gaps else "human_review",
                "reasons": reasons, "final_claim_decision": False,
                "fair_handling": "Same evidence checks for every case; no protected attributes used.",
                "next_step": "Provide missing evidence or request human review to dispute a gap." if gaps
                else "Request human claims assessment; this is not an approval or denial."}
    return {"stage": "review", "validated": True, "case_id": case["case_id"], "case": case,
            "status": "complete", "checks": checks, "gaps": gaps, "decision": decision,
            "disclaimer": DISCLAIMER}


def support(reviewed):
    validate_handoff(reviewed, "review")
    case = reviewed["case"]
    request = case["support_request"]
    question = request["question"].lower()
    if any(word in question for word in ("claim", "status", "denied", "reason", "document", "gap")):
        answer = ("Claim review action: " + reviewed["decision"]["action"] + ". "
                  + " ".join(reviewed["decision"]["reasons"]) + " " + reviewed["decision"]["next_step"])
        citations = ["review.decision"] + ["review.checks." + g["requirement"] for g in reviewed["gaps"]]
        topic = "claim_review"
    elif any(word in question for word in ("underwriting", "factor", "premium")):
        answer = "Disclosed underwriting factors: " + ", ".join(
            f["name"] + "=" + str(f["value"]) for f in case["underwriting"]["factors"])
        answer += ". No pricing formula is supplied, so a premium cannot be calculated."
        citations, topic = ["case.underwriting.factors"], "underwriting"
    elif any(word in question for word in ("coverage", "policy", "limit")):
        p = case["policy"]
        answer = (f"Stated policy limit: {p['coverage_limit']} {p['currency']}; policy period: "
                  f"{p['effective_date']} through {p['expiry_date']}. This does not establish "
                  "coverage for a particular loss; exclusions and human assessment may apply.")
        citations, topic = ["case.policy"], "policy"
    elif any(word in question for word in ("privacy", "data", "gdpr")):
        answer = ("Use synthetic policyholder references only. Do not submit names, contact details, "
                  "birth dates or government identifiers. This demo keeps no customer database, "
                  "but input files and stdout remain under your control; GDPR compliance is not certified.")
        citations, topic = ["validation.minimization"], "privacy"
    else:
        answer = ("The supplied case evidence does not answer that question. I can explain the policy "
                  "limit, claim review gaps, disclosed underwriting factors, or data minimization. "
                  "Contact a human for other requests; no answer or action has been invented.")
        citations, topic = [], "unknown"
    availability = ("Team marked online; contact a human using your existing support channel."
                    if request["team_online"] else
                    "Team marked offline; this grounded answer is available now. Contact the team "
                    "when available; no ticket has been created and no response time is promised.")
    return {"stage": "support", "validated": True, "case_id": case["case_id"],
            "status": "complete", "topic": topic, "answer": answer, "citations": citations,
            "availability": availability, "escalation_recommended": topic == "unknown" or
            bool(reviewed["gaps"]), "review_action": reviewed["decision"]["action"],
            "disclaimer": DISCLAIMER}


def run_pipeline(raw):
    case = validate_case(raw)
    setup = guided(case)
    if setup["status"] != "complete":
        return {"schema_version": "1.0", "synthetic": True, "status": "blocked",
                "case_id": case["case_id"], "guided": setup, "review": None, "support": None,
                "disclaimer": DISCLAIMER}
    checked = review(setup)
    answered = support(checked)
    # One normalized case is exposed, rather than three redundant copies.
    setup = {k: v for k, v in setup.items() if k != "case"}
    checked = {k: v for k, v in checked.items() if k != "case"}
    return {"schema_version": "1.0", "synthetic": True, "status": "ok", "case_id": case["case_id"],
            "case": case, "guided": setup, "review": checked, "support": answered,
            "disclaimer": DISCLAIMER}


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=no_duplicates)
        result = run_pipeline(raw)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        # Do not echo payloads, personal data, or local file paths in error output.
        message = str(exc) if isinstance(exc, ValidationError) else "Input file or data could not be processed"
        print(json.dumps({"status": "error", "error": message}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
