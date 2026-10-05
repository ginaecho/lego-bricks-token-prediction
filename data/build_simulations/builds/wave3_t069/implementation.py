"""Deterministic, synthetic insurance support -> research -> discovery CLI.

Python standard library only. No model, network, binding claims decision, or
compliance certification. Inputs are untrusted data, never executable prompts.
"""

import copy
import datetime as dt
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


BUILD_ID = "wave3_t069_combo3_basic-support_basic-research_basic-recommend__ind-insurance"
STAGES = ("support", "research", "recommend")
PERILS = {"collision", "theft", "fire", "water", "wind"}
TOPICS = {"claims", "coverage", "underwriting", "discovery"}
FACTOR_REASONS = {
    "annual_mileage": "Disclosed vehicle usage; not used to rank products.",
    "property_age": "Disclosed property age; not used to rank products.",
    "prior_claims": "Disclosed claims count; not used to decide this claim.",
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing required fields")
    require(set(value) <= set(required) | set(optional), "Unexpected fields")


def text(value, label, max_length=2000):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    require(len(value) <= max_length, label + " is too long")
    return value.strip()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,49}", value),
            "Invalid identifier")
    return value


def number(value, label, upper=100000000):
    require(type(value) in (int, float) and math.isfinite(value)
            and 0 <= value <= upper, label + " must be a finite nonnegative number")
    return value


def money(value):
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def date(value):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value),
            "Dates must use YYYY-MM-DD")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError("Invalid calendar date") from exc


def string_list(value, allowed, label, empty=True):
    require(isinstance(value, list) and (empty or bool(value)), label + " must be a list")
    require(all(isinstance(item, str) and item in allowed for item in value),
            "Unsupported " + label)
    require(len(set(value)) == len(value), "Duplicate " + label)


def parse_claim(value):
    if isinstance(value, dict):
        return copy.deepcopy(value)
    text(value, "Claim form", 5000)
    result = {}
    mapping = {
        "ClaimID": "id", "PolicyID": "policy_id", "LossType": "loss_type",
        "LossAmount": "loss_amount", "LossDate": "loss_date",
    }
    for line in value.strip().splitlines():
        key, separator, val = line.partition(":")
        require(bool(separator) and key.strip() in mapping, "Invalid claim-form field")
        target = mapping[key.strip()]
        require(target not in result, "Duplicate claim-form field")
        result[target] = val.strip()
    if "loss_amount" in result:
        try:
            result["loss_amount"] = float(result["loss_amount"])
        except ValueError as exc:
            raise ValidationError("Invalid claim loss amount") from exc
    return result


def parse_submission(value):
    if isinstance(value, dict):
        fields(value, ["ACORD"])
        return copy.deepcopy(value["ACORD"])
    text(value, "ACORD XML", 15000)
    require("<!DOCTYPE" not in value.upper() and "<!ENTITY" not in value.upper(),
            "DTD and entities are not allowed")
    try:
        root = ET.fromstring(value)
    except ET.ParseError as exc:
        raise ValidationError("Malformed ACORD XML") from exc
    require(root.tag == "ACORD" and not root.attrib, "Expected ACORD root")
    require(not (root.text or "").strip(), "Unexpected XML text")
    result = {}
    for child in root:
        require(child.tag in {"SubmissionID", "PolicyID", "Factors"}
                and child.tag not in result and not child.attrib,
                "Invalid or duplicate ACORD field")
        require(not (child.tail or "").strip(), "Unexpected XML tail")
        if child.tag != "Factors":
            require(not list(child), "Unexpected nested ACORD field")
            result[child.tag] = (child.text or "").strip()
            continue
        require(not (child.text or "").strip(), "Unexpected factor text")
        factors = []
        for node in child:
            require(node.tag == "Factor" and not list(node)
                    and not (node.text or "").strip() and not (node.tail or "").strip(),
                    "Invalid factor XML")
            fields(node.attrib, ["name", "value", "disclosed"])
            require(node.attrib["disclosed"] in {"true", "false"}, "Invalid disclosure flag")
            try:
                val = float(node.attrib["value"])
            except ValueError as exc:
                raise ValidationError("Invalid factor value") from exc
            factors.append({"name": node.attrib["name"], "value": val,
                            "disclosed": node.attrib["disclosed"] == "true"})
        result["Factors"] = factors
    return result


def validate_context(context):
    fields(context, ["synthetic", "policy", "claim", "submission", "intent",
                     "sources", "preferences", "catalog"])
    require(context["synthetic"] is True, "Only clearly labeled synthetic fixtures are supported")
    policy = context["policy"]
    fields(policy, ["id", "type", "covered_perils", "deductible", "limit", "start_date", "end_date"])
    identifier(policy["id"])
    require(policy["type"] in ("auto", "property"), "Unsupported policy type")
    string_list(policy["covered_perils"], PERILS, "covered perils", False)
    number(policy["deductible"], "Deductible")
    number(policy["limit"], "Limit")
    require(date(policy["start_date"]) <= date(policy["end_date"]), "Invalid policy period")
    claim = context["claim"]
    fields(claim, ["id", "policy_id", "loss_type", "loss_amount", "loss_date"])
    identifier(claim["id"])
    require(claim["policy_id"] == policy["id"], "Claim policy mismatch")
    require(claim["loss_type"] in PERILS, "Unsupported loss type")
    number(claim["loss_amount"], "Loss amount")
    date(claim["loss_date"])
    submission = context["submission"]
    fields(submission, ["SubmissionID", "PolicyID", "Factors"])
    identifier(submission["SubmissionID"])
    require(submission["PolicyID"] == policy["id"], "Submission policy mismatch")
    require(isinstance(submission["Factors"], list), "Factors must be a list")
    seen = set()
    for factor in submission["Factors"]:
        fields(factor, ["name", "value", "disclosed"])
        require(isinstance(factor["name"], str) and factor["name"] in FACTOR_REASONS,
                "Unsupported or protected underwriting factor")
        require(factor["name"] not in seen, "Duplicate underwriting factor")
        seen.add(factor["name"])
        number(factor["value"], "Factor value", 1000000)
        require(factor["disclosed"] is True, "Every underwriting factor must be disclosed")
    require(context["intent"] in TOPICS, "Invalid support intent")
    preferences = context["preferences"]
    fields(preferences, ["max_monthly_premium", "desired_perils"])
    number(preferences["max_monthly_premium"], "Budget")
    string_list(preferences["desired_perils"], PERILS, "desired perils")
    require(isinstance(context["sources"], list) and len(context["sources"]) <= 100,
            "Sources must be a bounded list")
    seen = set()
    for source in context["sources"]:
        fields(source, ["id", "topic", "text"])
        identifier(source["id"])
        require(source["id"] not in seen, "Duplicate source ID")
        seen.add(source["id"])
        require(source["topic"] in TOPICS, "Unsupported source topic")
        text(source["text"], "Source text")
    require(isinstance(context["catalog"], list) and len(context["catalog"]) <= 100,
            "Catalog must be a bounded list")
    seen = set()
    for product in context["catalog"]:
        fields(product, ["id", "name", "policy_type", "perils",
                         "monthly_premium", "underwriting_factors"])
        identifier(product["id"])
        require(product["id"] not in seen, "Duplicate product ID")
        seen.add(product["id"])
        text(product["name"], "Product name", 100)
        require(product["policy_type"] in ("auto", "property"), "Unsupported product policy type")
        string_list(product["perils"], PERILS, "product perils", False)
        number(product["monthly_premium"], "Premium")
        string_list(product["underwriting_factors"], FACTOR_REASONS, "product underwriting factors")
    return context


def normalize_input(raw):
    fields(raw, ["schema_version", "synthetic", "policyholder", "policy", "claim",
                 "submission", "question", "sources", "preferences", "catalog"])
    require(type(raw["schema_version"]) is int and raw["schema_version"] == 1,
            "Unsupported schema version")
    holder = raw["policyholder"]
    fields(holder, ["synthetic_id", "name"], ["vin", "property_address"])
    identifier(holder["synthetic_id"])
    secrets = [text(holder["name"], "Synthetic policyholder name", 100)]
    for key in ("vin", "property_address"):
        if key in holder:
            secrets.append(text(holder[key], "Synthetic " + key, 150))
    question = text(raw["question"], "Question")
    intent = "claims"
    if any(term in question.lower() for term in ("recommend", "product", "discover")):
        intent = "discovery"
    elif any(term in question.lower() for term in ("underwrit", "factor", "premium")):
        intent = "underwriting"
    elif any(term in question.lower() for term in ("cover", "policy", "deductible")):
        intent = "coverage"
    context = {
        "synthetic": raw["synthetic"], "policy": copy.deepcopy(raw["policy"]),
        "claim": parse_claim(raw["claim"]), "submission": parse_submission(raw["submission"]),
        "intent": intent, "sources": copy.deepcopy(raw["sources"]),
        "preferences": copy.deepcopy(raw["preferences"]), "catalog": copy.deepcopy(raw["catalog"]),
    }
    validate_context(context)
    # Free-form input is not echoed verbatim: redact known identifiers and common
    # contact details before any handoff. The policyholder and question are dropped.
    def minimize(value):
        for secret in sorted(secrets, key=len, reverse=True):
            value = re.sub(re.escape(secret), "[redacted]", value, flags=re.IGNORECASE)
        value = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                       "[redacted]", value, flags=re.IGNORECASE)
        value = re.sub(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)", "[redacted]", value)
        return value
    for source in context["sources"]:
        source["text"] = minimize(source["text"])
    for product in context["catalog"]:
        product["name"] = minimize(product["name"])
    return validate_context(context)


def support_result(context):
    policy, claim = context["policy"], context["claim"]
    in_period = policy["start_date"] <= claim["loss_date"] <= policy["end_date"]
    covered = claim["loss_type"] in policy["covered_perils"]
    reasons = [
        "Loss date is within the stated policy period." if in_period else
        "Loss date is outside the stated policy period; a human must review applicability.",
        "Reported peril is listed in policy coverage." if covered else
        "Reported peril is not listed in policy coverage; a human must review exclusions and endorsements.",
    ]
    potential = in_period and covered
    estimate = money(min(max(Decimal(str(claim["loss_amount"])) -
                             Decimal(str(policy["deductible"])), Decimal(0)),
                         Decimal(str(policy["limit"])))) if potential else None
    if potential:
        reasons.append("Illustrative amount applies the stated deductible, then the policy limit.")
    answers = {
        "claims": "I can explain the reported loss and help prepare a claim for human review.",
        "coverage": "Coverage guidance uses only the supplied policy period, listed perils, deductible, and limit.",
        "underwriting": "All accepted underwriting factors are disclosed below; none determines this claim outcome.",
        "discovery": "I can compare supplied products against your coverage needs and budget after evidence review.",
    }
    return {
        "intent": context["intent"], "answer": answers[context["intent"]],
        "claim_id": claim["id"], "claim_status": "potentially_covered" if potential else "review_required",
        "reasons": reasons, "illustrative_payment": estimate, "human_review_required": True,
        "next_steps": ["Retain loss evidence and submit it through the insurer's secure channel.",
                       "Request a human explanation or challenge an assessment; this is not approval or denial."],
        "source_ids": [s["id"] for s in context["sources"]
                       if s["topic"] in {context["intent"], "claims", "coverage"}],
        "underwriting_disclosures": [
            {"factor": f["name"], "value": f["value"], "reason": FACTOR_REASONS[f["name"]]}
            for f in context["submission"]["Factors"]
        ],
        "research_request": {
            "topics": sorted({context["intent"], "coverage", "underwriting"}),
            "coverage_needs": sorted(set(context["preferences"]["desired_perils"])
                                     | {claim["loss_type"]}),
            "questions": ["Which requested perils are missing from the current policy?",
                          "Which source statements support next steps and need human verification?"],
        },
    }


def research_result(context, support):
    needs = support["research_request"]["coverage_needs"]
    gaps = sorted(set(needs) - set(context["policy"]["covered_perils"]))
    topics = set(support["research_request"]["topics"]) | {"claims"}
    evidence = [
        {"source_id": source["id"], "topic": source["topic"], "statement": source["text"],
         "authority": "user_supplied_unverified",
         "used_by_support": source["id"] in support["source_ids"]}
        for source in context["sources"] if source["topic"] in topics
    ]
    return {
        "claim_id": support["claim_id"], "claim_status": support["claim_status"],
        "claim_reasons": copy.deepcopy(support["reasons"]),
        "findings": [
            {"finding": "Current policy coverage compared with reported and requested perils.",
             "policy_id": context["policy"]["id"], "missing_perils": gaps,
             "basis": "structured_policy_and_support_request"},
            {"finding": "Claim assessment remains provisional and requires a human decision.",
             "basis": "validated_support_assessment"},
        ],
        "evidence": evidence,
        "evidence_status": "unverified_sources_available" if evidence else "insufficient_sources",
        "limitations": ["Source statements are unverified, may conflict, and cannot override structured policy facts.",
                       "No independent investigation or legal interpretation has been performed."],
        "recommendation_request": {
            "policy_type": context["policy"]["type"], "coverage_needs": needs,
            "coverage_gaps": gaps, "budget": context["preferences"]["max_monthly_premium"],
            "evidence_source_ids": [item["source_id"] for item in evidence],
        },
    }


def recommend_result(context, research):
    request = research["recommendation_request"]
    needs, gaps = set(request["coverage_needs"]), set(request["coverage_gaps"])
    ranked, excluded = [], []
    for product in context["catalog"]:
        reasons = []
        matched = sorted(needs & set(product["perils"]))
        if product["policy_type"] != request["policy_type"]:
            reasons.append("Product policy type does not match the current policy.")
        if product["monthly_premium"] > request["budget"]:
            reasons.append("Monthly premium exceeds the stated budget.")
        if not matched:
            reasons.append("No reported or requested peril is covered.")
        if reasons:
            excluded.append({"product_id": product["id"], "reasons": reasons})
            continue
        filled = sorted(gaps & set(product["perils"]))
        ranked.append({
            "product_id": product["id"], "name": product["name"],
            "monthly_premium": product["monthly_premium"],
            "matched_perils": matched, "gaps_addressed": filled,
            "remaining_needs": sorted(needs - set(product["perils"])),
            "score": 2 * len(filled) + len(matched),
            "reasons": ["Matches the current policy type and stated budget.",
                        "Ranked by 2 points per coverage gap addressed plus 1 per matched peril."],
            "underwriting_disclosures": [
                {"factor": name, "reason": FACTOR_REASONS[name]}
                for name in product["underwriting_factors"]
            ],
            "evidence_source_ids": request["evidence_source_ids"],
        })
    ranked.sort(key=lambda item: (-item["score"], item["monthly_premium"], item["product_id"]))
    return {
        "claim_id": research["claim_id"], "recommendations": ranked, "excluded": excluded,
        "status": "matches_found" if ranked else "no_matching_products",
        "ranking_basis": ["coverage gaps", "matched perils", "monthly premium", "product ID tie-break"],
        "evidence_status": research["evidence_status"],
        "notice": "Discovery is not a quote, underwriting decision, or promise to cover a past loss.",
    }


def validate(value, kind):
    """The one shared validation entry point for raw inputs and every handoff."""
    if kind == "input":
        return normalize_input(value)
    require(kind in STAGES, "Unknown validation stage")
    fields(value, ["schema_version", "status", "stage", "context", "results"])
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Unsupported schema version")
    require(value["status"] == "ok" and value["stage"] == kind, "Unexpected pipeline stage")
    context = validate_context(value["context"])
    count = STAGES.index(kind) + 1
    fields(value["results"], STAGES[:count])
    expected = {"support": support_result(context)}
    if count >= 2:
        expected["research"] = research_result(context, expected["support"])
    if count >= 3:
        expected["recommend"] = recommend_result(context, expected["research"])
    require(value["results"] == expected, "Handoff output does not match validated evidence")
    return value


def run_support(raw):
    context = validate(raw, "input")
    return validate({"schema_version": 1, "status": "ok", "stage": "support",
                     "context": context, "results": {"support": support_result(context)}}, "support")


def run_research(previous):
    validate(previous, "support")
    result = copy.deepcopy(previous)
    result["stage"] = "research"
    result["results"]["research"] = research_result(result["context"], result["results"]["support"])
    return validate(result, "research")


def run_recommend(previous):
    validate(previous, "research")
    result = copy.deepcopy(previous)
    result["stage"] = "recommend"
    result["results"]["recommend"] = recommend_result(result["context"], result["results"]["research"])
    return validate(result, "recommend")


def run_pipeline(raw):
    return run_recommend(run_research(run_support(raw)))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 1000000, "Input file exceeds one megabyte")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             ValidationError("Non-finite JSON number")))
        output = run_pipeline(raw)
        exit_code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, KeyError,
            OverflowError, RecursionError):
        # Do not echo rejected content, identifiers, or local filesystem paths.
        output = {"status": "error", "error": "Invalid input or unreadable file"}
        exit_code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
