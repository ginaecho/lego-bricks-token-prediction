"""Synthetic insurance research -> comparison; Python standard library only.

The ACORD-style adapters implement a deliberately small local dialect, not ACORD
certification. Claim reasoning and factor disclosure checks are demonstrations,
not legal advice, underwriting recommendations, or compliance certification.
"""

import copy
import datetime as dt
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

VERSION = "1.0"
METRICS = ("annual_premium", "coverage_limit", "deductible")
FACTORS = {
    "driving_history": {"clean", "prior_incidents"},
    "property_type": {"house", "apartment"},
    "occupancy": {"owner", "tenant"},
    "safety_features": {"present", "absent"},
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing required fields")
    require(set(value) <= set(required) | set(optional), "Unexpected fields")


def identifier(value):
    require(isinstance(value, str) and
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is not None,
            "Invalid identifier")
    return value


def money(value, positive=False):
    require(type(value) in (int, float) and math.isfinite(value),
            "Money must be a finite number")
    require(0 <= value <= 1_000_000_000 and (not positive or value > 0),
            "Money out of range")
    require(abs(value * 100 - round(value * 100)) < 0.00001,
            "Money supports at most two decimal places")
    return round(float(value), 2)


def validate_policy(policy):
    keys(policy, ("policy_id", "premium", "premium_period", "currency",
                  "coverage_limit", "deductible"))
    identifier(policy["policy_id"])
    money(policy["premium"], positive=True)
    money(policy["coverage_limit"], positive=True)
    money(policy["deductible"])
    require(policy["premium_period"] in ("monthly", "annual"), "Invalid premium period")
    require(policy["currency"] in ("EUR", "USD", "GBP"), "Unsupported currency")
    require(policy["deductible"] <= policy["coverage_limit"], "Deductible exceeds coverage")


def expected_claim_reason(claim):
    if not claim["evidence_complete"]:
        return "review", "insufficient_evidence"
    if not claim["policy_active"]:
        return "deny", "policy_inactive"
    if not claim["covered"]:
        return "deny", "excluded_peril"
    return "approve", "covered_loss"


def validate_claim(claim):
    keys(claim, ("claim_id", "policy_id", "loss_amount", "loss_date", "covered",
                 "policy_active", "evidence_complete", "decision", "reasons"))
    identifier(claim["claim_id"])
    identifier(claim["policy_id"])
    money(claim["loss_amount"], positive=True)
    require(isinstance(claim["loss_date"], str), "Invalid loss date")
    try:
        parsed = dt.date.fromisoformat(claim["loss_date"])
    except ValueError:
        raise ValidationError("Invalid loss date") from None
    require(parsed.isoformat() == claim["loss_date"], "Use an ISO loss date")
    for name in ("covered", "policy_active", "evidence_complete"):
        require(type(claim[name]) is bool, "Claim facts must be booleans")
    decision, reason = expected_claim_reason(claim)
    require(claim["decision"] == decision and claim["reasons"] == [reason],
            "Claim decision must match stated evidence and reason; incomplete evidence needs review")


def validate_submission(submission):
    keys(submission, ("submission_id", "policy_id", "factors"))
    identifier(submission["submission_id"])
    identifier(submission["policy_id"])
    factors = submission["factors"]
    require(isinstance(factors, list) and 0 < len(factors) <= len(FACTORS),
            "Disclosed underwriting factors required")
    seen = set()
    for factor in factors:
        keys(factor, ("name", "value", "disclosed"))
        name = factor["name"]
        require(isinstance(name, str) and name in FACTORS,
                "Unsupported or sensitive underwriting factor")
        require(name not in seen, "Duplicate underwriting factor")
        seen.add(name)
        require(isinstance(factor["value"], str) and factor["value"] in FACTORS[name],
                "Unsupported factor value")
        require(factor["disclosed"] is True, "Hidden underwriting factors prohibited")


def validate_preferences(preferences):
    keys(preferences, ("weights",), ("max_annual_premium",))
    weights = preferences["weights"]
    require(isinstance(weights, dict) and bool(weights) and set(weights) <= set(METRICS),
            "Use supported comparison metrics")
    for weight in weights.values():
        require(type(weight) in (int, float) and math.isfinite(weight) and 0 <= weight <= 1000,
                "Weights must be finite nonnegative numbers")
    require(sum(weights.values()) > 0, "At least one positive weight required")
    if "max_annual_premium" in preferences:
        money(preferences["max_annual_premium"], positive=True)


def validate_document(document, phase):
    """One validation entry point for raw input and every cross-stage envelope."""
    if phase == "input":
        keys(document, ("schema_version", "synthetic", "question", "sources", "preferences"))
        require(document["schema_version"] == VERSION, "Unsupported schema version")
        require(document["synthetic"] is True, "Only explicitly synthetic inputs are supported")
        question = document["question"]
        require(isinstance(question, str) and 0 < len(question.strip()) <= 1000,
                "A bounded research question is required")
        validate_preferences(document["preferences"])
        require(isinstance(document["sources"], list) and 0 < len(document["sources"]) <= 100,
                "Provide 1 to 100 sources")
        seen = set()
        for source in document["sources"]:
            keys(source, ("source_id", "format", "content"))
            sid = identifier(source["source_id"])
            require(sid not in seen, "Duplicate source ID")
            seen.add(sid)
            require(source["format"] in ("acord_json", "acord_xml", "claim_form"),
                    "Unsupported source format")
        return
    keys(document, ("schema_version", "synthetic", "status", "research", "comparison", "preferences"))
    require(document["schema_version"] == VERSION and document["synthetic"] is True,
            "Invalid shared envelope")
    require(document["status"] == "ok", "Invalid envelope status")
    validate_preferences(document["preferences"])
    research = document["research"]
    keys(research, ("records", "evidence", "answer", "limitations"))
    records = research["records"]
    require(isinstance(records, list) and bool(records), "Research requires policy records")
    ids, claim_ids, submission_ids, currencies = set(), set(), set(), set()
    for record in records:
        keys(record, ("policy", "claims", "underwriting_submission", "source_ids"))
        policy, submission = record["policy"], record["underwriting_submission"]
        validate_policy(policy)
        validate_submission(submission)
        pid = policy["policy_id"]
        require(pid not in ids, "Duplicate policy ID")
        ids.add(pid)
        require(submission["submission_id"] not in submission_ids, "Duplicate submission ID")
        submission_ids.add(submission["submission_id"])
        currencies.add(policy["currency"])
        require(submission["policy_id"] == pid, "Submission references a different policy")
        require(isinstance(record["claims"], list) and bool(record["claims"]),
                "Each policy requires claim evidence")
        for claim in record["claims"]:
            validate_claim(claim)
            require(claim["policy_id"] == pid, "Claim references a different policy")
            require(claim["claim_id"] not in claim_ids, "Duplicate claim ID")
            claim_ids.add(claim["claim_id"])
        require(isinstance(record["source_ids"], list) and bool(record["source_ids"]),
                "Source provenance required")
        for sid in record["source_ids"]:
            identifier(sid)
    require(len(currencies) == 1, "Mixed currencies need an explicit FX model; not supported")
    evidence = research["evidence"]
    require(evidence == make_evidence(records), "Evidence must match validated source records")
    keys(research["answer"], ("evidence_ids", "finding"))
    chosen = research["answer"]["evidence_ids"]
    require(isinstance(chosen, list) and chosen and len(chosen) == len(set(chosen)),
            "Unique answer citations required")
    allowed = {item["evidence_id"] for item in evidence}
    require(all(isinstance(item, str) and item in allowed for item in chosen),
            "Answer references unknown evidence")
    require(research["answer"]["finding"] == finding(evidence, chosen),
            "Research finding must be grounded in cited evidence")
    require(research["limitations"] == LIMITATIONS, "Required limitations missing")
    if phase == "research":
        require(document["comparison"] is None, "Research cannot contain premature comparison")
    else:
        require(phase == "complete", "Invalid validation phase")
        require(document["comparison"] == comparison_payload(records, document["preferences"]),
                "Comparison must match the validated research handoff")


LIMITATIONS = [
    "SYNTHETIC FIXTURE DATA: not real policies, people, or losses.",
    "Evidence is supplied, not independently verified; no external research is performed.",
    "Coverage is simplified; claim facts are assumed and human review remains necessary.",
    "Demonstrative fairness, minimization and disclosure checks; no compliance certification.",
]


def xml_bundle(text):
    require(isinstance(text, str) and len(text) <= 100_000, "Invalid XML content")
    require("<!DOCTYPE" not in text.upper() and "<!ENTITY" not in text.upper(),
            "XML declarations and entities are not supported")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise ValidationError("Malformed ACORD-style XML") from None
    require(root.tag == "ACORD" and not root.attrib, "Expected ACORD root")
    require(all(child.tag in ("Policy", "Claim", "UnderwritingSubmission", "Policyholder")
                for child in root), "Unexpected ACORD element")
    for tag in ("Policy", "Claim", "UnderwritingSubmission"):
        require(len(root.findall(tag)) == 1, "Exactly one of each ACORD entity required")
    p, c, u = root.find("Policy"), root.find("Claim"), root.find("UnderwritingSubmission")
    require(not list(p), "Unexpected policy child elements")
    require(all(child.tag == "Reason" and not child.attrib and not list(child) for child in c),
            "Unexpected claim child elements")
    require(all(child.tag == "Factor" and not list(child) for child in u),
            "Unexpected underwriting child elements")
    policy, claim, submission = dict(p.attrib), dict(c.attrib), dict(u.attrib)
    try:
        for name in ("premium", "coverage_limit", "deductible"):
            policy[name] = float(policy[name])
        claim["loss_amount"] = float(claim["loss_amount"])
    except (KeyError, ValueError):
        raise ValidationError("Invalid XML monetary fields") from None
    for name in ("covered", "policy_active", "evidence_complete"):
        require(claim.get(name) in ("true", "false"), "Invalid XML boolean")
        claim[name] = claim[name] == "true"
    claim["reasons"] = [node.text for node in c]
    submission["factors"] = []
    for node in u:
        factor = dict(node.attrib)
        require(factor.get("disclosed") in ("true", "false"), "Invalid XML disclosure")
        factor["disclosed"] = factor["disclosed"] == "true"
        submission["factors"].append(factor)
    return {"policy": policy, "claim": claim, "underwriting_submission": submission}


def claim_form(text):
    require(isinstance(text, str) and len(text) <= 100_000, "Invalid claim form")
    fields = {}
    names = {
        "Claim ID": "claim_id", "Policy ID": "policy_id", "Loss Amount": "loss_amount",
        "Loss Date": "loss_date", "Covered": "covered", "Policy Active": "policy_active",
        "Evidence Complete": "evidence_complete", "Decision": "decision", "Reasons": "reasons",
    }
    ignored = {"Policyholder", "VIN", "Property Address"}
    for line in text.splitlines():
        if not line.strip():
            continue
        label, sep, value = line.partition(":")
        require(sep and label.strip() in set(names) | ignored, "Unknown claim-form field")
        label, value = label.strip(), value.strip()
        if label in ignored:
            continue
        require(names[label] not in fields, "Duplicate claim-form field")
        fields[names[label]] = value
    try:
        fields["loss_amount"] = float(fields["loss_amount"])
    except (KeyError, ValueError):
        raise ValidationError("Invalid claim-form loss amount") from None
    for name in ("covered", "policy_active", "evidence_complete"):
        require(fields.get(name) in ("true", "false"), "Invalid claim-form boolean")
        fields[name] = fields[name] == "true"
    fields["reasons"] = [part.strip() for part in fields.get("reasons", "").split(",")]
    validate_claim(fields)
    return fields


def make_evidence(records):
    evidence = []
    for record in records:
        p = record["policy"]
        attrs = {
            "annual_premium": round(p["premium"] * (12 if p["premium_period"] == "monthly" else 1), 2),
            "coverage_limit": p["coverage_limit"], "deductible": p["deductible"],
        }
        for attribute, value in attrs.items():
            evidence.append({
                "evidence_id": p["policy_id"] + "_" + attribute,
                "entity_id": p["policy_id"], "attribute": attribute,
                "value": value, "unit": p["currency"], "source_ids": record["source_ids"],
            })
    return evidence


def finding(evidence, selected):
    return "; ".join(
        f'{item["entity_id"]}: {item["attribute"]}={item["value"]} {item["unit"]}'
        for item in evidence if item["evidence_id"] in selected
    )


def research_answer(question, evidence, selector=None):
    """General deterministic evidence selection over attributed facts.

    An optional offline callable receives question + copied evidence and may
    return only evidence IDs, never ungrounded generated claims.
    """
    if selector is not None:
        try:
            selected = selector(question, copy.deepcopy(evidence))
        except Exception:
            raise ValidationError("Injected evidence selector failed") from None
    else:
        words = set(re.findall(r"[a-z]+", question.lower()))
        selected = [item["evidence_id"] for item in evidence
                    if words.intersection(item["attribute"].split("_"))]
        if not selected:
            selected = [item["evidence_id"] for item in evidence]
    allowed = {item["evidence_id"] for item in evidence}
    require(isinstance(selected, list) and bool(selected) and
            all(isinstance(item, str) and item in allowed for item in selected),
            "Selector must return known evidence IDs")
    require(len(selected) == len(set(selected)), "Duplicate selected evidence")
    return {"evidence_ids": selected, "finding": finding(evidence, selected)}


def research_stage(document, selector=None):
    validate_document(document, "input")
    records, loose_claims = [], []
    for source in document["sources"]:
        if source["format"] == "claim_form":
            loose_claims.append((claim_form(source["content"]), source["source_id"]))
            continue
        if source["format"] == "acord_xml":
            bundle = xml_bundle(source["content"])
        else:
            keys(source["content"], ("ACORD",))
            bundle = source["content"]["ACORD"]
        keys(bundle, ("policy", "underwriting_submission"), ("claim", "policyholder"))
        # Names, addresses, fabricated VINs and all policyholder metadata are discarded.
        records.append({
            "policy": copy.deepcopy(bundle["policy"]),
            "claims": [copy.deepcopy(bundle["claim"])] if "claim" in bundle else [],
            "underwriting_submission": copy.deepcopy(bundle["underwriting_submission"]),
            "source_ids": [source["source_id"]],
        })
    for record in records:
        validate_policy(record["policy"])
    for claim, sid in loose_claims:
        matching = [record for record in records
                    if record["policy"]["policy_id"] == claim["policy_id"]]
        require(len(matching) == 1, "Claim-form policy reference is missing or ambiguous")
        matching[0]["claims"].append(claim)
        matching[0]["source_ids"].append(sid)
    result = {
        "schema_version": VERSION, "synthetic": True, "status": "ok",
        "preferences": copy.deepcopy(document["preferences"]),
        "research": {"records": records, "evidence": make_evidence(records),
                     "answer": {}, "limitations": list(LIMITATIONS)},
        "comparison": None,
    }
    evidence = result["research"]["evidence"]
    result["research"]["answer"] = research_answer(document["question"], evidence, selector)
    validate_document(result, "research")
    return result


def comparison_payload(records, preferences):
    evidence = make_evidence(records)
    rows = []
    for record in records:
        policy = record["policy"]
        facts = [item for item in evidence if item["entity_id"] == policy["policy_id"]]
        metrics = {item["attribute"]: item["value"] for item in facts}
        claims = []
        for claim in record["claims"]:
            payout = max(0, min(claim["loss_amount"], policy["coverage_limit"]) - policy["deductible"])
            claims.append({
                "claim_id": claim["claim_id"], "decision": claim["decision"],
                "reasons": claim["reasons"],
                "illustrative_payment": round(payout, 2) if claim["decision"] == "approve" else None,
            })
        rows.append({
            "policy_id": policy["policy_id"], "currency": policy["currency"],
            "attributes": metrics, "evidence_ids": [item["evidence_id"] for item in facts],
            "source_ids": record["source_ids"],
            "disclosed_underwriting_factors": record["underwriting_submission"]["factors"],
            "claim_decisions": claims,
            "eligible": metrics["annual_premium"] <= preferences.get("max_annual_premium", math.inf),
        })
    eligible = [row for row in rows if row["eligible"]]
    weights = preferences["weights"]
    for row in rows:
        reasons, score = [], 0.0
        for metric, weight in sorted(weights.items()):
            values = [candidate["attributes"][metric] for candidate in eligible]
            utility = 0.0
            if row["eligible"]:
                low, high = min(values), max(values)
                utility = 1.0 if low == high else (row["attributes"][metric] - low) / (high - low)
                if metric != "coverage_limit" and low != high:
                    utility = 1 - utility
            contribution = utility * weight / sum(weights.values())
            score += contribution
            reasons.append({"metric": metric, "weight": weight,
                            "utility": round(utility, 6), "contribution": round(contribution, 6)})
        row.update(score=round(score, 6) if row["eligible"] else None,
                   score_reasons=reasons, rank=None,
                   exclusion_reasons=[] if row["eligible"] else ["annual_premium_exceeds_budget"])
    ranked = sorted(eligible, key=lambda row: (-row["score"], row["policy_id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return {
        "side_by_side": sorted(rows, key=lambda row: row["policy_id"]),
        "ranking": [row["policy_id"] for row in ranked],
        "recommended_policy_id": ranked[0]["policy_id"] if ranked else None,
        "method": "Eligible-set min-max weighted utility; ties by policy ID; no currency conversion.",
        "decision_boundary": "Preference ranking only; not a claim, eligibility, or underwriting decision.",
    }


def compare_stage(researched):
    validate_document(researched, "research")
    result = copy.deepcopy(researched)
    result["comparison"] = comparison_payload(
        result["research"]["records"], result["preferences"])
    validate_document(result, "complete")
    return result


def run_pipeline(document, selector=None):
    return compare_stage(research_stage(document, selector))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            text = stream.read(1_000_001)
        require(len(text) <= 1_000_000, "Input file too large")
        document = json.loads(text, object_pairs_hook=unique_object)
        result = run_pipeline(document)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, KeyError,
            OverflowError, RecursionError):
        # Never echo source content, policyholder values or filesystem paths.
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
