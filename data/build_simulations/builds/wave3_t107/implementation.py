"""Synthetic insurance discovery demo. Standard library only; not adjudication."""

import datetime as dt
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing required fields")
    require(set(value) <= set(required) | set(optional),
            "Unknown fields rejected for data minimization")


def identifier(value):
    require(isinstance(value, str) and
            re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is not None,
            "Invalid pseudonymous identifier")
    return value


def date(value):
    require(isinstance(value, str) and
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
            "Date must be YYYY-MM-DD")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError("Invalid calendar date") from exc


def number(value, low, high):
    require(type(value) in (int, float) and low <= value <= high
            and math.isfinite(value), "Number outside allowed range")
    return value


def decode_claim(value):
    if isinstance(value, dict):
        return value
    require(isinstance(value, str) and len(value) <= 4096,
            "Claim must be JSON or bounded claim-form text")
    names = {"ClaimID": "id", "PolicyID": "policy_id",
             "LossDate": "loss_date", "LossAmount": "loss_amount",
             "Evidence": "evidence_ready"}
    result = {}
    for line in value.splitlines():
        key, separator, item = line.partition(":")
        require(separator and key in names and names[key] not in result,
                "Invalid or duplicate claim-form field")
        result[names[key]] = item.strip()
    if "loss_amount" in result:
        try:
            result["loss_amount"] = float(result["loss_amount"])
        except ValueError as exc:
            raise ValidationError("Invalid loss amount") from exc
    if "evidence_ready" in result:
        require(result["evidence_ready"] in ("yes", "no"), "Evidence must be yes or no")
        result["evidence_ready"] = result["evidence_ready"] == "yes"
    return result


def decode_submission(value):
    if isinstance(value, dict):
        return value
    require(isinstance(value, str) and len(value) <= 16384,
            "Submission must be JSON or bounded ACORD-style XML")
    require("<!" not in value and "<?" not in value,
            "XML declarations, entities and DTDs are not supported")
    try:
        root = ET.fromstring(value)
    except ET.ParseError as exc:
        raise ValidationError("Invalid submission XML") from exc
    require(root.tag == "ACORDSubmission" and not root.attrib,
            "Expected ACORDSubmission root")
    names = {"SubmissionID": "id", "PolicyID": "policy_id", "VIN": "vin"}
    result = {}
    factors = []
    for child in root:
        require(not (child.tail or "").strip(), "Unexpected XML text")
        if child.tag == "Factor":
            require(set(child.attrib) == {"name", "value", "disclosed"}
                    and not list(child) and not (child.text or "").strip(),
                    "Invalid XML factor")
            require(child.attrib["disclosed"] in ("true", "false"),
                    "Invalid disclosure boolean")
            try:
                factor_value = float(child.attrib["value"])
            except ValueError as exc:
                raise ValidationError("Invalid factor value") from exc
            factors.append({"name": child.attrib["name"], "value": factor_value,
                            "disclosed": child.attrib["disclosed"] == "true"})
        else:
            require(child.tag in names and names[child.tag] not in result
                    and not child.attrib and not list(child), "Unknown XML field")
            result[names[child.tag]] = child.text or ""
    require(not (root.text or "").strip(), "Unexpected XML text")
    result["factors"] = factors
    return result


FACTOR_RANGES = {"annual_mileage": (0, 200000), "vehicle_age": (0, 100),
                 "prior_claim_count": (0, 100)}


def validate_input(raw):
    fields(raw, ("schema_version", "synthetic", "as_of", "goal",
                 "policy", "claim", "underwriting_submission"))
    require(type(raw["schema_version"]) is int and raw["schema_version"] == 1,
            "Unsupported schema version")
    require(raw["synthetic"] is True, "Only clearly labeled synthetic data is accepted")
    require(isinstance(raw["goal"], str) and
            raw["goal"] in ("auto", "claim", "underwriting", "policy"), "Invalid goal")
    as_of = date(raw["as_of"])
    policy = raw["policy"]
    fields(policy, ("id", "policyholder_ref", "start_date", "end_date"))
    identifier(policy["id"])
    identifier(policy["policyholder_ref"])
    start, end = date(policy["start_date"]), date(policy["end_date"])
    require(start <= end, "Policy dates are reversed")
    claim = decode_claim(raw["claim"])
    fields(claim, ("id", "policy_id", "loss_date", "loss_amount", "evidence_ready"))
    identifier(claim["id"])
    require(claim["policy_id"] == policy["id"], "Claim policy reference mismatch")
    require(date(claim["loss_date"]) <= as_of, "Loss date cannot be in the future")
    number(claim["loss_amount"], 0.01, 100000000)
    require(type(claim["evidence_ready"]) is bool, "Evidence must be boolean")
    submission = decode_submission(raw["underwriting_submission"])
    fields(submission, ("id", "policy_id", "vin", "factors"))
    identifier(submission["id"])
    require(submission["policy_id"] == policy["id"], "Submission policy reference mismatch")
    require(isinstance(submission["vin"], str) and
            re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", submission["vin"]) is not None,
            "VIN must have 17 VIN-style characters")
    require(isinstance(submission["factors"], list) and
            1 <= len(submission["factors"]) <= 3, "Provide one to three disclosed factors")
    seen = set()
    for factor in submission["factors"]:
        fields(factor, ("name", "value", "disclosed"))
        name = factor["name"]
        require(isinstance(name, str) and name in FACTOR_RANGES and name not in seen,
                "Unknown, sensitive or duplicate underwriting factor")
        require(factor["disclosed"] is True, "Hidden underwriting factors are prohibited")
        number(factor["value"], *FACTOR_RANGES[name])
        seen.add(name)
    return {"schema_version": 1, "synthetic": True, "as_of": raw["as_of"],
            "goal": raw["goal"], "policy": policy, "claim": claim,
            "underwriting_submission": submission}


ACTIONS = {
    "collect_claim_evidence": ("claim", {"evidence_ready": False}, {"evidence_ready": True},
                               "Collect loss evidence before a readiness check."),
    "check_claim_readiness": ("claim", {"evidence_ready": True},
                             {"readiness_checked": True},
                             "Check evidence and policy dates without denying a claim."),
    "prepare_claim_submission": ("claim", {"evidence_ready": True, "readiness_checked": True},
                                 {"claim_prepared": True},
                                 "Prepare a claim for human handling; no submission is sent."),
    "review_disclosed_factors": ("underwriting", {}, {"factors_reviewed": True},
                                 "Review every disclosed underwriting factor."),
    "prepare_underwriting_submission": ("underwriting", {"factors_reviewed": True},
                                        {"underwriting_prepared": True},
                                        "Prepare the disclosed submission for human review."),
    "review_policy": ("policy", {}, {"policy_reviewed": True},
                     "Review policy dates and coverage questions."),
    "review_renewal_options": ("policy", {"policy_reviewed": True},
                              {"renewal_reviewed": True},
                              "Review renewal options; no policy purchase is made."),
}


def eligible(action, state):
    return all(state.get(key, False) == value for key, value in ACTIONS[action][1].items())


def validate_journey(plan, goal, initial):
    require(isinstance(plan, list) and len(plan) == 2, "Journey must contain exactly two actions")
    state, steps, seen = dict(initial), [], set()
    for action in plan:
        require(isinstance(action, str) and action in ACTIONS and action not in seen,
                "Unknown or duplicate journey action")
        category, prerequisites, effects, reason = ACTIONS[action]
        require(category == goal and eligible(action, state),
                "Journey goal or prerequisite violation")
        before = dict(state)
        state.update(effects)
        steps.append({"action": action, "reason": reason, "prerequisites": prerequisites,
                      "state_before": before, "projected_state_after": dict(state)})
        seen.add(action)
    return steps


def recommend(raw, planner=None):
    data = validate_input(raw)
    claim, policy = data["claim"], data["policy"]
    goal = "claim" if data["goal"] == "auto" else data["goal"]
    initial = {"evidence_ready": claim["evidence_ready"], "readiness_checked": False,
               "factors_reviewed": False, "policy_reviewed": False}
    default = {
        "claim": (["check_claim_readiness", "prepare_claim_submission"]
                  if claim["evidence_ready"] else
                  ["collect_claim_evidence", "check_claim_readiness"]),
        "underwriting": ["review_disclosed_factors", "prepare_underwriting_submission"],
        "policy": ["review_policy", "review_renewal_options"],
    }[goal]
    # An injected planner only sees the minimized planning context, not raw entities.
    context = {"goal": goal, "state": dict(initial), "allowed_actions": [
        action for action, spec in ACTIONS.items() if spec[0] == goal]}
    if planner is None:
        plan = default
    else:
        try:
            plan = planner(context)
        except Exception as exc:
            raise ValidationError("Injected planner failed") from exc
    journey = validate_journey(plan, goal, initial)
    in_term = policy["start_date"] <= claim["loss_date"] <= policy["end_date"]
    reasons = ["Evidence is available." if claim["evidence_ready"]
               else "Loss evidence is still needed; assistance should be offered.",
               "Loss date falls within policy dates." if in_term else
               "Loss date is outside policy dates; refer for human review, not automatic denial."]
    return {"schema_version": 1, "status": "ok", "synthetic": True,
            "entity_refs": {"policy": policy["id"], "claim": claim["id"],
                            "underwriting_submission": data["underwriting_submission"]["id"]},
            "selected_goal": goal,
            "selection_reason": ("An existing claim receives priority." if data["goal"] == "auto"
                                 else "The explicitly requested goal receives priority."),
            "recommendations": [{"action": action, "reason": spec[3],
                                  "prerequisites": spec[1]}
                                 for action, spec in ACTIONS.items()
                                 if spec[0] == goal and eligible(action, initial)],
            "journey": journey, "journey_validated": True, "execution": "planning_only",
            "claim_handling": {"decision": ("human_review" if not in_term else
                                           "ready_for_preparation" if claim["evidence_ready"]
                                           else "needs_evidence"),
                               "reasons": reasons,
                               "notice": "Readiness guidance only, not a coverage or payment decision."},
            "underwriting_disclosure": data["underwriting_submission"]["factors"],
            "data_minimization": "Output omits policyholder alias, VIN, loss amount and loss date.",
            "limitations": "Synthetic demonstration; no compliance certification or live transactions."}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        source = Path(argv[0])
        require(source.stat().st_size <= 65536, "Input file exceeds 64 KiB")
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        output = recommend(raw)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError):
        print(json.dumps({"schema_version": 1, "status": "error",
                          "message": "Invalid input or unreadable file; verify the documented schema."}))
        return 2
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
