"""Synthetic insurance guided onboarding; standard-library reference, not compliance advice."""
import datetime as dt
import json
import math
import sys
import xml.etree.ElementTree as ET


STEPS = ("verify_policy", "review_underwriting", "review_claim", "finish")
FACTORS = {"property_type": {"house", "apartment"}, "protection": {"alarm", "none"}}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), label="object"):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(required) <= value.keys(), f"{label}: missing required fields")
    require(value.keys() <= set(required) | set(optional), f"{label}: unknown fields; minimize data")


def text(value, label):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 100, f"{label}: invalid text")
    return value


def money(value, label):
    require(type(value) in (int, float) and 0 <= value <= 1000000000000 and math.isfinite(value),
            f"{label}: expected finite amount between zero and one trillion")


def date(value, label):
    require(isinstance(value, str), f"{label}: expected ISO date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        raise ValidationError(f"{label}: invalid ISO date") from None
    require(parsed.isoformat() == value, f"{label}: use YYYY-MM-DD")
    return parsed


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValidationError("Nonfinite JSON number")))


def normalize_submission(wrapper):
    fields(wrapper, ("format", "content"), label="underwriting_submission")
    fmt, content = wrapper["format"], wrapper["content"]
    if fmt == "acord-json":
        fields(content, ("ACORD",), label="ACORD document")
        fields(content["ACORD"], ("Submission",), label="ACORD")
        return content["ACORD"]["Submission"]
    require(fmt == "acord-xml", "Unsupported submission format")
    require(isinstance(content, str) and len(content) <= 20000, "Invalid XML content")
    require("<!" not in content, "XML declarations/entities are not supported")
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        raise ValidationError("Malformed submission XML") from None
    for element in root.iter():
        require(not (element.tail or "").strip(), "Unexpected XML text")
        if len(element):
            require(not (element.text or "").strip(), "Unexpected XML container text")
    require(root.tag == "ACORD" and not root.attrib and len(root) == 1, "Invalid ACORD root")
    node = root[0]
    require(node.tag == "Submission" and not node.attrib, "Invalid Submission element")
    result = {}
    for child in node:
        require(not child.attrib and child.tag not in result, "Invalid or duplicate XML field")
        if child.tag == "Factors":
            require("factors" not in result and not (child.text or "").strip(), "Invalid or duplicate Factors")
            factors = []
            for factor in child:
                require(factor.tag == "Factor" and set(factor.attrib) == {"name", "value", "disclosed"}
                        and not len(factor) and not (factor.text or "").strip(), "Invalid XML factor")
                require(factor.attrib["disclosed"] in {"true", "false"}, "Invalid disclosure boolean")
                factors.append({"name": factor.attrib["name"], "value": factor.attrib["value"],
                                "disclosed": factor.attrib["disclosed"] == "true"})
            result["factors"] = factors
        else:
            require(child.tag in {"submission_id", "policy_id"} and not len(child), "Unknown XML field")
            result[child.tag] = child.text or ""
    return result


def normalize_claim(wrapper):
    fields(wrapper, ("format", "content"), label="claim")
    require(wrapper["format"] == "claim-form-text", "Unsupported claim format")
    raw = wrapper["content"]
    require(isinstance(raw, str) and len(raw) <= 10000, "Invalid claim form")
    result = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        require(bool(sep) and key.strip() not in result, "Malformed or duplicate claim form field")
        result[key.strip()] = value.strip()
    fields(result, ("claim_id", "policy_id", "loss_date", "loss_amount", "loss_type"), label="claim form")
    try:
        result["loss_amount"] = float(result["loss_amount"])
    except ValueError:
        raise ValidationError("Invalid loss amount") from None
    return result


def validate(data):
    """One boundary validator shared by the CLI and callable onboarding API."""
    fields(data, ("schema_version", "synthetic", "policy", "claim", "underwriting_submission", "onboarding"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "Unsupported schema version")
    require(data["synthetic"] is True, "Only clearly labeled synthetic fixtures are supported")
    p = data["policy"]
    if p is not None:
        fields(p, ("policy_id", "policyholder", "start_date", "end_date", "limit", "deductible", "covered_losses"),
               ("property_address",),
               label="policy")
        if "property_address" in p:
            text(p["property_address"], "synthetic property address")
            require(p["property_address"].startswith("SYNTHETIC: "), "Property address must be labeled synthetic")
        text(p["policy_id"], "policy ID")
        fields(p["policyholder"], ("synthetic_id",), label="policyholder")
        text(p["policyholder"]["synthetic_id"], "synthetic policyholder ID")
        require(p["policyholder"]["synthetic_id"].startswith("SYN-"), "Use a synthetic policyholder identifier")
        require(date(p["start_date"], "start date") <= date(p["end_date"], "end date"), "Inverted policy dates")
        money(p["limit"], "policy limit")
        money(p["deductible"], "deductible")
        losses = p["covered_losses"]
        require(isinstance(losses, list) and len(losses) > 0, "covered_losses must be nonempty")
        require(all(isinstance(x, str) and x in {"fire", "water", "theft"} for x in losses),
                "Unsupported covered loss")
        require(len(losses) == len(set(losses)), "Duplicate covered loss")
    c = normalize_claim(data["claim"]) if data["claim"] is not None else None
    if c is not None:
        text(c["claim_id"], "claim ID")
        text(c["policy_id"], "claim policy ID")
        date(c["loss_date"], "loss date")
        money(c["loss_amount"], "loss amount")
        require(c["loss_type"] in {"fire", "water", "theft", "other"}, "Unsupported loss type")
        require(p is None or c["policy_id"] == p["policy_id"], "Claim policy mismatch")
    s = normalize_submission(data["underwriting_submission"]) if data["underwriting_submission"] is not None else None
    if s is not None:
        fields(s, ("submission_id", "policy_id", "factors"), label="submission")
        text(s["submission_id"], "submission ID")
        text(s["policy_id"], "submission policy ID")
        require(p is None or s["policy_id"] == p["policy_id"], "Submission policy mismatch")
        require(isinstance(s["factors"], list) and len(s["factors"]) > 0, "Disclosed factors are required")
        names = []
        for factor in s["factors"]:
            fields(factor, ("name", "value", "disclosed"), label="factor")
            name, value = factor["name"], factor["value"]
            require(isinstance(name, str) and name in FACTORS, "Unsupported underwriting factor")
            require(isinstance(value, str) and value in FACTORS[name], "Unsupported factor value")
            require(factor["disclosed"] is True, "Every underwriting factor must be disclosed")
            names.append(name)
        require(len(names) == len(set(names)), "Duplicate underwriting factor")
    o = data["onboarding"]
    fields(o, ("completed_steps", "requested_steps"), label="onboarding")
    for key in o:
        require(isinstance(o[key], list) and all(isinstance(x, str) and x in STEPS for x in o[key]),
                f"{key}: unknown step")
        require(len(o[key]) == len(set(o[key])), f"{key}: duplicate step")
    completed = o["completed_steps"]
    require(completed == list(STEPS[:len(completed)]), "Completed steps must be a prerequisite prefix")
    evidence = (p is not None, s is not None, c is not None, True)
    require(all(evidence[:len(completed)]), "Completed steps lack validated evidence")
    return p, c, s


def claim_decision(policy, claim):
    """No demographic factors or underwriting factors influence claims decisions."""
    if not policy["start_date"] <= claim["loss_date"] <= policy["end_date"]:
        return {"decision": "not_payable", "amount": 0, "reasons": ["Loss date is outside the policy period."]}
    if claim["loss_type"] not in policy["covered_losses"]:
        return {"decision": "not_payable", "amount": 0, "reasons": ["Loss type is not included in the stated coverage."]}
    amount = round(min(max(claim["loss_amount"] - policy["deductible"], 0), policy["limit"]), 2)
    return {"decision": "payable" if amount > 0 else "not_payable", "amount": amount,
            "reasons": ["Covered loss within policy dates; amount = min(max(loss - deductible, 0), limit)."],
            "review_notice": "Synthetic rules only; a disputed decision requires human review."}


def run(data):
    p, c, s = validate(data)
    completed = list(data["onboarding"]["completed_steps"])
    evidence = {"verify_policy": p, "review_underwriting": s, "review_claim": c, "finish": True}
    events = []
    for step in data["onboarding"]["requested_steps"]:
        if step in completed:
            events.append({"step": step, "result": "already_complete"})
        elif step != STEPS[len(completed)]:
            events.append({"step": step, "result": "blocked", "reason": "Complete prerequisite steps first."})
        elif evidence[step] is None:
            events.append({"step": step, "result": "blocked", "reason": "Supply this step's validated entity."})
        else:
            completed.append(step)
            events.append({"step": step, "result": "completed"})
    next_step = STEPS[len(completed)] if len(completed) < len(STEPS) else None
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "policy": p, "claim": c, "underwriting_submission": s,
            "onboarding": {"completed_steps": completed, "requested_steps": [],
                           "total_steps": len(STEPS), "percent_complete": len(completed) * 25,
                           "next_step": next_step, "events": events,
                           "next_step_ready": next_step is not None and evidence[next_step] is not None},
            "claim_decision": claim_decision(p, c) if "review_claim" in completed else None,
            "underwriting_review": {"disclosed_factors": s["factors"], "result": "ready_for_human_review"}
            if "review_underwriting" in completed else None,
            "scope_notice": "Demonstrative validation only; not compliance certification or binding insurance advice."}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            raw = handle.read(1000001)
        require(len(raw) <= 1000000, "Input exceeds size limit")
        result = run(decode_json(raw))
    except (ValidationError, ValueError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
