"""Synthetic insurance intake and accountable triage; no coverage or eligibility decisions.

Run: python -B implementation.py example_input.json
ACORD-style formats are intentionally a small imitation, not an ACORD implementation.
Only canonical, minimized data crosses the normalization -> triage boundary.
"""

import datetime as dt
import json
import math
import re
import sys
import xml.etree.ElementTree as ET


ENTITIES = ("policy", "claim", "underwriting_submission")
FACTORS = {"roof_age_years", "vehicle_age_years", "coverage_requested"}
FIELDS = {
    "entity", "reference", "summary", "urgency", "loss_amount", "loss_date",
    "policyholder", "vin", "property_address", "underwriting_factors",
}
DEFAULTS = {
    "high_loss_amount": 25000,
    "categories": {
        "policy": "policy_service",
        "claim": "claims_handling",
        "underwriting_submission": "underwriting_review",
    },
    "routes": {
        "policy": {"team": "policy_operations", "owner": "policy_duty_manager"},
        "claim": {"team": "claims_operations", "owner": "claims_duty_adjuster"},
        "underwriting_submission": {
            "team": "underwriting_operations", "owner": "underwriting_duty_manager",
        },
    },
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def mapping(value, allowed, required, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) <= allowed, label + " contains unsupported fields")
    require(required <= set(value), label + " is missing required fields")


def number(value, label, positive=False):
    require(
        type(value) in (int, float)
        and (value > 0 if positive else value >= 0)
        and value <= 1_000_000_000
        and math.isfinite(value),
        label + " must be a finite bounded nonnegative number",
    )
    return value


def label(value, field):
    require(
        isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{1,63}", value),
        field + " must be a non-personal operational label",
    )
    return value


def configuration(raw):
    mapping(raw, set(DEFAULTS), set(), "config")
    threshold = number(raw.get("high_loss_amount", 25000), "high_loss_amount", True)
    categories = dict(DEFAULTS["categories"])
    routes = {key: dict(value) for key, value in DEFAULTS["routes"].items()}
    for name, target in (("categories", categories), ("routes", routes)):
        overrides = raw.get(name, {})
        mapping(overrides, set(ENTITIES), set(), name)
        for entity, value in overrides.items():
            if name == "categories":
                target[entity] = label(value, "category")
            else:
                mapping(value, {"team", "owner"}, {"team", "owner"}, "route")
                target[entity] = {key: label(value[key], key) for key in ("team", "owner")}
    return {"high_loss_amount": threshold, "categories": categories, "routes": routes}


def decode_payload(fmt, payload):
    if fmt == "acord_json":
        mapping(payload, {"ACORD"}, {"ACORD"}, "payload")
        mapping(payload["ACORD"], {"Submission"}, {"Submission"}, "ACORD")
        return payload["ACORD"]["Submission"]
    require(isinstance(payload, str) and len(payload) <= 100000, "payload must be bounded text")
    if fmt == "claim_form":
        result = {}
        for line in payload.splitlines():
            if not line.strip():
                continue
            key, separator, value = line.partition(":")
            key = key.strip()
            require(separator and key not in result, "claim form has malformed or duplicate fields")
            result[key] = value.strip()
        require(result.get("entity") == "claim", "claim form only accepts claims")
    else:
        require("<!DOCTYPE" not in payload.upper() and "<!ENTITY" not in payload.upper(),
                "XML declarations for DTDs/entities are prohibited")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            raise ValidationError("invalid ACORD-style XML") from None
        require(root.tag == "ACORD" and not root.attrib and len(root) == 1,
                "XML must contain one ACORD Submission")
        submission = root[0]
        require(submission.tag == "Submission" and not submission.attrib,
                "XML requires Submission")
        result = {}
        for child in submission:
            require(child.tag not in result and not child.attrib, "duplicate or attributed XML field")
            if child.tag == "underwriting_factors":
                factors = []
                for factor in child:
                    require(factor.tag == "Factor" and not factor.attrib, "invalid XML Factor")
                    parts = {}
                    for item in factor:
                        require(not item.attrib and len(item) == 0 and item.tag not in parts,
                                "invalid XML factor field")
                        parts[item.tag] = item.text or ""
                    if "disclosed" in parts:
                        require(parts["disclosed"] in ("true", "false"), "invalid disclosure boolean")
                        parts["disclosed"] = parts["disclosed"] == "true"
                    if "value" in parts:
                        parts["value"] = text_number(parts["value"])
                    factors.append(parts)
                result[child.tag] = factors
            else:
                require(len(child) == 0, "XML scalar fields cannot contain nested elements")
                result[child.tag] = child.text or ""
    if "loss_amount" in result:
        result["loss_amount"] = text_number(result["loss_amount"])
    return result


def text_number(value):
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValidationError("invalid numeric text") from None


def validate(document):
    """Shared validation and minimization layer used by every input adapter."""
    mapping(document, {"schema_version", "synthetic", "format", "payload", "config"},
            {"schema_version", "synthetic", "format", "payload"}, "input")
    require(type(document["schema_version"]) is int and document["schema_version"] == 1,
            "schema_version must be 1")
    require(document["synthetic"] is True, "only clearly labeled synthetic fixtures are accepted")
    fmt = document["format"]
    require(isinstance(fmt, str) and fmt in ("acord_json", "acord_xml", "claim_form"),
            "unsupported format")
    config = configuration(document.get("config", {}))
    raw = decode_payload(fmt, document["payload"])
    mapping(raw, FIELDS, {"entity", "reference", "summary"}, "submission")
    entity = raw["entity"]
    require(isinstance(entity, str) and entity in ENTITIES, "unsupported entity")
    reference = raw["reference"]
    require(isinstance(reference, str) and re.fullmatch(r"SYN-[A-Z]{2,8}-[0-9]{3,12}", reference),
            "reference must be a synthetic opaque identifier")
    require(isinstance(raw["summary"], str) and 0 < len(raw["summary"].strip()) <= 4000,
            "summary must be nonempty bounded text")
    urgency = raw.get("urgency", "normal")
    require(isinstance(urgency, str) and urgency in ("normal", "urgent"), "unsupported urgency")
    for field in ("vin", "property_address", "policyholder"):
        if field in raw:
            require(isinstance(raw[field], str) and len(raw[field]) <= 1000,
                    "personal fixture fields must be bounded strings")
    record = {"entity": entity, "reference": reference, "urgency": urgency}
    if entity == "claim":
        require({"loss_amount", "loss_date"} <= set(raw), "claims require amount and date")
        record["loss_amount"] = number(raw["loss_amount"], "loss_amount")
        date = raw["loss_date"]
        require(isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date),
                "loss_date must be ISO YYYY-MM-DD")
        try:
            dt.date.fromisoformat(date)
        except ValueError:
            raise ValidationError("invalid loss_date") from None
        record["loss_date"] = date
    else:
        require(not ({"loss_amount", "loss_date"} & set(raw)), "loss fields are only valid for claims")
    factors = raw.get("underwriting_factors", [])
    require(isinstance(factors, list) and len(factors) <= len(FACTORS), "invalid underwriting factors")
    require(entity == "underwriting_submission" or not factors, "factors only valid for underwriting")
    if entity == "underwriting_submission":
        require(bool(factors), "underwriting requires explicit disclosed factors")
    disclosed = []
    seen = set()
    for factor in factors:
        mapping(factor, {"name", "value", "disclosed"}, {"name", "value", "disclosed"}, "factor")
        name = factor["name"]
        require(isinstance(name, str) and name in FACTORS, "unsupported or sensitive underwriting factor")
        require(name not in seen, "duplicate underwriting factor")
        require(factor["disclosed"] is True, "all underwriting factors must be disclosed")
        value = number(factor["value"], "factor value")
        if name.endswith("_years"):
            require(type(value) in (int, float) and value <= 200 and int(value) == value,
                    "age factors require whole years between 0 and 200")
        disclosed.append({"name": name, "value": value, "disclosed": True})
        seen.add(name)
    record["underwriting_factors"] = sorted(disclosed, key=lambda item: item["name"])
    return record, config


def process(document):
    record, config = validate(document)
    entity = record["entity"]
    priority = "normal"
    reasons = ["ENTITY_ROUTE: entity selects the configured category and accountable duty role."]
    if record["urgency"] == "urgent":
        priority = "high"
        reasons.append("URGENT_REQUEST: explicit urgency requests expedited human review.")
    if entity == "claim" and record["loss_amount"] >= config["high_loss_amount"]:
        priority = "high"
        reasons.append("LOSS_THRESHOLD: reported loss meets the configured expedited-review threshold.")
    if priority == "normal":
        reasons.append("STANDARD_QUEUE: no expedited-review rule matched.")
    if entity == "claim":
        reasons.append("FAIR_HANDLING: amount and urgency affect queue order only; no claim is denied or approved.")
    if entity == "underwriting_submission":
        reasons.append("DISCLOSED_FACTORS: only listed disclosed factors are passed to the human reviewer.")
    return {
        "status": "ok",
        "schema_version": 1,
        "synthetic": True,
        "ticket": record,
        "triage": {
            "category": config["categories"][entity],
            "priority": priority,
            "route": dict(config["routes"][entity]),
            "reasons": reasons,
            "decision": "human_review_required",
            "coverage_decision": "not_made",
            "applied_rules": {"high_loss_amount": config["high_loss_amount"]},
        },
        "privacy": {
            "retained": "Opaque reference and task-required insurance fields only.",
            "discarded": ["policyholder", "vin", "property_address", "summary"],
        },
        "limitations": "Demonstrative validation only; not compliance certification or production insurance advice.",
    }


def reject_constant(_):
    raise ValidationError("nonfinite JSON constants are prohibited")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], "r", encoding="utf-8") as handle:
            text = handle.read(1_000_001)
        require(len(text) <= 1_000_000, "input exceeds maximum size")
        document = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = process(document)
    except (OSError, UnicodeError):
        output = {"status": "error", "message": "input file cannot be read"}
    except (json.JSONDecodeError, RecursionError):
        output = {"status": "error", "message": "invalid JSON document"}
    except ValidationError as error:
        output = {"status": "error", "message": str(error)}
    except ValueError:
        output = {"status": "error", "message": "invalid JSON value"}
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0 if output["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
