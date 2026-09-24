"""Bounded synthetic insurance triage; not an adjudication or compliance system."""

import datetime
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


LEXICON = {
    "helpful": 2, "excellent": 3, "thanks": 1, "satisfied": 2, "resolved": 2,
    "clear": 1, "delay": -2, "delayed": -2, "unfair": -3, "angry": -2,
    "confused": -1, "poor": -2, "denied": -2, "unsafe": -3,
}
SEVERITY = {"low": 0, "medium": 10, "high": 20, "critical": 30}
FACTORS = {"vehicle_age", "property_age", "prior_claims", "loss_amount", "coverage_limit"}
SOURCE_FIELDS = {
    "policyholder_ref", "feedback", "policyholder_name", "vin",
    "property_address", "loss_amount", "loss_date",
}
XML_FIELDS = {
    "PolicyholderRef": "policyholder_ref", "Feedback": "feedback",
    "PolicyholderName": "policyholder_name", "VIN": "vin",
    "PropertyAddress": "property_address", "LossAmount": "loss_amount",
    "LossDate": "loss_date",
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional, label):
    require(isinstance(value, dict), label + " must be an object")
    require(required <= value.keys() <= required | optional,
            label + " has missing or unsupported fields")


def text(value, label, limit=2000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            label + " must be nonempty bounded text")
    return value.strip()


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            label + " must be a finite nonnegative number")


def parse_source(submission):
    fields(submission, {"format", "content"}, set(), "submission")
    fmt, content = submission["format"], submission["content"]
    require(isinstance(fmt, str) and fmt in {"acord_json", "acord_xml", "claim_text"},
            "unsupported submission format")
    if fmt == "acord_json":
        fields(content, {"ACORD"}, set(), "ACORD-style JSON")
        payload = content["ACORD"]
    elif fmt == "acord_xml":
        content = text(content, "XML content", 12000)
        require("<!DOCTYPE" not in content.upper() and "<!ENTITY" not in content.upper(),
                "XML declarations are not supported")
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            raise ValidationError("invalid ACORD-style XML") from None
        require(root.tag == "ACORD" and not root.attrib and not (root.text or "").strip(),
                "XML must use a plain ACORD root")
        payload = {}
        for child in root:
            require(child.tag in XML_FIELDS and not child.attrib and len(child) == 0
                    and not (child.tail or "").strip(), "unsupported XML field structure")
            key = XML_FIELDS[child.tag]
            require(key not in payload, "duplicate source field")
            payload[key] = child.text or ""
    else:
        content = text(content, "claim-form content", 12000)
        payload = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            name, sep, value = line.partition(":")
            require(bool(sep) and name.strip() in XML_FIELDS, "invalid claim-form field")
            key = XML_FIELDS[name.strip()]
            require(key not in payload, "duplicate source field")
            payload[key] = value.strip()
    fields(payload, {"policyholder_ref", "feedback"}, SOURCE_FIELDS, "source payload")
    require(bool(re.fullmatch(r"SYN-[A-Z0-9-]{1,32}",
                             text(payload["policyholder_ref"], "synthetic reference", 36))),
            "only synthetic policyholder references are accepted")
    text(payload["feedback"], "feedback")
    for key in {"policyholder_name", "property_address", "vin"} & payload.keys():
        text(payload[key], key, 200)
    if "vin" in payload:
        require(bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", payload["vin"])),
                "fabricated VIN must have 17 valid characters")
    if "loss_amount" in payload:
        amount = payload["loss_amount"]
        if fmt != "acord_json":
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                raise ValidationError("invalid loss amount") from None
        number(amount, "loss amount")
    if "loss_date" in payload:
        value = payload["loss_date"]
        require(isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)),
                "loss date must be YYYY-MM-DD")
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            raise ValidationError("invalid loss date") from None
    # Keep only the text needed for scoring; identifiers and assets never reach output.
    return payload["feedback"]


def normalize(document):
    fields(document, {"schema_version", "synthetic", "records"}, set(), "input")
    require(document["schema_version"] == "1.0", "unsupported schema version")
    require(document["synthetic"] is True, "only clearly labeled synthetic data is accepted")
    records = document["records"]
    require(isinstance(records, list) and 1 <= len(records) <= 100,
            "records must contain 1 to 100 items")
    normalized, seen = [], set()
    for item in records:
        fields(item, {"record_id", "entity", "severity", "submission"},
               {"claim_handling", "underwriting_factors"}, "record")
        record_id = text(item["record_id"], "record ID", 40)
        require(bool(re.fullmatch(r"REC-[A-Z0-9-]+", record_id)),
                "record ID must be an opaque REC- identifier")
        require(record_id not in seen, "duplicate record ID")
        seen.add(record_id)
        entity = item["entity"]
        require(isinstance(entity, str) and entity in {"policy", "claim", "underwriting_submission"},
                "unsupported entity")
        severity = item["severity"]
        require(isinstance(severity, str) and severity in SEVERITY, "unsupported severity")
        feedback = parse_source(item["submission"])
        if item["submission"]["format"] == "claim_text":
            require(entity == "claim", "claim-form text is restricted to claims")
        require(("claim_handling" in item) == (entity == "claim"),
                "claim handling is required only for claims")
        if entity == "claim":
            handling = item["claim_handling"]
            fields(handling, {"decision", "reasons"}, set(), "claim handling")
            require(isinstance(handling["decision"], str)
                    and handling["decision"] in {"pending", "approved", "denied"},
                    "unsupported supplied claim decision")
            reasons = handling["reasons"]
            require(isinstance(reasons, list) and 1 <= len(reasons) <= 10,
                    "all supplied claim decisions require stated reasons")
            for reason in reasons:
                text(reason, "claim reason", 500)
        require(("underwriting_factors" in item) == (entity == "underwriting_submission"),
                "underwriting factors are required only for underwriting submissions")
        factors = item.get("underwriting_factors", [])
        if entity == "underwriting_submission":
            require(isinstance(factors, list) and 1 <= len(factors) <= len(FACTORS),
                    "underwriting requires disclosed factors")
            names = set()
            for factor in factors:
                fields(factor, {"name", "value", "disclosed"}, set(), "underwriting factor")
                name = factor["name"]
                require(isinstance(name, str) and name in FACTORS,
                        "unsupported or protected underwriting factor")
                require(name not in names, "duplicate underwriting factor")
                names.add(name)
                require(factor["disclosed"] is True, "hidden underwriting factors are forbidden")
                number(factor["value"], "underwriting factor value")
        normalized.append({
            "record_id": record_id, "entity": entity, "severity": severity,
            "feedback": feedback, "factors": factors,
            "claim_reason_count": len(item["claim_handling"]["reasons"]) if entity == "claim" else 0,
        })
    return normalized


def sentiment(feedback):
    tokens = re.findall(r"[a-z]+|[.!?;,]", feedback.lower())
    matches, negate = [], False
    for token in tokens:
        if token in ".!?;,":
            negate = False
        elif token in {"not", "never", "no"}:
            negate = True
        elif token in LEXICON:
            base = LEXICON[token]
            matches.append({"term": token, "base_weight": base,
                            "negated": negate, "contribution": -base if negate else base})
            negate = False
        else:
            negate = False
    score = max(-5, min(5, sum(m["contribution"] for m in matches)))
    return {"score": score, "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
            "matches": matches}


def run(document):
    records = normalize(document)
    results = []
    for record in records:
        result = sentiment(record["feedback"])
        penalty = max(0, -result["score"])
        priority = SEVERITY[record["severity"]] + penalty
        results.append({
            "record_id": record["record_id"], "entity": record["entity"],
            "severity": record["severity"], "sentiment": result,
            "priority": {
                "score": priority,
                "band": "urgent" if priority >= 30 else "high" if priority >= 20
                        else "normal" if priority >= 10 else "low",
                "reasons": [f"Declared severity contributes {SEVERITY[record['severity']]} points.",
                            f"Negative sentiment contributes {penalty} points."],
            },
            "claim_handling": {
                "action": "human_review",
                "reason": "Review the supplied claim decision and stated reasons independently of sentiment.",
                "supplied_reason_count": record["claim_reason_count"],
            } if record["entity"] == "claim" else None,
            "underwriting_factors": record["factors"],
        })
    results.sort(key=lambda r: (-r["priority"]["score"], r["record_id"]))
    for rank, result in enumerate(results, 1):
        result["priority"]["rank"] = rank
    return {
        "status": "ok", "schema_version": "1.0", "synthetic": True,
        "results": results,
        "method": {
            "sentiment": "Lexicon sum clipped to [-5,5]; immediately preceding no/not/never reverses a match.",
            "priority": "severity base (0/10/20/30) + max(0,-sentiment); ties by record_id",
            "lexicon": LEXICON,
        },
        "safeguards": {
            "data_minimization": "Source data, policyholder references, free text, names, VINs and addresses are omitted.",
            "claims": "Queue triage only; no automated coverage, payout or adverse claim decisions.",
            "underwriting": "Only disclosed allowlisted factors; factors do not affect sentiment or priority.",
            "limitation": "Demonstrative validation only, not fairness or GDPR compliance certification.",
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        path = Path(args[0])
        require(path.stat().st_size <= 1_000_000, "input exceeds 1 MB limit")
        document = json.loads(path.read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object)
        response = run(document)
        code = 0
    except ValidationError as exc:
        response, code = {"status": "error", "message": str(exc)}, 2
    except (OSError, UnicodeError, ValueError, RecursionError, OverflowError):
        response, code = {"status": "error", "message": "Cannot read or decode valid input."}, 2
    print(json.dumps(response, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
