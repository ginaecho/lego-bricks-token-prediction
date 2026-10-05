"""Offline, synthetic insurance research -> accountable triage reference CLI.

Run: python -B implementation.py example_input.json
Retrieval uses supplied HTTP-response fixtures only; no network or model calls.
This demonstrates validation rules, not legal compliance or coverage decisions.
"""

import copy
import datetime
import decimal
import hashlib
import json
import re
import sys
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


VERSION = "1.0"
KINDS = {"policy": "Policy", "claim": "Claim",
         "underwriting_submission": "UnderwritingSubmission"}
PRIVATE_FIELDS = ("PolicyholderName", "VIN", "PropertyAddress", "Email")
COMMON_FIELDS = {"Reference", "Description", *PRIVATE_FIELDS}
CLAIM_FIELDS = {"PolicyReference", "LossAmount", "LossDate"}
UW_FIELDS = {"UnderwritingFactors", "DisclosedFactors"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "Missing or unsupported fields")


def text(value, limit=4000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            "Expected nonempty bounded text")
    require(not any(ord(c) < 32 and c not in "\n\t\r" for c in value),
            "Unsupported control character")
    return value


def identifier(value):
    text(value, 50)
    require(re.fullmatch(r"[A-Z]{2,8}-[A-Z0-9-]{1,40}", value) is not None,
            "Invalid pseudonymous reference")


def host(value):
    text(value, 253)
    require(re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
                         r"[a-z]{2,63}", value) is not None,
            "Invalid allowlisted hostname")


def url(value, allowed=None):
    text(value, 2048)
    require(not re.search(r"[\s\\%]", value), "Unsupported URL encoding")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(parsed.scheme == "https" and hostname is not None
            and parsed.netloc == hostname and not parsed.query
            and not parsed.fragment, "Only plain HTTPS fixture URLs are accepted")
    host(hostname)
    require(allowed is None or hostname in allowed, "URL hostname is not allowlisted")
    return value


def amount(value):
    require(type(value) in (str, int, float), "Invalid loss amount")
    try:
        number = decimal.Decimal(str(value))
    except decimal.InvalidOperation as exc:
        raise ValidationError("Invalid loss amount") from exc
    require(number.is_finite() and 0 < number <= 100000000
            and number.as_tuple().exponent >= -2,
            "Loss amount must be positive, bounded, and use at most two decimals")
    return float(number)


def factors(values, disclosed):
    require(isinstance(values, dict) and bool(values)
            and set(values) <= {"occupancy", "building_age"},
            "Unsupported or missing underwriting factors")
    require(isinstance(disclosed, list) and all(isinstance(x, str) for x in disclosed)
            and len(disclosed) == len(set(disclosed))
            and set(disclosed) == set(values),
            "All underwriting factors must be explicitly disclosed")
    if "occupancy" in values:
        require(values["occupancy"] in ("residential", "commercial"),
                "Unsupported occupancy")
    if "building_age" in values:
        require(type(values["building_age"]) is int
                and 0 <= values["building_age"] <= 300, "Invalid building age")


def normalize(raw, kind):
    extras = CLAIM_FIELDS if kind == "claim" else UW_FIELDS if kind == "underwriting_submission" else set()
    fields(raw, {"Reference"} | extras, COMMON_FIELDS - {"Reference"})
    identifier(raw["Reference"])
    for key in COMMON_FIELDS - {"Reference"}:
        if key in raw:
            text(raw[key], 1000)
    result = {"kind": kind, "reference": raw["Reference"]}
    if kind == "claim":
        identifier(raw["PolicyReference"])
        date = text(raw["LossDate"], 10)
        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) is not None,
                "Loss date must use YYYY-MM-DD")
        try:
            datetime.date.fromisoformat(date)
        except ValueError as exc:
            raise ValidationError("Invalid loss date") from exc
        result.update(policy_reference=raw["PolicyReference"],
                      loss_amount=amount(raw["LossAmount"]), loss_date=date)
    elif kind == "underwriting_submission":
        factors(raw["UnderwritingFactors"], raw["DisclosedFactors"])
        result.update(underwriting_factors=copy.deepcopy(raw["UnderwritingFactors"]),
                      disclosed_factors=sorted(raw["DisclosedFactors"]))
    return result


def parse_submission(ticket):
    kind, fmt, submission = ticket["entity_type"], ticket["format"], ticket["submission"]
    wrapper = KINDS[kind]
    if fmt == "acord_json":
        fields(submission, {"ACORD"})
        fields(submission["ACORD"], {wrapper})
        raw = submission["ACORD"][wrapper]
    elif fmt == "acord_xml":
        text(submission, 20000)
        require("<!DOCTYPE" not in submission.upper()
                and "<!ENTITY" not in submission.upper(), "XML declarations are forbidden")
        try:
            root = ET.fromstring(submission)
        except ET.ParseError as exc:
            raise ValidationError("Invalid ACORD-style XML") from exc
        nodes = list(root.iter())
        require(len(nodes) <= 100 and root.tag == "ACORD" and len(root) == 1
                and root[0].tag == wrapper, "Invalid ACORD-style XML structure")
        require(all(not node.attrib or (node.tag == "Factor"
                    and set(node.attrib) == {"name"}) for node in nodes),
                "Unsupported XML attributes")
        require(all(not (node.tail or "").strip() for node in nodes)
                and not (root.text or "").strip()
                and not (root[0].text or "").strip(), "Unexpected XML text")
        raw = {}
        for child in root[0]:
            require(child.tag not in raw and not child.attrib, "Duplicate or invalid XML field")
            if child.tag == "UnderwritingFactors":
                require(not (child.text or "").strip(), "Unexpected XML factor text")
                values = {}
                for factor in child:
                    require(factor.tag == "Factor" and len(factor) == 0
                            and set(factor.attrib) == {"name"}, "Invalid XML factor")
                    name = factor.attrib["name"]
                    require(name not in values, "Duplicate underwriting factor")
                    value = (factor.text or "").strip()
                    if name == "building_age":
                        require(value.isdigit() and len(value) <= 3, "Invalid building age")
                        value = int(value)
                    values[name] = value
                raw[child.tag] = values
            elif child.tag == "DisclosedFactors":
                require(not (child.text or "").strip(), "Unexpected disclosure text")
                require(all(x.tag == "Factor" and not x.attrib and len(x) == 0
                            for x in child), "Invalid XML disclosures")
                raw[child.tag] = [(x.text or "").strip() for x in child]
            else:
                require(len(child) == 0, "Nested XML field is unsupported")
                raw[child.tag] = (child.text or "").strip()
    else:
        require(fmt == "claim_form_text" and kind == "claim",
                "Claim-form text is only valid for claims")
        text(submission, 20000)
        raw = {}
        for line in submission.splitlines():
            if not line.strip():
                continue
            key, separator, value = line.partition(":")
            key = key.strip()
            require(bool(separator) and key not in raw, "Invalid or duplicate claim-form field")
            raw[key] = value.strip()
    normalized = normalize(raw, kind)
    private_values = [raw[key] for key in PRIVATE_FIELDS if key in raw]
    return normalized, private_values


def label(value):
    text(value, 80)
    require(re.fullmatch(r"[a-z][a-z0-9_-]*", value) is not None,
            "Category, team and accountable owner must be stable identifiers")


def validate_route(route):
    fields(route, {"team", "owner"})
    for value in route.values():
        label(value)


def validate_config(config):
    fields(config, {"categories", "routes", "high_loss_amount"})
    amount(config["high_loss_amount"])
    fields(config["categories"], set(KINDS))
    for category in config["categories"].values():
        label(category)
    fields(config["routes"], set(config["categories"].values()))
    for route in config["routes"].values():
        validate_route(route)


def validate(value, stage):
    """One validation entry point for external input and both stage boundaries."""
    if stage == "input":
        fields(value, {"schema_version", "synthetic_fixture", "allowlisted_hosts",
                       "documents", "ticket", "config"})
        require(value["schema_version"] == VERSION and value["synthetic_fixture"] is True,
                "Only version 1.0 explicitly synthetic fixtures are supported")
        hosts = value["allowlisted_hosts"]
        require(isinstance(hosts, list) and 1 <= len(hosts) <= 20, "Invalid allowlist")
        for entry in hosts:
            host(entry)
        require(len(set(hosts)) == len(hosts), "Duplicate allowlisted host")
        docs = value["documents"]
        require(isinstance(docs, list) and 1 <= len(docs) <= 20, "Invalid document fixtures")
        seen = set()
        for doc in docs:
            fields(doc, {"url", "status", "content_type", "body"})
            url(doc["url"], hosts)
            require(doc["url"] not in seen, "Duplicate source URL")
            seen.add(doc["url"])
            require(type(doc["status"]) is int and 100 <= doc["status"] <= 599,
                    "Invalid response status")
            text(doc["content_type"], 80)
            text(doc["body"], 12000)
        ticket = value["ticket"]
        fields(ticket, {"ticket_id", "entity_type", "format", "submission", "source_urls"})
        identifier(ticket["ticket_id"])
        require(isinstance(ticket["entity_type"], str) and ticket["entity_type"] in KINDS,
                "Unsupported insurance entity")
        require(isinstance(ticket["format"], str) and ticket["format"] in
                {"acord_json", "acord_xml", "claim_form_text"}, "Unsupported submission format")
        refs = ticket["source_urls"]
        require(isinstance(refs, list) and 1 <= len(refs) <= 20, "Source URLs are required")
        for ref in refs:
            url(ref, hosts)
            require(ref in seen, "Requested source has no retrieval fixture")
        require(len(refs) == len(set(refs)), "Duplicate requested source")
        parse_submission(ticket)
        validate_config(value["config"])
        return

    require(stage in {"research", "triage"}, "Unknown validation stage")
    base = {"schema_version", "synthetic_fixture", "stage", "ticket_id",
            "entity", "source_urls", "findings"}
    fields(value, base if stage == "research" else base | {"status", "triage", "privacy"})
    require(value["schema_version"] == VERSION and value["synthetic_fixture"] is True
            and value["stage"] == stage, "Invalid stage envelope")
    identifier(value["ticket_id"])
    entity = value["entity"]
    require(isinstance(entity, dict) and isinstance(entity.get("kind"), str)
            and entity["kind"] in KINDS, "Invalid normalized entity")
    kind = entity["kind"]
    needed = {"kind", "reference"}
    needed |= {"policy_reference", "loss_amount", "loss_date"} if kind == "claim" else (
        {"underwriting_factors", "disclosed_factors"} if kind == "underwriting_submission" else set())
    fields(entity, needed)
    reverse = {"reference": "Reference", "policy_reference": "PolicyReference",
               "loss_amount": "LossAmount", "loss_date": "LossDate",
               "underwriting_factors": "UnderwritingFactors", "disclosed_factors": "DisclosedFactors"}
    normalize({reverse[k]: v for k, v in entity.items() if k != "kind"}, kind)
    refs = value["source_urls"]
    require(isinstance(refs, list) and 1 <= len(refs) <= 20, "Invalid provenance sources")
    for ref in refs:
        url(ref)
    require(len(refs) == len(set(refs)), "Duplicate provenance source")
    findings = value["findings"]
    require(isinstance(findings, list) and 1 <= len(findings) <= 100,
            "Missing or excessive validated findings")
    ids = set()
    for finding in findings:
        fields(finding, {"finding_id", "entity_type", "statement", "provenance"})
        identifier(finding["finding_id"])
        require(finding["finding_id"] not in ids and finding["entity_type"] == kind,
                "Duplicate or unrelated finding")
        ids.add(finding["finding_id"])
        text(finding["statement"], 4000)
        provenance = finding["provenance"]
        fields(provenance, {"url", "line_number", "content_sha256", "retrieval_mode", "redacted"})
        require(isinstance(provenance["url"], str) and provenance["url"] in refs,
                "Finding source is outside validated handoff")
        require(type(provenance["line_number"]) is int and provenance["line_number"] > 0,
                "Invalid provenance line")
        require(isinstance(provenance["content_sha256"], str) and re.fullmatch(
            r"[a-f0-9]{64}", provenance["content_sha256"]) is not None, "Invalid source digest")
        require(provenance["retrieval_mode"] == "synthetic_fixture"
                and type(provenance["redacted"]) is bool, "Invalid retrieval provenance")
    if stage == "research":
        return
    require(value["status"] == "ok", "Invalid success status")
    triage = value["triage"]
    fields(triage, {"category", "priority", "route", "reasons", "decision", "human_review_required"})
    label(triage["category"])
    require(triage["priority"] in ("normal", "high"), "Invalid priority")
    validate_route(triage["route"])
    require(triage["decision"] == "route_only_no_coverage_or_eligibility_decision"
            and triage["human_review_required"] is True, "Unfair automated decision is forbidden")
    reasons = triage["reasons"]
    require(isinstance(reasons, list) and bool(reasons), "Stated reasons are required")
    for reason in reasons:
        fields(reason, {"code", "detail", "evidence_ids"})
        text(reason["code"], 80)
        text(reason["detail"], 1000)
        evidence = reason["evidence_ids"]
        require(isinstance(evidence, list) and bool(evidence)
                and all(isinstance(x, str) and x in ids for x in evidence),
                "Reason must cite validated findings")
    fields(value["privacy"], {"raw_policyholder_data_retained", "excluded_fields"})
    require(value["privacy"]["raw_policyholder_data_retained"] is False
            and value["privacy"]["excluded_fields"] == list(PRIVATE_FIELDS) + ["Description"],
            "Invalid data minimization declaration")


def research(data):
    validate(data, "input")
    entity, private_values = parse_submission(data["ticket"])
    fixtures = {doc["url"]: doc for doc in data["documents"]}
    findings = []
    for source in data["ticket"]["source_urls"]:
        document = fixtures[source]
        require(document["status"] == 200, "Fixture retrieval failed; redirects are not followed")
        require(document["content_type"] == "text/plain", "Unsupported retrieval content type")
        body = document["body"]
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        lines = body.splitlines()
        require(len(lines) <= 100, "Too many source lines")
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            label, separator, statement = line.partition("|")
            require(bool(separator) and label in KINDS and bool(statement.strip()),
                    "Source lines must be entity_type|finding text")
            if label != entity["kind"]:
                continue
            statement = statement.strip()
            original = statement
            for private in sorted(private_values, key=len, reverse=True):
                statement = re.sub(re.escape(private), "[REDACTED]", statement, flags=re.I)
            statement = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
                               "[REDACTED]", statement)
            findings.append({
                "finding_id": "FND-" + str(len(findings) + 1),
                "entity_type": label,
                "statement": statement,
                "provenance": {"url": source, "line_number": line_number,
                               "content_sha256": digest, "retrieval_mode": "synthetic_fixture",
                               "redacted": statement != original},
            })
    result = {"schema_version": VERSION, "synthetic_fixture": True, "stage": "research",
              "ticket_id": data["ticket"]["ticket_id"], "entity": entity,
              "source_urls": list(data["ticket"]["source_urls"]), "findings": findings}
    validate(result, "research")
    return result


def triage(research_output, config):
    validate(research_output, "research")
    validate_config(config)
    result = copy.deepcopy(research_output)
    entity = result["entity"]
    kind = entity["kind"]
    category = config["categories"][kind]
    evidence = [finding["finding_id"] for finding in result["findings"]]
    high = kind == "claim" and entity["loss_amount"] >= amount(config["high_loss_amount"])
    detail = "Validated " + kind + " guidance supports routing to the configured accountable team."
    reasons = [{"code": "entity_guidance_match", "detail": detail, "evidence_ids": evidence}]
    if kind == "claim":
        detail = ("Loss amount meets the configured escalation threshold." if high else
                  "Loss amount is below the configured escalation threshold.")
        reasons.append({"code": "transparent_loss_threshold", "detail": detail +
                        " Threshold: " + str(amount(config["high_loss_amount"])) + "." +
                        " Queue priority is not a coverage or payment decision.",
                        "evidence_ids": evidence})
    if kind == "underwriting_submission":
        reasons.append({"code": "disclosed_factors_only",
                        "detail": "Only disclosed factors are passed to human review: " +
                        ", ".join(entity["disclosed_factors"]) + ". No risk score is inferred.",
                        "evidence_ids": evidence})
    result.update(stage="triage", status="ok", triage={
        "category": category, "priority": "high" if high else "normal",
        "route": copy.deepcopy(config["routes"][category]), "reasons": reasons,
        "decision": "route_only_no_coverage_or_eligibility_decision",
        "human_review_required": True,
    }, privacy={"raw_policyholder_data_retained": False,
                "excluded_fields": list(PRIVATE_FIELDS) + ["Description"]})
    validate(result, "triage")
    return result


def run_pipeline(data):
    return triage(research(data), data["config"])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def reject_constant(_):
    raise ValidationError("Nonfinite JSON number")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], "rb") as source:
            payload = source.read(1000001)
        require(len(payload) <= 1000000, "Input exceeds one megabyte")
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        output = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, ValidationError) else "Unable to read or decode input"
        print(json.dumps({"status": "error", "message": message}))
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
