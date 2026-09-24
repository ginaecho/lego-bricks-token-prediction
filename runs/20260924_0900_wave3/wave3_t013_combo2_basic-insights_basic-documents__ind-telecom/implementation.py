"""Synthetic telecom reference pipeline; Python standard library only."""

import csv
import hashlib
import io
import json
import re
import sys
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


REGIONS = {"EU", "US"}
THEMES = {
    "connectivity": ("disconnect", "outage", "offline", "dropped"),
    "coverage": ("signal", "coverage", "roaming"),
    "billing": ("bill", "charge", "refund"),
    "device": ("device", "handset", "battery"),
}
ACTIONS = {
    "connectivity": "Investigate network reliability and affected fault tickets.",
    "coverage": "Review coverage and roaming availability.",
    "billing": "Review charges and explain eligible corrections.",
    "device": "Offer device diagnostics.",
    "other": "Review uncategorized feedback.",
}
REASONS = {"service_outage", "billing_correction", "goodwill"}
CDR_FIELDS = [
    "usage_id", "subscriber_id", "residency", "call_minutes", "data_mb",
    "billing_adjustment", "adjustment_reason",
]


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), "Missing required fields")
    require(set(value) <= set(required) | set(optional), "Unexpected fields")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def integer(value):
    return type(value) is int and 0 <= value <= 10**12


def money(value):
    require(isinstance(value, str) and
            re.fullmatch(r"-?(0|[1-9]\d{0,8})\.\d{2}", value) is not None,
            "Money must be a bounded decimal string with two places")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValidationError("Invalid money") from None
    return result


def reference(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{24}", value) is not None


def envelope(stage, records):
    return {"schema_version": 1, "fixture_label": "SYNTHETIC", "stage": stage,
            "records": records}


def validate(data, expected_stage):
    """Single schema boundary for source, insights, and documents."""
    fields(data, ("schema_version", "fixture_label", "stage", "records"),
           ("cdr_csv", "privacy") if expected_stage == "source" else ())
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Unsupported schema version")
    require(data["fixture_label"] == "SYNTHETIC", "Synthetic fixture label required")
    require(data["stage"] == expected_stage, "Unexpected pipeline stage")
    require(isinstance(data["records"], list) and 0 < len(data["records"]) <= 10000,
            "Expected 1 to 10000 records")
    if expected_stage == "source":
        _validate_source(data)
    else:
        require(expected_stage in {"insights", "documents"}, "Unknown stage")
        seen = set()
        for record in data["records"]:
            common = ("type", "subscriber_ref", "residency", "themes",
                      "ticket_refs", "usage", "adjustments")
            fields(record, common + (("document_id", "checks") if
                                      expected_stage == "documents" else ()))
            require(record["type"] == ("insight" if expected_stage == "insights"
                                       else "document"), "Invalid record type")
            ref = record["subscriber_ref"]
            require(reference(ref) and ref not in seen, "Invalid or duplicate subscriber reference")
            seen.add(ref)
            require(record["residency"] in REGIONS, "Unsupported residency")
            require(isinstance(record["ticket_refs"], list) and
                    all(reference(t) for t in record["ticket_refs"]) and
                    len(set(record["ticket_refs"])) == len(record["ticket_refs"]),
                    "Invalid ticket references")
            fields(record["usage"], ("record_count", "call_minutes", "data_mb",
                                     "billing_adjustment_total"))
            require(all(integer(record["usage"][k]) for k in
                        ("record_count", "call_minutes", "data_mb")), "Invalid usage totals")
            total = money(record["usage"]["billing_adjustment_total"])
            require(isinstance(record["themes"], list), "Invalid themes")
            theme_names = set()
            for theme in record["themes"]:
                fields(theme, ("name", "evidence_count", "action"))
                name = theme["name"]
                require(isinstance(name, str) and name in ACTIONS and name not in theme_names,
                        "Invalid or duplicate theme")
                theme_names.add(name)
                require(integer(theme["evidence_count"]) and theme["evidence_count"] > 0,
                        "Invalid evidence count")
                require(theme["action"] == ACTIONS[name], "Unapproved action text")
            require(isinstance(record["adjustments"], list), "Invalid adjustments")
            adjustment_total = Decimal("0.00")
            adjustment_ids = set()
            for adjustment in record["adjustments"]:
                fields(adjustment, ("usage_ref", "residency", "amount", "reason"))
                require(reference(adjustment["usage_ref"]) and
                        adjustment["usage_ref"] not in adjustment_ids, "Invalid adjustment reference")
                adjustment_ids.add(adjustment["usage_ref"])
                require(adjustment["residency"] == record["residency"],
                        "Adjustment residency mismatch")
                require(isinstance(adjustment["reason"], str) and
                        adjustment["reason"] in REASONS, "Billing adjustment requires a reason")
                amount = money(adjustment["amount"])
                require(amount != 0, "Zero adjustment must be omitted")
                adjustment_total += amount
            require(total == adjustment_total, "Adjustment reconciliation failed")
            require(len(record["adjustments"]) <= record["usage"]["record_count"],
                    "Too many adjustments")
            if expected_stage == "documents":
                require(record["document_id"] == "service-review-" + ref,
                        "Invalid document identifier")
                require(record["checks"] == {
                    "residency": "passed", "billing_reconciliation": "passed",
                    "subscriber_minimization": "passed",
                }, "Document checks failed")
    return data


def _validate_source(data):
    fields(data.get("privacy"), ("processing_permitted", "do_not_sell",
                                "purpose", "pseudonymization_key"))
    privacy = data["privacy"]
    require(privacy["processing_permitted"] is True and privacy["do_not_sell"] is True,
            "Subscriber privacy policy forbids this processing")
    require(privacy["purpose"] == "service_improvement", "Unsupported processing purpose")
    require(isinstance(privacy["pseudonymization_key"], str) and
            16 <= len(privacy["pseudonymization_key"]) <= 64, "Invalid pseudonymization key")
    accounts, tickets, identifiers = {}, {}, set()
    for record in data["records"]:
        require(isinstance(record, dict), "Expected record object")
        kind = record.get("type")
        required = {
            "subscriber_account": ("subscriber_id", "phone", "imei", "erasure_requested"),
            "network_fault_ticket": ("ticket_id", "subscriber_id", "category"),
            "support_chat": ("chat_id", "subscriber_id", "ticket_id", "text"),
            "call_data_usage": tuple(CDR_FIELDS[:2] + CDR_FIELDS[3:]),
        }
        require(isinstance(kind, str) and kind in required, "Unknown source record type")
        fields(record, ("type", "residency") + required[kind])
        require(isinstance(record["residency"], str) and record["residency"] in REGIONS,
                "Unsupported residency")
        require(text(record["subscriber_id"]), "Missing subscriber identifier")
        id_key = {"subscriber_account": "subscriber_id", "network_fault_ticket": "ticket_id",
                  "support_chat": "chat_id", "call_data_usage": "usage_id"}[kind]
        identifier = record[id_key]
        require(text(identifier), "Missing record identifier")
        require((kind, identifier) not in identifiers, "Duplicate record identifier")
        identifiers.add((kind, identifier))
        if kind == "subscriber_account":
            require(isinstance(record["phone"], str) and
                    re.fullmatch(r"\+120255501\d{2}", record["phone"]) is not None,
                    "Use fictitious reserved phone numbers")
            require(isinstance(record["imei"], str) and
                    re.fullmatch(r"000000\d{9}", record["imei"]) is not None,
                    "Use synthetic IMEI identifiers")
            require(record["erasure_requested"] is False, "Erasure-requested account cannot be processed")
            accounts[identifier] = record
        elif kind == "network_fault_ticket":
            require(isinstance(record["category"], str) and record["category"] in ACTIONS,
                    "Unknown fault category")
            tickets[identifier] = record
        elif kind == "support_chat":
            require(text(record["ticket_id"]) and text(record["text"]) and
                    len(record["text"]) <= 20000, "Invalid support transcript")
        else:
            require(integer(record["call_minutes"]) and integer(record["data_mb"]),
                    "Usage values must be bounded nonnegative integers")
            amount = money(record["billing_adjustment"])
            reason = record["adjustment_reason"]
            require(isinstance(reason, str) and
                    (reason in REASONS if amount else reason == ""),
                    "Billing adjustment requires an approved stated reason")
    require(bool(accounts), "At least one subscriber account is required")
    for record in data["records"]:
        account = accounts.get(record["subscriber_id"])
        require(account is not None, "Unknown subscriber account")
        require(record["residency"] == account["residency"], "Subscriber residency mismatch")
        if record["type"] == "support_chat":
            ticket = tickets.get(record["ticket_id"])
            require(ticket is not None and ticket["subscriber_id"] == record["subscriber_id"],
                    "Support chat ticket linkage mismatch")


def parse_source(payload):
    require(isinstance(payload, dict), "Expected input object")
    # A JSON round trip separates callers' input from normalization mutations.
    source = json.loads(json.dumps(payload, allow_nan=False))
    raw_csv = source.pop("cdr_csv", None)
    if raw_csv is not None:
        require(isinstance(raw_csv, str) and len(raw_csv) <= 2_000_000, "Invalid CDR CSV")
        require(isinstance(source.get("records"), list), "Expected records")
        try:
            reader = csv.DictReader(io.StringIO(raw_csv), strict=True)
            require(reader.fieldnames == CDR_FIELDS, "Invalid CDR CSV header")
            for row in reader:
                require(set(row) == set(CDR_FIELDS) and all(v is not None for v in row.values()),
                        "Malformed CDR CSV row")
                for key in ("call_minutes", "data_mb"):
                    require(re.fullmatch(r"\d{1,12}", row[key]) is not None,
                            "Invalid CSV usage number")
                    row[key] = int(row[key])
                row["type"] = "call_data_usage"
                source["records"].append(row)
        except csv.Error:
            raise ValidationError("Malformed CDR CSV") from None
    return validate(source, "source")


def customer_insights(source):
    validate(source, "source")
    key = source["privacy"]["pseudonymization_key"].encode("utf-8")

    def pseudonym(kind, identifier):
        return hashlib.blake2b((kind + ":" + identifier).encode("utf-8"),
                               key=hashlib.sha256(key).digest(), digest_size=12).hexdigest()

    result = []
    accounts = sorted((r for r in source["records"] if r["type"] == "subscriber_account"),
                      key=lambda r: r["subscriber_id"])
    for account in accounts:
        related = [r for r in source["records"] if r["subscriber_id"] == account["subscriber_id"]]
        counts, ticket_refs, adjustments = {}, [], []
        minutes = megabytes = usage_count = 0
        total = Decimal("0.00")
        for record in related:
            if record["type"] == "network_fault_ticket":
                ticket_refs.append(pseudonym("ticket", record["ticket_id"]))
                names = {record["category"]}
            elif record["type"] == "support_chat":
                words = set(re.findall(r"[a-z]+", record["text"].lower()))
                names = {name for name, keywords in THEMES.items() if words.intersection(keywords)}
                names = names or {"other"}
            elif record["type"] == "call_data_usage":
                usage_count += 1
                minutes += record["call_minutes"]
                megabytes += record["data_mb"]
                amount = money(record["billing_adjustment"])
                total += amount
                if amount:
                    adjustments.append({
                        "usage_ref": pseudonym("usage", record["usage_id"]),
                        "residency": record["residency"], "amount": format(amount, ".2f"),
                        "reason": record["adjustment_reason"],
                    })
                continue
            else:
                continue
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        result.append({
            "type": "insight", "subscriber_ref": pseudonym("subscriber", account["subscriber_id"]),
            "residency": account["residency"],
            "themes": [{"name": name, "evidence_count": count, "action": ACTIONS[name]}
                       for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))],
            "ticket_refs": sorted(ticket_refs),
            "usage": {"record_count": usage_count, "call_minutes": minutes, "data_mb": megabytes,
                      "billing_adjustment_total": format(total, ".2f")},
            "adjustments": sorted(adjustments, key=lambda a: a["usage_ref"]),
        })
    return validate(envelope("insights", result), "insights")


def automate_documents(insights):
    validate(insights, "insights")
    documents = []
    for insight in insights["records"]:
        document = json.loads(json.dumps(insight))
        document.update(type="document",
                        document_id="service-review-" + insight["subscriber_ref"],
                        checks={"residency": "passed", "billing_reconciliation": "passed",
                                "subscriber_minimization": "passed"})
        documents.append(document)
    return validate(envelope("documents", documents), "documents")


def run_pipeline(payload):
    insights = customer_insights(parse_source(payload))
    documents = automate_documents(insights)
    return {"status": "ok", "insights": insights, "documents": documents}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Expected one input JSON filename")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        output = run_pipeline(payload)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Do not echo input values, transcript text, keys, or file paths on errors.
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable input file"}))
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
