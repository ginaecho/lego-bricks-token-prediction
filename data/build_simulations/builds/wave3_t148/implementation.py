"""Synthetic telecom research: deterministic retrieval, not compliance certification."""
import csv
import io
import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys() <= set(required) | set(optional),
            "Missing or unsupported fields")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected nonempty text")
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value),
            "Expected a finite number")
    return value


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


CDR_FIELDS = ["id", "account_id", "residency", "phone", "imei", "service",
              "volume_mb", "duration_seconds", "note"]


def validate(payload):
    """Single input/industry validation boundary, yielding safe exact passages."""
    fields(payload, ("schema_version", "synthetic", "policy", "query", "limit", "records"))
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "Unsupported schema version")
    require(payload["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    policy = payload["policy"]
    fields(policy, ("residency", "purpose", "lawful_basis", "role"))
    residency = policy["residency"]
    require(residency in ("EU", "US"), "Unsupported residency")
    require(policy["purpose"] == "support_research", "Purpose is not authorized")
    require(policy["lawful_basis"] in ("consent", "contract"), "Lawful basis required")
    require(policy["role"] == "researcher", "Research access required")
    query = text(payload["query"])
    require(len(query) <= 1000 and bool(tokens(query)), "Invalid research query")
    require(type(payload["limit"]) is int and 1 <= payload["limit"] <= 20,
            "Limit must be an integer from 1 to 20")
    records = payload["records"]
    require(isinstance(records, list) and 1 <= len(records) <= 1000,
            "Expected 1 to 1000 records")
    accounts, seen = {}, set()
    protected, free_text, passages = [], [query], []
    for record in records:
        require(isinstance(record, dict), "Expected record object")
        kind = record.get("kind")
        require(kind in ("subscriber_account", "network_fault_ticket", "usage_record"),
                "Unknown entity kind")
        common = ("kind", "id", "residency")
        extra = {
            "subscriber_account": ("phone", "imei", "processing_allowed"),
            "network_fault_ticket": ("account_id", "transcript"),
            "usage_record": ("account_id", "cdr_csv"),
        }[kind]
        fields(record, common + extra, ("billing_adjustment",))
        rid = text(record["id"])
        prefix = {"subscriber_account": "acct", "network_fault_ticket": "ticket",
                  "usage_record": "usage"}[kind]
        require(re.fullmatch(prefix + r"-[a-z0-9]{1,12}", rid) is not None,
                "Record IDs must be synthetic pseudonyms")
        require(rid not in seen, "Duplicate record ID")
        seen.add(rid)
        require(record["residency"] == residency, "Record residency mismatch")
        if "billing_adjustment" in record:
            adjustment = record["billing_adjustment"]
            fields(adjustment, ("amount", "reason", "residency"))
            number(adjustment["amount"])
            free_text.append(text(adjustment["reason"]))
            require(adjustment["residency"] == residency, "Adjustment residency mismatch")
        if kind == "subscriber_account":
            require(record["processing_allowed"] is True, "Subscriber processing denied")
            require(isinstance(record["phone"], str) and
                    re.fullmatch(r"\+1-202-555-01[0-9]{2}", record["phone"]) is not None,
                    "Phone must use the synthetic fictional range")
            require(isinstance(record["imei"], str) and
                    re.fullmatch(r"000000[0-9]{9}", record["imei"]) is not None,
                    "IMEI must use the synthetic zero prefix")
            protected.extend((record["phone"], record["imei"]))
            accounts[rid] = record
    for record in records:
        kind, rid = record["kind"], record["id"]
        if kind == "subscriber_account":
            continue
        require(isinstance(record["account_id"], str) and record["account_id"] in accounts,
                "Unknown subscriber account")
        if kind == "network_fault_ticket":
            transcript = text(record["transcript"])
            require(len(transcript) <= 100000, "Transcript is too large")
            free_text.append(transcript)
            offset = 0
            for line_number, line in enumerate(transcript.splitlines(keepends=True), 1):
                quote = line.rstrip("\r\n")
                if quote.strip():
                    passages.append({
                        "quote": quote,
                        "citation": {"record_id": rid, "format": "support_chat",
                                     "field": "transcript", "line": line_number,
                                     "start": offset, "end": offset + len(quote)},
                    })
                offset += len(line)
        else:
            raw = text(record["cdr_csv"])
            require(len(raw) <= 100000, "CDR is too large")
            try:
                reader = csv.DictReader(io.StringIO(raw, newline=""), strict=True)
                require(reader.fieldnames == CDR_FIELDS, "Invalid CDR header")
                rows = list(reader)
            except csv.Error:
                raise ValidationError("Malformed CDR CSV") from None
            require(len(rows) == 1 and set(rows[0]) == set(CDR_FIELDS) and
                    all(isinstance(v, str) for v in rows[0].values()),
                    "Each usage record must contain exactly one complete CDR row")
            row = rows[0]
            account = accounts[record["account_id"]]
            require(row["id"] == rid and row["account_id"] == record["account_id"],
                    "CDR identity mismatch")
            require(row["residency"] == residency, "CDR residency mismatch")
            require(row["phone"] == account["phone"] and row["imei"] == account["imei"],
                    "CDR subscriber identifier mismatch")
            require(row["service"] in ("call", "data"), "Unsupported CDR service")
            for column in ("volume_mb", "duration_seconds"):
                try:
                    amount = float(row[column])
                except ValueError:
                    raise ValidationError("Invalid CDR usage number") from None
                require(math.isfinite(amount) and amount >= 0, "Invalid CDR usage number")
            note = text(row["note"])
            free_text.append(note)
            passages.append({
                "quote": note,
                "citation": {"record_id": rid, "format": "cdr_csv",
                             "row": 2, "column": "note"},
            })
    # Never rewrite quoted evidence: reject unsafe text before any retrieval.
    # This bounded guard is not a general-purpose PII detector.
    for value in free_text:
        require(not any(secret in value for secret in protected) and
                re.search(r"[\w.+-]+@[\w.-]+\.\w+|\d{7,}|\+\d[\d ()-]{7,}", value) is None,
                "Research text contains prohibited subscriber data")
    return passages


def research(payload):
    passages = validate(payload)
    terms = tokens(payload["query"])
    ranked = []
    for passage in passages:
        score = len(terms & tokens(passage["quote"]))
        if score:
            ranked.append({**passage, "matched_terms": sorted(terms & tokens(passage["quote"])),
                           "score": score})
    ranked.sort(key=lambda item: (-item["score"], item["citation"]["record_id"],
                                 item["citation"].get("start", 0)))
    findings = ranked[:payload["limit"]]
    return {"status": "ok", "schema_version": 1, "synthetic": True,
            "residency": payload["policy"]["residency"],
            "findings": findings, "matched_passages": len(ranked),
            "outcome": "findings" if findings else "no_matches"}


def reject_constant(_):
    raise ValidationError("Nonfinite JSON number")


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "Duplicate JSON key")
        obj[key] = value
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: implementation.py INPUT.json")
        with Path(argv[0]).open(encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        result = research(payload)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable input file"}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
