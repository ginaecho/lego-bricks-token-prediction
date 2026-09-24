"""Synthetic telecom evidence review; demonstrative safeguards, not certification.

Schema version 1: records share id/residency. Documents hold support-chat text
or CSV CDRs. Requirements reference one entity and explicitly selected documents.
Evidence requires all case-insensitive literal terms in ONE linked document.
Raw subscriber data and document content are never returned.
"""

import csv
import io
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate(data):
    require(isinstance(data, dict), "Input must be an object")
    require(data.get("schema_version") == "1", "Unsupported schema_version")
    require(data.get("synthetic") is True, "Only labeled synthetic input is supported")
    region = data.get("processing_residency")
    require(region in ("EU", "US"), "processing_residency must be EU or US")
    tables = {}
    names = ("accounts", "tickets", "usage_records", "billing_adjustments",
             "documents", "requirements")
    all_ids = set()
    for name in names:
        rows = data.get(name)
        require(isinstance(rows, list), "Missing or invalid collection: " + name)
        tables[name] = {}
        for row in rows:
            require(isinstance(row, dict), "Records must be objects")
            rid = row.get("id")
            require(isinstance(rid, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", rid),
                    "Record identifiers must be short safe identifiers")
            require(rid not in all_ids, "Duplicate record identifier")
            all_ids.add(rid)
            require(row.get("residency") == region, "Record residency mismatch")
            tables[name][rid] = row
    for account in tables["accounts"].values():
        require(account.get("jurisdiction") in ("GDPR", "CCPA"), "Unknown privacy jurisdiction")
        require(account.get("subscriber_data_protected") is True,
                "Subscriber protection declaration is required")
        require(text(account.get("purpose")), "Subscriber processing purpose is required")
        require(account.get("lawful_basis") in ("consent", "contract", "legal_obligation"),
                "Demonstrative lawful basis is required")
        require(account.get("phone") == "+1-202-555-0100", "Use designated fictitious phone")
        require(account.get("imei") == "000000000000000", "Use designated dummy IMEI")
    for name in ("tickets", "usage_records", "billing_adjustments"):
        for row in tables[name].values():
            require(row.get("account_id") in tables["accounts"], "Unknown subscriber account")
    for row in tables["tickets"].values():
        require(row.get("state") in ("open", "resolved"), "Invalid fault ticket state")
        require(text(row.get("summary")), "Ticket summary is required")
    for row in tables["usage_records"].values():
        require(number(row.get("data_mb")) and row["data_mb"] >= 0, "Invalid data volume")
        require(type(row.get("call_seconds")) is int and row["call_seconds"] >= 0,
                "Invalid call duration")
    for row in tables["billing_adjustments"].values():
        require(number(row.get("amount")), "Invalid billing amount")
        require(text(row.get("reason")), "Billing adjustment requires a stated reason")
    entity_tables = {"account": "accounts", "ticket": "tickets", "usage": "usage_records"}
    for doc in tables["documents"].values():
        kind = doc.get("kind")
        require(kind in ("cdr_csv", "support_chat"), "Unsupported document kind")
        require(text(doc.get("content")), "Document content is required")
        links = doc.get("entity_ids")
        require(isinstance(links, list) and bool(links) and
                all(isinstance(i, str) and any(i in tables[n] for n in entity_tables.values())
                    for i in links), "Document entity links are invalid")
        if kind == "cdr_csv":
            validate_cdr(doc, tables, region)
    for row in tables["requirements"].values():
        table = entity_tables.get(row.get("entity_type"))
        require(table is not None and row.get("entity_id") in tables[table],
                "Unknown requirement entity")
        terms = row.get("required_terms")
        require(isinstance(terms, list) and bool(terms) and all(text(t) for t in terms),
                "Requirements need nonblank terms")
        refs = row.get("document_ids")
        require(isinstance(refs, list) and all(isinstance(i, str) and i in tables["documents"]
                                            for i in refs), "Unknown requirement document")
    return tables


def validate_cdr(doc, tables, region):
    try:
        reader = csv.DictReader(io.StringIO(doc["content"]), strict=True)
        expected = ["id", "account_id", "residency", "call_seconds", "data_mb"]
        require(reader.fieldnames == expected, "CDR CSV header mismatch")
        seen = set()
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()),
                    "Malformed CDR CSV row")
            rid = row["id"]
            require(rid in tables["usage_records"] and rid not in seen, "Unknown or duplicate CDR")
            seen.add(rid)
            require(rid in doc["entity_ids"], "CDR must link its usage record")
            usage = tables["usage_records"][rid]
            require(row["residency"] == region, "CDR residency mismatch")
            require(row["account_id"] == usage["account_id"], "CDR account mismatch")
            require(re.fullmatch(r"\d+", row["call_seconds"]) is not None,
                    "Invalid CDR call duration")
            volume = float(row["data_mb"])
            require(math.isfinite(volume) and volume >= 0, "Invalid CDR volume")
            require(int(row["call_seconds"]) == usage["call_seconds"] and
                    volume == usage["data_mb"], "CDR usage mismatch")
        require(bool(seen), "CDR CSV needs at least one row")
    except (csv.Error, ValueError, OverflowError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("Malformed CDR CSV") from None


def review(data):
    """Validate once, then produce content-free traceable review findings."""
    try:
        tables = validate(data)
    except (TypeError, KeyError):
        raise ValidationError("Invalid field type or structure") from None
    findings = []
    for req in tables["requirements"].values():
        inspected = []
        evidence = []
        for did in dict.fromkeys(req["document_ids"]):
            doc = tables["documents"][did]
            inspected.append(did)
            if req["entity_id"] in doc["entity_ids"] and all(
                    term.casefold() in doc["content"].casefold() for term in req["required_terms"]):
                evidence.append(did)
        findings.append({
            "requirement_id": req["id"],
            "entity_type": req["entity_type"],
            "entity_id": req["entity_id"],
            "residency": data["processing_residency"],
            "result": "supported" if evidence else "gap",
            "evidence_document_ids": evidence,
            "inspected_document_ids": inspected,
            "gap_reason": None if evidence else "No linked document contains all required terms",
        })
    gaps = sum(f["result"] == "gap" for f in findings)
    return {
        "schema_version": "1", "synthetic": True, "status": "ok",
        "processing_residency": data["processing_residency"],
        "review_state": "not_assessed" if not findings else ("gaps_found" if gaps else "supported"),
        "summary": {"requirements": len(findings), "supported": len(findings) - gaps, "gaps": gaps},
        "findings": findings,
        "notice": "Synthetic demonstrative evidence checking only; no GDPR/CCPA or certification claim.",
    }


def reject_constant(value):
    raise ValidationError("Non-finite JSON number")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Expected one input JSON path")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = review(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, ValidationError) else "Unable to read valid input JSON"
        print(json.dumps({"schema_version": "1", "status": "error", "message": message}))
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
