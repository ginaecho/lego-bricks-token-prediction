"""Synthetic financial document review; not a compliance certification."""

import csv
import io
import json
import re
import sys
from datetime import date
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "Missing or unsupported fields")


def identifier(value):
    require(isinstance(value, str) and
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", value) is not None,
            "Invalid identifier")
    return value


def text(value):
    require(isinstance(value, str) and 0 < len(value) <= 200, "Invalid text")
    return value


def day(value):
    require(isinstance(value, str) and
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None, "Invalid date")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValidationError("Invalid date") from None


def money(value, positive=False):
    require(not isinstance(value, bool) and isinstance(value, (str, int, float)),
            "Invalid monetary value")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError("Invalid monetary value") from None
    require(result.is_finite() and 0 <= result <= Decimal("1000000000000"),
            "Monetary value outside permitted range")
    require(result == result.quantize(Decimal("0.01")), "Use at most two decimal places")
    require(not positive or result > 0, "Monetary value must be positive")
    return result


def currency(value):
    require(isinstance(value, str) and re.fullmatch("[A-Z]{3}", value) is not None,
            "Invalid currency")


def synthetic_account(value):
    require(isinstance(value, str) and
            re.fullmatch(r"FAKE-IBAN-[A-Z0-9]{4,24}", value) is not None,
            "Only clearly fake account identifiers are accepted")


def mask(value):
    if isinstance(value, dict):
        return {key: mask(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\d{13,19}", lambda match: "*" * (len(match[0]) - 4) +
                      match[0][-4:], value)
    return value


LEDGER_FIELDS = ["id", "customer_id", "amount", "currency", "date", "card_number"]
REQUIREMENTS = {
    "customer": ("identity", "address", "aml_review"),
    "transaction": ("payment_authorization",),
    "loan": ("income", "affordability"),
}


def validate(payload):
    """Single input gate: normalize ledgers and validate all shared entities."""
    fields(payload, ("schema_version", "synthetic", "as_of", "customers",
                     "transaction_ledger", "loans", "payments", "documents"))
    require(payload["schema_version"] == "1.0", "Unsupported schema version")
    require(payload["synthetic"] is True, "Synthetic fixtures only")
    as_of = day(payload["as_of"])
    for key in ("customers", "loans", "payments", "documents"):
        require(isinstance(payload[key], list), "Expected a list")
        require(len(payload[key]) <= 1000, "Fixture limit exceeded")
    ledger = payload["transaction_ledger"]
    fields(ledger, ("format", "data"))
    require(ledger["format"] in ("json", "csv"), "Unsupported ledger format")
    if ledger["format"] == "json":
        require(isinstance(ledger["data"], list), "JSON ledger must be a list")
        transactions = ledger["data"]
    else:
        require(isinstance(ledger["data"], str), "CSV ledger must be text")
        try:
            reader = csv.DictReader(io.StringIO(ledger["data"]), strict=True)
            require(reader.fieldnames == LEDGER_FIELDS, "Unexpected CSV header")
            transactions = list(reader)
        except csv.Error:
            raise ValidationError("Malformed CSV ledger") from None
    require(len(transactions) <= 1000, "Fixture limit exceeded")
    customers, txs, loans, documents = {}, {}, {}, {}
    for customer in payload["customers"]:
        fields(customer, ("id", "name", "account", "kyc_status", "screening"))
        key = identifier(customer["id"])
        require(key not in customers, "Duplicate customer")
        text(customer["name"])
        synthetic_account(customer["account"])
        require(customer["kyc_status"] in ("complete", "pending", "rejected"),
                "Invalid KYC status")
        fields(customer["screening"], ("sanctions", "pep"))
        require(all(type(v) is bool for v in customer["screening"].values()),
                "Screening flags must be booleans")
        customers[key] = customer
    for tx in transactions:
        fields(tx, LEDGER_FIELDS)
        key = identifier(tx["id"])
        require(key not in txs, "Duplicate transaction")
        require(identifier(tx["customer_id"]) in customers, "Unknown customer")
        money(tx["amount"], positive=True)
        currency(tx["currency"])
        require(day(tx["date"]) <= as_of, "Future transaction")
        require(isinstance(tx["card_number"], str) and
                re.fullmatch(r"\d{13,19}", tx["card_number"]) is not None,
                "Card number must contain 13 to 19 digits")
        txs[key] = tx
    for loan in payload["loans"]:
        fields(loan, ("id", "customer_id", "amount", "currency", "monthly_income",
                      "monthly_payment"))
        key = identifier(loan["id"])
        require(key not in loans, "Duplicate loan")
        require(identifier(loan["customer_id"]) in customers, "Unknown customer")
        money(loan["amount"], positive=True)
        money(loan["monthly_income"])
        money(loan["monthly_payment"], positive=True)
        currency(loan["currency"])
        loans[key] = loan
    payment_ids, paid_transactions = set(), set()
    for payment in payload["payments"]:
        fields(payment, ("MsgId", "transaction_id", "PmtInf"))
        pid = identifier(payment["MsgId"])
        tid = identifier(payment["transaction_id"])
        require(pid not in payment_ids and tid not in paid_transactions,
                "Duplicate payment or transaction payment")
        require(tid in txs, "Unknown payment transaction")
        payment_ids.add(pid)
        paid_transactions.add(tid)
        info = payment["PmtInf"]
        fields(info, ("Dbtr", "Cdtr", "InstdAmt"))
        for role in ("Dbtr", "Cdtr"):
            fields(info[role], ("Nm", "IBAN"))
            text(info[role]["Nm"])
            synthetic_account(info[role]["IBAN"])
        fields(info["InstdAmt"], ("Ccy", "Value"))
        currency(info["InstdAmt"]["Ccy"])
        require(money(info["InstdAmt"]["Value"], True) == money(txs[tid]["amount"])
                and info["InstdAmt"]["Ccy"] == txs[tid]["currency"],
                "Payment amount or currency mismatch")
        require(info["Dbtr"]["IBAN"] == customers[txs[tid]["customer_id"]]["account"],
                "Payment debtor account mismatch")
    entities = {"customer": customers, "transaction": txs, "loan": loans}
    for doc in payload["documents"]:
        fields(doc, ("id", "entity_type", "entity_id", "kind", "verified",
                     "issued_on", "expires_on"))
        key = identifier(doc["id"])
        require(key not in documents, "Duplicate document")
        require(isinstance(doc["entity_type"], str) and
                doc["entity_type"] in entities, "Unknown document entity type")
        require(identifier(doc["entity_id"]) in entities[doc["entity_type"]],
                "Unknown document entity")
        require(doc["kind"] in REQUIREMENTS[doc["entity_type"]],
                "Invalid evidence kind")
        require(type(doc["verified"]) is bool, "Verified must be boolean")
        issued = day(doc["issued_on"])
        expires = day(doc["expires_on"]) if doc["expires_on"] is not None else None
        require(expires is None or expires >= issued, "Evidence dates inverted")
        documents[key] = doc
    return as_of, entities, documents, paid_transactions


def review(payload):
    as_of, entities, documents, paid = validate(payload)
    results, gaps = [], []
    for entity_type, items in entities.items():
        for entity_id, entity in sorted(items.items()):
            checks, contributions, flags = [], [], []

            def add_flag(code, points, basis):
                flags.append(code)
                contributions.append({"rule": code, "points": points, "basis": basis})

            for requirement in REQUIREMENTS[entity_type]:
                candidates = sorted(
                    (doc for doc in documents.values()
                     if doc["entity_type"] == entity_type and
                     doc["entity_id"] == entity_id and doc["kind"] == requirement),
                    key=lambda doc: doc["id"])
                evidence = []
                for doc in candidates:
                    reasons = []
                    if not doc["verified"]:
                        reasons.append("unverified")
                    if day(doc["issued_on"]) > as_of:
                        reasons.append("not_yet_issued")
                    if doc["expires_on"] is not None and day(doc["expires_on"]) < as_of:
                        reasons.append("expired")
                    evidence.append({"document_id": doc["id"], "accepted": not reasons,
                                     "rejection_reasons": reasons})
                passed = any(e["accepted"] for e in evidence)
                check = {"requirement": requirement,
                         "status": "satisfied" if passed else "gap", "evidence": evidence}
                checks.append(check)
                if not passed:
                    gap_id = f"{entity_type}:{entity_id}:{requirement}"
                    gaps.append({"id": gap_id, "entity_type": entity_type,
                                 "entity_id": entity_id, "requirement": requirement,
                                 "reason": "missing" if not candidates else "no_valid_evidence",
                                 "document_ids": [d["id"] for d in candidates],
                                 "action": f"Provide verified, current {requirement} evidence"})
                    contributions.append({"rule": "evidence_gap", "points": 10,
                                          "basis": gap_id})
            customer = entity if entity_type == "customer" else entities["customer"][entity["customer_id"]]
            if customer["kyc_status"] != "complete":
                add_flag("kyc_incomplete", 20, f"customer:{customer['id']}:kyc_status")
            for screening, points in (("sanctions", 40), ("pep", 15)):
                if customer["screening"][screening]:
                    add_flag(f"aml_{screening}", points,
                             f"customer:{customer['id']}:screening.{screening}")
            if entity_type == "transaction":
                if money(entity["amount"]) >= 10000:
                    add_flag("aml_large_transaction", 15, "amount >= 10000 currency units")
                if entity_id not in paid:
                    add_flag("payment_payload_missing", 10, "No linked ISO-style payment")
            if entity_type == "loan":
                if (money(entity["monthly_income"]) == 0 or
                        money(entity["monthly_payment"]) > money(entity["monthly_income"]) * Decimal("0.4")):
                    add_flag("affordability_review", 25,
                             "income = 0 or monthly_payment > 40% of monthly_income")
            raw = sum(c["points"] for c in contributions)
            result = {
                "entity_type": entity_type, "entity_id": entity_id,
                "checks": checks, "screening_flags": flags,
                "disposition": "manual_review" if contributions else "requirements_met",
                "risk_score": {"value": min(100, raw), "raw_total": raw,
                               "formula": "min(100, sum(contribution.points)); baseline 0",
                               "contributions": contributions,
                               "interpretation": "Demonstrative review priority, not creditworthiness"},
            }
            if entity_type == "transaction":
                result["card_number"] = mask(entity["card_number"])
            results.append(result)
    return mask({
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "as_of": payload["as_of"], "stage": "document_review",
        "notice": "Synthetic demonstration only. No compliance certification, AML clearance, or lending decision.",
        "results": results, "gaps": gaps,
    })


def reject_constant(_):
    raise ValidationError("Nonfinite JSON number")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        output = review(payload)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        print(json.dumps({"schema_version": "1.0", "status": "error",
                          "error": "Invalid input or unreadable input file"}))
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
