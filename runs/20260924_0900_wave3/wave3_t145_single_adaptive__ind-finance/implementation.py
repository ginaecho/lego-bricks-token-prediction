"""Synthetic financial onboarding reference; no regulatory certification or advice."""

import copy
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
    require(set(required) <= set(value), "Missing required fields")
    require(set(value) <= set(required) | set(optional), "Unknown fields")


def text(value, label):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 200,
            label + " must be nonempty text of at most 200 characters")
    return value


def money(value, label, positive=False):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            label + " must be a monetary value")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(label + " must be numeric") from None
    require(result.is_finite() and abs(result) <= Decimal("1000000000000"),
            label + " must be finite and bounded")
    require(result == result.quantize(Decimal("0.01")), label + " requires cents precision")
    require(not positive or result > 0, label + " must be positive")
    return format(result, ".2f")


def mask(value):
    """Recursively mask contiguous or space/hyphen-separated PAN-like digit runs."""
    if isinstance(value, dict):
        return {mask(k): mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
                      lambda m: "*" * (len(re.sub(r"\D", "", m[0])) - 4)
                      + re.sub(r"\D", "", m[0])[-4:], value)
    return value


def validate(payload, output=False):
    """Shared boundary validator for the versioned input/output envelope."""
    require(isinstance(payload, dict), "Root must be an object")
    require(payload.get("schema_version") == "1.0", "Unsupported schema_version")
    require(payload.get("synthetic") is True, "Only explicitly synthetic data is accepted")
    if output:
        fields(payload, ("schema_version", "synthetic", "status", "customer",
                         "transactions", "payment", "loan_application", "onboarding",
                         "scores", "disclaimer"))
        require(payload["status"] in ("ready", "in_progress", "review_required"),
                "Invalid output status")
        for score in payload["scores"]:
            fields(score, ("name", "value", "explanation"))
            require(type(score["value"]) is int and 0 <= score["value"] <= 100,
                    "Score out of bounds")
            text(score["explanation"], "Score explanation")
        require(mask(payload) == payload, "Output contains unmasked PAN-like data")
        return payload
    fields(payload, ("schema_version", "synthetic", "customer", "ledger", "payment"),
           ("loan_application", "completed_steps"))
    customer = payload["customer"]
    fields(customer, ("customer_id", "name", "experience", "preference",
                      "kyc_status", "aml_flags", "account_iban"), ("card_number",))
    for key in ("customer_id", "name"):
        text(customer[key], key)
    require(customer["experience"] in ("beginner", "intermediate", "expert"),
            "Invalid experience")
    require(customer["preference"] in ("guided", "concise"), "Invalid preference")
    require(customer["kyc_status"] in ("pending", "verified", "rejected"),
            "Invalid KYC status")
    require(isinstance(customer["aml_flags"], list)
            and len(customer["aml_flags"]) <= 20, "Invalid AML flags")
    for flag in customer["aml_flags"]:
        text(flag, "AML flag")
    require(isinstance(customer["account_iban"], str)
            and re.fullmatch(r"ZZ00[A-Z0-9]{8,26}", customer["account_iban"]),
            "Use a clearly fake ZZ00-prefixed account/IBAN")
    if "card_number" in customer:
        require(isinstance(customer["card_number"], str)
                and re.fullmatch(r"[0-9 -]+", customer["card_number"])
                and 13 <= len(re.sub(r"\D", "", customer["card_number"])) <= 19,
                "Invalid synthetic card number")
    ledger = payload["ledger"]
    fields(ledger, ("format", "data"))
    require(ledger["format"] in ("json", "csv"), "Ledger format must be json or csv")
    keys = ("transaction_id", "date", "amount", "currency", "description")
    if ledger["format"] == "csv":
        require(isinstance(ledger["data"], str) and len(ledger["data"]) <= 1000000,
                "CSV data must be bounded text")
        reader = csv.DictReader(io.StringIO(ledger["data"]), strict=True)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(keys)
                and set(reader.fieldnames) == set(keys), "Invalid CSV headers")
        try:
            rows = list(reader)
        except csv.Error:
            raise ValidationError("Malformed CSV") from None
    else:
        rows = ledger["data"]
    require(isinstance(rows, list) and len(rows) <= 1000, "Ledger must contain at most 1000 rows")
    normalized = []
    identifiers = set()
    for row in rows:
        fields(row, keys)
        identifier = text(row["transaction_id"], "transaction_id")
        require(identifier not in identifiers, "Duplicate transaction_id")
        identifiers.add(identifier)
        require(isinstance(row["date"], str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]), "Invalid date")
        try:
            date.fromisoformat(row["date"])
        except ValueError:
            raise ValidationError("Invalid calendar date") from None
        require(row["currency"] in ("EUR", "USD", "GBP"), "Unsupported currency")
        text(row["description"], "description")
        normalized.append(dict(row, amount=money(row["amount"], "Transaction amount")))
    payment = payload["payment"]
    fields(payment, ("MsgId", "EndToEndId", "InstdAmt", "DbtrAcct", "CdtrAcct"))
    for key in ("MsgId", "EndToEndId"):
        text(payment[key], key)
    fields(payment["InstdAmt"], ("Ccy", "Value"))
    require(payment["InstdAmt"]["Ccy"] in ("EUR", "USD", "GBP"), "Unsupported payment currency")
    for key in ("DbtrAcct", "CdtrAcct"):
        fields(payment[key], ("IBAN",))
        require(isinstance(payment[key]["IBAN"], str)
                and re.fullmatch(r"ZZ00[A-Z0-9]{8,26}", payment[key]["IBAN"]),
                "Payment accounts must be clearly fake ZZ00 accounts")
    require(payment["DbtrAcct"]["IBAN"] == customer["account_iban"],
            "Payment debtor must match customer")
    normalized_payment = dict(payment, InstdAmt={
        "Ccy": payment["InstdAmt"]["Ccy"],
        "Value": money(payment["InstdAmt"]["Value"], "Payment amount", positive=True)})
    loan = payload.get("loan_application")
    if loan is not None:
        fields(loan, ("application_id", "amount", "currency", "term_months", "purpose"))
        text(loan["application_id"], "application_id")
        text(loan["purpose"], "purpose")
        require(loan["currency"] in ("EUR", "USD", "GBP"), "Unsupported loan currency")
        require(type(loan["term_months"]) is int and 1 <= loan["term_months"] <= 600,
                "Invalid loan term")
        loan = dict(loan, amount=money(loan["amount"], "Loan amount", positive=True))
    completed = payload.get("completed_steps", [])
    require(isinstance(completed, list) and all(isinstance(x, str) for x in completed),
            "completed_steps must be a list of strings")
    require(len(completed) == len(set(completed)), "Duplicate completed steps")
    return customer, normalized, normalized_payment, loan, set(completed)


def onboard(payload, explanation_hook=None):
    customer, transactions, payment, loan, completed = validate(payload)
    detailed = customer["experience"] == "beginner" or customer["preference"] == "guided"
    gates = []
    if customer["kyc_status"] != "verified":
        gates.append("KYC identity verification requires human review")
    if customer["aml_flags"]:
        gates.append("AML screening flags require human review")
    definitions = [
        ("profile", [], "Review your synthetic identity and communication preferences."),
        ("screening", ["profile"], "KYC verification and AML screening must clear before proceeding."),
        ("ledger", ["screening"], "Review dates, signed amounts and currencies; do not aggregate unlike currencies."),
        ("payment", ["ledger"], "Confirm debtor, creditor, currency and amount in the ISO 20022-style payload."),
    ]
    if loan:
        definitions.append(("loan", ["payment"],
                            "Review the requested principal, purpose and term; this is not a lending decision."))
    require(completed <= {item[0] for item in definitions}, "Unknown completed step")
    steps = []
    done = set()
    for identifier, prerequisites, explanation in definitions:
        blocked = any(p not in done for p in prerequisites) or (identifier == "screening" and gates)
        require(not (identifier in completed and blocked),
                "Completed step has unmet prerequisites or unresolved screening")
        state = "complete" if identifier in completed else "blocked" if blocked else "available"
        if state == "complete":
            done.add(identifier)
        step = {"id": identifier, "prerequisites": prerequisites, "state": state,
                "explanation": explanation,
                "guidance": ("Read the explanation, inspect the supplied data, then acknowledge this step."
                             if detailed else "Inspect and acknowledge.")}
        if explanation_hook is not None:
            try:
                guidance = explanation_hook(copy.deepcopy(step))
            except Exception:
                raise ValidationError("Injected guidance callable failed") from None
            step["guidance"] = text(guidance, "Injected guidance")
        steps.append(step)
    readiness = len(done) * 100 // len(steps)
    result = {
        "schema_version": "1.0", "synthetic": True,
        "status": "review_required" if gates else "ready" if readiness == 100 else "in_progress",
        "customer": customer, "transactions": transactions,
        "payment": payment, "loan_application": loan,
        "onboarding": {"mode": "guided" if detailed else "concise", "steps": steps,
                       "screening_reasons": gates,
                       "next_step": next((s["id"] for s in steps if s["state"] == "available"), None)},
        "scores": [{"name": "onboarding_readiness", "value": readiness,
                    "explanation": f"floor({len(done)} completed / {len(steps)} required steps * 100). "
                    "Screening gates block dependent completion; this is not creditworthiness."}],
        "disclaimer": "SYNTHETIC DEMONSTRATION ONLY. No compliance certification, payment execution or loan approval."
    }
    return validate(mask(result), output=True)


def reject_constant(value):
    raise ValidationError("Nonfinite JSON constants are not permitted")


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
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = onboard(payload)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, csv.Error, RecursionError):
        # Never echo untrusted input or exception text into CLI error output.
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file"}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
