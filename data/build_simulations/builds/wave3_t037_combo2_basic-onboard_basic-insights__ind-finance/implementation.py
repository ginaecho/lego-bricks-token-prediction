"""Synthetic financial onboarding -> insights reference pipeline (Python stdlib).

Run: python -B implementation.py example_input.json
Ledger formats: json (list), csv (header plus rows), or iso20022 (the documented
fixture subset, not a complete ISO 20022 implementation). All paths converge on
one normalized envelope and validator. Rules are demonstrations, not compliance
certification, credit decisions, or financial advice.
"""

import copy
import csv
import datetime
import io
import json
import re
import sys
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


FLAGS = {"aml_review", "sanctions_match"}
PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
TX_FIELDS = {"id", "customer_id", "amount", "currency", "booking_date",
             "description", "screening_flags", "card_number"}
THEMES = (
    ("loans", ("loan", "repayment", "application")),
    ("verification", ("kyc", "verify", "verification", "identity", "document")),
    ("payments", ("payment", "transfer", "card", "transaction")),
    ("fees", ("fee", "charge", "cost")),
    ("support", ("support", "help", "response")),
)
ACTIONS = {
    "verification": "Clarify document requirements and route screening holds to a specialist.",
    "loans": "Explain application status and required documents; do not promise approval.",
    "payments": "Review payment friction with operations using masked evidence.",
    "fees": "Review fee explanations and highlight charges before confirmation.",
    "support": "Review response routing and publish the next support step.",
    "other": "Manually review feedback and assign an owner.",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required fields")
    require(value.keys() <= set(required) | set(optional), "Unexpected fields")


def text(value, label, max_length=2000):
    require(isinstance(value, str) and 0 < len(value.strip()) <= max_length,
            "Invalid " + label)
    return value.strip()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value),
            "Invalid identifier")
    return value


def masked(value):
    return PAN.sub(lambda m: "****" + re.sub(r"\D", "", m.group())[-4:], value)


def flags(value):
    require(isinstance(value, list) and all(isinstance(x, str) for x in value),
            "Screening flags must be a list")
    require(len(set(value)) == len(value) and set(value) <= FLAGS,
            "Unknown or duplicate screening flags")
    return sorted(value)


def money(value, positive=False):
    require(isinstance(value, str) and re.fullmatch(r"-?\d{1,12}(?:\.\d{1,2})?", value),
            "Amounts must be bounded decimal strings with at most two decimal places")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValidationError("Invalid amount") from exc
    require(number > 0 if positive else number != 0, "Invalid amount sign or zero")
    return format(number.quantize(Decimal("0.01")), "f")


def unique(records, label):
    require(isinstance(records, list), label + " must be a list")
    seen = set()
    for record in records:
        require(isinstance(record, dict), label + " entry must be an object")
        rid = identifier(record.get("id"))
        require(rid not in seen, "Duplicate " + label + " identifier")
        seen.add(rid)


def ledger_rows(ledger):
    keys(ledger, {"format", "data"})
    kind, data = ledger["format"], ledger["data"]
    if kind == "json":
        require(isinstance(data, list), "JSON ledger must be a list")
        return data
    if kind == "csv":
        require(isinstance(data, str), "CSV ledger must be text")
        reader = csv.DictReader(io.StringIO(data), strict=True)
        headers = reader.fieldnames
        require(headers is not None and len(headers) == len(set(headers))
                and set(headers) == TX_FIELDS, "Invalid CSV headers")
        rows = []
        try:
            for row in reader:
                require(None not in row and all(v is not None for v in row.values()),
                        "Malformed CSV row")
                row["screening_flags"] = row["screening_flags"].split("|") if row["screening_flags"] else []
                rows.append(row)
        except csv.Error as exc:
            raise ValidationError("Malformed CSV ledger") from exc
        return rows
    require(kind == "iso20022", "Unsupported ledger format")
    keys(data, {"Document"})
    keys(data["Document"], {"CstmrCdtTrfInitn"})
    initiation = data["Document"]["CstmrCdtTrfInitn"]
    keys(initiation, {"PmtInf"})
    require(isinstance(initiation["PmtInf"], list), "PmtInf must be a list")
    rows = []
    for group in initiation["PmtInf"]:
        keys(group, {"CdtTrfTxInf"})
        require(isinstance(group["CdtTrfTxInf"], list), "CdtTrfTxInf must be a list")
        for item in group["CdtTrfTxInf"]:
            keys(item, {"PmtId", "Dbtr", "Amt", "ReqdExctnDt", "RmtInf",
                        "ScreeningFlags", "CardNumber"})
            keys(item["PmtId"], {"EndToEndId"})
            keys(item["Dbtr"], {"Id"})
            keys(item["Amt"], {"InstdAmt"})
            keys(item["Amt"]["InstdAmt"], {"value", "Ccy"})
            keys(item["RmtInf"], {"Ustrd"})
            rows.append({
                "id": item["PmtId"]["EndToEndId"], "customer_id": item["Dbtr"]["Id"],
                "amount": money(item["Amt"]["InstdAmt"]["value"], positive=True),
                "currency": item["Amt"]["InstdAmt"]["Ccy"],
                "booking_date": item["ReqdExctnDt"], "description": item["RmtInf"]["Ustrd"],
                "screening_flags": item["ScreeningFlags"], "card_number": item["CardNumber"],
            })
    return rows


def normalize(raw):
    keys(raw, {"schema_version", "synthetic", "customers", "ledger", "loans", "feedback"})
    require(type(raw["schema_version"]) is int and raw["schema_version"] == 1,
            "Unsupported schema_version")
    require(raw["synthetic"] is True, "Only clearly labeled synthetic data is accepted")
    result = {
        "schema_version": 1, "synthetic": True, "stage": "validated",
        "customers": copy.deepcopy(raw["customers"]),
        "transactions": copy.deepcopy(ledger_rows(raw["ledger"])),
        "loans": copy.deepcopy(raw["loans"]), "feedback": copy.deepcopy(raw["feedback"]),
        "onboarding": [], "insights": [],
    }
    for name in ("customers", "transactions", "loans", "feedback"):
        unique(result[name], name)
    for customer in result["customers"]:
        keys(customer, {"id", "name", "goal", "kyc_status", "screening_flags", "account"})
        customer["name"] = masked(text(customer["name"], "customer name", 100))
        customer["screening_flags"] = flags(customer["screening_flags"])
    for tx in result["transactions"]:
        keys(tx, TX_FIELDS)
        tx["amount"] = money(tx["amount"])
        tx["description"] = masked(text(tx["description"], "transaction description"))
        tx["screening_flags"] = flags(tx["screening_flags"])
        card = tx.pop("card_number")
        require(isinstance(card, str), "Card number must be text")
        require(card == "" or re.fullmatch(r"(?:\d[ -]?){12,18}\d", card),
                "Card must contain 13 to 19 digits or be empty")
        tx["masked_card"] = "****" + re.sub(r"\D", "", card)[-4:] if card else ""
    for loan in result["loans"]:
        keys(loan, {"id", "customer_id", "amount", "currency", "status", "documents_complete"})
        loan["amount"] = money(loan["amount"], positive=True)
    for item in result["feedback"]:
        keys(item, {"id", "customer_id", "entity_type", "entity_id", "text", "sentiment"})
        item["text"] = masked(text(item["text"], "feedback text"))
    validate(result, "validated")
    return result


def score(value, formula, components):
    return {"value": value, "range": [0, 100], "model_version": "deterministic-rules-v1",
            "formula": formula, "components": components,
            "limitations": "Synthetic operational prioritization only; not creditworthiness or AML probability."}


def onboarding_records(envelope):
    records = []
    for customer in envelope["customers"]:
        cid = customer["id"]
        transactions = [t for t in envelope["transactions"] if t["customer_id"] == cid]
        loans = [loan for loan in envelope["loans"] if loan["customer_id"] == cid]
        flagged = bool(customer["screening_flags"] or any(t["screening_flags"] for t in transactions))
        verified = customer["kyc_status"] == "verified"
        components = {"kyc_verified": 50 if verified else 0,
                      "screening_clear": 30 if not flagged else 0,
                      "has_transaction": 20 if transactions else 0}
        value = sum(components.values()) if verified and not flagged else 0
        missing = [loan["id"] for loan in loans if loan["status"] == "draft" and not loan["documents_complete"]]
        if flagged or customer["kyc_status"] == "rejected":
            code, status = "specialist_review", "blocked"
            instruction = "Contact the verification specialist; do not initiate new financial activity."
        elif not verified:
            code, status = "complete_kyc", "action_required"
            instruction = "Submit the requested identity documents using the secure verification channel."
        elif missing:
            code, status = "complete_loan_documents", "action_required"
            instruction = "Complete documents for loan applications: " + ", ".join(missing) + "."
        elif customer["goal"] == "borrow" and not loans:
            code, status = "start_loan_application", "ready"
            instruction = "Start a loan application and review the required documents; approval is not implied."
        elif customer["goal"] == "borrow" and any(loan["status"] == "submitted" for loan in loans):
            code, status = "track_loan_application", "ready"
            instruction = "Track your submitted application and await a human lending review."
        elif customer["goal"] == "borrow" and any(loan["status"] == "draft" for loan in loans):
            code, status = "review_loan_application", "ready"
            instruction = "Review your completed application before submitting it for human assessment."
        elif not transactions:
            code, status = "review_first_payment", "ready"
            instruction = "Review account details and payment limits before making your first payment."
        else:
            code, status = "review_activity", "ready"
            instruction = "Review your recent transaction history and share any feedback."
        records.append({
            "customer_id": cid, "status": status,
            "next_step": {"code": code, "message": customer["name"] + ": " + instruction},
            "readiness_score": score(value, "0 if KYC is not verified or screening is flagged; otherwise sum(components)", components),
            "evidence": {"kyc_status": customer["kyc_status"], "screening_flagged": flagged,
                         "transaction_ids": [t["id"] for t in transactions],
                         "loan_ids": [loan["id"] for loan in loans]},
        })
    return records


def insight_records(envelope):
    onboarding = {item["customer_id"]: item for item in envelope["onboarding"]}
    grouped = {}
    for item in envelope["feedback"]:
        words = set(re.findall(r"[a-z]+", item["text"].lower()))
        theme = next((name for name, terms in THEMES if words.intersection(terms)), "other")
        grouped.setdefault(theme, []).append(item)
    records = []
    for theme, feedback in grouped.items():
        customers = sorted({f["customer_id"] for f in feedback})
        blocked = sum(onboarding[c]["status"] == "blocked" for c in customers)
        negatives = sum(f["sentiment"] == "negative" for f in feedback)
        parts = {"negative_feedback": 15 * negatives, "blocked_customers": 20 * blocked,
                 "feedback_volume": 5 * len(feedback)}
        records.append({
            "theme": theme, "feedback_count": len(feedback),
            "customer_ids": customers, "evidence_ids": [f["id"] for f in feedback],
            "priority_score": score(min(100, sum(parts.values())), "min(100, sum(components))", parts),
            "action": ACTIONS[theme],
            "onboarding_context": [
                {"customer_id": c, "status": onboarding[c]["status"],
                 "next_step_code": onboarding[c]["next_step"]["code"]}
                for c in customers],
        })
    return sorted(records, key=lambda r: (-r["priority_score"]["value"], r["theme"]))


def validate(envelope, expected_stage):
    """Shared schema, referential integrity, privacy, and stage-boundary validator."""
    keys(envelope, {"schema_version", "synthetic", "stage", "customers", "transactions",
                    "loans", "feedback", "onboarding", "insights"})
    require(envelope["schema_version"] == 1 and type(envelope["schema_version"]) is int
            and envelope["synthetic"] is True, "Invalid envelope metadata")
    require(expected_stage in {"validated", "onboarded", "insights"}
            and envelope["stage"] == expected_stage, "Invalid pipeline stage")
    for name in ("customers", "transactions", "loans", "feedback"):
        unique(envelope[name], name)
    customers = {c["id"]: c for c in envelope["customers"]}
    for customer in customers.values():
        keys(customer, {"id", "name", "goal", "kyc_status", "screening_flags", "account"})
        text(customer["name"], "customer name", 100)
        require(customer["goal"] in ("payments", "borrow", "save"), "Invalid customer goal")
        require(customer["kyc_status"] in ("verified", "pending", "rejected"), "Invalid KYC status")
        flags(customer["screening_flags"])
        keys(customer["account"], {"number", "iban"})
        require(isinstance(customer["account"]["number"], str)
                and re.fullmatch(r"SYNTH-[A-Z0-9]{1,20}", customer["account"]["number"]),
                "Account must be visibly synthetic")
        require(isinstance(customer["account"]["iban"], str)
                and re.fullmatch(r"ZZ00SYNTH[A-Z0-9]{1,20}", customer["account"]["iban"]),
                "IBAN must use synthetic ZZ00SYNTH prefix")
    for tx in envelope["transactions"]:
        keys(tx, (TX_FIELDS - {"card_number"}) | {"masked_card"})
        require(isinstance(tx["customer_id"], str) and tx["customer_id"] in customers,
                "Unknown transaction customer")
        require(money(tx["amount"]) == tx["amount"], "Noncanonical amount")
        require(isinstance(tx["currency"], str) and re.fullmatch(r"[A-Z]{3}", tx["currency"]),
                "Invalid currency")
        require(isinstance(tx["booking_date"], str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tx["booking_date"]), "Invalid date")
        try:
            datetime.date.fromisoformat(tx["booking_date"])
        except ValueError as exc:
            raise ValidationError("Invalid calendar date") from exc
        text(tx["description"], "description")
        flags(tx["screening_flags"])
        require(isinstance(tx["masked_card"], str)
                and (tx["masked_card"] == "" or re.fullmatch(r"\*{4}\d{4}", tx["masked_card"])),
                "Unsafe card number")
    for loan in envelope["loans"]:
        keys(loan, {"id", "customer_id", "amount", "currency", "status", "documents_complete"})
        require(isinstance(loan["customer_id"], str) and loan["customer_id"] in customers,
                "Unknown loan customer")
        require(money(loan["amount"], positive=True) == loan["amount"], "Noncanonical loan amount")
        require(isinstance(loan["currency"], str) and re.fullmatch(r"[A-Z]{3}", loan["currency"]),
                "Invalid loan currency")
        require(loan["status"] in ("draft", "submitted", "closed"), "Invalid loan status")
        require(type(loan["documents_complete"]) is bool, "documents_complete must be boolean")
        require(loan["status"] != "submitted" or loan["documents_complete"],
                "Submitted loan must have complete documents")
    entities = {"customer": customers,
                "transaction": {t["id"]: t for t in envelope["transactions"]},
                "loan": {loan["id"]: loan for loan in envelope["loans"]}}
    for item in envelope["feedback"]:
        keys(item, {"id", "customer_id", "entity_type", "entity_id", "text", "sentiment"})
        identifier(item["customer_id"])
        identifier(item["entity_id"])
        require(item["customer_id"] in customers, "Unknown feedback customer")
        require(isinstance(item["entity_type"], str) and item["entity_type"] in entities,
                "Invalid feedback entity type")
        target = entities[item["entity_type"]].get(item["entity_id"])
        require(target is not None, "Unknown feedback entity")
        owner = target["id"] if item["entity_type"] == "customer" else target["customer_id"]
        require(owner == item["customer_id"], "Feedback entity belongs to another customer")
        text(item["text"], "feedback")
        require(item["sentiment"] in ("negative", "neutral", "positive"), "Invalid sentiment")
    require(not PAN.search(json.dumps(envelope)), "Unmasked card-like digit sequence")
    require(isinstance(envelope["onboarding"], list) and isinstance(envelope["insights"], list),
            "Stage results must be lists")
    if expected_stage == "validated":
        require(envelope["onboarding"] == [] and envelope["insights"] == [], "Premature stage data")
    else:
        require(envelope["onboarding"] == onboarding_records(envelope), "Invalid onboarding handoff")
        require(envelope["insights"] == (insight_records(envelope) if expected_stage == "insights" else []),
                "Invalid insights results or explanations")
    return envelope


def onboard(envelope):
    validate(envelope, "validated")
    result = copy.deepcopy(envelope)
    result["onboarding"] = onboarding_records(result)
    result["stage"] = "onboarded"
    return validate(result, "onboarded")


def insights(envelope):
    validate(envelope, "onboarded")
    result = copy.deepcopy(envelope)
    result["insights"] = insight_records(result)
    result["stage"] = "insights"
    return validate(result, "insights")


def run_pipeline(raw):
    return {"status": "ok", "data": insights(onboard(normalize(raw))),
            "notice": "Clearly synthetic demonstration; no compliance certification or lending decisions."}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], "r", encoding="utf-8") as source:
            raw = json.load(source, object_pairs_hook=reject_duplicate_keys,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValidationError("Nonfinite JSON number")))
        output, exit_code = run_pipeline(raw), 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, KeyError,
            OverflowError, RecursionError, csv.Error):
        # Do not reflect filenames, free text, PANs, or other input in errors.
        output, exit_code = {"status": "error", "message": "Invalid input or unreadable file."}, 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
