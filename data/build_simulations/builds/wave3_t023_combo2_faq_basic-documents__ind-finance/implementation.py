"""Synthetic financial FAQ -> document pipeline; Python standard library only."""
import copy
import csv
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 10000,
            label + " must be nonempty text")
    return value.strip()


def mask(value):
    return re.sub(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
                  lambda m: "****" + re.sub(r"\D", "", m.group())[-4:], value)


def money(value, label):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            label + " must be numeric")
    try:
        number = Decimal(str(value))
        require(number.is_finite() and 0 < number <= Decimal("1000000000000"),
                label + " must be finite and positive")
        require(number == number.quantize(Decimal("0.01")),
                label + " must have at most two decimals")
        return format(number, ".2f")
    except InvalidOperation as exc:
        raise ValidationError(label + " is invalid") from exc


FIELDS = ["transaction_id", "date", "amount", "currency", "account", "card_number"]
NS = "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"


def parse_ledger(source):
    require(isinstance(source, dict), "transaction_source must be an object")
    fmt, data = source.get("format"), source.get("data")
    if fmt == "json":
        if isinstance(data, str):
            data = json.loads(data)
        rows = data
    elif fmt == "csv":
        require(isinstance(data, str), "CSV data must be text")
        reader = csv.DictReader(io.StringIO(data))
        require(reader.fieldnames == FIELDS, "CSV header must match shared transaction fields")
        rows = list(reader)
        require(all(None not in row and all(v is not None for v in row.values())
                    for row in rows), "Malformed CSV row")
    elif fmt == "iso20022":
        require(isinstance(data, str), "XML data must be text")
        require("<!DOCTYPE" not in data.upper() and "<!ENTITY" not in data.upper(),
                "XML declarations/entities are not supported")
        root = ET.fromstring(data)
        require(root.tag == "{" + NS + "}Document", "Unsupported payment namespace/root")
        ns = {"p": NS}
        rows = []
        payments = root.findall("p:CstmrCdtTrfInitn/p:PmtInf", ns)
        require(bool(payments), "Missing payment information")
        for payment in payments:
            for tx in payment.findall("p:CdtTrfTxInf", ns):
                amount = tx.find("p:Amt/p:InstdAmt", ns)
                require(amount is not None, "Missing instructed amount")
                rows.append({
                    "transaction_id": tx.findtext("p:PmtId/p:EndToEndId", namespaces=ns),
                    "date": payment.findtext("p:ReqdExctnDt/p:Dt", namespaces=ns),
                    "amount": amount.text,
                    "currency": amount.get("Ccy"),
                    "account": tx.findtext("p:CdtrAcct/p:Id/p:IBAN", namespaces=ns),
                    "card_number": "",
                })
    else:
        raise ValidationError("Unsupported ledger format")
    require(isinstance(rows, list) and len(rows) <= 1000, "Ledger must contain at most 1000 rows")
    return rows


def validate(envelope, phase):
    """One validation boundary used before and after each integrated stage."""
    require(isinstance(envelope, dict), "Input must be an object")
    require(envelope.get("schema_version") == "1.0", "Unsupported schema_version")
    require(envelope.get("synthetic") is True, "Only explicitly synthetic data is supported")
    customer = envelope.get("customer")
    require(isinstance(customer, dict), "customer must be an object")
    require(set(customer) == {"customer_id", "name", "account", "kyc_status", "aml_flags"},
            "Unexpected customer fields")
    for key in ("customer_id", "name", "account"):
        text(customer.get(key), "customer." + key)
    require(customer.get("kyc_status") in ("verified", "pending", "rejected"), "Invalid KYC status")
    flags = customer.get("aml_flags")
    require(isinstance(flags, list) and all(
        isinstance(f, str) and f in ("sanctions_match", "unusual_activity", "pep")
        for f in flags) and len(flags) == len(set(flags)), "Invalid AML flags")
    loan = envelope.get("loan_application")
    require(isinstance(loan, dict), "loan_application must be an object")
    require(set(loan) == {"loan_id", "amount", "annual_income", "currency"},
            "Unexpected loan fields")
    text(loan.get("loan_id"), "loan_id")
    money(loan.get("amount"), "loan amount")
    money(loan.get("annual_income"), "annual income")
    require(isinstance(loan.get("currency"), str) and
            re.fullmatch("[A-Z]{3}", loan["currency"]), "Invalid loan currency")
    text(envelope.get("question"), "question")
    kb = envelope.get("knowledge_base")
    require(isinstance(kb, list) and len(kb) <= 100, "knowledge_base must be a bounded list")
    ids = set()
    for entry in kb:
        require(isinstance(entry, dict), "Knowledge entry must be an object")
        require(set(entry) == {"id", "question", "answer"}, "Unexpected knowledge fields")
        for key in ("id", "question", "answer"):
            text(entry.get(key), "knowledge." + key)
        require(entry["id"] not in ids, "Duplicate knowledge id")
        ids.add(entry["id"])
    if phase == "input":
        return
    rows = envelope.get("transactions")
    require(isinstance(rows, list), "Missing normalized transactions")
    seen = set()
    for row in rows:
        require(isinstance(row, dict), "Transaction must be an object")
        tid = text(row.get("transaction_id"), "transaction_id")
        require(tid not in seen, "Duplicate transaction_id")
        seen.add(tid)
        stamp = text(row.get("date"), "transaction date")
        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp), "Invalid transaction date")
        date.fromisoformat(stamp)
        money(row.get("amount"), "transaction amount")
        require(isinstance(row.get("currency"), str) and
                re.fullmatch("[A-Z]{3}", row["currency"]), "Invalid transaction currency")
        text(row.get("account"), "transaction account")
        require(isinstance(row.get("card_number"), str) and
                (row["card_number"] == "" or re.fullmatch(r"\*{4}\d{4}", row["card_number"])),
                "Card must be masked")
    if phase == "normalized":
        return
    faq = envelope.get("faq")
    require(isinstance(faq, dict), "Missing FAQ handoff")
    require(faq.get("status") in ("answered", "abstained"), "Invalid FAQ status")
    if faq["status"] == "answered":
        matches = [e for e in kb if [e["id"]] == faq.get("evidence_ids")]
        require(len(matches) == 1 and faq.get("answer") == matches[0]["answer"],
                "Answer is not grounded in cited evidence")
    else:
        require(faq.get("answer") is None and faq.get("evidence_ids") == [] and
                faq.get("reason") == "No matching knowledge-base evidence",
                "Invalid abstention")
    if phase == "documents":
        docs = envelope.get("documents")
        require(isinstance(docs, dict) and docs.get("support_response") == faq,
                "Document FAQ propagation mismatch")
        require(docs.get("rows") == rows, "Document rows mismatch")
        for score in [docs.get("loan_risk")] + docs.get("transaction_risks", []):
            require(isinstance(score, dict) and isinstance(score.get("factors"), list)
                    and bool(score["factors"]), "Score must have explanatory factors")
            require(all(isinstance(f, dict) and isinstance(f.get("points"), int)
                        and isinstance(f.get("reason"), str) and f["reason"]
                        for f in score["factors"]), "Invalid score explanation")
            require(score.get("score") == min(100, sum(f["points"] for f in score["factors"])),
                    "Score does not match explanation")
        require(len(docs.get("transaction_risks", [])) == len(rows), "Missing transaction scores")


def normalize(raw):
    validate(raw, "input")
    out = {k: copy.deepcopy(raw[k]) for k in (
        "schema_version", "synthetic", "customer", "loan_application", "question", "knowledge_base")}
    rows = parse_ledger(raw.get("transaction_source"))
    normalized = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == set(FIELDS),
                "Transaction fields must match shared schema")
        item = dict(row)
        card = item["card_number"]
        require(isinstance(card, str), "Card must be text")
        digits = re.sub(r"[ -]", "", card)
        require(card == "" or re.fullmatch(r"\d{13,19}", digits), "Invalid card number")
        item["card_number"] = "****" + digits[-4:] if card else ""
        item["amount"] = money(item["amount"], "transaction amount")
        normalized.append(item)
    out["transactions"] = normalized
    out["loan_application"]["amount"] = money(out["loan_application"]["amount"], "loan amount")
    out["loan_application"]["annual_income"] = money(out["loan_application"]["annual_income"], "income")

    def scrub(obj):
        if isinstance(obj, str):
            return mask(obj)
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        if isinstance(obj, dict):
            return {k: v if k in ("amount", "annual_income") else scrub(v)
                    for k, v in obj.items()}
        return obj

    out = scrub(out)
    validate(out, "normalized")
    return out


STOP = {"a", "an", "the", "is", "are", "i", "my", "how", "do", "can", "to", "for", "what", "and"}


def tokens(value):
    return set(re.findall("[a-z0-9]+", value.lower())) - STOP


def answer_stage(envelope, answerer=None):
    validate(envelope, "normalized")
    out = copy.deepcopy(envelope)
    ranked = sorted(envelope["knowledge_base"],
                    key=lambda e: (-len(tokens(e["question"]) & tokens(envelope["question"])), e["id"]))
    selected = ranked[0] if ranked and tokens(ranked[0]["question"]) & tokens(envelope["question"]) else None
    if selected:
        answer = selected["answer"]
        if answerer is not None:
            answer = answerer(envelope["question"], copy.deepcopy(selected))
        out["faq"] = {"status": "answered", "answer": answer, "evidence_ids": [selected["id"]]}
    else:
        out["faq"] = {"status": "abstained", "answer": None, "evidence_ids": [],
                      "reason": "No matching knowledge-base evidence"}
    validate(out, "faq")
    return out


def risk(factors):
    return {"score": min(100, sum(f["points"] for f in factors)),
            "model": "synthetic-rule-risk-v1", "factors": factors,
            "explanation": "Sum factor points, capped at 100; illustrative only, not a credit decision."}


def csv_cell(value):
    # Spreadsheet formula injection protection is independent of PAN masking.
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def document_stage(envelope):
    validate(envelope, "faq")
    out = copy.deepcopy(envelope)
    customer, loan, rows = out["customer"], out["loan_application"], out["transactions"]
    factors = [{"reason": "Baseline synthetic risk", "points": 0}]
    if customer["kyc_status"] != "verified":
        factors.append({"reason": "KYC is " + customer["kyc_status"], "points": 40})
    for flag in customer["aml_flags"]:
        factors.append({"reason": "AML screening flag: " + flag, "points": 30})
    loan_factors = copy.deepcopy(factors)
    if Decimal(loan["amount"]) > Decimal(loan["annual_income"]) * 3:
        loan_factors.append({"reason": "Requested principal exceeds 3x annual income", "points": 25})
    scores, totals = [], {}
    for row in rows:
        tx_factors = copy.deepcopy(factors)
        if Decimal(row["amount"]) >= 10000:
            tx_factors.append({"reason": "Transaction amount >= 10000 in stated currency", "points": 20})
        scores.append({"transaction_id": row["transaction_id"], **risk(tx_factors)})
        currency = row["currency"]
        totals[currency] = totals.get(currency, Decimal(0)) + Decimal(row["amount"])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: csv_cell(row[k]) for k in FIELDS} for row in rows)
    out["documents"] = {
        "label": "SYNTHETIC FINANCIAL OPERATIONS DOCUMENT",
        "support_response": copy.deepcopy(out["faq"]),
        "customer_id": customer["customer_id"], "loan_id": loan["loan_id"],
        "screening": {"kyc_status": customer["kyc_status"], "aml_flags": customer["aml_flags"],
                      "requires_manual_review": customer["kyc_status"] != "verified" or bool(customer["aml_flags"]),
                      "notice": "Demonstrative checks only; no compliance certification."},
        "rows": copy.deepcopy(rows), "ledger_csv": output.getvalue(),
        "totals_by_currency": {k: format(v, ".2f") for k, v in sorted(totals.items())},
        "loan_risk": risk(loan_factors), "transaction_risks": scores,
        "checks": {"transaction_count": len(rows), "card_numbers_masked": True,
                   "faq_grounded_or_abstained": True},
    }
    validate(out, "documents")
    return out


def run(payload, answerer=None):
    result = document_stage(answer_stage(normalize(payload), answerer))
    return {"status": "ok", **result}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        result = run(payload)
        code = 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError,
            ET.ParseError, csv.Error):
        # Do not echo invalid input, filenames, or exception text containing PANs.
        result, code = {"status": "error", "message": "Invalid input or unreadable input file"}, 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
