"""Synthetic financial research reference. Python standard library; no network.

CLI: python -B implementation.py example_input.json
The shared schema is described in build_manifest.json. All input channels pass
through validate(). Findings are verbatim sentence spans in supplied sources.
Screening and relevance scores are transparent demonstration rules, not models
or regulatory compliance determinations.
"""

import csv
import io
import json
import math
import re
import sys
from datetime import date


class ValidationError(ValueError):
    pass


PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
WORDS = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "and", "are", "for", "in", "is", "of", "on", "the", "to", "with"}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, label):
    require(isinstance(value, dict), label + " must be an object")
    return value


def text(value, label, max_length=10000):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    require(len(value) <= max_length, label + " is too long")
    require(PAN.search(value) is None, label + " contains unmasked card-like digits")
    return value


def number(value, label, positive=False):
    require(type(value) in (int, float), label + " must be a number")
    require(value > 0 if positive else value >= 0, label + " is out of range")
    require(value <= 10**12, label + " is too large")
    require(math.isfinite(value), label + " must be finite")
    return value


def array(value, label, maximum=1000):
    require(isinstance(value, list), label + " must be an array")
    require(len(value) <= maximum, label + " has too many entries")
    return value


def unique(rows, label):
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)), label + " IDs must be unique")


def tokens(value):
    return set(WORDS.findall(value.lower())) - STOPWORDS


def score(contributions):
    return {
        "value": min(100, sum(item["points"] for item in contributions)),
        "range": [0, 100],
        "aggregation": "sum of disclosed contributions, capped at 100",
        "explanation": contributions or [{"rule": "no demonstration flags", "points": 0}],
    }


def validate(raw):
    """Normalize JSON/CSV/ISO-style records into one bounded schema."""
    obj(raw, "input")
    require(raw.get("schema_version") == "1.0", "schema_version must be 1.0")
    require(raw.get("synthetic") is True, "synthetic must be true")
    query = text(raw.get("query"), "query", 1000)
    limit = raw.get("max_findings", 5)
    require(type(limit) is int and 1 <= limit <= 20, "max_findings must be 1..20")
    sources = []
    for source in array(raw.get("sources"), "sources", 100):
        obj(source, "source")
        sources.append({
            "id": text(source.get("id"), "source.id", 100),
            "title": text(source.get("title"), "source.title", 200),
            "text": text(source.get("text"), "source.text"),
        })
    unique(sources, "source")
    customers = []
    for customer in array(raw.get("customers"), "customers"):
        obj(customer, "customer")
        status = customer.get("kyc_status")
        require(status in ("verified", "pending", "rejected"), "invalid kyc_status")
        flags = array(customer.get("screening_flags", []), "screening_flags", 2)
        require(all(flag in ("pep", "sanctions_match") for flag in flags),
                "unknown screening flag")
        require(len(set(flags)) == len(flags), "duplicate screening flag")
        card = customer.get("card_number")
        require(card is None or (isinstance(card, str) and
                re.fullmatch(r"[0-9]{13,19}", card) is not None),
                "card_number must be 13..19 digits")
        customers.append({
            "id": text(customer.get("id"), "customer.id", 100),
            "name": text(customer.get("name"), "customer.name", 200),
            "iban": text(customer.get("iban"), "customer.iban", 100),
            "kyc_status": status,
            "screening_flags": list(flags),
            "card_number": None if card is None else "*" * (len(card) - 4) + card[-4:],
        })
        require(customers[-1]["iban"].startswith("SYNTH-"), "IBAN must use SYNTH- prefix")
    unique(customers, "customer")
    customer_ids = {customer["id"] for customer in customers}
    ledger = raw.get("transactions", [])
    if isinstance(ledger, dict):
        require(ledger.get("format") == "csv", "ledger format must be csv")
        content = ledger.get("data")
        require(isinstance(content, str) and len(content) <= 200000, "invalid CSV data")
        reader = csv.DictReader(io.StringIO(content), strict=True)
        ledger = []
        try:
            require(reader.fieldnames == ["id", "customer_id", "amount", "currency", "date"],
                    "CSV header must be id,customer_id,amount,currency,date")
            for row in reader:
                require(None not in row and all(v is not None for v in row.values()),
                        "CSV row has incorrect column count")
                try:
                    row["amount"] = float(row["amount"])
                except (TypeError, ValueError):
                    raise ValidationError("CSV amount must be numeric") from None
                ledger.append(row)
                require(len(ledger) <= 1000, "too many CSV transactions")
        except csv.Error:
            raise ValidationError("malformed CSV") from None
    entries = list(array(ledger, "transactions"))
    for payment in array(raw.get("payments", []), "payments"):
        obj(payment, "payment")
        info = obj(payment.get("CdtTrfTxInf"), "CdtTrfTxInf")
        payment_id = obj(info.get("PmtId"), "PmtId")
        amount = obj(obj(info.get("Amt"), "Amt").get("InstdAmt"), "InstdAmt")
        debtor = obj(info.get("Dbtr"), "Dbtr")
        entries.append({
            "id": payment_id.get("EndToEndId"),
            "customer_id": debtor.get("CustomerId"),
            "amount": amount.get("value"),
            "currency": amount.get("Ccy"),
            "date": info.get("ReqdExctnDt"),
        })
    require(len(entries) <= 1000, "too many combined transactions")
    transactions = []
    for entry in entries:
        obj(entry, "transaction")
        customer_id = text(entry.get("customer_id"), "transaction.customer_id", 100)
        require(customer_id in customer_ids, "transaction references unknown customer")
        currency = text(entry.get("currency"), "currency", 3)
        require(re.fullmatch(r"[A-Z]{3}", currency) is not None, "invalid currency")
        when = text(entry.get("date"), "date", 10)
        try:
            require(date.fromisoformat(when).isoformat() == when, "date must be YYYY-MM-DD")
        except ValueError:
            raise ValidationError("date must be a valid YYYY-MM-DD") from None
        transactions.append({
            "id": text(entry.get("id"), "transaction.id", 100),
            "customer_id": customer_id,
            "amount": number(entry.get("amount"), "amount", positive=True),
            "currency": currency,
            "date": when,
        })
    unique(transactions, "transaction")
    loans = []
    for loan in array(raw.get("loan_applications", []), "loan_applications"):
        obj(loan, "loan")
        customer_id = text(loan.get("customer_id"), "loan.customer_id", 100)
        require(customer_id in customer_ids, "loan references unknown customer")
        currency = text(loan.get("currency"), "loan.currency", 3)
        require(re.fullmatch(r"[A-Z]{3}", currency) is not None, "invalid loan currency")
        loans.append({
            "id": text(loan.get("id"), "loan.id", 100),
            "customer_id": customer_id,
            "amount": number(loan.get("amount"), "loan.amount", positive=True),
            "annual_income": number(loan.get("annual_income"), "annual_income"),
            "currency": currency,
        })
    unique(loans, "loan")
    return {"query": query, "max_findings": limit, "sources": sources,
            "customers": customers, "transactions": transactions, "loan_applications": loans}


def research(data):
    terms = tokens(data["query"])
    candidates = []
    for source in data["sources"]:
        for match in re.finditer(r"[^.!?\n]+[.!?]*", source["text"]):
            passage = match.group()
            start = match.start() + len(passage) - len(passage.lstrip())
            quote = passage.strip()
            matched = sorted(terms & tokens(quote))
            if not matched:
                continue
            end = start + len(quote)
            candidates.append({
                "finding": quote,
                "citation": {"source_id": source["id"], "title": source["title"],
                             "start": start, "end": end, "quote": quote},
                "relevance_score": {
                    "value": len(matched),
                    "explanation": {"rule": "count of distinct non-stopword query terms in passage",
                                    "matched_terms": matched},
                },
            })
    candidates.sort(key=lambda item: (-item["relevance_score"]["value"],
                                     item["citation"]["source_id"], item["citation"]["start"]))
    return candidates[:data["max_findings"]]


def run(raw):
    data = validate(raw)
    findings = research(data)
    customers = {}
    assessments = []
    for customer in data["customers"]:
        rules = []
        if customer["kyc_status"] != "verified":
            rules.append({"rule": "KYC not verified; review required", "points": 40})
        for flag in customer["screening_flags"]:
            rules.append({"rule": flag + "; review required",
                          "points": 80 if flag == "sanctions_match" else 20})
        customers[customer["id"]] = rules
        assessments.append({"entity_type": "customer", "entity_id": customer["id"],
                            "flags": [r["rule"] for r in rules], "screening_score": score(rules)})
    for transaction in data["transactions"]:
        rules = list(customers[transaction["customer_id"]])
        if transaction["currency"] == "USD" and transaction["amount"] >= 10000:
            rules.append({"rule": "synthetic AML threshold: USD amount >= 10000", "points": 30})
        assessments.append({"entity_type": "transaction", "entity_id": transaction["id"],
                            "flags": [r["rule"] for r in rules], "screening_score": score(rules)})
    for loan in data["loan_applications"]:
        rules = list(customers[loan["customer_id"]])
        if loan["annual_income"] == 0:
            rules.append({"rule": "zero declared annual income", "points": 50})
        elif loan["amount"] > loan["annual_income"] * 3:
            rules.append({"rule": "requested amount > 3 times annual income in same currency",
                          "points": 30})
        assessments.append({"entity_type": "loan_application", "entity_id": loan["id"],
                            "flags": [r["rule"] for r in rules], "screening_score": score(rules)})
    return {
        "status": "ok", "schema_version": "1.0", "synthetic": True,
        "disclaimer": "Synthetic demonstration only; not compliance certification, financial advice, "
                      "credit approval, or real AML/KYC screening. Card masking is a PCI-inspired "
                      "demonstration, not PCI DSS certification.",
        "research": {"query": data["query"], "findings": findings,
                     "no_matches": not findings, "sources": data["sources"]},
        "entities": {key: data[key] for key in ("customers", "transactions", "loan_applications")},
        "assessments": assessments,
    }


def reject_constant(value):
    raise ValidationError("non-finite JSON numbers are not accepted")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            payload = handle.read(1000001)
        require(len(payload) <= 1000000, "input exceeds 1000000 characters")
        raw = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run(raw)
        code = 0
    except (OSError, UnicodeError):
        result, code = {"status": "error", "message": "input file cannot be read"}, 2
    except (ValueError, RecursionError, OverflowError):
        # Never echo rejected values or paths, which could contain card data.
        result, code = {"status": "error", "message": "input validation failed"}, 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
