"""Offline, synthetic financial research reference CLI; Python standard library only."""
import csv
import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, field):
    require(isinstance(value, str) and bool(value.strip()), field + " must be nonempty text")
    return value


def number(value, field, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), field + " must be finite numeric")
    require(value > 0 if positive else value >= 0, field + " has invalid range")
    return value


def mask(value):
    """Mask possible PANs, including space/hyphen-separated forms, on every output path."""
    if isinstance(value, str):
        return re.sub(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
                      lambda m: "*" * (len(re.sub(r"\D", "", m[0])) - 4)
                      + re.sub(r"\D", "", m[0])[-4:], value)
    if isinstance(value, list):
        return [mask(v) for v in value]
    if isinstance(value, dict):
        return {mask(k): v if k in ("sha256", "source_sha256") else mask(v)
                for k, v in value.items()}
    return value


def url(value):
    text(value, "URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValidationError("Malformed URL") from None
    require(parsed.scheme == "https" and parsed.hostname is not None
            and parsed.username is None and parsed.password is None
            and port in (None, 443) and not parsed.fragment
            and not re.search(r"[\s\\]", value), "Only credential-free HTTPS URLs are accepted")
    return value


def timestamp(value):
    try:
        result = datetime.fromisoformat(text(value, "retrieved_at").replace("Z", "+00:00"))
        require(result.utcoffset() is not None, "retrieved_at requires a timezone")
    except ValueError:
        raise ValidationError("Invalid timezone-aware retrieved_at") from None


def parse_ledger(ledger):
    require(isinstance(ledger, dict), "ledger must be an object")
    fmt, data = ledger.get("format"), ledger.get("data")
    if fmt == "json":
        require(isinstance(data, list), "JSON ledger data must be an array")
        return data
    if fmt == "csv":
        text(data, "CSV ledger")
        reader = csv.DictReader(io.StringIO(data))
        expected = {"id", "customer_id", "amount", "currency", "card_number", "aml_flag"}
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(expected)
                and set(reader.fieldnames) == expected, "CSV columns do not match shared schema")
        rows = []
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
            require(row["aml_flag"] in ("true", "false"), "CSV aml_flag must be true or false")
            try:
                row["amount"] = float(row["amount"])
            except ValueError:
                raise ValidationError("Invalid CSV amount") from None
            row["aml_flag"] = row["aml_flag"] == "true"
            rows.append(row)
        return rows
    raise ValidationError("ledger format must be json or csv")


def validate(data):
    """One normalization/validation boundary for CLI and injected-callable users."""
    require(isinstance(data, dict), "Input must be an object")
    require(data.get("schema_version") == "1.0", "Unsupported schema_version")
    require(data.get("synthetic") is True, "Only explicitly synthetic fixtures are supported")
    customers = data.get("customers")
    loans = data.get("loan_applications")
    require(isinstance(customers, list) and customers, "customers must be a nonempty array")
    require(isinstance(loans, list), "loan_applications must be an array")
    customer_ids = set()
    for c in customers:
        require(isinstance(c, dict), "Customer must be an object")
        cid = text(c.get("id"), "customer id")
        require(cid not in customer_ids, "Duplicate customer id")
        customer_ids.add(cid)
        text(c.get("name"), "customer name")
        require(c.get("kyc_status") in ("verified", "pending", "failed"), "Invalid KYC status")
        require(type(c.get("aml_flag")) is bool, "Customer aml_flag must be boolean")
        require(isinstance(c.get("iban"), str) and re.fullmatch(r"ZZ\d{20}", c["iban"]),
                "Synthetic IBAN must use reserved ZZ plus 20 digits")
    transactions = parse_ledger(data.get("ledger"))
    seen = set()
    for t in transactions:
        require(isinstance(t, dict), "Transaction must be an object")
        tid = text(t.get("id"), "transaction id")
        require(tid not in seen, "Duplicate transaction id")
        seen.add(tid)
        require(isinstance(t.get("customer_id"), str) and t["customer_id"] in customer_ids,
                "Unknown transaction customer")
        number(t.get("amount"), "transaction amount", True)
        require(isinstance(t.get("currency"), str) and re.fullmatch(r"[A-Z]{3}", t["currency"]),
                "Currency must be three uppercase letters")
        require(type(t.get("aml_flag")) is bool, "Transaction aml_flag must be boolean")
        require(isinstance(t.get("card_number"), str)
                and re.fullmatch(r"\d{13,19}", t["card_number"]), "Invalid synthetic card number")
    seen = set()
    for loan in loans:
        require(isinstance(loan, dict), "Loan application must be an object")
        lid = text(loan.get("id"), "loan id")
        require(lid not in seen, "Duplicate loan id")
        seen.add(lid)
        require(isinstance(loan.get("customer_id"), str) and loan["customer_id"] in customer_ids,
                "Unknown loan customer")
        number(loan.get("amount"), "loan amount", True)
        number(loan.get("annual_income"), "annual income", True)
        number(loan["amount"] / loan["annual_income"], "loan/income ratio")
    research = data.get("research")
    require(isinstance(research, dict), "research must be an object")
    allowed = research.get("allowlist")
    sources = research.get("sources")
    keywords = research.get("keywords")
    require(isinstance(allowed, list) and allowed, "allowlist must be nonempty")
    for item in allowed:
        url(item)
    require(len(set(allowed)) == len(allowed), "Duplicate allowlist URL")
    require(isinstance(sources, list) and sources, "sources must be nonempty")
    require(isinstance(keywords, list) and keywords, "keywords must be nonempty")
    for word in keywords:
        text(word, "keyword")
    seen = set()
    for source in sources:
        require(isinstance(source, dict), "Source must be an object")
        address = url(source.get("url"))
        require(address in allowed, "Source URL is not exactly allowlisted")
        require(address not in seen, "Duplicate source URL")
        seen.add(address)
        text(source.get("title"), "source title")
        text(source.get("content"), "source content")
        timestamp(source.get("retrieved_at"))
        require(len(source["content"]) <= 100000, "Source content exceeds fixture limit")
    payments = data.get("payment_payloads", [])
    require(isinstance(payments, list), "payment_payloads must be an array")
    normalized_payments = []
    transaction_map = {t["id"]: t for t in transactions}
    payment_ids = set()
    for payload in payments:
        text(payload, "payment payload")
        require(len(payload) <= 100000 and "<!DOCTYPE" not in payload.upper()
                and "<!ENTITY" not in payload.upper(), "Unsafe or oversized XML")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            raise ValidationError("Invalid payment XML") from None
        require(root.tag.split("}")[-1] == "Document", "Payment root must be Document")
        def one(tag):
            elements = [e for e in root.iter() if e.tag.split("}")[-1] == tag]
            require(len(elements) == 1, "Payment requires exactly one " + tag)
            return elements[0]
        tid = text(one("EndToEndId").text, "payment transaction id")
        require(tid in transaction_map and tid not in payment_ids, "Unknown or duplicate payment transaction")
        payment_ids.add(tid)
        amount_node = one("InstdAmt")
        try:
            amount = float(amount_node.text)
        except (ValueError, TypeError):
            raise ValidationError("Invalid payment amount") from None
        number(amount, "payment amount", True)
        transaction = transaction_map[tid]
        require(amount == transaction["amount"] and amount_node.get("Ccy") == transaction["currency"],
                "Payment does not match transaction amount/currency")
        iban = text(one("IBAN").text, "payment IBAN")
        customer = next(c for c in customers if c["id"] == transaction["customer_id"])
        require(iban == customer["iban"], "Payment IBAN does not match customer")
        normalized_payments.append({"transaction_id": tid, "amount": amount,
                                    "currency": transaction["currency"], "iban": iban})
    return customers, transactions, loans, research, normalized_payments


def score(components):
    return {"value": min(100, sum(c["points"] for c in components)), "scale": "0-100 risk",
            "method": "synthetic_rules_v1", "explanation": components,
            "aggregation": "sum of contributions capped at 100"}


def run(data, extractor=None):
    customers, transactions, loans, research, payments = validate(data)
    profiles = {}
    for c in customers:
        flags = ([] if c["kyc_status"] == "verified" else ["KYC_REVIEW"]) + (
            ["AML_REVIEW"] if c["aml_flag"] else [])
        profiles[c["id"]] = {"id": c["id"], "name": c["name"], "iban": c["iban"], "flags": flags,
                            "score": score([
                                {"reason": "KYC not verified", "points": 40 if "KYC_REVIEW" in flags else 0},
                                {"reason": "AML screening flag", "points": 60 if c["aml_flag"] else 0}])}
    results = []
    for t in transactions:
        flags = list(profiles[t["customer_id"]]["flags"])
        if t["aml_flag"] and "AML_REVIEW" not in flags:
            flags.append("AML_REVIEW")
        if t["amount"] >= 10000:
            flags.append("LARGE_TRANSACTION_REVIEW")
        results.append({"id": t["id"], "customer_id": t["customer_id"], "amount": t["amount"],
                        "currency": t["currency"], "card_number": t["card_number"], "flags": flags,
                        "score": score([{"reason": "Customer screening requires review",
                                         "points": 40 if profiles[t["customer_id"]]["flags"] else 0},
                                        {"reason": "Transaction AML flag", "points": 40 if t["aml_flag"] else 0},
                                        {"reason": "Amount at least 10000 currency units",
                                         "points": 20 if t["amount"] >= 10000 else 0}])})
    loan_results = []
    for loan in loans:
        ratio = loan["amount"] / loan["annual_income"]
        flags = list(profiles[loan["customer_id"]]["flags"])
        loan_results.append({"id": loan["id"], "customer_id": loan["customer_id"],
                             "flags": flags, "disposition": "manual_review_only",
                             "score": score([{"reason": "Loan/income ratio greater than 2",
                                              "observed_ratio": ratio, "points": 50 if ratio > 2 else 0},
                                             {"reason": "Customer screening requires review",
                                              "points": 50 if flags else 0}])})
    findings, provenance = [], []
    for source in research["sources"]:
        content = source["content"]
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        provenance.append({"url": source["url"], "title": source["title"],
                           "retrieved_at": source["retrieved_at"], "sha256": digest,
                           "retrieval_mode": "supplied_offline_snapshot"})
        if extractor is None:
            spans = []
            for match in re.finditer(r"[^\n.!?]+[.!?]?", content):
                if any(k.casefold() in match[0].casefold() for k in research["keywords"]):
                    spans.append({"start": match.start(), "end": match.end(), "quote": match[0]})
        else:
            # The callable receives only immutable text, not the shared input object.
            try:
                spans = extractor(content, tuple(research["keywords"]))
            except Exception:
                raise ValidationError("Injected extractor failed") from None
        require(isinstance(spans, list) and len(spans) <= 1000, "Extractor must return a bounded list")
        for span in spans:
            require(isinstance(span, dict), "Finding must be an object")
            start, end = span.get("start"), span.get("end")
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(content),
                    "Invalid evidence offsets")
            require(span.get("quote") == content[start:end], "Finding quote must match original evidence")
            findings.append({"source_url": source["url"], "source_sha256": digest,
                             "retrieved_at": source["retrieved_at"], "start": start, "end": end,
                             "quote": span["quote"], "offset_basis": "original snapshot Unicode characters"})
    return mask({"status": "ok", "schema_version": "1.0", "synthetic": True,
                 "notice": "Demonstration only; no compliance certification or lending decisions. "
                           "No live retrieval; source metadata is fixture-supplied, not independently verified.",
                 "customers": list(profiles.values()), "transactions": results,
                 "loan_applications": loan_results, "payments": payments,
                 "research": {"sources": provenance, "findings": findings,
                              "redaction": "PAN-like digits masked after original-snapshot evidence validation"}})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle)
        result = run(data)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, csv.Error, RecursionError):
        # Deliberately avoid echoing untrusted input or filesystem details.
        print(json.dumps({"status": "error", "message": "Invalid input, file, or fixture; validation failed."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
