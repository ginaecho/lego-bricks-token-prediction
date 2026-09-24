"""Synthetic loan-product comparison; educational rules, not compliance certification.

Run: python -B implementation.py example_input.json
All input formats converge on one validated schema before scoring.
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


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, field):
    require(isinstance(value, dict), field + " must be an object")
    return value


def text(value, field):
    require(isinstance(value, str) and bool(value.strip()), field + " must be text")
    return value.strip()


def number(value, field, low=0, high=1e12):
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            field + " must be numeric")
    require(math.isfinite(value) and low <= value <= high,
            field + " is outside the allowed finite range")
    return float(value)


def integer(value, field, low, high):
    number(value, field, low, high)
    require(isinstance(value, int), field + " must be an integer")
    return value


def money(value, currency, field):
    obj(value, field)
    require(value.get("currency") == currency, field + " currency mismatch")
    amount = number(value.get("value"), field + ".value")
    unit = value.get("unit", "major")
    require(unit in ("major", "minor"), field + " unit must be major or minor")
    return amount / 100 if unit == "minor" else amount


def pan(value):
    require(isinstance(value, str), "card_number must be text")
    require(bool(re.fullmatch(r"[0-9 -]+", value)), "invalid card_number")
    digits = re.sub(r"[ -]", "", value)
    require(13 <= len(digits) <= 19, "invalid card_number length")
    return "*" * (len(digits) - 4) + digits[-4:]


def safe_output(value):
    """Defense-in-depth: redact PAN-like strings in any echoed text field."""
    if isinstance(value, dict):
        return {safe_output(k): safe_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_output(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
                      lambda m: pan(m.group()), value)
    return value


def transactions(ledger, currency):
    obj(ledger, "ledger")
    fmt = ledger.get("format")
    rows = ledger.get("data")
    if fmt == "csv":
        require(isinstance(rows, str), "CSV ledger data must be text")
        try:
            reader = csv.DictReader(io.StringIO(rows), strict=True)
            required = {"id", "date", "amount", "currency", "direction"}
            require(reader.fieldnames is not None and
                    required.issubset(reader.fieldnames), "CSV headers missing")
            require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                    "duplicate CSV headers")
            rows = list(reader)
            for row in rows:
                require(None not in row and None not in row.values(),
                        "malformed CSV row")
                try:
                    row["amount"] = float(row["amount"])
                except (ValueError, TypeError):
                    raise ValidationError("CSV amount must be numeric") from None
        except csv.Error:
            raise ValidationError("invalid CSV ledger") from None
    elif fmt == "iso20022":
        obj(rows, "ISO payment payload")
        document = obj(rows.get("Document"), "ISO Document")
        initiation = obj(document.get("CstmrCdtTrfInitn"), "CstmrCdtTrfInitn")
        rows = initiation.get("PmtInf")
        require(isinstance(rows, list), "ISO payload requires PmtInf list")
        converted = []
        for payment in rows:
            obj(payment, "ISO payment")
            amount = obj(payment.get("InstdAmt"), "InstdAmt")
            converted.append({
                "id": payment.get("EndToEndId"),
                "date": payment.get("ReqdExctnDt"),
                "amount": amount.get("value"),
                "currency": amount.get("Ccy"),
                "direction": payment.get("Direction"),
                "account": payment.get("DbtrAcct"),
                **({"card_number": payment["CardNumber"]}
                   if "CardNumber" in payment else {}),
            })
        rows = converted
    else:
        require(fmt == "json", "unsupported ledger format")
    require(isinstance(rows, list) and len(rows) <= 10000,
            "ledger must contain at most 10000 transactions")
    normalized = []
    seen = set()
    for row in rows:
        obj(row, "transaction")
        ident = text(row.get("id"), "transaction.id")
        require(ident not in seen, "duplicate transaction id")
        seen.add(ident)
        day = text(row.get("date"), "transaction.date")
        require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)),
                "transaction date must be YYYY-MM-DD")
        try:
            date.fromisoformat(day)
        except ValueError:
            raise ValidationError("invalid transaction date") from None
        require(row.get("currency") == currency, "transaction currency mismatch")
        direction = row.get("direction")
        require(direction in ("credit", "debit"), "invalid transaction direction")
        item = {"id": ident, "date": day, "currency": currency,
                "amount": number(row.get("amount"), "transaction.amount"),
                "direction": direction}
        if "card_number" in row and not (fmt == "csv" and row["card_number"] == ""):
            item["masked_card"] = pan(row["card_number"])
        normalized.append(item)
    return normalized


def normalize(data):
    obj(data, "input")
    require(data.get("schema_version") == "1.0", "unsupported schema_version")
    require(data.get("synthetic") is True, "synthetic must be true")
    customer = obj(data.get("customer"), "customer")
    customer_id = text(customer.get("id"), "customer.id")
    kyc = customer.get("kyc_status")
    require(kyc in ("verified", "pending", "rejected"), "invalid kyc_status")
    flags = customer.get("aml_flags")
    require(isinstance(flags, list) and all(
        isinstance(flag, str) and flag.strip() for flag in flags),
        "aml_flags must be a list of nonempty strings")
    application = obj(data.get("loan_application"), "loan_application")
    currency = application.get("currency")
    require(currency in ("USD", "EUR", "GBP"), "unsupported currency")
    principal = number(application.get("amount"), "loan amount", 1, 1e9)
    months = integer(application.get("term_months"), "term_months", 1, 600)
    income = number(customer.get("monthly_income"), "monthly_income", 1)
    debt = number(customer.get("monthly_debt"), "monthly_debt")
    ledger = transactions(data.get("ledger"), currency)
    weights = obj(data.get("preferences"), "preferences")
    require(set(weights) == {"apr", "fee", "payment"},
            "preferences require apr, fee and payment weights")
    weights = {k: number(v, "preference weight", 0, 1000)
               for k, v in weights.items()}
    total = sum(weights.values())
    require(total > 0, "at least one preference must be positive")
    weights = {k: v / total for k, v in weights.items()}
    products = data.get("products")
    require(isinstance(products, list) and 1 <= len(products) <= 100,
            "products must contain 1 to 100 entries")
    normalized = []
    seen = set()
    for product in products:
        obj(product, "product")
        ident = text(product.get("id"), "product.id")
        require(ident not in seen, "duplicate product id")
        seen.add(ident)
        name = text(product.get("name"), "product.name")
        apr = obj(product.get("apr"), "apr")
        unit = apr.get("unit")
        require(unit in ("percent", "fraction", "basis_points"), "invalid APR unit")
        rate = number(apr.get("value"), "apr.value", 0, 10000)
        rate *= {"percent": 0.01, "fraction": 1, "basis_points": 0.0001}[unit]
        require(rate <= 1, "APR must not exceed 100 percent")
        fee = money(product.get("fee"), currency, "fee")
        minimum = money(product.get("min_amount"), currency, "min_amount")
        maximum = money(product.get("max_amount"), currency, "max_amount")
        require(minimum <= maximum, "min_amount exceeds max_amount")
        min_term = integer(product.get("min_term_months"), "min_term_months", 1, 600)
        max_term = integer(product.get("max_term_months"), "max_term_months", 1, 600)
        require(min_term <= max_term, "invalid product term range")
        monthly = rate / 12
        payment = principal / months if not monthly else (
            principal * monthly / -math.expm1(-months * math.log1p(monthly)))
        dti = (debt + payment) / income
        reasons = []
        if kyc != "verified":
            reasons.append("KYC_NOT_VERIFIED")
        if flags:
            reasons.append("AML_REVIEW_REQUIRED")
        if not minimum <= principal <= maximum:
            reasons.append("AMOUNT_OUT_OF_RANGE")
        if not min_term <= months <= max_term:
            reasons.append("TERM_OUT_OF_RANGE")
        if dti > 0.4:
            reasons.append("DEBT_TO_INCOME_EXCEEDS_40_PERCENT")
        normalized.append({
            "id": ident, "name": name, "apr": rate, "fee": fee,
            "payment": payment, "total_repayment": payment * months + fee,
            "debt_to_income": dti, "eligible": not reasons,
            "exclusion_reasons": reasons,
        })
    return {"customer_id": customer_id, "currency": currency,
            "principal": principal, "term_months": months, "weights": weights,
            "products": normalized, "transactions": ledger,
            "screening": {"kyc_status": kyc, "aml_flags": flags,
                          "review_required": kyc != "verified" or bool(flags)}}


def compare(data):
    normalized = normalize(data)
    products = normalized["products"]
    eligible = [p for p in products if p["eligible"]]
    bounds = {
        key: (min(p[key] for p in eligible), max(p[key] for p in eligible))
        for key in normalized["weights"]
    } if eligible else {}
    for product in products:
        if not product["eligible"]:
            product["score"] = None
            product["explanation"] = {
                "rule": "Excluded before preference scoring",
                "reasons": product["exclusion_reasons"],
                "dti_formula": "(monthly debt + loan payment) / monthly income",
                "dti_limit": 0.4,
            }
            continue
        parts = []
        for key, weight in normalized["weights"].items():
            low, high = bounds[key]
            utility = 1.0 if high == low else (high - product[key]) / (high - low)
            parts.append({"attribute": key, "value": product[key],
                          "minimum": low, "maximum": high,
                          "weight": weight, "utility": utility,
                          "contribution": 100 * weight * utility})
        product["score"] = sum(p["contribution"] for p in parts)
        product["explanation"] = {
            "rule": "100 * sum(weight * utility); lower attributes are preferred",
            "utility_formula": "(maximum - value) / (maximum - minimum)",
            "equal_attribute_rule": "utility = 1 when all eligible values equal",
            "comparison_population": "eligible products only",
            "components": parts,
            "dti_formula": "(monthly debt + loan payment) / monthly income",
            "dti_limit": 0.4,
        }
    ranked = sorted(eligible, key=lambda p: (-p["score"], p["id"]))
    ledger = normalized["transactions"]
    result = {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "notice": "Synthetic demonstration only; not lending advice or compliance certification.",
        "customer_id": normalized["customer_id"],
        "application": {k: normalized[k] for k in ("currency", "principal", "term_months")},
        "screening": normalized["screening"],
        "transactions": ledger,
        "transaction_summary": {
            "count": len(ledger),
            "credits": sum(t["amount"] for t in ledger if t["direction"] == "credit"),
            "debits": sum(t["amount"] for t in ledger if t["direction"] == "debit"),
            "use": "Context only; history is not used to infer creditworthiness",
        },
        "comparison": {
            "columns": ["id", "name", "apr", "fee", "payment", "total_repayment",
                        "debt_to_income", "eligible", "score"],
            "units": {"apr": "fraction", "fee": "currency major units",
                      "payment": "currency major units per month",
                      "total_repayment": "currency major units", "score": "0 to 100"},
            "rows": products,
        },
        "ranking": [{"rank": i + 1, "product_id": p["id"], "score": p["score"]}
                    for i, p in enumerate(ranked)],
        "model": {
            "version": "deterministic-preference-v1",
            "tie_breaker": "product id ascending",
            "payment": "Fixed-rate amortization; zero APR uses principal / months",
            "fees": "One-time upfront fee, not financed; no compounding fees",
            "weights": normalized["weights"],
            "limits": "No credit bureau, real AML checks, FX conversion or live provider",
        },
    }
    return safe_output(result)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle)
        result = compare(data)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, AttributeError, OverflowError, RecursionError):
        # Do not echo exception text: paths and rejected values can contain sensitive data.
        print(json.dumps({"schema_version": "1.0", "status": "error",
                          "error": "Invalid input or unreadable input file"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
