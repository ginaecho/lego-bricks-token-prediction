"""Synthetic, deterministic financial product discovery followed by smart search.

Demonstrative screening/masking rules only; not compliance certification,
credit underwriting, investment advice, or an ISO 20022 implementation.
"""

import copy
import csv
import difflib
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


def text(value, field):
    require(isinstance(value, str) and bool(value.strip()), field + " must be nonempty text")
    return value.strip()


def number(value, field, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            field + " must be a finite number in range")
    return value


def strings(value, field):
    require(isinstance(value, list), field + " must be an array")
    return [text(item, field) for item in value]


def fields(value, required, optional=()):
    require(isinstance(value, dict), "expected an object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "missing or unsupported fields")


def mask_card(value):
    if value is None:
        return None
    require(isinstance(value, str) and re.fullmatch(r"\d{13,19}", value) is not None,
            "card_number must contain 13-19 digits")
    return "*" * (len(value) - 4) + value[-4:]


def normalize_ledger(ledger):
    fields(ledger, ("format", "data"))
    fmt, data = ledger["format"], ledger["data"]
    if fmt == "json":
        require(isinstance(data, list), "JSON ledger must be an array")
        rows = data
    elif fmt == "csv":
        require(isinstance(data, str), "CSV ledger must be text")
        reader = csv.DictReader(io.StringIO(data))
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "CSV headers must be present and unique")
        require(set(reader.fieldnames) == {"id", "date", "amount", "currency", "category", "card_number"},
                "invalid CSV headers")
        rows = list(reader)
        for row in rows:
            require(None not in row and all(v is not None for v in row.values()), "invalid CSV row")
            try:
                row["amount"] = float(row["amount"])
            except ValueError:
                raise ValidationError("invalid CSV amount") from None
            row["card_number"] = row["card_number"] or None
    elif fmt == "iso20022":
        fields(data, ("CstmrCdtTrfInitn",))
        fields(data["CstmrCdtTrfInitn"], ("PmtInf",))
        payments = data["CstmrCdtTrfInitn"]["PmtInf"]
        require(isinstance(payments, list), "PmtInf must be an array")
        rows = []
        for payment in payments:
            fields(payment, ("EndToEndId", "ReqdExctnDt", "InstdAmt", "CtgyPurp"),
                   ("CardNumber",))
            fields(payment["InstdAmt"], ("value", "Ccy"))
            rows.append({"id": payment["EndToEndId"], "date": payment["ReqdExctnDt"],
                         "amount": payment["InstdAmt"]["value"],
                         "currency": payment["InstdAmt"]["Ccy"],
                         "category": payment["CtgyPurp"],
                         "card_number": payment.get("CardNumber")})
    else:
        raise ValidationError("unsupported ledger format")
    normalized = []
    for row in rows:
        fields(row, ("id", "date", "amount", "currency", "category"), ("card_number",))
        normalized.append({key: value for key, value in row.items() if key != "card_number"} |
                          {"masked_card": mask_card(row.get("card_number"))})
    return normalized


def validate_state(state):
    """One shared contract, checked on ingress and both sides of every handoff."""
    fields(state, ("schema_version", "status", "synthetic", "stage", "customer",
                   "transactions", "loan_application", "products", "query",
                   "recommendations", "results", "screening"))
    require(state["schema_version"] == "1.0" and state["status"] == "ok"
            and state["synthetic"] is True, "invalid schema header")
    require(state["stage"] in ("validated", "recommend", "search"), "invalid pipeline stage")
    customer = state["customer"]
    fields(customer, ("id", "kyc_status", "aml_flags", "preferences", "risk_tolerance",
                      "account_iban"))
    text(customer["id"], "customer.id")
    require(customer["kyc_status"] in ("verified", "pending", "rejected"), "invalid KYC status")
    strings(customer["aml_flags"], "aml_flags")
    strings(customer["preferences"], "preferences")
    require(type(customer["risk_tolerance"]) is int and 0 <= customer["risk_tolerance"] <= 5,
            "risk_tolerance must be an integer from 0 to 5")
    require(isinstance(customer["account_iban"], str) and
            re.fullmatch(r"SYNTH-[A-Z0-9]{8,24}", customer["account_iban"]) is not None,
            "account_iban must be an explicitly fake SYNTH identifier")
    require(isinstance(state["query"], str) and len(state["query"]) <= 500,
            "query must be text of at most 500 characters")
    transactions = state["transactions"]
    require(isinstance(transactions, list), "transactions must be an array")
    transaction_ids = set()
    for row in transactions:
        fields(row, ("id", "date", "amount", "currency", "category", "masked_card"))
        tid = text(row["id"], "transaction.id")
        require(tid not in transaction_ids, "duplicate transaction id")
        transaction_ids.add(tid)
        require(isinstance(row["date"], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]),
                "date must be YYYY-MM-DD")
        try:
            date.fromisoformat(row["date"])
        except ValueError:
            raise ValidationError("invalid transaction date") from None
        number(row["amount"], "amount", -1e15)
        require(isinstance(row["currency"], str) and re.fullmatch(r"[A-Z]{3}", row["currency"]),
                "currency must be three uppercase letters")
        text(row["category"], "category")
        require(row["masked_card"] is None or
                isinstance(row["masked_card"], str) and re.fullmatch(r"\*{9,15}\d{4}", row["masked_card"]),
                "unmasked card in pipeline")
    loan = state["loan_application"]
    if loan is not None:
        fields(loan, ("id", "amount", "purpose", "currency", "status"))
        text(loan["id"], "loan.id")
        number(loan["amount"], "loan.amount", 1)
        text(loan["purpose"], "loan.purpose")
        require(isinstance(loan["currency"], str) and re.fullmatch(r"[A-Z]{3}", loan["currency"]),
                "invalid loan currency")
        require(loan["status"] in ("draft", "submitted", "declined"), "invalid loan status")
    require(isinstance(state["products"], list), "products must be an array")
    products = {}
    for product in state["products"]:
        fields(product, ("id", "name", "category", "tags", "risk", "kind", "currency", "max_amount"))
        pid = text(product["id"], "product.id")
        require(pid not in products, "duplicate product id")
        products[pid] = product
        text(product["name"], "product.name")
        text(product["category"], "product.category")
        strings(product["tags"], "tags")
        require(type(product["risk"]) is int and 0 <= product["risk"] <= 5, "invalid product risk")
        require(product["kind"] in ("account", "card", "loan", "investment"), "invalid product kind")
        require(isinstance(product["currency"], str) and re.fullmatch(r"[A-Z]{3}", product["currency"]),
                "invalid product currency")
        number(product["max_amount"], "max_amount")
    expected_screening = screening(customer)
    require(state["screening"] == expected_screening, "inconsistent screening")
    for field in ("recommendations", "results"):
        require(isinstance(state[field], list), field + " must be an array")
        ids = set()
        for scored in state[field]:
            fields(scored, ("product_id", "score", "explanation"))
            pid = scored["product_id"]
            require(isinstance(pid, str) and pid in products and pid not in ids, "invalid scored product id")
            ids.add(pid)
            require(not expected_screening["blocked"] and eligible(products[pid], customer, loan),
                    "ineligible product in output")
            number(scored["score"], "score")
            require(isinstance(scored["explanation"], list) and bool(scored["explanation"]),
                    "every score must have explanations")
            total = 0
            for component in scored["explanation"]:
                fields(component, ("reason", "points"))
                text(component["reason"], "score reason")
                total += number(component["points"], "score component")
            require(math.isclose(total, scored["score"], abs_tol=1e-9), "score explanation does not sum")
    recommendation_map = {r["product_id"]: r for r in state["recommendations"]}
    require(all(r["product_id"] in recommendation_map for r in state["results"]),
            "search results must come from discovery")
    for result in state["results"]:
        first = result["explanation"][0]
        require(first == {"reason": "validated discovery score",
                          "points": recommendation_map[result["product_id"]]["score"]},
                "search must preserve discovery score")
    if state["stage"] == "validated":
        require(not state["recommendations"] and not state["results"], "premature scores")
    if state["stage"] == "recommend":
        require(not state["results"], "premature search results")
    return state


def screening(customer):
    reasons = []
    if customer["kyc_status"] != "verified":
        reasons.append("KYC not verified")
    if customer["aml_flags"]:
        reasons.append("AML flags require review")
    return {"blocked": bool(reasons), "reasons": reasons,
            "notice": "Demonstrative screening only; no compliance certification."}


def eligible(product, customer, loan):
    if product["risk"] > customer["risk_tolerance"]:
        return False
    if product["kind"] == "loan":
        return (loan is not None and loan["status"] != "declined"
                and loan["currency"] == product["currency"] and loan["amount"] <= product["max_amount"])
    return True


def normalize_input(payload):
    fields(payload, ("schema_version", "synthetic", "customer", "ledger",
                     "loan_application", "products", "query"))
    customer = payload["customer"]
    fields(customer, ("id", "kyc_status", "aml_flags", "preferences", "risk_tolerance", "account_iban"))
    state = {"schema_version": payload["schema_version"], "status": "ok",
             "synthetic": payload["synthetic"], "stage": "validated",
             "customer": copy.deepcopy(customer), "transactions": normalize_ledger(payload["ledger"]),
             "loan_application": copy.deepcopy(payload["loan_application"]),
             "products": copy.deepcopy(payload["products"]), "query": payload["query"],
             "recommendations": [], "results": [], "screening": screening(customer)}
    return validate_state(state)


def tokens(value):
    aliases = {"savings": "saving", "save": "saving", "saver": "saving",
               "borrowing": "loan", "borrow": "loan", "loans": "loan",
               "automobile": "car", "auto": "car", "cashback": "reward", "rewards": "reward"}
    return {aliases.get(word, word) for word in re.findall(r"[a-z0-9]+", value.lower())}


def product_tokens(product):
    return tokens(" ".join([product["name"], product["category"], product["kind"]] + product["tags"]))


def recommend(state):
    validate_state(state)
    require(state["stage"] == "validated", "discovery requires validated input")
    output = copy.deepcopy(state)
    output["stage"] = "recommend"
    if not state["screening"]["blocked"]:
        preferences = tokens(" ".join(state["customer"]["preferences"]))
        history = tokens(" ".join(t["category"] for t in state["transactions"]))
        for product in state["products"]:
            if not eligible(product, state["customer"], state["loan_application"]):
                continue
            vocabulary = product_tokens(product)
            preferred = sorted(preferences & vocabulary)
            observed = sorted(history & vocabulary)
            explanation = [
                {"reason": "eligible product baseline; not a credit or suitability rating", "points": 10},
                {"reason": "preference matches: " + (", ".join(preferred) or "none"),
                 "points": 20 * len(preferred)},
                {"reason": "transaction category matches: " + (", ".join(observed) or "none"),
                 "points": 5 * len(observed)},
                {"reason": "matches active loan application" if product["kind"] == "loan"
                 else "not a loan application match", "points": 25 if product["kind"] == "loan" else 0}]
            output["recommendations"].append({"product_id": product["id"],
                                             "score": sum(c["points"] for c in explanation),
                                             "explanation": explanation})
    output["recommendations"].sort(key=lambda r: (-r["score"], r["product_id"]))
    return validate_state(output)


def search(state):
    validate_state(state)
    require(state["stage"] == "recommend", "search requires validated discovery output")
    output = copy.deepcopy(state)
    output["stage"] = "search"
    query = tokens(state["query"]) - {"a", "an", "the", "for", "with", "i", "want", "me"}
    products = {p["id"]: p for p in state["products"]}
    for candidate in state["recommendations"]:
        vocabulary = product_tokens(products[candidate["product_id"]])
        explanation = [{"reason": "validated discovery score", "points": candidate["score"]}]
        matched = True
        for token in sorted(query):
            if token in vocabulary:
                explanation.append({"reason": "exact/alias query match: " + token, "points": 40})
            else:
                fuzzy = difflib.get_close_matches(token, sorted(vocabulary), n=1, cutoff=0.8) if len(token) >= 4 else []
                if not fuzzy:
                    matched = False
                    break
                explanation.append({"reason": "typo query match: " + token + " -> " + fuzzy[0], "points": 25})
        if matched:
            if not query:
                explanation.append({"reason": "empty query preserves personalized discovery", "points": 0})
            output["results"].append({"product_id": candidate["product_id"],
                                      "score": sum(c["points"] for c in explanation),
                                      "explanation": explanation})
    output["results"].sort(key=lambda r: (-r["score"], r["product_id"]))
    return validate_state(output)


def run(payload):
    return search(recommend(normalize_input(payload)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as stream:
            payload = json.load(stream)
        result = run(payload)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file; check the shared schema."}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
