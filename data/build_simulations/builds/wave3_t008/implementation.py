"""Synthetic financial search -> prerequisite-driven onboarding reference CLI.

Standard library only. Embeddings may be supplied as a local callable to run();
no provider is configured or called. Rules are demonstrations, not certification.
"""

import csv
import io
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, name):
    require(isinstance(value, dict), name + " must be an object")
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    return value.strip()


def synthetic_id(value, name):
    value = text(value, name)
    require(bool(re.fullmatch(r"SYN-[A-Za-z0-9-]+", value)), name + " must have a SYN- identifier")
    return value


def strings(value, name):
    require(isinstance(value, list), name + " must be a list")
    result = [text(item, name + " item") for item in value]
    require(len(result) == len(set(result)), name + " contains duplicates")
    return result


def money(value, name, positive=False):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            name + " must be a decimal amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(name + " must be a decimal amount") from None
    require(amount.is_finite() and amount.copy_abs() <= Decimal("1000000000"),
            name + " must be finite and within the demonstration limit")
    require(amount == amount.quantize(Decimal("0.01")), name + " has excess precision")
    require(not positive or amount > 0, name + " must be positive")
    return format(amount, ".2f")


def currency(value):
    require(value in ("EUR", "USD", "GBP"), "currency must be EUR, USD or GBP")
    return value


def fake_account(value):
    value = text(value, "account")
    require(bool(re.fullmatch(r"FAKE-[A-Z0-9-]{4,40}", value)),
            "account/IBAN must be explicitly fake (FAKE-...)")
    return value


def mask_card(value):
    if value is None or value == "":
        return None
    value = text(value, "card_number")
    if re.fullmatch(r"\*{12}\d{4}", value):
        return value
    digits = re.sub(r"[ -]", "", value)
    require(bool(re.fullmatch(r"\d{13,19}", digits)), "card_number must have 13-19 digits")
    return "*" * 12 + digits[-4:]


def safe_text(value, name):
    value = text(value, name)
    require(not re.search(r"(?:\d[ -]?){13,19}", value),
            name + " must not contain an unmasked card number")
    return value


def normalize_ledger(ledger, customer_id):
    ledger = obj(ledger, "ledger")
    kind = ledger.get("format")
    data = ledger.get("data")
    if kind == "json":
        require(isinstance(data, list), "JSON ledger data must be a list")
        rows = data
    elif kind == "csv":
        require(isinstance(data, str), "CSV ledger data must be text")
        reader = csv.DictReader(io.StringIO(data), strict=True)
        required = {"id", "customer_id", "amount", "currency", "account", "description"}
        try:
            require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                    "CSV ledger is missing required columns")
            require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                    "CSV ledger has duplicate columns")
            rows = list(reader)
        except csv.Error:
            raise ValidationError("Malformed CSV ledger") from None
        require(all(None not in row and None not in row.values() for row in rows),
                "CSV ledger has inconsistent row width")
    elif kind == "iso20022":
        payment = obj(data, "ISO-style payload")
        transfers = obj(payment.get("CstmrCdtTrfInitn"), "CstmrCdtTrfInitn").get("CdtTrfTxInf")
        require(isinstance(transfers, list), "CdtTrfTxInf must be a list")
        rows = []
        for transfer in transfers:
            transfer = obj(transfer, "transfer")
            amount = obj(transfer.get("InstdAmt"), "InstdAmt")
            rows.append({
                "id": transfer.get("EndToEndId"),
                "customer_id": transfer.get("CustomerId"),
                "amount": amount.get("Value"),
                "currency": amount.get("Ccy"),
                "account": transfer.get("DbtrAcct"),
                "description": transfer.get("RmtInf"),
                "card_number": transfer.get("CardNumber"),
            })
    else:
        raise ValidationError("ledger format must be json, csv or iso20022")
    require(len(rows) <= 1000, "ledger exceeds 1000 transactions")
    normalized = []
    seen = set()
    for row in rows:
        row = obj(row, "transaction")
        transaction_id = synthetic_id(row.get("id"), "transaction id")
        require(transaction_id not in seen, "duplicate transaction id")
        seen.add(transaction_id)
        require(row.get("customer_id") == customer_id, "transaction customer mismatch")
        normalized.append({
            "id": transaction_id,
            "customer_id": customer_id,
            "amount": money(row.get("amount"), "transaction amount"),
            "currency": currency(row.get("currency")),
            "account": fake_account(row.get("account")),
            "description": safe_text(row.get("description"), "transaction description"),
            "card_number": mask_card(row.get("card_number")),
        })
    return normalized


def validate_input(payload):
    payload = obj(payload, "input")
    require(payload.get("schema_version") == "1.0", "schema_version must be 1.0")
    require(payload.get("synthetic") is True, "synthetic must be true")
    customer = obj(payload.get("customer"), "customer")
    customer_id = synthetic_id(customer.get("id"), "customer id")
    kyc = customer.get("kyc_status")
    require(kyc in ("pending", "verified", "rejected"), "invalid kyc_status")
    customer = {
        "id": customer_id,
        "name": safe_text(customer.get("name"), "customer name"),
        "kyc_status": kyc,
        "kyc_flags": strings(customer.get("kyc_flags", []), "kyc_flags"),
        "aml_flags": strings(customer.get("aml_flags", []), "aml_flags"),
    }
    for flag in customer["kyc_flags"] + customer["aml_flags"]:
        require(bool(re.fullmatch(r"[a-z_]{1,60}", flag)), "screening flags must be symbolic codes")
    transactions = normalize_ledger(payload.get("ledger"), customer_id)
    loan = obj(payload.get("loan_application"), "loan_application")
    require(loan.get("customer_id") == customer_id, "loan customer mismatch")
    require(type(loan.get("term_months")) is int and 1 <= loan["term_months"] <= 600,
            "term_months must be an integer from 1 to 600")
    loan = {
        "id": synthetic_id(loan.get("id"), "loan id"),
        "customer_id": customer_id,
        "amount": money(loan.get("amount"), "loan amount", positive=True),
        "currency": currency(loan.get("currency")),
        "term_months": loan["term_months"],
        "purpose": safe_text(loan.get("purpose"), "loan purpose"),
    }
    products = payload.get("products")
    require(isinstance(products, list) and 0 < len(products) <= 100, "products must contain 1-100 items")
    clean_products = []
    ids = set()
    for product in products:
        product = obj(product, "product")
        product_id = synthetic_id(product.get("id"), "product id")
        require(product_id not in ids, "duplicate product id")
        ids.add(product_id)
        require(product.get("kind") in ("loan", "payment", "savings"), "invalid product kind")
        clean_products.append({
            "id": product_id,
            "kind": product["kind"],
            "name": safe_text(product.get("name"), "product name"),
            "description": safe_text(product.get("description"), "product description"),
            "currency": currency(product.get("currency")),
        })
    top_k = payload.get("top_k", 3)
    require(type(top_k) is int and 1 <= top_k <= 20, "top_k must be an integer from 1 to 20")
    return {
        "schema_version": "1.0", "synthetic": True,
        "customer": customer, "transactions": transactions, "loan_application": loan,
        "products": clean_products, "query": safe_text(payload.get("query"), "query"),
        "top_k": top_k,
        "completed_steps": strings(payload.get("completed_steps", []), "completed_steps"),
    }


SYNONYMS = {
    "borrow": "loan", "borrowing": "loan", "credit": "loan", "loans": "loan",
    "transfer": "payment", "transfers": "payment", "payments": "payment",
    "saving": "savings", "deposit": "savings", "deposits": "savings",
}


def tokens(value):
    return {SYNONYMS.get(word, word) for word in re.findall(r"[a-z0-9]+", value.lower())}


def vector(embedder, value):
    try:
        result = embedder(value)
    except Exception:
        raise ValidationError("embedding callable failed") from None
    require(isinstance(result, (list, tuple)) and 1 <= len(result) <= 4096,
            "embedding must be a nonempty numeric vector")
    require(all(type(item) in (int, float) and abs(item) <= 1e6 and math.isfinite(item)
                for item in result), "embedding values must be finite bounded numbers")
    result = [float(item) for item in result]
    require(math.hypot(*result) > 0, "embedding vector must have nonzero norm")
    return result


def semantic_search(state, embedder=None):
    query_tokens = tokens(state["query"])
    require(bool(query_tokens), "query must contain searchable alphanumeric tokens")
    query_vector = vector(embedder, state["query"]) if embedder is not None else None
    index = {}
    documents = {}
    for product in state["products"]:
        document = product["name"] + " " + product["description"] + " " + product["kind"]
        documents[product["id"]] = document
        for token in tokens(document):
            index.setdefault(token, set()).add(product["id"])
    matches = []
    for product in state["products"]:
        # Currency suitability is a hard filter, never a learned score.
        if product["currency"] != state["loan_application"]["currency"]:
            continue
        matched = sorted(token for token in query_tokens if product["id"] in index.get(token, set()))
        lexical = len(matched) / len(query_tokens)
        cosine = None
        if query_vector is not None:
            product_vector = vector(embedder, documents[product["id"]])
            require(len(product_vector) == len(query_vector), "embedding dimensions must agree")
            cosine = sum((a / math.hypot(*query_vector)) * (b / math.hypot(*product_vector))
                         for a, b in zip(query_vector, product_vector))
            cosine = max(-1.0, min(1.0, cosine))
        score = lexical if cosine is None else 0.7 * lexical + 0.3 * max(0.0, cosine)
        if score <= 0:
            continue
        matches.append({
            "product": dict(product), "score": score,
            "explanation": {
                "matched_tokens": matched, "query_tokens": sorted(query_tokens),
                "lexical_coverage": lexical, "embedding_cosine": cosine,
                "formula": "lexical_coverage" if cosine is None else
                           "0.7 * lexical_coverage + 0.3 * max(0, embedding_cosine)",
                "purpose": "search relevance only; not creditworthiness or approval",
                "currency_filter": product["currency"],
            },
        })
    matches.sort(key=lambda match: (-match["score"], match["product"]["id"]))
    return {
        "schema_version": state["schema_version"],
        "customer_id": state["customer"]["id"],
        "loan_application_id": state["loan_application"]["id"],
        "transaction_ids": [transaction["id"] for transaction in state["transactions"]],
        "query": state["query"], "matches": matches[:state["top_k"]],
        "selected_product_id": matches[0]["product"]["id"] if matches else None,
    }


def validate_handoff(state, handoff):
    handoff = obj(handoff, "semantic handoff")
    require(handoff.get("schema_version") == state["schema_version"], "handoff schema mismatch")
    require(handoff.get("customer_id") == state["customer"]["id"], "handoff customer mismatch")
    require(handoff.get("loan_application_id") == state["loan_application"]["id"], "handoff loan mismatch")
    require(handoff.get("transaction_ids") == [item["id"] for item in state["transactions"]],
            "handoff transaction mismatch")
    require(handoff.get("query") == state["query"], "handoff query mismatch")
    matches = handoff.get("matches")
    require(isinstance(matches, list) and len(matches) <= state["top_k"], "invalid handoff matches")
    catalog = {product["id"]: product for product in state["products"]}
    seen = set()
    for match in matches:
        match = obj(match, "match")
        product = obj(match.get("product"), "match product")
        product_id = product.get("id")
        require(isinstance(product_id, str) and product_id in catalog and product == catalog[product_id],
                "handoff product is not in validated catalog")
        require(product_id not in seen, "duplicate handoff product")
        seen.add(product_id)
        score = match.get("score")
        require(type(score) in (int, float) and math.isfinite(score) and 0 < score <= 1,
                "invalid relevance score")
        explanation = obj(match.get("explanation"), "score explanation")
        query_tokens = sorted(tokens(state["query"]))
        matched_tokens = sorted(set(query_tokens) & tokens(
            product["name"] + " " + product["description"] + " " + product["kind"]))
        lexical = len(matched_tokens) / len(query_tokens)
        require(explanation.get("query_tokens") == query_tokens and
                explanation.get("matched_tokens") == matched_tokens and
                explanation.get("lexical_coverage") == lexical, "invalid lexical explanation")
        cosine = explanation.get("embedding_cosine")
        require(cosine is None or (type(cosine) in (int, float) and math.isfinite(cosine)
                                  and -1 <= cosine <= 1), "invalid embedding explanation")
        expected = lexical if cosine is None else 0.7 * lexical + 0.3 * max(0, cosine)
        require(math.isclose(score, expected, rel_tol=1e-12, abs_tol=1e-12),
                "score does not match explanation")
        require(explanation.get("formula") == ("lexical_coverage" if cosine is None else
                "0.7 * lexical_coverage + 0.3 * max(0, embedding_cosine)"),
                "invalid scoring formula")
        require(product["currency"] == state["loan_application"]["currency"],
                "handoff currency mismatch")
    require(matches == sorted(matches, key=lambda item: (-item["score"], item["product"]["id"])),
            "handoff matches must be ranked")
    require(handoff.get("selected_product_id") == (matches[0]["product"]["id"] if matches else None),
            "handoff selection must be the highest-ranked product")
    return handoff


def guided_setup(state, handoff):
    handoff = validate_handoff(state, handoff)
    completed = set(state["completed_steps"])
    if not handoff["matches"]:
        require(not completed, "completed steps require a selected product")
        return {"status": "not_started", "reason": "No relevant product; refine the query.",
                "selected_product_id": None, "steps": [], "next_steps": [],
                "progress": {"completed": 0, "total": 0, "percent": 0}}
    customer = state["customer"]
    chosen = handoff["matches"][0]["product"]
    rules = [
        ("kyc_review", [], customer["kyc_status"] == "verified" and not customer["kyc_flags"],
         "KYC must be verified with no screening flags."),
        ("aml_review", ["kyc_review"], not customer["aml_flags"],
         "AML flags require human review and clearance."),
        ("transaction_review", ["aml_review"], bool(state["transactions"]),
         "At least one validated synthetic transaction is required."),
    ]
    if chosen["kind"] == "loan":
        rules.append(("loan_application", ["transaction_review"], True,
                      "Validated loan details may be submitted for human review; no approval is implied."))
    require(completed <= {rule[0] for rule in rules}, "unknown or inapplicable completed step")
    steps = []
    for step_id, prerequisites, evidence_ok, reason in rules:
        unmet = sorted(set(prerequisites) - completed)
        if step_id in completed:
            require(evidence_ok and not unmet, "completed step violates evidence or prerequisites: " + step_id)
            status = "completed"
        else:
            status = "ready" if evidence_ok and not unmet else "blocked"
        steps.append({"id": step_id, "prerequisites": prerequisites, "status": status,
                      "evidence_satisfied": evidence_ok, "unmet_prerequisites": unmet,
                      "explanation": reason})
    next_steps = [step["id"] for step in steps if step["status"] == "ready"]
    count = len(completed)
    return {
        "status": "complete" if count == len(steps) else ("in_progress" if next_steps else "blocked"),
        "selected_product_id": chosen["id"],
        "customer_id": handoff["customer_id"],
        "loan_application_id": handoff["loan_application_id"],
        "transaction_ids": list(handoff["transaction_ids"]),
        "selection_explanation": dict(handoff["matches"][0]["explanation"]),
        "steps": steps, "next_steps": next_steps,
        "progress": {"completed": count, "total": len(steps), "percent": round(100 * count / len(steps), 2)},
        "resume": {"selected_product_id": chosen["id"],
                   "completed_steps": [step["id"] for step in steps if step["status"] == "completed"]},
        "notice": "Synthetic demonstration only. Screening flags block setup; no credit decision or compliance certification.",
    }


def run(payload, embedder=None):
    state = validate_input(payload)
    semantic = semantic_search(state, embedder=embedder)
    guided = guided_setup(state, semantic)
    return {
        "schema_version": state["schema_version"], "synthetic": True, "status": "ok",
        "customer": state["customer"], "transactions": state["transactions"],
        "loan_application": state["loan_application"], "semantic": semantic, "guided": guided,
        "notice": "Synthetic fixture data. Card masking is a demonstration, not PCI DSS certification.",
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        result = run(payload)
    except (ValueError, OSError, UnicodeError, RecursionError):
        # Never echo untrusted payloads, card data, provider exceptions, or file contents.
        print(json.dumps({"schema_version": "1.0", "status": "error",
                          "message": "Invalid input or unreadable file."}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
