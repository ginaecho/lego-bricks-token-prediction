"""Synthetic financial discovery -> onboarding -> comparison reference CLI.

No real compliance or lending decisions. Costs use a disclosed flat-interest
estimate, not an amortization schedule. All data is processed locally in memory.
"""

import copy
import csv
import datetime
import io
import json
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value, field, minimum=Decimal("0"), maximum=Decimal("1000000000")):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            field + " must be numeric")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(field + " must be numeric") from None
    require(result.is_finite() and minimum <= result <= maximum,
            field + " is outside its allowed range")
    return result


def money(value, field, positive=False):
    result = number(value, field, Decimal("0.01") if positive else Decimal("0"))
    require(result == result.quantize(Decimal("0.01")), field + " needs at most two decimals")
    return result


def rounded(value, places=6):
    return float(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def text(value, field):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 120,
            field + " must be nonempty text of at most 120 characters")
    # Prevent free-text fields from carrying unmasked PAN-like values.
    require(not re.search(r"(?:\d[ -]?){13,19}", value), field + " contains sensitive-looking digits")
    return value


def mapping(value, field):
    require(isinstance(value, dict), field + " must be an object")
    return value


def sequence(value, field, limit=1000):
    require(isinstance(value, list) and len(value) <= limit, field + " must be a bounded list")
    return value


def exact_keys(value, required, optional=()):
    mapping(value, "record")
    require(set(required) <= set(value) <= set(required) | set(optional),
            "record has missing or unsupported fields")


def synthetic_account(value):
    require(isinstance(value, str) and
            re.fullmatch(r"SYNTH-(?:ACCT|IBAN)-[A-Z0-9]{4,24}", value) is not None,
            "account must be a clearly synthetic account or IBAN")
    return value


def mask_card(value):
    require(isinstance(value, str), "card_number must be text")
    require(re.fullmatch(r"[0-9 -]+", value) is not None, "card_number has invalid characters")
    digits = value.replace(" ", "").replace("-", "")
    require(13 <= len(digits) <= 19, "card_number must contain 13 to 19 digits")
    return "*" * (len(digits) - 4) + digits[-4:]


def parse_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            require(key not in result, "JSON contains duplicate fields")
            result[key] = item
        return result
    try:
        return json.loads(value, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValidationError("nonfinite JSON number")))
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("invalid JSON document") from None


class Validator:
    """One schema/validation boundary, used on input and every stage handoff."""

    @staticmethod
    def transaction(row, currency, source):
        exact_keys(row, ("id", "date", "amount", "currency", "direction", "account"),
                   ("card_number",))
        try:
            require(isinstance(row["date"], str), "date must be text")
            require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]) is not None,
                    "date must use YYYY-MM-DD")
            datetime.date.fromisoformat(row["date"])
        except ValueError:
            raise ValidationError("invalid transaction date") from None
        require(row["currency"] == currency, "transaction currency mismatch")
        require(row["direction"] in ("credit", "debit"), "invalid transaction direction")
        result = {
            "id": text(row["id"], "transaction id"), "date": row["date"],
            "amount": str(money(row["amount"], "transaction amount", True)),
            "currency": currency, "direction": row["direction"],
            "account": synthetic_account(row["account"]), "source": source,
        }
        if row.get("card_number"):
            result["masked_card"] = mask_card(row["card_number"])
        elif "card_number" in row:
            require(row["card_number"] == "", "invalid optional card_number")
        return result

    @staticmethod
    def input(payload):
        exact_keys(payload, ("schema_version", "synthetic", "customer", "loan_application",
                             "transaction_ledger", "payment_payload", "products"))
        require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
                "unsupported schema_version")
        require(payload["synthetic"] is True, "only explicitly synthetic fixtures are accepted")
        customer = copy.deepcopy(payload["customer"])
        exact_keys(customer, ("id", "name", "account", "kyc", "aml_flags", "preferences"))
        text(customer["id"], "customer id")
        text(customer["name"], "customer name")
        synthetic_account(customer["account"])
        exact_keys(customer["kyc"], ("status", "screening_flags"))
        require(customer["kyc"]["status"] in ("verified", "pending", "rejected"),
                "invalid KYC status")
        for key, flags in (("KYC", customer["kyc"]["screening_flags"]),
                           ("AML", customer["aml_flags"])):
            for flag in sequence(flags, key + " flags", 20):
                text(flag, key + " flag")
        preferences = customer["preferences"]
        exact_keys(preferences, ("cost", "affordability", "term"))
        weights = {key: number(value, "preference", maximum=Decimal("100"))
                   for key, value in preferences.items()}
        require(sum(weights.values()) > 0, "at least one preference weight must be positive")
        customer["preferences"] = {key: str(value) for key, value in weights.items()}

        loan = copy.deepcopy(payload["loan_application"])
        exact_keys(loan, ("id", "customer_id", "amount", "currency", "desired_term_months",
                          "monthly_budget"))
        text(loan["id"], "loan id")
        require(loan["customer_id"] == customer["id"], "loan/customer mismatch")
        require(loan["currency"] in ("EUR", "USD", "GBP"), "unsupported loan currency")
        loan["amount"] = str(money(loan["amount"], "loan amount", True))
        loan["monthly_budget"] = str(money(loan["monthly_budget"], "monthly budget", True))
        term = number(loan["desired_term_months"], "desired term", Decimal("1"), Decimal("600"))
        require(term == int(term), "desired term must be whole months")
        loan["desired_term_months"] = int(term)

        ledger = payload["transaction_ledger"]
        exact_keys(ledger, ("format", "data"))
        if ledger["format"] == "json":
            rows = sequence(ledger["data"], "ledger")
        elif ledger["format"] == "csv":
            require(isinstance(ledger["data"], str) and len(ledger["data"]) <= 1_000_000,
                    "CSV ledger must be bounded text")
            try:
                reader = csv.DictReader(io.StringIO(ledger["data"]), strict=True)
                headers = reader.fieldnames
                required = {"id", "date", "amount", "currency", "direction", "account"}
                require(headers is not None and len(headers) == len(set(headers))
                        and required <= set(headers) <= required | {"card_number"},
                        "invalid CSV headers")
                rows = list(reader)
            except csv.Error:
                raise ValidationError("invalid CSV ledger") from None
            sequence(rows, "ledger")
        else:
            raise ValidationError("ledger format must be json or csv")
        transactions = [Validator.transaction(row, loan["currency"], "ledger") for row in rows]
        payment = payload["payment_payload"]
        exact_keys(payment, ("Document",))
        exact_keys(payment["Document"], ("CstmrCdtTrfInitn",))
        initiation = payment["Document"]["CstmrCdtTrfInitn"]
        exact_keys(initiation, ("MsgId", "CdtTrfTxInf"))
        text(initiation["MsgId"], "payment message id")
        for transfer in sequence(initiation["CdtTrfTxInf"], "payment transfers"):
            exact_keys(transfer, ("PmtId", "ReqdExctnDt", "Amt", "DbtrAcct", "CdtrAcct"))
            exact_keys(transfer["Amt"], ("InstdAmt", "Ccy"))
            exact_keys(transfer["DbtrAcct"], ("IBAN",))
            exact_keys(transfer["CdtrAcct"], ("IBAN",))
            debtor = synthetic_account(transfer["DbtrAcct"]["IBAN"])
            synthetic_account(transfer["CdtrAcct"]["IBAN"])
            require(debtor == customer["account"], "payment debtor/customer mismatch")
            transactions.append(Validator.transaction({
                "id": transfer["PmtId"], "date": transfer["ReqdExctnDt"],
                "amount": transfer["Amt"]["InstdAmt"], "currency": transfer["Amt"]["Ccy"],
                "direction": "debit", "account": debtor,
            }, loan["currency"], "iso20022_style"))
        require(len(transactions) <= 1000, "too many combined transactions")
        ids = [row["id"] for row in transactions]
        require(len(ids) == len(set(ids)), "duplicate transaction id across sources")
        require(all(row["account"] == customer["account"] for row in transactions),
                "transaction/customer account mismatch")

        products = []
        for product in sequence(payload["products"], "products", 100):
            exact_keys(product, ("id", "name", "currency", "apr", "monthly_fee", "term",
                                 "minimum_amount", "maximum_amount"))
            text(product["id"], "product id")
            text(product["name"], "product name")
            require(product["currency"] == loan["currency"], "product currency mismatch")
            for attribute in ("apr", "monthly_fee", "term"):
                exact_keys(product[attribute], ("value", "unit"))
            require(product["apr"]["unit"] in ("percent", "basis_points"), "unsupported APR unit")
            apr = number(product["apr"]["value"], "APR")
            if product["apr"]["unit"] == "basis_points":
                apr /= 100
            require(apr <= 100, "APR exceeds reference range")
            require(product["monthly_fee"]["unit"] in ("currency_per_month", "cents_per_month"),
                    "unsupported monthly fee unit")
            fee = number(product["monthly_fee"]["value"], "monthly fee")
            if product["monthly_fee"]["unit"] == "cents_per_month":
                fee /= 100
            fee = money(str(fee), "monthly fee")
            require(product["term"]["unit"] in ("months", "years"), "unsupported term unit")
            months = number(product["term"]["value"], "product term", Decimal("0.01"))
            if product["term"]["unit"] == "years":
                months *= 12
            require(months == int(months) and 1 <= months <= 600, "invalid product term")
            lower = money(product["minimum_amount"], "minimum amount", True)
            upper = money(product["maximum_amount"], "maximum amount", True)
            require(lower <= upper, "product amount range is reversed")
            products.append({
                "id": product["id"], "name": product["name"], "currency": product["currency"],
                "apr_percent": str(apr), "monthly_fee": str(fee), "term_months": int(months),
                "minimum_amount": str(lower), "maximum_amount": str(upper),
            })
        require(len({product["id"] for product in products}) == len(products),
                "duplicate product id")
        credits = sum((Decimal(t["amount"]) for t in transactions if t["direction"] == "credit"),
                      Decimal("0"))
        debits = sum((Decimal(t["amount"]) for t in transactions if t["direction"] == "debit"),
                     Decimal("0"))
        return {
            "customer": customer, "loan_application": loan, "transactions": transactions,
            "ledger_summary": {"count": len(transactions), "credit_total": str(credits),
                               "debit_total": str(debits), "net": str(credits - debits),
                               "currency": loan["currency"]},
            "products": products,
        }

    @staticmethod
    def envelope(envelope, expected_stage):
        exact_keys(envelope, ("schema_version", "synthetic", "status", "stage", "context",
                              "journey"), ("onboarding", "comparison"))
        require(envelope["schema_version"] == 1 and envelope["synthetic"] is True
                and envelope["status"] == "ok" and envelope["stage"] == expected_stage,
                "invalid pipeline envelope")
        context = mapping(envelope["context"], "context")
        exact_keys(context, ("customer", "loan_application", "transactions",
                             "ledger_summary", "products"))
        # Reconstruct the external schema to reuse its complete validation rules.
        transactions = []
        for row in context["transactions"]:
            exact_keys(row, ("id", "date", "amount", "currency", "direction", "account", "source"),
                       ("masked_card",))
            require(row["source"] in ("ledger", "iso20022_style"), "invalid transaction source")
            if "masked_card" in row:
                require(isinstance(row["masked_card"], str) and
                        re.fullmatch(r"\*{9,15}\d{4}", row["masked_card"]) is not None,
                        "unmasked card in handoff")
            transactions.append({key: row[key] for key in
                                 ("id", "date", "amount", "currency", "direction", "account")})
        products = [{
            "id": p["id"], "name": p["name"], "currency": p["currency"],
            "apr": {"value": p["apr_percent"], "unit": "percent"},
            "monthly_fee": {"value": p["monthly_fee"], "unit": "currency_per_month"},
            "term": {"value": p["term_months"], "unit": "months"},
            "minimum_amount": p["minimum_amount"], "maximum_amount": p["maximum_amount"],
        } for p in context["products"]]
        normalized = Validator.input({
            "schema_version": 1, "synthetic": True, "customer": context["customer"],
            "loan_application": context["loan_application"],
            "transaction_ledger": {"format": "json", "data": transactions},
            "payment_payload": {"Document": {"CstmrCdtTrfInitn":
                               {"MsgId": "SYNTH-HANDOFF", "CdtTrfTxInf": []}}},
            "products": products,
        })
        require(context["ledger_summary"] == normalized["ledger_summary"],
                "ledger summary does not match transactions")
        require(context["customer"] == normalized["customer"]
                and context["loan_application"] == normalized["loan_application"]
                and context["products"] == normalized["products"], "noncanonical handoff context")
        require(envelope["journey"] == plan(context), "invalid journey or prerequisites")
        if expected_stage in ("onboarding", "comparison"):
            require(envelope.get("onboarding") == onboard_plan(context, envelope["journey"]),
                    "invalid onboarding handoff")
        else:
            require("onboarding" not in envelope and "comparison" not in envelope,
                    "unexpected future-stage output")
        if expected_stage == "comparison":
            require(envelope.get("comparison") ==
                    comparison_plan(context, envelope["onboarding"]), "invalid comparison output")
        elif expected_stage == "onboarding":
            require("comparison" not in envelope, "unexpected comparison output")
        return envelope


def blockers(context):
    customer = context["customer"]
    result = []
    if customer["kyc"]["status"] != "verified":
        result.append("kyc_" + customer["kyc"]["status"])
    if customer["kyc"]["screening_flags"]:
        result.append("kyc_screening_review")
    if customer["aml_flags"]:
        result.append("aml_screening_review")
    return result


def plan(context):
    blocked = blockers(context)
    first = "clear_profile" if blocked else "prepare_comparison"
    first_requires = ["profile_available", "application_validated"]
    steps = [
        {"action": first, "requires": first_requires, "provides": ["comparison_ready"],
         "instruction": ("Resolve screening and identity checks with a human reviewer; "
                         "this plan does not clear any flags." if blocked else
                         "Use the supplied loan application and preferences to prepare comparison.")},
        {"action": "compare_products", "requires": ["comparison_ready"],
         "provides": ["comparison_available"],
         "instruction": "Compare normalized loan attributes and explained preference scores."},
    ]
    available = {"profile_available", "application_validated"}
    for step in steps:
        require(set(step["requires"]) <= available, "unsatisfied planned prerequisite")
        available.update(step["provides"])
    return {
        "customer_id": context["customer"]["id"],
        "application_id": context["loan_application"]["id"],
        "next_action": first, "blockers": blocked, "steps": steps,
        "two_step_validation": "valid_projected_path_not_proof_of_completion",
        "reason": {"desired_amount": context["loan_application"]["amount"],
                   "desired_term_months": context["loan_application"]["desired_term_months"],
                   "preferences": context["customer"]["preferences"],
                   "observed_transactions": context["ledger_summary"]["count"]},
    }


def onboard_plan(context, journey):
    ready = not journey["blockers"]
    return {
        "customer_id": journey["customer_id"], "application_id": journey["application_id"],
        "consumed_action": journey["next_action"],
        "status": "ready" if ready else "requires_review",
        "personalized_message": ("Hello " + context["customer"]["name"] + ". " +
                                 journey["steps"][0]["instruction"]),
        "next_step": "compare_products" if ready else journey["next_action"],
        "completed_actions": [journey["next_action"]] if ready else [],
        "comparison_ready": ready,
        "blockers": list(journey["blockers"]),
        "comparison_request": {
            "application_id": journey["application_id"],
            "amount": journey["reason"]["desired_amount"],
            "currency": context["loan_application"]["currency"],
            "desired_term_months": journey["reason"]["desired_term_months"],
            "monthly_budget": context["loan_application"]["monthly_budget"],
            "preferences": copy.deepcopy(journey["reason"]["preferences"]),
        },
    }


def comparison_plan(context, onboarding):
    request = onboarding["comparison_request"]
    common = {"customer_id": onboarding["customer_id"],
              "application_id": request["application_id"],
              "consumed_onboarding_status": onboarding["status"]}
    if not onboarding["comparison_ready"]:
        return dict(common, status="blocked", blockers=list(onboarding["blockers"]),
                    side_by_side=[], ranking=[], excluded=[], recommended_product_id=None)
    amount = Decimal(request["amount"])
    budget = Decimal(request["monthly_budget"])
    desired = Decimal(request["desired_term_months"])
    weights = {key: Decimal(value) for key, value in request["preferences"].items()}
    total_weight = sum(weights.values())
    rows, ranking, excluded = [], [], []
    for product in context["products"]:
        if not Decimal(product["minimum_amount"]) <= amount <= Decimal(product["maximum_amount"]):
            excluded.append({"product_id": product["id"], "reason": "requested_amount_outside_range"})
            continue
        apr, fee, term = (Decimal(product["apr_percent"]), Decimal(product["monthly_fee"]),
                          Decimal(product["term_months"]))
        payment = amount / term + amount * apr / 1200 + fee
        cost_index = apr + fee * 12 * 100 / amount
        components = {
            "cost": (100 / (1 + cost_index), "100 / (1 + apr_percent + monthly_fee * 1200 / amount)",
                     {"apr_percent": str(apr), "monthly_fee": str(fee), "amount": str(amount)}),
            "affordability": (100 * min(Decimal(1), budget / payment),
                              "100 * min(1, monthly_budget / estimated_payment)",
                              {"monthly_budget": str(budget), "estimated_payment": str(payment)}),
            "term": (100 * min(term / desired, desired / term),
                     "100 * min(term_months / desired_months, desired_months / term_months)",
                     {"term_months": int(term), "desired_months": int(desired)}),
        }
        explanations = {}
        score = Decimal("0")
        for key, (value, formula, inputs) in components.items():
            contribution = Decimal(str(rounded(value * weights[key] / total_weight)))
            score += contribution
            explanations[key] = {
                "value": rounded(value), "formula": formula, "inputs": inputs,
                "raw_weight": str(weights[key]), "weight_total": str(total_weight),
                "normalized_weight": rounded(weights[key] / total_weight),
                "weighted_contribution": float(contribution),
            }
        rows.append({
            "product_id": product["id"], "name": product["name"],
            "currency": product["currency"], "apr_percent": float(apr), "monthly_fee": float(fee),
            "term_months": int(term), "estimated_monthly_payment": rounded(payment, 2),
            "within_monthly_budget": payment <= budget,
            "payment_explanation": {
                "formula": "amount / term_months + amount * apr_percent / 1200 + monthly_fee",
                "method": "flat-interest illustrative estimate; not an amortized quote",
                "amount": str(amount), "unrounded_estimate": str(payment)},
        })
        ranking.append({"product_id": product["id"], "score": float(score),
                        "score_explanation": {"components": explanations,
                                              "aggregation": "sum rounded weighted contributions"}})
    ranking.sort(key=lambda item: (-item["score"], item["product_id"]))
    rows.sort(key=lambda item: item["product_id"])
    excluded.sort(key=lambda item: item["product_id"])
    return dict(common, status="compared" if ranking else "no_eligible_products", blockers=[],
                side_by_side=rows, ranking=ranking, excluded=excluded,
                recommended_product_id=ranking[0]["product_id"] if ranking else None,
                disclaimer="Synthetic preference ranking, not credit scoring, approval, financial "
                           "advice, or compliance certification. Budget is supplied, not inferred "
                           "from transaction history. Unaffordable options remain labeled.")


def journey_stage(payload):
    context = Validator.input(payload)
    output = {"schema_version": 1, "synthetic": True, "status": "ok", "stage": "journey",
              "context": context, "journey": plan(context)}
    return Validator.envelope(output, "journey")


def onboarding_stage(previous):
    Validator.envelope(previous, "journey")
    output = copy.deepcopy(previous)
    output["stage"] = "onboarding"
    output["onboarding"] = onboard_plan(output["context"], output["journey"])
    return Validator.envelope(output, "onboarding")


def comparison_stage(previous):
    Validator.envelope(previous, "onboarding")
    output = copy.deepcopy(previous)
    output["stage"] = "comparison"
    output["comparison"] = comparison_plan(output["context"], output["onboarding"])
    return Validator.envelope(output, "comparison")


def run_pipeline(payload):
    return comparison_stage(onboarding_stage(journey_stage(payload)))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with open(args[0], "r", encoding="utf-8-sig") as handle:
            raw = handle.read(2_000_001)
        require(len(raw) <= 2_000_000, "input exceeds size limit")
        result = run_pipeline(parse_json(raw))
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (ValidationError, OSError, UnicodeError, KeyError, TypeError, ValueError,
            ArithmeticError, RecursionError):
        # Do not echo file paths, input values, cards, or parser exception payloads.
        print(json.dumps({"status": "error", "error": "Input validation or file access failed."}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
