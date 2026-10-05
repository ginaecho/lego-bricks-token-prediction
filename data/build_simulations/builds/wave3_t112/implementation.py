"""Synthetic financial-services reference pipeline; Python standard library only.

All stages use the same versioned envelope and validate earlier deterministic
results before consuming them. This is not compliance certification, financial
advice, a credit decision, or a substitute for real KYC/AML screening.
"""

import copy
import csv
import io
import json
import re
import sys
from datetime import date
from decimal import Decimal, InvalidOperation


VERSION = "1.0"
STAGES = ("input", "sentiment", "adaptive", "review", "faq")
FLAGS = {"aml_alert", "sanctions_match", "pep_review"}
SEVERITIES = {"low": 20, "medium": 40, "high": 60, "critical": 80}
LEXICON = {
    "excellent": 2, "helpful": 1, "happy": 1, "thanks": 1, "good": 1,
    "bad": -1, "confusing": -1, "delayed": -1, "angry": -2,
    "fraud": -3, "unauthorized": -3, "terrible": -2,
}
FIELDS = {
    "identity_document": ("legal_name", "document_reference"),
    "address_proof": ("address",),
    "loan_application": ("requested_amount", "currency", "purpose"),
    "income_proof": ("annual_income",),
    "source_of_funds": ("source",),
    "manual_screening_review": ("reviewer", "outcome"),
    "incident_report": ("incident_reference",),
}
PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
STOP = {"a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is",
        "it", "my", "of", "on", "the", "to", "what", "with"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, label):
    require(isinstance(value, dict) and set(value) == set(required),
            label + ": unexpected or missing fields")


def text(value, label, maximum=2000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            label + ": expected nonempty bounded text")
    return value


def sequence(value, label, maximum=200):
    require(isinstance(value, list) and len(value) <= maximum,
            label + ": expected bounded list")
    return value


def identifier(value, label):
    text(value, label, 80)
    require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", value) is not None,
            label + ": invalid identifier")
    return value


def unique(items, label):
    ids = [item["id"] for item in items]
    require(len(ids) == len(set(ids)), label + ": duplicate IDs")


def same_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            same_value(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_value(left, right) for left, right in zip(actual, expected))
    return actual == expected


def money(value, label):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            label + ": invalid amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(label + ": invalid amount") from None
    require(amount.is_finite() and 0 < amount <= Decimal("1000000000"),
            label + ": amount must be finite, positive, and bounded")
    require(amount == amount.quantize(Decimal("0.01")),
            label + ": at most two fractional digits")
    return format(amount, ".2f")


def day(value, label):
    text(value, label, 10)
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
            label + ": expected ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValidationError(label + ": invalid date") from None


def flags(value):
    sequence(value, "screening_flags", 3)
    require(all(isinstance(flag, str) and flag in FLAGS for flag in value),
            "screening_flags: unsupported flag")
    require(len(value) == len(set(value)), "screening_flags: duplicate flag")
    return sorted(value)


def redact(value):
    if isinstance(value, str):
        return PAN.sub(lambda m: "****" + re.sub(r"\D", "", m.group())[-4:], value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def tokens(value):
    return [word for word in re.findall(r"[a-z]+", value.lower()) if word not in STOP]


def normalize_ledger(ledger):
    keys(ledger, ("format", "data"), "ledger")
    require(isinstance(ledger["format"], str), "ledger: invalid format")
    columns = ("id", "amount", "currency", "account", "counterparty_iban",
               "card_number", "screening_flags")
    if ledger["format"] == "json":
        rows = copy.deepcopy(sequence(ledger["data"], "transactions"))
    elif ledger["format"] == "csv":
        text(ledger["data"], "CSV ledger", 100000)
        try:
            reader = csv.DictReader(io.StringIO(ledger["data"]), strict=True)
            require(reader.fieldnames == list(columns), "CSV ledger: invalid header")
            rows = list(reader)
        except csv.Error:
            raise ValidationError("CSV ledger: malformed CSV") from None
        for row in rows:
            require(set(row) == set(columns) and all(v is not None for v in row.values()),
                    "CSV ledger: malformed row")
            row["screening_flags"] = row["screening_flags"].split("|") if row["screening_flags"] else []
    elif ledger["format"] == "iso20022":
        payload = ledger["data"]
        keys(payload, ("MsgId", "PmtInf"), "ISO-style payload")
        identifier(payload["MsgId"], "MsgId")
        rows = []
        for payment in sequence(payload["PmtInf"], "PmtInf"):
            keys(payment, ("DbtrAcct", "CdtTrfTxInf"), "PmtInf item")
            keys(payment["DbtrAcct"], ("Id",), "DbtrAcct")
            for tx in sequence(payment["CdtTrfTxInf"], "CdtTrfTxInf"):
                keys(tx, ("PmtId", "Amt", "CdtrAcct", "CardNumber", "ScreeningFlags"),
                     "ISO-style transaction")
                keys(tx["PmtId"], ("EndToEndId",), "PmtId")
                keys(tx["Amt"], ("InstdAmt",), "Amt")
                keys(tx["Amt"]["InstdAmt"], ("Value", "Ccy"), "InstdAmt")
                keys(tx["CdtrAcct"], ("IBAN",), "CdtrAcct")
                rows.append({
                    "id": tx["PmtId"]["EndToEndId"],
                    "amount": tx["Amt"]["InstdAmt"]["Value"],
                    "currency": tx["Amt"]["InstdAmt"]["Ccy"],
                    "account": payment["DbtrAcct"]["Id"],
                    "counterparty_iban": tx["CdtrAcct"]["IBAN"],
                    "card_number": tx["CardNumber"],
                    "screening_flags": tx["ScreeningFlags"],
                })
    else:
        raise ValidationError("ledger: unsupported format")
    sequence(rows, "transactions")
    for row in rows:
        keys(row, columns, "transaction")
        identifier(row["id"], "transaction.id")
        row["amount"] = money(row["amount"], "transaction.amount")
        require(row["currency"] in ("USD", "EUR", "GBP"), "transaction: unsupported currency")
        require(isinstance(row["account"], str) and
                re.fullmatch(r"SYN-ACC-\d{6}", row["account"]) is not None,
                "transaction: only synthetic account numbers are accepted")
        require(isinstance(row["counterparty_iban"], str) and
                re.fullmatch(r"ZZ\d{2}[A-Z0-9]{12,26}", row["counterparty_iban"]) is not None,
                "transaction: only fabricated ZZ IBANs are accepted")
        card = row["card_number"]
        require(isinstance(card, str) and
                (card == "" or re.fullmatch(r"\*{4}\d{4}", card) is not None or
                 re.fullmatch(r"(?:\d[ -]?){12,18}\d", card) is not None),
                "transaction: invalid card number")
        row["card_number"] = redact(card)
        row["screening_flags"] = flags(row["screening_flags"])
    unique(rows, "transactions")
    return {"format": "json", "data": rows}


def normalize_data(data):
    keys(data, ("fixture_label", "as_of", "customer", "ledger", "loan_application",
                "issues", "documents", "knowledge_base", "question"), "data")
    data = copy.deepcopy(data)
    require(data["fixture_label"] == "SYNTHETIC - NOT REAL CUSTOMER DATA",
            "data: synthetic fixture label required")
    day(data["as_of"], "as_of")
    customer = data["customer"]
    keys(customer, ("id", "name", "experience", "preferences", "kyc"), "customer")
    identifier(customer["id"], "customer.id")
    text(customer["name"], "customer.name", 150)
    require(customer["experience"] in ("beginner", "experienced"),
            "customer: unsupported experience")
    keys(customer["preferences"], ("format", "pace"), "preferences")
    require(customer["preferences"]["format"] in ("checklist", "narrative"),
            "preferences: unsupported format")
    require(customer["preferences"]["pace"] in ("guided", "fast"),
            "preferences: unsupported pace")
    kyc = customer["kyc"]
    keys(kyc, ("identity_verified", "address_verified", "screening_flags"), "kyc")
    require(type(kyc["identity_verified"]) is bool and type(kyc["address_verified"]) is bool,
            "kyc: verification fields must be boolean")
    kyc["screening_flags"] = flags(kyc["screening_flags"])
    data["ledger"] = normalize_ledger(data["ledger"])
    loan = data["loan_application"]
    keys(loan, ("id", "customer_id", "requested_amount", "currency", "purpose"), "loan")
    identifier(loan["id"], "loan.id")
    require(loan["customer_id"] == customer["id"], "loan: customer mismatch")
    loan["requested_amount"] = money(loan["requested_amount"], "loan.requested_amount")
    require(loan["currency"] in ("USD", "EUR", "GBP"), "loan: unsupported currency")
    text(loan["purpose"], "loan.purpose", 500)
    for issue in sequence(data["issues"], "issues", 100):
        keys(issue, ("id", "text", "severity"), "issue")
        identifier(issue["id"], "issue.id")
        text(issue["text"], "issue.text")
        require(isinstance(issue["severity"], str) and issue["severity"] in SEVERITIES,
                "issue: unsupported severity")
    unique(data["issues"], "issues")
    for doc in sequence(data["documents"], "documents", 100):
        keys(doc, ("id", "type", "customer_id", "loan_id", "status", "issued_on",
                   "expires_on", "fields"), "document")
        identifier(doc["id"], "document.id")
        require(isinstance(doc["type"], str) and doc["type"] in FIELDS,
                "document: unsupported type")
        identifier(doc["customer_id"], "document.customer_id")
        identifier(doc["loan_id"], "document.loan_id")
        require(doc["status"] in ("draft", "final"), "document: unsupported status")
        issued = day(doc["issued_on"], "document.issued_on")
        if doc["expires_on"] is not None:
            require(day(doc["expires_on"], "document.expires_on") >= issued,
                    "document: expiry precedes issuance")
        require(isinstance(doc["fields"], dict) and
                set(doc["fields"]).issubset(FIELDS[doc["type"]]),
                "document: unsupported evidence fields")
        for field, value in doc["fields"].items():
            if field in ("requested_amount", "annual_income"):
                doc["fields"][field] = money(value, "document amount")
            else:
                require(isinstance(value, str) and len(value) <= 2000,
                        "document: invalid evidence field")
    unique(data["documents"], "documents")
    for entry in sequence(data["knowledge_base"], "knowledge_base", 100):
        keys(entry, ("id", "title", "text", "tags", "applies_to"), "knowledge entry")
        identifier(entry["id"], "knowledge.id")
        text(entry["title"], "knowledge.title", 200)
        text(entry["text"], "knowledge.text")
        for tag in sequence(entry["tags"], "knowledge.tags", 20):
            text(tag, "knowledge.tag", 80)
        scopes = sequence(entry["applies_to"], "knowledge.applies_to", 4)
        require(bool(scopes) and all(isinstance(scope, str) and scope in
                ("general", "review_gap", "screening_hold", "ready") for scope in scopes),
                "knowledge: invalid applicability")
    unique(data["knowledge_base"], "knowledge_base")
    text(data["question"], "question")
    return redact(data)


def sentiment(data, previous):
    scored = []
    for issue in data["issues"]:
        contributions = [{"token": word, "weight": LEXICON[word]}
                         for word in tokens(issue["text"]) if word in LEXICON]
        raw = sum(item["weight"] for item in contributions)
        score = max(-5, min(5, raw))
        fraud_terms = sorted(set(tokens(issue["text"])) & {"fraud", "unauthorized"})
        severity = "critical" if fraud_terms else issue["severity"]
        base = SEVERITIES[severity]
        boost = min(20, max(0, -score) * 4)
        scored.append({
            "id": issue["id"],
            "sentiment": {"score": score,
                          "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
                          "explanation": {"contributions": contributions, "raw_sum": raw,
                                          "rule": "clip(sum(token weights), -5, 5)"}},
            "severity": severity,
            "severity_explanation": {"reported": issue["severity"],
                                     "critical_trigger_terms": fraud_terms},
            "priority": {"score": base + boost,
                         "explanation": {"severity_base": base, "negative_sentiment_boost": boost,
                                         "rule": "severity_base + min(20, max(0, -sentiment) * 4)"}},
        })
    scored.sort(key=lambda item: (-item["priority"]["score"], item["id"]))
    all_flags = set(data["customer"]["kyc"]["screening_flags"])
    sources = [{"source": "customer:" + data["customer"]["id"], "flag": flag}
               for flag in sorted(all_flags)]
    for tx in data["ledger"]["data"]:
        for flag in tx["screening_flags"]:
            all_flags.add(flag)
            sources.append({"source": "transaction:" + tx["id"], "flag": flag})
    return {"issues": scored, "screening_flags": sorted(all_flags),
            "flag_sources": sources, "urgent_issue_ids": [
                item["id"] for item in scored if item["severity"] == "critical"],
            "limitations": "Lexical scoring does not infer intent, sarcasm, or creditworthiness."}


def adaptive(data, previous):
    insights = previous["sentiment"]
    customer = data["customer"]
    kyc = customer["kyc"]
    modules = []

    def module(name, prerequisites, satisfied, reason, needs_review=False):
        completed = {item["id"] for item in modules if item["status"] == "complete"}
        blocked_by = [name for name in prerequisites if name not in completed]
        status = ("blocked" if blocked_by else "needs_review" if needs_review
                  else "complete" if satisfied else "ready")
        modules.append({"id": name, "prerequisites": prerequisites, "blocked_by": blocked_by,
                        "status": status, "explanation": reason})

    module("profile", [], True, "Validated synthetic customer and linked loan profile.")
    if customer["experience"] == "beginner" or customer["preferences"]["pace"] == "guided":
        module("orientation", ["profile"], False,
               "Optional orientation selected by experience or guided-pace preference.")
    module("identity", ["profile"], kyc["identity_verified"] and kyc["address_verified"],
           "Both identity and address assertions are prerequisites; documents are reviewed next.")
    module("screening", ["identity"], not insights["screening_flags"],
           "Supplied KYC/AML flags require human review; absence of flags is not real screening.",
           bool(insights["screening_flags"]))
    module("ledger", ["screening"], True, "Validated ledger follows the screening prerequisite.")
    module("loan_preparation", ["ledger"], False, "Prepare linked loan evidence after ledger review.")
    requirements = [{"id": "req_" + kind, "document_type": kind,
                     "reason": "Baseline customer/KYC and loan evidence."}
                    for kind in ("identity_document", "address_proof", "loan_application", "income_proof")]
    if insights["screening_flags"]:
        for kind in ("source_of_funds", "manual_screening_review"):
            requirements.append({"id": "req_" + kind, "document_type": kind,
                                 "reason": "Screening flags: " + ", ".join(insights["screening_flags"])})
    if insights["urgent_issue_ids"]:
        requirements.append({"id": "req_incident_report", "document_type": "incident_report",
                             "reason": "Critical issues: " + ", ".join(insights["urgent_issue_ids"])})
    return {"format": customer["preferences"]["format"],
            "pace": customer["preferences"]["pace"], "modules": modules,
            "requirements": requirements, "focus_issue_ids": [i["id"] for i in insights["issues"]],
            "urgent_issue_ids": insights["urgent_issue_ids"],
            "screening_flags": insights["screening_flags"],
            "explanation": "Preferences change presentation, never mandatory evidence or screening gates."}


def review(data, previous):
    plan = previous["adaptive"]
    checks = []
    for requirement in plan["requirements"]:
        kind = requirement["document_type"]
        candidates = sorted((doc for doc in data["documents"] if doc["type"] == kind),
                            key=lambda doc: doc["id"])
        evaluations = []
        for doc in candidates:
            reasons = []
            if doc["customer_id"] != data["customer"]["id"] or doc["loan_id"] != data["loan_application"]["id"]:
                reasons.append("owner_or_loan_mismatch")
            if doc["status"] != "final":
                reasons.append("not_final")
            if doc["issued_on"] > data["as_of"]:
                reasons.append("future_issued")
            if doc["expires_on"] is not None and doc["expires_on"] < data["as_of"]:
                reasons.append("expired")
            for field in FIELDS[kind]:
                if not str(doc["fields"].get(field, "")).strip():
                    reasons.append("missing_field:" + field)
            if kind == "identity_document" and doc["fields"].get("legal_name") != data["customer"]["name"]:
                reasons.append("legal_name_mismatch")
            if kind == "loan_application":
                for field in FIELDS[kind]:
                    if doc["fields"].get(field) != data["loan_application"][field]:
                        reasons.append("loan_field_mismatch:" + field)
            evaluations.append({"document_id": doc["id"], "reasons": reasons})
        evidence = [item["document_id"] for item in evaluations if not item["reasons"]]
        checks.append({"requirement_id": requirement["id"], "document_type": kind,
                       "requirement_reason": requirement["reason"],
                       "status": "satisfied" if evidence else "gap",
                       "evidence_ids": evidence, "evaluations": evaluations,
                       "gap_reason": None if evidence else "no_usable_evidence" if candidates else "missing_document"})
    gaps = [item["requirement_id"] for item in checks if item["status"] == "gap"]
    holds = list(plan["screening_flags"])
    if not data["customer"]["kyc"]["identity_verified"]:
        holds.append("identity_unverified")
    if not data["customer"]["kyc"]["address_verified"]:
        holds.append("address_unverified")
    return {"checks": checks, "gap_ids": gaps, "hold_reasons": holds,
            "screening_flags": plan["screening_flags"],
            "urgent_issue_ids": plan["urgent_issue_ids"],
            "status": "needs_action" if gaps or holds else "evidence_complete",
            "decision": "not_a_loan_decision",
            "disclaimer": "Demonstrative evidence checks only; no certification or compliance determination."}


def faq(data, previous):
    reviewed = previous["review"]
    scopes = {"general"}
    if reviewed["gap_ids"]:
        scopes.add("review_gap")
    if reviewed["hold_reasons"]:
        scopes.add("screening_hold")
    if reviewed["status"] == "evidence_complete":
        scopes.add("ready")
    query = set(tokens(data["question"]))
    context_words = set(tokens(" ".join(reviewed["gap_ids"] + reviewed["hold_reasons"])))
    ranked = []
    for entry in data["knowledge_base"]:
        if not scopes.intersection(entry["applies_to"]):
            continue
        title_words = sorted(query & set(tokens(entry["title"])))
        body = set(tokens(entry["text"] + " " + " ".join(entry["tags"])))
        body_words = sorted(query & body)
        if not title_words and not body_words:
            continue
        context_matches = sorted(context_words & body)
        score = 2 * len(title_words) + len(body_words) + len(context_matches)
        ranked.append({"id": entry["id"], "score": score,
                       "explanation": {"title_matches": title_words, "body_tag_matches": body_words,
                                       "review_context_matches": context_matches,
                                       "rule": "2 * title matches + body/tag matches + review-context matches"}})
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    by_id = {entry["id"]: entry for entry in data["knowledge_base"]}
    selected = ranked[:2]
    citations = [{"knowledge_id": item["id"], "excerpt": by_id[item["id"]]["text"]}
                 for item in selected]
    return {"status": "answered" if selected else "abstained",
            "answer": "\n\n".join(item["excerpt"] for item in citations) if selected else None,
            "abstention_reason": None if selected else "No applicable knowledge entry overlaps the question.",
            "citations": citations, "retrieval": ranked,
            "case_context": {"review_status": reviewed["status"], "gap_ids": reviewed["gap_ids"],
                             "hold_reasons": reviewed["hold_reasons"],
                             "urgent_issue_ids": reviewed["urgent_issue_ids"]},
            "limitations": "Exact excerpts from supplied synthetic knowledge; lexical matches may be incomplete."}


BUILDERS = {"sentiment": sentiment, "adaptive": adaptive, "review": review, "faq": faq}


def validate_state(state, expected_stage):
    keys(state, ("schema_version", "synthetic", "status", "stage", "data", "results"), "envelope")
    require(state["schema_version"] == VERSION, "envelope: unsupported schema_version")
    require(state["synthetic"] is True, "envelope: synthetic must be true")
    require(state["status"] == "ok" and state["stage"] == expected_stage,
            "envelope: wrong status or stage")
    require(expected_stage in STAGES, "envelope: unsupported stage")
    data = normalize_data(state["data"])
    count = STAGES.index(expected_stage)
    keys(state["results"], STAGES[1:count + 1], "results")
    if count:
        require(data == state["data"], "stage data: noncanonical or unmasked data")
    validated = {}
    # Recompute this bounded reference's deterministic outputs to reject stale or
    # tampered handoffs, not merely well-shaped but inconsistent stage results.
    for name in STAGES[1:count + 1]:
        expected = BUILDERS[name](data, validated)
        require(same_value(state["results"][name], expected), name + ": inconsistent stage output")
        validated[name] = state["results"][name]
    return data


def advance(state, next_stage):
    require(isinstance(next_stage, str) and next_stage in BUILDERS, "transition: invalid target")
    expected_stage = STAGES[STAGES.index(next_stage) - 1]
    data = validate_state(state, expected_stage)
    result = copy.deepcopy(state)
    result["data"] = data
    result["results"][next_stage] = BUILDERS[next_stage](data, result["results"])
    result["stage"] = next_stage
    validate_state(result, next_stage)
    return result


def run_pipeline(payload):
    validate_state(payload, "input")
    state = payload
    for name in STAGES[1:]:
        state = advance(state, name)
    return state


def reject_constant(value):
    raise ValidationError("JSON: nonfinite numeric constant")


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON: duplicate object key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        try:
            with open(args[0], "r", encoding="utf-8") as handle:
                raw = handle.read(1000001)
        except (OSError, UnicodeError):
            raise ValidationError("input: unable to read UTF-8 file") from None
        require(len(raw) <= 1000000, "input: file exceeds size limit")
        try:
            payload = json.loads(raw, parse_constant=reject_constant,
                                 object_pairs_hook=reject_duplicates)
        except ValidationError:
            raise
        except (ValueError, RecursionError, OverflowError):
            raise ValidationError("input: invalid JSON") from None
        result = run_pipeline(payload)
    except (ValidationError, RecursionError) as error:
        message = str(error) if isinstance(error, ValidationError) else "input: nesting too deep"
        print(json.dumps({"schema_version": VERSION, "status": "error",
                          "error": {"code": "validation_error", "message": message}}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
