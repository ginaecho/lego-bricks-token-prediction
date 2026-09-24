"""Synthetic telecom feedback reference. Python standard library; no network calls."""
import csv
import io
import json
import re
import sys
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(record, required, optional=()):
    require(isinstance(record, dict), "Expected an object")
    require(set(required) <= set(record) <= set(required) | set(optional),
            "Missing or unexpected fields")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def identifier(value, prefix):
    require(isinstance(value, str) and re.fullmatch(prefix + r"_[0-9]{3,12}", value),
            "Invalid opaque identifier")


def number(value, nonnegative=False):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            "Invalid numeric value")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError("Invalid numeric value") from None
    require(result.is_finite() and (not nonnegative or result >= 0),
            "Invalid numeric range")
    return result


def validate(document, output=False):
    """Single validation boundary for input entities and published output."""
    require(isinstance(document, dict), "Expected a JSON object")
    if output:
        fields(document, ("status", "synthetic", "residency", "summary", "themes"))
        require(document["status"] == "ok" and document["synthetic"] is True,
                "Invalid result envelope")
        for theme in document["themes"]:
            require(theme["residency"] == document["residency"], "Output residency mismatch")
            require(theme["count"] == len(theme["evidence"]), "Invalid evidence count")
            for item in theme["evidence"]:
                require(item["residency"] == document["residency"], "Output residency mismatch")
                require(bool(item["source_feedback_ids"]), "Missing provenance")
        return document
    fields(document, ("synthetic", "residency", "subscribers", "tickets", "usage_csv", "feedback"))
    require(document["synthetic"] is True, "Only explicitly synthetic fixtures are accepted")
    require(document["residency"] in ("EU", "US"), "Unsupported residency")
    for collection in ("subscribers", "tickets", "feedback"):
        require(isinstance(document[collection], list), "Expected entity array")
    require(isinstance(document["usage_csv"], str), "Expected CSV string")
    region = document["residency"]

    def indexed(records, prefix, required, optional=()):
        result = {}
        for record in records:
            fields(record, required, optional)
            identifier(record["id"], prefix)
            require(record["id"] not in result, "Duplicate entity identifier")
            require(record["residency"] == region, "Record residency mismatch")
            result[record["id"]] = record
        return result

    accounts = indexed(document["subscribers"], "sub",
                       ("id", "residency", "phone", "imei", "name", "email",
                        "processing_basis", "analytics_allowed"))
    for account in accounts.values():
        require(all(text(account[k]) for k in ("phone", "imei", "name", "email")),
                "Invalid subscriber fields")
        require(re.fullmatch(r"\+1-202-555-01[0-9]{2}", account["phone"]),
                "Use fictitious reserved-range phone numbers")
        require(re.fullmatch(r"SYNTHETIC-IMEI-[0-9]{6}", account["imei"]),
                "Use labeled fictitious IMEIs")
        require(account["email"].endswith("@example.invalid"), "Use synthetic email domain")
        require(account["processing_basis"] in ("consent", "legitimate_interest")
                and account["analytics_allowed"] is True,
                "Subscriber lacks allowed analytics processing")
    tickets = indexed(document["tickets"], "tkt",
                      ("id", "residency", "subscriber_id", "fault"), ("billing_adjustment",))
    for ticket in tickets.values():
        require(ticket["subscriber_id"] in accounts, "Unknown ticket subscriber")
        require(ticket["fault"] in ("network", "billing", "support", "usage"),
                "Unsupported fault type")
        if "billing_adjustment" in ticket:
            adjustment = ticket["billing_adjustment"]
            fields(adjustment, ("amount", "reason"))
            number(adjustment["amount"])
            require(text(adjustment["reason"]), "Billing adjustment requires a stated reason")
    columns = ["id", "residency", "subscriber_id", "ticket_id", "call_seconds", "data_mb"]
    try:
        reader = csv.DictReader(io.StringIO(document["usage_csv"]), strict=True)
        require(reader.fieldnames == columns, "Invalid call detail CSV header")
        usage = indexed(list(reader), "cdr", columns)
    except csv.Error:
        raise ValidationError("Malformed call detail CSV") from None
    for record in usage.values():
        require(record["subscriber_id"] in accounts and record["ticket_id"] in tickets,
                "Unknown usage relationship")
        require(tickets[record["ticket_id"]]["subscriber_id"] == record["subscriber_id"],
                "Usage account/ticket mismatch")
        number(record["call_seconds"], True)
        number(record["data_mb"], True)
    feedback = indexed(document["feedback"], "fb",
                       ("id", "residency", "subscriber_id", "ticket_id", "channel", "text"))
    for record in feedback.values():
        require(record["subscriber_id"] in accounts and record["ticket_id"] in tickets,
                "Unknown feedback relationship")
        require(tickets[record["ticket_id"]]["subscriber_id"] == record["subscriber_id"],
                "Feedback account/ticket mismatch")
        require(record["channel"] == "support_chat", "Expected support chat transcript")
        require(text(record["text"]) and len(record["text"]) <= 10000,
                "Transcript must contain 1–10000 characters")
    return accounts, tickets, usage, feedback


def redact(value, accounts):
    # Exact known subscriber values plus common unsolicited contact/device forms.
    private = {account[key] for account in accounts.values()
               for key in ("phone", "imei", "name", "email")}
    for item in sorted(private, key=lambda item: (-len(item), item)):
        value = re.sub(re.escape(item), "[REDACTED]", value, flags=re.IGNORECASE)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED]", value)
    value = re.sub(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)", "[REDACTED]", value)
    return " ".join(value.split())


THEMES = {
    "billing": r"\b(bill|billing|charge|charged|refund|invoice)\b",
    "connectivity": r"\b(network|signal|outage|dropped|disconnect|coverage)\b",
    "support": r"\b(agent|support|wait|waiting|resolved)\b",
    "usage": r"\b(data|roaming|allowance|usage|speed)\b",
}


def analyze(document):
    accounts, tickets, usage, feedback = validate(document)
    groups = {}
    for record in feedback.values():
        cleaned = redact(record["text"], accounts)
        key = (record["subscriber_id"], record["ticket_id"], cleaned.casefold())
        if key not in groups:
            groups[key] = {"residency": document["residency"],
                           "subscriber_id": record["subscriber_id"],
                           "ticket_id": record["ticket_id"], "source_feedback_ids": [],
                           "usage_record_ids": sorted(
                               row["id"] for row in usage.values()
                               if row["ticket_id"] == record["ticket_id"]),
                           "excerpt": cleaned[:240], "_text": cleaned}
        groups[key]["source_feedback_ids"].append(record["id"])
    themes = {}
    for group in groups.values():
        labels = [label for label, pattern in THEMES.items()
                  if re.search(pattern, group["_text"], re.IGNORECASE)] or ["other"]
        evidence = {key: value for key, value in group.items() if key != "_text"}
        evidence["source_feedback_ids"].sort()
        for label in labels:
            themes.setdefault(label, []).append(evidence)
    result = {
        "status": "ok", "synthetic": True, "residency": document["residency"],
        "summary": {"feedback_records": len(feedback), "unique_feedback": len(groups),
                    "duplicates_removed": len(feedback) - len(groups)},
        "themes": [{"theme": label, "residency": document["residency"],
                    "count": len(items),
                    "evidence": sorted(items, key=lambda item: item["source_feedback_ids"])}
                   for label, items in sorted(themes.items())],
    }
    return validate(result, output=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            document = json.load(stream)
        result = analyze(document)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError):
        # Never echo source text, paths, or validation values into error output.
        print(json.dumps({"status": "error", "message": "Input file or validation failed"}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
