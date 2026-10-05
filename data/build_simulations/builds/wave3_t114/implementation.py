"""Synthetic, deterministic telecom FAQ CLI; standard library only.

Run: python -B implementation.py example_input.json
An optional answerer callable receives only a grounded candidate, not records.
These privacy/residency rules are demonstrations, not legal certification.
"""

import csv
import io
import json
import math
import re
import sys
from datetime import datetime

BUILD_ID = "wave3_t114_single_faq__ind-telecom"
STOP = {"a", "an", "the", "is", "are", "my", "why", "how", "do", "does",
        "i", "can", "please", "what", "to", "of", "it"}
CDR_FIELDS = ["record_id", "account_id", "timestamp", "call_minutes",
              "data_mb", "phone_number", "imei", "residency", "synthetic"]


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required):
    require(isinstance(value, dict) and set(value) == set(required),
            "Invalid record fields.")


def text(value, limit=2000):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit,
            "Invalid text value.")


def timestamp(value):
    text(value, 50)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, "Timestamp requires a timezone.")
    except ValueError:
        raise ValidationError("Invalid timestamp.") from None


def record(value, region):
    require(value["residency"] == region, "Record violates data residency.")
    require(value["synthetic"] is True, "Only labeled synthetic records allowed.")


def tokens(value):
    return set(re.findall(r"[a-z]+", value.lower())) - STOP


def envelope(region=None, status="abstained", answer=None, sources=None,
             reason=None, counts=None):
    return {"status": status, "build_id": BUILD_ID, "synthetic": True,
            "residency": region, "answer": answer, "source_ids": sources or [],
            "reason": reason, "context_counts": counts or {}}


def validate(value, direction="input"):
    """One shared validation boundary for all records and CLI result envelopes."""
    if direction == "output":
        fields(value, ["status", "build_id", "synthetic", "residency", "answer",
                       "source_ids", "reason", "context_counts"])
        require(value["status"] in {"answered", "abstained", "error"},
                "Invalid result status.")
        require(value["build_id"] == BUILD_ID and value["synthetic"] is True,
                "Invalid result metadata.")
        require(value["residency"] in {None, "EU", "US"},
                "Invalid result residency.")
        require(isinstance(value["source_ids"], list) and
                all(isinstance(x, str) for x in value["source_ids"]),
                "Invalid source identifiers.")
        require(isinstance(value["context_counts"], dict) and
                all(type(x) is int and x >= 0
                    for x in value["context_counts"].values()),
                "Invalid context counts.")
        if value["status"] == "answered":
            text(value["answer"])
            require(bool(value["source_ids"]) and value["reason"] is None,
                    "Answers require grounding.")
        else:
            require(value["answer"] is None and not value["source_ids"],
                    "Nonanswers cannot assert facts.")
            text(value["reason"])
        return value

    require(direction == "input", "Invalid validation direction.")
    fields(value, ["synthetic", "residency", "authenticated_account_id",
                   "subscriber_account", "network_fault_tickets", "cdr_csv",
                   "support_chat_transcripts", "knowledge_base", "question"])
    region = value["residency"]
    require(isinstance(region, str) and region in {"EU", "US"},
            "Unsupported residency.")
    require(value["synthetic"] is True, "Synthetic fixture label required.")
    account = value["subscriber_account"]
    fields(account, ["account_id", "display_name", "phone_number", "imei",
                     "support_processing_allowed", "residency", "synthetic"])
    record(account, region)
    for key in ("account_id", "display_name"):
        text(account[key], 100)
    require(account["support_processing_allowed"] is True,
            "Subscriber support processing is not permitted.")
    require(value["authenticated_account_id"] == account["account_id"],
            "Subscriber authorization mismatch.")
    require(isinstance(account["phone_number"], str) and
            re.fullmatch(r"\+1-202-555-01\d{2}", account["phone_number"]),
            "Use fictitious reserved phone numbers.")
    require(isinstance(account["imei"], str) and
            re.fullmatch(r"000000\d{9}", account["imei"]),
            "Use synthetic zero-prefix IMEIs.")
    question = value["question"]
    fields(question, ["text", "residency", "synthetic"])
    record(question, region)
    text(question["text"], 1000)
    for key in ("network_fault_tickets", "support_chat_transcripts",
                "knowledge_base"):
        require(isinstance(value[key], list) and len(value[key]) <= 100,
                "Invalid record collection.")
    identifiers = set()

    def unique(identifier):
        text(identifier, 100)
        require(identifier not in identifiers, "Duplicate record identifier.")
        identifiers.add(identifier)

    for ticket in value["network_fault_tickets"]:
        fields(ticket, ["ticket_id", "account_id", "category", "status",
                        "billing_adjustment", "residency", "synthetic"])
        record(ticket, region)
        unique(ticket["ticket_id"])
        require(ticket["account_id"] == account["account_id"],
                "Cross-subscriber record rejected.")
        require(isinstance(ticket["category"], str) and ticket["category"] in
                {"network", "billing"}, "Invalid ticket category.")
        require(isinstance(ticket["status"], str) and ticket["status"] in
                {"open", "resolved"}, "Invalid ticket status.")
        adjustment = ticket["billing_adjustment"]
        if adjustment is not None:
            fields(adjustment, ["amount", "currency", "reason", "residency",
                                "synthetic"])
            record(adjustment, region)
            require(type(adjustment["amount"]) in (int, float) and
                    math.isfinite(adjustment["amount"]), "Invalid adjustment.")
            require(adjustment["currency"] in ("USD", "EUR"),
                    "Invalid adjustment currency.")
            text(adjustment["reason"], 500)
    text(value["cdr_csv"], 100000)
    try:
        reader = csv.DictReader(io.StringIO(value["cdr_csv"]), strict=True)
        require(reader.fieldnames == CDR_FIELDS, "Invalid CDR header.")
        usage = list(reader)
    except csv.Error:
        raise ValidationError("Invalid CDR CSV.") from None
    require(len(usage) <= 1000, "Too many CDR records.")
    for row in usage:
        fields(row, CDR_FIELDS)
        require(all(isinstance(x, str) for x in row.values()),
                "Malformed CDR row.")
        require(row["residency"] == region and row["synthetic"] == "true",
                "CDR residency or synthetic label invalid.")
        require(row["account_id"] == account["account_id"] and
                row["phone_number"] == account["phone_number"] and
                row["imei"] == account["imei"], "CDR subscriber mismatch.")
        unique(row["record_id"])
        timestamp(row["timestamp"])
        for key in ("call_minutes", "data_mb"):
            try:
                number = float(row[key])
            except ValueError:
                raise ValidationError("Invalid CDR usage volume.") from None
            require(math.isfinite(number) and number >= 0,
                    "Usage volumes must be finite and nonnegative.")
    for chat in value["support_chat_transcripts"]:
        fields(chat, ["transcript_id", "account_id", "messages", "residency",
                      "synthetic"])
        record(chat, region)
        unique(chat["transcript_id"])
        require(chat["account_id"] == account["account_id"],
                "Cross-subscriber transcript rejected.")
        require(isinstance(chat["messages"], list) and len(chat["messages"]) <= 100,
                "Invalid chat messages.")
        for message in chat["messages"]:
            fields(message, ["role", "text", "timestamp", "residency", "synthetic"])
            record(message, region)
            require(message["role"] in ("subscriber", "agent"), "Invalid chat role.")
            text(message["text"])
            timestamp(message["timestamp"])
    private_values = [account[k].lower() for k in
                      ("account_id", "display_name", "phone_number", "imei")]
    for article in value["knowledge_base"]:
        fields(article, ["article_id", "title", "keywords", "answer",
                         "visibility", "residency", "synthetic"])
        record(article, region)
        unique(article["article_id"])
        require(article["visibility"] == "public", "Only public KB articles allowed.")
        require(isinstance(article["keywords"], list) and
                1 <= len(article["keywords"]) <= 30, "Invalid KB keywords.")
        for item in [article["article_id"], article["title"], article["answer"],
                     *article["keywords"]]:
            text(item)
            require(not any(secret in item.lower() for secret in private_values)
                    and not re.search(r"@|\d{6,}|\+?\d[\d ()-]{8,}\d", item),
                    "KB must not contain subscriber identifiers.")
    return usage


def run(payload, answerer=None):
    usage = validate(payload)
    counts = {"usage_records": len(usage),
              "network_fault_tickets": len(payload["network_fault_tickets"]),
              "support_chat_transcripts": len(payload["support_chat_transcripts"])}
    region = payload["residency"]
    query = tokens(payload["question"]["text"])
    ranked = []
    for article in payload["knowledge_base"]:
        vocabulary = tokens(" ".join(article["keywords"]) + " " + article["title"])
        overlap = len(query & vocabulary)
        if overlap >= 2 and overlap / max(len(query), 1) >= 0.5:
            ranked.append((overlap, article))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["article_id"]))
    if not ranked:
        result = envelope(region, reason="No sufficiently grounded knowledge-base match.",
                          counts=counts)
    elif len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        result = envelope(region, reason="Ambiguous knowledge-base matches.",
                          counts=counts)
    else:
        article = ranked[0][1]
        result = envelope(region, status="answered", answer=article["answer"],
                          sources=[article["article_id"]], counts=counts)
        if answerer is not None:
            # The injected callable can only reproduce the approved public answer.
            candidate = {"answer": result["answer"],
                         "source_ids": list(result["source_ids"])}
            try:
                generated = answerer(dict(candidate, source_ids=list(candidate["source_ids"])))
                require(generated == candidate, "Ungrounded injected answer.")
            except Exception:
                result = envelope(region, reason="Injected answer failed grounding validation.",
                                  counts=counts)
    return validate(result, "output")


def reject_constant(_):
    raise ValidationError("Nonfinite JSON numbers are forbidden.")


def distinct_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON keys are forbidden.")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Expected one input JSON file.")
        with open(argv[0], encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=reject_constant,
                                object_pairs_hook=distinct_object)
        result = run(payload)
        code = 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError):
        # Never echo exception details, file paths, chat text, or personal data.
        result = validate(envelope(status="error",
                                   reason="Input validation or file access failed."),
                          "output")
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
