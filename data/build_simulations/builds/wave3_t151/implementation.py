"""Synthetic public-service discovery. No eligibility or entitlement decisions."""

import json
import math
import re
import sys
from datetime import datetime, timezone


class ValidationError(ValueError):
    pass


ENTITIES = {
    "citizen_service_request": "Citizen service request",
    "benefits_application": "Benefits application",
    "policy_document": "Policy document",
}
TOPICS = {"housing", "transport", "family", "environment"}
PII = re.compile(
    r"[\w.+-]+@[\w.-]+\.\w+|\b\d{3}-\d{2}-\d{4}\b|"
    r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"
)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected):
    require(isinstance(value, dict), "An object is required.")
    require(set(value) == set(expected), "Missing or unsupported fields.")


def identifier(value, prefix):
    require(isinstance(value, str) and
            re.fullmatch(prefix + r"-[0-9]{4}", value) is not None,
            "Use a synthetic identifier in the required format.")


def text(value):
    require(isinstance(value, str) and 1 <= len(value) <= 2000,
            "Text must contain 1 to 2000 characters.")
    require(not PII.search(value), "Remove personal contact or identity numbers.")


def instant(value):
    require(isinstance(value, str), "Time must be an ISO 8601 string.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.tzinfo is not None, "Time must include a time zone.")
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise ValidationError("Use a valid ISO 8601 time with a time zone.") from None


def validate(data):
    """One input validation boundary; never quote rejected values in errors."""
    fields(data, {"schema_version", "synthetic", "as_of", "consent",
                  "persona", "items", "events", "limit"})
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Only schema version 1 is supported.")
    require(data["synthetic"] is True, "Only clearly marked synthetic data is allowed.")
    require(type(data["consent"]) is bool, "Consent must be true or false.")
    require(type(data["limit"]) is int and 1 <= data["limit"] <= 50,
            "Limit must be an integer from 1 to 50.")
    now = instant(data["as_of"])
    fields(data["persona"], {"id", "synthetic_address"})
    identifier(data["persona"]["id"], "persona")
    require(data["persona"]["synthetic_address"] == "NOT A REAL ADDRESS",
            "Use the non-traceable synthetic address placeholder.")
    require(isinstance(data["items"], list) and len(data["items"]) <= 500,
            "Items must be a list of at most 500 entries.")
    items = {}
    for item in data["items"]:
        fields(item, {"id", "entity", "topic", "public_priority", "form", "policy_text"})
        identifier(item["id"], "item")
        require(item["id"] not in items, "Item identifiers must be unique.")
        require(isinstance(item["entity"], str) and item["entity"] in ENTITIES,
                "Unsupported public-service entity.")
        require(isinstance(item["topic"], str) and item["topic"] in TOPICS,
                "Unsupported service topic.")
        require(type(item["public_priority"]) is int and
                0 <= item["public_priority"] <= 100,
                "Public priority must be an integer from 0 to 100.")
        if item["entity"] == "policy_document":
            require(item["form"] is None, "Policy documents cannot contain form data.")
            text(item["policy_text"])
        else:
            require(item["policy_text"] is None, "Forms cannot contain policy text.")
            fields(item["form"], {"case_number", "summary"})
            require(isinstance(item["form"]["case_number"], str) and
                    re.fullmatch(r"CASE-SYN-[0-9]{4}", item["form"]["case_number"]),
                    "Use an invented synthetic case number.")
            text(item["form"]["summary"])
        items[item["id"]] = item
    require(isinstance(data["events"], list) and len(data["events"]) <= 1000,
            "Events must be a list of at most 1000 entries.")
    seen = set()
    events = []
    for event in data["events"]:
        fields(event, {"id", "item_id", "action", "at"})
        identifier(event["id"], "event")
        identifier(event["item_id"], "item")
        require(event["id"] not in seen, "Event identifiers must be unique.")
        seen.add(event["id"])
        require(event["item_id"] in items, "Every event must reference a catalog item.")
        require(isinstance(event["action"], str) and event["action"] in {"browse", "purchase"},
                "Action must be browse or purchase.")
        when = instant(event["at"])
        require(when <= now, "Events cannot be later than the ranking time.")
        events.append((event, (now - when).total_seconds() / 86400))
    return items, events


def personalize(data):
    items, events = validate(data)
    affinities = dict.fromkeys(TOPICS, 0.0)
    if data["consent"]:
        for event, age in sorted(events, key=lambda pair: pair[0]["id"]):
            weight = 2.0 if event["action"] == "purchase" else 1.0
            affinities[items[event["item_id"]]["topic"]] += weight * 2.0 ** (-age / 30.0)
    # Keep absolute recency strength: a lone ancient event must not become strong again.
    total = math.fsum(affinities.values())
    cold = total < 1e-9
    reason = ("consent_not_given" if not data["consent"] else
              "no_recent_history" if cold else "recent_history")
    results = []
    for item in items.values():
        interest = 0.0 if cold else affinities[item["topic"]] / max(1.0, total)
        public = item["public_priority"] / 100.0
        score = public if cold else 0.2 * public + 0.8 * interest
        results.append({
            "item_id": item["id"],
            "entity": item["entity"],
            "label": ENTITIES[item["entity"]] + ": " + item["topic"],
            "score": round(score, 10),
            "explanation": {
                "plain_language": (
                    "Shown by public priority. Your browsing history was not used."
                    if cold else
                    "Shown using public priority and your recent interest in this topic."
                ),
                "public_priority": public,
                "topic_interest": round(interest, 10),
                "public_weight": 1.0 if cold else 0.2,
                "interest_weight": 0.0 if cold else 0.8,
            },
        })
    results.sort(key=lambda row: (-row["score"], row["item_id"]))
    return {
        "status": "ok", "schema_version": 1, "synthetic": True,
        "mode": "cold_start" if cold else "personalized",
        "reason": reason,
        "recommendations": results[:data["limit"]],
        "audit": {
            "algorithm": "recency-topic-v1",
            "as_of": data["as_of"],
            "half_life_days": 30,
            "action_weights": {"browse": 1, "purchase": 2},
            "formula": "interest = topic weight / max(1, total weight); score = public weight * priority + interest weight * interest",
            "tie_break": "item_id ascending after score rounded to 10 decimal places",
            "events_used": len(events) if data["consent"] else 0,
            "privacy": "No persona, address, case number, form text or policy text is returned.",
            "notice": "Discovery only. This does not decide benefit eligibility or service access.",
            "limits": "Synthetic demonstration, not a Privacy Act, GDPR, FOIA or Section 508 certification. Text screening is not a complete PII detector.",
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("JSON keys must be unique.")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("JSON numbers must be finite.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Provide exactly one input JSON file.")
        with open(argv[0], encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = personalize(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError):
        print(json.dumps({"status": "error",
                          "message": "Input could not be read or validated. Use the synthetic schema and omit personal data."}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
