"""Synthetic public-service comparison; Python standard library, no providers.

Run: python -B implementation.py example_input.json
This is a demonstration, not a benefits decision or compliance certification.
"""

import json
import math
import re
import sys
from decimal import Decimal


ATTRIBUTES = {
    "turnaround": ("days", "lower"),
    "fee": ("USD", "lower"),
    "accessibility": ("percent", "higher"),
    "online": ("boolean", "higher"),
}
MAX_BYTES = 256_000


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, location):
    require(isinstance(value, dict), location + " must be an object.")
    require(set(value) == set(expected), location + " has missing or unsupported fields.")


def text(value, location, limit=160):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit,
            location + " must be nonempty text within its length limit.")
    require(all(char.isprintable() or char in "\n\t" for char in value),
            location + " contains unsupported control characters.")
    # Do not repeat rejected text in errors. This screening is illustrative,
    # not a guarantee that arbitrary prose is free of personal information.
    require(not re.search(r"@|\b\d{3}[- ]\d{2}[- ]\d{4}\b|(?:\d[\s()+.-]*){7,}", value),
            location + " may contain personal contact or identity information.")


def identifier(value, prefix, location):
    require(isinstance(value, str) and
            re.fullmatch(re.escape(prefix) + r"[A-Z0-9-]{1,32}", value) is not None,
            location + " must use its synthetic identifier prefix.")


def number(value, location, maximum):
    require(type(value) in (int, float) and 0 <= value <= maximum
            and math.isfinite(value), location + " must be a finite number in range.")
    return value


def normalize(name, value):
    if value is None:
        return None
    fields(value, {"value", "unit"}, "Attribute")
    unit = value["unit"]
    require(isinstance(unit, str), "Attribute unit must be text.")
    raw = value["value"]
    if name == "online":
        require(unit == "boolean", "Online availability needs the boolean unit.")
        require(type(raw) is bool or (isinstance(raw, str) and raw.lower() in ("yes", "no")),
                "Online availability must be true, false, yes, or no.")
        return raw if type(raw) is bool else raw.lower() == "yes"
    raw = number(raw, "Attribute value", 1_000_000)
    if name == "turnaround":
        require(unit in ("hours", "days", "weeks"), "Unsupported turnaround unit.")
        return float(Decimal(str(raw)) * {
            "hours": Decimal(1) / Decimal(24),
            "days": Decimal(1), "weeks": Decimal(7),
        }[unit])
    if name == "fee":
        require(unit in ("USD", "cents"), "Unsupported fee unit.")
        return float(Decimal(str(raw)) / (100 if unit == "cents" else 1))
    require(unit in ("percent", "rating5"), "Unsupported accessibility unit.")
    require(raw <= (100 if unit == "percent" else 5), "Accessibility is out of range.")
    return float(Decimal(str(raw)) * (20 if unit == "rating5" else 1))


def validate(payload):
    """Single validation boundary; returns normalized, PII-minimized data."""
    fields(payload, {"schema_version", "synthetic", "citizen_service_request",
                     "preferences", "options"}, "Input")
    require(payload["schema_version"] == "1.0", "Unsupported schema version.")
    require(payload["synthetic"] is True, "Only explicitly synthetic fixtures are accepted.")
    request = payload["citizen_service_request"]
    fields(request, {"case_number", "persona_id", "address", "service"}, "Citizen request")
    identifier(request["case_number"], "SYN-REQ-", "Request case number")
    identifier(request["persona_id"], "SYN-PERSONA-", "Persona identifier")
    require(isinstance(request["address"], str) and
            re.fullmatch(r"NOT-A-REAL-ADDRESS [0-9]{3}", request["address"]) is not None,
            "Use a non-traceable synthetic address.")
    require(request["service"] in ("housing_support", "food_support", "transport_support"),
            "Unsupported service category.")
    preferences = payload["preferences"]
    fields(preferences, ATTRIBUTES, "Preferences")
    for weight in preferences.values():
        number(weight, "Preference weight", 100)
    require(sum(preferences.values()) > 0, "At least one preference weight must be positive.")
    options = payload["options"]
    require(isinstance(options, list) and 1 <= len(options) <= 30,
            "Provide between one and thirty options.")
    seen = set()
    applications = set()
    normalized = []
    for option in options:
        fields(option, {"id", "label", "benefits_application", "policy_document",
                        "attributes"}, "Option")
        identifier(option["id"], "SYN-OPT-", "Option identifier")
        require(option["id"] not in seen, "Option identifiers must be unique.")
        seen.add(option["id"])
        text(option["label"], "Option label")
        application = option["benefits_application"]
        fields(application, {"case_number", "request_case_number", "status"}, "Benefits application")
        identifier(application["case_number"], "SYN-APP-", "Application case number")
        require(application["case_number"] not in applications, "Application case numbers must be unique.")
        applications.add(application["case_number"])
        require(application["request_case_number"] == request["case_number"],
                "An application must reference the shared citizen request.")
        require(application["status"] == "illustrative", "Application status must be illustrative.")
        policy = option["policy_document"]
        fields(policy, {"id", "text"}, "Policy document")
        identifier(policy["id"], "SYN-POL-", "Policy identifier")
        text(policy["text"], "Policy text", 4000)
        require(policy["text"].startswith("SYNTHETIC POLICY: "),
                "Policy text must be labeled SYNTHETIC POLICY.")
        fields(option["attributes"], ATTRIBUTES, "Attributes")
        values = {name: normalize(name, option["attributes"][name]) for name in ATTRIBUTES}
        normalized.append({
            "id": option["id"],
            "label": option["label"],
            "benefits_application": {"case_number": application["case_number"],
                                     "status": "illustrative"},
            "policy_document": {"id": policy["id"]},
            "normalized_attributes": values,
        })
    total = sum(preferences.values())
    weights = {key: preferences[key] / total for key in ATTRIBUTES}
    return {"case_number": request["case_number"], "service": request["service"],
            "options": normalized, "weights": weights}


def compare(payload):
    data = validate(payload)
    options = data["options"]
    weights = data["weights"]
    ranges = {}
    for name in ATTRIBUTES:
        available = [option["normalized_attributes"][name] for option in options
                     if option["normalized_attributes"][name] is not None]
        ranges[name] = (min(available), max(available)) if available else (None, None)
    for option in options:
        contributions = {}
        total = 0
        missing = []
        for name, (_, direction) in ATTRIBUTES.items():
            value = option["normalized_attributes"][name]
            low, high = ranges[name]
            if value is None:
                score = 0.0
                missing.append(name)
                explanation = "Not provided. It receives zero points."
            elif name == "online":
                score = float(value)
                explanation = "Online service gets one point; no online service gets zero."
            elif low == high:
                score = 1.0
                explanation = "All provided values are equal. Each gets one point."
            else:
                score = (high - value) / (high - low) if direction == "lower" else (value - low) / (high - low)
                explanation = ("A lower value is better." if direction == "lower" else
                               "A higher value is better.") + " Points run from zero to one within this comparison."
            weighted = score * weights[name]
            total += weighted
            contributions[name] = {
                "points": score, "weight": weights[name],
                "weighted_points": weighted, "explanation": explanation,
            }
        option["score"] = total
        option["missing_attributes"] = missing
        option["explanation"] = contributions
    ordered = sorted(options, key=lambda item: (-item["score"], item["id"]))
    ranking = [{"rank": index, "option_id": option["id"], "score": option["score"]}
               for index, option in enumerate(ordered, 1)]
    return {
        "schema_version": "1.0", "status": "ok", "synthetic": True,
        "citizen_service_request": {"case_number": data["case_number"], "service": data["service"]},
        "comparison": {
            "columns": [option["id"] for option in sorted(options, key=lambda item: item["id"])],
            "rows": [
                {"attribute": name, "unit": unit, "preferred_direction": direction,
                 "range": {"minimum": ranges[name][0], "maximum": ranges[name][1]},
                 "values": {option["id"]: option["normalized_attributes"][name]
                            for option in sorted(options, key=lambda item: item["id"])}}
                for name, (unit, direction) in ATTRIBUTES.items()
            ],
        },
        "options": sorted(options, key=lambda item: item["id"]),
        "ranking": ranking,
        "summary": "First for your preferences: " + ranking[0]["option_id"] +
                   ". This is a service comparison, not a decision about benefits.",
        "review": {
            "method": "Weighted sum of attribute points. Highest score comes first.",
            "numeric_formula": "When values differ: lower is better uses (maximum - value) / (maximum - minimum); higher is better uses (value - minimum) / (maximum - minimum).",
            "tie_rule": "Equal scores are ordered by synthetic option identifier.",
            "missing_rule": "Missing values receive zero points. Weights are not redistributed.",
            "policy_use": "Policy text is context only. Attributes are supplied facts, not extracted legal conclusions.",
            "privacy": "Persona identifiers, addresses, and policy text are omitted from the result.",
            "limitations": "Synthetic demonstration only. No Privacy Act, GDPR, FOIA, or Section 508 certification.",
            "plain_language": "Units and point explanations are included as text. No colors or images are required.",
        },
    }


def reject_constant(_value):
    raise ValidationError("JSON must use finite numbers.")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON contains duplicate fields.")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], "rb") as source:
            content = source.read(MAX_BYTES + 1)
        require(len(content) <= MAX_BYTES, "Input file is too large.")
        payload = json.loads(content.decode("utf-8"), parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
        output = compare(payload)
        code = 0
    except ValidationError as error:
        output, code = {"schema_version": "1.0", "status": "error", "message": str(error)}, 2
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, OverflowError):
        output, code = {"schema_version": "1.0", "status": "error",
                        "message": "Cannot read or validate the input JSON file."}, 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
