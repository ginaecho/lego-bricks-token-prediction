"""Synthetic insurance discovery demo; not a claims or underwriting decision engine."""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing fields: " + ", ".join(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "Unknown fields are not retained: data minimization")


def text(value, label, limit=200):
    require(isinstance(value, str) and 0 < len(value) <= limit, "Invalid " + label)
    return value


def number(value, label, minimum=0, maximum=1e12):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, "Invalid " + label)
    return value


def timestamp(value):
    text(value, "timestamp", 40)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.tzinfo is not None, "Timestamps require an explicit timezone")
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError("Invalid timestamp") from exc


def identifier(value):
    text(value, "synthetic identifier", 64)
    require(value.startswith("SYN-") and all(c.isascii() and (c.isalnum() or c == "-") for c in value),
            "Identifiers must be synthetic SYN- references")
    return value


def source_payload(source, entity):
    fields(source, ("format", "payload"))
    fmt, payload = source["format"], source["payload"]
    if fmt == "acord_json":
        fields(payload, ("ACORD",))
        result = payload["ACORD"]
    elif fmt == "acord_xml":
        text(payload, "XML payload", 12000)
        require("<!" not in payload and "<?" not in payload, "XML declarations and entities are prohibited")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ValidationError("Malformed ACORD-style XML") from exc
        require(root.tag == "ACORD" and not root.attrib and not (root.text or "").strip(),
                "Expected a simple ACORD root")
        result = {}
        for child in root:
            require(child.tag not in result and not child.attrib and len(child) == 0
                    and not (child.tail or "").strip(), "XML fields must be unique simple values")
            result[child.tag] = child.text or ""
        for key in ("coverage_amount", "loss_amount"):
            if key in result:
                try:
                    result[key] = float(result[key])
                except ValueError as exc:
                    raise ValidationError("Invalid XML amount") from exc
        if "documents_complete" in result:
            require(result["documents_complete"] in ("true", "false"), "Invalid documents_complete")
            result["documents_complete"] = result["documents_complete"] == "true"
    elif fmt == "claim_text":
        require(entity == "claim", "Claim-form text is only for claims")
        text(payload, "claim-form text", 12000)
        result = {}
        for line in payload.splitlines():
            require(":" in line, "Claim form requires key: value lines")
            key, value = (part.strip() for part in line.split(":", 1))
            require(key not in result, "Duplicate claim-form field")
            result[key] = value
        if "loss_amount" in result:
            try:
                result["loss_amount"] = float(result["loss_amount"])
            except ValueError as exc:
                raise ValidationError("Invalid loss amount") from exc
        if "documents_complete" in result:
            require(result["documents_complete"] in ("true", "false"), "Invalid documents_complete")
            result["documents_complete"] = result["documents_complete"] == "true"
    else:
        raise ValidationError("Unsupported source format")
    return result


def normalize(document):
    fields(document, ("schema_version", "synthetic", "as_of", "policyholder_ref", "items", "events"),
           ("half_life_days", "limit"))
    require(type(document["schema_version"]) is int and document["schema_version"] == 1,
            "Unsupported schema version")
    require(document["synthetic"] is True, "Only clearly labeled synthetic fixtures are accepted")
    identifier(document["policyholder_ref"])
    now = timestamp(document["as_of"])
    half_life = number(document.get("half_life_days", 30), "half_life_days", 0.001, 36500)
    limit = document.get("limit", 10)
    require(type(limit) is int and 1 <= limit <= 100, "limit must be an integer from 1 to 100")
    require(isinstance(document["items"], list) and len(document["items"]) <= 1000, "Invalid items")
    require(isinstance(document["events"], list) and len(document["events"]) <= 10000, "Invalid events")
    items = {}
    for raw in document["items"]:
        fields(raw, ("id", "entity", "source"))
        item_id = identifier(raw["id"])
        require(item_id not in items, "Duplicate item id")
        entity = raw["entity"]
        require(isinstance(entity, str) and entity in ("policy", "claim", "underwriting_submission"),
                "Unsupported insurance entity")
        payload = source_payload(raw["source"], entity)
        common = ("line",)
        if entity == "claim":
            fields(payload, common + ("loss_amount", "loss_date", "documents_complete"))
            number(payload["loss_amount"], "loss_amount")
            require(timestamp(payload["loss_date"]) <= now, "Loss dates cannot be in the future")
            require(type(payload["documents_complete"]) is bool, "documents_complete must be boolean")
        elif entity == "policy":
            fields(payload, common + ("coverage_amount",))
            number(payload["coverage_amount"], "coverage_amount", 1)
        else:
            fields(payload, common + ("factors",))
            require(isinstance(payload["factors"], list) and 1 <= len(payload["factors"]) <= 3,
                    "Underwriting factors must be explicitly disclosed")
            names = set()
            for factor in payload["factors"]:
                fields(factor, ("name", "value", "disclosed"))
                name = factor["name"]
                require(isinstance(name, str) and name in ("coverage_amount", "property_age_years", "vehicle_age_years"),
                        "Only disclosed, non-sensitive demonstration factors are allowed")
                require(name not in names, "Duplicate underwriting factor")
                names.add(name)
                require(factor["disclosed"] is True, "Hidden underwriting factors are prohibited")
                number(factor["value"], "factor value", 0, 1e12 if name == "coverage_amount" else 300)
        require(isinstance(payload["line"], str) and payload["line"] in ("auto", "property"),
                "line must be auto or property")
        items[item_id] = {"id": item_id, "entity": entity, "attributes": payload}
    events = []
    event_ids = set()
    for event in document["events"]:
        fields(event, ("id", "item_id", "action", "at"))
        event_id = identifier(event["id"])
        require(event_id not in event_ids, "Duplicate event id")
        event_ids.add(event_id)
        item_id = identifier(event["item_id"])
        require(item_id in items, "Event references an unknown item")
        action = event["action"]
        require(isinstance(action, str) and action in ("browse", "purchase"), "Unsupported action")
        require(action != "purchase" or items[item_id]["entity"] == "policy",
                "Only policies can be purchased")
        at = timestamp(event["at"])
        require(at <= now, "Future behavior events are prohibited")
        age = (now - at).total_seconds() / 86400
        weight = (3 if action == "purchase" else 1) * 2 ** (-age / half_life)
        events.append((item_id, weight))
    return items, events, limit


def run(document):
    items, events, limit = normalize(document)
    direct = {key: [] for key in items}
    affinities = {}
    for key, weight in events:
        direct[key].append(weight)
        entity = items[key]["entity"]
        affinities.setdefault(entity, []).append(weight)
    entity_scores = {key: math.fsum(values) for key, values in affinities.items()}
    cold_start = not any(weight > 0 for _, weight in events)
    ranked = []
    claim_handling = []
    for key, item in items.items():
        direct_score = math.fsum(direct[key])
        affinity = 0.2 * entity_scores.get(item["entity"], 0)
        result = dict(item)
        result["score"] = direct_score + affinity
        result["reasons"] = (
            ["Cold start: equal scores; stable synthetic-id ordering"]
            if cold_start else
            [f"Recency-weighted direct interest: {direct_score:.8f}",
             f"Disclosed entity affinity contribution: {affinity:.8f}"]
        )
        ranked.append(result)
        if item["entity"] == "claim":
            complete = item["attributes"]["documents_complete"]
            claim_handling.append({
                "id": key,
                "decision": "ready_for_human_review" if complete else "request_information",
                "reason": "Required documents are complete" if complete else "Required documents are missing",
                "note": "Administrative routing only; no denial, coverage, payment, or queue-priority decision",
            })
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "status": "ok", "schema_version": 1, "synthetic": True,
        "cold_start": cold_start, "recommendations": ranked[:limit],
        "claim_handling": sorted(claim_handling, key=lambda item: item["id"]),
        "safeguards": {
            "claims": "All claims are handled independently of behavior and discovery rank",
            "privacy": "Strict allowlists; synthetic reference only; no names, VINs, addresses or free text retained",
            "underwriting": "All allowed factors are disclosed; no risk, eligibility or premium determination",
            "scope": "Demonstrative validation, not compliance certification",
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        path = Path(args[0])
        require(path.stat().st_size <= 2_000_000, "Input exceeds 2 MB")
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValidationError("Non-finite JSON number")))
        result = run(document)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        # Avoid echoing potentially identifying input or filesystem paths.
        print(json.dumps({"status": "error", "message": "Invalid input or unreadable file; check schema and constraints"}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
