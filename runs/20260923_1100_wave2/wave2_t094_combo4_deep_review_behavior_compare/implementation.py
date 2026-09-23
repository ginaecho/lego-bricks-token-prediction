"""Synthetic, deterministic research -> review -> behavior -> comparison CLI.

No certification is performed. Evidence is supplied by the caller, not verified
against the real world. Timestamps are UTC ISO 8601 strings ending in Z.
"""

import copy
import datetime as dt
import json
import math
import sys


SCHEMA_VERSION = "1.0"
STAGES = ("research", "review", "behavior", "compare")
UNITS = {
    "price": {"USD": 1.0},
    "weight": {"g": 1.0, "kg": 1000.0},
    "runtime": {"h": 1.0, "min": 1.0 / 60.0},
}
CANONICAL_UNITS = {"price": "USD", "weight": "g", "runtime": "h"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unexpected fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()),
            path + " must be a nonempty string")


def number(value, path, minimum=0, maximum=None):
    require(type(value) in (int, float), path + " must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= minimum, path + " is not a finite valid number")
    require(maximum is None or value <= maximum, path + " is too large")


def array(value, path):
    require(isinstance(value, list), path + " must be an array")


def timestamp(value, path):
    text(value, path)
    require(value.endswith("Z"), path + " must be a UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValidationError(path + " is not an ISO 8601 timestamp") from exc
    require("T" in value and parsed.utcoffset() == dt.timedelta(0),
            path + " must be a UTC datetime")
    return parsed


def unique_ids(rows, path):
    result = set()
    for row in rows:
        require(isinstance(row, dict), path + " entries must be objects")
        text(row.get("id"), path + ".id")
        require(row["id"] not in result, path + " contains a duplicate id")
        result.add(row["id"])
    return result


def validate_input(data):
    obj(data, ("schema_version", "synthetic", "as_of", "documents",
               "requirements", "products", "events", "preferences"), "input")
    require(data["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    require(data["synthetic"] is True, "fixture must be labeled synthetic")
    now = timestamp(data["as_of"], "as_of")
    for key in ("documents", "requirements", "products", "events"):
        array(data[key], key)
    require(bool(data["products"]), "products must not be empty")
    require(bool(data["requirements"]), "requirements must not be empty")
    product_ids = unique_ids(data["products"], "products")
    requirement_ids = unique_ids(data["requirements"], "requirements")
    unique_ids(data["documents"], "documents")
    unique_ids(data["events"], "events")
    for product in data["products"]:
        obj(product, ("id", "name", "category", "attributes"), "product")
        text(product["name"], "product.name")
        text(product["category"], "product.category")
        obj(product["attributes"], UNITS, "product.attributes")
        for attribute, unit_table in UNITS.items():
            measurement = product["attributes"][attribute]
            if measurement is None:
                continue
            obj(measurement, ("value", "unit"), attribute)
            text(measurement["unit"], attribute + ".unit")
            require(measurement["unit"] in unit_table,
                    attribute + " uses an unsupported unit")
            number(measurement["value"], attribute + ".value", maximum=1e12)
    for requirement in data["requirements"]:
        obj(requirement, ("id", "description", "mandatory"), "requirement")
        text(requirement["description"], "requirement.description")
        require(type(requirement["mandatory"]) is bool,
                "requirement.mandatory must be boolean")
    for document in data["documents"]:
        obj(document, ("id", "title", "published_at", "claims"), "document")
        text(document["title"], "document.title")
        require(timestamp(document["published_at"], "published_at") <= now,
                "document cannot be published in the future")
        array(document["claims"], "document.claims")
        seen = set()
        for claim in document["claims"]:
            obj(claim, ("product_id", "requirement_id", "stance", "quote"), "claim")
            text(claim["product_id"], "claim.product_id")
            text(claim["requirement_id"], "claim.requirement_id")
            require(claim["product_id"] in product_ids, "claim references unknown product")
            require(claim["requirement_id"] in requirement_ids,
                    "claim references unknown requirement")
            require(claim["stance"] in ("supports", "opposes", "uncertain"),
                    "invalid claim stance")
            text(claim["quote"], "claim.quote")
            pair = (claim["product_id"], claim["requirement_id"])
            require(pair not in seen, "duplicate product/requirement claim within document")
            seen.add(pair)
    for event in data["events"]:
        obj(event, ("id", "product_id", "type", "at"), "event")
        text(event["product_id"], "event.product_id")
        require(event["product_id"] in product_ids, "event references unknown product")
        require(event["type"] in ("browse", "purchase"), "invalid event type")
        require(timestamp(event["at"], "event.at") <= now, "event cannot be in the future")
    preferences = data["preferences"]
    obj(preferences, ("half_life_days", "preferred_categories", "weights",
                      "max_price_usd"), "preferences")
    number(preferences["half_life_days"], "half_life_days", minimum=0.001, maximum=1e6)
    array(preferences["preferred_categories"], "preferred_categories")
    for category in preferences["preferred_categories"]:
        text(category, "preferred category")
    require(len(set(preferences["preferred_categories"])) ==
            len(preferences["preferred_categories"]), "duplicate preferred category")
    obj(preferences["weights"], ("price", "weight", "runtime", "behavior"), "weights")
    for name, weight in preferences["weights"].items():
        number(weight, "weights." + name, maximum=1e6)
    require(sum(preferences["weights"].values()) > 0, "at least one weight must be positive")
    if preferences["max_price_usd"] is not None:
        number(preferences["max_price_usd"], "max_price_usd", maximum=1e12)
    return data


def artifact(stage, records, warnings=()):
    return {"schema_version": SCHEMA_VERSION, "stage": stage,
            "records": records, "warnings": list(warnings)}


def evidence_for(data, product_id, requirement_id):
    evidence = []
    for document in sorted(data["documents"], key=lambda row: row["id"]):
        for index, claim in enumerate(document["claims"]):
            if (claim["product_id"], claim["requirement_id"]) == (product_id, requirement_id):
                evidence.append({"document_id": document["id"], "claim_index": index,
                                 "stance": claim["stance"], "quote": claim["quote"]})
    return evidence


def finding_state(evidence):
    stances = {item["stance"] for item in evidence}
    if "supports" in stances and "opposes" in stances:
        return "disputed"
    if "uncertain" in stances:
        return "unresolved"
    if "supports" in stances:
        return "supported"
    if "opposes" in stances:
        return "contradicted"
    return "missing"


def research(data):
    rows = []
    for product in sorted(data["products"], key=lambda row: row["id"]):
        findings = []
        for requirement in data["requirements"]:
            evidence = evidence_for(data, product["id"], requirement["id"])
            state = finding_state(evidence)
            question = (None if state == "supported" else
                        "What evidence resolves " + requirement["id"] + " for " + product["id"] + "?")
            findings.append({"requirement_id": requirement["id"], "state": state,
                             "evidence": evidence, "unresolved_question": question})
        rows.append({"product_id": product["id"], "findings": findings})
    return artifact("research", rows, ["Caller-supplied evidence; no external verification."])


def review(data, previous):
    mandatory = {row["id"]: row["mandatory"] for row in data["requirements"]}
    records = []
    for row in previous["records"]:
        checks = []
        for finding in row["findings"]:
            checks.append({**copy.deepcopy(finding),
                           "mandatory": mandatory[finding["requirement_id"]]})
        gaps = [check["requirement_id"] for check in checks if check["state"] != "supported"]
        eligible = all(not check["mandatory"] or check["state"] == "supported"
                       for check in checks)
        records.append({"product_id": row["product_id"], "checks": checks,
                        "gaps": gaps, "eligible": eligible})
    return artifact("review", records, ["Requirement review is not certification or a compliance claim."])


def behavior(data, previous):
    now = timestamp(data["as_of"], "as_of")
    prefs = data["preferences"]
    products = {row["id"]: row for row in data["products"]}
    scores = {}
    for row in previous["records"]:
        if not row["eligible"]:
            continue
        score = 0.0
        for event in data["events"]:
            if event["product_id"] == row["product_id"]:
                age = (now - timestamp(event["at"], "event.at")).total_seconds() / 86400
                score += (3.0 if event["type"] == "purchase" else 1.0) * (
                    2.0 ** (-age / prefs["half_life_days"]))
        scores[row["product_id"]] = score
    cold_start = not any(scores.values())
    if cold_start:
        scores = {pid: float(products[pid]["category"] in prefs["preferred_categories"])
                  for pid in scores}
    maximum = max(scores.values(), default=0.0)
    records = []
    for row in previous["records"]:
        eligible = row["eligible"]
        signal = scores.get(row["product_id"], 0.0)
        records.append({"product_id": row["product_id"], "eligible": eligible,
                        "review_gaps": copy.deepcopy(row["gaps"]),
                        "review_checks": copy.deepcopy(row["checks"]),
                        "raw_signal": signal,
                        "score": signal / maximum if maximum else 0.0,
                        "mode": "excluded" if not eligible else (
                            "cold_start" if cold_start else "history")})
    return artifact("behavior", records,
                    ["Ineligible products cannot gain eligibility through activity.",
                     "Cold start uses explicit category preferences; ties remain neutral."])


def normalized_attributes(product):
    return {key: None if measurement is None else
            measurement["value"] * UNITS[key][measurement["unit"]]
            for key, measurement in product["attributes"].items()}


def compare(data, previous):
    products = {row["id"]: row for row in data["products"]}
    preferences = data["preferences"]
    attributes = {pid: normalized_attributes(product) for pid, product in products.items()}
    candidates = []
    excluded = []
    budget = preferences["max_price_usd"]
    for row in previous["records"]:
        pid = row["product_id"]
        price = attributes[pid]["price"]
        if not row["eligible"]:
            excluded.append(pid + ": mandatory evidence gap")
        elif budget is not None and (price is None or price > budget):
            excluded.append(pid + ": price missing or above budget")
        else:
            candidates.append(row)
    ranges = {}
    for key in UNITS:
        values = [attributes[row["product_id"]][key] for row in candidates
                  if attributes[row["product_id"]][key] is not None]
        ranges[key] = (min(values), max(values)) if values else (0.0, 0.0)
    records = []
    total_weight = sum(preferences["weights"].values())
    for row in candidates:
        pid = row["product_id"]
        utilities = {"behavior": row["score"]}
        for key in UNITS:
            value = attributes[pid][key]
            low, high = ranges[key]
            if value is None:
                utility = 0.0
            elif low == high:
                utility = 1.0
            elif key == "runtime":
                utility = (value - low) / (high - low)
            else:
                utility = (high - value) / (high - low)
            utilities[key] = utility
        score = sum(utilities[key] * weight for key, weight in
                    preferences["weights"].items()) / total_weight
        records.append({"product_id": pid, "name": products[pid]["name"],
                        "attributes": attributes[pid], "units": dict(CANONICAL_UNITS),
                        "utilities": utilities, "score": score, "rank": 0,
                        "behavior_score": row["score"], "behavior_mode": row["mode"],
                        "review_gaps": copy.deepcopy(row["review_gaps"]),
                        "review_checks": copy.deepcopy(row["review_checks"])})
    records.sort(key=lambda row: (-row["score"], row["product_id"]))
    for rank, row in enumerate(records, 1):
        row["rank"] = rank
    return artifact("compare", records, excluded + [
        "Lower price/weight and higher runtime are preferred.",
        "Missing attributes receive zero utility; equal known values receive one.",
        "Scores are relative to eligible, within-budget candidates, not quality guarantees."])


def validate_artifact(value, stage, data, previous=None):
    """One shared boundary validator checks shape, provenance and derived values.

    Exact deterministic reconstruction rejects invented citations, lost gaps,
    changed eligibility and invalid scores, not just superficially valid JSON.
    """
    require(stage in STAGES, "unknown pipeline stage")
    obj(value, ("schema_version", "stage", "records", "warnings"), stage)
    require(value["schema_version"] == SCHEMA_VERSION and value["stage"] == stage,
            "invalid artifact header")
    array(value["records"], stage + ".records")
    array(value["warnings"], stage + ".warnings")
    for warning in value["warnings"]:
        text(warning, "warning")
    if stage == "research":
        expected = research(data)
    else:
        require(previous is not None, "missing validated predecessor")
        require(previous["stage"] == STAGES[STAGES.index(stage) - 1],
                "incorrect predecessor stage")
        expected = {"review": review, "behavior": behavior, "compare": compare}[stage](data, previous)
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        expected_encoded = json.dumps(expected, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValidationError("artifact contains invalid JSON values") from exc
    require(encoded == expected_encoded, stage + " violates schema or provenance invariants")
    return value


def run_pipeline(data):
    validate_input(data)
    stages = {}
    previous = None
    for stage, builder in zip(STAGES, (research, review, behavior, compare)):
        result = builder(data) if previous is None else builder(data, previous)
        previous = validate_artifact(result, stage, data, previous)
        stages[stage] = previous
    return {"schema_version": SCHEMA_VERSION, "synthetic": True, "status": "ok",
            "as_of": data["as_of"], "stages": stages}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=reject_duplicates,
                             parse_constant=reject_constant)
        output = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                          "error": str(exc)}, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
