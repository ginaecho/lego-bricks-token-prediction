"""Synthetic comparison -> evidence review reference CLI (Python standard library).

Inputs and outputs use schema_version 1. Missing attributes score zero; numeric
preferences use observed min/max utility and weights. Ties break by product ID.
Evidence is a user-supplied structured assertion with an exact source quote, not
an independently verified interpretation of that quote or a certification.
"""

import copy
import json
import math
import sys


class ValidationError(ValueError):
    pass


UNITS = {
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1),
    "g": ("mass", 0.001), "kg": ("mass", 1),
    "USD": ("currency", 1), "count": ("count", 1),
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing fields: " + str(set(required) - value.keys()))
    require(value.keys() <= set(required) | set(optional), "Unknown fields")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected nonempty text")
    return value


def number(value):
    require(type(value) in (int, float), "Expected a finite number, not boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, "Expected a finite number")
    return value


def indexed(items, label, allow_empty=False):
    require(isinstance(items, list) and (allow_empty or bool(items)), label + " must be a list")
    result = {}
    for item in items:
        require(isinstance(item, dict) and "id" in item, label + " entries need IDs")
        ident = text(item["id"])
        require(ident not in result, "Duplicate " + label + " ID: " + ident)
        result[ident] = item
    return result


def normalize(raw, definition):
    keys(raw, ["value"], ["unit"])
    if definition["kind"] == "text":
        require("unit" not in raw, "Text attributes cannot have units")
        return " ".join(text(raw["value"]).split()).casefold()
    value = number(raw["value"])
    unit = raw.get("unit", definition["unit"])
    require(isinstance(unit, str) and unit in UNITS, "Unknown unit")
    base = definition["unit"]
    require(UNITS[unit][0] == UNITS[base][0], "Incompatible units")
    try:
        normalized = value * (UNITS[unit][1] / UNITS[base][1])
    except OverflowError:
        raise ValidationError("Normalized value overflow")
    return number(normalized)


def equal(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def validate_input(data):
    keys(data, ["schema_version", "synthetic", "attributes", "products",
                "preferences", "requirements", "documents", "evidence"])
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Unsupported schema version")
    require(data["synthetic"] is True, "Fixture must be labeled synthetic")
    attrs = indexed(data["attributes"], "attribute")
    for attr in attrs.values():
        keys(attr, ["id", "kind"], ["unit"])
        require(attr["kind"] in ("number", "text"), "Unknown attribute kind")
        if attr["kind"] == "number":
            require(isinstance(attr.get("unit"), str) and attr["unit"] in UNITS,
                    "Numeric attributes require a supported canonical unit")
        else:
            require("unit" not in attr, "Text definitions cannot have units")
    products = indexed(data["products"], "product")
    for product in products.values():
        keys(product, ["id", "name", "attributes"])
        text(product["name"])
        require(isinstance(product["attributes"], dict), "Product attributes must be an object")
        for attr, value in product["attributes"].items():
            require(attr in attrs, "Unknown product attribute")
            normalize(value, attrs[attr])
    preferences = data["preferences"]
    require(isinstance(preferences, list) and bool(preferences), "Preferences must be nonempty")
    seen = set()
    for pref in preferences:
        keys(pref, ["attribute", "direction", "weight"])
        attr = text(pref["attribute"])
        require(attr in attrs and attrs[attr]["kind"] == "number", "Preference must be numeric")
        require(attr not in seen, "Duplicate preference attribute")
        seen.add(attr)
        require(pref["direction"] in ("min", "max"), "Direction must be min or max")
        require(number(pref["weight"]) > 0, "Weight must be positive")
    number(sum(p["weight"] for p in preferences))
    requirements = indexed(data["requirements"], "requirement")
    for req in requirements.values():
        keys(req, ["id", "attribute", "operator", "target", "evidence_required"], ["expected"])
        attr = text(req["attribute"])
        require(attr in attrs, "Unknown requirement attribute")
        require(req["operator"] in ("min", "max", "equals", "exists"), "Unknown operator")
        require(req["target"] in ("winner", "all"), "Unknown requirement target")
        require(type(req["evidence_required"]) is bool, "evidence_required must be boolean")
        if req["operator"] == "exists":
            require("expected" not in req, "exists cannot specify expected")
        else:
            require("expected" in req, "Expected requirement value missing")
            normalize(req["expected"], attrs[attr])
        if req["operator"] in ("min", "max"):
            require(attrs[attr]["kind"] == "number", "Bounds must be numeric")
    documents = indexed(data["documents"], "document", allow_empty=True)
    for doc in documents.values():
        keys(doc, ["id", "text"])
        text(doc["text"])
    evidence = indexed(data["evidence"], "evidence", allow_empty=True)
    for item in evidence.values():
        keys(item, ["id", "requirement_id", "product_id", "document_id",
                    "locator", "quote", "asserted"])
        for name in ("requirement_id", "product_id", "document_id", "locator", "quote"):
            text(item[name])
        require(item["requirement_id"] in requirements, "Unknown evidence requirement")
        require(item["product_id"] in products, "Unknown evidence product")
        require(item["document_id"] in documents, "Unknown evidence document")
        require(item["quote"] in documents[item["document_id"]]["text"],
                "Evidence quote not found in document")
        attr = requirements[item["requirement_id"]]["attribute"]
        normalize(item["asserted"], attrs[attr])
    return data


def _comparison(data):
    attrs = {a["id"]: a for a in data["attributes"]}
    products = sorted(data["products"], key=lambda p: p["id"])
    normalized = {
        p["id"]: {a: normalize(v, attrs[a]) for a, v in p["attributes"].items()}
        for p in products
    }
    components = {p["id"]: {} for p in products}
    total_weight = sum(p["weight"] for p in data["preferences"])
    for pref in data["preferences"]:
        attr = pref["attribute"]
        values = [v[attr] for v in normalized.values() if attr in v]
        # Scaling first avoids overflow when subtracting very large finite values.
        scale = max([abs(v) for v in values] + [1])
        lo, hi = (min(values) / scale, max(values) / scale) if values else (0, 0)
        for product in products:
            value = normalized[product["id"]].get(attr)
            if value is None:
                utility = 0.0
            elif hi == lo:
                utility = 1.0
            else:
                utility = (value / scale - lo) / (hi - lo)
                if pref["direction"] == "min":
                    utility = 1 - utility
            components[product["id"]][attr] = utility * (pref["weight"] / total_weight)
    ranking = [
        {"product_id": p["id"], "score": sum(components[p["id"]].values()),
         "contributions": components[p["id"]]}
        for p in products
    ]
    ranking.sort(key=lambda r: (-r["score"], r["product_id"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    return {
        "normalized_products": [
            {"id": p["id"], "name": p["name"], "attributes": normalized[p["id"]]}
            for p in products
        ],
        "side_by_side": [
            {"attribute": a["id"], "unit": a.get("unit"),
             "values": {p["id"]: normalized[p["id"]].get(a["id"]) for p in products}}
            for a in data["attributes"]
        ],
        "ranking": ranking,
        "winner_id": ranking[0]["product_id"],
    }


def _review(data, comparison):
    attrs = {a["id"]: a for a in data["attributes"]}
    checks = []
    for req in data["requirements"]:
        attr = req["attribute"]
        expected = normalize(req["expected"], attrs[attr]) if "expected" in req else None
        for product in comparison["normalized_products"]:
            if req["target"] == "winner" and product["id"] != comparison["winner_id"]:
                continue
            actual = product["attributes"].get(attr)
            passed = actual is not None
            if passed and req["operator"] != "exists":
                if req["operator"] == "equals":
                    passed = equal(actual, expected)
                elif req["operator"] == "min":
                    passed = actual >= expected or equal(actual, expected)
                else:
                    passed = actual <= expected or equal(actual, expected)
            traces = []
            for evidence in data["evidence"]:
                if (evidence["product_id"] != product["id"]
                        or evidence["requirement_id"] != req["id"]):
                    continue
                asserted = normalize(evidence["asserted"], attrs[attr])
                traces.append({
                    "evidence_id": evidence["id"], "document_id": evidence["document_id"],
                    "locator": evidence["locator"], "quote": evidence["quote"],
                    "normalized_assertion": asserted,
                    "matches_product": actual is not None and equal(actual, asserted),
                })
            gaps = []
            if actual is None:
                gaps.append("missing_attribute")
            elif not passed:
                gaps.append("requirement_not_met")
            if any(not trace["matches_product"] for trace in traces):
                gaps.append("conflicting_evidence")
            if req["evidence_required"] and not any(t["matches_product"] for t in traces):
                gaps.append("missing_supporting_evidence")
            checks.append({
                "requirement_id": req["id"], "product_id": product["id"],
                "comparison_rank": next(r["rank"] for r in comparison["ranking"]
                                        if r["product_id"] == product["id"]),
                "attribute": attr, "operator": req["operator"], "expected": expected,
                "actual": actual, "status": "gap" if gaps else "met",
                "gaps": gaps, "evidence": traces,
            })
    return {
        "basis_winner_id": comparison["winner_id"],
        "checks": checks, "gap_count": sum(c["status"] == "gap" for c in checks),
        "notice": "Structured evidence checks only; no certification or compliance claim.",
    }


def validate_envelope(envelope, stage):
    """One shared boundary validator; recomputation rejects fabricated handoffs."""
    required = ["schema_version", "status", "synthetic", "input", "comparison"]
    if stage == "review":
        required.append("review")
    require(stage in ("compare", "review"), "Unknown pipeline stage")
    keys(envelope, required)
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "Invalid output version")
    require(envelope["status"] == "ok" and envelope["synthetic"] is True, "Invalid output status")
    validate_input(envelope["input"])
    require(envelope["comparison"] == _comparison(envelope["input"]), "Invalid comparison handoff")
    if stage == "review":
        require(envelope["review"] == _review(envelope["input"], envelope["comparison"]),
                "Invalid review output")
    # Also enforce JSON serializability and forbid nonfinite stage values.
    json.dumps(envelope, allow_nan=False)
    return envelope


def compare(data):
    data = copy.deepcopy(validate_input(data))
    return validate_envelope({
        "schema_version": 1, "status": "ok", "synthetic": True,
        "input": data, "comparison": _comparison(data),
    }, "compare")


def review(compared):
    validate_envelope(compared, "compare")
    result = copy.deepcopy(compared)
    result["review"] = _review(result["input"], result["comparison"])
    return validate_envelope(result, "review")


def run(data):
    return review(compare(data))


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=reject_duplicates)
        result = run(data)
        output = json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "message": str(exc)}))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
