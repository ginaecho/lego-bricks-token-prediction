"""Deterministic product comparison for explicitly synthetic fixtures."""

import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


UNITS = {
    "g": ("mass", 1.0), "kg": ("mass", 1000.0),
    "GB": ("storage", 1.0), "TB": ("storage", 1000.0),
    "mm": ("length", 1.0), "cm": ("length", 10.0),
    "m": ("length", 1000.0), "USD": ("currency", 1.0),
    "": ("unitless", 1.0),
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Missing required fields: " +
            ", ".join(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "Unknown object fields")
    return value


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected nonempty text")
    require(len(value) <= 200, "Text exceeds 200 characters")
    return value.strip()


def key(value):
    return re.sub(r"\s+", "_", text(value).casefold())


def category(value):
    return " ".join(text(value).casefold().split())


def number(value):
    require(type(value) in (int, float), "Expected a finite number, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValidationError("Number is outside supported range") from None
    require(math.isfinite(result) and abs(result) <= 1e12,
            "Number must be finite and within +/- 1e12")
    return result


def sequence(value, minimum, maximum):
    require(isinstance(value, list) and minimum <= len(value) <= maximum,
            f"Expected a list with {minimum} to {maximum} items")
    return value


def normalize(data):
    """Validate once and produce the shared canonical representation."""
    obj(data, ("fixture_label", "attributes", "products", "preferences"))
    require(text(data["fixture_label"]).casefold().startswith("synthetic"),
            "fixture_label must start with 'synthetic'")
    attributes, lookup = [], {}
    for raw in sequence(data["attributes"], 1, 30):
        obj(raw, ("key", "kind"), ("label", "unit", "aliases"))
        name = key(raw["key"])
        require(raw["kind"] in ("number", "category"), "Unsupported attribute kind")
        unit = raw.get("unit", "")
        require(isinstance(unit, str) and unit in UNITS, "Unsupported unit")
        require(raw["kind"] == "number" or unit == "", "Categories cannot have units")
        aliases = sequence(raw.get("aliases", []), 0, 10)
        for alias in [name] + [key(a) for a in aliases]:
            require(alias not in lookup, "Duplicate or ambiguous attribute alias")
            lookup[alias] = name
        attributes.append({"key": name, "kind": raw["kind"],
                           "label": text(raw.get("label", raw["key"])), "unit": unit})
    by_key = {a["key"]: a for a in attributes}
    products, ids = [], set()
    for raw in sequence(data["products"], 1, 100):
        obj(raw, ("id", "name", "attributes"))
        identifier = text(raw["id"])
        require(identifier not in ids, "Duplicate product id")
        ids.add(identifier)
        require(isinstance(raw["attributes"], dict), "Product attributes must be an object")
        values, seen = {a["key"]: None for a in attributes}, set()
        for source, value in raw["attributes"].items():
            source = key(source)
            require(source in lookup, "Unknown product attribute")
            name = lookup[source]
            require(name not in seen, "Attribute supplied multiple times through aliases")
            seen.add(name)
            spec = by_key[name]
            if value is None:
                continue
            if spec["kind"] == "category":
                values[name] = category(value)
            else:
                unit = spec["unit"]
                if isinstance(value, dict):
                    obj(value, ("value", "unit"))
                    unit = value["unit"]
                    require(isinstance(unit, str) and unit in UNITS, "Unsupported unit")
                    require(UNITS[unit][0] == UNITS[spec["unit"]][0],
                            "Incompatible unit dimensions")
                    value = value["value"]
                converted = number(value) * UNITS[unit][1] / UNITS[spec["unit"]][1]
                values[name] = number(converted)
        products.append({"id": identifier, "name": text(raw["name"]), "attributes": values})
    preferences, seen = [], set()
    for raw in sequence(data["preferences"], 1, 30):
        obj(raw, ("attribute", "weight"), ("direction", "order"))
        source = key(raw["attribute"])
        require(source in lookup, "Unknown preference attribute")
        name = lookup[source]
        require(name not in seen, "Duplicate preference attribute")
        seen.add(name)
        weight = number(raw["weight"])
        require(weight > 0, "Preference weight must be positive")
        pref = {"attribute": name, "weight": weight}
        if by_key[name]["kind"] == "number":
            require(raw.get("direction") in ("lower", "higher") and "order" not in raw,
                    "Numeric preference needs lower/higher direction and no order")
            pref["direction"] = raw["direction"]
        else:
            require("direction" not in raw and "order" in raw,
                    "Category preference needs order and no direction")
            order = [category(v) for v in sequence(raw["order"], 1, 100)]
            require(len(set(order)) == len(order), "Duplicate normalized category preference")
            pref["order"] = order
        preferences.append(pref)
    return {"fixture_label": text(data["fixture_label"]), "attributes": attributes,
            "products": products, "preferences": preferences}


def compare(data):
    canonical = normalize(data)
    products, preferences = canonical["products"], canonical["preferences"]
    total_weight = sum(p["weight"] for p in preferences)
    utilities = {}
    for pref in preferences:
        name = pref["attribute"]
        available = [p["attributes"][name] for p in products
                     if p["attributes"][name] is not None]
        if "direction" in pref and available:
            low, high = min(available), max(available)
        for product in products:
            value = product["attributes"][name]
            utility = 0.0
            if value is not None:
                if "order" in pref:
                    order = pref["order"]
                    utility = ((len(order) - order.index(value)) / len(order)
                               if value in order else 0.0)
                elif high == low:
                    utility = 1.0
                else:
                    utility = (value - low) / (high - low)
                    if pref["direction"] == "lower":
                        utility = 1.0 - utility
            utilities[(product["id"], name)] = utility
    ranking = []
    for product in products:
        details = []
        score = 0.0
        for pref in preferences:
            utility = utilities[(product["id"], pref["attribute"])]
            contribution = utility * (pref["weight"] / total_weight)
            score += contribution
            details.append({"attribute": pref["attribute"], "utility": round(utility, 6),
                            "weighted_score": round(contribution, 6),
                            "missing": product["attributes"][pref["attribute"]] is None})
        ranking.append({"id": product["id"], "name": product["name"],
                        "score": score, "contributions": details})
    ranking.sort(key=lambda row: (-row["score"], row["id"]))
    for position, row in enumerate(ranking, 1):
        row["rank"], row["score"] = position, round(row["score"], 6)
    comparison = [
        {**attribute, "values": {p["id"]: p["attributes"][attribute["key"]]
                                for p in products}}
        for attribute in canonical["attributes"]
    ]
    return {"status": "ok", **canonical, "comparison": comparison, "ranking": ranking}


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, "Duplicate JSON object key")
        result[name] = value
    return result


def reject_constant(value):
    raise ValidationError("Nonfinite JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py input.json")
        with Path(argv[0]).open("rb") as stream:
            payload = stream.read(1_000_001)
        require(len(payload) <= 1_000_000, "Input exceeds 1 MB")
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        result = compare(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
