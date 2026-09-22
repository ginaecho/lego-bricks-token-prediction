"""Fixture-only product comparison. Run: python implementation.py example_input.json.

Input schema declares numeric attributes (optional canonical unit) or categorical
attributes. Quantities are numbers in the canonical unit or {value, unit} objects.
Preferences use positive weights and either min/max direction or categorical
target. Constraints use numeric min/max or categorical equals. Missing values
receive zero utility and fail constraints. Ranking uses eligible-set min/max
scaling; ties retain input order. An injected explanation callback may return only
verified structured claims, never unvalidated prose. No provider is contacted.
"""

import argparse
import copy
import json
import math
import sys


class ValidationError(ValueError):
    """Invalid input or ungrounded explanation."""


# Factors convert to a dimension's base unit. Currency dimensions stay separate.
UNITS = {
    "m": ("length", 1), "cm": ("length", .01), "mm": ("length", .001),
    "km": ("length", 1000), "in": ("length", .0254), "ft": ("length", .3048),
    "kg": ("mass", 1), "g": ("mass", .001), "lb": ("mass", .45359237),
    "oz": ("mass", .028349523125),
    "L": ("volume", 1), "mL": ("volume", .001),
    "W": ("power", 1), "kW": ("power", 1000),
    "Wh": ("energy", 1), "kWh": ("energy", 1000),
    "s": ("time", 1), "min": ("time", 60), "h": ("time", 3600),
    "B": ("storage", 1), "KB": ("storage", 1000),
    "MB": ("storage", 1000 ** 2), "GB": ("storage", 1000 ** 3),
    "KiB": ("storage", 1024), "MiB": ("storage", 1024 ** 2),
    "GiB": ("storage", 1024 ** 3), "%": ("percentage", 1),
    "USD": ("USD", 1), "EUR": ("EUR", 1),
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def mapping(value, label):
    require(isinstance(value, dict), label + " must be an object")
    return value


def keys(value, allowed, label, required=()):
    mapping(value, label)
    require(not set(value) - set(allowed), label + " has unsupported fields")
    require(set(required) <= set(value), label + " is missing required fields")


def number(value, label):
    require(type(value) in (int, float), label + " must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValidationError(label + " must be a finite number") from None
    require(math.isfinite(result), label + " must be a finite number")
    return result


def quantity(value, spec, label):
    target = spec.get("unit")
    if isinstance(value, dict):
        keys(value, ("value", "unit"), label, ("value", "unit"))
        unit = value["unit"]
        require(isinstance(unit, str) and unit in UNITS, label + " has unknown unit")
        require(target is not None, label + " supplies a unit to a unitless attribute")
        require(UNITS[unit][0] == UNITS[target][0], label + " has incompatible unit")
        result = number(value["value"], label) * (UNITS[unit][1] / UNITS[target][1])
        require(math.isfinite(result), label + " conversion overflow")
        return result
    return number(value, label)


def validate_schema(schema):
    mapping(schema, "schema")
    for attribute, spec in schema.items():
        require(isinstance(attribute, str) and bool(attribute.strip()), "invalid attribute name")
        keys(spec, ("type", "unit"), attribute, ("type",))
        require(spec["type"] in ("numeric", "categorical"), attribute + " has invalid type")
        if "unit" in spec:
            require(spec["type"] == "numeric", attribute + " has conflicting schema")
            require(isinstance(spec["unit"], str) and spec["unit"] in UNITS,
                    attribute + " has unknown canonical unit")


def policies(raw, schema, is_preference):
    mapping(raw, "preferences" if is_preference else "constraints")
    result = {}
    for attribute, policy in raw.items():
        require(attribute in schema, "unknown policy attribute: " + attribute)
        numeric = schema[attribute]["type"] == "numeric"
        if is_preference:
            required = ("weight", "direction") if numeric else ("weight", "target")
            keys(policy, required, attribute + " preference", required)
            weight = number(policy["weight"], attribute + " weight")
            require(weight > 0, "weights must be positive")
            result[attribute] = dict(policy, weight=weight)
            if numeric:
                require(policy["direction"] in ("min", "max"), "invalid preference direction")
            else:
                require(isinstance(policy["target"], str), "categorical target must be a string")
        elif numeric:
            keys(policy, ("min", "max"), attribute + " constraint")
            require(bool(policy), "constraint must not be empty")
            normalized = {key: quantity(value, schema[attribute], attribute + " " + key)
                          for key, value in policy.items()}
            require(normalized.get("min", -math.inf) <= normalized.get("max", math.inf),
                    attribute + " has conflicting bounds")
            result[attribute] = normalized
        else:
            keys(policy, ("equals",), attribute + " constraint", ("equals",))
            require(isinstance(policy["equals"], str), "categorical equals must be a string")
            result[attribute] = dict(policy)
    return result


def validate_explanation(explanation, normalized):
    keys(explanation, ("claims",), "explanation", ("claims",))
    require(isinstance(explanation["claims"], list), "claims must be an array")
    for claim in explanation["claims"]:
        keys(claim, ("product_id", "attribute", "value"), "claim",
             ("product_id", "attribute", "value"))
        product_id, attribute = claim["product_id"], claim["attribute"]
        require(isinstance(product_id, str) and product_id in normalized,
                "explanation references an unlisted product")
        require(isinstance(attribute, str) and attribute in normalized[product_id],
                "explanation references an unsupported attribute")
        actual = normalized[product_id][attribute]
        require(actual is not None, "explanation references a missing attribute")
        if isinstance(actual, str):
            require(isinstance(claim["value"], str) and claim["value"] == actual,
                    "explanation contains an unsupported value")
        else:
            require(number(claim["value"], "claim value") == actual,
                    "explanation contains an unsupported value")
    return copy.deepcopy(explanation)


def compare(data, explanation_callback=None):
    """Return deterministic comparison; callback receives an isolated result copy."""
    keys(data, ("fixture", "schema", "products", "preferences", "constraints"),
         "input", ("fixture", "schema", "products"))
    require(data["fixture"] is True, "only explicitly labeled fixtures are supported")
    schema = data["schema"]
    validate_schema(schema)
    require(isinstance(data["products"], list), "products must be an array")
    preferences = policies(data.get("preferences", {}), schema, True)
    constraints = policies(data.get("constraints", {}), schema, False)
    normalized, missing = {}, {}
    for product in data["products"]:
        keys(product, ("id", "attributes"), "product", ("id", "attributes"))
        product_id = product["id"]
        require(isinstance(product_id, str) and bool(product_id.strip()), "invalid product ID")
        require(product_id not in normalized, "duplicate product ID: " + product_id)
        attributes = mapping(product["attributes"], product_id + " attributes")
        require(not set(attributes) - set(schema), product_id + " has undeclared attributes")
        values = {}
        for attribute, spec in schema.items():
            value = attributes.get(attribute)
            if value is not None:
                if spec["type"] == "numeric":
                    value = quantity(value, spec, product_id + "." + attribute)
                else:
                    require(isinstance(value, str), product_id + "." + attribute + " must be a string")
            values[attribute] = value
        normalized[product_id] = values
        missing[product_id] = [a for a, v in values.items() if v is None]

    eligible, excluded = [], []
    for product_id, values in normalized.items():
        reasons = []
        for attribute, constraint in constraints.items():
            value = values[attribute]
            if value is None:
                reasons.append({"attribute": attribute, "reason": "missing"})
            elif any((operator == "min" and value < bound)
                     or (operator == "max" and value > bound)
                     or (operator == "equals" and value != bound)
                     for operator, bound in constraint.items()):
                reasons.append({"attribute": attribute, "reason": "constraint_failed",
                                "value": value, "constraint": constraint})
        if reasons:
            excluded.append({"product_id": product_id, "reasons": reasons})
        else:
            eligible.append(product_id)

    comparison = []
    noncomparable = []
    for attribute, spec in schema.items():
        present = [p for p in normalized if normalized[p][attribute] is not None]
        comparable = len(present) >= 2
        row = {"attribute": attribute, "type": spec["type"], "unit": spec.get("unit"),
               "values": {p: v[attribute] for p, v in normalized.items()},
               "missing_product_ids": [p for p in normalized if p not in present],
               "comparable": comparable}
        comparison.append(row)
        if not comparable:
            noncomparable.append({"attribute": attribute, "reason": "fewer_than_two_known_values"})

    # Divide weights by their maximum first to avoid overflow in their sum.
    scale = max((p["weight"] for p in preferences.values()), default=1)
    total = math.fsum(p["weight"] / scale for p in preferences.values())
    utilities = {p: {} for p in eligible}
    for attribute, policy in preferences.items():
        known = [normalized[p][attribute] for p in eligible
                 if normalized[p][attribute] is not None]
        numeric = schema[attribute]["type"] == "numeric"
        low, high = (min(known), max(known)) if known and numeric else (0, 0)
        for product_id in eligible:
            value = normalized[product_id][attribute]
            if value is None:
                utility = 0.0
            elif not numeric:
                utility = float(value == policy["target"])
            elif low == high:
                utility = 1.0
            else:
                # Scaling prevents overflow for opposite-sign large finite values.
                magnitude = max(abs(low), abs(high))
                utility = ((value / magnitude - low / magnitude)
                           / (high / magnitude - low / magnitude))
                if policy["direction"] == "min":
                    utility = 1 - utility
            utilities[product_id][attribute] = utility
    ranking = []
    for product_id in eligible:
        contributions = {
            a: utilities[product_id][a] * (p["weight"] / scale / total)
            for a, p in preferences.items()
        }
        ranking.append({"product_id": product_id, "score": math.fsum(contributions.values()),
                        "utilities": utilities[product_id], "contributions": contributions,
                        "missing_attributes": missing[product_id]})
    ranking.sort(key=lambda entry: -entry["score"])
    for index, entry in enumerate(ranking, 1):
        entry["rank"] = index

    tradeoffs = []
    if ranking:
        selected = ranking[0]["product_id"]
        for other in ranking[1:]:
            other_id = other["product_id"]
            for attribute, policy in preferences.items():
                left, right = normalized[selected][attribute], normalized[other_id][attribute]
                if left is None or right is None:
                    continue
                difference = utilities[selected][attribute] - utilities[other_id][attribute]
                if difference:
                    tradeoffs.append({
                        "product_id": selected, "compared_to": other_id,
                        "attribute": attribute, "value": left, "compared_value": right,
                        "unit": schema[attribute].get("unit"),
                        "assessment": "advantage" if difference > 0 else "disadvantage",
                        "preference": policy,
                    })
    result = {
        "fixture": True, "status": "empty" if not normalized else
        ("ok" if eligible else "no_compatible_products"),
        "product_ids": list(normalized), "normalized_products": normalized,
        "comparison": comparison, "missing_attributes": missing,
        "noncomparable_attributes": noncomparable, "ranking": ranking,
        "excluded": excluded, "tradeoffs": tradeoffs,
        "ranking_method": "eligible-set min/max weighted utility; missing=0; stable input-order ties",
    }
    if explanation_callback is not None:
        require(callable(explanation_callback), "explanation callback must be callable")
        result["explanation"] = validate_explanation(
            explanation_callback(copy.deepcopy(result)), normalized)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Fixture JSON path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            data = json.load(sys.stdin, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        else:
            with open(args.input, encoding="utf-8") as handle:
                data = json.load(handle, object_pairs_hook=unique_object,
                                 parse_constant=reject_constant)
        result = compare(data)
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
