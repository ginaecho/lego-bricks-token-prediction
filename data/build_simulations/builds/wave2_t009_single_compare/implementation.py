"""Deterministic synthetic product comparison; Python standard library only."""

import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


UNITS = {
    "g": ("mass", 1.0), "kg": ("mass", 1000.0),
    "oz": ("mass", 28.349523125), "lb": ("mass", 453.59237),
    "mm": ("length", 1.0), "cm": ("length", 10.0),
    "m": ("length", 1000.0), "in": ("length", 25.4),
    "mb": ("storage", 1.0), "gb": ("storage", 1000.0),
    "tb": ("storage", 1000000.0), "usd": ("currency", 1.0),
}
NUMBER = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)")


def fail(message):
    raise ValidationError(message)


def object_fields(value, required, optional, path):
    if not isinstance(value, dict):
        fail(path + " must be an object")
    if set(required) - value.keys():
        fail(path + " missing fields: " + ", ".join(sorted(set(required) - value.keys())))
    if value.keys() - set(required) - set(optional):
        fail(path + " contains unknown fields")


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        fail(path + " must be nonempty text")
    return " ".join(value.split())


def sequence(value, path, low=1, high=50):
    if not isinstance(value, list) or not low <= len(value) <= high:
        fail(f"{path} must be a list with {low}..{high} entries")
    return value


def finite(value, path):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        fail(path + " must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        fail(path + " must be a finite number")
    if not math.isfinite(result):
        fail(path + " must be a finite number")
    return result


def normalize(value, spec, path):
    if value is None:
        return None
    kind = spec["type"]
    if kind == "text":
        return text(value, path).casefold()
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().casefold() in ("true", "false", "yes", "no"):
            return value.strip().casefold() in ("true", "yes")
        fail(path + " must be boolean or true/false/yes/no")
    source_unit = spec.get("unit")
    if isinstance(value, str):
        match = NUMBER.fullmatch(value.strip())
        if not match:
            fail(path + " must be a number optionally followed by a supported unit")
        value = finite(float(match[1]), path)
        suffix = match[2].lower()
        if suffix:
            source_unit = suffix
    else:
        value = finite(value, path)
    canonical = spec.get("unit")
    if source_unit:
        if source_unit not in UNITS or canonical not in UNITS:
            fail(path + " has an unsupported unit")
        if UNITS[source_unit][0] != UNITS[canonical][0]:
            fail(path + " has an incompatible unit")
        value *= UNITS[source_unit][1] / UNITS[canonical][1]
    return finite(value, path)


def validate(data):
    """Shared boundary: validate and normalize all stages' inputs once."""
    object_fields(data, ("schema_version", "fixture_label", "attributes", "products", "preferences"), (), "input")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        fail("schema_version must be 1")
    label = text(data["fixture_label"], "fixture_label")
    specs, aliases = {}, {}
    for raw in sequence(data["attributes"], "attributes", high=30):
        object_fields(raw, ("id", "label", "type"), ("unit", "aliases"), "attribute")
        spec = dict(raw)
        key = text(raw["id"], "attribute.id").casefold()
        if key in specs:
            fail("duplicate attribute id")
        spec["id"] = key
        spec["label"] = text(raw["label"], "attribute.label")
        if raw["type"] not in ("number", "text", "boolean"):
            fail("unsupported attribute type")
        if "unit" in raw:
            unit = text(raw["unit"], "attribute.unit").lower()
            if raw["type"] != "number" or unit not in UNITS:
                fail("unit requires a numeric attribute and supported unit")
            spec["unit"] = unit
        names = [key] + sequence(raw.get("aliases", []), "attribute.aliases", low=0, high=20)
        for name in names:
            alias = text(name, "attribute alias").casefold()
            if alias in aliases:
                fail("duplicate or ambiguous attribute alias")
            aliases[alias] = key
        specs[key] = spec
    products, ids = [], set()
    for raw in sequence(data["products"], "products", low=2):
        object_fields(raw, ("id", "name", "attributes"), (), "product")
        pid = text(raw["id"], "product.id")
        if pid in ids:
            fail("duplicate product id")
        ids.add(pid)
        name = text(raw["name"], "product.name")
        if not isinstance(raw["attributes"], dict):
            fail("product.attributes must be an object")
        values, seen = dict.fromkeys(specs), set()
        for raw_key, value in raw["attributes"].items():
            alias = text(raw_key, "product attribute key").casefold()
            if alias not in aliases:
                fail("unknown product attribute: " + alias)
            key = aliases[alias]
            if key in seen:
                fail("multiple values for attribute: " + key)
            seen.add(key)
            values[key] = normalize(value, specs[key], pid + "." + key)
        products.append({"id": pid, "name": name, "attributes": values})
    prefs, seen = [], set()
    for raw in sequence(data["preferences"], "preferences", high=30):
        object_fields(raw, ("attribute", "mode", "weight"), ("value",), "preference")
        alias = text(raw["attribute"], "preference.attribute").casefold()
        if alias not in aliases:
            fail("unknown preference attribute")
        key = aliases[alias]
        if key in seen:
            fail("duplicate preference attribute")
        seen.add(key)
        weight = finite(raw["weight"], "preference.weight")
        if weight <= 0:
            fail("preference.weight must be positive")
        mode = raw["mode"]
        if mode not in ("higher", "lower", "target", "equals"):
            fail("unknown preference mode")
        if mode in ("higher", "lower", "target") and specs[key]["type"] != "number":
            fail("numeric preference mode requires numeric attribute")
        pref = {"attribute": key, "mode": mode, "weight": weight}
        if mode in ("target", "equals"):
            if "value" not in raw or raw["value"] is None:
                fail("target/equals requires a non-null value")
            pref["value"] = normalize(raw["value"], specs[key], "preference.value")
        elif "value" in raw:
            fail("higher/lower must not specify value")
        prefs.append(pref)
    return label, specs, products, prefs


def compare(data):
    label, specs, products, prefs = validate(data)
    rows = [{
        "attribute": key, "label": spec["label"], "type": spec["type"],
        "unit": spec.get("unit"),
        "values": {p["id"]: p["attributes"][key] for p in products},
    } for key, spec in specs.items()]
    # Normalize by the largest weight first to avoid overflowing their sum.
    largest = max(p["weight"] for p in prefs)
    weights = [p["weight"] / largest for p in prefs]
    total = sum(weights)
    details = {p["id"]: [] for p in products}
    for pref, weight in zip(prefs, weights):
        key, mode = pref["attribute"], pref["mode"]
        available = [p["attributes"][key] for p in products if p["attributes"][key] is not None]
        # Scale before subtracting so even very large finite inputs remain safe.
        scale = max([abs(v) for v in available] + [abs(pref.get("value", 0)), 1.0]) if specs[key]["type"] == "number" else 1.0
        low = min(available) / scale if available and specs[key]["type"] == "number" else 0
        high = max(available) / scale if available and specs[key]["type"] == "number" else 0
        for product in products:
            value = product["attributes"][key]
            if value is None:
                score = 0.0
            elif mode == "equals":
                score = float(value == pref["value"])
            elif mode == "target":
                distance = abs(value / scale - pref["value"] / scale)
                span = high - low or max(abs(pref["value"]) / scale, 1.0 / scale)
                score = 1.0 / (1.0 + distance / span)
            elif high == low:
                score = 1.0
            elif mode == "higher":
                score = (value / scale - low) / (high - low)
            else:
                score = (high - value / scale) / (high - low)
            details[product["id"]].append({
                "attribute": key, "mode": mode, "missing": value is None,
                "score": round(score, 6), "weighted_score": score * weight / total,
            })
    ranking = []
    for product in products:
        contributions = details[product["id"]]
        score = sum(item["weighted_score"] for item in contributions)
        ranking.append({"product_id": product["id"], "score": round(score, 6), "contributions": contributions})
    ranking.sort(key=lambda item: (-item["score"], item["product_id"]))
    for rank, entry in enumerate(ranking, 1):
        entry["rank"] = rank
        for item in entry["contributions"]:
            item["weighted_score"] = round(item["weighted_score"], 6)
    return {
        "status": "ok", "schema_version": 1, "fixture_label": label,
        "normalized_products": products,
        "comparison": {"product_order": [p["id"] for p in products], "rows": rows},
        "ranking": ranking,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            fail("usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=lambda _: fail("nonfinite JSON number"))
        result = compare(data)
        output = json.dumps(result, ensure_ascii=True, allow_nan=False)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
