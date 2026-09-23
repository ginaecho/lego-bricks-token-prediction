"""Synthetic, deterministic compare -> research -> extraction reference pipeline.

Run: python -B implementation.py example_input.json
Offsets are zero-based, end-exclusive Python Unicode character offsets.
No provider calls, dependencies, persistence, or implicit document downloads.
"""

import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path


class ValidationError(ValueError):
    pass


def obj(properties, required=None, additional=False):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": additional}


def arr(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


def enum(*values):
    return {"enum": list(values)}


STRING = {"type": "string"}
TEXT = {"type": "string", "minLength": 1}
NUMBER = {"type": "number"}
INTEGER = {"type": "integer", "minimum": 0}
BOOL = {"type": "boolean"}
NULL = {"type": "null"}
VALUE = {"anyOf": [NUMBER, STRING]}
MAYBE_VALUE = {"anyOf": [VALUE, NULL]}
MAYBE_TEXT = {"anyOf": [STRING, NULL]}
RAW_VALUE = {"anyOf": [VALUE, obj({"value": NUMBER, "unit": TEXT})]}

ATTRIBUTE = obj({
    "id": TEXT, "aliases": arr(TEXT), "kind": enum("number", "text"),
    "unit": MAYBE_TEXT,
    "units": obj({}, required=[], additional=NUMBER),
})
PREFERENCE = obj({
    "attribute": TEXT, "direction": enum("min", "max", "match"),
    "weight": {"type": "number", "exclusiveMinimum": 0}, "target": VALUE,
}, required=["attribute", "direction", "weight"])
FIELD = obj({
    "id": TEXT, "type": enum("string", "number", "integer", "boolean"),
    "required": BOOL,
    "source": enum("name", "score", "attribute", "summary", "evidence_count",
                   "unresolved_count", "has_disagreement"),
    "product_id": TEXT, "attribute": TEXT,
}, required=["id", "type", "required", "source"])
INPUT_SCHEMA = obj({
    "schema_version": enum("1.0"), "synthetic": enum(True),
    "attributes": arr(ATTRIBUTE, 1),
    "products": arr(obj({
        "id": TEXT, "name": TEXT,
        "attributes": obj({}, required=[], additional=RAW_VALUE),
    }), 1),
    "preferences": arr(PREFERENCE, 1),
    "documents": arr(obj({"id": TEXT, "product_id": TEXT, "text": STRING})),
    "extraction_schema": arr(FIELD, 1),
})
SPAN = obj({"document_id": TEXT, "start": INTEGER, "end": INTEGER, "quote": TEXT})
NORMALIZED = obj({"value": VALUE, "unit": MAYBE_TEXT})
PRODUCT = obj({
    "id": TEXT, "name": TEXT, "rank": {"type": "integer", "minimum": 1},
    "score": {"type": "number", "minimum": 0, "maximum": 1},
    "attributes": obj({}, required=[], additional=NORMALIZED),
    "preference_scores": obj({}, required=[], additional={
        "type": "number", "minimum": 0, "maximum": 1}),
})
COMPARE_SCHEMA = obj({
    "stage": enum("compare"), "products": arr(PRODUCT, 1),
    "columns": arr(TEXT, 1), "winner_id": TEXT,
    "side_by_side": arr(obj({
        "attribute": TEXT, "unit": MAYBE_TEXT,
        "values": obj({}, required=[], additional=MAYBE_VALUE),
    }), 1),
})
EVIDENCE = obj({"value": VALUE, "source_span": SPAN})
FINDING = obj({
    "attribute": TEXT, "unit": MAYBE_TEXT, "catalog_value": MAYBE_VALUE,
    "consensus_value": MAYBE_VALUE,
    "status": enum("supported", "disputed", "unresolved"),
    "evidence": arr(EVIDENCE), "distinct_values": arr(VALUE),
    "document_count": INTEGER,
})
RESEARCH_PRODUCT = obj({
    **PRODUCT["properties"], "findings": arr(FINDING, 1),
    "summary": TEXT, "unresolved_questions": arr(TEXT),
})
RESEARCH_SCHEMA = obj({
    "stage": enum("deep"), "winner_id": TEXT,
    "products": arr(RESEARCH_PRODUCT, 1),
    "documents_used": arr(TEXT), "disagreements": arr(obj({
        "product_id": TEXT, "attribute": TEXT, "values": arr(VALUE),
        "catalog_value": MAYBE_VALUE,
    })),
})
EXTRACTED_FIELD = obj({
    "value": {"anyOf": [VALUE, BOOL, NULL]},
    "status": enum("found", "missing"), "reason": MAYBE_TEXT,
    "unit": MAYBE_TEXT, "source_path": TEXT, "source_spans": arr(SPAN),
})
EXTRACT_SCHEMA = obj({
    "stage": enum("extract"), "complete": BOOL,
    "fields": obj({}, required=[], additional=EXTRACTED_FIELD),
    "missing_fields": arr(TEXT), "required_missing": arr(TEXT),
})
OUTPUT_SCHEMA = obj({
    "status": enum("ok"), "schema_version": enum("1.0"), "synthetic": enum(True),
    "comparison": COMPARE_SCHEMA, "research": RESEARCH_SCHEMA,
    "extraction": EXTRACT_SCHEMA,
})


def validate(value, schema, path="$"):
    """Small strict schema subset shared by input and all stage boundaries."""
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                validate(value, choice, path)
                return
            except ValidationError:
                pass
        raise ValidationError(f"{path}: does not match an allowed type")
    if "enum" in schema:
        if not any(type(value) is type(option) and value == option
                   for option in schema["enum"]):
            raise ValidationError(f"{path}: expected one of {schema['enum']}")
        return
    kind = schema["type"]
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": type(value) in (int, float),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
    }[kind]
    if not valid:
        raise ValidationError(f"{path}: expected {kind}")
    if kind in ("number", "integer"):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValidationError(f"{path}: number must be finite")
        for limit, op in (
            ("minimum", lambda a, b: a >= b),
            ("maximum", lambda a, b: a <= b),
            ("exclusiveMinimum", lambda a, b: a > b),
        ):
            if limit in schema and not op(value, schema[limit]):
                raise ValidationError(f"{path}: violates {limit}")
    elif kind == "string" and len(value.strip()) < schema.get("minLength", 0):
        raise ValidationError(f"{path}: must not be blank")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0):
            raise ValidationError(f"{path}: too few items")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
    elif kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValidationError(f"{path}.{key}: required")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path}: keys must be strings")
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValidationError(f"{path}.{key}: unknown field")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate(item, schema["additionalProperties"], f"{path}.{key}")


def key(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().casefold()).strip("_")


def unique(items, label):
    if len(items) != len(set(items)):
        raise ValidationError(f"{label}: duplicate identifier")


def normalize(raw, definition):
    if definition["kind"] == "text":
        if not isinstance(raw, str) or not raw.strip():
            raise ValidationError(f"{definition['id']}: expected nonempty text")
        return " ".join(raw.casefold().split())
    if type(raw) in (int, float):
        amount, unit = raw, definition["unit"]
    elif isinstance(raw, dict):
        amount, unit = raw["value"], raw["unit"]
    elif isinstance(raw, str):
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*",
            raw,
        )
        if not match:
            raise ValidationError(f"{definition['id']}: expected quantity")
        amount = float(match[1])
        unit = match[2] or definition["unit"]
    else:
        raise ValidationError(f"{definition['id']}: invalid quantity")
    units = {u.casefold(): factor for u, factor in definition["units"].items()}
    if unit is None:
        factor = 1
    elif unit.casefold() in units:
        factor = units[unit.casefold()]
    else:
        raise ValidationError(f"{definition['id']}: unsupported unit {unit!r}")
    try:
        result = amount * factor
        if not math.isfinite(result):
            raise ValidationError(f"{definition['id']}: quantity must be finite")
        return round(result, 12)
    except OverflowError as exc:
        raise ValidationError(f"{definition['id']}: quantity out of range") from exc


def validate_input(data):
    validate(data, INPUT_SCHEMA)
    definitions = {a["id"]: a for a in data["attributes"]}
    unique([a["id"] for a in data["attributes"]], "attributes")
    aliases = {}
    for definition in definitions.values():
        unit = definition["unit"]
        units = definition["units"]
        unique([u.casefold() for u in units], "units")
        if definition["kind"] == "text":
            if unit is not None or units:
                raise ValidationError("text attributes cannot have units")
        elif unit is None:
            if units:
                raise ValidationError("dimensionless attributes cannot have units")
        elif not unit.strip() or units.get(unit) != 1:
            raise ValidationError("canonical unit must have conversion factor 1")
        if any(factor <= 0 for factor in units.values()):
            raise ValidationError("unit factors must be positive")
        for alias in [definition["id"], *definition["aliases"]]:
            normalized = key(alias)
            if not normalized:
                raise ValidationError("attribute aliases must contain ASCII letters/digits")
            if normalized in aliases and aliases[normalized] != definition["id"]:
                raise ValidationError(f"ambiguous attribute alias: {alias}")
            aliases[normalized] = definition["id"]
    ids = [p["id"] for p in data["products"]]
    unique(ids, "products")
    for product in data["products"]:
        seen = []
        for alias, value in product["attributes"].items():
            if key(alias) not in aliases:
                raise ValidationError(f"unknown product attribute: {alias}")
            attribute = aliases[key(alias)]
            seen.append(attribute)
            normalize(value, definitions[attribute])
        unique(seen, f"product {product['id']} attributes")
    unique([p["attribute"] for p in data["preferences"]], "preferences")
    for preference in data["preferences"]:
        attribute = preference["attribute"]
        if attribute not in definitions:
            raise ValidationError(f"unknown preference attribute: {attribute}")
        if preference["direction"] == "match":
            if "target" not in preference:
                raise ValidationError("match preference requires target")
            normalize(preference["target"], definitions[attribute])
        elif definitions[attribute]["kind"] != "number" or "target" in preference:
            raise ValidationError("min/max preferences require numeric attributes and no target")
    unique([d["id"] for d in data["documents"]], "documents")
    for document in data["documents"]:
        if document["product_id"] not in ids:
            raise ValidationError("document references unknown product")
    unique([f["id"] for f in data["extraction_schema"]], "extraction fields")
    for field in data["extraction_schema"]:
        if "product_id" in field and field["product_id"] not in ids:
            raise ValidationError("extraction field references unknown product")
        if field["source"] == "attribute":
            if field.get("attribute") not in definitions:
                raise ValidationError("attribute extraction requires a known attribute")
            expected = "number" if definitions[field["attribute"]]["kind"] == "number" else "string"
        else:
            if "attribute" in field:
                raise ValidationError("attribute selector only applies to attribute extraction")
            expected = {
                "name": "string", "score": "number", "summary": "string",
                "evidence_count": "integer", "unresolved_count": "integer",
                "has_disagreement": "boolean",
            }[field["source"]]
        if field["type"] != expected:
            raise ValidationError(f"field {field['id']}: expected declared type {expected}")
    return definitions, aliases


def compare(data):
    definitions, aliases = validate_input(data)
    products = []
    for raw in data["products"]:
        attributes = {}
        for alias, value in raw["attributes"].items():
            attribute = aliases[key(alias)]
            attributes[attribute] = {
                "value": normalize(value, definitions[attribute]),
                "unit": definitions[attribute]["unit"],
            }
        products.append({"id": raw["id"], "name": raw["name"],
                         "attributes": attributes, "preference_scores": {}})
    # Scaling first avoids overflow for large, individually finite weights.
    scale = max(p["weight"] for p in data["preferences"])
    weights = [p["weight"] / scale for p in data["preferences"]]
    for preference in data["preferences"]:
        attribute = preference["attribute"]
        values = [p["attributes"][attribute]["value"] for p in products
                  if attribute in p["attributes"]]
        for product in products:
            if attribute not in product["attributes"]:
                score = 0.0
            else:
                value = product["attributes"][attribute]["value"]
                if preference["direction"] == "match":
                    target = normalize(preference["target"], definitions[attribute])
                    score = float(value == target)
                else:
                    low, high = min(values), max(values)
                    if low == high:
                        score = 1.0
                    else:
                        # Exact intermediate differences avoid overflow and
                        # loss of a small integer gap near a very large value.
                        score = float((Fraction(value) - Fraction(low))
                                      / (Fraction(high) - Fraction(low)))
                        if preference["direction"] == "min":
                            score = 1 - score
                        score = min(1.0, max(0.0, score))
            product["preference_scores"][attribute] = round(score, 12)
    for product in products:
        product["score"] = round(sum(
            weight * product["preference_scores"][preference["attribute"]]
            for weight, preference in zip(weights, data["preferences"])
        ) / sum(weights), 12)
    products.sort(key=lambda p: (-p["score"], p["id"]))
    for rank, product in enumerate(products, 1):
        product["rank"] = rank
    result = {
        "stage": "compare", "products": products,
        "columns": [p["id"] for p in products], "winner_id": products[0]["id"],
        "side_by_side": [
            {"attribute": attribute, "unit": definition["unit"],
             "values": {p["id"]: p["attributes"].get(attribute, {}).get("value")
                        for p in products}}
            for attribute, definition in definitions.items()
        ],
    }
    validate(result, COMPARE_SCHEMA)
    return result


def validate_handoff(stage, schema, data):
    validate(stage, schema)
    products = stage["products"]
    if ([p["rank"] for p in products] != list(range(1, len(products) + 1))
            or stage["winner_id"] != products[0]["id"]
            or len(products) != len(data["products"])
            or {p["id"] for p in products} != {p["id"] for p in data["products"]}):
        raise ValidationError("inconsistent stage product identities/ranks")
    if products != sorted(products, key=lambda p: (-p["score"], p["id"])):
        raise ValidationError("inconsistent stage ranking")
    definitions = {a["id"]: a for a in data["attributes"]}
    documents = {d["id"]: d for d in data["documents"]}
    preference_ids = {p["attribute"] for p in data["preferences"]}
    used, disagreements = set(), []
    if stage["stage"] == "compare":
        expected_table = [
            {"attribute": attribute, "unit": definition["unit"],
             "values": {p["id"]: p["attributes"].get(attribute, {}).get("value")
                        for p in products}}
            for attribute, definition in definitions.items()
        ]
        if (stage["columns"] != [p["id"] for p in products]
                or stage["side_by_side"] != expected_table):
            raise ValidationError("inconsistent side-by-side comparison")
    for product in products:
        if set(product["preference_scores"]) != preference_ids:
            raise ValidationError("inconsistent preference score attributes")
        for attribute, normalized in product["attributes"].items():
            if attribute not in definitions or normalized["unit"] != definitions[attribute]["unit"]:
                raise ValidationError("inconsistent normalized stage attribute")
            expected = NUMBER if definitions[attribute]["kind"] == "number" else TEXT
            validate(normalized["value"], expected)
        if stage["stage"] == "deep":
            if [f["attribute"] for f in product["findings"]] != list(definitions):
                raise ValidationError("research findings must cover schema attributes in order")
            for finding in product["findings"]:
                attribute = finding["attribute"]
                definition = definitions[attribute]
                expected_type = NUMBER if definition["kind"] == "number" else TEXT
                distinct, document_ids = [], set()
                for evidence in finding["evidence"]:
                    validate(evidence["value"], expected_type)
                    span = evidence["source_span"]
                    document = documents.get(span["document_id"])
                    if (document is None or document["product_id"] != product["id"]
                            or not 0 <= span["start"] < span["end"] <= len(document["text"])
                            or document["text"][span["start"]:span["end"]] != span["quote"]):
                        raise ValidationError("invalid research evidence source span")
                    if normalize(span["quote"], definition) != evidence["value"]:
                        raise ValidationError("evidence value does not match its source")
                    if evidence["value"] not in distinct:
                        distinct.append(evidence["value"])
                    document_ids.add(span["document_id"])
                used.update(document_ids)
                catalog = product["attributes"].get(attribute, {}).get("value")
                status, consensus = "unresolved", None
                if distinct:
                    if len(distinct) > 1 or (catalog is not None and catalog != distinct[0]):
                        status = "disputed"
                        disagreements.append({
                            "product_id": product["id"], "attribute": attribute,
                            "values": distinct, "catalog_value": catalog,
                        })
                    else:
                        status, consensus = "supported", distinct[0]
                if (finding["unit"] != definition["unit"]
                        or finding["catalog_value"] != catalog
                        or finding["distinct_values"] != distinct
                        or finding["document_count"] != len(document_ids)
                        or finding["status"] != status
                        or finding["consensus_value"] != consensus):
                    raise ValidationError("research synthesis is inconsistent with evidence")
    if stage["stage"] == "deep":
        if stage["documents_used"] != sorted(used) or stage["disagreements"] != disagreements:
            raise ValidationError("research aggregate evidence is inconsistent")


def deep_research(data, comparison):
    definitions, aliases = validate_input(data)
    validate_handoff(comparison, COMPARE_SCHEMA, data)
    products, disagreements, used = [], [], set()
    for compared in comparison["products"]:
        evidence = {attribute: [] for attribute in definitions}
        unresolved = []
        for document in data["documents"]:
            if document["product_id"] != compared["id"]:
                continue
            offset = 0
            for line in document["text"].splitlines(keepends=True):
                match = re.fullmatch(r"\s*([^:\r\n]+):[ \t]*(.*?)[ \t]*(?:\r?\n)?", line)
                if match and key(match[1]) in aliases:
                    attribute = aliases[key(match[1])]
                    try:
                        value = normalize(match[2], definitions[attribute])
                    except ValidationError:
                        unresolved.append(
                            f"Can the invalid {attribute} claim in {document['id']} "
                            f"at offset {offset} be clarified?")
                    else:
                        start, end = offset + match.start(2), offset + match.end(2)
                        evidence[attribute].append({
                            "value": value, "source_span": {
                                "document_id": document["id"], "start": start,
                                "end": end, "quote": document["text"][start:end],
                            },
                        })
                        used.add(document["id"])
                offset += len(line)
        findings = []
        for attribute, definition in definitions.items():
            claims = evidence[attribute]
            values = []
            for claim in claims:
                if claim["value"] not in values:
                    values.append(claim["value"])
            catalog = compared["attributes"].get(attribute, {}).get("value")
            if not values:
                status, consensus = "unresolved", None
                unresolved.append(f"What independent evidence establishes {attribute}?")
            elif len(values) > 1 or (catalog is not None and catalog != values[0]):
                status, consensus = "disputed", None
                disagreements.append({"product_id": compared["id"], "attribute": attribute,
                                      "values": values, "catalog_value": catalog})
                unresolved.append(f"Which conflicting {attribute} value is correct?")
            else:
                status, consensus = "supported", values[0]
            findings.append({
                "attribute": attribute, "unit": definition["unit"],
                "catalog_value": catalog, "consensus_value": consensus,
                "status": status, "evidence": claims, "distinct_values": values,
                "document_count": len({c["source_span"]["document_id"] for c in claims}),
            })
        counts = {status: sum(f["status"] == status for f in findings)
                  for status in ("supported", "disputed", "unresolved")}
        summary = (
            f"{compared['name']} (comparison rank {compared['rank']}): "
            f"{counts['supported']} supported, {counts['disputed']} disputed, "
            f"{counts['unresolved']} without evidence. "
            f"{len(unresolved)} unresolved questions. Support is not independent verification."
        )
        products.append({**compared, "findings": findings, "summary": summary,
                         "unresolved_questions": unresolved})
    result = {"stage": "deep", "winner_id": comparison["winner_id"],
              "products": products, "documents_used": sorted(used),
              "disagreements": disagreements}
    validate_handoff(result, RESEARCH_SCHEMA, data)
    return result


def extract(data, research):
    validate_input(data)
    validate_handoff(research, RESEARCH_SCHEMA, data)
    fields, missing, required_missing = {}, [], []
    products = {p["id"]: p for p in research["products"]}
    for spec in data["extraction_schema"]:
        product_id = spec.get("product_id", research["winner_id"])
        product = products[product_id]
        source = spec["source"]
        path = f"research.products[{product['rank'] - 1}]"
        spans, reason, unit = [], None, None
        if source == "attribute":
            index = next(i for i, f in enumerate(product["findings"])
                         if f["attribute"] == spec["attribute"])
            finding = product["findings"][index]
            path += f".findings[{index}].consensus_value"
            value, unit = finding["consensus_value"], finding["unit"]
            if finding["status"] != "supported":
                value = None
                reason = ("conflicting evidence" if finding["status"] == "disputed"
                          else "no usable documentary evidence")
            else:
                spans = [claim["source_span"] for claim in finding["evidence"]]
        elif source in ("name", "score", "summary"):
            value = product[source]
            path += f".{source}"
            if source == "summary":
                spans = [claim["source_span"] for f in product["findings"]
                         for claim in f["evidence"]]
        elif source == "evidence_count":
            value = sum(len(f["evidence"]) for f in product["findings"])
            path += ".findings[*].evidence"
            spans = [claim["source_span"] for f in product["findings"]
                     for claim in f["evidence"]]
        elif source == "unresolved_count":
            value = len(product["unresolved_questions"])
            path += ".unresolved_questions"
        else:
            value = any(f["status"] == "disputed" for f in product["findings"])
            path += ".findings[*].status"
            spans = [claim["source_span"] for f in product["findings"]
                     if f["status"] == "disputed" for claim in f["evidence"]]
        if value is None:
            missing.append(spec["id"])
            if spec["required"]:
                required_missing.append(spec["id"])
        else:
            validate(value, {"type": spec["type"]}, f"extraction.{spec['id']}")
        fields[spec["id"]] = {
            "value": value, "status": "missing" if value is None else "found",
            "reason": reason, "unit": unit, "source_path": path, "source_spans": spans,
        }
    result = {"stage": "extract", "complete": not required_missing, "fields": fields,
              "missing_fields": missing, "required_missing": required_missing}
    validate(result, EXTRACT_SCHEMA)
    return result


def run_pipeline(data):
    validate_input(data)
    comparison = compare(data)
    research = deep_research(data, comparison)
    extraction = extract(data, research)
    result = {"status": "ok", "schema_version": "1.0", "synthetic": True,
              "comparison": comparison, "research": research, "extraction": extraction}
    validate(result, OUTPUT_SCHEMA)
    return result


def strict_object(pairs):
    result = {}
    for key_, value in pairs:
        if key_ in result:
            raise ValidationError(f"duplicate JSON key: {key_}")
        result[key_] = value
    return result


def reject_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=strict_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
