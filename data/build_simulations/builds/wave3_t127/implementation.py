"""Synthetic insurance extraction reference. Python standard library only.

Spans are zero-based, end-exclusive Unicode character offsets into the original
document content. JSON spans include the complete value literal; XML spans cover
escaped element text; claim-form spans cover the value after its label.
"""

import datetime
import json
import math
import re
import sys
import xml.etree.ElementTree as ET


class ValidationError(ValueError):
    pass


FIELDS = {
    "policy": {
        "policy_id": "identifier", "policyholder_id": "identifier",
        "effective_date": "date", "coverage_limit": "amount",
        "property_region": "string",
    },
    "claim": {
        "claim_id": "identifier", "policy_id": "identifier",
        "loss_date": "date", "loss_amount": "amount",
        "decision": "string", "decision_reason": "string",
    },
    "underwriting_submission": {
        "submission_id": "identifier", "policy_id": "identifier",
        "risk_factors": "factors", "disclosed_factors": "factors",
    },
}
FACTORS = {"roof_age", "building_material", "annual_mileage", "prior_claim_count"}
REASONS = {
    "approved": "Covered loss under the stated policy",
    "denied": "Loss excluded by stated policy terms",
    "pending": "Additional loss documentation required",
}
MAX_CONTENT = 100_000


def fail(message):
    raise ValidationError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("Duplicate JSON keys are not allowed")
        result[key] = value
    return result


def load_json(text):
    return json.loads(
        text, object_pairs_hook=unique_object,
        parse_constant=lambda _: fail("Non-finite JSON numbers are not allowed"),
    )


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"synthetic", "schema", "documents"}:
        fail("Input must contain exactly synthetic, schema and documents")
    if request["synthetic"] is not True:
        fail("Only explicitly labeled synthetic inputs are supported")
    schema, documents = request["schema"], request["documents"]
    if not isinstance(schema, list) or not schema or len(schema) > 50:
        fail("schema must be a nonempty list of at most 50 fields")
    seen = set()
    for field in schema:
        if not isinstance(field, dict) or set(field) != {"entity", "name", "required"}:
            fail("Each schema field requires entity, name and required")
        entity, name = field["entity"], field["name"]
        if not isinstance(entity, str) or not isinstance(name, str):
            fail("Schema entity and name must be strings")
        if entity not in FIELDS or name not in FIELDS[entity]:
            fail("Requested field is outside the minimized extraction allowlist")
        if type(field["required"]) is not bool:
            fail("Schema required must be boolean")
        if (entity, name) in seen:
            fail("Duplicate schema field")
        seen.add((entity, name))
    if not isinstance(documents, list) or not documents or len(documents) > 20:
        fail("documents must be a nonempty list of at most 20 documents")
    ids = set()
    for doc in documents:
        if not isinstance(doc, dict) or set(doc) != {"id", "entity", "format", "content"}:
            fail("Each document requires id, entity, format and content")
        if not isinstance(doc["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", doc["id"]):
            fail("Invalid document identifier")
        if doc["id"] in ids:
            fail("Duplicate document identifier")
        ids.add(doc["id"])
        if not isinstance(doc["entity"], str) or doc["entity"] not in FIELDS:
            fail("Unknown document entity")
        if not any(field["entity"] == doc["entity"] for field in schema):
            fail("Every document entity must have an extraction schema")
        if not isinstance(doc["format"], str) or doc["format"] not in {"json", "xml", "text"}:
            fail("Unsupported document format")
        if not isinstance(doc["content"], str) or not doc["content"].strip():
            fail("Document content must be nonempty text")
        if len(doc["content"]) > MAX_CONTENT:
            fail("Document exceeds bounded content limit")


def extract_candidates(doc):
    content, fmt = doc["content"], doc["format"]
    allowed = FIELDS[doc["entity"]]
    candidates = {}

    def add(name, value, start, end):
        if name not in allowed:
            return
        if name in candidates:
            fail("Ambiguous duplicate source field")
        candidates[name] = (value, start, end)

    if fmt == "json":
        obj = load_json(content)
        if not isinstance(obj, dict):
            fail("JSON submission must be an object")
        decoder = json.JSONDecoder()

        def skip(pos):
            while pos < len(content) and content[pos].isspace():
                pos += 1
            return pos

        def walk(pos):
            pos = skip(pos + 1)
            while content[pos] != "}":
                key, pos = decoder.raw_decode(content, pos)
                pos = skip(pos)
                pos = skip(pos + 1)
                start = pos
                value, end = decoder.raw_decode(content, pos)
                if key in allowed:
                    add(key, value, start, end)
                elif isinstance(value, dict):
                    walk(start)
                elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
                    fail("Object arrays are outside the supported submission subset")
                pos = skip(end)
                if content[pos] == ",":
                    pos = skip(pos + 1)
            return pos + 1

        walk(skip(0))
    elif fmt == "xml":
        if re.search(r"<!|<\?", content):
            fail("XML declarations, entities, comments and CDATA are unsupported")
        root = ET.fromstring(content)
        elements = list(root.iter())
        if any(element.attrib or "}" in element.tag for element in elements):
            fail("XML attributes and namespaces are outside the supported subset")
        if any(element.tag in allowed and len(element) for element in elements):
            fail("Extractable XML fields must contain text only")
        if any((len(element) and (element.text or "").strip()) or
               (element.tail or "").strip() for element in elements):
            fail("Mixed XML content is unsupported")
        pattern = r"<([A-Za-z_][\w.-]*)\s*>([^<]*)</\1\s*>|<([A-Za-z_][\w.-]*)\s*/>"
        for match in re.finditer(pattern, content):
            name = match.group(1) or match.group(3)
            if match.group(1):
                value = ET.fromstring(match.group(0)).text or ""
                start, end = match.span(2)
            else:
                value = ""
                start = end = match.end()
            add(name, value, start, end)
    else:
        pattern = r"^[ \t]*([a-z_]+)[ \t]*:[ \t]*([^\r\n]*)"
        for match in re.finditer(pattern, content, re.MULTILINE):
            add(match.group(1), match.group(2), *match.span(2))
    return candidates


def normalize(value, kind):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if kind == "amount":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            fail("Amount must be a finite nonnegative number")
        try:
            number = float(value)
        except (ValueError, OverflowError):
            fail("Amount must be a finite nonnegative number")
        if not math.isfinite(number) or number < 0 or number > 1_000_000_000:
            fail("Amount is outside the demonstrative allowed range")
        return number
    if kind == "factors":
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            fail("Underwriting factors must be a string list or comma-separated text")
        if len(value) != len(set(value)) or any(item not in FACTORS for item in value):
            fail("Underwriting factors must be unique disclosed, permitted non-protected factors")
        return value
    if not isinstance(value, str):
        fail("Text field must be a string")
    value = value.strip()
    if len(value) > 200:
        fail("Text field exceeds bounded length")
    if kind == "identifier" and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        fail("Invalid entity identifier")
    if kind == "date":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            fail("Date must use YYYY-MM-DD")
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            fail("Invalid calendar date")
    return value


def validate_record(entity, values):
    """One shared validation layer, independent of requested output fields."""
    if entity == "claim":
        decision, reason = values.get("decision"), values.get("decision_reason")
        if decision is not None:
            if decision not in REASONS or reason != REASONS[decision]:
                fail("Claim decisions require a compatible, stated policy/evidence-based reason")
        elif reason is not None:
            fail("Claim reason requires a decision")
    if entity == "underwriting_submission":
        used, disclosed = values.get("risk_factors"), values.get("disclosed_factors")
        if used is not None and disclosed is None:
            fail("Used underwriting factors must be explicitly disclosed")
        if used is not None and not set(used).issubset(set(disclosed)):
            fail("Hidden underwriting factors are not permitted")


def process(request):
    validate_request(request)
    records = []
    incomplete = False
    for doc in request["documents"]:
        candidates = extract_candidates(doc)
        values = {
            name: normalize(candidate[0], FIELDS[doc["entity"]][name])
            for name, candidate in candidates.items()
        }
        validate_record(doc["entity"], values)
        fields, missing = {}, []
        for spec in request["schema"]:
            if spec["entity"] != doc["entity"]:
                continue
            name = spec["name"]
            value = values.get(name)
            if value is None:
                missing.append({"field": name, "required": spec["required"]})
                incomplete |= spec["required"]
            else:
                _, start, end = candidates[name]
                fields[name] = {
                    "value": value,
                    "source": {"document_id": doc["id"], "start": start, "end": end},
                }
        records.append({
            "document_id": doc["id"], "entity": doc["entity"],
            "fields": fields, "missing_fields": missing,
        })
    return {
        "status": "incomplete" if incomplete else "ok",
        "synthetic": True, "records": records,
        "validation": {
            "claim_decisions": "Extracted only; compatible stated reasons required",
            "data_minimization": "Allowlisted output; no raw documents or unrequested fields",
            "underwriting": "Only permitted factors; every used factor must be disclosed",
            "notice": "Demonstrative rules only; not compliance certification or adjudication",
        },
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            fail("Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            text = handle.read(2_100_001)
        if len(text) > 2_100_000:
            fail("Input file exceeds bounded size limit")
        result = process(load_json(text))
    except ValidationError as exc:
        result = {"status": "error", "message": str(exc)}
    except (OSError, UnicodeError):
        result = {"status": "error", "message": "Unable to read UTF-8 input file"}
    except (ValueError, ET.ParseError, RecursionError, OverflowError):
        result = {"status": "error", "message": "Malformed or excessively nested input"}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
