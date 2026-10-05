"""Synthetic-reference document extraction -> interest ranking CLI.

Run: python -B implementation.py example_input.json
Only JSON documents are supported; dotted paths traverse objects, not arrays.
"""

import json
import math
from pathlib import Path
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_keys(value, required, optional=(), where="object"):
    require(isinstance(value, dict), f"{where} must be an object")
    require(set(required) <= value.keys(), f"{where} missing required fields")
    require(value.keys() <= set(required) | set(optional), f"{where} has unknown fields")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where} must be nonempty text")
    return " ".join(value.split())


def number(value, low, high, where):
    require(type(value) in (int, float), f"{where} must be numeric")
    require(low <= value <= high, f"{where} must be finite and between {low} and {high}")
    return float(value)


def strings(value, where, normalize=False):
    require(isinstance(value, list), f"{where} must be an array")
    result = [text(x, where) for x in value]
    if normalize:
        result = [x.casefold() for x in result]
    return sorted(set(result))


def validate_item(item):
    """The single canonical contract used by both stages."""
    object_keys(item, ("id", "title", "description", "tags", "quality", "source"), where="item")
    for key in ("id", "title", "description"):
        require(text(item[key], key) == item[key], f"{key} is not canonical")
    require(strings(item["tags"], "tags", True) == item["tags"], "tags are not canonical")
    number(item["quality"], 0, 1, "quality")
    object_keys(item["source"], ("document_id", "row", "fields"), where="source")
    text(item["source"]["document_id"], "document_id")
    require(type(item["source"]["row"]) is int and item["source"]["row"] >= 0,
            "source.row must be a nonnegative integer")
    object_keys(item["source"]["fields"], ("id", "title", "description", "tags"),
                ("quality",), "source.fields")
    for path in item["source"]["fields"].values():
        validate_path(path)
    return item


def validate_path(path):
    require(isinstance(path, str) and bool(path), "field path must be nonempty text")
    require(all(part and part == part.strip() for part in path.split(".")),
            "field path has an empty or padded segment")
    return path


def extract(row, path):
    node = row
    for part in path.split("."):
        require(isinstance(node, dict) and part in node, f"missing source field: {path}")
        node = node[part]
    return node


def automate_documents(config):
    """Extract configured columns, normalize them and reject invalid batches atomically."""
    require(isinstance(config, list), "documents must be an array")
    items, ids, document_ids = [], set(), set()
    for doc in config:
        object_keys(doc, ("document_id", "fields", "rows"), where="document")
        document_id = text(doc["document_id"], "document_id")
        require(document_id not in document_ids, "duplicate document_id")
        document_ids.add(document_id)
        fields = doc["fields"]
        object_keys(fields, ("id", "title", "description", "tags"), ("quality",), "fields")
        for path in fields.values():
            validate_path(path)
        require(isinstance(doc["rows"], list), "rows must be an array")
        for index, row in enumerate(doc["rows"]):
            require(isinstance(row, dict), "row must be an object")
            values = {name: extract(row, path) for name, path in fields.items()}
            raw_tags = values["tags"]
            if isinstance(raw_tags, str):
                raw_tags = [tag for tag in raw_tags.split(",") if tag.strip()]
            item = {
                "id": text(values["id"], "id"),
                "title": text(values["title"], "title"),
                "description": text(values["description"], "description"),
                "tags": strings(raw_tags, "tags", True),
                "quality": number(values.get("quality", 0), 0, 1, "quality"),
                "source": {"document_id": document_id, "row": index, "fields": dict(fields)},
            }
            validate_item(item)
            require(item["id"] not in ids, f"duplicate item id: {item['id']}")
            ids.add(item["id"])
            items.append(item)
    return items


def validate_profile(profile):
    object_keys(profile, ("weights", "exclusions", "limit"), where="profile")
    require(isinstance(profile["weights"], dict), "weights must be an object")
    weights = {}
    for tag, weight in profile["weights"].items():
        key = text(tag, "interest tag").casefold()
        require(key not in weights, "duplicate normalized interest tag")
        weights[key] = number(weight, -10, 10, "interest weight")
    object_keys(profile["exclusions"], ("ids", "tags", "terms"), where="exclusions")
    exclusions = {
        key: strings(value, f"exclusions.{key}", key != "ids")
        for key, value in profile["exclusions"].items()
    }
    require(type(profile["limit"]) is int and 0 <= profile["limit"] <= 100,
            "limit must be an integer between 0 and 100")
    return weights, exclusions, profile["limit"]


def recommend(items, profile):
    """Consume canonical documents only; exclusions take precedence over all scores."""
    weights, exclusions, limit = validate_profile(profile)
    require(isinstance(items, list), "catalog must be an array")
    ranked, excluded, ids = [], [], set()
    for item in items:
        validate_item(item)
        require(item["id"] not in ids, "duplicate catalog id")
        ids.add(item["id"])
        reasons = []
        if item["id"] in exclusions["ids"]:
            reasons.append({"kind": "id", "value": item["id"]})
        for tag in item["tags"]:
            if tag in exclusions["tags"]:
                reasons.append({"kind": "tag", "value": tag})
        for term in exclusions["terms"]:
            for field in ("title", "description"):
                if term in item[field].casefold():
                    reasons.append({"kind": "term", "value": term, "field": field})
        if reasons:
            excluded.append({"id": item["id"], "reasons": reasons})
            continue
        contributions = [
            {"tag": tag, "weight": weights[tag]}
            for tag in item["tags"] if tag in weights and weights[tag] != 0
        ]
        score = math.fsum([item["quality"]] + [x["weight"] for x in contributions])
        explanation = {
            "quality": item["quality"],
            "matched_interests": contributions,
            "basis": "quality + sum of matched tag weights",
            "fallback": not contributions,
        }
        ranked.append({"id": item["id"], "title": item["title"],
                       "score": score, "explanation": explanation,
                       "source": item["source"]})
    ranked.sort(key=lambda x: (-x["score"], x["id"]))
    excluded.sort(key=lambda x: x["id"])
    return {"recommendations": ranked[:limit], "excluded": excluded,
            "eligible_count": len(ranked)}


def run(payload):
    object_keys(payload, ("schema_version", "fixture_label", "documents", "profile"), where="input")
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "schema_version must be 1")
    label = text(payload["fixture_label"], "fixture_label")
    catalog = automate_documents(payload["documents"])
    result = recommend(catalog, payload["profile"])
    return {"status": "ok", "schema_version": 1, "fixture_label": label,
            "catalog": catalog, **result,
            "counts": {"documents": len(payload["documents"]), "extracted": len(catalog),
                       "excluded": len(result["excluded"]),
                       "returned": len(result["recommendations"])}}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonfinite JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        payload = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run(payload)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
