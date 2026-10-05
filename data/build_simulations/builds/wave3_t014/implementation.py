"""Synthetic reference pipeline: labeled document fields -> intent-aware catalog search."""
import difflib
import json
import math
import re
import sys
import unicodedata


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class Schema:
    """One validation boundary for the input, extraction handoff and ranked output."""

    @staticmethod
    def validate_input(data):
        require(isinstance(data, dict), "input must be an object")
        require(data.get("schema_version") == "1", "schema_version must be '1'")
        require(isinstance(data.get("document"), str), "document must be a string")
        fields = data.get("fields")
        require(isinstance(fields, list) and fields, "fields must be a nonempty array")
        names, labels = set(), set()
        for field in fields:
            require(isinstance(field, dict), "field must be an object")
            name = field.get("name")
            require(text(name) and name not in names, "field names must be nonempty and unique")
            names.add(name)
            require(field.get("type") in ("string", "number", "boolean"), "unsupported field type")
            require(type(field.get("required")) is bool, "required must be boolean")
            aliases = field.get("aliases", [])
            require(isinstance(aliases, list) and all(text(a) for a in aliases), "invalid aliases")
            local = {normalized(label) for label in [name] + aliases}
            require(not local & labels, "field labels must not overlap")
            require(all(":" not in label and "\n" not in label and "\r" not in label
                        for label in [name] + aliases), "labels cannot contain colon or newline")
            labels.update(local)
        catalog = data.get("catalog")
        require(isinstance(catalog, list), "catalog must be an array")
        ids = set()
        for product in catalog:
            require(isinstance(product, dict), "product must be an object")
            require(all(text(product.get(k)) for k in ("id", "name", "category")),
                    "products require id, name and category")
            require(product["id"] not in ids, "duplicate product id")
            ids.add(product["id"])
            require(isinstance(product.get("description"), str), "description must be a string")
            require(isinstance(product.get("tags"), list) and all(text(t) for t in product["tags"]),
                    "tags must be strings")
            require(number(product.get("price")) and product["price"] >= 0, "invalid price")
            require(type(product.get("in_stock")) is bool, "in_stock must be boolean")
        config = data.get("search")
        require(isinstance(config, dict), "search must be an object")
        types = {f["name"]: f["type"] for f in fields}
        for key, expected in (("query_field", "string"), ("max_price_field", "number"),
                              ("category_field", "string"), ("in_stock_field", "boolean")):
            if key == "query_field" or key in config:
                require(isinstance(config.get(key), str) and types.get(config[key]) == expected,
                        key + " must reference a " + expected + " field")
        require(type(config.get("limit", 5)) is int and 1 <= config.get("limit", 5) <= 100,
                "limit must be an integer between 1 and 100")
        require(number(config.get("min_score", 0.3)) and 0 < config.get("min_score", 0.3) <= 1,
                "min_score must be in (0, 1]")
        return data

    @staticmethod
    def validate_extraction(result, data):
        require(isinstance(result, dict), "extraction must be an object")
        values, missing = result.get("fields"), result.get("missing_fields")
        require(isinstance(values, dict) and isinstance(missing, list), "invalid extraction shape")
        definitions = {f["name"]: f for f in data["fields"]}
        require(set(values) == set(definitions), "extraction fields mismatch")
        expected_missing = []
        for name, definition in definitions.items():
            item = values[name]
            require(isinstance(item, dict), "invalid extracted field")
            value, span = item.get("value"), item.get("span")
            if value is None:
                require(span is None, "missing field cannot have a source span")
                expected_missing.append(name)
                continue
            kind = definition["type"]
            require((kind == "string" and text(value)) or (kind == "number" and number(value))
                    or (kind == "boolean" and type(value) is bool), "extracted value type mismatch")
            require(isinstance(span, dict) and type(span.get("start")) is int
                    and type(span.get("end")) is int, "invalid source span")
            start, end = span["start"], span["end"]
            require(0 <= start < end <= len(data["document"]), "source span out of bounds")
            raw = data["document"][start:end]
            require(raw == item.get("source_text"), "source text mismatch")
            require(convert(raw, kind) == value, "source value mismatch")
        require(missing == expected_missing, "missing field list mismatch")
        required = [f["name"] for f in data["fields"] if f["required"] and f["name"] in missing]
        require(result.get("missing_required") == required, "missing required list mismatch")
        return result

    @staticmethod
    def validate_search(result, data):
        require(isinstance(result, dict) and result.get("status") in ("ok", "skipped"),
                "invalid search result")
        hits = result.get("results")
        require(isinstance(hits, list), "results must be an array")
        require(len(hits) <= data["search"].get("limit", 5), "too many results")
        catalog = {p["id"]: p for p in data["catalog"]}
        seen = set()
        for hit in hits:
            require(isinstance(hit, dict) and hit.get("product_id") in catalog, "unknown product")
            require(hit["product_id"] not in seen, "duplicate result")
            seen.add(hit["product_id"])
            require(number(hit.get("score")) and 0 < hit["score"] <= 1, "invalid score")
            require(hit.get("product") == catalog[hit["product_id"]], "product mismatch")
        require(result["status"] != "skipped" or not hits, "skipped search must have no results")
        return result


def convert(raw, kind):
    if kind == "string":
        return raw
    if kind == "boolean":
        value = normalized(raw)
        require(value in ("true", "false", "yes", "no"), "invalid boolean value: " + raw)
        return value in ("true", "yes")
    require(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw) is not None,
            "invalid number value: " + raw)
    value = float(raw)
    require(number(value), "number must be finite")
    return value


def extract(data):
    by_label = {}
    for field in data["fields"]:
        for label in [field["name"]] + field.get("aliases", []):
            by_label[normalized(label)] = field
    values = {f["name"]: {"value": None, "span": None, "source_text": None} for f in data["fields"]}
    seen = set()
    offset = 0
    for line in data["document"].splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if ":" in content:
            label, raw = content.split(":", 1)
            definition = by_label.get(normalized(label))
            if definition is not None:
                name = definition["name"]
                require(name not in seen, "duplicate document field: " + name)
                seen.add(name)
                stripped = raw.strip()
                if stripped:
                    start = offset + content.index(":") + 1 + len(raw) - len(raw.lstrip())
                    values[name] = {"value": convert(stripped, definition["type"]),
                                    "span": {"start": start, "end": start + len(stripped)},
                                    "source_text": stripped}
        offset += len(line)
    missing = [name for name, item in values.items() if item["value"] is None]
    result = {"fields": values, "missing_fields": missing,
              "missing_required": [f["name"] for f in data["fields"]
                                   if f["required"] and f["name"] in missing]}
    return Schema.validate_extraction(result, data)


SYNONYMS = {
    "sneaker": "shoe", "sneakers": "shoe", "shoes": "shoe", "trainers": "shoe",
    "notebook": "laptop", "notebooks": "laptop", "laptops": "laptop",
    "sofa": "couch", "sofas": "couch", "couches": "couch",
    "wireless": "cordless", "headphones": "headphone", "earphones": "headphone",
    "running": "run", "jogging": "run", "jog": "run",
}
STOP_WORDS = {"a", "an", "the", "for", "and", "i", "want", "need", "please", "some", "me", "find"}


def tokens(value):
    words = re.findall(r"[^\W_]+", normalized(value), flags=re.UNICODE)
    return sorted({SYNONYMS.get(word, word) for word in words if word not in STOP_WORDS})


def similarity(query, candidate):
    if query == candidate:
        return 1.0
    if min(len(query), len(candidate)) >= 4 and (query.startswith(candidate) or candidate.startswith(query)):
        return 0.85
    if min(len(query), len(candidate)) >= 4:
        ratio = difflib.SequenceMatcher(None, query, candidate).ratio()
        if ratio >= 0.8:
            return 0.75 * ratio
    return 0.0


def search(extraction, data):
    Schema.validate_extraction(extraction, data)
    config = data["search"]
    fields = extraction["fields"]
    query = fields[config["query_field"]]["value"]
    if extraction["missing_required"] or query is None or not tokens(query):
        reason = "missing_required_fields" if extraction["missing_required"] else "missing_search_terms"
        return Schema.validate_search({"status": "skipped", "reason": reason, "results": []}, data)
    intent = {"query": query, "terms": tokens(query)}
    for key in ("max_price", "category", "in_stock"):
        reference = config.get(key + "_field")
        intent[key] = fields[reference]["value"] if reference else None
    require(intent["max_price"] is None or intent["max_price"] >= 0, "maximum price cannot be negative")
    hits = []
    for product in data["catalog"]:
        if intent["max_price"] is not None and product["price"] > intent["max_price"]:
            continue
        if intent["category"] is not None and normalized(product["category"]) != normalized(intent["category"]):
            continue
        if intent["in_stock"] is not None and product["in_stock"] != intent["in_stock"]:
            continue
        weighted = [(word, weight) for value, weight in
                    [(product["name"], 1.0), (" ".join(product["tags"]), 0.95),
                     (product["category"], 0.8), (product["description"], 0.7)]
                    for word in set(tokens(value) + re.findall(r"[^\W_]+", normalized(value)))]
        matches = []
        for term in intent["terms"]:
            best = max(((similarity(term, word) * weight, word) for word, weight in weighted),
                       default=(0.0, ""), key=lambda pair: (pair[0], pair[1]))
            matches.append({"term": term, "matched_token": best[1] if best[0] else None,
                            "score": round(best[0], 6)})
        score = sum(match["score"] for match in matches) / len(matches)
        if score >= config.get("min_score", 0.3):
            hits.append({"product_id": product["id"], "product": product,
                         "score": round(score, 6), "matches": matches})
    hits.sort(key=lambda hit: (-hit["score"], hit["product"]["price"], hit["product_id"]))
    result = {"status": "ok", "intent": intent, "total_matches": len(hits),
              "results": hits[:config.get("limit", 5)]}
    return Schema.validate_search(result, data)


def run_pipeline(data):
    Schema.validate_input(data)
    extraction = extract(data)
    ranked = search(extraction, data)
    return {"schema_version": "1", "status": "needs_input" if ranked["status"] == "skipped" else "ok",
            "extraction": extraction, "search": ranked}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle)
        output = run_pipeline(data)
        code = 0
    except (ValidationError, OSError, ValueError, UnicodeError, RecursionError) as exc:
        output, code = {"status": "error", "error": str(exc)}, 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
