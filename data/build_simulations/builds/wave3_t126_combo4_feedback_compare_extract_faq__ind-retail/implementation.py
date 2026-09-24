"""Deterministic synthetic retail pipeline; Python standard library only."""

import copy
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


STAGES = ("feedback", "compare", "extract", "faq")
THEMES = {
    "price": {"price", "expensive", "cheap", "affordable", "cost"},
    "quality": {"quality", "durable", "broken", "leak", "leaks"},
    "shipping": {"shipping", "delivery", "late"},
    "capacity": {"capacity", "small", "large", "volume"},
}
RESERVED = {"sku": "string", "price": "number", "stock": "integer",
            "order_id": "string", "customer_id": "string"}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")


def number(value, label, integer=False):
    require(type(value) in ((int,) if integer else (int, float)),
            label + " must be " + ("an integer" if integer else "numeric"))
    require(math.isfinite(value) and value >= 0, label + " must be finite and nonnegative")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def normalize(attributes):
    result = {}
    for key, value in attributes.items():
        name = key.strip().casefold()
        require(name not in result, "duplicate normalized attribute")
        value = re.sub(r"\s+", " ", str(value).strip().casefold())
        if name == "capacity":
            match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ml|l)", value)
            require(match is not None, "capacity requires ml or l")
            value = float(match[1]) * (1000 if match[2] == "l" else 1)
            require(math.isfinite(value), "capacity must be finite")
            name = "capacity_ml"
        elif name == "color":
            value = {"grey": "gray"}.get(value, value)
        require(name not in result, "colliding normalized attribute")
        result[name] = value
    return result


def validate_input(data):
    require(isinstance(data, dict), "input must be an object")
    required = {"schema_version", "synthetic", "catalog", "customer", "basket",
                "feedback", "clickstream", "preferences", "documents",
                "extraction_schema", "knowledge_base", "question"}
    require(set(data) == required, "input fields must match shared schema")
    require(data["schema_version"] == "1.0" and data["synthetic"] is True,
            "schema_version 1.0 and synthetic=true required")
    for name in ("catalog", "feedback", "clickstream", "documents",
                 "extraction_schema", "knowledge_base"):
        require(isinstance(data[name], list), name + " must be an array")
    require(bool(data["catalog"]), "catalog must not be empty")
    catalog = {}
    currencies = set()
    for product in data["catalog"]:
        require(isinstance(product, dict) and set(product) ==
                {"sku", "name", "price", "stock", "currency", "attributes"},
                "invalid catalog product")
        for key in ("sku", "name", "currency"):
            text(product[key], key)
        require(product["sku"] not in catalog, "duplicate catalog SKU")
        number(product["price"], "price")
        number(product["stock"], "stock", True)
        require(product["currency"] in {"EUR", "USD", "GBP"}, "unsupported currency")
        currencies.add(product["currency"])
        require(isinstance(product["attributes"], dict), "attributes must be an object")
        for key, value in product["attributes"].items():
            text(key, "attribute name")
            text(value, "attribute value")
        normalize(product["attributes"])
        catalog[product["sku"]] = product
    require(len(currencies) == 1, "mixed currencies cannot be ranked")
    customer = data["customer"]
    require(isinstance(customer, dict) and set(customer) == {"id", "persona", "consent"},
            "invalid customer profile")
    text(customer["id"], "customer id")
    text(customer["persona"], "synthetic persona")
    consent = customer["consent"]
    require(isinstance(consent, dict) and set(consent) == {"gdpr", "ccpa"}
            and all(type(v) is bool for v in consent.values()), "explicit boolean consent required")
    basket = data["basket"]
    require(isinstance(basket, dict) and set(basket) == {"order_id", "customer_id", "items"},
            "invalid basket")
    text(basket["order_id"], "order id")
    require(basket["customer_id"] == customer["id"], "basket customer mismatch")
    require(isinstance(basket["items"], list), "basket items must be an array")
    seen = set()
    for item in basket["items"]:
        require(isinstance(item, dict) and set(item) == {"sku", "quantity"}, "invalid basket item")
        require(isinstance(item["sku"], str) and item["sku"] in catalog, "unknown basket SKU")
        require(item["sku"] not in seen, "duplicate basket SKU")
        seen.add(item["sku"])
        number(item["quantity"], "quantity", True)
        require(item["quantity"] > 0, "quantity must be positive")
        require(item["quantity"] <= catalog[item["sku"]]["stock"], "basket exceeds catalog stock")
    ids = set()
    for item in data["feedback"]:
        require(isinstance(item, dict) and set(item) == {"id", "sku", "text", "synthetic"},
                "invalid feedback record")
        text(item["id"], "feedback id")
        text(item["text"], "feedback text")
        require(item["synthetic"] is True, "feedback must be clearly labeled synthetic")
        require(isinstance(item["sku"], str) and item["sku"] in catalog, "unknown feedback SKU")
        require(item["id"] not in ids, "duplicate feedback id")
        ids.add(item["id"])
    for event in data["clickstream"]:
        require(isinstance(event, dict) and set(event) == {"customer_id", "sku", "event"},
                "invalid clickstream event")
        require(event["customer_id"] == customer["id"], "clickstream customer mismatch")
        require(isinstance(event["sku"], str) and event["sku"] in catalog, "unknown clickstream SKU")
        require(event["event"] in ("view", "add_to_basket"), "unsupported clickstream event")
    prefs = data["preferences"]
    require(isinstance(prefs, dict) and set(prefs) == {"max_price", "in_stock", "attributes"},
            "invalid preferences")
    if prefs["max_price"] is not None:
        number(prefs["max_price"], "max_price")
    require(type(prefs["in_stock"]) is bool, "in_stock must be boolean")
    require(isinstance(prefs["attributes"], dict), "preference attributes must be an object")
    for key, value in prefs["attributes"].items():
        text(key, "preference attribute")
        text(value, "preference value")
    normalize(prefs["attributes"])
    seen = set()
    for document in data["documents"]:
        require(isinstance(document, dict) and set(document) == {"sku", "text"}, "invalid document")
        require(isinstance(document["sku"], str) and document["sku"] in catalog, "unknown document SKU")
        require(document["sku"] not in seen, "duplicate SKU document")
        seen.add(document["sku"])
        text(document["text"], "document text")
    seen = set()
    require(bool(data["extraction_schema"]), "extraction schema must not be empty")
    for field in data["extraction_schema"]:
        require(isinstance(field, dict) and set(field) == {"name", "label", "type", "required"},
                "invalid extraction field")
        text(field["name"], "field name")
        text(field["label"], "field label")
        require("\n" not in field["label"] and ":" not in field["label"], "invalid field label")
        require(field["name"] not in seen, "duplicate extraction field")
        seen.add(field["name"])
        require(field["type"] in ("string", "number", "integer"), "invalid field type")
        require(type(field["required"]) is bool, "required must be boolean")
        require(field["name"] not in RESERVED or RESERVED[field["name"]] == field["type"],
                "reserved field has incorrect type")
    ids = set()
    for article in data["knowledge_base"]:
        require(isinstance(article, dict) and set(article) == {"id", "sku", "text", "claims"},
                "invalid knowledge-base article")
        text(article["id"], "article id")
        text(article["text"], "article text")
        require(article["id"] not in ids, "duplicate article id")
        ids.add(article["id"])
        require(article["sku"] is None or
                (isinstance(article["sku"], str) and article["sku"] in catalog), "unknown article SKU")
        require(isinstance(article["claims"], dict) and
                set(article["claims"]) <= {"price", "stock"}, "invalid article claims")
        for key, value in article["claims"].items():
            require(article["sku"] is not None, "catalog claims require SKU")
            number(value, key, key == "stock")
            require(value == catalog[article["sku"]][key], "article claim contradicts catalog")
        # Numeric commercial answers are rendered only from catalog fields, never prose.
        require(not re.search(r"\b(price|stock|cost|costs)\b|[$€£]", article["text"], re.I),
                "commercial claims belong in structured claims, not article text")
    text(data["question"], "question")
    return catalog


def personalized(data):
    return all(data["customer"]["consent"].values())


def convert(raw, kind):
    if kind == "string":
        return raw
    if kind == "integer":
        require(bool(re.fullmatch(r"\d+", raw)), "invalid extracted integer")
        value = int(raw)
    else:
        require(bool(re.fullmatch(r"\d+(?:\.\d+)?", raw)), "invalid extracted number")
        value = float(raw)
    number(value, "extracted value", kind == "integer")
    return value


def validate_state(state, expected=None):
    require(isinstance(state, dict) and set(state) ==
            {"schema_version", "status", "synthetic", "input", "stages"}, "invalid shared envelope")
    require(state["schema_version"] == "1.0" and state["status"] == "ok"
            and state["synthetic"] is True, "invalid envelope metadata")
    data = state["input"]
    catalog = validate_input(data)
    stages = state["stages"]
    require(isinstance(stages, dict), "stages must be an object")
    require(list(stages) == list(STAGES[:len(stages)]), "invalid stage order")
    if expected is not None:
        require(list(stages) == list(STAGES[:expected]), "stage handoff out of order")
    if "feedback" in stages:
        feedback = stages["feedback"]
        originals = {f["id"]: f for f in data["feedback"]}
        covered = []
        for group in feedback["groups"]:
            require(group["source_ids"] and group["representative_id"] == group["source_ids"][0],
                    "invalid feedback representative")
            representative = originals.get(group["representative_id"])
            require(representative is not None, "unknown feedback source")
            for source in group["source_ids"]:
                require(source in originals, "unknown feedback source")
                require(originals[source]["sku"] == representative["sku"] and
                        dedup_key(originals[source]) == dedup_key(representative),
                        "incorrect feedback deduplication")
            covered.extend(group["source_ids"])
            require(group["sku"] == representative["sku"], "feedback SKU mismatch")
        require(sorted(covered) == sorted(originals), "feedback coverage mismatch")
        representatives = {g["representative_id"] for g in feedback["groups"]}
        for theme, supports in feedback["themes"].items():
            require(theme in {*THEMES, "other"}, "invalid theme")
            for support in supports:
                require(support["source_id"] in representatives, "unsupported feedback excerpt")
                original = originals[support["source_id"]]["text"]
                require(support["span"] == [0, len(original)] and support["excerpt"] == original,
                        "fabricated feedback excerpt")
                require(theme in themes_for(original), "unsupported feedback theme")
    if "compare" in stages:
        comparison = stages["compare"]
        require(comparison["personalized"] == personalized(data), "consent gate mismatch")
        rows = comparison["rows"]
        require(len(rows) == len(catalog) and {r["sku"] for r in rows} == set(catalog),
                "comparison must cover catalog")
        for row in rows:
            product = catalog[row["sku"]]
            require(all(row[key] == product[key] for key in ("name", "price", "stock", "currency")),
                    "comparison contradicts catalog")
            require(row["attributes"] == normalize(product["attributes"]), "attribute mismatch")
            number(row["score"], "ranking score")
            if not comparison["personalized"]:
                require(row["score"] == 0, "personalization without consent")
        require(rows == sorted(rows, key=lambda r: (-r["score"], r["sku"])), "invalid ranking")
        require(comparison["selected_sku"] == rows[0]["sku"], "selection mismatch")
        require(comparison["feedback_theme_counts"] ==
                {k: len(v) for k, v in stages["feedback"]["themes"].items()}, "feedback handoff mismatch")
    if "extract" in stages:
        extraction = stages["extract"]
        selected = stages["compare"]["selected_sku"]
        require(extraction["selected_sku"] == selected, "extraction selection mismatch")
        document = next((d["text"] for d in data["documents"] if d["sku"] == selected), "")
        fields = {f["name"]: f for f in data["extraction_schema"]}
        require(set(extraction["fields"]) | set(extraction["missing_fields"]) == set(fields)
                and not set(extraction["fields"]) & set(extraction["missing_fields"]),
                "extraction coverage mismatch")
        require(extraction["missing_required"] ==
                [f["name"] for f in data["extraction_schema"]
                 if f["required"] and f["name"] in extraction["missing_fields"]],
                "missing required field mismatch")
        expected_values = dict(catalog[selected], order_id=data["basket"]["order_id"],
                               customer_id=data["customer"]["id"])
        for name, extracted in extraction["fields"].items():
            start, end = extracted["span"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(document),
                    "invalid extraction span")
            require(extracted["raw"] == document[start:end], "invalid extraction provenance")
            require(extracted["value"] == convert(extracted["raw"], fields[name]["type"]),
                    "extraction conversion mismatch")
            if name in RESERVED:
                require(extracted["value"] == expected_values[name], "extracted field contradicts context")
    if "faq" in stages:
        faq = stages["faq"]
        require(faq["selected_sku"] == stages["extract"]["selected_sku"], "FAQ handoff mismatch")
        if faq["abstained"]:
            require(faq["answer"] is None and not faq["citations"] and faq["reason"],
                    "invalid abstention")
        else:
            require("sku" in stages["extract"]["fields"], "FAQ requires extracted SKU")
            articles = {a["id"]: a for a in data["knowledge_base"]}
            excerpts = []
            require(bool(faq["citations"]), "answer requires sources")
            for citation in faq["citations"]:
                require(citation["source_id"] in articles, "unknown FAQ source")
                article = articles[citation["source_id"]]
                require(article["sku"] in (None, faq["selected_sku"]), "wrong SKU source")
                require(citation["span"] == [0, len(article["text"])] and
                        citation["excerpt"] == article["text"], "ungrounded FAQ excerpt")
                excerpts.append(article["text"])
            require(faq["answer"] == "\n".join(excerpts), "ungrounded FAQ answer")
        require(faq["catalog_facts"] == {k: catalog[faq["selected_sku"]][k]
                                       for k in ("sku", "price", "stock", "currency")},
                "FAQ facts contradict catalog")
    return state


def dedup_key(record):
    return record["sku"], re.sub(r"\W+", " ", record["text"].casefold()).strip()


def themes_for(value):
    return [theme for theme, words in THEMES.items() if tokens(value) & words] or ["other"]


def finish(state, name, result):
    output = copy.deepcopy(state)
    output["stages"][name] = result
    return validate_state(output)


def feedback_stage(state):
    validate_state(state, 0)
    groups = {}
    themes = {}
    for item in state["input"]["feedback"]:
        key = dedup_key(item)
        if key not in groups:
            groups[key] = {"sku": item["sku"], "representative_id": item["id"], "source_ids": []}
            for theme in themes_for(item["text"]):
                themes.setdefault(theme, []).append(
                    {"source_id": item["id"], "excerpt": item["text"], "span": [0, len(item["text"])]})
        groups[key]["source_ids"].append(item["id"])
    return finish(state, "feedback", {"groups": list(groups.values()), "themes": themes})


def compare_stage(state):
    validate_state(state, 1)
    data = state["input"]
    enabled = personalized(data)
    counts = {key: len(value) for key, value in state["stages"]["feedback"]["themes"].items()}
    prefs = data["preferences"]
    attributes = normalize(prefs["attributes"])
    rows = []
    for product in data["catalog"]:
        row = copy.deepcopy(product)
        row["attributes"] = normalize(product["attributes"])
        score = 0
        reasons = []
        if enabled:
            if prefs["max_price"] is not None and product["price"] <= prefs["max_price"]:
                score += 3 + min(counts.get("price", 0), 3)
                reasons.append("within_budget")
            if prefs["in_stock"] and product["stock"] > 0:
                score += 4
                reasons.append("available")
            for name, wanted in attributes.items():
                if row["attributes"].get(name) == wanted:
                    score += 2
                    reasons.append("attribute:" + name)
            activity = sum(1 if event["event"] == "view" else 2 for event in data["clickstream"]
                           if event["sku"] == product["sku"])
            score += min(activity, 3)
            if activity:
                reasons.append("consented_clickstream")
        row.update(score=score, reasons=reasons)
        rows.append(row)
    rows.sort(key=lambda row: (-row["score"], row["sku"]))
    return finish(state, "compare", {"personalized": enabled, "rows": rows,
                  "selected_sku": rows[0]["sku"], "feedback_theme_counts": counts,
                  "ranking_policy": "soft preferences; SKU tie-break" if enabled
                  else "neutral SKU ordering; personal signals ignored"})


def extract_stage(state):
    validate_state(state, 2)
    data = state["input"]
    selected = state["stages"]["compare"]["selected_sku"]
    document = next((d["text"] for d in data["documents"] if d["sku"] == selected), "")
    extracted = {}
    missing = []
    required = []
    for field in data["extraction_schema"]:
        pattern = r"^[ \t]*" + re.escape(field["label"]) + r":[ \t]*([^\r\n]*\S)[ \t]*\r?$"
        matches = list(re.finditer(pattern, document, re.M | re.I))
        require(len(matches) <= 1, "ambiguous duplicate document field: " + field["name"])
        if not matches:
            missing.append(field["name"])
            if field["required"]:
                required.append(field["name"])
            continue
        match = matches[0]
        raw = match[1].rstrip()
        start = match.start(1)
        extracted[field["name"]] = {"value": convert(raw, field["type"]),
                                   "raw": raw, "span": [start, start + len(raw)]}
    return finish(state, "extract", {"selected_sku": selected, "fields": extracted,
                  "missing_fields": missing, "missing_required": required,
                  "document_found": bool(document)})


def faq_stage(state):
    validate_state(state, 3)
    data = state["input"]
    extraction = state["stages"]["extract"]
    selected = extraction["selected_sku"]
    catalog = {p["sku"]: p for p in data["catalog"]}
    stopwords = {"a", "an", "the", "is", "are", "it", "i", "my", "can", "how", "what",
                 "does", "do", "for", "of", "to", "this", "and", "with"}
    query = tokens(data["question"]) - stopwords
    candidates = []
    if "sku" in extraction["fields"]:
        for article in data["knowledge_base"]:
            if article["sku"] in (None, selected):
                overlap = query & (tokens(article["text"]) - stopwords)
                if query and overlap == query:
                    candidates.append((len(overlap), article["id"], article))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    citations = []
    if candidates:
        article = candidates[0][2]
        citations = [{"source_id": article["id"], "excerpt": article["text"],
                      "span": [0, len(article["text"])]}]
    return finish(state, "faq", {
        "selected_sku": selected,
        "abstained": not bool(citations),
        "reason": None if citations else ("missing_extracted_sku" if "sku" not in extraction["fields"]
                                         else "no_relevant_knowledge"),
        "answer": citations[0]["excerpt"] if citations else None,
        "citations": citations,
        "catalog_facts": {key: catalog[selected][key] for key in ("sku", "price", "stock", "currency")},
    })


def run_pipeline(data):
    state = {"schema_version": "1.0", "status": "ok", "synthetic": True,
             "input": copy.deepcopy(data), "stages": {}}
    validate_state(state, 0)
    for stage in (feedback_stage, compare_stage, extract_stage, faq_stage):
        state = stage(state)
    return state


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py input.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=reject_duplicates)
        result = run_pipeline(data)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
