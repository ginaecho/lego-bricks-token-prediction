"""Offline synthetic research -> comparison -> feedback reference pipeline.

Usage: python -B implementation.py example_input.json
Offsets are zero-based, half-open Python Unicode character offsets. Evidence
establishes product eligibility, not independent verification of supplied specs.
All stages use validate() at their boundaries; no generated prose or providers.
"""

import json
import math
import re
import sys
from pathlib import Path


ATTRIBUTES = ("price_usd", "weight_g", "battery_hours")
STOPWORDS = frozenset("a an the and or of for in with to is are".split())
THEMES = {
    "battery": frozenset("battery charge charging runtime".split()),
    "price": frozenset("price cost expensive cheap affordable value".split()),
    "weight": frozenset("weight heavy light lightweight portable".split()),
    "usability": frozenset("easy difficult controls setup interface".split()),
    "reliability": frozenset("broken failed reliable failure durable".split()),
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, label):
    require(isinstance(value, dict), label + " must be an object")
    return value


def array(value, label):
    require(isinstance(value, list), label + " must be an array")
    return value


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()),
            label + " must be nonblank text")
    return value


def number(value, label, positive=False):
    require(type(value) in (int, float), label + " must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and (value > 0 if positive else value >= 0),
            label + " must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)


def dedup_key(value):
    return tuple(tokens(value))


def records(value, label):
    rows = array(value, label)
    seen = set()
    for row in rows:
        obj(row, label + " item")
        identifier = text(row.get("id"), label + ".id")
        require(identifier not in seen, "duplicate " + label + " id: " + identifier)
        seen.add(identifier)
    return rows


def normalize(attributes):
    result = dict.fromkeys(ATTRIBUTES)
    if "price" in attributes:
        result["price_usd"] = float(attributes["price"]["value"])
    if "weight" in attributes:
        weight = attributes["weight"]
        result["weight_g"] = float(weight["value"]) * (
            1000 if weight["unit"] == "kg" else 1)
    if "battery" in attributes:
        battery = attributes["battery"]
        result["battery_hours"] = float(battery["value"]) / (
            60 if battery["unit"] == "min" else 1)
    for key, value in result.items():
        if value is not None:
            number(value, key)
    return result


def validate(stage, payload, data=None, previous=None):
    """Shared schema and provenance validation for inputs and every handoff."""
    obj(payload, stage)
    if stage == "input":
        require(type(payload.get("schema_version")) is int
                and payload["schema_version"] == 1, "schema_version must be 1")
        require(payload.get("synthetic") is True, "synthetic must be true")
        query = text(payload.get("query"), "query")
        require(bool(set(tokens(query)) - STOPWORDS), "query needs searchable terms")
        limit = payload.get("max_findings", 20)
        require(type(limit) is int and 1 <= limit <= 1000,
                "max_findings must be an integer from 1 to 1000")
        sources = records(payload.get("sources"), "sources")
        source_ids = {row["id"] for row in sources}
        for source in sources:
            text(source.get("title"), "source.title")
            text(source.get("text"), "source.text")
        products = records(payload.get("products"), "products")
        product_ids = {row["id"] for row in products}
        for product in products:
            text(product.get("name"), "product.name")
            refs = array(product.get("source_ids"), "product.source_ids")
            for ref in refs:
                require(isinstance(ref, str) and ref in source_ids,
                        "unknown product source reference")
            require(len(refs) == len(set(refs)), "duplicate product source reference")
            attrs = obj(product.get("attributes"), "product.attributes")
            require(set(attrs) <= {"price", "weight", "battery"},
                    "unsupported product attribute")
            for name, measure in attrs.items():
                obj(measure, "attribute")
                number(measure.get("value"), name + ".value", positive=name != "price")
                if name == "price":
                    require(measure.get("currency") == "USD", "price currency must be USD")
                else:
                    allowed = {"g", "kg"} if name == "weight" else {"h", "min"}
                    unit = measure.get("unit")
                    require(isinstance(unit, str) and unit in allowed,
                            "unsupported " + name + " unit")
            normalize(attrs)
        preferences = array(payload.get("preferences"), "preferences")
        require(bool(preferences), "preferences must not be empty")
        seen = set()
        for pref in preferences:
            obj(pref, "preference")
            key = pref.get("attribute")
            require(isinstance(key, str) and key in ATTRIBUTES,
                    "unsupported preference attribute")
            require(key not in seen, "duplicate preference attribute")
            seen.add(key)
            require(pref.get("direction") in ("min", "max"),
                    "preference direction must be min or max")
            number(pref.get("weight"), "preference.weight", positive=True)
        for feedback in records(payload.get("feedback"), "feedback"):
            ref = feedback.get("product_id")
            require(isinstance(ref, str) and ref in product_ids,
                    "unknown feedback product reference")
            body = text(feedback.get("text"), "feedback.text")
            require(bool(tokens(body)), "feedback.text needs words")
        return payload

    require(data is not None, "stage validation requires input context")
    if stage == "research":
        findings = records(payload.get("findings"), "findings")
        sources = {source["id"]: source for source in data["sources"]}
        for finding in findings:
            citation = obj(finding.get("citation"), "citation")
            source_id = citation.get("source_id")
            require(isinstance(source_id, str) and source_id in sources,
                    "unknown citation source")
            source = sources[source_id]
            start, end = citation.get("start"), citation.get("end")
            require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(source["text"]), "invalid citation offsets")
            require(citation.get("quote") == source["text"][start:end],
                    "citation quote does not match exact source slice")
            require(citation.get("title") == source["title"], "citation title mismatch")
            require(finding.get("text") == citation["quote"], "finding is not extractive")
            expected_terms = sorted((set(tokens(data["query"])) - STOPWORDS)
                                    & set(tokens(finding["text"])))
            require(bool(expected_terms) and finding.get("matched_terms") == expected_terms,
                    "finding does not support query terms")
        candidates = array(payload.get("product_ids"), "research.product_ids")
        retrieved = {finding["citation"]["source_id"] for finding in findings}
        expected = sorted(product["id"] for product in data["products"]
                          if retrieved.intersection(product["source_ids"]))
        require(candidates == expected, "research product handoff mismatch")
    elif stage == "comparison":
        require(previous is not None, "comparison requires research handoff")
        validate("research", previous, data)
        rows = records(payload.get("rows"), "comparison.rows")
        require([row["id"] for row in rows] == previous["product_ids"],
                "comparison must contain exactly research-supported products")
        require(payload.get("columns") == list(ATTRIBUTES), "comparison columns mismatch")
        products = {product["id"]: product for product in data["products"]}
        for row in rows:
            product = products[row["id"]]
            require(row.get("name") == product["name"], "product name mismatch")
            require(row.get("attributes") == normalize(product["attributes"]),
                    "normalized attributes mismatch")
            expected = [finding["id"] for finding in previous["findings"]
                        if finding["citation"]["source_id"] in product["source_ids"]]
            require(row.get("finding_ids") == expected, "comparison evidence mismatch")
        ranking = array(payload.get("ranking"), "comparison.ranking")
        require(ranking == rank_rows(rows, data["preferences"]), "ranking mismatch")
    elif stage == "feedback":
        require(previous is not None, "feedback requires comparison handoff")
        selected = [entry["product_id"] for entry in previous["ranking"]]
        require(payload.get("selected_product_ids") == selected, "feedback scope mismatch")
        expected = aggregate_feedback(data["feedback"], previous["ranking"])
        require(payload == expected, "feedback provenance, deduplication or theme mismatch")
    else:
        raise ValidationError("unknown validation stage: " + stage)
    return payload


def research(data):
    validate("input", data)
    terms = set(tokens(data["query"])) - STOPWORDS
    hits = []
    for source in data["sources"]:
        for match in re.finditer(r"[^.!?\n]+[.!?]?", source["text"]):
            raw = match.group()
            quote = raw.strip()
            matched = sorted(terms & set(tokens(quote)))
            if not matched:
                continue
            start = match.start() + len(raw) - len(raw.lstrip())
            hits.append({
                "id": source["id"] + ":" + str(start),
                "text": quote,
                "matched_terms": matched,
                "citation": {"source_id": source["id"], "title": source["title"],
                             "start": start, "end": start + len(quote), "quote": quote},
            })
    hits.sort(key=lambda hit: (-len(hit["matched_terms"]),
                              hit["citation"]["source_id"], hit["citation"]["start"]))
    hits = hits[:data.get("max_findings", 20)]
    source_ids = {hit["citation"]["source_id"] for hit in hits}
    output = {
        "findings": hits,
        "product_ids": sorted(product["id"] for product in data["products"]
                              if source_ids.intersection(product["source_ids"])),
    }
    return validate("research", output, data)


def rank_rows(rows, preferences):
    if not rows:
        return []
    # Scale weights before summation to avoid overflow for valid finite weights.
    largest_weight = max(pref["weight"] for pref in preferences)
    scaled = [pref["weight"] / largest_weight for pref in preferences]
    total_weight = sum(scaled)
    utilities = {row["id"]: {} for row in rows}
    for pref in preferences:
        key = pref["attribute"]
        values = [row["attributes"][key] for row in rows
                  if row["attributes"][key] is not None]
        low, high = (min(values), max(values)) if values else (0, 0)
        for row in rows:
            value = row["attributes"][key]
            if value is None:
                utility = 0.0
            elif high == low:
                utility = 1.0
            else:
                utility = (value - low) / (high - low)
                if pref["direction"] == "min":
                    utility = 1.0 - utility
            utilities[row["id"]][key] = utility
    ranking = []
    for row in rows:
        contributions = utilities[row["id"]]
        score = sum(weight * contributions[pref["attribute"]]
                    for pref, weight in zip(preferences, scaled)) / total_weight
        ranking.append({"product_id": row["id"], "score": round(score, 8),
                        "utilities": contributions})
    ranking.sort(key=lambda entry: (-entry["score"], entry["product_id"]))
    for index, entry in enumerate(ranking, 1):
        entry["rank"] = index
    return ranking


def compare(data, research_output):
    validate("input", data)
    validate("research", research_output, data)
    products = {product["id"]: product for product in data["products"]}
    rows = []
    for identifier in research_output["product_ids"]:
        product = products[identifier]
        rows.append({
            "id": identifier, "name": product["name"],
            "attributes": normalize(product["attributes"]),
            "finding_ids": [finding["id"] for finding in research_output["findings"]
                            if finding["citation"]["source_id"] in product["source_ids"]],
        })
    output = {"columns": list(ATTRIBUTES), "rows": rows,
              "ranking": rank_rows(rows, data["preferences"])}
    return validate("comparison", output, data, research_output)


def aggregate_feedback(feedback, ranking):
    ranks = {entry["product_id"]: entry["rank"] for entry in ranking}
    groups = {}
    for item in sorted(feedback, key=lambda row: row["id"]):
        if item["product_id"] in ranks:
            key = (item["product_id"], dedup_key(item["text"]))
            groups.setdefault(key, []).append(item)
    themes = {}
    for (product_id, words), items in groups.items():
        representative = items[0]
        names = [name for name, vocabulary in THEMES.items() if set(words) & vocabulary]
        for name in names or ["other"]:
            themes.setdefault(name, []).append({
                "product_id": product_id, "comparison_rank": ranks[product_id],
                "feedback_id": representative["id"],
                "text": representative["text"],
                "duplicate_ids": [item["id"] for item in items[1:]],
            })
    summaries = []
    for name, excerpts in sorted(themes.items()):
        excerpts.sort(key=lambda item: (item["comparison_rank"], item["feedback_id"]))
        summaries.append({"theme": name, "count": len(excerpts),
                          "supporting_excerpts": excerpts})
    return {
        "selected_product_ids": [entry["product_id"] for entry in ranking],
        "unique_feedback_count": len(groups),
        "duplicate_count": sum(len(items) - 1 for items in groups.values()),
        "themes": summaries,
    }


def analyze_feedback(data, research_output, comparison_output):
    validate("input", data)
    validate("comparison", comparison_output, data, research_output)
    output = aggregate_feedback(data["feedback"], comparison_output["ranking"])
    return validate("feedback", output, data, comparison_output)


def run_pipeline(data):
    validate("input", data)
    findings = research(data)
    comparison = compare(data, findings)
    feedback = analyze_feedback(data, findings, comparison)
    return {"schema_version": 1, "status": "ok", "synthetic": True,
            "research": findings, "comparison": comparison, "feedback": feedback}


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("r", encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
        result = run_pipeline(data)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, OverflowError,
            RecursionError) as error:
        result = {"schema_version": 1, "status": "error", "error": str(error)}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
