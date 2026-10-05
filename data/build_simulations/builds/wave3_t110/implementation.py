"""Deterministic synthetic marketplace pipeline; Python standard library only."""

import difflib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path


class ValidationError(ValueError):
    pass


STAGES = ("search", "interests", "normal", "faq")
STOP = set("a an the and or for to of in with is are what which does do it this i want need me about".split())
ALIASES = {
    "rucksack": "backpack", "rucksacks": "backpack", "backpacks": "backpack",
    "bags": "bag", "hiking": "hike", "trekking": "hike", "trek": "hike",
    "lightweight": "light", "lightweightness": "light",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, expected, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(value) == set(expected), label + " has missing or unknown fields")


def text(value, label, maximum=10000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            label + " must be a nonempty bounded string")


def strings(value, label, maximum=100):
    require(isinstance(value, list) and len(value) <= maximum, label + " must be a bounded list")
    for item in value:
        text(item, label + " entry", 200)
    require(len(value) == len(set(value)), label + " must contain unique entries")


def number(value, label):
    require(type(value) in (int, float) and 0 <= value <= 1e12 and math.isfinite(value),
            label + " must be a finite number between 0 and 1e12")


def tokens(value):
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {ALIASES.get(word, word) for word in re.findall(r"[^\W_]+", normalized)
            if word not in STOP}


def overlap(query, candidate, fuzzy=False):
    if not query:
        return 0.0
    score = 0.0
    for term in sorted(query):
        if term in candidate:
            score += 1
        elif fuzzy and len(term) >= 4 and any(
                len(word) >= 4 and difflib.SequenceMatcher(None, term, word).ratio() >= .82
                for word in candidate):
            score += .7
    return round(score / len(query), 6)


def validate_input(data):
    keys(data, ("synthetic", "query", "products", "preferences", "research",
                "knowledge_base", "question", "limits"), "input")
    require(data["synthetic"] is True, "fixtures must be clearly labeled synthetic")
    text(data["query"], "query", 500)
    text(data["question"], "question", 500)
    products = data["products"]
    require(isinstance(products, list) and len(products) <= 100, "products must be a bounded list")
    ids = set()
    for p in products:
        keys(p, ("id", "name", "description", "category", "tags", "price"), "product")
        for key in ("id", "name", "description", "category"):
            text(p[key], "product." + key, 2000 if key == "description" else 200)
        require(p["id"] not in ids, "duplicate product id")
        ids.add(p["id"])
        strings(p["tags"], "tags")
        number(p["price"], "price")
    prefs = data["preferences"]
    keys(prefs, ("liked_tags", "liked_categories", "excluded_product_ids",
                 "excluded_tags", "budget_max"), "preferences")
    for key in ("liked_tags", "liked_categories", "excluded_product_ids", "excluded_tags"):
        strings(prefs[key], key)
    require(set(prefs["excluded_product_ids"]) <= ids, "unknown excluded product id")
    if prefs["budget_max"] is not None:
        number(prefs["budget_max"], "budget_max")
    source_ids = set()
    for collection in ("research", "knowledge_base"):
        require(isinstance(data[collection], list) and len(data[collection]) <= 100,
                collection + " must be a bounded list")
        for source in data[collection]:
            keys(source, ("id", "title", "text", "product_ids"), "source")
            for key in ("id", "title", "text"):
                text(source[key], "source." + key)
            require(source["id"] not in source_ids, "duplicate source id")
            source_ids.add(source["id"])
            strings(source["product_ids"], "source.product_ids")
            require(bool(source["product_ids"]) and set(source["product_ids"]) <= ids,
                    "source product references must exist")
    keys(data["limits"], STAGES, "limits")
    for name, value in data["limits"].items():
        require(type(value) is int and 1 <= value <= 20, "invalid limit: " + name)
    return data


def casefold_set(values):
    return {unicodedata.normalize("NFKC", value).casefold() for value in values}


def eligible(product, prefs):
    return (product["id"] not in prefs["excluded_product_ids"]
            and not casefold_set(product["tags"]) & casefold_set(prefs["excluded_tags"])
            and (prefs["budget_max"] is None or product["price"] <= prefs["budget_max"]))


def record(identifier, product_id, score, value, upstream=(), citations=()):
    return {"id": identifier, "product_id": product_id, "score": round(score, 6),
            "text": value, "upstream_ids": list(upstream), "citations": list(citations)}


def envelope(stage, records):
    return {"schema_version": 1, "stage": stage,
            "status": "ok" if records else ("abstained" if stage == "faq" else "empty"),
            "product_ids": sorted({r["product_id"] for r in records}), "records": records}


def validate_stage(output, data, previous=None):
    keys(output, ("schema_version", "stage", "status", "product_ids", "records"), "stage")
    stage = output["stage"]
    require(stage in STAGES and type(output["schema_version"]) is int
            and output["schema_version"] == 1, "invalid stage schema")
    index = STAGES.index(stage)
    require((index == 0 and previous is None) or
            (index > 0 and isinstance(previous, dict) and previous["stage"] == STAGES[index - 1]),
            "invalid stage order")
    records = output["records"]
    require(isinstance(records, list) and len(records) <= data["limits"][stage],
            "invalid stage records")
    require(output["status"] == ("ok" if records else ("abstained" if stage == "faq" else "empty")),
            "invalid stage status")
    products = {p["id"]: p for p in data["products"]}
    sources = {s["id"]: s for s in data["research"] + data["knowledge_base"]}
    parents = {r["id"]: r for r in previous["records"]} if previous else {}
    seen = set()
    for r in records:
        keys(r, ("id", "product_id", "score", "text", "upstream_ids", "citations"), "record")
        text(r["id"], "record.id")
        require(r["id"] not in seen, "duplicate record id")
        seen.add(r["id"])
        require(isinstance(r["product_id"], str) and r["product_id"] in products,
                "unknown output product")
        number(r["score"], "record.score")
        text(r["text"], "record.text")
        strings(r["upstream_ids"], "upstream_ids")
        if previous:
            require(bool(r["upstream_ids"]), "missing upstream evidence")
            require(all(u in parents and parents[u]["product_id"] == r["product_id"]
                        for u in r["upstream_ids"]), "invalid upstream reference")
            require(eligible(products[r["product_id"]], data["preferences"]),
                    "excluded product propagated")
        else:
            require(not r["upstream_ids"], "search cannot have upstream references")
        require(isinstance(r["citations"], list), "citations must be a list")
        if index >= 2:
            require(bool(r["citations"]), "grounded stage requires citations")
        else:
            require(not r["citations"], "discovery records have no passage citations")
        for cite in r["citations"]:
            keys(cite, ("source_id", "start", "end", "quote"), "citation")
            require(isinstance(cite["source_id"], str) and cite["source_id"] in sources,
                    "unknown citation source")
            source = sources[cite["source_id"]]
            allowed = data["research"] if stage == "normal" else data["knowledge_base"]
            require(source in allowed and r["product_id"] in source["product_ids"],
                    "citation source is outside stage/product scope")
            require(type(cite["start"]) is int and type(cite["end"]) is int
                    and 0 <= cite["start"] < cite["end"] <= len(source["text"]),
                    "invalid citation offsets")
            require(cite["quote"] == source["text"][cite["start"]:cite["end"]],
                    "citation must match source exactly")
        if index >= 2:
            require(r["text"] == " ".join(c["quote"] for c in r["citations"]),
                    "extractive text must equal cited passages")
    strings(output["product_ids"], "output.product_ids")
    require(output["product_ids"] == sorted({r["product_id"] for r in records}),
            "product summary mismatch")
    return output


def search(data):
    query = tokens(data["query"])
    results = []
    for p in data["products"]:
        terms = tokens(" ".join([p["name"], p["description"], p["category"]] + p["tags"]))
        score = overlap(query, terms, fuzzy=True)
        if score:
            results.append(record("search:" + p["id"], p["id"], score,
                                  p["name"] + ": " + p["description"]))
    results.sort(key=lambda r: (-r["score"], r["product_id"]))
    return envelope("search", results[:data["limits"]["search"]])


def interests(data, previous):
    products = {p["id"]: p for p in data["products"]}
    prefs = data["preferences"]
    results = []
    for found in previous["records"]:
        p = products[found["product_id"]]
        if not eligible(p, prefs):
            continue
        tags = sorted(tag for tag in p["tags"]
                      if casefold_set([tag]) & casefold_set(prefs["liked_tags"]))
        category = bool(casefold_set([p["category"]]) & casefold_set(prefs["liked_categories"]))
        reasons = ["Search relevance " + str(found["score"])]
        if tags:
            reasons.append("Preferred tags: " + ", ".join(tags))
        if category:
            reasons.append("Preferred category: " + p["category"])
        reasons.append("Catalog price: " + str(p["price"]))
        score = found["score"] + 2 * len(tags) + (1 if category else 0)
        results.append(record("interests:" + p["id"], p["id"], score,
                              "; ".join(reasons), [found["id"]]))
    results.sort(key=lambda r: (-r["score"], r["product_id"]))
    return envelope("interests", results[:data["limits"]["interests"]])


def passages(source):
    # Offsets index Unicode characters in the unmodified source, not UTF-8 bytes.
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", source["text"]):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        if start < end:
            yield {"source_id": source["id"], "start": start, "end": end,
                   "quote": source["text"][start:end]}


def normal(data, previous):
    results = []
    query = tokens(data["query"] + " " + data["question"])
    preference_terms = tokens(" ".join(data["preferences"]["liked_tags"]
                                     + data["preferences"]["liked_categories"]))
    for rank, parent in enumerate(previous["records"]):
        for source in data["research"]:
            if parent["product_id"] not in source["product_ids"]:
                continue
            for cite in passages(source):
                relevance = overlap(query, tokens(cite["quote"]))
                if not relevance:
                    continue
                score = relevance + .1 * overlap(preference_terms, tokens(cite["quote"])) + .01 / (rank + 1)
                results.append(record(
                    "normal:" + parent["product_id"] + ":" + source["id"] + ":" + str(cite["start"]),
                    parent["product_id"], score, cite["quote"], [parent["id"]], [cite]))
    results.sort(key=lambda r: (-r["score"], r["id"]))
    return envelope("normal", results[:data["limits"]["normal"]])


def faq(data, previous):
    query = tokens(data["question"])
    results = []
    for product_id in previous["product_ids"]:
        evidence = [r for r in previous["records"] if r["product_id"] == product_id]
        evidence_terms = tokens(" ".join(r["text"] for r in evidence))
        for source in data["knowledge_base"]:
            if product_id not in source["product_ids"]:
                continue
            for cite in passages(source):
                passage_terms = tokens(cite["quote"])
                relevance = overlap(query, passage_terms)
                # Conservative lexical coverage: never answer from research alone.
                if relevance < .6 or not query:
                    continue
                score = relevance + .1 * overlap(evidence_terms, passage_terms)
                results.append(record(
                    "faq:" + product_id + ":" + source["id"] + ":" + str(cite["start"]),
                    product_id, score, cite["quote"],
                    [r["id"] for r in evidence], [cite]))
    results.sort(key=lambda r: (-r["score"], r["id"]))
    return envelope("faq", results[:data["limits"]["faq"]])


def run_pipeline(data):
    validate_input(data)
    outputs = []
    previous = None
    for function in (search, interests, normal, faq):
        result = function(data) if previous is None else function(data, previous)
        previous = validate_stage(result, data, previous)
        outputs.append(previous)
    return {"status": "ok", "synthetic": True, "schema_version": 1, "stages": outputs,
            "answer": {"status": outputs[-1]["status"],
                       "text": " ".join(r["text"] for r in outputs[-1]["records"])
                       or "I cannot answer from the available linked knowledge-base evidence."}}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
