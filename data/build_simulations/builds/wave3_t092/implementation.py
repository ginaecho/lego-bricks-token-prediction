"""Deterministic synthetic-reference research -> search -> insights pipeline."""

import copy
import difflib
import json
import math
import re
import sys
from collections import Counter


class ValidationError(ValueError):
    pass


STOP = set("a an and are as at be by can do for from how i in is it of on or our the to we what which with".split())
ALIASES = {
    "headphones": "headphone", "headsets": "headphone", "headset": "headphone",
    "wireless": "bluetooth", "cordless": "bluetooth",
    "inexpensive": "budget", "affordable": "budget", "cheap": "budget",
    "comfortable": "comfort", "comfy": "comfort",
    "durable": "durability", "sturdy": "durability",
    "batteries": "battery", "shipping": "delivery",
}
THEMES = {
    "battery": {"battery", "charge", "charging"},
    "comfort": {"comfort", "pain", "fit"},
    "delivery": {"delivery", "late", "arrived"},
    "durability": {"durability", "broken", "broke", "fragile"},
    "usability": {"easy", "difficult", "setup", "confusing"},
    "value": {"budget", "price", "expensive", "value"},
}
POSITIVE = {"good", "great", "excellent", "love", "easy", "reliable"}
NEGATIVE = {"bad", "poor", "broken", "broke", "late", "pain", "terrible", "difficult", "confusing"}


def tokens(text):
    return [ALIASES.get(word, word) for word in re.findall(r"[^\W_]+", text.casefold())
            if word not in STOP]


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected.split()), path + " has missing or unknown fields")


def text(value, path, empty=False):
    require(isinstance(value, str) and (empty or bool(value.strip())),
            path + " must be a string" + ("" if empty else " with non-whitespace content"))


def number(value, path, lower, upper=float("inf")):
    require(type(value) in (int, float) and
            (type(value) is int or math.isfinite(value)) and lower <= value <= upper,
            path + " must be a finite number in range")


def records(value, path, expected):
    require(isinstance(value, list), path + " must be an array")
    ids = set()
    for row in value:
        fields(row, expected, path + " item")
        text(row["id"], path + ".id")
        require(row["id"] not in ids, path + " IDs must be unique")
        ids.add(row["id"])
    return ids


def validate_request(request):
    fields(request, "schema_version fixture_label question sources query products feedback", "input")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "schema_version must be 1")
    text(request["fixture_label"], "fixture_label")
    text(request["question"], "question")
    records(request["sources"], "sources", "id text credibility")
    for source in request["sources"]:
        text(source["text"], "source.text")
        number(source["credibility"], "source.credibility", 0, 1)
    query = request["query"]
    fields(query, "text max_price limit", "query")
    text(query["text"], "query.text", empty=True)
    if query["max_price"] is not None:
        number(query["max_price"], "query.max_price", 0)
    require(type(query["limit"]) is int and 1 <= query["limit"] <= 100,
            "query.limit must be an integer from 1 to 100")
    product_ids = records(request["products"], "products",
                          "id name description tags price available")
    for product in request["products"]:
        text(product["name"], "product.name")
        text(product["description"], "product.description", empty=True)
        require(isinstance(product["tags"], list), "product.tags must be an array")
        for tag in product["tags"]:
            text(tag, "product tag")
        number(product["price"], "product.price", 0)
        require(type(product["available"]) is bool, "product.available must be boolean")
    records(request["feedback"], "feedback", "id product_id text rating")
    for feedback in request["feedback"]:
        text(feedback["product_id"], "feedback.product_id")
        require(feedback["product_id"] in product_ids, "feedback references unknown product")
        text(feedback["text"], "feedback.text")
        if feedback["rating"] is not None:
            number(feedback["rating"], "feedback.rating", 1, 5)


def validate(envelope, expected_stage):
    """One validation boundary shared by every stage and the CLI pipeline."""
    fields(envelope, "schema_version status stage data", "envelope")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "envelope schema_version must be 1")
    require(envelope["status"] == "ok", "stage status must be ok")
    stages = ["input", "research", "search", "insights"]
    require(envelope["stage"] == expected_stage and expected_stage in stages,
            "unexpected pipeline stage")
    index = stages.index(expected_stage)
    fields(envelope["data"], " ".join(["request"] + stages[1:index + 1]), "data")
    data = envelope["data"]
    validate_request(data["request"])
    request = data["request"]
    if index >= 1:
        research = data["research"]
        fields(research, "question evidence needs limitations", "research")
        require(research["question"] == request["question"], "research question changed")
        source_ids = {row["id"] for row in request["sources"]}
        evidence_ids = records(research["evidence"], "evidence", "id excerpt relevance credibility")
        require(evidence_ids <= source_ids, "evidence references unknown source")
        originals = {row["id"]: row for row in request["sources"]}
        for row in research["evidence"]:
            require(row["excerpt"] == originals[row["id"]]["text"], "evidence text changed")
            number(row["relevance"], "evidence.relevance", 0, 1)
            require(row["credibility"] == originals[row["id"]]["credibility"],
                    "evidence credibility changed")
        for key in ("needs", "limitations"):
            require(isinstance(research[key], list), "research." + key + " must be an array")
            for value in research[key]:
                text(value, "research." + key)
    if index >= 2:
        search = data["search"]
        fields(search, "query interpreted_terms research_source_ids results", "search")
        require(search["query"] == request["query"]["text"], "search query changed")
        require(search["research_source_ids"] == [row["id"] for row in research["evidence"]],
                "search evidence provenance changed")
        require(isinstance(search["interpreted_terms"], list), "interpreted_terms must be an array")
        for term in search["interpreted_terms"]:
            text(term, "interpreted term")
        result_ids = records(search["results"], "results",
                             "id name price score matched_query matched_research")
        products = {row["id"]: row for row in request["products"]}
        require(result_ids <= products.keys(), "result references unknown product")
        require(len(search["results"]) <= request["query"]["limit"], "too many results")
        for row in search["results"]:
            product = products[row["id"]]
            require(row["name"] == product["name"] and row["price"] == product["price"],
                    "result product metadata changed")
            require(product["available"], "unavailable search result")
            maximum = request["query"]["max_price"]
            require(maximum is None or row["price"] <= maximum, "result exceeds budget")
            number(row["score"], "result.score", 0)
            for key, allowed in (("matched_query", search["interpreted_terms"]),
                                 ("matched_research", research["needs"])):
                require(isinstance(row[key], list), key + " must be an array")
                require(all(isinstance(term, str) and term in allowed for term in row[key]),
                        "result terms lack provenance")
    if index >= 3:
        insights = data["insights"]
        fields(insights, "selected_product_ids feedback_ids themes limitations", "insights")
        require(insights["selected_product_ids"] == [row["id"] for row in search["results"]],
                "insights selection changed")
        eligible = [row for row in request["feedback"] if row["product_id"] in result_ids]
        require(insights["feedback_ids"] == [row["id"] for row in eligible],
                "insights feedback scope changed")
        theme_ids = records(insights["themes"], "themes",
                            "id count negative positive neutral feedback_ids product_ids action")
        require(theme_ids <= set(THEMES) | {"other"}, "unknown feedback theme")
        for row in insights["themes"]:
            for key in ("count", "negative", "positive", "neutral"):
                require(type(row[key]) is int and row[key] >= 0, "theme counts must be integers")
            require(row["count"] == row["negative"] + row["positive"] + row["neutral"],
                    "theme sentiment counts disagree")
            require(isinstance(row["feedback_ids"], list) and
                    all(isinstance(item, str) for item in row["feedback_ids"]),
                    "theme feedback IDs must be strings")
            require(len(set(row["feedback_ids"])) == row["count"] == len(row["feedback_ids"])
                    and set(row["feedback_ids"]) <= {f["id"] for f in eligible},
                    "theme feedback provenance invalid")
            expected_products = sorted({f["product_id"] for f in eligible
                                        if f["id"] in row["feedback_ids"]})
            require(row["product_ids"] == expected_products, "theme product provenance invalid")
            text(row["action"], "theme.action")
        require(isinstance(insights["limitations"], list), "insights limitations must be an array")
        for item in insights["limitations"]:
            text(item, "insights limitation")
    return envelope


def advance(previous, stage, result):
    envelope = copy.deepcopy(previous)
    envelope["stage"] = stage
    envelope["data"][stage] = result
    return validate(envelope, stage)


def research_stage(previous):
    validate(previous, "input")
    request = previous["data"]["request"]
    question_terms = set(tokens(request["question"]))
    evidence = []
    counts = Counter()
    for source in request["sources"]:
        terms = set(tokens(source["text"]))
        overlap = terms & question_terms
        if overlap and source["credibility"] > 0:
            evidence.append({"id": source["id"], "excerpt": source["text"],
                             "relevance": round(len(overlap) / len(question_terms), 6),
                             "credibility": source["credibility"]})
            for term in terms:
                counts[term] += source["credibility"]
    evidence.sort(key=lambda row: (-row["relevance"] * row["credibility"], row["id"]))
    needs = sorted(counts, key=lambda term: (-counts[term], term))[:12]
    limitations = ["Source credibility is supplied, not independently verified.",
                   "Evidence is lexical relevance, not a truth or causality assessment."]
    if not evidence:
        limitations.append("No relevant evidence; search will use only the customer query.")
    return advance(previous, "research", {
        "question": request["question"], "evidence": evidence,
        "needs": needs, "limitations": limitations,
    })


def search_stage(previous):
    validate(previous, "research")
    request = previous["data"]["request"]
    research = previous["data"]["research"]
    indexed = {p["id"]: set(tokens(" ".join([p["name"], p["description"]] + p["tags"])))
               for p in request["products"]}
    vocabulary = sorted(set().union(*indexed.values())) if indexed else []
    interpreted = set()
    for term in tokens(request["query"]["text"]):
        matches = difflib.get_close_matches(term, vocabulary, n=1, cutoff=0.84)
        interpreted.add(term if term in vocabulary or not matches else matches[0])
    results = []
    for product in request["products"]:
        maximum = request["query"]["max_price"]
        if not product["available"] or (maximum is not None and product["price"] > maximum):
            continue
        terms = indexed[product["id"]]
        direct = sorted(terms & interpreted)
        inferred = sorted(terms & set(research["needs"]))
        score = 3 * len(direct) + len(inferred)
        if score:
            results.append({"id": product["id"], "name": product["name"],
                            "price": product["price"], "score": score,
                            "matched_query": direct, "matched_research": inferred})
    results.sort(key=lambda row: (-row["score"], row["price"], row["id"]))
    return advance(previous, "search", {
        "query": request["query"]["text"], "interpreted_terms": sorted(interpreted),
        "research_source_ids": [row["id"] for row in research["evidence"]],
        "results": results[:request["query"]["limit"]],
    })


def sentiment(feedback):
    if feedback["rating"] is not None:
        return "negative" if feedback["rating"] < 3 else "positive" if feedback["rating"] > 3 else "neutral"
    terms = set(tokens(feedback["text"]))
    score = len(terms & POSITIVE) - len(terms & NEGATIVE)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def insights_stage(previous):
    validate(previous, "search")
    request = previous["data"]["request"]
    selected = [row["id"] for row in previous["data"]["search"]["results"]]
    feedback = [row for row in request["feedback"] if row["product_id"] in selected]
    groups = {}
    for item in feedback:
        terms = set(tokens(item["text"]))
        themes = [name for name, keywords in THEMES.items() if terms & keywords] or ["other"]
        for name in themes:
            group = groups.setdefault(name, {"id": name, "count": 0, "negative": 0,
                                             "positive": 0, "neutral": 0,
                                             "feedback_ids": [], "product_ids": [], "action": ""})
            group["count"] += 1
            group[sentiment(item)] += 1
            group["feedback_ids"].append(item["id"])
            group["product_ids"].append(item["product_id"])
    themes = list(groups.values())
    for group in themes:
        group["product_ids"] = sorted(set(group["product_ids"]))
        group["action"] = (f"Investigate {group['id']} complaints for selected products."
                           if group["negative"] else
                           f"Review {group['id']} feedback before changing selected products.")
    themes.sort(key=lambda row: (-row["negative"], -row["count"], row["id"]))
    limitations = ["Only feedback for returned products is included; not a population estimate.",
                   "Themes can overlap; ratings take precedence over lexical sentiment.",
                   "Keyword sentiment does not interpret negation, sarcasm, or language context."]
    if not feedback:
        limitations.append("No feedback is available for the selected products.")
    return advance(previous, "insights", {
        "selected_product_ids": selected, "feedback_ids": [row["id"] for row in feedback],
        "themes": themes, "limitations": limitations,
    })


def run(request):
    current = validate({"schema_version": 1, "status": "ok", "stage": "input",
                        "data": {"request": copy.deepcopy(request)}}, "input")
    return insights_stage(search_stage(research_stage(current)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            request = json.load(handle)
        result = run(request)
    except (ValidationError, OSError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
