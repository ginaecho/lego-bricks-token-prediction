"""Deterministic synthetic marketplace pipeline; Python standard library only."""
import difflib
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
    require(len(value) <= 10000, name + " is too long")
    return value


def exact_keys(value, keys, name):
    require(isinstance(value, dict) and set(value) == set(keys), name + " has invalid fields")


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            name + " must be a finite nonnegative number")


ALIASES = {"sneakers": "shoe", "shoes": "shoe", "trainers": "shoe",
           "earbuds": "headphone", "headphones": "headphone",
           "wireless": "bluetooth", "cheap": "budget", "inexpensive": "budget"}
STOP = {"a", "an", "the", "for", "i", "want", "need", "with", "and", "to", "of", "is"}


def tokens(value):
    return [ALIASES.get(t, t) for t in re.findall(r"\w+", value.lower()) if t not in STOP]


def validate(value, kind, context=None):
    """All external inputs and every handoff pass this shared validation boundary."""
    if kind == "input":
        exact_keys(value, ["schema_version", "synthetic", "query", "question",
                           "catalog", "policies", "feedback", "options"], kind)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "unsupported schema_version")
        require(value["synthetic"] is True, "reference input must be labeled synthetic")
        text(value["query"], "query")
        text(value["question"], "question")
        exact_keys(value["options"], ["limit", "max_price", "in_stock_only"], "options")
        opts = value["options"]
        require(type(opts["limit"]) is int and 1 <= opts["limit"] <= 20, "limit must be 1..20")
        if opts["max_price"] is not None:
            number(opts["max_price"], "max_price")
        require(type(opts["in_stock_only"]) is bool, "in_stock_only must be boolean")
        schemas = {"catalog": ["id", "name", "description", "tags", "price", "in_stock"],
                   "policies": ["id", "topic", "text"],
                   "feedback": ["id", "text", "product_id"]}
        for collection, keys in schemas.items():
            rows = value[collection]
            require(isinstance(rows, list) and len(rows) <= 1000, collection + " must be a bounded list")
            seen = set()
            for row in rows:
                exact_keys(row, keys, collection)
                text(row["id"], "id")
                require(row["id"] not in seen, "duplicate " + collection + " id")
                seen.add(row["id"])
                if collection == "catalog":
                    text(row["name"], "name")
                    text(row["description"], "description")
                    number(row["price"], "price")
                    require(type(row["in_stock"]) is bool, "in_stock must be boolean")
                    require(isinstance(row["tags"], list), "tags must be a list")
                    for tag in row["tags"]:
                        text(tag, "tag")
                else:
                    text(row["text"], "text")
                    if collection == "policies":
                        text(row["topic"], "topic")
                    elif row["product_id"] is not None:
                        text(row["product_id"], "product_id")
        ids = {p["id"] for p in value["catalog"]}
        require(all(f["product_id"] is None or f["product_id"] in ids for f in value["feedback"]),
                "feedback references unknown product")
    elif kind == "search":
        exact_keys(value, ["query", "normalized_terms", "matches"], kind)
        require(value["query"] == context["query"], "query changed in handoff")
        require(value["normalized_terms"] == sorted(set(tokens(context["query"]))),
                "invalid normalized terms")
        require(isinstance(value["matches"], list), "matches must be a list")
        require(len(value["matches"]) <= context["options"]["limit"], "too many matches")
        ids = set()
        for hit in value["matches"]:
            exact_keys(hit, ["product", "score", "matched_terms"], "match")
            require(hit["product"] in context["catalog"], "unknown product in handoff")
            pid = hit["product"]["id"]
            require(pid not in ids, "duplicate result")
            ids.add(pid)
            number(hit["score"], "score")
            require(0 < hit["score"] <= 1, "score outside range")
            require(isinstance(hit["matched_terms"], list) and bool(hit["matched_terms"]),
                    "missing match evidence")
            require(all(t in value["normalized_terms"] for t in hit["matched_terms"]),
                    "unknown match evidence")
    elif kind == "support":
        exact_keys(value, ["search", "question", "answer", "citations", "resolved", "feedback"], kind)
        validate(value["search"], "search", context)
        require(value["question"] == context["question"], "question changed")
        require(value["feedback"] == context["feedback"], "feedback changed")
        require(type(value["resolved"]) is bool, "resolved must be boolean")
        require(isinstance(value["citations"], list), "citations must be a list")
        text(value["answer"], "answer")
        sources = evidence(context, value["search"])
        for citation in value["citations"]:
            exact_keys(citation, ["source", "text"], "citation")
            require(citation in sources, "ungrounded citation")
        require(len({c["source"] for c in value["citations"]}) == len(value["citations"]),
                "duplicate citations")
        require(value["resolved"] == bool(value["citations"]), "resolution lacks evidence")
        expected = "\n".join(c["text"] for c in value["citations"]) if value["resolved"] else FALLBACK
        require(value["answer"] == expected, "answer is not grounded in cited evidence")
    elif kind == "output":
        exact_keys(value, ["schema_version", "synthetic", "status", "search", "support", "insights"], kind)
        require(value["schema_version"] == 1 and value["synthetic"] is True and value["status"] == "ok",
                "invalid output envelope")
        validate(value["support"], "support", context)
        require(value["search"] == value["support"]["search"], "search handoff changed")
        require(value["insights"] == insights(value["support"]), "insights handoff changed")
    else:
        raise ValidationError("unknown validation kind")
    return value


def search(data):
    terms = sorted(set(tokens(data["query"])))
    matches = []
    for product in data["catalog"]:
        opts = data["options"]
        if opts["in_stock_only"] and not product["in_stock"]:
            continue
        if opts["max_price"] is not None and product["price"] > opts["max_price"]:
            continue
        words = set(tokens(" ".join([product["name"], product["description"]] + product["tags"])))
        matched = [term for term in terms if term in words or (
            len(term) >= 4 and bool(difflib.get_close_matches(term, sorted(words), n=1, cutoff=0.82)))]
        if matched:
            matches.append({"product": product, "score": round(len(matched) / len(terms), 6),
                            "matched_terms": matched})
    matches.sort(key=lambda m: (-m["score"], m["product"]["price"], m["product"]["id"]))
    return validate({"query": data["query"], "normalized_terms": terms,
                     "matches": matches[:data["options"]["limit"]]}, "search", data)


FALLBACK = "I cannot answer from the available product and policy information. Please contact the support team."
POLICY_TOPICS = {
    "returns": {"return", "returns", "refund", "refunds"},
    "shipping": {"shipping", "delivery", "deliver", "arrive"},
    "warranty": {"warranty", "guarantee"},
    "payment": {"payment", "pay", "billing"},
}


def evidence(data, result):
    words = set(tokens(data["question"]))
    found = []
    for policy in data["policies"]:
        keywords = POLICY_TOPICS.get(policy["topic"].lower(), set(tokens(policy["topic"])))
        if words & keywords:
            found.append({"source": "policy:" + policy["id"], "text": policy["text"]})
    if words & {"price", "cost", "stock", "available", "availability", "describe", "description", "features"}:
        for hit in result["matches"]:
            p = hit["product"]
            found.append({"source": "product:" + p["id"],
                          "text": f'{p["name"]}: {p["description"]} Price: {p["price"]:.2f}. '
                                  f'In stock: {"yes" if p["in_stock"] else "no"}.'})
    return found


def support(data, result, selector=None):
    validate(result, "search", data)
    sources = evidence(data, result)
    if selector is not None:
        # Injection chooses evidence, never generates unverified factual prose.
        choice = selector(json.loads(json.dumps(sources)))
        require(isinstance(choice, list) and all(isinstance(s, str) for s in choice),
                "selector must return source IDs")
        require(len(choice) == len(set(choice)), "selector duplicated sources")
        index = {s["source"]: s for s in sources}
        require(all(s in index for s in choice), "selector returned unknown source")
        sources = [index[s] for s in choice]
    answer = "\n".join(s["text"] for s in sources) if sources else FALLBACK
    return validate({"search": result, "question": data["question"], "answer": answer,
                     "citations": sources, "resolved": bool(sources),
                     "feedback": data["feedback"]}, "support", data)


THEMES = {
    "delivery": ({"shipping", "delivery", "late", "arrive"}, "Review delivery expectations and delays."),
    "quality": ({"broken", "quality", "defect", "durable"}, "Review product quality reports."),
    "value": ({"price", "expensive", "budget", "cost"}, "Review pricing and value communication."),
    "usability": ({"confusing", "easy", "difficult", "instructions"}, "Improve product instructions."),
    "returns": ({"return", "returns", "refund"}, "Review the returns experience."),
}
NEGATIVE = {"bad", "broken", "late", "expensive", "confusing", "difficult", "disappointed"}
POSITIVE = {"good", "great", "love", "easy", "durable", "excellent"}


def insights(supported):
    selected = {h["product"]["id"] for h in supported["search"]["matches"]}
    relevant = [f for f in supported["feedback"] if f["product_id"] is None or f["product_id"] in selected]
    buckets = {}
    sentiment = {"positive": 0, "negative": 0, "mixed": 0, "neutral": 0}
    for row in relevant:
        words = set(tokens(row["text"]))
        pos, neg = bool(words & POSITIVE), bool(words & NEGATIVE)
        sentiment["mixed" if pos and neg else "positive" if pos else "negative" if neg else "neutral"] += 1
        themes = [name for name, (keys, _) in THEMES.items() if words & keys] or ["other"]
        for theme in themes:
            buckets.setdefault(theme, []).append(row["id"])
    actions = [{"theme": name, "count": len(ids), "feedback_ids": sorted(ids),
                "action": THEMES[name][1] if name in THEMES else "Read uncategorized feedback."}
               for name, ids in buckets.items()]
    actions.sort(key=lambda row: (-row["count"], row["theme"]))
    return {"matched_product_ids": sorted(selected), "feedback_count": len(relevant),
            "excluded_feedback_count": len(supported["feedback"]) - len(relevant),
            "sentiment": sentiment, "themes": actions,
            "support_gap": None if supported["resolved"] else
                {"question": supported["question"], "action": "Add verified support knowledge or follow up."}}


def run(data, selector=None):
    validate(data, "input")
    found = search(data)
    answered = support(data, found, selector)
    validate(answered, "support", data)
    return validate({"schema_version": 1, "synthetic": True, "status": "ok",
                     "search": found, "support": answered, "insights": insights(answered)},
                    "output", data)


def strict_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def main(argv=None):
    try:
        args = sys.argv[1:] if argv is None else argv
        require(len(args) == 1, "usage: python -B implementation.py input.json")
        with open(args[0], encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=strict_object)
        output = run(data)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
