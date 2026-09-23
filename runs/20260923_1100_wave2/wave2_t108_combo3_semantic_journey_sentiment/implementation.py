"""Synthetic discovery pipeline. Run: python -B implementation.py example_input.json.

Python API: run(payload, embedding=None). The optional callable accepts a string
and returns a finite, nonzero numeric vector of consistent dimension. It must be
deterministic for reproducibility. No provider is configured or contacted.
"""

import copy
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


class ValidationError(ValueError):
    pass


SYNONYMS = {
    "beginner": "starter", "beginners": "starter", "intro": "starter",
    "photography": "camera", "photos": "camera", "photographic": "camera",
    "lightweight": "portable", "compact": "portable",
}
LEXICON = {"love": 2, "great": 2, "good": 1, "helpful": 1,
           "bad": -1, "broken": -2, "hate": -2, "confusing": -1, "slow": -1}
SEVERITIES = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, fields, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(value) == set(fields), where + " has missing or unknown fields")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()), where + " must be nonblank text")


def strings(value, where):
    require(isinstance(value, list), where + " must be an array")
    for item in value:
        text(item, where)
    require(len(value) == len(set(value)), where + " contains duplicates")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_input(data):
    shape(data, ("schema_version", "fixture_label", "query", "limit", "products",
                 "actions", "profile", "feedback"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "unsupported schema_version")
    text(data["fixture_label"], "fixture_label")
    text(data["query"], "query")
    require(type(data["limit"]) is int and 1 <= data["limit"] <= 100, "limit must be 1..100")
    ids = {}
    for kind in ("products", "actions", "feedback"):
        require(isinstance(data[kind], list), kind + " must be an array")
        ids[kind] = set()
        for row in data[kind]:
            fields = {"products": ("id", "name", "description", "tags"),
                      "actions": ("id", "product_id", "title", "requires", "interest"),
                      "feedback": ("id", "product_id", "action_id", "text", "severity")}[kind]
            shape(row, fields, kind)
            text(row["id"], kind + ".id")
            require(row["id"] not in ids[kind], kind + " has duplicate id")
            ids[kind].add(row["id"])
            if kind == "products":
                text(row["name"], "name")
                text(row["description"], "description")
                strings(row["tags"], "tags")
            elif kind == "actions":
                text(row["title"], "title")
                text(row["interest"], "interest")
                text(row["product_id"], "product_id")
                strings(row["requires"], "requires")
            else:
                text(row["text"], "feedback.text")
                text(row["product_id"], "product_id")
                if row["action_id"] is not None:
                    text(row["action_id"], "action_id")
                require(isinstance(row["severity"], str) and row["severity"] in SEVERITIES,
                        "invalid severity")
    actions = {a["id"]: a for a in data["actions"]}
    for a in actions.values():
        require(a["product_id"] in ids["products"], "unknown action product")
        require(set(a["requires"]) <= ids["actions"], "unknown prerequisite")
    # Iterative topological validation also handles long prerequisite chains.
    resolved = set()
    while len(resolved) < len(actions):
        ready = {k for k, a in actions.items() if k not in resolved
                 and set(a["requires"]) <= resolved}
        require(bool(ready), "cyclic prerequisites")
        resolved.update(ready)
    shape(data["profile"], ("completed_actions", "interests"), "profile")
    strings(data["profile"]["completed_actions"], "completed_actions")
    strings(data["profile"]["interests"], "interests")
    completed = set(data["profile"]["completed_actions"])
    require(completed <= ids["actions"], "unknown completed action")
    for k in completed:
        require(set(actions[k]["requires"]) <= completed, "completed history misses prerequisites")
    for f in data["feedback"]:
        require(f["product_id"] in ids["products"], "unknown feedback product")
        if f["action_id"] is not None:
            require(f["action_id"] in actions, "unknown feedback action")
            require(actions[f["action_id"]]["product_id"] == f["product_id"],
                    "feedback action/product mismatch")


def tokens(value):
    return [SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", value.lower())]


class SearchIndex:
    """Normalized TF-IDF inverted index with explicit small synonym vocabulary."""

    def __init__(self, products):
        self.documents = {}
        self.postings = {}
        for p in products:
            counts = Counter(tokens(" ".join([p["name"], p["description"], *p["tags"]])))
            self.documents[p["id"]] = counts
            for term in counts:
                self.postings.setdefault(term, set()).add(p["id"])
        self.idf = {t: 1 + math.log((1 + len(products)) / (1 + len(postings)))
                    for t, postings in self.postings.items()}

    def scores(self, query):
        q = Counter(tokens(query))
        qv = {t: c * self.idf.get(t, 1 + math.log(1 + len(self.documents))) for t, c in q.items()}
        candidates = set().union(*(self.postings.get(t, set()) for t in q))
        result = {}
        for key in candidates:
            dv = {t: c * self.idf[t] for t, c in self.documents[key].items()}
            result[key] = cosine(list(qv.values()), [dv.get(t, 0) for t in qv],
                                 right_norm=math.hypot(*dv.values()))
        return result


def cosine(left, right, right_norm=None):
    ln = math.hypot(*left)
    rn = math.hypot(*right) if right_norm is None else right_norm
    if not ln or not rn:
        return 0.0
    return max(-1.0, min(1.0, sum((a / ln) * (b / rn) for a, b in zip(left, right))))


def vector(embed, value, dimension=None):
    try:
        v = embed(value)
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(v, (list, tuple)) and 1 <= len(v) <= 4096, "invalid embedding vector")
    require(all(number(x) for x in v), "embedding must contain finite numbers")
    require(dimension is None or len(v) == dimension, "embedding dimension mismatch")
    norm = math.hypot(*v)
    require(math.isfinite(norm) and norm > 0, "embedding must have finite nonzero norm")
    return v


def semantic(data, embed=None):
    scores = SearchIndex(data["products"]).scores(data["query"])
    qv = vector(embed, data["query"]) if embed is not None else None
    hits = []
    for p in data["products"]:
        lexical = scores.get(p["id"], 0.0)
        similarity = None
        score = lexical
        if qv is not None:
            pv = vector(embed, p["name"] + " " + p["description"], len(qv))
            similarity = max(0.0, cosine(qv, pv))
            score = 0.5 * lexical + 0.5 * similarity
        if round(score, 8) > 0:
            hits.append({"product_id": p["id"], "score": round(score, 8),
                         "lexical_score": round(lexical, 8),
                         "embedding_score": None if similarity is None else round(similarity, 8)})
    hits.sort(key=lambda h: (-h["score"], h["product_id"]))
    return {"query": data["query"], "hits": hits[:data["limit"]]}


def journey(data, search):
    actions = data["actions"]
    completed = set(data["profile"]["completed_actions"])
    interests = set(data["profile"]["interests"])
    ranks = {h["product_id"]: i for i, h in enumerate(search["hits"])}
    candidates = [a for a in actions if a["product_id"] in ranks and a["id"] not in completed]

    def order(a):
        return (ranks[a["product_id"]], a["interest"] not in interests, a["id"])

    ready = sorted((a for a in candidates if set(a["requires"]) <= completed), key=order)
    pair = []
    # Look ahead instead of choosing a dead-end first action greedily.
    for first in ready:
        second = sorted((a for a in candidates if a["id"] != first["id"]
                         and set(a["requires"]) <= completed | {first["id"]}), key=order)
        if second:
            pair = [first, second[0]]
            break
    steps = [{"action_id": a["id"], "product_id": a["product_id"],
              "requires": list(a["requires"]), "interest_match": a["interest"] in interests}
             for a in pair]
    return {"source_product_ids": list(ranks), "next_actions": [a["id"] for a in ready],
            "status": "ready" if pair else "unavailable", "steps": steps,
            "reason": None if pair else "No feasible two-step journey within search results"}


def score_sentiment(value):
    words = re.findall(r"[a-z0-9]+", value.lower())
    evidence = []
    for i, word in enumerate(words):
        if word in LEXICON:
            negated = i > 0 and words[i - 1] in {"not", "never", "no"}
            evidence.append({"token": word, "position": i, "negated": negated,
                             "weight": LEXICON[word] * (-1 if negated else 1)})
    total = sum(e["weight"] for e in evidence)
    score = round(total / sum(abs(e["weight"]) for e in evidence), 8) if evidence else 0.0
    return {"score": score, "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "evidence": evidence}


def sentiment(data, recommendation):
    steps = recommendation["steps"]
    products = {s["product_id"] for s in steps}
    selected = {s["action_id"] for s in steps}
    issues = []
    for f in data["feedback"]:
        if f["product_id"] not in products or (f["action_id"] is not None and f["action_id"] not in selected):
            continue
        analysis = score_sentiment(f["text"])
        priority = round(100 * SEVERITIES[f["severity"]] + 10 * max(0, -analysis["score"]), 8)
        issues.append({"feedback_id": f["id"], "product_id": f["product_id"],
                       "action_id": f["action_id"], "severity": f["severity"],
                       "priority": priority, "sentiment": analysis})
    issues.sort(key=lambda i: (-i["priority"], i["feedback_id"]))
    return {"source_action_ids": [s["action_id"] for s in steps], "issues": issues,
            "excluded_feedback_count": len(data["feedback"]) - len(issues)}


def validate(data, stage=None, result=None, previous=None):
    """Single validation boundary for input and every internal stage handoff."""
    validate_input(data)
    if stage is None:
        return
    require(isinstance(result, dict), "stage output must be an object")
    if stage == "semantic":
        shape(result, ("query", "hits"), "semantic")
        require(result["query"] == data["query"], "query provenance mismatch")
        require(isinstance(result["hits"], list) and len(result["hits"]) <= data["limit"], "invalid hits")
        known = {p["id"] for p in data["products"]}
        seen = set()
        for hit in result["hits"]:
            shape(hit, ("product_id", "score", "lexical_score", "embedding_score"), "hit")
            text(hit["product_id"], "hit.product_id")
            require(hit["product_id"] in known and hit["product_id"] not in seen, "invalid hit product")
            seen.add(hit["product_id"])
            require(number(hit["score"]) and 0 < hit["score"] <= 1, "invalid relevance")
            require(number(hit["lexical_score"]) and 0 <= hit["lexical_score"] <= 1, "invalid lexical score")
            ev = hit["embedding_score"]
            require(ev is None or (number(ev) and 0 <= ev <= 1), "invalid embedding score")
            expected = hit["lexical_score"] if ev is None else (hit["lexical_score"] + ev) / 2
            require(abs(hit["score"] - expected) <= 1e-7, "inconsistent relevance")
        require(result["hits"] == sorted(result["hits"], key=lambda h: (-h["score"], h["product_id"])),
                "hits not ranked")
    elif stage == "journey":
        validate(data, "semantic", previous)
        require(result == journey(data, previous), "invalid journey or semantic provenance")
    elif stage == "sentiment":
        shape(previous, ("semantic", "journey"), "sentiment handoff")
        validate(data, "journey", previous["journey"], previous["semantic"])
        require(result == sentiment(data, previous["journey"]), "invalid sentiment or journey provenance")
    else:
        raise ValidationError("unknown stage")


def run(payload, embedding=None):
    data = copy.deepcopy(payload)
    validate(data)
    require(embedding is None or callable(embedding), "embedding must be callable")
    search = semantic(data, embedding)
    validate(data, "semantic", search)
    recommendations = journey(data, search)
    validate(data, "journey", recommendations, search)
    insights = sentiment(data, recommendations)
    validate(data, "sentiment", insights, {"semantic": search, "journey": recommendations})
    return {"status": "ok", "schema_version": 1, "fixture_label": data["fixture_label"],
            "semantic": search, "journey": recommendations, "sentiment": insights}


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = run(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
