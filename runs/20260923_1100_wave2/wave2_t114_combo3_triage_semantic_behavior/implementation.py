"""Synthetic, deterministic triage -> search -> personalization reference CLI.

Run: python -B implementation.py example_input.json
Library injection: run_pipeline(request, embed=callable). The callable receives a
list of strings (query, then catalog texts) and returns equal-length numeric
vectors. No providers, persistence, or network access are implemented.
"""

import copy
import json
import math
import re
import sys
from datetime import datetime


class ValidationError(ValueError):
    pass


DEFAULT_CONFIG = {
    "routes": [
        {"category": "billing", "keywords": ["refund", "payment", "invoice"],
         "priority": "high", "team": "payments"},
        {"category": "technical", "keywords": ["broken", "error", "setup"],
         "priority": "normal", "team": "technical-support"},
    ],
    "default_route": {"category": "general", "priority": "normal", "team": "support"},
    "search_limit": 10,
    "final_limit": 5,
    "half_life_days": 30.0,
    "behavior_weight": 0.5,
    "cold_start_weight": 0.1,
}


class Validator:
    """Shared validation for input, injected vectors, and every stage handoff."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def fields(cls, obj, required, optional=()):
        cls.require(isinstance(obj, dict), "Expected an object")
        cls.require(set(required) <= obj.keys(), "Missing fields: " +
                    ", ".join(sorted(set(required) - obj.keys())))
        cls.require(obj.keys() <= set(required) | set(optional), "Unknown fields")

    @classmethod
    def text(cls, value, name, allow_empty=False):
        cls.require(isinstance(value, str) and (allow_empty or value.strip()),
                    name + " must be a nonempty string")

    @classmethod
    def number(cls, value, name, minimum=0, maximum=None):
        cls.require(type(value) in (int, float), name + " must be numeric")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        cls.require(finite and value >= minimum and
                    (maximum is None or value <= maximum),
                    name + " is outside its finite range")

    @classmethod
    def timestamp(cls, value):
        cls.text(value, "timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("Invalid ISO 8601 timestamp") from exc
        cls.require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
                    "Timestamps require an explicit timezone")
        return parsed

    @classmethod
    def route(cls, route, keywords):
        cls.fields(route, ("category", "priority", "team", "keywords") if keywords
                   else ("category", "priority", "team"))
        for key in ("category", "team"):
            cls.text(route[key], key)
        cls.require(route["priority"] in ("low", "normal", "high", "urgent"),
                    "Unknown priority")
        if keywords:
            cls.require(isinstance(route["keywords"], list) and route["keywords"],
                        "Route keywords must be a nonempty list")
            for keyword in route["keywords"]:
                cls.text(keyword, "keyword")
                cls.require(tokens(keyword), "Keywords must contain searchable tokens")

    @classmethod
    def request(cls, data):
        cls.fields(data, ("schema_version", "fixture_label", "as_of", "ticket",
                          "catalog", "events"), ("config",))
        cls.require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                    "Only schema_version 1 is supported")
        cls.text(data["fixture_label"], "fixture_label")
        as_of = cls.timestamp(data["as_of"])
        cls.fields(data["ticket"], ("id", "text"))
        for key in ("id", "text"):
            cls.text(data["ticket"][key], "ticket." + key)
        cls.require(tokens(data["ticket"]["text"]), "Ticket needs searchable tokens")
        cls.require(isinstance(data["catalog"], list), "catalog must be a list")
        ids = set()
        for product in data["catalog"]:
            cls.fields(product, ("id", "title", "description", "category", "popularity"))
            for key in ("id", "title", "category", "description"):
                cls.text(product[key], "product." + key, allow_empty=key == "description")
            cls.require(product["id"] not in ids, "Duplicate product id")
            ids.add(product["id"])
            cls.number(product["popularity"], "popularity", maximum=1)
        cls.require(isinstance(data["events"], list), "events must be a list")
        for event in data["events"]:
            cls.fields(event, ("product_id", "action", "timestamp"))
            cls.text(event["product_id"], "event.product_id")
            cls.require(event["product_id"] in ids, "Unknown event product")
            cls.require(event["action"] in ("browse", "purchase"), "Unknown event action")
            cls.require(cls.timestamp(event["timestamp"]) <= as_of, "Future event")
        config = copy.deepcopy(DEFAULT_CONFIG)
        overrides = data.get("config", {})
        cls.fields(overrides, (), config.keys())
        config.update(copy.deepcopy(overrides))
        cls.require(isinstance(config["routes"], list), "routes must be a list")
        for route in config["routes"]:
            cls.route(route, True)
        cls.route(config["default_route"], False)
        for name in ("search_limit", "final_limit"):
            cls.require(type(config[name]) is int and 1 <= config[name] <= 1000,
                        name + " must be an integer between 1 and 1000")
        cls.number(config["half_life_days"], "half_life_days", 0.000001, 1000000)
        for name in ("behavior_weight", "cold_start_weight"):
            cls.number(config[name], name, 0, 100)
        result = copy.deepcopy(data)
        result["config"] = config
        return result

    @classmethod
    def vectors(cls, result, count):
        cls.require(isinstance(result, list) and len(result) == count,
                    "Embedding must return one vector per text")
        dimension = None
        for vector in result:
            cls.require(isinstance(vector, list) and vector, "Empty or invalid vector")
            dimension = len(vector) if dimension is None else dimension
            cls.require(len(vector) == dimension, "Embedding dimensions differ")
            for value in vector:
                cls.number(value, "embedding coordinate", -1e100, 1e100)
        return result

    @classmethod
    def envelope(cls, envelope, stage):
        stages = ("triage", "semantic", "behavior")
        index = stages.index(stage)
        cls.fields(envelope, ("schema_version", "status", "request") + stages[:index + 1])
        cls.require(envelope["status"] == "ok" and envelope["schema_version"] == 1,
                    "Invalid envelope status or version")
        request = cls.request(envelope["request"])
        config = request["config"]
        triage = envelope["triage"]
        cls.fields(triage, ("ticket_id", "category", "priority", "accountable_team",
                            "query", "matched_keywords"))
        cls.require(triage["ticket_id"] == request["ticket"]["id"], "Ticket handoff mismatch")
        cls.require(triage["query"] == request["ticket"]["text"], "Query handoff mismatch")
        cls.route({"category": triage["category"], "priority": triage["priority"],
                   "team": triage["accountable_team"]}, False)
        cls.require(isinstance(triage["matched_keywords"], list), "Invalid keyword evidence")
        for word in triage["matched_keywords"]:
            cls.text(word, "matched keyword")
        if index >= 1:
            search = envelope["semantic"]
            cls.fields(search, ("ticket_id", "query", "category", "embedding_used", "candidates"))
            for key in ("ticket_id", "query", "category"):
                cls.require(search[key] == triage[key], "Semantic handoff mismatch: " + key)
            cls.require(type(search["embedding_used"]) is bool, "Invalid embedding flag")
            cls.require(isinstance(search["candidates"], list), "Invalid candidates")
            cls.require(len(search["candidates"]) <= config["search_limit"], "Too many candidates")
            ids = {p["id"] for p in request["catalog"]}
            seen = set()
            for candidate in search["candidates"]:
                cls.fields(candidate, ("product_id", "semantic_score"))
                cls.text(candidate["product_id"], "candidate.product_id")
                cls.require(candidate["product_id"] in ids and candidate["product_id"] not in seen,
                            "Unknown or duplicate candidate")
                seen.add(candidate["product_id"])
                cls.number(candidate["semantic_score"], "semantic_score", 0, 1.2)
        if index >= 2:
            behavior = envelope["behavior"]
            cls.fields(behavior, ("ticket_id", "accountable_team", "priority",
                                  "cold_start", "recommendations"))
            for key in ("ticket_id", "accountable_team", "priority"):
                cls.require(behavior[key] == triage[key], "Behavior handoff mismatch: " + key)
            cls.require(type(behavior["cold_start"]) is bool, "Invalid cold-start flag")
            cls.require(behavior["cold_start"] == (not request["events"]),
                        "Cold-start flag does not match history")
            recommendations = behavior["recommendations"]
            cls.require(isinstance(recommendations, list) and
                        len(recommendations) <= config["final_limit"], "Invalid recommendations")
            candidates = {c["product_id"]: c for c in search["candidates"]}
            products = {p["id"]: p for p in request["catalog"]}
            seen = set()
            for row in recommendations:
                cls.fields(row, ("product_id", "title", "semantic_score",
                                 "personalization_score", "score"))
                cls.text(row["product_id"], "recommendation.product_id")
                cls.require(row["product_id"] in candidates and row["product_id"] not in seen,
                            "Recommendation did not originate in search")
                seen.add(row["product_id"])
                cls.require(row["title"] == products[row["product_id"]]["title"],
                            "Product title handoff mismatch")
                cls.require(row["semantic_score"] == candidates[row["product_id"]]["semantic_score"],
                            "Semantic score handoff mismatch")
                for key in ("semantic_score", "personalization_score", "score"):
                    cls.number(row[key], key)
                cls.require(math.isclose(row["score"], row["semantic_score"] +
                                        row["personalization_score"], abs_tol=1e-9),
                            "Final score mismatch")
        return envelope


def tokens(text):
    return set(re.findall(r"\w+", text.casefold()))


def triage_stage(request):
    request = Validator.request(request)
    query_tokens = tokens(request["ticket"]["text"])
    config = request["config"]
    selected = config["default_route"]
    matched = []
    for route in config["routes"]:
        hits = [keyword for keyword in route["keywords"] if tokens(keyword) <= query_tokens]
        # Configured order resolves equal-evidence routes, including priority conflicts.
        if len(hits) > len(matched):
            selected, matched = route, hits
    result = {
        "schema_version": 1, "status": "ok", "request": request,
        "triage": {
            "ticket_id": request["ticket"]["id"], "category": selected["category"],
            "priority": selected["priority"], "accountable_team": selected["team"],
            "query": request["ticket"]["text"], "matched_keywords": matched,
        },
    }
    return Validator.envelope(result, "triage")


class SearchIndex:
    """In-memory inverted token index; optional cosine vectors augment relevance."""

    def __init__(self, products):
        self.products = products
        self.documents = []
        self.postings = {}
        for index, product in enumerate(products):
            document = tokens(product["title"] + " " + product["description"])
            self.documents.append(document)
            for token in document:
                self.postings.setdefault(token, set()).add(index)

    def rank(self, triage, embed):
        query = tokens(triage["query"])
        overlaps = {}
        for token in query:
            for index in self.postings.get(token, ()):
                overlaps[index] = overlaps.get(index, 0) + 1
        vectors = None
        if embed is not None and self.products:
            Validator.require(callable(embed), "embed must be callable")
            texts = [triage["query"]] + [
                p["title"] + " " + p["description"] for p in self.products]
            try:
                vectors = Validator.vectors(embed(texts), len(texts))
            except ValidationError:
                raise
            except Exception as exc:
                raise ValidationError("Injected embedding callable failed") from exc
        rows = []
        for index, product in enumerate(self.products):
            overlap = overlaps.get(index, 0)
            lexical = overlap / len(query | self.documents[index])
            score = lexical
            if vectors is not None:
                left, right = vectors[0], vectors[index + 1]
                a, b = math.hypot(*left), math.hypot(*right)
                cosine = sum((x / a) * (y / b) for x, y in zip(left, right)) if a and b else 0
                score = 0.5 * lexical + 0.5 * max(0, min(1, cosine))
            if product["category"].casefold() == triage["category"].casefold():
                score += 0.2
            rows.append({"product_id": product["id"], "semantic_score": round(score, 10)})
        return sorted(rows, key=lambda row: (-row["semantic_score"], row["product_id"]))


def semantic_stage(envelope, embed=None):
    Validator.envelope(envelope, "triage")
    Validator.require(embed is None or callable(embed), "embed must be callable")
    result = copy.deepcopy(envelope)
    request, triage = result["request"], result["triage"]
    rows = SearchIndex(request["catalog"]).rank(triage, embed)
    result["semantic"] = {
        "ticket_id": triage["ticket_id"], "query": triage["query"],
        "category": triage["category"],
        "embedding_used": embed is not None and bool(request["catalog"]),
        "candidates": rows[:request["config"]["search_limit"]],
    }
    return Validator.envelope(result, "semantic")


def behavior_stage(envelope):
    Validator.envelope(envelope, "semantic")
    result = copy.deepcopy(envelope)
    request, triage = result["request"], result["triage"]
    config = request["config"]
    as_of = Validator.timestamp(request["as_of"])
    history = {}
    for event in request["events"]:
        age_days = (as_of - Validator.timestamp(event["timestamp"])).total_seconds() / 86400
        strength = (3 if event["action"] == "purchase" else 1) * (
            2 ** (-age_days / config["half_life_days"]))
        history[event["product_id"]] = history.get(event["product_id"], 0) + strength
    cold_start = not request["events"]
    products = {p["id"]: p for p in request["catalog"]}
    rows = []
    for candidate in result["semantic"]["candidates"]:
        product = products[candidate["product_id"]]
        strength = history.get(product["id"], 0)
        bonus = (config["cold_start_weight"] * product["popularity"] if cold_start else
                 config["behavior_weight"] * strength / (1 + strength))
        bonus = round(bonus, 10)
        rows.append({
            **candidate, "title": product["title"], "personalization_score": bonus,
            "score": round(candidate["semantic_score"] + bonus, 10),
        })
    rows.sort(key=lambda row: (-row["score"], row["product_id"]))
    result["behavior"] = {
        "ticket_id": triage["ticket_id"], "accountable_team": triage["accountable_team"],
        "priority": triage["priority"], "cold_start": cold_start,
        "recommendations": rows[:config["final_limit"]],
    }
    return Validator.envelope(result, "behavior")


def run_pipeline(data, embed=None):
    return behavior_stage(semantic_stage(triage_stage(data), embed))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 1:
            raise ValidationError("Usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle)
        output = run_pipeline(data)
        serialized = json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True)
    except (ValidationError, OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
