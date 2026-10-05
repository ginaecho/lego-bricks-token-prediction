"""Deterministic synthetic-feedback pipeline. Python 3.9+, standard library only.

Run: python -B implementation.py example_input.json
Library: run(payload, embedder=None); embedder(text) must return a finite,
nonzero numeric list of fixed dimension. No provider or network is used here.
"""

import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


class ValidationError(ValueError):
    pass


STAGES = ("input", "insights", "triage", "semantic")
PRIORITIES = ("urgent", "high", "normal", "low")
POSITIVE_WORDS = {"great", "love", "excellent", "helpful", "easy", "happy"}
NEGATIVE_WORDS = {"broken", "failed", "confusing", "late", "bad", "blocked", "hate"}
STOPWORDS = set(
    "a an the and or to of in on for my our your it is are was were be been "
    "i we you they this that with but have has had not no very please can "
    "could would should me us so as at from".split()
)
DEFAULT_CONFIG = {
    "theme_rules": [
        {"id": "billing", "label": "Billing", "keywords": ["payment", "refund"],
         "action": "Audit payment and refund failures with the affected customers."},
        {"id": "access", "label": "Account access", "keywords": ["login", "password"],
         "action": "Reproduce account access problems and improve recovery."},
        {"id": "delivery", "label": "Delivery", "keywords": ["shipping", "delivery"],
         "action": "Review delivery delays and provide tracking updates."},
        {"id": "usability", "label": "Usability", "keywords": ["confusing", "navigation"],
         "action": "Review reported user journeys and test a clearer interface."},
    ],
    "category_rules": [
        {"category": "account", "theme_ids": ["access"], "keywords": [],
         "priority": "high", "team": "Support", "owner": "account-duty"},
        {"category": "payments", "theme_ids": ["billing"], "keywords": [],
         "priority": "normal", "team": "Finance", "owner": "payments-duty"},
        {"category": "fulfillment", "theme_ids": ["delivery"], "keywords": [],
         "priority": "normal", "team": "Operations", "owner": "delivery-duty"},
        {"category": "product", "theme_ids": ["usability"], "keywords": [],
         "priority": "low", "team": "Product", "owner": "product-duty"},
    ],
    "priority_rules": [
        {"keywords": ["outage", "fraud", "security"], "priority": "urgent"},
        {"keywords": ["broken", "failed", "blocked"], "priority": "high"},
    ],
    "default_route": {
        "category": "general", "priority": "normal",
        "team": "Support", "owner": "support-duty",
    },
    "synonyms": {
        "payment": ["billing", "charge", "charged"],
        "refund": ["reimbursement", "reimburse"],
        "login": ["signin", "authentication"],
        "shipping": ["shipment"],
    },
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), name="object"):
    require(isinstance(value, dict), name + " must be an object")
    require(set(required) <= value.keys(), name + " is missing required fields")
    require(value.keys() <= set(required) | set(optional), name + " has unknown fields")


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + " must be nonblank text")


def string_list(value, name, allow_empty=True):
    require(isinstance(value, list), name + " must be a list")
    require(allow_empty or bool(value), name + " must not be empty")
    for item in value:
        text(item, name + " item")
    require(len(value) == len(set(value)), name + " must not contain duplicates")


def words(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def tokens(value, config):
    aliases = {
        alias.casefold(): canonical.casefold()
        for canonical, variants in config["synonyms"].items()
        for alias in [canonical] + variants
    }
    return [aliases.get(word, word) for word in words(value) if word not in STOPWORDS]


def matches(value, keywords, config):
    haystack = tokens(value, config)
    for keyword in keywords:
        needle = tokens(keyword, config)
        if needle and any(haystack[i:i + len(needle)] == needle
                          for i in range(len(haystack) - len(needle) + 1)):
            return True
    return False


def validate_config(config):
    fields(config, DEFAULT_CONFIG, name="config")
    require(isinstance(config["synonyms"], dict), "synonyms must be an object")
    seen = set()
    for canonical, aliases in config["synonyms"].items():
        string_list(aliases, "synonym aliases")
        for token in [canonical] + aliases:
            text(token, "synonym")
            require(words(token) == [token.casefold()], "synonyms must be single words")
            require(token.casefold() not in seen, "synonyms must be unambiguous")
            seen.add(token.casefold())
    theme_ids = set()
    for key in ("theme_rules", "category_rules", "priority_rules"):
        require(isinstance(config[key], list), key + " must be a list")
    for rule in config["theme_rules"]:
        fields(rule, ("id", "label", "keywords", "action"), name="theme rule")
        for key in ("id", "label", "action"):
            text(rule[key], "theme " + key)
        require(rule["id"] not in theme_ids and not rule["id"].startswith("inferred:"),
                "theme IDs must be unique and cannot start with inferred:")
        theme_ids.add(rule["id"])
        string_list(rule["keywords"], "theme keywords", False)
    categories = set()
    for rule in config["category_rules"]:
        fields(rule, ("category", "theme_ids", "keywords", "priority", "team", "owner"),
               name="category rule")
        string_list(rule["theme_ids"], "category theme_ids")
        string_list(rule["keywords"], "category keywords")
        require(set(rule["theme_ids"]) <= theme_ids, "category references unknown theme")
        require(rule["theme_ids"] or rule["keywords"], "category rule needs a match condition")
        require(isinstance(rule["category"], str), "category must be text")
        require(rule["category"] not in categories, "category rules must be unique")
        categories.add(rule["category"])
        validate_route({key: rule[key] for key in ("category", "priority", "team", "owner")})
    for rule in config["priority_rules"]:
        fields(rule, ("keywords", "priority"), name="priority rule")
        string_list(rule["keywords"], "priority keywords", False)
        require(rule["priority"] in PRIORITIES, "unknown priority")
    validate_route(config["default_route"])


def validate_route(route):
    fields(route, ("category", "priority", "team", "owner"), name="route")
    for key in ("category", "team", "owner"):
        text(route[key], key)
    require(route["priority"] in PRIORITIES, "unknown priority")


def validate(state, expected_stage):
    """Shared envelope and referential-integrity validation at every boundary."""
    require(expected_stage in STAGES, "unknown stage")
    completed = STAGES[1:STAGES.index(expected_stage) + 1]
    fields(state, ("schema_version", "fixture_label", "feedback", "queries", "config",
                   "status", "stage") + completed, name="pipeline envelope")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "schema_version must be 1")
    require(state["status"] == "ok" and state["stage"] == expected_stage,
            "invalid pipeline stage or status")
    text(state["fixture_label"], "fixture_label")
    require(isinstance(state["feedback"], list), "feedback must be a list")
    feedback = {}
    for item in state["feedback"]:
        fields(item, ("id", "customer_id", "source", "text"), name="feedback")
        for key, value in item.items():
            text(value, "feedback." + key)
        require(item["id"] not in feedback, "duplicate feedback id")
        feedback[item["id"]] = item
    require(isinstance(state["queries"], list), "queries must be a list")
    for query in state["queries"]:
        fields(query, ("text", "limit"), name="query")
        text(query["text"], "query text")
        require(type(query["limit"]) is int and 1 <= query["limit"] <= 100,
                "query limit must be an integer from 1 to 100")
    validate_config(state["config"])
    if "insights" not in completed:
        return
    insights = state["insights"]
    fields(insights, ("themes", "signals"), name="insights")
    require(isinstance(insights["themes"], list) and isinstance(insights["signals"], list),
            "insights themes and signals must be lists")
    themes = {}
    for theme in insights["themes"]:
        fields(theme, ("id", "label", "action", "feedback_ids", "customer_ids",
                       "count", "sentiment_counts"), name="theme")
        for key in ("id", "label", "action"):
            text(theme[key], "theme " + key)
        require(theme["id"] not in themes, "duplicate output theme")
        string_list(theme["feedback_ids"], "theme feedback_ids", False)
        string_list(theme["customer_ids"], "theme customer_ids", False)
        require(set(theme["feedback_ids"]) <= feedback.keys(), "theme references unknown feedback")
        require(type(theme["count"]) is int and theme["count"] == len(theme["feedback_ids"]),
                "theme count does not match evidence")
        require(set(theme["customer_ids"]) ==
                {feedback[fid]["customer_id"] for fid in theme["feedback_ids"]},
                "theme customer evidence does not match")
        fields(theme["sentiment_counts"], ("positive", "negative", "neutral"),
               name="sentiment counts")
        require(all(type(n) is int and n >= 0 for n in theme["sentiment_counts"].values()),
                "invalid sentiment counts")
        themes[theme["id"]] = theme
    signals = {}
    for signal in insights["signals"]:
        fields(signal, ("feedback_id", "theme_ids", "sentiment"), name="signal")
        text(signal["feedback_id"], "signal feedback_id")
        require(signal["feedback_id"] in feedback and signal["feedback_id"] not in signals,
                "invalid or duplicate signal feedback reference")
        string_list(signal["theme_ids"], "signal theme_ids", False)
        require(set(signal["theme_ids"]) <= themes.keys(), "signal references unknown theme")
        require(signal["sentiment"] in ("positive", "negative", "neutral"), "invalid sentiment")
        signals[signal["feedback_id"]] = signal
    require(signals.keys() == feedback.keys(), "signals must cover every feedback item")
    for tid, theme in themes.items():
        members = [fid for fid, signal in signals.items() if tid in signal["theme_ids"]]
        require(set(members) == set(theme["feedback_ids"]), "theme evidence mismatch")
        counts = Counter(signals[fid]["sentiment"] for fid in members)
        require(all(theme["sentiment_counts"][key] == counts[key]
                    for key in ("positive", "negative", "neutral")), "sentiment evidence mismatch")
    if "triage" not in completed:
        return
    fields(state["triage"], ("tickets",), name="triage")
    require(isinstance(state["triage"]["tickets"], list), "tickets must be a list")
    tickets = {}
    for ticket in state["triage"]["tickets"]:
        fields(ticket, ("id", "feedback_id", "customer_id", "source", "text", "theme_ids",
                        "actions", "sentiment", "category", "priority", "team", "owner",
                        "routing_reason"), name="ticket")
        text(ticket["id"], "ticket id")
        text(ticket["feedback_id"], "ticket feedback_id")
        require(ticket["id"] not in tickets and ticket["feedback_id"] in feedback,
                "invalid ticket identity")
        fid = ticket["feedback_id"]
        require(ticket["id"] == "ticket:" + fid, "ticket id must derive from feedback id")
        for key in ("customer_id", "source", "text"):
            require(ticket[key] == feedback[fid][key], "ticket lost feedback " + key)
        require(ticket["theme_ids"] == signals[fid]["theme_ids"]
                and ticket["sentiment"] == signals[fid]["sentiment"], "ticket lost insight signal")
        require(ticket["actions"] == [themes[tid]["action"] for tid in ticket["theme_ids"]],
                "ticket lost recommended actions")
        validate_route({key: ticket[key] for key in ("category", "priority", "team", "owner")})
        text(ticket["routing_reason"], "routing_reason")
        tickets[ticket["id"]] = ticket
    require({ticket["feedback_id"] for ticket in tickets.values()} == feedback.keys(),
            "tickets must cover every feedback item")
    if "semantic" not in completed:
        return
    search = state["semantic"]
    fields(search, ("mode", "index", "results"), name="semantic")
    require(search["mode"] in ("lexical_synonyms", "hybrid_embeddings"), "invalid search mode")
    fields(search["index"], ("documents", "postings"), name="index")
    require(isinstance(search["index"]["documents"], list), "documents must be a list")
    documents = {}
    expected_postings = defaultdict(list)
    for doc in search["index"]["documents"]:
        fields(doc, ("id", "text", "terms"), name="document")
        text(doc["id"], "document id")
        require(doc["id"] in tickets and doc["id"] not in documents, "invalid document reference")
        require(doc["text"] == document_text(tickets[doc["id"]], themes),
                "document does not reflect validated ticket")
        require(doc["terms"] == tokens(doc["text"], state["config"]), "invalid indexed terms")
        documents[doc["id"]] = doc
        for term in sorted(set(doc["terms"])):
            expected_postings[term].append(doc["id"])
    require(documents.keys() == tickets.keys(), "search index must cover all tickets")
    require(search["index"]["postings"] == dict(expected_postings), "invalid search postings")
    require(isinstance(search["results"], list)
            and len(search["results"]) == len(state["queries"]), "query result count mismatch")
    for query, result in zip(state["queries"], search["results"]):
        fields(result, ("query", "hits"), name="query result")
        require(result["query"] == query["text"], "query propagation mismatch")
        require(isinstance(result["hits"], list) and len(result["hits"]) <= query["limit"],
                "invalid hit list")
        seen = set()
        for hit in result["hits"]:
            fields(hit, ("ticket_id", "feedback_id", "score", "category", "priority",
                         "team", "owner", "theme_ids"), name="search hit")
            text(hit["ticket_id"], "hit ticket_id")
            require(hit["ticket_id"] in tickets and hit["ticket_id"] not in seen,
                    "invalid or duplicate search hit")
            seen.add(hit["ticket_id"])
            score = hit["score"]
            require(type(score) in (int, float) and math.isfinite(score)
                    and 0 < score <= 1, "invalid relevance score")
            for key in ("feedback_id", "category", "priority", "team", "owner", "theme_ids"):
                require(hit[key] == tickets[hit["ticket_id"]][key], "hit lost ticket " + key)
        require(result["hits"] == sorted(result["hits"], key=lambda h: (-h["score"], h["ticket_id"])),
                "hits must be ranked by score then ticket ID")


def prepare(payload):
    fields(payload, ("schema_version", "fixture_label", "feedback"),
           ("config", "queries"), name="input")
    state = copy.deepcopy(payload)
    overrides = state.get("config", {})
    require(isinstance(overrides, dict) and overrides.keys() <= DEFAULT_CONFIG.keys(),
            "config must be an object with known keys")
    state["config"] = {**copy.deepcopy(DEFAULT_CONFIG), **overrides}
    state.setdefault("queries", [])
    for query in state["queries"] if isinstance(state["queries"], list) else []:
        if isinstance(query, dict):
            query.setdefault("limit", 5)
    state.update(status="ok", stage="input")
    validate(state, "input")
    return state


def sentiment(value):
    terms = set(words(value))
    positive = len(terms & POSITIVE_WORDS)
    negative = len(terms & NEGATIVE_WORDS)
    return "positive" if positive > negative else "negative" if negative > positive else "neutral"


def customer_insights(state):
    validate(state, "input")
    result = copy.deepcopy(state)
    config = state["config"]
    frequencies = Counter(term for item in state["feedback"]
                          for term in set(tokens(item["text"], config)))
    rules = {rule["id"]: rule for rule in config["theme_rules"]}
    signals, members = [], defaultdict(list)
    for item in state["feedback"]:
        ids = [rule["id"] for rule in config["theme_rules"]
               if matches(item["text"], rule["keywords"], config)]
        if not ids:
            candidates = set(tokens(item["text"], config)) - POSITIVE_WORDS - NEGATIVE_WORDS
            topic = min(candidates, key=lambda word: (-frequencies[word], word)) if candidates else "other"
            tid = "inferred:" + topic
            rules[tid] = {"label": "Emerging theme: " + topic,
                          "action": "Review feedback about " + topic + " and select an improvement."}
            ids = [tid]
        signal = {"feedback_id": item["id"], "theme_ids": ids,
                  "sentiment": sentiment(item["text"])}
        signals.append(signal)
        for tid in ids:
            members[tid].append((item, signal))
    themes = []
    for tid, evidence in members.items():
        counts = Counter(signal["sentiment"] for _, signal in evidence)
        themes.append({
            "id": tid, "label": rules[tid]["label"], "action": rules[tid]["action"],
            "feedback_ids": sorted(item["id"] for item, _ in evidence),
            "customer_ids": sorted({item["customer_id"] for item, _ in evidence}),
            "count": len(evidence),
            "sentiment_counts": {name: counts[name] for name in ("positive", "negative", "neutral")},
        })
    result["insights"] = {
        "themes": sorted(themes, key=lambda theme: (-theme["count"], theme["id"])),
        "signals": sorted(signals, key=lambda signal: signal["feedback_id"]),
    }
    result["stage"] = "insights"
    validate(result, "insights")
    return result


def triage_tickets(state):
    validate(state, "insights")
    result = copy.deepcopy(state)
    config = state["config"]
    signals = {item["feedback_id"]: item for item in state["insights"]["signals"]}
    themes = {item["id"]: item for item in state["insights"]["themes"]}
    tickets = []
    for item in sorted(state["feedback"], key=lambda item: item["id"]):
        signal = signals[item["id"]]
        route = config["default_route"]
        reason = "default_route"
        for index, rule in enumerate(config["category_rules"]):
            if (set(rule["theme_ids"]) & set(signal["theme_ids"])
                    or matches(item["text"], rule["keywords"], config)):
                route = rule
                reason = "category_rules[" + str(index) + "]"
                break
        priority = route["priority"]
        for rule in config["priority_rules"]:
            if matches(item["text"], rule["keywords"], config):
                priority = min(priority, rule["priority"], key=PRIORITIES.index)
        tickets.append({
            **item, "id": "ticket:" + item["id"], "feedback_id": item["id"],
            "theme_ids": list(signal["theme_ids"]), "sentiment": signal["sentiment"],
            "actions": [themes[tid]["action"] for tid in signal["theme_ids"]],
            **{key: route[key] for key in ("category", "team", "owner")},
            "priority": priority, "routing_reason": reason,
        })
    result.update(stage="triage", triage={"tickets": tickets})
    validate(result, "triage")
    return result


def document_text(ticket, themes):
    return " ".join(
        [ticket[key] for key in ("text", "category", "priority", "team", "owner")]
        + [themes[tid]["label"] for tid in ticket["theme_ids"]] + ticket["actions"]
    )


def vector_for(embedder, value, dimension):
    try:
        vector = embedder(value)
    except Exception as error:
        raise ValidationError("embedding callable failed: " + type(error).__name__) from error
    require(isinstance(vector, list) and bool(vector), "embedding must be a nonempty numeric list")
    require(all(type(n) in (float, int) for n in vector), "embedding values must be numbers")
    try:
        vector = [float(n) for n in vector]
    except OverflowError as error:
        raise ValidationError("embedding values exceed numeric range") from error
    require(all(math.isfinite(n) for n in vector), "embedding values must be finite numbers")
    require(dimension is None or len(vector) == dimension, "embedding dimensions must agree")
    norm = math.hypot(*vector)
    require(math.isfinite(norm) and norm > 0, "embedding must have a finite nonzero norm")
    return [n / norm for n in vector]


def semantic_search(state, embedder=None):
    validate(state, "triage")
    require(embedder is None or callable(embedder), "embedder must be callable")
    result = copy.deepcopy(state)
    config = state["config"]
    tickets = state["triage"]["tickets"]
    themes = {theme["id"]: theme for theme in state["insights"]["themes"]}
    documents, postings = [], defaultdict(list)
    for ticket in tickets:
        value = document_text(ticket, themes)
        terms = tokens(value, config)
        documents.append({"id": ticket["id"], "text": value, "terms": terms})
        for term in sorted(set(terms)):
            postings[term].append(ticket["id"])
    idf = {term: math.log((1 + len(documents)) / (1 + len(ids))) + 1
           for term, ids in postings.items()}

    def lexical_vector(terms):
        weights = {term: (1 + math.log(count)) * idf[term]
                   for term, count in Counter(terms).items() if term in idf}
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {term: weight / norm for term, weight in weights.items()} if norm else {}

    lexical = [lexical_vector(doc["terms"]) for doc in documents]
    embedded, dimension = [], None
    if embedder is not None:
        for doc in documents:
            vector = vector_for(embedder, doc["text"], dimension)
            dimension = len(vector)
            embedded.append(vector)
    results = []
    for query in state["queries"]:
        query_vector = lexical_vector(tokens(query["text"], config))
        dense_query = None
        if embedder is not None:
            dense_query = vector_for(embedder, query["text"], dimension)
            dimension = len(dense_query)
        hits = []
        for index, ticket in enumerate(tickets):
            score = sum(weight * lexical[index].get(term, 0)
                        for term, weight in query_vector.items())
            if dense_query is not None:
                cosine = max(0.0, sum(a * b for a, b in zip(dense_query, embedded[index])))
                score = 0.65 * score + 0.35 * cosine
            score = round(min(1.0, max(0.0, score)), 8)
            if score > 0:
                hits.append({
                    "ticket_id": ticket["id"], "score": score,
                    **{key: copy.deepcopy(ticket[key]) for key in
                       ("feedback_id", "category", "priority", "team", "owner", "theme_ids")},
                })
        hits.sort(key=lambda hit: (-hit["score"], hit["ticket_id"]))
        results.append({"query": query["text"], "hits": hits[:query["limit"]]})
    result.update(stage="semantic", semantic={
        "mode": "hybrid_embeddings" if embedder is not None else "lexical_synonyms",
        "index": {"documents": documents, "postings": dict(sorted(postings.items()))},
        "results": results,
    })
    validate(result, "semantic")
    return result


def run(payload, embedder=None):
    return semantic_search(triage_tickets(customer_insights(prepare(payload))), embedder)


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        payload = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                             parse_constant=reject_constant, object_pairs_hook=unique_object)
        output = run(payload)
        code = 0
    except (ValueError, OSError, RecursionError, OverflowError) as error:
        output = {"status": "error", "error": str(error)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
