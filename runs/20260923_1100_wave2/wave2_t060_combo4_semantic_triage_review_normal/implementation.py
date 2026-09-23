"""Synthetic reference pipeline; Python 3.10+ standard library only.

Run: python -B implementation.py example_input.json
Library: run_pipeline(payload, embedding=None), where embedding(text) returns
a nonempty, finite numeric vector of consistent dimension. No provider is used.

Input/output use one versioned envelope. Every handoff validates the complete
envelope, including stage order, reference integrity, and exact citation spans.
Review checks literal evidence only; it is not certification or legal advice.
"""

import copy
import json
import math
import re
import sys
from collections import Counter


VERSION = "1.0"
STAGES = ("semantic", "triage", "review", "normal")
PRIORITIES = {"low", "normal", "high", "urgent"}


class ValidationError(ValueError):
    """Invalid request, plugin response, or stage handoff."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), path="object"):
    require(isinstance(value, dict), f"{path} must be an object")
    keys = set(value)
    require(set(required) <= keys, f"{path} missing fields: {sorted(set(required) - keys)}")
    require(keys <= set(required) | set(optional), f"{path} has unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonempty text")


def integer(value, path, lower=1, upper=100):
    require(type(value) is int and lower <= value <= upper,
            f"{path} must be an integer in [{lower}, {upper}]")


def strings(value, path, nonempty=False):
    require(isinstance(value, list), f"{path} must be a list")
    require(not nonempty or bool(value), f"{path} must not be empty")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), f"{path} contains duplicates")


def records(value, path):
    require(isinstance(value, list), f"{path} must be a list")
    ids = set()
    for item in value:
        require(isinstance(item, dict), f"{path} item must be an object")
        text(item.get("id"), f"{path}.id")
        require(item["id"] not in ids, f"{path} duplicate id: {item['id']}")
        ids.add(item["id"])
    return ids


def tokens(value):
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def term_present(term, value):
    needle, haystack = tokens(term), tokens(value)
    return bool(needle) and any(
        haystack[i:i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def validate_terms(value, path):
    strings(value, path, nonempty=True)
    for term in value:
        require(bool(tokens(term)), f"{path} terms must contain words")
    normalized = [tuple(tokens(term)) for term in value]
    require(len(normalized) == len(set(normalized)), f"{path} has equivalent terms")


def validate_request(data):
    fields(data, ("schema_version", "synthetic", "query", "products", "tickets",
                  "requirements", "sources", "config"), path="input")
    require(data["schema_version"] == VERSION, "unsupported schema_version")
    require(data["synthetic"] is True, "reference input must be labeled synthetic")
    text(data["query"], "query")
    require(bool(tokens(data["query"])), "query must contain words")
    product_ids = records(data["products"], "products")
    ticket_ids = records(data["tickets"], "tickets")
    requirement_ids = records(data["requirements"], "requirements")
    source_ids = records(data["sources"], "sources")
    for source in data["sources"]:
        fields(source, ("id", "title", "passages"), path="source")
        text(source["title"], "source.title")
        records(source["passages"], "passages")
        for passage in source["passages"]:
            fields(passage, ("id", "text"), path="passage")
            text(passage["text"], "passage.text")
    for product in data["products"]:
        fields(product, ("id", "name", "description", "source_ids"), path="product")
        text(product["name"], "product.name")
        text(product["description"], "product.description")
        strings(product["source_ids"], "product.source_ids")
        require(set(product["source_ids"]) <= source_ids, "unknown product source")
    for ticket in data["tickets"]:
        fields(ticket, ("id", "subject", "body"), path="ticket")
        text(ticket["subject"], "ticket.subject")
        text(ticket["body"], "ticket.body")
    config = data["config"]
    fields(config, ("search_limit", "research_limit", "routing_rules", "default_route"),
           path="config")
    integer(config["search_limit"], "search_limit")
    integer(config["research_limit"], "research_limit")
    require(isinstance(config["routing_rules"], list), "routing_rules must be a list")
    categories = set()
    for rule in config["routing_rules"]:
        fields(rule, ("category", "terms", "priority", "owner"), path="routing_rule")
        validate_route(rule)
        validate_terms(rule["terms"], "routing_rule.terms")
        require(rule["category"] not in categories, "duplicate routing category")
        categories.add(rule["category"])
    fields(config["default_route"], ("category", "priority", "owner"), path="default_route")
    validate_route(config["default_route"])
    categories.add(config["default_route"]["category"])
    for requirement in data["requirements"]:
        fields(requirement, ("id", "description", "category", "terms"), path="requirement")
        text(requirement["description"], "requirement.description")
        text(requirement["category"], "requirement.category")
        require(requirement["category"] == "*" or requirement["category"] in categories,
                "requirement category has no route")
        validate_terms(requirement["terms"], "requirement.terms")
    return product_ids, ticket_ids, requirement_ids, source_ids


def validate_route(route):
    text(route["category"], "route.category")
    require(route["category"] != "*", "route.category cannot be wildcard")
    text(route["priority"], "route.priority")
    require(route["priority"] in PRIORITIES, "invalid route priority")
    text(route["owner"], "route.owner")


def passage_map(data):
    return {(s["id"], p["id"]): p["text"]
            for s in data["sources"] for p in s["passages"]}


def citation(source_id, passage_id, value, start=0, end=None):
    end = len(value) if end is None else end
    return {"source_id": source_id, "passage_id": passage_id,
            "start": start, "end": end, "quote": value[start:end]}


def validate_citation(value, passages, allowed):
    fields(value, ("source_id", "passage_id", "start", "end", "quote"), path="citation")
    text(value["source_id"], "citation.source_id")
    text(value["passage_id"], "citation.passage_id")
    key = (value["source_id"], value["passage_id"])
    require(key in passages and key[0] in allowed, "citation outside eligible sources")
    original = passages[key]
    require(type(value["start"]) is int and type(value["end"]) is int,
            "citation offsets must be integers")
    require(0 <= value["start"] < value["end"] <= len(original), "invalid citation span")
    require(value["quote"] == original[value["start"]:value["end"]],
            "citation quote differs from source")


def validate_envelope(envelope, completed):
    fields(envelope, ("schema_version", "status", "synthetic", "input", "stages"),
           path="envelope")
    require(envelope["schema_version"] == VERSION and envelope["synthetic"] is True,
            "invalid envelope metadata")
    require(envelope["status"] == ("ok" if completed == 4 else "processing"),
            "invalid envelope status")
    data = envelope["input"]
    product_ids, ticket_ids, requirement_ids, source_ids = validate_request(data)
    stages = envelope["stages"]
    require(isinstance(stages, dict) and list(stages) == list(STAGES[:completed]),
            "stages must appear in pipeline order")
    products = {p["id"]: p for p in data["products"]}
    passages = passage_map(data)
    if completed >= 1:
        result = stages["semantic"]
        fields(result, ("query", "method", "hits"))
        require(result["query"] == data["query"], "semantic query mismatch")
        require(result["method"] in ("lexical", "lexical+embedding"), "invalid search method")
        require(isinstance(result["hits"], list), "search hits must be a list")
        require(len(result["hits"]) <= data["config"]["search_limit"], "too many hits")
        seen = set()
        previous = None
        for hit in result["hits"]:
            fields(hit, ("product_id", "score", "matched_tokens", "source_ids"))
            text(hit["product_id"], "hit.product_id")
            require(hit["product_id"] in product_ids and hit["product_id"] not in seen,
                    "unknown or duplicate product hit")
            seen.add(hit["product_id"])
            require(type(hit["score"]) in (int, float) and math.isfinite(hit["score"])
                    and hit["score"] > 0, "invalid search score")
            order = (-hit["score"], hit["product_id"])
            require(previous is None or previous <= order, "hits not ranked")
            previous = order
            expected = sorted(set(tokens(data["query"])) & set(tokens(
                products[hit["product_id"]]["name"] + " " +
                products[hit["product_id"]]["description"])))
            require(hit["matched_tokens"] == expected, "incorrect matched tokens")
            require(hit["source_ids"] == products[hit["product_id"]]["source_ids"],
                    "search source provenance mismatch")
    if completed >= 2:
        result = stages["triage"]
        fields(result, ("routes",))
        require(isinstance(result["routes"], list), "routes must be a list")
        require([r.get("ticket_id") for r in result["routes"] if isinstance(r, dict)]
                == [t["id"] for t in data["tickets"]], "ticket coverage mismatch")
        for route in result["routes"]:
            fields(route, ("ticket_id", "category", "priority", "owner", "product_ids",
                           "source_ids", "matched_terms", "reason"))
            require(route == route_ticket(data, stages["semantic"], next(
                t for t in data["tickets"] if t["id"] == route["ticket_id"])),
                "triage handoff mismatch")
    if completed >= 3:
        result = stages["review"]
        fields(result, ("assurance", "checks"))
        require(result["assurance"] == "literal-evidence-check; not certification",
                "invalid review assurance")
        require(isinstance(result["checks"], list), "review checks must be a list")
        expected_checks = [(r, q) for r in stages["triage"]["routes"]
                           for q in data["requirements"]
                           if q["category"] in ("*", r["category"])]
        require(len(result["checks"]) == len(expected_checks), "review coverage mismatch")
        for check, (route, requirement) in zip(result["checks"], expected_checks):
            fields(check, ("ticket_id", "requirement_id", "owner", "priority",
                           "status", "missing_terms", "evidence", "gap"))
            require(check == review_requirement(data, route, requirement),
                    "review handoff mismatch")
            for evidence in check["evidence"]:
                validate_citation(evidence["citation"], passages, route["source_ids"])
    if completed >= 4:
        result = stages["normal"]
        fields(result, ("mode", "findings"))
        require(result["mode"] == "extractive; no unsupported conclusions", "invalid research mode")
        require(isinstance(result["findings"], list), "findings must be a list")
        require(len(result["findings"]) == len(stages["review"]["checks"]),
                "research coverage mismatch")
        for finding, check in zip(result["findings"], stages["review"]["checks"]):
            fields(finding, ("ticket_id", "requirement_id", "owner", "priority",
                             "review_status", "unresolved_terms", "query", "passages"))
            require(finding == research_check(data, stages["triage"], check),
                    "research handoff mismatch")
            route = next(r for r in stages["triage"]["routes"]
                         if r["ticket_id"] == check["ticket_id"])
            for retrieved in finding["passages"]:
                validate_citation(retrieved["citation"], passages, route["source_ids"])
                require(retrieved["finding"] == retrieved["citation"]["quote"],
                        "research finding must be an exact extract")
    return envelope


def vector(value):
    require(isinstance(value, (list, tuple)) and len(value) > 0,
            "embedding must return a nonempty list or tuple")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value),
            "embedding elements must be finite numbers")
    # Scaling before normalization avoids overflow on large but finite vectors.
    scale = max(abs(x) for x in value)
    if not scale:
        return [0.0] * len(value)
    scaled = [x / scale for x in value]
    norm = math.sqrt(sum(x * x for x in scaled))
    return [x / norm for x in scaled]


class SearchIndex:
    """TF-IDF cosine index with optional additive positive embedding cosine."""

    def __init__(self, documents):
        self.documents = documents
        self.counts = [Counter(tokens(value)) for _, value in documents]
        df = Counter(word for counts in self.counts for word in counts)
        self.idf = {word: 1 + math.log((1 + len(documents)) / (1 + count))
                    for word, count in df.items()}

    def rank(self, query, limit, embedding=None):
        query_counts = Counter(tokens(query))
        q = {word: count * self.idf.get(word, 1 + math.log(1 + len(self.documents)))
             for word, count in query_counts.items()}
        qnorm = math.sqrt(sum(weight * weight for weight in q.values()))
        qvector = None
        if embedding is not None:
            require(callable(embedding), "embedding must be callable")
            qvector = self._embed(embedding, query)
        hits = []
        for (identifier, value), counts in zip(self.documents, self.counts):
            d = {word: count * self.idf[word] for word, count in counts.items()}
            dnorm = math.sqrt(sum(weight * weight for weight in d.values()))
            score = (sum(q.get(word, 0) * weight for word, weight in d.items())
                     / (qnorm * dnorm)) if qnorm and dnorm else 0.0
            if qvector is not None:
                dvector = self._embed(embedding, value)
                require(len(qvector) == len(dvector), "embedding dimension mismatch")
                score += max(0.0, sum(x * y for x, y in zip(qvector, dvector)))
            score = round(score, 10)
            if score > 0:
                hits.append((identifier, score))
        return sorted(hits, key=lambda pair: (-pair[1], pair[0]))[:limit]

    @staticmethod
    def _embed(embedding, value):
        try:
            return vector(embedding(value))
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError("embedding callable failed") from exc


def semantic_stage(envelope, embedding=None):
    validate_envelope(envelope, 0)
    data = envelope["input"]
    products = {p["id"]: p for p in data["products"]}
    documents = [(p["id"], p["name"] + " " + p["description"]) for p in data["products"]]
    ranked = SearchIndex(documents).rank(data["query"], data["config"]["search_limit"], embedding)
    hits = [{"product_id": identifier, "score": score,
             "matched_tokens": sorted(set(tokens(data["query"])) & set(tokens(
                 products[identifier]["name"] + " " + products[identifier]["description"]))),
             "source_ids": list(products[identifier]["source_ids"])}
            for identifier, score in ranked]
    envelope["stages"]["semantic"] = {
        "query": data["query"], "method": "lexical+embedding" if embedding is not None else "lexical",
        "hits": hits}
    return validate_envelope(envelope, 1)


def route_ticket(data, semantic, ticket):
    product_ids = [h["product_id"] for h in semantic["hits"]]
    products = {p["id"]: p for p in data["products"]}
    context = " ".join(products[p]["name"] + " " + products[p]["description"]
                       for p in product_ids)
    ticket_text = ticket["subject"] + " " + ticket["body"]
    best, best_score, matched = None, 0, []
    for rule in data["config"]["routing_rules"]:
        direct = [term for term in rule["terms"] if term_present(term, ticket_text)]
        contextual = [term for term in rule["terms"] if term_present(term, context)]
        score = 3 * len(direct) + len(contextual)
        if score > best_score:
            best, best_score = rule, score
            matched = sorted(set(direct + contextual))
    selected = best or data["config"]["default_route"]
    return {
        "ticket_id": ticket["id"], "category": selected["category"],
        "priority": selected["priority"], "owner": selected["owner"],
        "product_ids": product_ids,
        "source_ids": sorted({s for h in semantic["hits"] for s in h["source_ids"]}),
        "matched_terms": matched,
        "reason": f"rule score {best_score}; ticket terms weight 3, product terms weight 1"
                  if best else "default route; no rule terms matched",
    }


def triage_stage(envelope):
    validate_envelope(envelope, 1)
    data = envelope["input"]
    envelope["stages"]["triage"] = {
        "routes": [route_ticket(data, envelope["stages"]["semantic"], t) for t in data["tickets"]]}
    return validate_envelope(envelope, 2)


def eligible_passages(data, route):
    return [(s["id"], p["id"], p["text"])
            for s in data["sources"] if s["id"] in route["source_ids"]
            for p in s["passages"]]


def review_requirement(data, route, requirement):
    candidates = []
    for source_id, passage_id, value in eligible_passages(data, route):
        matched = [term for term in requirement["terms"] if term_present(term, value)]
        if matched:
            candidates.append((source_id, passage_id, value, matched))
    candidates.sort(key=lambda item: (-len(item[3]), item[0], item[1]))
    best = candidates[:1]
    matched = best[0][3] if best else []
    missing = [term for term in requirement["terms"] if term not in matched]
    status = "supported" if not missing else ("partial" if matched else "missing")
    return {
        "ticket_id": route["ticket_id"], "requirement_id": requirement["id"],
        "owner": route["owner"], "priority": route["priority"], "status": status,
        "missing_terms": missing,
        "evidence": [{"matched_terms": terms, "citation": citation(s, p, value)}
                     for s, p, value, terms in best],
        "gap": None if status == "supported" else {
            "description": requirement["description"],
            "reason": "No single eligible passage contains every required term.",
            "missing_terms": missing,
        },
    }


def review_stage(envelope):
    validate_envelope(envelope, 2)
    data = envelope["input"]
    envelope["stages"]["review"] = {
        "assurance": "literal-evidence-check; not certification",
        "checks": [review_requirement(data, route, requirement)
                   for route in envelope["stages"]["triage"]["routes"]
                   for requirement in data["requirements"]
                   if requirement["category"] in ("*", route["category"])]}
    return validate_envelope(envelope, 3)


def sentence_spans(value):
    # Offsets are Python Unicode character positions, not UTF-8 byte positions.
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)", value):
        start, end = match.span()
        while start < end and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end


def research_check(data, triage, check):
    route = next(r for r in triage["routes"] if r["ticket_id"] == check["ticket_id"])
    requirement = next(q for q in data["requirements"] if q["id"] == check["requirement_id"])
    focus = check["missing_terms"] or requirement["terms"]
    query = " ".join(focus) + " " + requirement["description"]
    available = eligible_passages(data, route)
    # Tuple identifiers preserve source/passage identity without delimiter collisions.
    ranked = SearchIndex([((s, p), value) for s, p, value in available]).rank(
        query, data["config"]["research_limit"])
    lookup = {(s, p): value for s, p, value in available}
    retrieved = []
    for (source_id, passage_id), score in ranked:
        value = lookup[(source_id, passage_id)]
        spans = list(sentence_spans(value)) or [(0, len(value))]
        query_tokens = set(tokens(query))
        start, end = max(spans, key=lambda span: (
            len(query_tokens & set(tokens(value[span[0]:span[1]]))), -span[0]))
        quote = citation(source_id, passage_id, value, start, end)
        retrieved.append({"score": score, "finding": quote["quote"], "citation": quote})
    return {
        "ticket_id": check["ticket_id"], "requirement_id": check["requirement_id"],
        "owner": check["owner"], "priority": check["priority"],
        "review_status": check["status"], "unresolved_terms": list(check["missing_terms"]),
        "query": query, "passages": retrieved,
    }


def normal_stage(envelope):
    validate_envelope(envelope, 3)
    data = envelope["input"]
    envelope["stages"]["normal"] = {
        "mode": "extractive; no unsupported conclusions",
        "findings": [research_check(data, envelope["stages"]["triage"], check)
                     for check in envelope["stages"]["review"]["checks"]]}
    envelope["status"] = "ok"
    return validate_envelope(envelope, 4)


def run_pipeline(payload, embedding=None):
    validate_request(payload)
    envelope = {"schema_version": VERSION, "synthetic": True, "status": "processing",
                "input": copy.deepcopy(payload), "stages": {}}
    semantic_stage(envelope, embedding)
    triage_stage(envelope)
    review_stage(envelope)
    return normal_stage(envelope)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object,
                                parse_constant=invalid_constant)
        output = run_pipeline(payload)
        code = 0
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        output = {"schema_version": VERSION, "status": "error",
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
