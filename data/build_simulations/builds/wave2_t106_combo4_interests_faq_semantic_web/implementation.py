"""Synthetic, offline discovery -> FAQ -> semantic search -> research pipeline.

Run: python -B implementation.py example_input.json
Embeddings may be injected into run_pipeline as a batch Python callable.
Web ingestion accepts supplied snapshots only; it never fetches a URL.
"""

import collections
import json
import math
import re
import sys
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys.split()), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")
    require(len(value) <= 20000, path + " is too long")
    return value


def array(value, path):
    require(isinstance(value, list) and len(value) <= 1000, path + " must be a bounded array")
    return value


def strings(value, path):
    array(value, path)
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), path + " contains duplicates")
    return value


def number(value, path, low=0, high=1):
    require(type(value) in (int, float), path + " must be a number")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    require(valid, path + " is outside its finite range")
    return value


def normalized(value):
    return value.strip().casefold()


def tokens(value):
    stop = {"a", "an", "and", "are", "can", "do", "for", "how", "i", "in",
            "is", "it", "of", "on", "the", "to", "what", "with"}
    return [word for word in re.findall(r"[^\W_]+", value.casefold()) if word not in stop]


def allowed_url(url, hosts):
    text(url, "web URL")
    require(not any(ch.isspace() or ord(ch) < 32 for ch in url) and "\\" not in url,
            "web URL contains ambiguous characters")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname in hosts
                 and parsed.username is None and parsed.password is None
                 and parsed.port in (None, 443) and not parsed.fragment
                 and parsed.netloc.casefold() in
                 {parsed.hostname, parsed.hostname + ":443"})
    except (ValueError, TypeError):
        valid = False
    require(valid, "web URL must be HTTPS on an exact allowlisted host without credentials or fragments")
    return url


def validate_input(value):
    shape(value, "schema_version fixture_label profile products faqs web settings", "input")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "unsupported schema_version")
    require(text(value["fixture_label"], "fixture_label").startswith("SYNTHETIC"),
            "fixture_label must start with SYNTHETIC")
    profile = value["profile"]
    shape(profile, "interests excluded_tags excluded_product_ids question", "profile")
    text(profile["question"], "profile.question")
    array(profile["interests"], "interests")
    seen = set()
    for interest in profile["interests"]:
        shape(interest, "tag weight", "interest")
        tag = normalized(text(interest["tag"], "interest.tag"))
        require(tag not in seen, "duplicate interest tag")
        seen.add(tag)
        number(interest["weight"], "interest.weight", 0.000001, 1000)
    strings(profile["excluded_tags"], "excluded_tags")
    strings(profile["excluded_product_ids"], "excluded_product_ids")
    products = {}
    for product in array(value["products"], "products"):
        shape(product, "id title description tags", "product")
        for field in ("id", "title", "description"):
            text(product[field], "product." + field)
        strings(product["tags"], "product.tags")
        require(product["id"] not in products, "duplicate product ID")
        products[product["id"]] = product
    require(set(profile["excluded_product_ids"]) <= products.keys(), "unknown excluded product ID")
    faqs = set()
    for faq in array(value["faqs"], "faqs"):
        shape(faq, "id question answer product_ids", "faq")
        for field in ("id", "question", "answer"):
            text(faq[field], "faq." + field)
        require(faq["id"] not in faqs, "duplicate FAQ ID")
        faqs.add(faq["id"])
        require(set(strings(faq["product_ids"], "faq.product_ids")) <= products.keys(),
                "unknown FAQ product ID")
    web = value["web"]
    shape(web, "allowed_hosts documents", "web")
    hosts = strings(web["allowed_hosts"], "allowed_hosts")
    for host in hosts:
        require(host == host.lower() and len(host) <= 253 and "." in host
                and all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part)
                        for part in host.split(".")),
                "allowed_hosts must contain lowercase DNS hostnames, not URLs")
    urls = set()
    for document in array(web["documents"], "web.documents"):
        shape(document, "url title text product_ids", "web document")
        allowed_url(document["url"], hosts)
        require(document["url"] not in urls, "duplicate web URL")
        urls.add(document["url"])
        text(document["title"], "web.title")
        text(document["text"], "web.text")
        require(set(strings(document["product_ids"], "web.product_ids")) <= products.keys()
                and bool(document["product_ids"]), "web documents need known product IDs")
    settings = value["settings"]
    shape(settings, "limit faq_threshold search_threshold web_threshold", "settings")
    require(type(settings["limit"]) is int and 1 <= settings["limit"] <= 20, "limit must be 1..20")
    for key in ("faq_threshold", "search_threshold", "web_threshold"):
        number(settings[key], key)
    return value


def excluded_ids(request):
    profile = request["profile"]
    tags = {normalized(tag) for tag in profile["excluded_tags"]}
    return set(profile["excluded_product_ids"]) | {
        product["id"] for product in request["products"]
        if tags.intersection(normalized(tag) for tag in product["tags"])
    }


def product_evidence(product):
    return {"id": "product:" + product["id"], "kind": "product", "source_id": product["id"],
            "title": product["title"], "text": product["description"], "url": None,
            "product_ids": [product["id"]]}


def faq_evidence(faq):
    return {"id": "faq:" + faq["id"], "kind": "faq", "source_id": faq["id"],
            "title": faq["question"], "text": faq["answer"], "url": None,
            "product_ids": faq["product_ids"][:]}


def web_evidence(document):
    return {"id": "web:" + document["url"], "kind": "web", "source_id": document["url"],
            "title": document["title"], "text": document["text"], "url": document["url"],
            "product_ids": document["product_ids"][:]}


def source_map(request):
    records = ([product_evidence(p) for p in request["products"]]
               + [faq_evidence(f) for f in request["faqs"]]
               + [web_evidence(d) for d in request["web"]["documents"]])
    return {record["id"]: record for record in records}


def interest_matches(request, product):
    tags = {normalized(tag) for tag in product["tags"]}
    return [interest for interest in request["profile"]["interests"]
            if normalized(interest["tag"]) in tags]


def explanation(matches):
    return "Matches your interests: " + ", ".join(
        f'{normalized(match["tag"])} (weight {match["weight"]:g})' for match in matches) + "."


def validate_stage(request, result, expected, previous=None):
    """Shared envelope, grounding, exclusion, and handoff validation."""
    shape(result, "stage status reason query product_ids items evidence", "stage output")
    require(result["stage"] == expected, "unexpected stage")
    text(result["query"], "stage.query")
    require(result["status"] in ("ok", "abstained"), "invalid stage status")
    ids = strings(result["product_ids"], "stage.product_ids")
    allowed = {product["id"] for product in request["products"]} - excluded_ids(request)
    require(set(ids) <= allowed, "stage exposes an unknown or excluded product")
    if previous is not None:
        require(set(ids) <= set(previous["product_ids"]), "stage widened its product scope")
        if expected == "faq":
            require(ids == previous["product_ids"], "FAQ must preserve recommendation scope")
        require(result["evidence"][:len(previous["evidence"])] == previous["evidence"],
                "stage discarded upstream provenance")
        expected_query = previous["query"]
        if expected == "faq" and result["items"]:
            expected_query += " " + result["items"][0]["text"]
        require(result["query"] == expected_query, "stage query does not match its handoff")
    evidence = array(result["evidence"], "stage.evidence")
    sources = source_map(request)
    evidence_ids = set()
    for record in evidence:
        shape(record, "id kind source_id title text url product_ids", "evidence")
        require(isinstance(record["id"], str) and record["id"] in sources,
                "unknown evidence")
        require(record == sources[record["id"]], "evidence differs from its source")
        require(record["id"] not in evidence_ids, "duplicate evidence")
        require(not excluded_ids(request).intersection(record["product_ids"]),
                "evidence mentions an excluded product")
        evidence_ids.add(record["id"])
    items = array(result["items"], "stage.items")
    require(len(items) <= request["settings"]["limit"], "too many stage results")
    require((result["status"] == "ok") == bool(items), "status does not match results")
    if items:
        require(result["reason"] is None, "successful output cannot have an abstention reason")
    else:
        text(result["reason"], "abstention reason")
    seen = set()
    products = {product["id"]: product for product in request["products"]}
    for item in items:
        shape(item, "id product_ids score text citations", "stage item")
        text(item["id"], "item.id")
        require(item["id"] not in seen, "duplicate result ID")
        seen.add(item["id"])
        number(item["score"], "item.score")
        text(item["text"], "item.text")
        item_ids = strings(item["product_ids"], "item.product_ids")
        require(set(item_ids) <= set(ids), "item exceeds stage scope")
        citations = strings(item["citations"], "item.citations")
        require(len(citations) == 1 and set(citations) <= evidence_ids, "missing or invalid citation")
        source = sources[citations[0]]
        kind = {"interests": "product", "faq": "faq", "semantic": "product", "web": "web"}[expected]
        require(source["kind"] == kind and item["id"] == source["source_id"],
                "item cites the wrong source")
        require(item_ids == source["product_ids"], "item source scope mismatch")
        if expected == "interests":
            matches = interest_matches(request, products[item["id"]])
            require(bool(matches) and item["text"] == explanation(matches), "ungrounded preference explanation")
        else:
            require(item["text"] == source["text"], "answer or finding is not source-grounded")
    if expected != "faq":
        expected_ids = [pid for item in items for pid in item["product_ids"]]
        if expected == "web":
            expected_ids = list(dict.fromkeys(expected_ids))
        require(ids == expected_ids, "stage product IDs do not match its results")
    return result


class SearchIndex:
    """Small deterministic TF-IDF cosine index; ties use source IDs."""

    def __init__(self, documents):
        self.documents = documents
        counts = [collections.Counter(tokens(body)) for _, body in documents]
        frequencies = collections.Counter(word for count in counts for word in count)
        self.idf = {word: math.log((1 + len(counts)) / (1 + frequency)) + 1
                    for word, frequency in frequencies.items()}
        self.vectors = [self.vector(count) for count in counts]

    def vector(self, count):
        vector = {word: frequency * self.idf[word] for word, frequency in count.items()
                  if word in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {word: value / norm for word, value in vector.items()} if norm else {}

    def scores(self, query):
        query_vector = self.vector(collections.Counter(tokens(query)))
        return {key: min(1.0, sum(value * query_vector.get(word, 0)
                                 for word, value in vector.items()))
                for (key, _), vector in zip(self.documents, self.vectors)}


def embedding_scores(documents, query, embedder):
    try:
        vectors = embedder([body for _, body in documents] + [query])
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(vectors, list) and len(vectors) == len(documents) + 1,
            "embedding batch size mismatch")
    dimension = None
    unit_vectors = []
    for vector in vectors:
        require(isinstance(vector, list) and 1 <= len(vector) <= 4096, "invalid embedding vector")
        if dimension is None:
            dimension = len(vector)
        require(len(vector) == dimension, "embedding dimensions differ")
        for component in vector:
            number(component, "embedding component", -1e100, 1e100)
        norm = math.hypot(*vector)
        require(norm > 0, "embedding vectors must be nonzero")
        unit_vectors.append([component / norm for component in vector])
    query_vector = unit_vectors[-1]
    return {key: max(0.0, min(1.0, sum(a * b for a, b in zip(vector, query_vector))))
            for (key, _), vector in zip(documents, unit_vectors[:-1])}


def item(source, score, body=None):
    return {"id": source["source_id"], "product_ids": source["product_ids"][:],
            "score": round(score, 10), "text": source["text"] if body is None else body,
            "citations": [source["id"]]}


def envelope(stage, query, ids, items, new_evidence, previous=None):
    evidence = previous["evidence"][:] if previous else []
    known = {record["id"] for record in evidence}
    for record in new_evidence:
        if record["id"] not in known:
            evidence.append(record)
            known.add(record["id"])
    return {"stage": stage, "status": "ok" if items else "abstained",
            "reason": None if items else "No eligible source met this stage's evidence requirements.",
            "query": query, "product_ids": ids, "items": items, "evidence": evidence}


def run_interests(request):
    blocked = excluded_ids(request)
    total = sum(interest["weight"] for interest in request["profile"]["interests"])
    candidates = []
    for product in request["products"]:
        matches = interest_matches(request, product)
        if product["id"] not in blocked and matches:
            score = sum(match["weight"] for match in matches) / total
            candidates.append((score, product, matches))
    candidates.sort(key=lambda entry: (-entry[0], entry[1]["id"]))
    candidates = candidates[:request["settings"]["limit"]]
    evidence = [product_evidence(product) for _, product, _ in candidates]
    items = [item(source, score, explanation(matches))
             for (score, _, matches), source in zip(candidates, evidence)]
    tags = list(dict.fromkeys(normalized(match["tag"]) for _, _, matches in candidates for match in matches))
    query = request["profile"]["question"] + ((" " + " ".join(tags)) if tags else "")
    result = envelope("interests", query, [entry["id"] for entry in items], items, evidence)
    return validate_stage(request, result, "interests")


def ranked(documents, query, threshold, limit, embedder=None):
    scores = SearchIndex(documents).scores(query)
    if embedder is not None and documents:
        semantic = embedding_scores(documents, query, embedder)
        scores = {key: (score + semantic[key]) / 2 for key, score in scores.items()}
    return sorted(((key, score) for key, score in scores.items() if score > 0 and score >= threshold),
                  key=lambda pair: (-pair[1], pair[0]))[:limit]


def run_faq(request, previous):
    validate_stage(request, previous, "interests")
    scope = set(previous["product_ids"])
    question_words = set(tokens(request["profile"]["question"]))
    eligible = [faq for faq in request["faqs"] if scope
                and set(faq["product_ids"]) <= scope
                and question_words.intersection(tokens(faq["question"]))]
    lookup = {faq["id"]: faq for faq in eligible}
    documents = [(faq["id"], faq["question"] + " " + faq["answer"]) for faq in eligible]
    hits = ranked(documents, previous["query"], request["settings"]["faq_threshold"], 1)
    evidence = [faq_evidence(lookup[key]) for key, _ in hits]
    items = [item(source, score) for (_, score), source in zip(hits, evidence)]
    query = previous["query"] + ((" " + items[0]["text"]) if items else "")
    result = envelope("faq", query, previous["product_ids"][:], items, evidence, previous)
    return validate_stage(request, result, "faq", previous)


def run_semantic(request, previous, embedder=None):
    validate_stage(request, previous, "faq")
    scope = set(previous["product_ids"])
    products = {product["id"]: product for product in request["products"] if product["id"] in scope}
    documents = [(key, product["title"] + " " + product["description"] + " " + " ".join(product["tags"]))
                 for key, product in sorted(products.items())]
    hits = ranked(documents, previous["query"], request["settings"]["search_threshold"],
                  request["settings"]["limit"], embedder)
    evidence = [product_evidence(products[key]) for key, _ in hits]
    items = [item(source, score) for (_, score), source in zip(hits, evidence)]
    result = envelope("semantic", previous["query"], [entry["id"] for entry in items], items, evidence, previous)
    return validate_stage(request, result, "semantic", previous)


def run_web(request, previous):
    validate_stage(request, previous, "semantic")
    scope = set(previous["product_ids"])
    documents = {document["url"]: document for document in request["web"]["documents"]
                 if set(document["product_ids"]) <= scope}
    index_documents = [(url, document["title"] + " " + document["text"])
                       for url, document in sorted(documents.items())]
    hits = ranked(index_documents, previous["query"], request["settings"]["web_threshold"],
                  request["settings"]["limit"])
    evidence = [web_evidence(documents[url]) for url, _ in hits]
    items = [item(source, score) for (_, score), source in zip(hits, evidence)]
    ids = list(dict.fromkeys(pid for entry in items for pid in entry["product_ids"]))
    result = envelope("web", previous["query"], ids, items, evidence, previous)
    return validate_stage(request, result, "web", previous)


def run_pipeline(request, embedder=None):
    validate_input(request)
    stages = [run_interests(request)]
    stages.append(run_faq(request, stages[-1]))
    stages.append(run_semantic(request, stages[-1], embedder))
    stages.append(run_web(request, stages[-1]))
    return {"schema_version": 1, "fixture_label": request["fixture_label"],
            "status": "ok", "stages": stages}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        require(len(arguments) == 1, "usage: python -B implementation.py INPUT.json")
        with open(arguments[0], encoding="utf-8") as handle:
            request = json.load(handle, object_pairs_hook=reject_duplicate_keys,
                                parse_constant=lambda value: (_ for _ in ()).throw(
                                    ValidationError("nonfinite JSON number: " + value)))
        result = run_pipeline(request)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
