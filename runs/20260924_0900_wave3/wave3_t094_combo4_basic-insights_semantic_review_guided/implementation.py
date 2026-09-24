"""Deterministic, synthetic-friendly insights -> search -> review -> setup CLI.

Run: python -B implementation.py example_input.json
Library: run_pipeline(input_dict, embedder=None). An optional embedder receives
one list of query/document strings and returns one finite, equal-length numeric
vector per string. No providers, network calls, or dependencies are built in.
Review support is literal evidence coverage, never a certification or legal judgment.
"""

import copy
import json
import math
import re
import sys
from collections import Counter


VERSION = 1
MAX_INPUT_BYTES = 2_000_000
WORD = re.compile(r"[^\W_]+", re.UNICODE)
STOP = frozenset(
    "a an and are as at be been but by can for from had has have i in is it its "
    "me my of on or our please that the their them there they this to us was we "
    "were with would you your".split()
)
POSITIVE = frozenset("good great excellent helpful love easy fast clear happy".split())
NEGATIVE = frozenset("bad poor confusing slow broken missing difficult hate late unhappy".split())


class ValidationError(ValueError):
    pass


class Validation:
    """One validation layer for input, injected vectors, and every stage boundary."""

    @staticmethod
    def check(condition, path, message):
        if not condition:
            raise ValidationError(f"{path}: {message}")

    @classmethod
    def obj(cls, value, path, required, optional=()):
        cls.check(isinstance(value, dict), path, "must be an object")
        cls.check(set(required) <= value.keys(), path, "missing required fields")
        cls.check(value.keys() <= set(required) | set(optional), path, "unknown fields")

    @classmethod
    def text(cls, value, path, limit=12000):
        cls.check(isinstance(value, str) and bool(value.strip()), path, "must be nonempty text")
        cls.check(len(value) <= limit, path, f"exceeds {limit} characters")

    @classmethod
    def ident(cls, value, path):
        cls.text(value, path, 80)
        cls.check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is not None,
                  path, "invalid identifier")

    @classmethod
    def array(cls, value, path, limit=500):
        cls.check(isinstance(value, list), path, "must be an array")
        cls.check(len(value) <= limit, path, f"exceeds {limit} items")

    @classmethod
    def integer(cls, value, path, low, high):
        cls.check(type(value) is int and low <= value <= high, path,
                  f"must be an integer in [{low}, {high}]")

    @classmethod
    def number(cls, value, path, low, high):
        cls.check(type(value) in (int, float), path, "must be numeric, not boolean")
        try:
            valid = math.isfinite(value) and low <= value <= high
        except OverflowError:
            valid = False
        cls.check(valid, path, "must be finite and within range")

    @classmethod
    def refs(cls, values, allowed, path):
        cls.array(values, path)
        for value in values:
            cls.ident(value, path)
        cls.check(len(values) == len(set(values)), path, "duplicate references")
        cls.check(set(values) <= set(allowed), path, "unknown reference")

    @classmethod
    def records(cls, records, path, required, optional=()):
        cls.array(records, path)
        seen = set()
        for i, record in enumerate(records):
            label = f"{path}[{i}]"
            cls.obj(record, label, required, optional)
            cls.ident(record["id"], label + ".id")
            cls.check(record["id"] not in seen, label, "duplicate id")
            seen.add(record["id"])
        return seen

    @classmethod
    def input(cls, raw):
        cls.obj(raw, "input", ("schema_version", "synthetic", "feedback", "documents",
                              "requirements", "steps"), ("search", "completed_step_ids"))
        cls.integer(raw["schema_version"], "schema_version", VERSION, VERSION)
        cls.check(type(raw["synthetic"]) is bool, "synthetic", "must be boolean")
        cls.records(raw["feedback"], "feedback", ("id", "text"))
        for f in raw["feedback"]:
            cls.text(f["text"], "feedback.text")
        cls.records(raw["documents"], "documents", ("id", "title", "text"))
        for d in raw["documents"]:
            cls.text(d["title"], "document.title", 200)
            cls.text(d["text"], "document.text")
        requirement_ids = cls.records(
            raw["requirements"], "requirements",
            ("id", "description", "required_terms"), ("min_evidence",))
        for r in raw["requirements"]:
            cls.text(r["description"], "requirement.description", 1000)
            cls.array(r["required_terms"], "requirement.required_terms", 30)
            cls.check(bool(r["required_terms"]), "requirement.required_terms", "must not be empty")
            normalized = []
            for term in r["required_terms"]:
                cls.text(term, "requirement.term", 80)
                cls.check(WORD.fullmatch(term) is not None, "requirement.term",
                          "must be one word; phrases are not supported")
                normalized.append(term.casefold())
            cls.check(len(set(normalized)) == len(normalized), "requirement.required_terms",
                      "duplicate normalized terms")
            cls.integer(r.get("min_evidence", 1), "requirement.min_evidence", 1, 500)
        step_ids = cls.records(raw["steps"], "steps", ("id", "title", "prerequisites",
                                                     "requirement_ids"))
        for s in raw["steps"]:
            cls.text(s["title"], "step.title", 200)
            cls.refs(s["prerequisites"], step_ids, "step.prerequisites")
            cls.refs(s["requirement_ids"], requirement_ids, "step.requirement_ids")
        order = topological_steps(raw["steps"])
        cls.check(len(order) == len(raw["steps"]), "steps", "prerequisite cycle")
        cls.refs(raw.get("completed_step_ids", []), step_ids, "completed_step_ids")
        search = raw.get("search", {})
        cls.obj(search, "search", (), ("max_themes", "top_k", "min_score", "embedding_weight"))
        cls.integer(search.get("max_themes", 8), "search.max_themes", 1, 30)
        cls.integer(search.get("top_k", 5), "search.top_k", 1, 50)
        cls.number(search.get("min_score", 0.05), "search.min_score", 0, 1)
        cls.number(search.get("embedding_weight", 0.35), "search.embedding_weight", 0, 1)
        return copy.deepcopy(raw)

    @classmethod
    def vectors(cls, values, count):
        cls.array(values, "embedding", 530)
        cls.check(len(values) == count, "embedding", "wrong vector count")
        dimensions = None
        result = []
        for vector in values:
            cls.array(vector, "embedding.vector", 4096)
            cls.check(bool(vector), "embedding.vector", "empty vector")
            if dimensions is None:
                dimensions = len(vector)
            cls.check(len(vector) == dimensions, "embedding.vector", "inconsistent dimensions")
            for number in vector:
                cls.number(number, "embedding.coordinate", -1e100, 1e100)
            scale = max(abs(v) for v in vector)
            cls.check(scale > 0, "embedding.vector", "zero vector")
            scaled = [v / scale for v in vector]
            norm = math.sqrt(sum(v * v for v in scaled))
            result.append([v / norm for v in scaled])
        return result

    @classmethod
    def stage(cls, value, name, source, previous=None):
        cls.obj(value, name, ("schema_version", "stage", "items", "summary"))
        cls.integer(value["schema_version"], name + ".schema_version", VERSION, VERSION)
        cls.check(value["stage"] == name, name, "incorrect stage")
        cls.array(value["items"], name + ".items")
        cls.check(isinstance(value["summary"], dict), name, "summary must be an object")
        items = value["items"]
        if name == "insights":
            cls.records(items, name, ("id", "keyword", "feedback_ids", "count", "sentiment",
                                     "action", "query"))
            feedback = {f["id"]: f for f in source["feedback"]}
            keywords = []
            for item in items:
                cls.text(item["keyword"], name + ".keyword", 12000)
                cls.refs(item["feedback_ids"], feedback, name + ".feedback_ids")
                cls.check(bool(item["feedback_ids"]), name, "theme needs feedback")
                cls.check(item["count"] == len(item["feedback_ids"]), name, "incorrect count")
                cls.check(all(item["keyword"] in tokens(feedback[f]["text"])
                              for f in item["feedback_ids"]), name, "untraceable theme")
                cls.check(item["sentiment"] in ("positive", "negative", "mixed", "neutral"),
                          name, "invalid sentiment")
                cls.text(item["action"], name + ".action")
                cls.check(item["query"] == item["keyword"], name, "query must match theme")
                keywords.append(item["keyword"])
            cls.check(len(keywords) == len(set(keywords)), name, "duplicate keyword")
        elif name == "semantic":
            cls.check(previous is not None and previous["stage"] == "insights",
                      name, "requires validated insights")
            cls.records(items, name, ("id", "theme_id", "feedback_ids", "query", "hits"))
            themes = {x["id"]: x for x in previous["items"]}
            docs = {d["id"]: d for d in source["documents"]}
            cls.check(len(items) == len(themes) and
                      {x["theme_id"] for x in items} == set(themes), name, "theme coverage mismatch")
            for item in items:
                theme = themes[item["theme_id"]]
                cls.check(item["feedback_ids"] == theme["feedback_ids"] and
                          item["query"] == theme["query"], name, "broken feedback handoff")
                cls.array(item["hits"], name + ".hits", 50)
                seen = set()
                for hit in item["hits"]:
                    cls.obj(hit, name + ".hit", ("document_id", "title", "text", "score",
                                                "lexical_score", "embedding_score"))
                    cls.ident(hit["document_id"], name + ".document_id")
                    cls.check(hit["document_id"] in docs and hit["document_id"] not in seen,
                              name, "invalid or duplicate document")
                    seen.add(hit["document_id"])
                    doc = docs[hit["document_id"]]
                    cls.check(hit["text"] == doc["text"] and hit["title"] == doc["title"],
                              name, "document provenance mismatch")
                    for key in ("score", "lexical_score", "embedding_score"):
                        cls.number(hit[key], name + "." + key, 0, 1)
                cls.check(item["hits"] == sorted(item["hits"],
                          key=lambda h: (-h["score"], h["document_id"])),
                          name, "hits must be ranked")
        elif name == "review":
            cls.check(previous is not None and previous["stage"] == "semantic",
                      name, "requires validated semantic output")
            cls.records(items, name, ("id", "status", "required_terms", "min_evidence",
                                     "evidence", "gap"))
            requirements = {r["id"]: r for r in source["requirements"]}
            cls.check({x["id"] for x in items} == set(requirements), name, "requirement mismatch")
            hits, routes = retrieved(previous)
            for item in items:
                r = requirements[item["id"]]
                terms = [t.casefold() for t in r["required_terms"]]
                minimum = r.get("min_evidence", 1)
                cls.check(item["required_terms"] == terms and item["min_evidence"] == minimum,
                          name, "requirement changed")
                cls.array(item["evidence"], name + ".evidence")
                seen = set()
                for e in item["evidence"]:
                    cls.obj(e, name + ".evidence", ("document_id", "query_ids", "start", "end",
                                                  "quote", "matched_terms"))
                    cls.ident(e["document_id"], name + ".evidence.document_id")
                    cls.check(e["document_id"] in hits and e["document_id"] not in seen,
                              name, "evidence must be uniquely retrieved")
                    seen.add(e["document_id"])
                    text = hits[e["document_id"]]["text"]
                    cls.integer(e["start"], name + ".start", 0, len(text))
                    cls.integer(e["end"], name + ".end", e["start"] + 1, len(text))
                    cls.check(e["quote"] == text[e["start"]:e["end"]], name, "quote mismatch")
                    cls.check(set(terms) <= set(tokens(e["quote"])) and
                              e["matched_terms"] == terms, name, "unsupported evidence terms")
                    cls.check(e["query_ids"] == routes[e["document_id"]], name, "broken search trace")
                supported = len(item["evidence"]) >= minimum
                cls.check(item["status"] == ("supported" if supported else "gap"),
                          name, "incorrect evidence status")
                cls.check(item["gap"] == (None if supported else gap_record(
                    terms, minimum, hits, item["evidence"])), name, "incorrect gap trace")
        elif name == "guided":
            cls.check(previous is not None and previous["stage"] == "review",
                      name, "requires validated review output")
            cls.records(items, name, ("id", "title", "status", "prerequisites", "requirement_ids",
                                     "blockers"))
            steps = {s["id"]: s for s in source["steps"]}
            cls.check({x["id"] for x in items} == set(steps), name, "step coverage mismatch")
            complete = set(source.get("completed_step_ids", []))
            reviews = {r["id"]: r for r in previous["items"]}
            for item in items:
                step = steps[item["id"]]
                cls.check(all(item[k] == step[k] for k in
                              ("title", "prerequisites", "requirement_ids")), name, "step changed")
                blockers = step_blockers(step, complete, reviews)
                cls.check(item["blockers"] == blockers, name, "incorrect blockers")
                cls.check(not (item["id"] in complete and blockers),
                          "completed_step_ids", f"step {item['id']} has unmet prerequisites or evidence")
                status = "completed" if item["id"] in complete else ("blocked" if blockers else "ready")
                cls.check(item["status"] == status, name, "incorrect progress state")
            expected = progress(items)
            cls.check(value["summary"] == expected, name, "incorrect progress summary")
        else:
            raise ValidationError(f"unknown stage: {name}")
        return value


def tokens(text):
    return [m.group().casefold() for m in WORD.finditer(text)]


def topological_steps(steps):
    pending = {s["id"]: s for s in steps}
    done, order = set(), []
    while pending:
        ready = sorted(k for k, s in pending.items() if set(s["prerequisites"]) <= done)
        if not ready:
            break
        for key in ready:
            order.append(pending.pop(key))
            done.add(key)
    return order


def envelope(stage, items, summary):
    return {"schema_version": VERSION, "stage": stage, "items": items, "summary": summary}


def insights(source):
    groups = {}
    by_id = {f["id"]: f for f in source["feedback"]}
    ignored = []
    for feedback in source["feedback"]:
        words = set(tokens(feedback["text"])) - STOP - POSITIVE - NEGATIVE
        if not words:
            ignored.append(feedback["id"])
        for word in words:
            groups.setdefault(word, []).append(feedback["id"])
    selected = sorted(groups, key=lambda w: (-len(groups[w]), w))[
        :source.get("search", {}).get("max_themes", 8)]
    items = []
    for i, word in enumerate(selected):
        feedback_ids = sorted(groups[word])
        sentiments = set()
        for fid in feedback_ids:
            words = set(tokens(by_id[fid]["text"]))
            if words & POSITIVE:
                sentiments.add("positive")
            if words & NEGATIVE:
                sentiments.add("negative")
        sentiment = "mixed" if len(sentiments) == 2 else next(iter(sentiments), "neutral")
        items.append({"id": f"theme-{i + 1}", "keyword": word, "feedback_ids": feedback_ids,
                      "count": len(feedback_ids), "sentiment": sentiment, "query": word,
                      "action": f"Investigate '{word}' using {len(feedback_ids)} feedback item(s)."})
    return Validation.stage(envelope("insights", items, {
        "feedback_count": len(source["feedback"]), "theme_count": len(items),
        "ignored_feedback_ids": sorted(ignored), "omitted_keyword_count": len(groups) - len(items),
        "method": "frequency themes; heuristic sentiment, not an LLM judgment",
    }), "insights", source)


def cosine(left, right):
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def semantic(source, previous, embedder=None):
    Validation.stage(previous, "insights", source)
    documents = sorted(source["documents"], key=lambda d: d["id"])
    counts = [Counter(tokens(d["title"] + " " + d["text"])) for d in documents]
    df = Counter(word for count in counts for word in count)
    idf = {w: math.log((1 + len(documents)) / (1 + n)) + 1 for w, n in df.items()}
    index = {word: [d["id"] for d, count in zip(documents, counts) if word in count]
             for word in sorted(df)}
    doc_vectors = [{w: (1 + math.log(n)) * idf[w] for w, n in count.items()} for count in counts]
    doc_norms = [math.sqrt(sum(v * v for v in vector.values())) for vector in doc_vectors]
    queries = previous["items"]
    vectors = None
    if embedder is not None and queries and documents:
        Validation.check(callable(embedder), "embedder", "must be callable")
        texts = [q["query"] for q in queries] + [d["title"] + "\n" + d["text"] for d in documents]
        try:
            values = embedder(list(texts))
        except Exception as exc:
            raise ValidationError("embedder: injected callable failed") from exc
        vectors = Validation.vectors(values, len(texts))
    elif embedder is not None:
        Validation.check(callable(embedder), "embedder", "must be callable")
    config = source.get("search", {})
    weight = config.get("embedding_weight", 0.35) if vectors is not None else 0
    items = []
    for qi, query in enumerate(queries):
        qcounts = Counter(tokens(query["query"]))
        qvector = {w: (1 + math.log(n)) * idf.get(w, math.log(1 + len(documents)) + 1)
                   for w, n in qcounts.items()}
        qnorm = math.sqrt(sum(v * v for v in qvector.values()))
        hits = []
        for di, doc in enumerate(documents):
            denom = qnorm * doc_norms[di]
            lexical = (sum(v * doc_vectors[di].get(w, 0) for w, v in qvector.items()) / denom
                       if denom else 0)
            lexical = min(1.0, max(0.0, lexical))
            embedded = cosine(vectors[qi], vectors[len(queries) + di]) if vectors is not None else 0
            score = round((1 - weight) * lexical + weight * embedded, 10)
            if score > 0 and score >= config.get("min_score", 0.05):
                hits.append({"document_id": doc["id"], "title": doc["title"], "text": doc["text"],
                             "score": score, "lexical_score": round(lexical, 10),
                             "embedding_score": round(embedded, 10)})
        hits.sort(key=lambda h: (-h["score"], h["document_id"]))
        items.append({"id": f"query-{qi + 1}", "theme_id": query["id"],
                      "feedback_ids": list(query["feedback_ids"]), "query": query["query"],
                      "hits": hits[:config.get("top_k", 5)]})
    return Validation.stage(envelope("semantic", items, {
        "document_count": len(documents), "index": index,
        "ranking": "tf-idf cosine" if vectors is None else "weighted tf-idf and injected cosine",
        "embedding_used": vectors is not None,
    }), "semantic", source, previous)


def retrieved(previous):
    hits, routes = {}, {}
    for query in previous["items"]:
        for hit in query["hits"]:
            did = hit["document_id"]
            hits[did] = hit
            routes.setdefault(did, []).append(query["id"])
    return hits, routes


def gap_record(terms, minimum, hits, evidence):
    candidates = [{"document_id": did, "missing_terms": [
        t for t in terms if t not in set(tokens(hit["text"]))]}
        for did, hit in sorted(hits.items())]
    return {"reason": "no_retrieved_documents" if not hits else "insufficient_evidence",
            "needed_count": max(0, minimum - len(evidence)),
            "candidate_checks": candidates}


def review(source, previous, insight_output):
    Validation.stage(previous, "semantic", source, insight_output)
    hits, routes = retrieved(previous)
    items = []
    for requirement in source["requirements"]:
        terms = [t.casefold() for t in requirement["required_terms"]]
        evidence = []
        for did, hit in sorted(hits.items()):
            positions = {}
            for match in WORD.finditer(hit["text"]):
                positions.setdefault(match.group().casefold(), (match.start(), match.end()))
            if not set(terms) <= positions.keys():
                continue
            start = min(positions[t][0] for t in terms)
            end = max(positions[t][1] for t in terms)
            evidence.append({"document_id": did, "query_ids": routes[did], "start": start,
                             "end": end, "quote": hit["text"][start:end], "matched_terms": terms})
        minimum = requirement.get("min_evidence", 1)
        supported = len(evidence) >= minimum
        items.append({"id": requirement["id"], "status": "supported" if supported else "gap",
                      "required_terms": terms, "min_evidence": minimum, "evidence": evidence,
                      "gap": None if supported else gap_record(terms, minimum, hits, evidence)})
    return Validation.stage(envelope("review", items, {
        "supported": sum(r["status"] == "supported" for r in items),
        "gaps": sum(r["status"] == "gap" for r in items),
        "notice": "Literal term coverage in retrieved documents only; no certification, "
                  "truth, entailment, or compliance claim. Negation requires human review.",
    }), "review", source, previous)


def step_blockers(step, complete, reviews):
    return ([{"kind": "prerequisite", "id": key} for key in step["prerequisites"]
             if key not in complete] +
            [{"kind": "requirement_gap", "id": key} for key in step["requirement_ids"]
             if reviews[key]["status"] != "supported"])


def progress(items):
    total = len(items)
    counts = Counter(s["status"] for s in items)
    return {"total": total, "completed": counts["completed"], "ready": counts["ready"],
            "blocked": counts["blocked"],
            "percent_complete": round(100 * counts["completed"] / total, 2) if total else 100.0}


def guided(source, previous, search_output):
    Validation.stage(previous, "review", source, search_output)
    reviews = {r["id"]: r for r in previous["items"]}
    complete = set(source.get("completed_step_ids", []))
    items = []
    for step in topological_steps(source["steps"]):
        blockers = step_blockers(step, complete, reviews)
        status = "completed" if step["id"] in complete else ("blocked" if blockers else "ready")
        items.append(dict(step, status=status, blockers=blockers))
    return Validation.stage(envelope("guided", items, progress(items)), "guided", source, previous)


def run_pipeline(raw, embedder=None):
    source = Validation.input(raw)
    a = insights(source)
    b = semantic(source, a, embedder)
    c = review(source, b, a)
    d = guided(source, c, b)
    return {"schema_version": VERSION, "status": "ok", "synthetic": source["synthetic"],
            "stages": {"insights": a, "semantic": b, "review": c, "guided": d}}


def reject_constant(value):
    raise ValidationError(f"JSON: non-finite constant {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        Validation.check(key not in result, "JSON", f"duplicate key {key}")
        result[key] = value
    return result


def main(argv=None):
    try:
        args = sys.argv[1:] if argv is None else argv
        Validation.check(len(args) == 1, "CLI", "usage: python -B implementation.py INPUT.json")
        with open(args[0], "rb") as handle:
            data = handle.read(MAX_INPUT_BYTES + 1)
        Validation.check(len(data) <= MAX_INPUT_BYTES, "input", "file exceeds 2000000 bytes")
        raw = json.loads(data.decode("utf-8"), parse_constant=reject_constant,
                         object_pairs_hook=unique_object)
        result = run_pipeline(raw)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(exc)},
                         ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
