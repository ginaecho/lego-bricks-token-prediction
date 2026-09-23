"""Deterministic, standard-library-only research-to-onboarding reference CLI.

Run: python -B implementation.py example_input.json
Embedding injection: run_pipeline(data, embedder=callable). The callable receives
one batch [query, product_text, ...] and returns equally sized finite nonzero
vectors. No providers, downloads, persistent index, or network calls are used.
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


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def enum(*values):
    return ("enum", values)


def nullable(spec):
    return ("nullable", spec)


STR = "string"
NUM = "number"
LEVEL = enum("beginner", "intermediate", "advanced")
PRIORITY = enum("normal", "high", "urgent")
STANCE = enum("support", "oppose", "uncertain")
ROUTE = {
    "category": STR, "priority": PRIORITY, "team": STR, "owner": STR,
    "onboarding_track": STR,
}
EVIDENCE = {
    "evidence_id": STR, "document_id": STR, "source": STR, "excerpt": STR,
    "stance": STANCE, "product_ids": [STR],
}
INPUT_SCHEMA = {
    "schema_version": enum("1.0"),
    "fixture_label": STR,
    "questions": [{"id": STR, "question": STR, "proposition": STR}],
    "documents": [{
        "id": STR, "title": STR, "source": STR,
        "evidence": [{
            "question_id": STR, "stance": STANCE, "excerpt": STR,
            "product_ids": [STR],
        }],
    }],
    "products": [{"id": STR, "name": STR, "description": STR, "tags": [STR]}],
    "search": {
        "query": STR, "top_k": "integer", "embedding_weight": NUM,
        "synonyms": [{"term": STR, "equivalents": [STR]}],
    },
    "ticket": {"id": STR, "subject": STR, "body": STR},
    "routing": {
        "rules": [{
            "id": STR, "keywords": [STR], **ROUTE,
        }],
        "fallback": ROUTE,
        "urgent_keywords": [STR],
    },
    "profile": {
        "experience": LEVEL, "preferred_formats": [STR],
        "completed_step_ids": [STR],
    },
    "onboarding": {
        "review_step_id": STR,
        "steps": [{
            "id": STR, "title": STR, "tracks": [STR], "levels": [LEVEL],
            "formats": [STR], "prerequisites": [STR], "product_ids": [STR],
        }],
    },
}
OUTPUT_SCHEMAS = {
    "research": {
        "findings": [{
            "question_id": STR, "question": STR, "proposition": STR,
            "status": enum("supported", "disputed", "contested", "uncertain", "unanswered"),
            "source_count": "integer", "evidence": [EVIDENCE],
            "summary": STR,
        }],
        "unresolved_question_ids": [STR],
    },
    "semantic": {
        "query": STR, "embedding_used": "boolean",
        "index": [{"product_id": STR, "term_count": "integer", "evidence_ids": [STR]}],
        "results": [{
            "product_id": STR, "name": STR, "score": NUM,
            "lexical_score": NUM, "embedding_score": nullable(NUM),
            "evidence_ids": [STR], "attention_question_ids": [STR],
        }],
    },
    "triage": {
        "ticket_id": STR, "rule_id": nullable(STR), **ROUTE,
        "selected_product_id": nullable(STR), "evidence_ids": [STR],
        "attention_question_ids": [STR], "reasons": [STR],
    },
    "adaptive": {
        "experience": LEVEL, "track": STR, "accountable_owner": STR,
        "selected_product_id": nullable(STR), "needs_human_review": "boolean",
        "attention_question_ids": [STR], "explanations": [STR],
        "steps": [{
            "step_id": STR, "title": STR, "format": STR,
            "prerequisites": [STR], "reason": STR,
        }],
    },
}


def check_shape(value, spec, path):
    """One strict recursive schema checker for inputs and every stage output."""
    if isinstance(spec, dict):
        require(isinstance(value, dict), f"{path}: expected object")
        require(set(value) == set(spec), f"{path}: missing or unknown fields")
        for key, child in spec.items():
            check_shape(value[key], child, f"{path}.{key}")
    elif isinstance(spec, list):
        require(isinstance(value, list), f"{path}: expected array")
        for index, item in enumerate(value):
            check_shape(item, spec[0], f"{path}[{index}]")
    elif isinstance(spec, tuple):
        if spec[0] == "nullable":
            if value is not None:
                check_shape(value, spec[1], path)
        else:
            require(isinstance(value, str) and value in spec[1],
                    f"{path}: expected one of {spec[1]}")
    elif spec == STR:
        require(isinstance(value, str) and bool(value.strip()),
                f"{path}: expected nonblank string")
    elif spec == NUM:
        require(type(value) in (int, float), f"{path}: expected finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        require(finite, f"{path}: expected finite number")
    elif spec == "integer":
        require(type(value) is int, f"{path}: expected integer")
    elif spec == "boolean":
        require(type(value) is bool, f"{path}: expected boolean")


def unique(values, path):
    require(len(values) == len(set(values)), f"{path}: duplicate values")


def tokens(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def evidence_status(evidence):
    stances = {item["stance"] for item in evidence}
    if {"support", "oppose"} <= stances:
        return "disputed"
    if "oppose" in stances:
        return "contested"
    if "uncertain" in stances:
        return "uncertain"
    return "supported" if "support" in stances else "unanswered"


def expected_evidence(data):
    by_question = {q["id"]: [] for q in data["questions"]}
    for doc in data["documents"]:
        for index, entry in enumerate(doc["evidence"], 1):
            by_question[entry["question_id"]].append({
                "evidence_id": f"{doc['id']}:{index}",
                "document_id": doc["id"], "source": doc["source"],
                "excerpt": entry["excerpt"], "stance": entry["stance"],
                "product_ids": list(entry["product_ids"]),
            })
    return by_question


def validate(kind, value, data=None, previous=None):
    """Shared shape, referential-integrity, and handoff validation boundary."""
    check_shape(value, INPUT_SCHEMA if kind == "input" else OUTPUT_SCHEMAS[kind], kind)
    if kind == "input":
        data = value
        collections = {
            "questions": data["questions"], "documents": data["documents"],
            "products": data["products"], "rules": data["routing"]["rules"],
            "steps": data["onboarding"]["steps"],
        }
        ids = {}
        for name, entries in collections.items():
            ids[name] = {item["id"] for item in entries}
            unique([item["id"] for item in entries], name)
        for doc in data["documents"]:
            for entry in doc["evidence"]:
                require(entry["question_id"] in ids["questions"], "unknown evidence question")
                require(set(entry["product_ids"]) <= ids["products"], "unknown evidence product")
                unique(entry["product_ids"], "evidence.product_ids")
        search = data["search"]
        require(1 <= search["top_k"] <= 100, "search.top_k must be in 1..100")
        require(0 <= search["embedding_weight"] <= 1, "embedding_weight must be in 0..1")
        require(bool(tokens(search["query"])), "search.query needs a searchable token")
        synonym_keys = []
        for item in search["synonyms"]:
            words = tokens(item["term"])
            require(len(words) == 1, "synonym term must be a single token")
            synonym_keys.append(words[0])
            require(bool(item["equivalents"]), "synonym equivalents must not be empty")
            for word in item["equivalents"]:
                require(len(tokens(word)) == 1, "synonym equivalent must be a single token")
        unique(synonym_keys, "search.synonyms")
        for rule in data["routing"]["rules"]:
            require(bool(rule["keywords"]), "routing rule requires keywords")
        for phrase in data["routing"]["urgent_keywords"] + [
            word for rule in data["routing"]["rules"] for word in rule["keywords"]
        ]:
            require(bool(tokens(phrase)), "routing keyword needs a word")
        steps = {step["id"]: step for step in data["onboarding"]["steps"]}
        require(data["onboarding"]["review_step_id"] in steps, "unknown review step")
        tracks = set()
        for step in steps.values():
            for key in ("tracks", "levels", "formats"):
                require(bool(step[key]), f"step.{key} must not be empty")
                unique(step[key], f"step.{key}")
            unique(step["prerequisites"], "step.prerequisites")
            unique(step["product_ids"], "step.product_ids")
            require(set(step["prerequisites"]) <= ids["steps"], "unknown prerequisite")
            require(set(step["product_ids"]) <= ids["products"], "unknown step product")
            tracks.update(step["tracks"])
        for route in data["routing"]["rules"] + [data["routing"]["fallback"]]:
            require(route["onboarding_track"] in tracks, "unknown onboarding track")
        completed = data["profile"]["completed_step_ids"]
        unique(completed, "completed_step_ids")
        require(set(completed) <= ids["steps"], "unknown completed step")
        unique(data["profile"]["preferred_formats"], "preferred_formats")
        # Iterative topological traversal avoids recursion limits on deep prerequisites.
        resolved = set()
        while len(resolved) < len(steps):
            ready = {sid for sid, step in steps.items()
                     if sid not in resolved and set(step["prerequisites"]) <= resolved}
            require(bool(ready), "onboarding prerequisites contain a cycle")
            resolved.update(ready)
        return value

    require(data is not None, f"{kind}: missing validated input context")
    if kind == "research":
        expected = expected_evidence(data)
        require([f["question_id"] for f in value["findings"]] ==
                [q["id"] for q in data["questions"]], "research question coverage/order changed")
        for finding, question in zip(value["findings"], data["questions"]):
            evidence = expected[question["id"]]
            require(finding["evidence"] == evidence, "research evidence provenance changed")
            require(finding["question"] == question["question"] and
                    finding["proposition"] == question["proposition"], "research question changed")
            require(finding["status"] == evidence_status(evidence), "research status inconsistent")
            require(finding["source_count"] == len({e["source"] for e in evidence}),
                    "research source count inconsistent")
        require(value["unresolved_question_ids"] ==
                [f["question_id"] for f in value["findings"] if f["status"] != "supported"],
                "research unresolved questions inconsistent")
    elif kind == "semantic":
        require(previous is not None, "semantic: missing research handoff")
        products = {p["id"]: p for p in data["products"]}
        require([item["product_id"] for item in value["index"]] == list(products),
                "semantic index coverage/order changed")
        for item in value["index"]:
            expected_ids, _ = product_evidence(previous, item["product_id"])
            require(item["evidence_ids"] == expected_ids, "index evidence provenance changed")
            require(item["term_count"] > 0, "index must contain searchable terms")
        unique([item["product_id"] for item in value["results"]], "semantic results")
        require(value["query"] == data["search"]["query"], "semantic query changed")
        require(value["embedding_used"] == (data["search"]["embedding_weight"] > 0),
                "semantic embedding mode inconsistent")
        require(len(value["results"]) <= data["search"]["top_k"], "too many search results")
        require(value["results"] == sorted(value["results"],
                key=lambda item: (-item["score"], item["product_id"])), "results not ranked")
        for result in value["results"]:
            pid = result["product_id"]
            require(pid in products, "search returned unknown product")
            require(result["name"] == products[pid]["name"], "search product name changed")
            evidence, attention = product_evidence(previous, pid)
            require(result["evidence_ids"] == evidence and
                    result["attention_question_ids"] == attention, "search provenance changed")
            require(0 < result["score"] <= 1 and 0 <= result["lexical_score"] <= 1,
                    "search scores out of range")
            embedding = result["embedding_score"]
            require((embedding is not None) == value["embedding_used"],
                    "embedding score presence inconsistent")
            if embedding is not None:
                require(0 <= embedding <= 1, "embedding score out of range")
            weight = data["search"]["embedding_weight"]
            combined = (1 - weight) * result["lexical_score"] + weight * (embedding or 0)
            require(abs(result["score"] - combined) <= 0.000002, "combined score inconsistent")
    elif kind == "triage":
        require(previous is not None, "triage: missing semantic handoff")
        selected = previous["results"][0] if previous["results"] else None
        require(value["ticket_id"] == data["ticket"]["id"], "ticket id changed")
        require(value["selected_product_id"] == (selected["product_id"] if selected else None),
                "triage product handoff changed")
        for key in ("evidence_ids", "attention_question_ids"):
            require(value[key] == (selected[key] if selected else []), f"triage {key} changed")
        route, rule_id, _ = choose_route(data)
        require(value["rule_id"] == rule_id, "triage chose incorrect rule")
        for key in ("category", "team", "owner", "onboarding_track"):
            require(value[key] == route[key], f"triage {key} changed")
        priority, _ = choose_priority(data, route, value["attention_question_ids"])
        require(value["priority"] == priority, "triage priority inconsistent")
        require(bool(value["reasons"]), "triage requires explanations")
    elif kind == "adaptive":
        require(previous is not None, "adaptive: missing triage handoff")
        require(value["track"] == previous["onboarding_track"] and
                value["accountable_owner"] == previous["owner"] and
                value["selected_product_id"] == previous["selected_product_id"],
                "onboarding routing handoff changed")
        require(value["experience"] == data["profile"]["experience"], "experience changed")
        require(value["attention_question_ids"] == previous["attention_question_ids"],
                "onboarding attention handoff changed")
        require(value["needs_human_review"] == bool(previous["attention_question_ids"]),
                "onboarding review flag inconsistent")
        unique([s["step_id"] for s in value["steps"]], "onboarding output steps")
        expected_ids, _ = plan_steps(data, previous)
        require([s["step_id"] for s in value["steps"]] == expected_ids,
                "onboarding step selection/order changed")
        catalog = {s["id"]: s for s in data["onboarding"]["steps"]}
        satisfied = set(data["profile"]["completed_step_ids"])
        for item in value["steps"]:
            step = catalog[item["step_id"]]
            require(set(step["prerequisites"]) <= satisfied, "prerequisite order violated")
            require(item["prerequisites"] == step["prerequisites"], "prerequisites changed")
            require(item["format"] == select_format(data, step)[0], "format selection changed")
            require(item["title"] == step["title"], "onboarding title changed")
            satisfied.add(item["step_id"])
        require(bool(value["explanations"]), "onboarding requires explanations")
    return value


def research(data):
    grouped = expected_evidence(data)
    findings = []
    for question in data["questions"]:
        evidence = grouped[question["id"]]
        status = evidence_status(evidence)
        source_count = len({item["source"] for item in evidence})
        counts = Counter(item["stance"] for item in evidence)
        summary = (
            f"{status}: {counts['support']} supporting, {counts['oppose']} opposing, "
            f"{counts['uncertain']} uncertain excerpts across {source_count} distinct sources. "
            "Counts describe supplied evidence, not a probability or truth verdict."
        )
        findings.append({
            "question_id": question["id"], "question": question["question"],
            "proposition": question["proposition"], "status": status,
            "source_count": source_count, "evidence": evidence, "summary": summary,
        })
    return {
        "findings": findings,
        "unresolved_question_ids": [f["question_id"] for f in findings
                                    if f["status"] != "supported"],
    }


def product_evidence(research_output, product_id):
    evidence_ids, attention = [], []
    for finding in research_output["findings"]:
        matches = [e["evidence_id"] for e in finding["evidence"]
                   if product_id in e["product_ids"]]
        evidence_ids.extend(matches)
        if matches and finding["question_id"] in research_output["unresolved_question_ids"]:
            attention.append(finding["question_id"])
    return evidence_ids, attention


def unit_vector(vector):
    scale = max(abs(component) for component in vector)
    require(scale > 0, "embedding vectors must be nonzero")
    scaled = [component / scale for component in vector]
    magnitude = math.sqrt(sum(component * component for component in scaled))
    return [component / magnitude for component in scaled]


def embedding_scores(embedder, texts):
    require(callable(embedder), "positive embedding_weight requires an injected embedder")
    try:
        vectors = embedder(list(texts))
    except Exception as exc:
        raise ValidationError("injected embedder failed") from exc
    require(isinstance(vectors, list) and len(vectors) == len(texts),
            "embedder must return one vector per input text")
    dimensions = None
    normalized = []
    for index, vector in enumerate(vectors):
        check_shape(vector, [NUM], f"embedding[{index}]")
        require(bool(vector), "embedding vectors must not be empty")
        dimensions = dimensions if dimensions is not None else len(vector)
        require(len(vector) == dimensions, "embedding dimensions must match")
        normalized.append(unit_vector(vector))
    query = normalized[0]
    return [max(0.0, min(1.0, math.fsum(a * b for a, b in zip(query, vector))))
            for vector in normalized[1:]]


def semantic(data, previous, embedder=None):
    products = data["products"]
    synonyms = {tokens(item["term"])[0]: [tokens(w)[0] for w in item["equivalents"]]
                for item in data["search"]["synonyms"]}

    def terms(text):
        base = tokens(text)
        return Counter(base + [alias for word in base for alias in synonyms.get(word, [])])

    texts, index = [], []
    for product in products:
        sections = [product["name"], product["description"], " ".join(product["tags"])]
        for finding in previous["findings"]:
            matches = [e for e in finding["evidence"] if product["id"] in e["product_ids"]]
            if matches:
                sections.extend([finding["question"], finding["proposition"]])
                sections.extend(e["excerpt"] for e in matches)
        text = " ".join(sections)
        texts.append(text)
        evidence, _ = product_evidence(previous, product["id"])
        index.append({"product_id": product["id"], "term_count": len(terms(text)),
                      "evidence_ids": evidence})
    bags = [terms(text) for text in texts]
    query = terms(data["search"]["query"])
    document_frequency = Counter(word for bag in bags for word in bag)
    vocabulary = set(query) | set(document_frequency)
    idf = {word: math.log((1 + len(bags)) / (1 + document_frequency[word])) + 1
           for word in vocabulary}

    def weighted(bag):
        return {word: (1 + math.log(count)) * idf[word] for word, count in bag.items()}

    query_weights = weighted(query)
    query_norm = math.sqrt(sum(v * v for v in query_weights.values()))
    weight = data["search"]["embedding_weight"]
    embedded = embedding_scores(embedder, [data["search"]["query"]] + texts) if weight else None
    results = []
    for position, (product, bag) in enumerate(zip(products, bags)):
        weights = weighted(bag)
        norm = math.sqrt(sum(v * v for v in weights.values()))
        dot = sum(value * weights.get(word, 0) for word, value in query_weights.items())
        lexical = max(0.0, min(1.0, dot / (query_norm * norm))) if norm else 0.0
        embed_score = embedded[position] if embedded is not None else None
        score = round((1 - weight) * lexical + weight * (embed_score or 0), 6)
        if score <= 0:
            continue
        evidence, attention = product_evidence(previous, product["id"])
        results.append({
            "product_id": product["id"], "name": product["name"], "score": score,
            "lexical_score": round(lexical, 6),
            "embedding_score": round(embed_score, 6) if embed_score is not None else None,
            "evidence_ids": evidence, "attention_question_ids": attention,
        })
    results.sort(key=lambda result: (-result["score"], result["product_id"]))
    return {"query": data["search"]["query"], "embedding_used": weight > 0,
            "index": index, "results": results[:data["search"]["top_k"]]}


def contains_phrase(text_tokens, phrase):
    words = tokens(phrase)
    return any(text_tokens[i:i + len(words)] == words
               for i in range(len(text_tokens) - len(words) + 1))


def choose_route(data):
    ticket = data["ticket"]
    words = tokens(ticket["subject"] + " " + ticket["body"])
    winner, matches = None, []
    for rule in data["routing"]["rules"]:
        current = [phrase for phrase in rule["keywords"] if contains_phrase(words, phrase)]
        if len(current) > len(matches):
            winner, matches = rule, current
    if winner is None:
        return data["routing"]["fallback"], None, ["No category keyword matched; fallback route."]
    return winner, winner["id"], [
        f"Rule {winner['id']} matched: {', '.join(matches)}. "
        "Most matching configured keywords wins; ties use configuration order."
    ]


def choose_priority(data, route, attention):
    priority = route["priority"]
    reasons = [f"Configured route priority: {priority}."]
    words = tokens(data["ticket"]["subject"] + " " + data["ticket"]["body"])
    urgent = [p for p in data["routing"]["urgent_keywords"] if contains_phrase(words, p)]
    if attention:
        priority = "high" if priority == "normal" else priority
        reasons.append("Selected product has unresolved evidence: " + ", ".join(attention) + ".")
    if urgent:
        priority = "urgent"
        reasons.append("Urgent keyword matched: " + ", ".join(urgent) + ".")
    return priority, reasons


def triage(data, previous):
    route, rule_id, reasons = choose_route(data)
    selected = previous["results"][0] if previous["results"] else None
    attention = list(selected["attention_question_ids"]) if selected else []
    priority, priority_reasons = choose_priority(data, route, attention)
    reasons += priority_reasons
    reasons.append("Selected the highest-ranked product." if selected
                   else "No positive search result; routed without assuming a product.")
    return {
        "ticket_id": data["ticket"]["id"], "rule_id": rule_id,
        **{key: route[key] for key in ROUTE}, "priority": priority,
        "selected_product_id": selected["product_id"] if selected else None,
        "evidence_ids": list(selected["evidence_ids"]) if selected else [],
        "attention_question_ids": attention, "reasons": reasons,
    }


def plan_steps(data, triage_output):
    catalog = {step["id"]: step for step in data["onboarding"]["steps"]}
    completed = set(data["profile"]["completed_step_ids"])
    review_id = data["onboarding"]["review_step_id"]
    targets, reasons = [], {}
    if triage_output["attention_question_ids"]:
        targets.append(review_id)
        reasons[review_id] = "Required evidence review for unresolved selected-product findings."
    for sid, step in catalog.items():
        if sid == review_id:
            continue
        product_ok = not step["product_ids"] or (
            triage_output["selected_product_id"] in step["product_ids"])
        if (triage_output["onboarding_track"] in step["tracks"] and
                data["profile"]["experience"] in step["levels"] and product_ok):
            targets.append(sid)
            reasons[sid] = "Matches routed track, experience, and product applicability."
    needed = set()
    pending = list(targets)
    while pending:
        sid = pending.pop()
        if sid in completed or sid in needed:
            continue
        needed.add(sid)
        for prerequisite in catalog[sid]["prerequisites"]:
            reasons.setdefault(prerequisite, f"Required prerequisite for {sid}; overrides level/track filters.")
            pending.append(prerequisite)
    ordered, satisfied = [], set(completed)
    while needed:
        ready = [sid for sid in catalog if sid in needed
                 and set(catalog[sid]["prerequisites"]) <= satisfied]
        require(bool(ready), "onboarding plan cannot satisfy prerequisites")
        if review_id in ready:
            ready.remove(review_id)
            ready.insert(0, review_id)
        for sid in ready:
            ordered.append(sid)
            satisfied.add(sid)
            needed.remove(sid)
    return ordered, reasons


def select_format(data, step):
    for preference in data["profile"]["preferred_formats"]:
        if preference in step["formats"]:
            return preference, f"Uses preferred {preference} format."
    return step["formats"][0], "No preferred format available; uses first supported format."


def adaptive(data, previous):
    ordered, reasons = plan_steps(data, previous)
    catalog = {step["id"]: step for step in data["onboarding"]["steps"]}
    steps = []
    for sid in ordered:
        step = catalog[sid]
        format_name, format_reason = select_format(data, step)
        steps.append({
            "step_id": sid, "title": step["title"], "format": format_name,
            "prerequisites": list(step["prerequisites"]),
            "reason": reasons[sid] + " " + format_reason,
        })
    explanations = [
        "Completed steps are treated as satisfied and omitted; prerequisites override experience filters.",
        "Topological order is stable in catalog order; a ready evidence-review step is prioritized.",
        f"Accountable handoff: {previous['team']} / {previous['owner']}.",
    ]
    if previous["attention_question_ids"]:
        explanations.append("Human review remains required even if the review tutorial was completed.")
    if not steps:
        explanations.append("No remaining applicable onboarding steps.")
    return {
        "experience": data["profile"]["experience"], "track": previous["onboarding_track"],
        "accountable_owner": previous["owner"], "selected_product_id": previous["selected_product_id"],
        "needs_human_review": bool(previous["attention_question_ids"]),
        "attention_question_ids": list(previous["attention_question_ids"]),
        "explanations": explanations, "steps": steps,
    }


def run_pipeline(data, embedder=None):
    """Validate before computation and between every adjacent pair of stages."""
    validate("input", data)
    data = copy.deepcopy(data)
    research_output = validate("research", research(data), data)
    semantic_output = validate("semantic", semantic(data, research_output, embedder),
                               data, research_output)
    triage_output = validate("triage", triage(data, semantic_output), data, semantic_output)
    adaptive_output = validate("adaptive", adaptive(data, triage_output), data, triage_output)
    return {
        "schema_version": "1.0", "status": "ok", "fixture_label": data["fixture_label"],
        "stages": {"research": research_output, "semantic": semantic_output,
                   "triage": triage_output, "adaptive": adaptive_output},
    }


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
