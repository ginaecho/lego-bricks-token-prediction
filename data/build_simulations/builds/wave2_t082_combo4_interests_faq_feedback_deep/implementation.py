"""Synthetic, deterministic interests -> FAQ -> feedback -> research pipeline.

Python 3.10+, standard library only. No provider, network, or model integration.
The strict schemas below define the public contract; unknown fields are errors.
"""

import json
import math
from pathlib import Path
import re
import sys


class ValidationError(ValueError):
    """An input or stage result violates the shared contract."""


def array(spec):
    return ("array", spec)


def enum(*values):
    return ("enum", values)


def nullable(spec):
    return ("nullable", spec)


TEXT = "text"
NUMBER = "weight"
COUNT = "count"
THEME = enum("battery", "durability", "usability", "price", "support", "other")
POSITION = enum("supports", "opposes", "uncertain")
EVIDENCE = {"source_id": TEXT, "excerpt": TEXT}
PREFERENCE = {"topic": TEXT, "weight": NUMBER}
INPUT_SCHEMA = {
    "schema_version": enum("1.0"),
    "fixture_label": TEXT,
    "synthetic": enum(True),
    "interests": {
        "preferences": array(PREFERENCE),
        "excluded_product_ids": array(TEXT),
        "excluded_topics": array(TEXT),
        "limit": "limit",
    },
    "catalog": array({
        "id": TEXT, "name": TEXT, "topics": array(TEXT), "description": TEXT,
    }),
    "faq": {
        "questions": array({"id": TEXT, "product_id": TEXT, "text": TEXT}),
        "documents": array({
            "id": TEXT, "product_ids": array(TEXT), "text": TEXT,
        }),
    },
    "feedback": array({"id": TEXT, "question_id": TEXT, "text": TEXT}),
    "research": {
        "documents": array({
            "id": TEXT,
            "product_ids": array(TEXT),
            "text": TEXT,
            "claims": array({
                "theme": THEME, "claim_key": TEXT, "statement": TEXT,
                "position": POSITION, "excerpt": TEXT,
            }),
        }),
    },
}
FEEDBACK_EXCERPT = {
    "feedback_id": TEXT, "excerpt": TEXT, "duplicate_ids": array(TEXT),
}
OUTPUT_SCHEMAS = {
    "interests": {
        "stage": enum("interests"),
        "input_ids": array(TEXT),
        "recommendations": array({
            "product_id": TEXT, "score": "score",
            "matched_preferences": array(PREFERENCE),
            "explanation": TEXT, "evidence": array(EVIDENCE),
        }),
        "excluded_product_ids": array(TEXT),
    },
    "faq": {
        "stage": enum("faq"),
        "input_ids": array(TEXT),
        "answers": array({
            "question_id": TEXT, "product_id": TEXT, "question": TEXT,
            "status": enum("answered", "abstained"),
            "answer": nullable(TEXT), "citations": array(EVIDENCE), "reason": TEXT,
        }),
        "skipped_question_ids": array(TEXT),
    },
    "feedback": {
        "stage": enum("feedback"),
        "input_ids": array(TEXT),
        "items": array({
            "feedback_id": TEXT, "product_id": TEXT, "question_id": TEXT,
            "text": TEXT, "duplicate_ids": array(TEXT), "themes": array(THEME),
        }),
        "themes": array({
            "theme": THEME, "count": COUNT, "product_ids": array(TEXT),
            "question_ids": array(TEXT), "excerpts": array(FEEDBACK_EXCERPT),
        }),
        "skipped_feedback_ids": array(TEXT),
        "unanswered_question_ids": array(TEXT),
    },
    "deep": {
        "stage": enum("deep"),
        "input_ids": array(TEXT),
        "findings": array({
            "theme": THEME, "product_id": TEXT,
            "feedback_ids": array(TEXT),
            "feedback_excerpts": array(FEEDBACK_EXCERPT),
            "claims": array({
                "claim_key": TEXT, "statement": TEXT,
                "assessment": enum("supported", "opposed", "mixed", "uncertain"),
                "evidence": array({
                    "source_id": TEXT, "excerpt": TEXT, "position": POSITION,
                }),
            }),
            "summary": TEXT, "unresolved": array(TEXT),
        }),
        "disagreements": array({
            "theme": THEME, "product_id": TEXT,
            "claim_key": TEXT, "source_ids": array(TEXT),
        }),
        "unresolved_questions": array({
            "question_id": TEXT, "question": TEXT, "reason": TEXT,
        }),
        "documents_used": array(TEXT),
    },
}
STOPWORDS = set(
    "a an the what which is are was were do does did how can could i my "
    "it its for of to in on and or please tell me about this that".split()
)
THEME_WORDS = {
    "battery": {"battery", "charge", "charging", "runtime", "power"},
    "durability": {"durability", "durable", "broken", "break", "cracked", "sturdy"},
    "usability": {"usability", "easy", "difficult", "setup", "controls", "confusing"},
    "price": {"price", "cost", "expensive", "cheap", "value"},
    "support": {"support", "service", "warranty", "refund", "return", "help"},
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def validate(value, spec, path="$"):
    """One recursive, bounded, exact-type validator for input and every output."""
    if isinstance(spec, dict):
        require(type(value) is dict, f"{path}: expected object")
        require(set(value) == set(spec), f"{path}: missing or unknown fields")
        for key, child_spec in spec.items():
            validate(value[key], child_spec, f"{path}.{key}")
    elif isinstance(spec, tuple):
        kind, child_spec = spec
        if kind == "array":
            require(type(value) is list, f"{path}: expected array")
            require(len(value) <= 500, f"{path}: too many entries (maximum 500)")
            for index, child in enumerate(value):
                validate(child, child_spec, f"{path}[{index}]")
        elif kind == "enum":
            require(any(type(value) is type(v) and value == v for v in child_spec),
                    f"{path}: unsupported value")
        elif kind == "nullable":
            if value is not None:
                validate(value, child_spec, path)
    elif spec == TEXT:
        require(type(value) is str and bool(value.strip()), f"{path}: expected nonblank text")
        require(len(value) <= 20000, f"{path}: text exceeds 20000 characters")
        require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value),
                f"{path}: invalid Unicode surrogate")
    elif spec in ("weight", "score"):
        require(type(value) in (int, float), f"{path}: expected finite number")
        require(0 < value <= (100 if spec == "weight" else 50000),
                f"{path}: number outside permitted range")
        require(math.isfinite(value), f"{path}: expected finite number")
    elif spec in ("count", "limit"):
        upper = 20 if spec == "limit" else 500
        lower = 1 if spec == "limit" else 0
        require(type(value) is int and lower <= value <= upper,
                f"{path}: expected integer in [{lower}, {upper}]")


def norm(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def tokens(text):
    return set(norm(text).split()) - STOPWORDS


def unique(values, label):
    require(len(values) == len(set(values)), f"{label}: duplicate values")


def indexed(records, label):
    unique([r["id"] for r in records], label)
    return {r["id"]: r for r in records}


def validate_input(data):
    validate(data, INPUT_SCHEMA)
    require("synthetic" in data["fixture_label"].casefold(),
            "fixture_label must explicitly identify synthetic data")
    catalog = indexed(data["catalog"], "catalog IDs")
    questions = indexed(data["faq"]["questions"], "question IDs")
    indexed(data["faq"]["documents"], "FAQ document IDs")
    indexed(data["feedback"], "feedback IDs")
    indexed(data["research"]["documents"], "research document IDs")
    preferences = data["interests"]["preferences"]
    unique([norm(p["topic"]) for p in preferences], "preference topics")
    for topic in [p["topic"] for p in preferences] + data["interests"]["excluded_topics"]:
        require(bool(norm(topic)), "topics must contain letters or numbers")
    for field in ("excluded_product_ids", "excluded_topics"):
        values = data["interests"][field]
        unique([norm(v) for v in values] if field.endswith("topics") else values, field)
    require(set(data["interests"]["excluded_product_ids"]) <= catalog.keys(),
            "excluded product references unknown catalog ID")
    for product in catalog.values():
        unique([norm(t) for t in product["topics"]], "catalog topics")
        require(all(norm(t) for t in product["topics"]), "catalog topics cannot be punctuation")
    for question in questions.values():
        require(question["product_id"] in catalog, "question references unknown product")
    for feedback in data["feedback"]:
        require(feedback["question_id"] in questions, "feedback references unknown question")
        require(bool(norm(feedback["text"])), "feedback must contain letters or numbers")
    claim_statements = {}
    for section in ("faq", "research"):
        for document in data[section]["documents"]:
            unique(document["product_ids"], "document product IDs")
            require(bool(document["product_ids"]), "document requires at least one product")
            require(set(document["product_ids"]) <= catalog.keys(),
                    "document references unknown product")
            if section == "research":
                seen_claims = set()
                for claim in document["claims"]:
                    require(claim["excerpt"] in document["text"],
                            "research excerpt is not verbatim source text")
                    key = (claim["theme"], claim["claim_key"])
                    old = claim_statements.setdefault(key, claim["statement"])
                    require(old == claim["statement"],
                            "a claim key must have one consistent proposition")
                    signature = (key, claim["position"], claim["excerpt"])
                    require(signature not in seen_claims, "duplicate research claim")
                    seen_claims.add(signature)
    return data


def recommend(data):
    config = data["interests"]
    excluded_topics = {norm(t) for t in config["excluded_topics"]}
    excluded_ids = set(config["excluded_product_ids"])
    excluded, results = [], []
    for product in data["catalog"]:
        topics = {norm(t) for t in product["topics"]}
        if product["id"] in excluded_ids or topics & excluded_topics:
            excluded.append(product["id"])
            continue
        matches = [dict(p) for p in config["preferences"] if norm(p["topic"]) in topics]
        if not matches:
            continue
        matches.sort(key=lambda p: norm(p["topic"]))
        score = sum(p["weight"] for p in matches)
        labels = ", ".join(f'{p["topic"]} ({p["weight"]:g})' for p in matches)
        results.append({
            "product_id": product["id"], "score": score,
            "matched_preferences": matches,
            "explanation": f"Matched preferences: {labels}. Score: {score:g}. "
                           f"Source: {product['id']}.",
            "evidence": [{"source_id": product["id"], "excerpt": product["description"]}],
        })
    results.sort(key=lambda r: (-r["score"], r["product_id"]))
    return {
        "stage": "interests", "input_ids": [p["id"] for p in data["catalog"]],
        "recommendations": results[:config["limit"]],
        "excluded_product_ids": sorted(excluded),
    }


def answer_faq(data, interests):
    selected = [r["product_id"] for r in interests["recommendations"]]
    answers, skipped = [], []
    for question in data["faq"]["questions"]:
        if question["product_id"] not in selected:
            skipped.append(question["id"])
            continue
        query = tokens(question["text"])
        # Conservative lexical retrieval: every content token must occur in a source.
        documents = [
            doc for doc in data["faq"]["documents"]
            if question["product_id"] in doc["product_ids"]
            and query and query <= tokens(doc["text"])
        ]
        documents.sort(key=lambda doc: (
            len(tokens(doc["text"]) - query), doc["id"],
        ))
        citations = [{"source_id": d["id"], "excerpt": d["text"]} for d in documents[:2]]
        answers.append({
            "question_id": question["id"], "product_id": question["product_id"],
            "question": question["text"],
            "status": "answered" if citations else "abstained",
            "answer": "\n".join(c["excerpt"] for c in citations) if citations else None,
            "citations": citations,
            "reason": "Verbatim knowledge-base excerpts; not independently verified."
                      if citations else "No product-scoped source covers all query terms.",
        })
    return {
        "stage": "faq", "input_ids": selected, "answers": answers,
        "skipped_question_ids": skipped,
    }


def analyze_feedback(data, faq):
    questions = {a["question_id"]: a for a in faq["answers"]}
    canonical, skipped = {}, []
    for feedback in data["feedback"]:
        question = questions.get(feedback["question_id"])
        if question is None:
            skipped.append(feedback["id"])
            continue
        key = (feedback["question_id"], norm(feedback["text"]))
        if key in canonical:
            canonical[key]["duplicate_ids"].append(feedback["id"])
            continue
        words = set(norm(feedback["text"]).split())
        themes = sorted(theme for theme, vocabulary in THEME_WORDS.items() if words & vocabulary)
        canonical[key] = {
            "feedback_id": feedback["id"], "product_id": question["product_id"],
            "question_id": feedback["question_id"], "text": feedback["text"],
            "duplicate_ids": [], "themes": themes or ["other"],
        }
    items = list(canonical.values())
    themes = []
    for theme in sorted({theme for item in items for theme in item["themes"]}):
        members = [item for item in items if theme in item["themes"]]
        themes.append({
            "theme": theme, "count": len(members),
            "product_ids": sorted({item["product_id"] for item in members}),
            "question_ids": sorted({item["question_id"] for item in members}),
            "excerpts": [feedback_excerpt(item) for item in members],
        })
    return {
        "stage": "feedback", "input_ids": list(questions), "items": items, "themes": themes,
        "skipped_feedback_ids": skipped,
        "unanswered_question_ids": [
            a["question_id"] for a in faq["answers"] if a["status"] == "abstained"
        ],
    }


def feedback_excerpt(item):
    return {
        "feedback_id": item["feedback_id"], "excerpt": item["text"],
        "duplicate_ids": list(item["duplicate_ids"]),
    }


def synthesize(data, feedback):
    findings, disagreements, used = [], [], set()
    for theme in feedback["themes"]:
        for product_id in theme["product_ids"]:
            members = [
                item for item in feedback["items"]
                if item["product_id"] == product_id and theme["theme"] in item["themes"]
            ]
            grouped = {}
            for doc in sorted(data["research"]["documents"], key=lambda d: d["id"]):
                if product_id not in doc["product_ids"]:
                    continue
                for claim in doc["claims"]:
                    if claim["theme"] != theme["theme"]:
                        continue
                    group = grouped.setdefault(claim["claim_key"], {
                        "claim_key": claim["claim_key"], "statement": claim["statement"],
                        "evidence": [],
                    })
                    group["evidence"].append({
                        "source_id": doc["id"], "excerpt": claim["excerpt"],
                        "position": claim["position"],
                    })
                    used.add(doc["id"])
            claims, unresolved = [], []
            for key, group in sorted(grouped.items()):
                positions = {e["position"] for e in group["evidence"]}
                source_ids = sorted({e["source_id"] for e in group["evidence"]})
                if {"supports", "opposes"} <= positions:
                    assessment = "mixed"
                    disagreements.append({
                        "theme": theme["theme"], "product_id": product_id,
                        "claim_key": key, "source_ids": source_ids,
                    })
                    unresolved.append(f"Conflicting evidence for {key}; no consensus inferred.")
                elif "supports" in positions:
                    assessment = "supported"
                elif "opposes" in positions:
                    assessment = "opposed"
                else:
                    assessment = "uncertain"
                    unresolved.append(f"Only uncertain evidence for {key}.")
                if len(source_ids) < 2:
                    unresolved.append(f"Only one document for {key}; independent corroboration missing.")
                if "uncertain" in positions and assessment != "uncertain":
                    unresolved.append(f"Uncertain evidence remains for {key}.")
                group["assessment"] = assessment
                claims.append(group)
            if not claims:
                unresolved.append("No research evidence for this product and feedback theme.")
            source_count = len({e["source_id"] for c in claims for e in c["evidence"]})
            counts = {label: sum(c["assessment"] == label for c in claims)
                      for label in ("supported", "opposed", "mixed", "uncertain")}
            summary = (
                f"{len(members)} unique feedback item(s); {source_count} research document(s); "
                + ", ".join(f"{count} {label}" for label, count in counts.items())
                + ". Assessments describe supplied evidence, not verified facts."
            )
            findings.append({
                "theme": theme["theme"], "product_id": product_id,
                "feedback_ids": [m["feedback_id"] for m in members],
                "feedback_excerpts": [feedback_excerpt(m) for m in members],
                "claims": claims, "summary": summary, "unresolved": unresolved,
            })
    questions = {q["id"]: q for q in data["faq"]["questions"]}
    return {
        "stage": "deep", "input_ids": [i["feedback_id"] for i in feedback["items"]],
        "findings": findings, "disagreements": disagreements,
        "unresolved_questions": [{
            "question_id": qid, "question": questions[qid]["text"],
            "reason": "FAQ abstention remains unresolved; research is not a substitute FAQ answer.",
        } for qid in feedback["unanswered_question_ids"]],
        "documents_used": sorted(used),
    }


def validate_stage(name, output, data, previous=None):
    """Validate structure plus provenance and handoff invariants before consumption."""
    require(name in OUTPUT_SCHEMAS, "unknown pipeline stage")
    validate(output, OUTPUT_SCHEMAS[name], f"$.stages.{name}")
    if name == "interests":
        expected = [p["id"] for p in data["catalog"]]
        products = {p["id"]: p for p in data["catalog"]}
        ids = [r["product_id"] for r in output["recommendations"]]
        unique(ids, "recommended product IDs")
        require(len(ids) <= data["interests"]["limit"], "recommendation limit exceeded")
        excluded_topics = {norm(t) for t in data["interests"]["excluded_topics"]}
        excluded = {
            p["id"] for p in data["catalog"]
            if p["id"] in data["interests"]["excluded_product_ids"]
            or excluded_topics & {norm(t) for t in p["topics"]}
        }
        require(output["excluded_product_ids"] == sorted(excluded), "incorrect exclusion audit")
        require(not (set(ids) & excluded), "excluded recommendation")
        for rec in output["recommendations"]:
            require(rec["product_id"] in products, "unknown recommendation")
            product = products[rec["product_id"]]
            matches = sorted(
                [p for p in data["interests"]["preferences"]
                 if norm(p["topic"]) in {norm(t) for t in product["topics"]}],
                key=lambda p: norm(p["topic"]),
            )
            require(bool(matches) and rec["matched_preferences"] == matches,
                    "ungrounded preference matches")
            require(rec["score"] == sum(p["weight"] for p in matches), "incorrect preference score")
            require(rec["evidence"] == [{"source_id": product["id"],
                                        "excerpt": product["description"]}],
                    "ungrounded recommendation evidence")
        require(output["recommendations"] == sorted(
            output["recommendations"], key=lambda r: (-r["score"], r["product_id"])),
            "recommendations are not ranked")
    elif name == "faq":
        require(previous is not None, "FAQ requires validated interests")
        expected = [r["product_id"] for r in previous["recommendations"]]
        questions = {q["id"]: q for q in data["faq"]["questions"]}
        docs = {d["id"]: d for d in data["faq"]["documents"]}
        require([a["question_id"] for a in output["answers"]] ==
                [q["id"] for q in data["faq"]["questions"] if q["product_id"] in expected],
                "FAQ answer set violates recommendation handoff")
        require(output["skipped_question_ids"] ==
                [q["id"] for q in data["faq"]["questions"] if q["product_id"] not in expected],
                "incorrect skipped question audit")
        for answer in output["answers"]:
            question = questions[answer["question_id"]]
            require(answer["product_id"] == question["product_id"]
                    and answer["question"] == question["text"], "FAQ question changed")
            for citation in answer["citations"]:
                require(citation["source_id"] in docs, "unknown FAQ citation")
                doc = docs[citation["source_id"]]
                require(answer["product_id"] in doc["product_ids"]
                        and citation["excerpt"] == doc["text"], "ungrounded FAQ citation")
                require(bool(tokens(question["text"])) and
                        tokens(question["text"]) <= tokens(doc["text"]),
                        "FAQ citation does not cover query")
            if answer["status"] == "answered":
                require(bool(answer["citations"]) and answer["answer"] ==
                        "\n".join(c["excerpt"] for c in answer["citations"]),
                        "FAQ answer must be verbatim cited text")
            else:
                require(answer["answer"] is None and not answer["citations"],
                        "abstention cannot invent an answer or citations")
    elif name == "feedback":
        require(previous is not None, "feedback requires validated FAQ")
        expected = [a["question_id"] for a in previous["answers"]]
        originals = {f["id"]: f for f in data["feedback"]}
        questions = {a["question_id"]: a for a in previous["answers"]}
        observed = []
        for item in output["items"]:
            require(item["feedback_id"] in originals, "unknown feedback")
            original = originals[item["feedback_id"]]
            require(item["question_id"] in questions, "feedback escaped FAQ scope")
            require(item["text"] == original["text"]
                    and item["question_id"] == original["question_id"], "feedback source changed")
            require(item["product_id"] == questions[item["question_id"]]["product_id"],
                    "feedback product changed")
            observed.append(item["feedback_id"])
            for duplicate_id in item["duplicate_ids"]:
                require(duplicate_id in originals, "unknown duplicate feedback")
                duplicate = originals[duplicate_id]
                require(duplicate["question_id"] == item["question_id"]
                        and norm(duplicate["text"]) == norm(item["text"]),
                        "incorrect feedback deduplication")
                observed.append(duplicate_id)
        unique(observed, "feedback lineage IDs")
        require(set(observed) == {f["id"] for f in data["feedback"]
                                  if f["question_id"] in questions}, "feedback lineage incomplete")
        require(output["skipped_feedback_ids"] ==
                [f["id"] for f in data["feedback"] if f["question_id"] not in questions],
                "incorrect skipped feedback audit")
        for theme in output["themes"]:
            members = [i for i in output["items"] if theme["theme"] in i["themes"]]
            require(theme["count"] == len(members)
                    and theme["excerpts"] == [feedback_excerpt(i) for i in members]
                    and theme["product_ids"] == sorted({i["product_id"] for i in members})
                    and theme["question_ids"] == sorted({i["question_id"] for i in members}),
                    "theme aggregation or evidence changed")
        require(output["unanswered_question_ids"] ==
                [a["question_id"] for a in previous["answers"] if a["status"] == "abstained"],
                "FAQ abstentions lost")
    else:
        require(previous is not None, "research requires validated feedback")
        expected = [i["feedback_id"] for i in previous["items"]]
        docs = {d["id"]: d for d in data["research"]["documents"]}
        seen_docs, expected_disagreements = set(), []
        require([(f["theme"], f["product_id"]) for f in output["findings"]] ==
                [(t["theme"], pid) for t in previous["themes"] for pid in t["product_ids"]],
                "research findings escaped feedback scope")
        for finding in output["findings"]:
            members = [i for i in previous["items"]
                       if i["product_id"] == finding["product_id"]
                       and finding["theme"] in i["themes"]]
            require(finding["feedback_ids"] == [i["feedback_id"] for i in members]
                    and finding["feedback_excerpts"] == [feedback_excerpt(i) for i in members],
                    "research lost feedback provenance")
            for claim in finding["claims"]:
                require(bool(claim["evidence"]), "research claim has no evidence")
                positions, source_ids = set(), set()
                for evidence in claim["evidence"]:
                    require(evidence["source_id"] in docs, "unknown research source")
                    doc = docs[evidence["source_id"]]
                    require(finding["product_id"] in doc["product_ids"], "wrong product evidence")
                    require(any(
                        c["theme"] == finding["theme"] and c["claim_key"] == claim["claim_key"]
                        and c["statement"] == claim["statement"]
                        and c["excerpt"] == evidence["excerpt"]
                        and c["position"] == evidence["position"] for c in doc["claims"]
                    ), "research evidence does not match source annotation")
                    source_ids.add(doc["id"])
                    positions.add(evidence["position"])
                assessment = (
                    "mixed" if {"supports", "opposes"} <= positions else
                    "supported" if "supports" in positions else
                    "opposed" if "opposes" in positions else "uncertain"
                )
                require(claim["assessment"] == assessment, "incorrect research assessment")
                if assessment == "mixed":
                    expected_disagreements.append({
                        "theme": finding["theme"], "product_id": finding["product_id"],
                        "claim_key": claim["claim_key"], "source_ids": sorted(source_ids),
                    })
                seen_docs.update(source_ids)
        require(output["documents_used"] == sorted(seen_docs), "research source audit changed")
        require(output["disagreements"] == expected_disagreements, "disagreement audit changed")
        questions = {q["id"]: q["text"] for q in data["faq"]["questions"]}
        require([(q["question_id"], q["question"]) for q in output["unresolved_questions"]] ==
                [(qid, questions[qid]) for qid in previous["unanswered_question_ids"]],
                "unresolved FAQ questions lost")
    require(output["input_ids"] == expected, f"{name}: invalid upstream lineage")
    return output


def run_pipeline(data):
    """Run each stage only after the previous result passes shared validation."""
    validate_input(data)
    interests = validate_stage("interests", recommend(data), data)
    faq = validate_stage("faq", answer_faq(data, interests), data, interests)
    feedback = validate_stage("feedback", analyze_feedback(data, faq), data, faq)
    deep = validate_stage("deep", synthesize(data, feedback), data, feedback)
    return {
        "schema_version": "1.0", "status": "ok", "fixture_label": data["fixture_label"],
        "stages": {"interests": interests, "faq": faq, "feedback": feedback, "deep": deep},
    }


def reject_constant(value):
    raise ValidationError(f"nonfinite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py example_input.json")
        with Path(args[0]).open("rb") as stream:
            raw = stream.read(1_000_001)
        require(len(raw) <= 1_000_000, "input exceeds 1000000 bytes")
        data = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                          object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (ValueError, OSError, RecursionError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
