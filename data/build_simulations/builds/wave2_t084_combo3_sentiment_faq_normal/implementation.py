"""Deterministic synthetic customer-insights -> FAQ -> research reference CLI.

All stages exchange the same validated envelope. Evidence is extractive, not a
claim of real-world truth. Citation offsets are Python Unicode character offsets
with an exclusive end. No network, model, or third-party package is used.
"""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    """An input or intermediate envelope violates the shared contract."""


STAGES = ("input", "sentiment", "faq", "normal")
SEVERITIES = {"low": 0, "medium": 1, "high": 2, "critical": 3}
LEXICON = {
    "love": 2, "excellent": 2, "great": 2, "good": 1, "happy": 1,
    "helpful": 1, "thanks": 1, "bad": -1, "broken": -2, "terrible": -2,
    "angry": -2, "failed": -2, "late": -1, "missing": -1, "unsafe": -3,
}
NEGATORS = {"not", "never", "no"}
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "i", "my", "me",
    "we", "our", "you", "your", "it", "its", "and", "or", "to", "of",
    "for", "in", "on", "with", "please", "how", "can", "do", "does",
    "this", "that", "have", "has", "be", "am", "need",
} | NEGATORS
ROOT_KEYS = {
    "schema_version", "synthetic", "status", "stage", "feedback",
    "knowledge_base", "research_sources", "results",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def tokens(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def terms(text):
    return set(tokens(text)) - STOPWORDS


def nonblank(value, label, limit=20000):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonblank text")
    require(len(value) <= limit, label + " is too long")


def sentiments(feedback):
    records = []
    for item in feedback:
        words = tokens(item["text"])
        evidence = []
        for index, token in enumerate(words):
            if token in LEXICON:
                base = LEXICON[token]
                negated = index > 0 and words[index - 1] in NEGATORS
                evidence.append({
                    "token": token, "token_index": index, "base": base,
                    "negated": negated, "contribution": -base if negated else base,
                })
        score = sum(e["contribution"] for e in evidence)
        severity = item["severity"]
        rank = (0 if severity == "critical" else
                1 if severity == "high" or score <= -3 else
                2 if severity == "medium" or score < 0 else 3)
        records.append({
            "feedback_id": item["id"], "text": item["text"],
            "severity": severity, "score": score,
            "sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "evidence": evidence, "priority": "P" + str(rank),
            "priority_rank": rank,
        })
    return sorted(records, key=lambda r: (
        r["priority_rank"], -SEVERITIES[r["severity"]], r["score"], r["feedback_id"],
    ))


def passages(source):
    # Sentence/newline chunks preserve original positions, including Unicode.
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", source["text"]):
        raw = match.group()
        quote = raw.strip()
        if quote:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield {
                "source_id": source["id"], "title": source["title"],
                "start": start, "end": start + len(quote), "quote": quote,
            }


def retrieve(query, sources, limit):
    query_terms = terms(query)
    hits = []
    for source in sources:
        for citation in passages(source):
            overlap = sorted(query_terms & terms(citation["quote"]))
            coverage = len(overlap) / len(query_terms) if query_terms else 0.0
            if len(overlap) >= 2 and coverage >= 0.2:
                hits.append({
                    "citation": citation, "matched_terms": overlap,
                    "coverage": round(coverage, 6),
                })
    hits.sort(key=lambda h: (
        -len(h["matched_terms"]), -h["coverage"],
        h["citation"]["source_id"], h["citation"]["start"],
    ))
    return hits[:limit]


def faq_answers(sentiment_records, knowledge_base):
    answers = []
    for record in sentiment_records:
        hits = retrieve(record["text"], knowledge_base, 1)
        answer = hits[0]["citation"]["quote"] if hits else None
        answers.append({
            "feedback_id": record["feedback_id"], "priority": record["priority"],
            "severity": record["severity"], "sentiment": record["sentiment"],
            "query": record["text"], "status": "answered" if hits else "abstained",
            "answer": answer, "evidence": hits,
            "reason": None if hits else "No KB passage meets the lexical evidence threshold.",
            "research_query": record["text"] + ("\n" + answer if answer else ""),
        })
    return answers


def research_findings(answers, research_sources):
    findings = []
    for answer in answers:
        hits = retrieve(answer["research_query"], research_sources, 3)
        findings.append({
            "feedback_id": answer["feedback_id"], "priority": answer["priority"],
            "faq_status": answer["status"], "query": answer["research_query"],
            "status": "found" if hits else "abstained",
            "findings": [{"text": hit["citation"]["quote"], **hit} for hit in hits],
            "reason": None if hits else "No research passage meets the lexical evidence threshold.",
        })
    return findings


def validate(envelope, expected_stage=None):
    """One schema/validation layer for input and every cross-stage handoff.

    Deterministic intermediate records are recomputed and compared, validating
    lineage, priority, rankings, abstentions, and exact citation spans together.
    """
    require(isinstance(envelope, dict), "Envelope must be an object")
    require(set(envelope) == ROOT_KEYS, "Envelope has missing or unknown fields")
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "schema_version must be integer 1")
    require(envelope["synthetic"] is True, "This reference accepts labeled synthetic data only")
    require(envelope["status"] == "ok", "Envelope status must be ok")
    stage = envelope["stage"]
    require(isinstance(stage, str) and stage in STAGES, "Unknown stage")
    require(expected_stage is None or stage == expected_stage, "Unexpected stage: " + stage)
    for collection in ("feedback", "knowledge_base", "research_sources"):
        rows = envelope[collection]
        require(isinstance(rows, list) and len(rows) <= 100, collection + " must be a list of at most 100")
        identifiers = set()
        for row in rows:
            require(isinstance(row, dict), collection + " entries must be objects")
            keys = {"id", "text", "severity"} if collection == "feedback" else {"id", "title", "text"}
            require(set(row) == keys, collection + " entry has missing or unknown fields")
            nonblank(row["id"], collection + ".id", 100)
            require(row["id"] not in identifiers, collection + " contains duplicate IDs")
            identifiers.add(row["id"])
            nonblank(row["text"], collection + ".text")
            if collection == "feedback":
                require(isinstance(row["severity"], str) and row["severity"] in SEVERITIES,
                        "severity must be low, medium, high, or critical")
            else:
                nonblank(row["title"], collection + ".title", 500)
    results = envelope["results"]
    require(isinstance(results, dict), "results must be an object")
    count = STAGES.index(stage)
    require(set(results) == set(STAGES[1:count + 1]), "results do not match stage")
    expected = {}
    if count >= 1:
        expected["sentiment"] = sentiments(envelope["feedback"])
    if count >= 2:
        expected["faq"] = faq_answers(expected["sentiment"], envelope["knowledge_base"])
    if count >= 3:
        expected["normal"] = research_findings(expected["faq"], envelope["research_sources"])
    # JSON comparison also distinguishes boolean values from numeric values.
    try:
        actual_json = json.dumps(results, sort_keys=True, ensure_ascii=False, allow_nan=False)
        expected_json = json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("results must be finite JSON data") from exc
    require(actual_json == expected_json, "Invalid derived results or cross-stage lineage")
    return envelope


def advance(envelope, previous, following, records):
    validate(envelope, previous)
    output = copy.deepcopy(envelope)
    output["stage"] = following
    output["results"][following] = records(output)
    return validate(output, following)


def sentiment_stage(envelope):
    return advance(envelope, "input", "sentiment", lambda e: sentiments(e["feedback"]))


def faq_stage(envelope):
    return advance(envelope, "sentiment", "faq",
                   lambda e: faq_answers(e["results"]["sentiment"], e["knowledge_base"]))


def normal_stage(envelope):
    return advance(envelope, "faq", "normal",
                   lambda e: research_findings(e["results"]["faq"], e["research_sources"]))


def run_pipeline(envelope):
    return normal_stage(faq_stage(sentiment_stage(envelope)))


def reject_constant(value):
    raise ValidationError("Non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as stream:
            envelope = json.load(stream, object_pairs_hook=unique_object,
                                 parse_constant=reject_constant)
        output = run_pipeline(envelope)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
