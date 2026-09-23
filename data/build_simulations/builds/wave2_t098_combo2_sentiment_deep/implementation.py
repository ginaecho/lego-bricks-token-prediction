"""Deterministic synthetic customer-insight -> evidence-research reference CLI."""

import json
import re
import sys

SCHEMA_VERSION = "1.0"
SEVERITY = {"low": 0, "medium": 100, "high": 200, "critical": 300}
LEXICON = {
    "excellent": 2, "great": 2, "love": 2, "helpful": 1, "good": 1,
    "happy": 1, "fast": 1, "bad": -1, "slow": -1, "broken": -2,
    "hate": -2, "terrible": -2, "crash": -2, "unsafe": -3, "lost": -2,
}
NEGATORS = {"not", "never", "no"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, keys, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def array(value, path):
    require(type(value) is list, path + " must be an array")


def same_structure(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(same_structure(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(same_structure(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def validate(value, kind, context=None):
    """The single validation entry point for input and every stage handoff."""
    if kind == "input":
        fields(value, ["schema_version", "synthetic", "documents", "issues"], "input")
        require(value["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
        require(value["synthetic"] is True, "fixtures must be explicitly synthetic")
        array(value["documents"], "documents")
        array(value["issues"], "issues")
        ids = set()
        for doc in value["documents"]:
            fields(doc, ["id", "title", "text", "claims"], "document")
            for key in ("id", "title", "text"):
                text(doc[key], "document." + key)
            require(doc["id"] not in ids, "duplicate document id")
            ids.add(doc["id"])
            array(doc["claims"], "claims")
            seen = set()
            for claim in doc["claims"]:
                fields(claim, ["topic", "stance", "quote"], "claim")
                for key in claim:
                    text(claim[key], "claim." + key)
                require(claim["topic"] == claim["topic"].strip().casefold(),
                        "claim.topic must be trimmed lowercase canonical text")
                require(claim["stance"] in ("supports", "opposes", "uncertain"),
                        "invalid claim stance")
                require(claim["quote"] in doc["text"], "claim quote is not in its document")
                signature = (claim["topic"], claim["stance"], claim["quote"])
                require(signature not in seen, "duplicate claim")
                seen.add(signature)
        issue_ids = set()
        for issue in value["issues"]:
            fields(issue, ["id", "text", "severity", "document_ids"], "issue")
            text(issue["id"], "issue.id")
            text(issue["text"], "issue.text")
            text(issue["severity"], "issue.severity")
            require(issue["severity"] in SEVERITY, "invalid severity")
            require(issue["id"] not in issue_ids, "duplicate issue id")
            issue_ids.add(issue["id"])
            array(issue["document_ids"], "issue.document_ids")
            refs = issue["document_ids"]
            for ref in refs:
                text(ref, "document reference")
                require(ref in ids, "unknown document reference: " + ref)
            require(len(refs) == len(set(refs)), "duplicate document reference")
    elif kind == "sentiment":
        require(context is not None, "sentiment validation requires input context")
        validate(context, "input")
        fields(value, ["schema_version", "issues"], "sentiment")
        require(value["schema_version"] == SCHEMA_VERSION, "invalid sentiment version")
        array(value["issues"], "sentiment.issues")
        # Exact deterministic replay verifies scores, explanations, ordering and provenance.
        require(same_structure(value, _sentiment(context)), "sentiment handoff differs from validated input")
    elif kind == "research":
        require(context is not None and type(context) is tuple and len(context) == 2,
                "research validation requires input and sentiment")
        source, sentiment = context
        validate(sentiment, "sentiment", source)
        fields(value, ["schema_version", "findings"], "research")
        require(value["schema_version"] == SCHEMA_VERSION, "invalid research version")
        array(value["findings"], "research.findings")
        require(same_structure(value, _research(source, sentiment)), "research output has invalid evidence or propagation")
    else:
        raise ValidationError("unknown validation kind")
    return value


def score_sentiment(content):
    """Score lexical hits; a negator flips the next hit within three word tokens."""
    tokens = re.findall(r"[^\W_]+|[.!?;,]", content.casefold(), re.UNICODE)
    contributions = []
    pending = 0
    word_index = -1
    for token in tokens:
        if token in ".!?;,":
            pending = 0
            continue
        word_index += 1
        if token in NEGATORS:
            pending = 3
            continue
        if token in LEXICON:
            base = LEXICON[token]
            contributions.append({
                "token": token, "word_index": word_index, "base": base,
                "negated": pending > 0, "contribution": -base if pending else base,
            })
            pending = 0
        elif pending:
            pending -= 1
    raw = sum(item["contribution"] for item in contributions)
    label = "positive" if raw > 0 else "negative" if raw < 0 else "neutral"
    return {"score": raw, "label": label, "contributions": contributions}


def _sentiment(source):
    records = []
    for issue in source["issues"]:
        sentiment = score_sentiment(issue["text"])
        penalty = min(99, max(0, -sentiment["score"]) * 10)
        records.append({
            "issue_id": issue["id"], "text": issue["text"],
            "severity": issue["severity"], "document_ids": list(issue["document_ids"]),
            "sentiment": sentiment, "priority_score": SEVERITY[issue["severity"]] + penalty,
            "priority_explanation": {
                "severity_base": SEVERITY[issue["severity"]],
                "negative_sentiment_adjustment": penalty,
            },
        })
    records.sort(key=lambda record: (-record["priority_score"], record["issue_id"]))
    for rank, record in enumerate(records, 1):
        record["rank"] = rank
    return {"schema_version": SCHEMA_VERSION, "issues": records}


def sentiment_stage(source):
    validate(source, "input")
    return validate(_sentiment(source), "sentiment", source)


def _research(source, sentiment):
    documents = {doc["id"]: doc for doc in source["documents"]}
    findings = []
    for issue in sentiment["issues"]:
        topics = {}
        empty_documents = []
        for ref in issue["document_ids"]:
            doc = documents[ref]
            if not doc["claims"]:
                empty_documents.append(ref)
            for claim in doc["claims"]:
                topics.setdefault(claim["topic"], []).append({
                    "document_id": ref, "document_title": doc["title"],
                    "stance": claim["stance"], "quote": claim["quote"],
                })
        syntheses, unresolved = [], []
        if not topics:
            unresolved.append("No grounded claims: what evidence addresses this issue?")
        for ref in sorted(empty_documents):
            unresolved.append("Document " + ref + " has no annotated claims; what relevant evidence is missing?")
        for topic, evidence in sorted(topics.items()):
            evidence.sort(key=lambda e: (e["document_id"], e["stance"], e["quote"]))
            counts = {stance: len({e["document_id"] for e in evidence if e["stance"] == stance})
                      for stance in ("supports", "opposes", "uncertain")}
            disagreement = counts["supports"] > 0 and counts["opposes"] > 0
            source_count = len({e["document_id"] for e in evidence})
            if disagreement:
                conclusion = "disputed"
                unresolved.append(topic + ": what explains the conflicting evidence?")
            elif counts["uncertain"]:
                conclusion = "inconclusive"
            else:
                conclusion = "supported" if counts["supports"] else "opposed"
            if counts["uncertain"]:
                unresolved.append(topic + ": what would resolve the uncertain evidence?")
            if source_count < 2:
                unresolved.append(topic + ": can another independent source corroborate this?")
            syntheses.append({
                "topic": topic, "conclusion": conclusion, "disagreement": disagreement,
                "source_count": source_count, "stance_source_counts": counts,
                "evidence": evidence,
                "summary": (
                    f"{topic}: {conclusion}; {counts['supports']} supporting, "
                    f"{counts['opposes']} opposing, {counts['uncertain']} uncertain document(s)."
                ),
            })
        findings.append({
            "issue_id": issue["issue_id"], "severity": issue["severity"],
            "priority_score": issue["priority_score"], "rank": issue["rank"],
            "sentiment_label": issue["sentiment"]["label"],
            "sentiment_score": issue["sentiment"]["score"],
            "document_ids": list(issue["document_ids"]),
            "topics": syntheses, "unresolved_questions": unresolved,
        })
    return {"schema_version": SCHEMA_VERSION, "findings": findings}


def research_stage(source, sentiment):
    validate(sentiment, "sentiment", source)
    return validate(_research(source, sentiment), "research", (source, sentiment))


def run_pipeline(source):
    validate(source, "input")
    sentiment = sentiment_stage(source)
    research = research_stage(source, sentiment)
    return {"status": "ok", "schema_version": SCHEMA_VERSION, "synthetic": True,
            "sentiment": sentiment, "research": research}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json (or - for stdin)")
        if argv[0] == "-":
            raw = sys.stdin.read()
        else:
            with open(argv[0], encoding="utf-8") as handle:
                raw = handle.read()
        source = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(source)
    except (ValidationError, ValueError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
