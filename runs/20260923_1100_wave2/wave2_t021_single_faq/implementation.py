"""Synthetic KB reference CLI. No generation or external dependencies.

Input: schema_version=1, question, knowledge_base=[{id,title,content}],
and optional min_coverage (default 0.6). Unknown fields are rejected.
Output: schema_version, status (answered/abstained/error), answer, citations,
retrieval, reason, errors. Answers are verbatim source content, not a guarantee
that the supplied source is true. Lexical coverage is not a probability.
"""

import json
import math
from pathlib import Path
import re
import sys


class ValidationError(ValueError):
    pass


STOP_WORDS = frozenset(
    "a an the is are was were be been do does did can could would should "
    "i me my we our you your it its to of for and or in on at with "
    "what how when where please tell about".split()
)


def terms(text):
    return set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)) - STOP_WORDS


def validate(data):
    """Single validation boundary used by library and CLI."""
    if not isinstance(data, dict):
        raise ValidationError("input must be an object")
    required = {"schema_version", "question", "knowledge_base"}
    if not required <= data.keys():
        raise ValidationError("missing required input fields")
    if data.keys() - required - {"min_coverage"}:
        raise ValidationError("unknown input fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    question = data["question"]
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValidationError("question must be nonblank text of at most 2000 characters")
    threshold = data.get("min_coverage", 0.6)
    if (type(threshold) not in (float, int) or not 0 < threshold <= 1
            or not math.isfinite(threshold)):
        raise ValidationError("min_coverage must be a finite number in (0, 1]")
    kb = data["knowledge_base"]
    if not isinstance(kb, list) or len(kb) > 1000:
        raise ValidationError("knowledge_base must be a list of at most 1000 documents")
    ids = set()
    for index, document in enumerate(kb):
        if not isinstance(document, dict) or set(document) != {"id", "title", "content"}:
            raise ValidationError("each document must have exactly id, title, content")
        for field, limit in (("id", 128), ("title", 500), ("content", 20000)):
            value = document[field]
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValidationError(f"document {index}: invalid {field}")
        if document["id"] != document["id"].strip():
            raise ValidationError("document ids must not have surrounding whitespace")
        if document["id"] in ids:
            raise ValidationError("document ids must be unique")
        ids.add(document["id"])
    return question, kb, float(threshold)


def envelope(status, *, answer=None, citations=None, retrieval=None, reason=None, errors=None):
    return {
        "schema_version": 1, "status": status, "answer": answer,
        "citations": citations or [], "retrieval": retrieval or [],
        "reason": reason, "errors": errors or [],
    }


def run(data):
    try:
        question, documents, threshold = validate(data)
    except ValidationError as exc:
        return envelope("error", reason="invalid_input", errors=[str(exc)])
    query = terms(question)
    if not query:
        return envelope("abstained", reason="no_searchable_terms")
    candidates = []
    for doc in documents:
        # Content-only retrieval prevents a matching title grounding an unrelated answer.
        overlap = query & terms(doc["content"])
        if overlap:
            candidates.append((len(overlap) / len(query), doc, sorted(overlap)))
    candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
    retrieval = [
        {"id": doc["id"], "coverage": round(score, 6), "matched_terms": matched}
        for score, doc, matched in candidates[:5]
    ]
    if not candidates or candidates[0][0] < threshold:
        return envelope("abstained", retrieval=retrieval, reason="insufficient_evidence")
    best_score, best, _ = candidates[0]
    tied = [doc for score, doc, _ in candidates if score == best_score]
    if any(doc["content"] != best["content"] for doc in tied):
        return envelope("abstained", retrieval=retrieval, reason="ambiguous_evidence")
    return envelope(
        "answered", answer=best["content"], retrieval=retrieval,
        citations=[{"id": best["id"], "title": best["title"], "quote": best["content"]}],
        reason="verbatim_source",
    )


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON number: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        result = envelope("error", reason="usage", errors=["usage: python -B implementation.py INPUT.json"])
    else:
        try:
            text = Path(argv[0]).read_text(encoding="utf-8")
            data = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
            result = run(data)
        except (OSError, UnicodeError, ValueError, RecursionError, OverflowError) as exc:
            result = envelope("error", reason="file_or_json_error", errors=[str(exc)])
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
