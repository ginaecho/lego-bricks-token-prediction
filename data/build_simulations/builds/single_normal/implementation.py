"""Bounded, offline extractive research. Run: python implementation.py input.json.

Input: {"question": str, "documents": [{"id": str, "text": str}]}.
Offsets are half-open Python Unicode character offsets into original document text.
Only exact source quotations are findings; lexical gaps are not proof of absence.
No dependency, network, model, or provider is used. An optional Python callback
may select/reorder grounded quotes, but cannot introduce unverified paraphrases.
"""

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter


MAX_DOCUMENTS = 64
MAX_TOTAL_CHARACTERS = 200_000
MAX_QUESTION_CHARACTERS = 2_000
MAX_QUERY_TERMS = 64
MAX_INPUT_BYTES = 2_000_000
PASSAGE_CHARACTERS = 600
PASSAGE_OVERLAP = 100
STOP = frozenset(
    "a an and are as at be been by can could did do does for from had has have "
    "how i in is it its of on or that the their there these this to was were "
    "what when where which who why will with would".split()
)
NEGATIVE = frozenset(("not", "no", "never", "cannot", "without", "denied"))


class ValidationError(ValueError):
    """Invalid request or ungrounded callback response."""


def tokens(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def content_tokens(text):
    return [term for term in tokens(text) if term not in STOP]


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"question", "documents"}:
        raise ValidationError("request must contain exactly question and documents")
    question, documents = request["question"], request["documents"]
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("question must be a nonempty string")
    if len(question) > MAX_QUESTION_CHARACTERS:
        raise ValidationError("question exceeds character limit")
    if not isinstance(documents, list) or len(documents) > MAX_DOCUMENTS:
        raise ValidationError("documents must be a list of at most 64 documents")
    seen, total = set(), 0
    for document in documents:
        if not isinstance(document, dict) or set(document) != {"id", "text"}:
            raise ValidationError("each document must contain exactly id and text")
        identifier, text = document["id"], document["text"]
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128:
            raise ValidationError("document id must be a nonempty string of at most 128 characters")
        if identifier in seen:
            raise ValidationError("duplicate document id: " + identifier)
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("document text must be a nonempty string")
        seen.add(identifier)
        total += len(text)
        if total > MAX_TOTAL_CHARACTERS:
            raise ValidationError("combined document text exceeds character limit")
    query_terms = list(dict.fromkeys(content_tokens(question)))
    if len(query_terms) > MAX_QUERY_TERMS:
        raise ValidationError("question exceeds 64 distinct content terms")
    return question, documents, query_terms


def passages(document):
    """Sentence-like spans, splitting long spans with a bounded overlap."""
    text = document["text"]
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)", text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end:
            stop = min(start + PASSAGE_CHARACTERS, end)
            yield {
                "source_id": document["id"],
                "start": start,
                "end": stop,
                "quote": text[start:stop],
            }
            if stop == end:
                break
            start = stop - PASSAGE_OVERLAP


def overlapping(left, right):
    if left["source_id"] != right["source_id"]:
        return False
    overlap = min(left["end"], right["end"]) - max(left["start"], right["start"])
    return overlap > 0


def conflict_candidates(findings):
    """Conservative lexical warnings, not a semantic contradiction adjudicator."""
    warnings = []
    for i, left in enumerate(findings):
        left_terms = set(content_tokens(left["text"]))
        left_core = {t for t in left_terms if t not in NEGATIVE and not t.isdigit()}
        for j in range(i + 1, len(findings)):
            right = findings[j]
            right_terms = set(content_tokens(right["text"]))
            right_core = {t for t in right_terms if t not in NEGATIVE and not t.isdigit()}
            shared = left_core & right_core
            union = left_core | right_core
            if len(shared) < 2 or len(shared) / max(1, len(union)) < 0.6:
                continue
            polarity = bool(left_terms & NEGATIVE) != bool(right_terms & NEGATIVE)
            left_numbers = {t for t in left_terms if t.isdigit()}
            right_numbers = {t for t in right_terms if t.isdigit()}
            numeric = bool(left_numbers and right_numbers and left_numbers != right_numbers)
            if polarity or numeric:
                warnings.append({
                    "finding_indices": [i, j],
                    "reason": "polarity_difference" if polarity else "numeric_difference",
                    "status": "potential_conflict_requires_review",
                })
    return warnings


def validated_synthesis(callback, result, documents):
    response = callback(copy.deepcopy(result))
    if not isinstance(response, dict) or set(response) != {"findings"}:
        raise ValidationError("callback must return exactly findings")
    items = response["findings"]
    if not isinstance(items, list) or len(items) > 12:
        raise ValidationError("callback findings must be a list of at most 12 items")
    sources = {d["id"]: d["text"] for d in documents}
    evidence = [c for f in result["findings"] for c in f["citations"]]
    for item in items:
        if not isinstance(item, dict) or set(item) != {"text", "citations"}:
            raise ValidationError("callback finding must contain exactly text and citations")
        citations = item["citations"]
        if not isinstance(citations, list) or not 1 <= len(citations) <= 12:
            raise ValidationError("callback finding needs 1 to 12 citations")
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {
                "source_id", "start", "end", "quote"
            }:
                raise ValidationError("malformed callback citation")
            source_id, start, end, quote = (
                citation["source_id"], citation["start"], citation["end"], citation["quote"]
            )
            if not isinstance(source_id, str) or source_id not in sources:
                raise ValidationError("unknown callback source id")
            if type(start) is not int or type(end) is not int:
                raise ValidationError("citation offsets must be integers")
            if not 0 <= start < end <= len(sources[source_id]):
                raise ValidationError("citation offsets out of range")
            if not isinstance(quote, str) or quote != sources[source_id][start:end]:
                raise ValidationError("callback quote does not match exact source span")
            if not any(
                c["source_id"] == source_id and c["start"] <= start and end <= c["end"]
                for c in evidence
            ):
                raise ValidationError("callback citation is outside retrieved evidence")
        if not isinstance(item["text"], str) or item["text"] != " ".join(
            c["quote"] for c in citations
        ):
            raise ValidationError("callback text must be its exact quotes joined by spaces")
    return copy.deepcopy(items)


def research(request, *, max_findings=6, synthesis_callback=None):
    if type(max_findings) is not int or not 1 <= max_findings <= 12:
        raise ValidationError("max_findings must be an integer from 1 to 12")
    question, documents, query = validate_request(request)
    candidates = list(c for document in documents for c in passages(document))
    counts = [Counter(content_tokens(c["quote"])) for c in candidates]
    sizes = [sum(c.values()) for c in counts]
    mean_size = sum(sizes) / max(1, len(sizes))
    frequencies = {term: sum(term in c for c in counts) for term in query}
    ranked = []
    for index, (citation, count, size) in enumerate(zip(candidates, counts, sizes)):
        matched = [term for term in query if count[term]]
        if not matched:
            continue
        score = 0.0
        for term in matched:
            idf = math.log(1 + (len(candidates) - frequencies[term] + 0.5) /
                           (frequencies[term] + 0.5))
            tf = count[term]
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * size / mean_size))
        score *= 1 + len(matched) / max(1, len(query))
        ranked.append((score, index, citation, matched))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    findings, canonical, duplicate_count, overlap_count = [], {}, 0, 0
    for score, _, citation, matched in ranked:
        key = " ".join(citation["quote"].casefold().split())
        if key in canonical:
            canonical[key]["citations"].append(citation)
            duplicate_count += 1
            continue
        if any(overlapping(citation, c) for f in findings for c in f["citations"]):
            overlap_count += 1
            continue
        if len(findings) >= max_findings:
            continue
        finding = {
            "text": citation["quote"],
            "citations": [citation],
            "score": round(score, 6),
            "matched_terms": matched,
        }
        findings.append(finding)
        canonical[key] = finding
    selected_terms = {t for f in findings for t in f["matched_terms"]}
    missing = [t for t in query if frequencies[t] == 0]
    uncovered = [t for t in query if frequencies[t] and t not in selected_terms]
    result = {
        "question": question,
        "status": "evidence_found" if findings else "no_matching_evidence",
        "method": "bounded_bm25_extractive",
        "findings": findings,
        "missing_evidence": {
            "terms_absent_from_documents": missing,
            "terms_not_in_selected_findings": uncovered,
            "reason": (
                "no_content_query_terms" if not query else
                "no_lexical_matches" if not findings else
                "partial_lexical_coverage" if missing or uncovered else
                "no_lexical_gap_detected"
            ),
            "caveat": "Lexical coverage does not establish that the question is answered.",
        },
        "conflicts": conflict_candidates(findings),
        "statistics": {
            "documents": len(documents),
            "candidate_passages": len(candidates),
            "matching_passages": len(ranked),
            "duplicate_passages_merged": duplicate_count,
            "overlapping_passages_suppressed": overlap_count,
            "finding_limit_reached": len(findings) == max_findings,
        },
    }
    if synthesis_callback is not None:
        if not callable(synthesis_callback):
            raise ValidationError("synthesis_callback must be callable")
        result["synthesized_findings"] = validated_synthesis(synthesis_callback, result, documents)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="JSON file; otherwise read JSON from stdin")
    parser.add_argument("--max-findings", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        if args.input:
            with open(args.input, "rb") as handle:
                raw = handle.read(MAX_INPUT_BYTES + 1)
        else:
            raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValidationError("JSON input exceeds byte limit")
        request = json.loads(raw.decode("utf-8"))
        result = research(request, max_findings=args.max_findings)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
