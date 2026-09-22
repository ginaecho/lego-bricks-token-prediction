"""Evidence-first analysis of explicit, source-anchored claims (Python stdlib).

Run: python implementation.py example_input.json
Use "-" instead of a filename to read JSON from stdin.

Input has exactly question and documents. Each document has id, title, text,
and claims; each claim has id, subject, value, and evidence (a literal excerpt).
IDs are globally unique within their respective document/claim namespaces.
Subject/value normalization is NFKC + casefold + whitespace collapse, not
semantic equivalence. Claims must have lexical subject/value support in their
excerpt; this is NOT an entailment or truth check.

analyze(data, synthesis_callback=None) is the injectable Python API. A callback
receives a detached copy of the evidence, groups, question, and synthesis plan.
It returns {"statements": [{"subject": ..., "value": ..., "claim_ids": [...],
"citations": [{"document_id": ..., "claim_id": ...}, ...]}, ...]}.
Every retrieved claim must occur exactly once, in one statement per normalized
subject/value pair. Only known claims and their own document citations pass.
Free-form model prose is deliberately forbidden: verified facts are rendered
locally, preventing uncited prose from bypassing structural validation.
No callback is invoked for empty evidence; the CLI never calls a model.
"""

import argparse
import copy
import json
import re
import sys
import unicodedata
from pathlib import Path


class ValidationError(ValueError):
    """Malformed input or unsupported synthesis."""


STOPWORDS = frozenset(
    "a an and are as at be by compare did do does for from how in is it of on "
    "or reported says the these this to was were what which with".split()
)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text):
    return set(re.findall(r"\w+", normalize(text))) - STOPWORDS


def _object(value, keys, location):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValidationError(f"{location} must be an object with exactly {sorted(keys)}")


def _list(value, location):
    if not isinstance(value, list):
        raise ValidationError(f"{location} must be a list")


def _string(value, location, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValidationError(f"{location} must be a {'possibly empty ' if allow_empty else 'nonempty '}string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError(f"{location} must contain valid Unicode") from exc


def _identifier(value, location):
    _string(value, location)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValidationError(f"{location} must be an ASCII identifier")


def _contains_phrase(excerpt, phrase):
    return re.search(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)",
                     normalize(excerpt)) is not None


def _validate_input(data):
    _object(data, {"question", "documents"}, "input")
    _string(data["question"], "question")
    _list(data["documents"], "documents")
    documents = set()
    claim_ids = set()
    records = []
    for index, document in enumerate(data["documents"]):
        location = f"documents[{index}]"
        _object(document, {"id", "title", "text", "claims"}, location)
        _identifier(document["id"], f"{location}.id")
        if document["id"] in documents:
            raise ValidationError(f"duplicate document id: {document['id']}")
        documents.add(document["id"])
        _string(document["title"], f"{location}.title")
        _string(document["text"], f"{location}.text", allow_empty=True)
        _list(document["claims"], f"{location}.claims")
        source_claims = set()
        for number, claim in enumerate(document["claims"]):
            claim_location = f"{location}.claims[{number}]"
            _object(claim, {"id", "subject", "value", "evidence"}, claim_location)
            _identifier(claim["id"], f"{claim_location}.id")
            if claim["id"] in claim_ids:
                raise ValidationError(f"duplicate claim id: {claim['id']}")
            claim_ids.add(claim["id"])
            for field in ("subject", "value", "evidence"):
                _string(claim[field], f"{claim_location}.{field}")
            start = document["text"].find(claim["evidence"])
            if start < 0:
                raise ValidationError(f"{claim_location}.evidence is not an exact source excerpt")
            for field in ("subject", "value"):
                if not _contains_phrase(claim["evidence"], claim[field]):
                    raise ValidationError(f"{claim_location}.{field} lacks lexical excerpt support")
            subject, value = normalize(claim["subject"]), normalize(claim["value"])
            duplicate_key = (subject, value, claim["evidence"])
            if duplicate_key in source_claims:
                raise ValidationError(f"duplicate source claim in {document['id']}")
            source_claims.add(duplicate_key)
            records.append({
                "claim_id": claim["id"],
                "document_id": document["id"],
                "document_title": document["title"],
                "subject": claim["subject"],
                "value": claim["value"],
                "normalized_subject": subject,
                "normalized_value": value,
                "quote": claim["evidence"],
                "source_span": {"start": start, "end": start + len(claim["evidence"])},
            })
    return records


def _retrieve(question, records):
    terms = tokens(question)
    selected_subjects = set()
    seeds = set()
    scored = []
    for record in records:
        subject_matches = terms & tokens(record["subject"])
        value_matches = terms & tokens(record["value"])
        quote_matches = terms & tokens(record["quote"])
        score = 4 * len(subject_matches) + 2 * len(value_matches) + len(quote_matches)
        if score:
            seeds.add(record["claim_id"])
            selected_subjects.add(record["normalized_subject"])
        scored.append((record, score, sorted(subject_matches | value_matches | quote_matches)))
    evidence = []
    for record, score, matched_terms in scored:
        if record["normalized_subject"] in selected_subjects:
            evidence.append({
                **record,
                "retrieval": {
                    "score": score,
                    "matched_terms": matched_terms,
                    "reason": "question_match" if record["claim_id"] in seeds else "same_subject_expansion",
                },
            })
    evidence.sort(key=lambda record: (
        record["normalized_subject"], record["normalized_value"],
        record["document_id"], record["claim_id"],
    ))
    return evidence, {
        "method": "lexical_overlap_then_complete_normalized_subject_expansion",
        "question_terms": sorted(terms),
        "candidate_claim_count": len(records),
        "matched_claim_count": len(seeds),
        "selected_claim_count": len(evidence),
    }


def _group(evidence):
    subjects = {}
    for record in evidence:
        subjects.setdefault(record["normalized_subject"], {}).setdefault(
            record["normalized_value"], []
        ).append(record)
    groups = []
    for subject, values in sorted(subjects.items()):
        alternatives = []
        for value, records in sorted(values.items()):
            alternatives.append({
                "value": value,
                "claim_ids": sorted(record["claim_id"] for record in records),
                "document_ids": sorted({record["document_id"] for record in records}),
            })
        document_count = len({record["document_id"] for records in values.values() for record in records})
        status = "disagreement" if len(values) > 1 else (
            "agreement" if document_count > 1 else "single_source"
        )
        groups.append({"subject": subject, "status": status, "alternatives": alternatives})
    return groups


def _planning(question, groups):
    if not groups:
        return ([{
            "kind": "no_relevant_evidence",
            "question": f"What source-backed explicit claims can address: {question}",
            "claim_ids": [],
        }], [{
            "action": "request_evidence",
            "subject": None,
            "claim_ids": [],
            "instruction": "Do not answer from absent evidence; request relevant source claims.",
        }])
    unresolved = []
    plan = []
    for group in groups:
        subject = group["subject"]
        claim_ids = sorted(cid for alt in group["alternatives"] for cid in alt["claim_ids"])
        status = group["status"]
        if status == "disagreement":
            unresolved.append({
                "kind": "conflicting_values",
                "question": f"What explains or resolves the different values for {subject}?",
                "claim_ids": claim_ids,
            })
            instruction = "Present every alternative with citations; do not select a winner by source count."
        elif status == "single_source":
            unresolved.append({
                "kind": "needs_corroboration",
                "question": f"What additional independent evidence can corroborate {subject}?",
                "claim_ids": claim_ids,
            })
            instruction = "Report as a single-source assertion, not an established fact."
        else:
            instruction = "Report matching assertions across documents; agreement does not establish truth."
        plan.append({
            "action": f"report_{status}",
            "subject": subject,
            "claim_ids": claim_ids,
            "instruction": instruction,
        })
    unresolved.append({
        "kind": "answer_coverage_unverified",
        "question": f"Do these lexically relevant claims fully answer: {question}",
        "claim_ids": [],
    })
    return unresolved, plan


def _statements(groups, evidence):
    known = {record["claim_id"]: record for record in evidence}
    return [{
        "subject": group["subject"],
        "value": alt["value"],
        "claim_ids": alt["claim_ids"][:],
        "citations": [{
            "document_id": known[cid]["document_id"], "claim_id": cid,
        } for cid in alt["claim_ids"]],
    } for group in groups for alt in group["alternatives"]]


def _validate_synthesis(response, evidence):
    _object(response, {"statements"}, "synthesis")
    _list(response["statements"], "synthesis.statements")
    known = {record["claim_id"]: record for record in evidence}
    used = set()
    pairs = set()
    canonical = []
    for index, statement in enumerate(response["statements"]):
        location = f"synthesis.statements[{index}]"
        _object(statement, {"subject", "value", "claim_ids", "citations"}, location)
        for field in ("subject", "value"):
            _string(statement[field], f"{location}.{field}")
        subject, value = normalize(statement["subject"]), normalize(statement["value"])
        if (subject, value) in pairs:
            raise ValidationError("duplicate synthesis subject/value statement")
        pairs.add((subject, value))
        _list(statement["claim_ids"], f"{location}.claim_ids")
        if not statement["claim_ids"]:
            raise ValidationError("synthesis statement needs at least one known claim")
        ids = set()
        for cid in statement["claim_ids"]:
            _identifier(cid, f"{location}.claim_ids item")
            if cid not in known:
                raise ValidationError(f"unknown or unretrieved synthesis claim: {cid}")
            if cid in used:
                raise ValidationError(f"duplicate synthesis claim: {cid}")
            record = known[cid]
            if (subject, value) != (record["normalized_subject"], record["normalized_value"]):
                raise ValidationError(f"unsupported synthesis assertion for claim: {cid}")
            used.add(cid)
            ids.add(cid)
        _list(statement["citations"], f"{location}.citations")
        cited = set()
        for citation in statement["citations"]:
            _object(citation, {"document_id", "claim_id"}, f"{location}.citation")
            _identifier(citation["document_id"], f"{location}.citation.document_id")
            _identifier(citation["claim_id"], f"{location}.citation.claim_id")
            cid = citation["claim_id"]
            if cid not in ids or known[cid]["document_id"] != citation["document_id"]:
                raise ValidationError("citation does not identify the asserted claim's source")
            if cid in cited:
                raise ValidationError("duplicate synthesis citation")
            cited.add(cid)
        if cited != ids:
            raise ValidationError("every synthesis claim needs its own citation")
        canonical.append({
            "subject": subject, "value": value, "claim_ids": sorted(ids),
            "citations": [
                {"document_id": known[cid]["document_id"], "claim_id": cid}
                for cid in sorted(ids)
            ],
        })
    if used != set(known):
        raise ValidationError("synthesis must preserve every retrieved claim, including disagreements")
    return sorted(canonical, key=lambda item: (item["subject"], item["value"]))


def _render(groups, statements):
    if not groups:
        return "No relevant explicit claims were found. The question remains unanswered."
    by_subject = {}
    for statement in statements:
        citations = ", ".join(
            f"{item['document_id']}/{item['claim_id']}" for item in statement["citations"]
        )
        by_subject.setdefault(statement["subject"], []).append(
            f"{statement['value']} [{citations}]"
        )
    labels = {
        "agreement": "Documents agree (not independently verified)",
        "disagreement": "Sources disagree (unresolved)",
        "single_source": "Single-source assertion (uncorroborated)",
    }
    return "\n".join(
        f"{labels[group['status']]} on {group['subject']}: "
        + "; ".join(by_subject[group["subject"]]) + "."
        for group in groups
    )


def analyze(data, synthesis_callback=None):
    """Validate, retrieve, group, plan, and optionally validate callback synthesis."""
    if synthesis_callback is not None and not callable(synthesis_callback):
        raise ValidationError("synthesis_callback must be callable")
    records = _validate_input(data)
    question = data["question"].strip()
    evidence, retrieval = _retrieve(question, records)
    groups = _group(evidence)
    unresolved, plan = _planning(question, groups)
    statements = _statements(groups, evidence)
    mode = "deterministic"
    if synthesis_callback is not None and evidence:
        request = copy.deepcopy({
            "question": question, "evidence": evidence,
            "grouped_claims": groups, "synthesis_plan": plan,
        })
        try:
            response = synthesis_callback(request)
        except Exception as exc:
            raise ValidationError("synthesis callback failed") from exc
        statements = _validate_synthesis(response, evidence)
        mode = "validated_callback"
    return {
        "question": question,
        "retrieval": retrieval,
        "evidence": evidence,
        "grouped_claims": groups,
        "unresolved_questions": unresolved,
        "synthesis_plan": plan,
        "synthesis": {
            "mode": mode, "statements": statements, "text": _render(groups, statements),
        },
        "limitations": [
            "Explicit claims and literal excerpts only; no automatic claim extraction or semantic entailment.",
            "Lexical relevance is not answer completeness; absence of retrieval is not absence of truth.",
            "NFKC/case/whitespace normalization only; no synonym, unit, time, or entity resolution.",
            "Different normalized values flag potential disagreement, not proven logical contradiction.",
            "Distinct document IDs do not prove independent sources; agreement is not verification.",
        ],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError(f"nonstandard JSON number: {value}")


def load_json(text):
    try:
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except ValidationError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="UTF-8 JSON input file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        result = analyze(load_json(text))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))
    except (ValidationError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
