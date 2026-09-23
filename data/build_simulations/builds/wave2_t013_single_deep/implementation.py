"""Bounded, deterministic evidence synthesis; no retrieval or provider calls."""

import json
import sys
from pathlib import Path


MAX_FILE_BYTES = 2_000_000
QUALITIES = {"low": 1, "medium": 2, "high": 3}
STANCES = {"support", "oppose", "neutral"}
STATUSES = {"supported", "opposed", "contested", "context_dependent", "insufficient"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def record(value, keys, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(keys), f"{path}: expected keys {', '.join(sorted(keys))}")


def text(value, path, maximum=10000):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonblank string")
    require(len(value) <= maximum, f"{path}: exceeds {maximum} characters")


def sequence(value, path, maximum, minimum=0):
    require(isinstance(value, list), f"{path}: expected array")
    require(minimum <= len(value) <= maximum, f"{path}: invalid array length")


def validate(value, kind="input"):
    """Shared boundary validation for requests and generated result envelopes."""
    if kind == "output":
        record(value, {"status", "synthetic", "research_question", "summary",
                       "findings", "disagreements", "unresolved_questions", "limitations"}, "$")
        require(value["status"] == "ok", "output.status: expected ok")
        require(value["synthetic"] is True, "output.synthetic: must be true")
        text(value["research_question"], "output.research_question")
        record(value["summary"], {"documents", "claims", "disagreements", "open_questions"}, "summary")
        for count in value["summary"].values():
            require(type(count) is int and count >= 0, "summary: invalid count")
        sequence(value["findings"], "findings", 100, 1)
        for finding in value["findings"]:
            record(finding, {"claim_id", "statement", "status", "contexts", "citations"}, "finding")
            text(finding["claim_id"], "finding.claim_id", 100)
            text(finding["statement"], "finding.statement")
            require(isinstance(finding["status"], str) and finding["status"] in STATUSES,
                    "finding.status: invalid")
            sequence(finding["contexts"], "contexts", 500)
            for context in finding["contexts"]:
                record(context, {"context", "status", "source_counts", "quality_weight_totals"}, "context")
                text(context["context"], "context.context", 200)
                require(isinstance(context["status"], str) and context["status"] in STATUSES,
                        "context.status: invalid")
                for field in ("source_counts", "quality_weight_totals"):
                    record(context[field], STANCES, field)
                    for count in context[field].values():
                        require(type(count) is int and count >= 0, f"{field}: invalid count")
            sequence(finding["citations"], "citations", 500)
            for citation in finding["citations"]:
                record(citation, {"document_id", "title", "quote", "stance", "quality", "context"}, "citation")
                for key in ("document_id", "title", "quote", "context"):
                    text(citation[key], f"citation.{key}", 100000)
                require(isinstance(citation["stance"], str) and citation["stance"] in STANCES,
                        "citation.stance: invalid")
                require(isinstance(citation["quality"], str) and citation["quality"] in QUALITIES,
                        "citation.quality: invalid")
        sequence(value["disagreements"], "disagreements", 50000)
        for item in value["disagreements"]:
            record(item, {"claim_id", "context", "supporting_documents", "opposing_documents"}, "disagreement")
            for key in ("claim_id", "context"):
                text(item[key], f"disagreement.{key}")
            for key in ("supporting_documents", "opposing_documents"):
                sequence(item[key], key, 100, 1)
                for source in item[key]:
                    text(source, key, 100)
        sequence(value["unresolved_questions"], "unresolved_questions", 200)
        for item in value["unresolved_questions"]:
            record(item, {"id", "question", "claim_ids", "reason"}, "unresolved_question")
            for key in ("id", "question", "reason"):
                text(item[key], f"unresolved_question.{key}", 11000)
            sequence(item["claim_ids"], "claim_ids", 100)
            for claim_id in item["claim_ids"]:
                text(claim_id, "claim_id", 100)
        sequence(value["limitations"], "limitations", 20, 1)
        for limitation in value["limitations"]:
            text(limitation, "limitation")
        require(value["summary"]["claims"] == len(value["findings"]), "summary.claims: mismatch")
        require(value["summary"]["disagreements"] == len(value["disagreements"]), "summary.disagreements: mismatch")
        require(value["summary"]["open_questions"] == len(value["unresolved_questions"]), "summary.open_questions: mismatch")
        return value

    require(kind == "input", "unknown validation boundary")
    record(value, {"schema_version", "synthetic", "research_question", "documents", "claims", "questions"}, "$")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "schema_version: expected integer 1")
    require(value["synthetic"] is True, "synthetic: fixtures must be explicitly labeled true")
    text(value["research_question"], "research_question")
    sequence(value["documents"], "documents", 100)
    documents = {}
    for document in value["documents"]:
        record(document, {"id", "title", "text"}, "document")
        text(document["id"], "document.id", 100)
        text(document["title"], "document.title", 1000)
        text(document["text"], "document.text", 100000)
        require(document["id"] not in documents, "duplicate document id")
        documents[document["id"]] = document
    sequence(value["claims"], "claims", 100, 1)
    claim_ids = set()
    for claim in value["claims"]:
        record(claim, {"id", "statement", "evidence"}, "claim")
        text(claim["id"], "claim.id", 100)
        text(claim["statement"], "claim.statement")
        require(claim["id"] not in claim_ids, "duplicate claim id")
        claim_ids.add(claim["id"])
        sequence(claim["evidence"], "claim.evidence", 500)
        for evidence in claim["evidence"]:
            record(evidence, {"document_id", "quote", "stance", "quality", "context"}, "evidence")
            text(evidence["document_id"], "evidence.document_id", 100)
            text(evidence["quote"], "evidence.quote")
            text(evidence["context"], "evidence.context", 200)
            require(isinstance(evidence["stance"], str) and evidence["stance"] in STANCES,
                    "evidence.stance: invalid")
            require(isinstance(evidence["quality"], str) and evidence["quality"] in QUALITIES,
                    "evidence.quality: invalid")
            require(evidence["document_id"] in documents, "evidence: unknown document")
            require(evidence["quote"] in documents[evidence["document_id"]]["text"],
                    "evidence.quote: must be an exact substring of its document")
    sequence(value["questions"], "questions", 100)
    question_ids = set()
    for question in value["questions"]:
        record(question, {"id", "question", "claim_ids"}, "question")
        text(question["id"], "question.id", 100)
        text(question["question"], "question.question")
        require(question["id"] not in question_ids, "duplicate question id")
        question_ids.add(question["id"])
        sequence(question["claim_ids"], "question.claim_ids", 100)
        for claim_id in question["claim_ids"]:
            text(claim_id, "question.claim_id", 100)
            require(claim_id in claim_ids, "question: unknown claim")
        require(len(set(question["claim_ids"])) == len(question["claim_ids"]),
                "question: duplicate claim reference")
    return value


def stance_status(stances):
    support, oppose = "support" in stances, "oppose" in stances
    if support and oppose:
        return "contested"
    if support:
        return "supported"
    if oppose:
        return "opposed"
    return "insufficient"


def synthesize(request):
    validate(request)
    documents = {document["id"]: document for document in request["documents"]}
    findings, disagreements, unresolved = [], [], []
    for claim in sorted(request["claims"], key=lambda item: item["id"]):
        unique = {}
        for evidence in claim["evidence"]:
            key = tuple(evidence[field] for field in ("context", "document_id", "stance", "quote"))
            previous = unique.get(key)
            if previous is None or QUALITIES[evidence["quality"]] > QUALITIES[previous["quality"]]:
                unique[key] = evidence
        citations = [
            dict(unique[key], title=documents[unique[key]["document_id"]]["title"])
            for key in sorted(unique)
        ]
        contexts = []
        for context_name in sorted({item["context"] for item in citations}):
            group = [item for item in citations if item["context"] == context_name]
            sources = {stance: {} for stance in sorted(STANCES)}
            for evidence in group:
                scores = sources[evidence["stance"]]
                doc_id = evidence["document_id"]
                scores[doc_id] = max(scores.get(doc_id, 0), QUALITIES[evidence["quality"]])
            status = stance_status({stance for stance, scores in sources.items() if scores})
            contexts.append({
                "context": context_name, "status": status,
                "source_counts": {stance: len(scores) for stance, scores in sources.items()},
                "quality_weight_totals": {stance: sum(scores.values()) for stance, scores in sources.items()},
            })
            if status == "contested":
                disagreements.append({
                    "claim_id": claim["id"], "context": context_name,
                    "supporting_documents": sorted(sources["support"]),
                    "opposing_documents": sorted(sources["oppose"]),
                })
        states = {context["status"] for context in contexts}
        if "contested" in states:
            status = "contested"
        elif "supported" in states and "opposed" in states:
            status = "context_dependent"
        else:
            status = stance_status({item["stance"] for item in citations})
        findings.append({"claim_id": claim["id"], "statement": claim["statement"],
                         "status": status, "contexts": contexts, "citations": citations})
        if status in {"contested", "context_dependent", "insufficient"}:
            reason = {
                "contested": "Opposing evidence exists in the same supplied context; weights do not settle truth.",
                "context_dependent": "Evidence direction differs across supplied contexts; generalization remains unresolved.",
                "insufficient": "No directional evidence is supplied; neutral citations cannot resolve the claim.",
            }[status]
            unresolved.append({"id": "generated:" + claim["id"],
                               "question": "What further evidence would resolve: " + claim["statement"],
                               "claim_ids": [claim["id"]], "reason": reason})
    for question in sorted(request["questions"], key=lambda item: item["id"]):
        unresolved.append({"id": "requested:" + question["id"], "question": question["question"],
                           "claim_ids": sorted(question["claim_ids"]),
                           "reason": "Explicit follow-up question; claim-level stance aggregation does not establish its answer."})
    result = {
        "status": "ok", "synthetic": True, "research_question": request["research_question"],
        "summary": {"documents": len(documents), "claims": len(findings),
                    "disagreements": len(disagreements), "open_questions": len(unresolved)},
        "findings": findings, "disagreements": disagreements, "unresolved_questions": unresolved,
        "limitations": [
            "Synthetic fixture synthesis only; no web retrieval, semantic extraction, or fact verification.",
            "Claims, stances, quality labels, and exact context labels are supplied by the caller.",
            "Quality weights are ordinal descriptors, not probabilities or a truth vote.",
            "Repeated citations count once per document, stance, and context; distinct documents may still be dependent.",
            "A supported or opposed status describes supplied evidence only, not a proven conclusion.",
        ],
    }
    return validate(result, "output")


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
        require(len(raw) <= MAX_FILE_BYTES, "input file exceeds 2000000 bytes")
        request = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys,
                             parse_constant=reject_constant)
        result = synthesize(request)
    except (OSError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
