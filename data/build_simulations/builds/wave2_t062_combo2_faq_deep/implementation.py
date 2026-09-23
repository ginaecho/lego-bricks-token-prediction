"""Synthetic, deterministic FAQ -> evidence synthesis reference pipeline."""

import copy
import json
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(expected), location + " has missing or unknown fields")


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), location + " must be nonempty text")


def string_list(value, location):
    require(isinstance(value, list), location + " must be a list")
    for item in value:
        text(item, location)
    require(len(value) == len(set(value)), location + " must contain unique strings")


def validate(kind, value, context=None):
    """Single validation boundary used for input, handoff, injected answer and output."""
    if kind == "input":
        fields(value, ("schema_version", "synthetic", "question", "topics",
                       "min_sources", "documents"), "input")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "schema_version must be 1")
        require(value["synthetic"] is True, "fixtures must be labeled synthetic")
        text(value["question"], "question")
        string_list(value["topics"], "topics")
        require(bool(value["topics"]), "topics cannot be empty")
        require(type(value["min_sources"]) is int and 1 <= value["min_sources"] <= 100,
                "min_sources must be an integer from 1 to 100")
        require(isinstance(value["documents"], list), "documents must be a list")
        ids = []
        for doc in value["documents"]:
            fields(doc, ("id", "title", "claims"), "document")
            text(doc["id"], "document id")
            text(doc["title"], "document title")
            ids.append(doc["id"])
            require(isinstance(doc["claims"], list), "claims must be a list")
            claim_ids = []
            for claim in doc["claims"]:
                fields(claim, ("id", "topic", "position", "statement"), "claim")
                for key in claim:
                    text(claim[key], "claim " + key)
                claim_ids.append(claim["id"])
            require(len(claim_ids) == len(set(claim_ids)), "duplicate claim id")
        require(len(ids) == len(set(ids)), "duplicate document id")
    elif kind == "answer":
        fields(value, ("status", "reason", "citations", "answer"), "answer")
        require(value["status"] in ("answered", "abstained"), "invalid answer status")
        require(isinstance(value["reason"], str), "reason must be text")
        require(isinstance(value["answer"], str), "answer must be text")
        require(isinstance(value["citations"], list), "citations must be a list")
        evidence = {(e["document_id"], e["claim_id"]): e for e in context["evidence"]}
        seen = set()
        for citation in value["citations"]:
            fields(citation, ("document_id", "claim_id"), "citation")
            text(citation["document_id"], "citation document_id")
            text(citation["claim_id"], "citation claim_id")
            key = (citation["document_id"], citation["claim_id"])
            require(key in evidence and key not in seen, "unknown or duplicate citation")
            seen.add(key)
        if value["status"] == "abstained":
            require(value["answer"] == "" and not value["citations"] and value["reason"].strip(),
                    "abstention requires a reason and no answer or citations")
        else:
            require(not context["blocking_reasons"], "answer must abstain on blocked evidence")
            require(value["reason"] == "", "answered response cannot have abstention reason")
            require(seen == set(evidence), "answer must cite all retrieved evidence")
            expected = "\n".join(evidence[(c["document_id"], c["claim_id"])]["statement"]
                                 for c in value["citations"])
            require(value["answer"] == expected and bool(expected),
                    "answer must exactly quote its evidence in citation order")
    elif kind == "faq":
        fields(value, ("question", "topics", "min_sources", "evidence", "blocking_reasons",
                       "response"), "faq")
        require(context is not None, "faq validation requires original input")
        validate("input", context)
        expected = retrieve(context)
        for key in ("question", "topics", "min_sources"):
            require(value[key] == context[key], "FAQ changed input " + key)
        require(value["evidence"] == expected, "FAQ evidence differs from source claims")
        require(value["blocking_reasons"] == blockers(context, expected),
                "FAQ blocking reasons differ from evidence")
        validate("answer", value["response"], value)
    elif kind == "deep":
        require(value == synthesize(context), "deep synthesis differs from validated FAQ evidence")
    elif kind == "output":
        fields(value, ("schema_version", "synthetic", "status", "faq", "deep"), "output")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1
                and value["synthetic"] is True and value["status"] == "ok",
                "invalid output metadata")
        validate("faq", value["faq"], context)
        validate("deep", value["deep"], value["faq"])
    else:
        raise ValidationError("unknown validation schema")
    return value


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def retrieve(data):
    """Rank by lexical overlap; exact requested topic is the relevance gate."""
    query = tokens(data["question"])
    result = []
    for doc in data["documents"]:
        for claim in doc["claims"]:
            if claim["topic"] not in data["topics"]:
                continue
            score = len(query & tokens(doc["title"] + " " + claim["statement"]))
            result.append({"document_id": doc["id"], "claim_id": claim["id"],
                           "topic": claim["topic"], "position": claim["position"],
                           "statement": claim["statement"], "score": score})
    return sorted(result, key=lambda e: (-e["score"], e["document_id"], e["claim_id"]))


def blockers(data, evidence):
    reasons = []
    for topic in data["topics"]:
        matches = [e for e in evidence if e["topic"] == topic]
        if not matches:
            reasons.append("No evidence for topic: " + topic)
        elif len({e["document_id"] for e in matches}) < data["min_sources"]:
            reasons.append("Insufficient distinct sources for topic: " + topic)
        if len({e["position"] for e in matches}) > 1:
            reasons.append("Conflicting positions for topic: " + topic)
    return reasons


def citation(evidence):
    return {key: evidence[key] for key in ("document_id", "claim_id")}


def faq_stage(data, answerer=None):
    validate("input", data)
    evidence = retrieve(data)
    faq = {key: copy.deepcopy(data[key]) for key in ("question", "topics", "min_sources")}
    faq.update(evidence=evidence, blocking_reasons=blockers(data, evidence))
    if answerer is not None:
        # The injected callable cannot mutate the actual handoff or source input.
        try:
            response = answerer(copy.deepcopy(faq))
        except Exception as exc:
            raise ValidationError("injected answerer failed") from exc
    elif faq["blocking_reasons"]:
        response = {"status": "abstained", "reason": "; ".join(faq["blocking_reasons"]),
                    "answer": "", "citations": []}
    else:
        response = {"status": "answered", "reason": "",
                    "answer": "\n".join(e["statement"] for e in evidence),
                    "citations": [citation(e) for e in evidence]}
    faq["response"] = response
    return validate("faq", faq, data)


def synthesize(faq):
    findings, disagreements, unresolved = [], [], []
    for topic in faq["topics"]:
        evidence = [e for e in faq["evidence"] if e["topic"] == topic]
        positions = sorted({e["position"] for e in evidence})
        variants = []
        for position in positions:
            matches = [e for e in evidence if e["position"] == position]
            variants.append({"position": position,
                             "statements": sorted({e["statement"] for e in matches}),
                             "citations": [citation(e) for e in matches],
                             "source_count": len({e["document_id"] for e in matches})})
        findings.append({"topic": topic, "positions": variants,
                         "source_count": len({e["document_id"] for e in evidence})})
        if len(positions) > 1:
            disagreements.append({"topic": topic, "positions": variants})
            unresolved.append("Which position is authoritative for topic: " + topic + "?")
        if not evidence:
            unresolved.append("What evidence addresses topic: " + topic + "?")
        elif len({e["document_id"] for e in evidence}) < faq["min_sources"]:
            unresolved.append("Can additional distinct sources corroborate topic: " + topic + "?")
    if faq["response"]["status"] == "abstained":
        unresolved.append("FAQ abstained: " + faq["response"]["reason"])
    return {"faq_status": faq["response"]["status"],
            "faq_citations": copy.deepcopy(faq["response"]["citations"]),
            "findings": findings, "disagreements": disagreements,
            "unresolved_questions": unresolved}


def deep_stage(faq, original_input):
    validate("faq", faq, original_input)
    return validate("deep", synthesize(faq), faq)


def run_pipeline(data, answerer=None):
    validate("input", data)
    faq = faq_stage(data, answerer)
    result = {"schema_version": 1, "synthetic": True, "status": "ok",
              "faq": faq, "deep": deep_stage(faq, data)}
    return validate("output", result, data)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonstandard JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py input.json")
        with open(args[0], encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
