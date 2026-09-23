"""Synthetic, deterministic FAQ -> guided setup -> document review reference.

Run: python -B implementation.py example_input.json
All references and evidence are validated before execution. No external services.
"""

import json
import re
import sys
from pathlib import Path


VERSION = "1.0"
DISCLAIMER = (
    "Synthetic reference review only. Evidence matches are not certification, "
    "legal advice, or a determination of compliance."
)


class ValidationError(ValueError):
    pass


def require(condition, path, message):
    if not condition:
        raise ValidationError(f"{path}: {message}")


def shape(value, keys, path):
    require(type(value) is dict, path, "expected object")
    require(set(value) == set(keys), path, "unexpected or missing fields")


def text(value, path):
    require(type(value) is str and bool(value.strip()), path, "expected nonempty string")


def sequence(value, path):
    require(type(value) is list, path, "expected array")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def identifier(value, path):
    text(value, path)
    require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value) is not None,
            path, "invalid identifier")


def id_list(value, path, allowed=None):
    sequence(value, path)
    for item in value:
        identifier(item, path)
        if allowed is not None:
            require(item in allowed, path, f"unknown reference {item}")
    require(len(set(value)) == len(value), path, "duplicate references")


def term_list(value, path):
    sequence(value, path)
    require(bool(value), path, "at least one term required")
    for item in value:
        text(item, path)
        require(re.fullmatch(r"[a-z0-9]+", item) is not None,
                path, "terms must be lowercase ASCII tokens")
    require(len(set(value)) == len(value), path, "duplicate terms")


def catalog(value, keys, path):
    sequence(value, path)
    index = {}
    for i, item in enumerate(value):
        item_path = f"{path}[{i}]"
        shape(item, keys, item_path)
        identifier(item["id"], item_path + ".id")
        require(item["id"] not in index, item_path, "duplicate identifier")
        index[item["id"]] = item
    return index


def validate_input(value):
    shape(value, ("schema_version", "synthetic", "query", "knowledge_base",
                  "setup", "requirements", "documents", "evidence"), "$")
    require(value["schema_version"] == VERSION, "$.schema_version", "unsupported version")
    require(value["synthetic"] is True, "$.synthetic", "synthetic fixtures required")
    text(value["query"], "$.query")
    kb = catalog(value["knowledge_base"],
                 ("id", "text", "match_terms", "step_ids"), "$.knowledge_base")
    steps = catalog(value["setup"],
                    ("id", "title", "prerequisites", "completed", "requirement_ids"),
                    "$.setup")
    requirements = catalog(value["requirements"], ("id", "description", "required_terms"),
                           "$.requirements")
    documents = catalog(value["documents"], ("id", "text"), "$.documents")
    evidence = catalog(value["evidence"],
                       ("id", "requirement_id", "document_id", "quote", "start"),
                       "$.evidence")
    for key, item in requirements.items():
        text(item["description"], f"requirements.{key}.description")
        term_list(item["required_terms"], f"requirements.{key}.required_terms")
    for key, item in steps.items():
        text(item["title"], f"setup.{key}.title")
        require(type(item["completed"]) is bool, f"setup.{key}.completed", "expected boolean")
        id_list(item["prerequisites"], f"setup.{key}.prerequisites", steps)
        id_list(item["requirement_ids"], f"setup.{key}.requirement_ids", requirements)
    visiting, visited = set(), set()

    def visit(key):
        require(key not in visiting, f"setup.{key}", "prerequisite cycle")
        if key in visited:
            return
        visiting.add(key)
        for dependency in steps[key]["prerequisites"]:
            visit(dependency)
            require(not steps[key]["completed"] or steps[dependency]["completed"],
                    f"setup.{key}", "completed step has incomplete prerequisite")
        visiting.remove(key)
        visited.add(key)

    for key in steps:
        visit(key)
    for key, item in kb.items():
        text(item["text"], f"knowledge_base.{key}.text")
        term_list(item["match_terms"], f"knowledge_base.{key}.match_terms")
        id_list(item["step_ids"], f"knowledge_base.{key}.step_ids", steps)
    for key, item in documents.items():
        text(item["text"], f"documents.{key}.text")
    for key, item in evidence.items():
        path = f"evidence.{key}"
        identifier(item["requirement_id"], path + ".requirement_id")
        identifier(item["document_id"], path + ".document_id")
        require(item["requirement_id"] in requirements, path, "unknown requirement")
        require(item["document_id"] in documents, path, "unknown document")
        text(item["quote"], path + ".quote")
        require(type(item["start"]) is int and item["start"] >= 0,
                path + ".start", "expected nonnegative integer")
        body = documents[item["document_id"]]["text"]
        start = item["start"]
        require(body[start:start + len(item["quote"])] == item["quote"],
                path, "quote does not match document at supplied character offset")
    return {"articles": kb, "steps": steps, "requirements": requirements,
            "documents": documents, "evidence": evidence}


def packet(status, article_ids, step_ids, requirement_ids, data):
    return {"status": status, "article_ids": article_ids, "step_ids": step_ids,
            "requirement_ids": requirement_ids, "data": data}


def validate_packet(value, indexes, path):
    """One shared handoff envelope, checked before every stage consumes it."""
    shape(value, ("status", "article_ids", "step_ids", "requirement_ids", "data"), path)
    require(value["status"] in ("complete", "abstained", "skipped"), path, "invalid status")
    require(type(value["data"]) is dict, path + ".data", "expected object")
    for field, target in (("article_ids", "articles"), ("step_ids", "steps"),
                          ("requirement_ids", "requirements")):
        id_list(value[field], path + "." + field, indexes[target])
    if value["status"] != "complete":
        require(not any(value[key] for key in ("article_ids", "step_ids", "requirement_ids")),
                path, "inactive handoff must have empty scope")
    else:
        require(len(value["article_ids"]) == 1, path, "one grounding article required")
    return value


def unique(values):
    return list(dict.fromkeys(values))


def requirement_scope(step_ids, indexes):
    return unique(rid for sid in step_ids for rid in indexes["steps"][sid]["requirement_ids"])


def answer_faq(query, indexes):
    matches = [article for article in indexes["articles"].values()
               if set(article["match_terms"]) <= tokens(query)]
    if len(matches) != 1:
        return packet("abstained", [], [], [], {
            "answer": None, "citations": [],
            "reason": "no_grounded_match" if not matches else "ambiguous_matches",
        })
    article = matches[0]
    return packet("complete", [article["id"]], list(article["step_ids"]),
                  requirement_scope(article["step_ids"], indexes), {
                      "answer": article["text"],
                      "citations": [{"article_id": article["id"], "quote": article["text"]}],
                      "reason": None,
                  })


def guided_setup(faq, indexes):
    validate_packet(faq, indexes, "faq_handoff")
    if faq["status"] != "complete":
        return packet("skipped", [], [], [], {"reason": "faq_abstained", "steps": []})
    article = indexes["articles"][faq["article_ids"][0]]
    require(faq["step_ids"] == article["step_ids"], "faq_handoff", "article scope mismatch")
    require(faq["requirement_ids"] == requirement_scope(faq["step_ids"], indexes),
            "faq_handoff", "requirement scope mismatch")
    require(faq["data"].get("answer") == article["text"], "faq_handoff", "ungrounded answer")
    ordered = []
    seen = set()

    def expand(sid):
        if sid in seen:
            return
        for dependency in indexes["steps"][sid]["prerequisites"]:
            expand(dependency)
        seen.add(sid)
        ordered.append(sid)

    for sid in faq["step_ids"]:
        expand(sid)
    rows = []
    for sid in ordered:
        step = indexes["steps"][sid]
        pending = [dep for dep in step["prerequisites"] if not indexes["steps"][dep]["completed"]]
        rows.append({
            "step_id": sid, "title": step["title"], "prerequisites": list(step["prerequisites"]),
            "status": "completed" if step["completed"] else ("blocked" if pending else "ready"),
            "blocked_by": pending, "requirement_ids": list(step["requirement_ids"]),
        })
    completed = sum(row["status"] == "completed" for row in rows)
    return packet("complete", list(faq["article_ids"]), ordered,
                  requirement_scope(ordered, indexes), {
                      "steps": rows,
                      "progress": {"completed": completed, "total": len(rows),
                                   "fraction": completed / len(rows) if rows else 1.0},
                      "next_step_ids": [row["step_id"] for row in rows if row["status"] == "ready"],
                  })


def review_documents(guided, indexes):
    validate_packet(guided, indexes, "guided_handoff")
    if guided["status"] != "complete":
        return packet("skipped", [], [], [], {
            "reason": "guided_skipped", "checks": [], "gaps": [], "disclaimer": DISCLAIMER,
        })
    # Reconstruct the trusted guided state to reject altered handoffs.
    article = indexes["articles"][guided["article_ids"][0]]
    expected_faq = packet("complete", [article["id"]], list(article["step_ids"]),
                          requirement_scope(article["step_ids"], indexes),
                          {"answer": article["text"]})
    require(guided == guided_setup(expected_faq, indexes),
            "guided_handoff", "guided state or scope mismatch")
    checks, gaps = [], []
    for row in guided["data"]["steps"]:
        if row["status"] != "completed":
            gaps.append({"kind": "incomplete_setup", "step_id": row["step_id"],
                         "blocked_by": row["blocked_by"], "article_ids": guided["article_ids"]})
    for rid in guided["requirement_ids"]:
        requirement = indexes["requirements"][rid]
        support = []
        for evidence in indexes["evidence"].values():
            if evidence["requirement_id"] != rid:
                continue
            missing = sorted(set(requirement["required_terms"]) - tokens(evidence["quote"]))
            support.append({
                "evidence_id": evidence["id"], "document_id": evidence["document_id"],
                "quote": evidence["quote"], "start": evidence["start"],
                "end": evidence["start"] + len(evidence["quote"]), "missing_terms": missing,
            })
        matched = any(not item["missing_terms"] for item in support)
        source_steps = [sid for sid in guided["step_ids"]
                        if rid in indexes["steps"][sid]["requirement_ids"]]
        checks.append({
            "requirement_id": rid, "description": requirement["description"],
            "status": "evidence_match" if matched else "gap",
            "step_ids": source_steps, "article_ids": list(guided["article_ids"]),
            "evidence": support,
        })
        if not matched:
            gaps.append({
                "kind": "insufficient_evidence" if support else "missing_evidence",
                "requirement_id": rid, "step_ids": source_steps,
                "article_ids": list(guided["article_ids"]),
                "evidence_ids": [item["evidence_id"] for item in support],
                "required_terms": list(requirement["required_terms"]),
            })
    return packet("complete", list(guided["article_ids"]), list(guided["step_ids"]),
                  list(guided["requirement_ids"]), {
                      "checks": checks, "gaps": gaps, "disclaimer": DISCLAIMER,
                      "method": "All required tokens must occur in one exact evidence quote; "
                                "token matching does not establish semantic truth.",
                  })


def run_pipeline(value):
    indexes = validate_input(value)
    faq = validate_packet(answer_faq(value["query"], indexes), indexes, "faq")
    guided = validate_packet(guided_setup(faq, indexes), indexes, "guided")
    review = validate_packet(review_documents(guided, indexes), indexes, "review")
    attention = faq["status"] == "abstained" or bool(review["data"]["gaps"])
    return {"schema_version": VERSION, "synthetic": True,
            "status": "needs_attention" if attention else "complete",
            "stages": {"faq": faq, "guided": guided, "review": review}}


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "$", f"duplicate JSON key {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"$: nonfinite JSON number {value}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "arguments", "usage: python -B implementation.py INPUT.json")
        with Path(argv[0]).open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=strict_object,
                              parse_constant=reject_constant)
        result = run_pipeline(value)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error",
                          "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
