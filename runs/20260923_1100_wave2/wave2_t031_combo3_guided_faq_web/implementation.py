"""Synthetic, offline guided -> FAQ -> web reference pipeline (Python stdlib).

Usage: python -B implementation.py example_input.json
Web ingestion accepts caller-supplied snapshots; it never fetches URLs.
An optional answerer(stage, query, evidence) may select exact evidence excerpts:
return {"answer": "...", "source_ids": ["..."]}. Unsupported prose is rejected.
"""

import copy
import ipaddress
import json
import re
import sys
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, fields, path):
    require(type(value) is dict, path + " must be an object")
    require(set(value) == set(fields), path + " has missing or unknown fields")


def text(value, path, maximum=20000):
    require(type(value) is str and bool(value.strip()), path + " must be nonempty text")
    require(len(value) <= maximum, path + " is too long")
    return value


def sequence(value, path, maximum=200):
    require(type(value) is list and len(value) <= maximum, path + " must be a bounded list")
    return value


def identifiers(value, path):
    sequence(value, path)
    for item in value:
        text(item, path, 100)
    require(len(value) == len(set(value)), path + " contains duplicates")
    return value


def hostname(value):
    text(value, "allowlisted host", 253)
    require(value == value.lower(), "allowlisted hosts must be lowercase")
    require("." in value and not value.endswith("."), "host must be a dotted DNS name")
    require(all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", p)
                for p in value.split(".")), "invalid DNS host")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    raise ValidationError("IP address hosts are not supported")


def validate_url(value, hosts):
    text(value, "page URL", 2048)
    require(not any(c.isspace() or ord(c) < 32 for c in value)
            and "\\" not in value, "URL contains unsafe characters")
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "https" and parsed.hostname in hosts
                 and parsed.username is None and parsed.password is None
                 and parsed.port in (None, 443) and not parsed.fragment)
    except ValueError as exc:
        raise ValidationError("invalid page URL") from exc
    require(valid, "URL must use HTTPS on an exact allowlisted host without credentials or fragment")


def validate_input(request):
    shape(request, ("schema_version", "fixture_label", "question", "setup",
                    "knowledge", "research"), "input")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "schema_version must be 1")
    require(request["fixture_label"] == "synthetic", "fixture_label must be synthetic")
    text(request["question"], "question", 2000)
    setup = request["setup"]
    shape(setup, ("product", "steps", "completed"), "setup")
    text(setup["product"], "product", 100)
    steps = sequence(setup["steps"], "steps", 100)
    require(bool(steps), "at least one onboarding step is required")
    ids = []
    for step in steps:
        shape(step, ("id", "requires", "required"), "step")
        ids.append(text(step["id"], "step.id", 100))
        identifiers(step["requires"], "step.requires")
        require(type(step["required"]) is bool, "step.required must be boolean")
    identifiers(ids, "step IDs")
    by_id = {s["id"]: s for s in steps}
    for step in steps:
        require(set(step["requires"]) <= set(ids), "unknown prerequisite")
    visited, active = set(), set()

    def visit(step_id):
        require(step_id not in active, "cyclic onboarding prerequisites")
        if step_id in visited:
            return
        active.add(step_id)
        for dependency in by_id[step_id]["requires"]:
            visit(dependency)
        active.remove(step_id)
        visited.add(step_id)

    for step_id in ids:
        visit(step_id)
    completed = set(identifiers(setup["completed"], "completed"))
    require(completed <= set(ids), "unknown completed step")
    for step_id in completed:
        require(set(by_id[step_id]["requires"]) <= completed,
                "completed step has incomplete prerequisites")
    documents = sequence(request["knowledge"], "knowledge")
    doc_ids = []
    for doc in documents:
        shape(doc, ("id", "product", "text"), "knowledge document")
        doc_ids.append(text(doc["id"], "document.id", 100))
        text(doc["product"], "document.product", 100)
        text(doc["text"], "document.text")
    identifiers(doc_ids, "knowledge IDs")
    research = request["research"]
    shape(research, ("allowlisted_hosts", "pages"), "research")
    hosts = identifiers(research["allowlisted_hosts"], "allowlisted_hosts")
    for host in hosts:
        hostname(host)
    pages = sequence(research["pages"], "pages")
    page_ids, urls = [], []
    for page in pages:
        shape(page, ("id", "url", "title", "text"), "page")
        page_ids.append(text(page["id"], "page.id", 100))
        validate_url(page["url"], hosts)
        urls.append(page["url"])
        text(page["title"], "page.title", 300)
        text(page["text"], "page.text")
    identifiers(page_ids, "page IDs")
    require(len(urls) == len(set(urls)), "duplicate page URLs")
    return request


STAGE_FIELDS = ("stage", "status", "context", "evidence", "answer", "reason",
                "progress", "handoff")
EVIDENCE_FIELDS = ("id", "kind", "text", "url", "title")
HANDOFF_FIELDS = ("ready", "query", "source_ids", "upstream_source_ids")


def validate_stage(record, stage):
    """One validation boundary shared by all three producer/consumer stages."""
    shape(record, STAGE_FIELDS, "stage output")
    require(record["stage"] == stage, "unexpected upstream stage")
    shape(record["context"], ("product", "question"), "context")
    for value in record["context"].values():
        text(value, "context value")
    handoff = record["handoff"]
    shape(handoff, HANDOFF_FIELDS, "handoff")
    require(type(handoff["ready"]) is bool, "handoff.ready must be boolean")
    text(handoff["query"], "handoff.query")
    identifiers(handoff["source_ids"], "handoff sources")
    identifiers(handoff["upstream_source_ids"], "upstream sources")
    evidence = sequence(record["evidence"], "evidence")
    ids = []
    for item in evidence:
        shape(item, EVIDENCE_FIELDS, "evidence item")
        ids.append(text(item["id"], "evidence.id", 100))
        text(item["text"], "evidence.text")
        require(item["kind"] == ("knowledge" if stage == "faq" else "web"),
                "incorrect evidence kind")
        if stage == "web":
            text(item["url"], "evidence.url")
            text(item["title"], "evidence.title")
        else:
            require(item["url"] is None and item["title"] is None,
                    "knowledge evidence must not invent web provenance")
    identifiers(ids, "evidence IDs")
    require(handoff["source_ids"] == ids, "handoff sources do not match evidence")
    if stage == "guided":
        require(record["status"] in ("ready", "blocked") and not evidence
                and record["answer"] is None, "invalid guided output")
        require(handoff["ready"] == (record["status"] == "ready"), "invalid readiness")
        progress = record["progress"]
        shape(progress, ("completed", "remaining", "available", "total",
                         "completed_count"), "progress")
        for field in ("completed", "remaining", "available"):
            identifiers(progress[field], "progress." + field)
        require(type(progress["total"]) is int and progress["total"] > 0
                and type(progress["completed_count"]) is int
                and progress["completed_count"] == len(progress["completed"])
                and progress["total"] == len(progress["completed"]) + len(progress["remaining"])
                and not set(progress["completed"]) & set(progress["remaining"])
                and set(progress["available"]) <= set(progress["remaining"]),
                "invalid progress")
    else:
        require(record["progress"] is None, "only guided output may include progress")
        require(record["status"] in ("answered", "abstained"), "invalid answer status")
        if record["status"] == "answered":
            require(bool(evidence) and record["answer"] == "\n".join(e["text"] for e in evidence)
                    and record["reason"] is None, "answer is not grounded in exact evidence")
        else:
            require(not evidence and record["answer"] is None, "abstention cannot contain an answer")
    if record["status"] in ("abstained", "blocked"):
        text(record["reason"], "reason")
    else:
        require(record["reason"] is None, "successful output cannot have a failure reason")
    return record


def record(stage, context, query, ready, upstream=()):
    return {"stage": stage, "status": "abstained", "context": copy.deepcopy(context),
            "evidence": [], "answer": None, "reason": "no_relevant_evidence",
            "progress": None,
            "handoff": {"ready": ready, "query": query, "source_ids": [],
                        "upstream_source_ids": list(upstream)}}


def guided_stage(request):
    setup = request["setup"]
    completed = set(setup["completed"])
    remaining = [s["id"] for s in setup["steps"] if s["id"] not in completed]
    ready = all(not s["required"] or s["id"] in completed for s in setup["steps"])
    context = {"product": setup["product"], "question": request["question"]}
    result = record("guided", context, context["product"] + " " + context["question"], ready)
    result.update(status="ready" if ready else "blocked",
                  reason=None if ready else "onboarding_incomplete",
                  progress={"completed": [s["id"] for s in setup["steps"] if s["id"] in completed],
                            "remaining": remaining,
                            "available": [s["id"] for s in setup["steps"]
                                          if s["id"] not in completed
                                          and set(s["requires"]) <= completed],
                            "total": len(setup["steps"]), "completed_count": len(completed)})
    return validate_stage(result, "guided")


STOP_WORDS = set("a an the is are of for to how do does can i my with and in on it".split())


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOP_WORDS


def retrieve(query, candidates):
    terms = tokens(query)
    ranked = [(len(terms & tokens(item["text"])), item) for item in candidates]
    ranked = [(score, item) for score, item in ranked if score > 0]
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [item for _, item in ranked[:3]]


def answer_from_evidence(result, evidence, answerer):
    if not evidence:
        return result
    selected = evidence
    if answerer is not None:
        require(callable(answerer), "answerer must be callable")
        try:
            proposal = answerer(result["stage"], result["handoff"]["query"],
                                copy.deepcopy(evidence))
        except Exception as exc:
            raise ValidationError("injected answerer failed") from exc
        shape(proposal, ("answer", "source_ids"), "injected answer")
        ids = identifiers(proposal["source_ids"], "injected source IDs")
        require(bool(ids), "injected answer must cite evidence")
        by_id = {item["id"]: item for item in evidence}
        require(set(ids) <= set(by_id), "injected answer cites unknown evidence")
        selected = [by_id[source_id] for source_id in ids]
        require(proposal["answer"] == "\n".join(item["text"] for item in selected),
                "injected answer must contain only exact cited excerpts")
    result.update(status="answered", reason=None, evidence=selected,
                  answer="\n".join(item["text"] for item in selected))
    result["handoff"]["source_ids"] = [item["id"] for item in selected]
    return result


def faq_stage(upstream, documents, answerer=None):
    validate_stage(upstream, "guided")
    result = record("faq", upstream["context"], upstream["handoff"]["query"],
                    upstream["handoff"]["ready"], upstream["handoff"]["source_ids"])
    if not result["handoff"]["ready"]:
        result["reason"] = "onboarding_incomplete"
    else:
        candidates = [{"id": d["id"], "kind": "knowledge", "text": d["text"],
                       "url": None, "title": None} for d in documents
                      if d["product"] == result["context"]["product"]]
        # Product selects the corpus; it must not alone make a FAQ relevant.
        evidence = retrieve(result["context"]["question"], candidates)
        answer_from_evidence(result, evidence, answerer)
    return validate_stage(result, "faq")


def web_stage(upstream, research, answerer=None):
    validate_stage(upstream, "faq")
    result = record("web", upstream["context"], upstream["handoff"]["query"],
                    upstream["handoff"]["ready"], upstream["handoff"]["source_ids"])
    if not result["handoff"]["ready"]:
        result["reason"] = "onboarding_incomplete"
    else:
        candidates = []
        for page in research["pages"]:
            validate_url(page["url"], research["allowlisted_hosts"])
            candidates.append({"id": page["id"], "kind": "web", "text": page["text"],
                               "url": page["url"], "title": page["title"]})
        evidence = retrieve(result["context"]["question"], candidates)
        answer_from_evidence(result, evidence, answerer)
    return validate_stage(result, "web")


def run_pipeline(request, answerer=None):
    request = copy.deepcopy(validate_input(request))
    require(answerer is None or callable(answerer), "answerer must be callable")
    guided = guided_stage(request)
    faq = faq_stage(guided, request["knowledge"], answerer)
    web = web_stage(faq, request["research"], answerer)
    return {"schema_version": 1, "fixture_label": "synthetic",
            "status": "ok" if guided["handoff"]["ready"] else "blocked",
            "stages": [guided, faq, web]}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], "r", encoding="utf-8") as stream:
            payload = stream.read(2_000_001)
        require(len(payload) <= 2_000_000, "input exceeds two million characters")
        request = json.loads(payload, object_pairs_hook=reject_duplicates,
                             parse_constant=reject_constant)
        result = run_pipeline(request)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
