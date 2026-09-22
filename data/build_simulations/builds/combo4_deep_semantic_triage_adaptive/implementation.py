"""Fictional-fixture, stdlib reference pipeline. No networking or provider code."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Callable


class ValidationError(ValueError):
    """Invalid schema, unsupported grounding, or invalid injected output."""


PRIORITIES = ("low", "normal", "high", "critical")
FIELDS = ("subject", "predicate", "value", "category_id")


def fail(message):
    raise ValidationError(message)


def obj(value, required, optional=(), path="input"):
    if not isinstance(value, dict):
        fail(f"{path}: expected object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        fail(f"{path}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        fail(f"{path}: expected nonempty string")
    return value


def array(value, path):
    if not isinstance(value, list):
        fail(f"{path}: expected array")
    return value


def choice(value, options, path):
    if not isinstance(value, str) or value not in options:
        fail(f"{path}: expected one of {list(options)}")
    return value


def unique_strings(value, path):
    values = tuple(text(v, path) for v in array(value, path))
    if len(set(values)) != len(values):
        fail(f"{path}: duplicate IDs/values")
    return values


def register(table, key, item, path):
    if key in table:
        fail(f"{path}: duplicate ID {key}")
    table[key] = item


def references(values, table, path):
    unknown = set(values) - table.keys()
    if unknown:
        fail(f"{path}: unknown IDs {sorted(unknown)}")


@dataclass(frozen=True)
class Statement:
    id: str
    source_id: str
    subject: str
    predicate: str
    value: str
    category_id: str
    text: str


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    statement_ids: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    id: str
    statement_id: str
    source_id: str


@dataclass(frozen=True)
class Category:
    id: str
    owner: str
    priority: str
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prerequisites: tuple[str, ...]
    required_resources: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    id: str
    experience: str
    preferred_format: str
    completed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    available_resources: tuple[str, ...]


@dataclass(frozen=True)
class Ticket:
    id: str
    question: str
    urgency: str


@dataclass(frozen=True)
class Request:
    sources: dict[str, Source]
    statements: dict[str, Statement]
    claims: tuple[Claim, ...]
    categories: dict[str, Category]
    tasks: dict[str, Task]
    profile: Profile
    ticket: Ticket
    top_k: int
    ambiguity_margin: float


def validate(raw):
    obj(raw, ("schema_version", "sources", "claims", "categories", "tasks",
              "profile", "ticket"), ("search",))
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        fail("schema_version: expected integer 1")
    tasks = {}
    for i, item in enumerate(array(raw["tasks"], "tasks")):
        p = f"tasks[{i}]"
        obj(item, ("id", "title", "prerequisites", "required_resources"), path=p)
        task = Task(text(item["id"], p), text(item["title"], p),
                    unique_strings(item["prerequisites"], p),
                    unique_strings(item["required_resources"], p))
        register(tasks, task.id, task, p)
    for task in tasks.values():
        references(task.prerequisites, tasks, f"task {task.id}.prerequisites")
    # Kahn's algorithm validates the whole graph, including currently unused tasks.
    seen = set()
    while len(seen) < len(tasks):
        ready = {t.id for t in tasks.values()
                 if t.id not in seen and set(t.prerequisites) <= seen}
        if not ready:
            fail("tasks: prerequisite cycle")
        seen.update(ready)

    categories = {}
    for i, item in enumerate(array(raw["categories"], "categories")):
        p = f"categories[{i}]"
        obj(item, ("id", "owner", "priority", "task_ids"), path=p)
        category = Category(text(item["id"], p), text(item["owner"], p),
                            choice(item["priority"], PRIORITIES, p),
                            unique_strings(item["task_ids"], p))
        references(category.task_ids, tasks, p)
        register(categories, category.id, category, p)
    sources, statements = {}, {}
    for i, item in enumerate(array(raw["sources"], "sources")):
        p = f"sources[{i}]"
        obj(item, ("id", "title", "statements"), path=p)
        source_id = text(item["id"], p)
        ids = []
        for j, s in enumerate(array(item["statements"], p)):
            sp = f"{p}.statements[{j}]"
            obj(s, ("id", "text") + FIELDS, path=sp)
            fields = {field: text(s[field], sp) for field in FIELDS}
            references((fields["category_id"],), categories, sp)
            statement = Statement(text(s["id"], sp), source_id,
                                  **fields, text=text(s["text"], sp))
            register(statements, statement.id, statement, sp)
            ids.append(statement.id)
        register(sources, source_id,
                 Source(source_id, text(item["title"], p), tuple(ids)), p)
    claims = {}
    for i, item in enumerate(array(raw["claims"], "claims")):
        p = f"claims[{i}]"
        obj(item, ("id", "source_id", "statement_id") + FIELDS, path=p)
        claim = Claim(text(item["id"], p), text(item["statement_id"], p),
                      text(item["source_id"], p))
        references((claim.source_id,), sources, p)
        references((claim.statement_id,), statements, p)
        statement = statements[claim.statement_id]
        if statement.source_id != claim.source_id:
            fail(f"{p}: unsupported source/statement pairing")
        for field in FIELDS:
            if text(item[field], p) != getattr(statement, field):
                fail(f"{p}: unsupported claim field {field}")
        register(claims, claim.id, claim, p)

    p = obj(raw["profile"], ("id", "experience", "preferred_format",
                            "completed_task_ids", "blocked_task_ids",
                            "available_resources"), path="profile")
    profile = Profile(
        text(p["id"], "profile.id"),
        choice(p["experience"], ("beginner", "intermediate", "expert"),
               "profile.experience"),
        choice(p["preferred_format"], ("text", "video", "exercise"),
               "profile.preferred_format"),
        unique_strings(p["completed_task_ids"], "profile.completed_task_ids"),
        unique_strings(p["blocked_task_ids"], "profile.blocked_task_ids"),
        unique_strings(p["available_resources"], "profile.available_resources"))
    references(profile.completed_task_ids, tasks, "profile.completed_task_ids")
    references(profile.blocked_task_ids, tasks, "profile.blocked_task_ids")
    completed = set(profile.completed_task_ids)
    if completed & set(profile.blocked_task_ids):
        fail("profile: task cannot be both completed and blocked")
    for tid in completed:
        if not set(tasks[tid].prerequisites) <= completed:
            fail("profile: completed tasks must include completed prerequisites")
    t = obj(raw["ticket"], ("id", "question", "urgency"), path="ticket")
    ticket = Ticket(text(t["id"], "ticket.id"), text(t["question"], "ticket.question"),
                    choice(t["urgency"], PRIORITIES, "ticket.urgency"))
    search = obj(raw.get("search", {}), (), ("top_k", "ambiguity_margin"),
                 path="search")
    top_k = search.get("top_k", 5)
    margin = search.get("ambiguity_margin", 0.1)
    if type(top_k) is not int or top_k < 1:
        fail("search.top_k: expected positive integer")
    if (type(margin) not in (int, float) or not math.isfinite(margin)
            or not 0 <= margin <= 1):
        fail("search.ambiguity_margin: expected finite number in [0, 1]")
    return Request(sources, statements, tuple(claims.values()), categories,
                   tasks, profile, ticket, top_k, float(margin))


def consolidate(request):
    groups = defaultdict(lambda: defaultdict(list))
    for claim in request.claims:
        s = request.statements[claim.statement_id]
        groups[(s.subject, s.predicate)][(s.value, s.category_id)].append(claim)
    findings = []
    for i, ((subject, predicate), alternatives) in enumerate(sorted(groups.items()), 1):
        finding = {"id": f"finding-{i}", "subject": subject, "predicate": predicate,
                   "disagreement": len(alternatives) > 1, "alternatives": []}
        for j, ((value, category), claims) in enumerate(sorted(alternatives.items()), 1):
            provenance = []
            for claim in sorted(claims, key=lambda c: c.id):
                s = request.statements[claim.statement_id]
                provenance.append({
                    "claim_id": claim.id, "source_id": s.source_id,
                    "source_title": request.sources[s.source_id].title,
                    "statement_id": s.id, "text": s.text})
            finding["alternatives"].append({
                "id": f"finding-{i}:alternative-{j}", "value": value,
                "category_id": category, "provenance": provenance})
        findings.append(finding)
    return findings


def documents(findings):
    docs = []
    for finding in findings:
        for alternative in finding["alternatives"]:
            # Repeated assertions retain provenance but cannot inflate term frequency.
            snippets = sorted({p["text"] for p in alternative["provenance"]})
            docs.append({
                "id": alternative["id"], "finding_id": finding["id"],
                "category_id": alternative["category_id"],
                "disagreement": finding["disagreement"],
                "text": "\n".join(snippets),
                "index_text": " ".join((finding["subject"], finding["predicate"],
                                        alternative["value"], *snippets)),
                "provenance": alternative["provenance"]})
    return docs


def tokens(value):
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def unit(vector):
    # Scale first to keep finite but very large or tiny injected vectors usable.
    scale = max((abs(v) for v in vector), default=0)
    if scale == 0:
        return [0.0 for _ in vector]
    scaled = [v / scale for v in vector]
    norm = math.sqrt(sum(v * v for v in scaled))
    return [v / norm for v in scaled]


def lexical_scores(docs, query):
    counts = [Counter(tokens(d["index_text"])) for d in docs]
    vocabulary = sorted({token for counts_ in counts for token in counts_})
    document_frequency = Counter(token for counts_ in counts for token in counts_)
    idf = {t: math.log((1 + len(docs)) / (1 + document_frequency[t])) + 1
           for t in vocabulary}

    def vector(count):
        return unit([(1 + math.log(count[t])) * idf[t] if count[t] else 0
                     for t in vocabulary])

    query_vector = vector(Counter(tokens(query)))
    return [sum(a * b for a, b in zip(query_vector, vector(count)))
            for count in counts]


def embedding_scores(docs, query, embed):
    vectors = embed([query] + [d["index_text"] for d in docs])
    if not isinstance(vectors, (list, tuple)) or len(vectors) != len(docs) + 1:
        fail("embedding: expected one vector per query/document")
    width = None
    normalized = []
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            fail("embedding: vectors must be nonempty arrays")
        if width is None:
            width = len(vector)
        if len(vector) != width:
            fail("embedding: mismatched dimensions")
        if any(type(v) not in (float, int) for v in vector):
            fail("embedding: expected finite numeric coordinates")
        try:
            numeric = [float(v) for v in vector]
        except (OverflowError, ValueError):
            fail("embedding: coordinate outside supported numeric range")
        if not all(math.isfinite(v) for v in numeric):
            fail("embedding: expected finite numeric coordinates")
        normalized.append(unit(numeric))
    return [max(-1.0, min(1.0, sum(a * b for a, b in zip(normalized[0], v))))
            for v in normalized[1:]]


def retrieve(request, findings, embed=None):
    docs = documents(findings)
    scores = (embedding_scores(docs, request.ticket.question, embed)
              if embed is not None and docs else lexical_scores(docs, request.ticket.question))
    positive = [dict(d, score=score) for d, score in zip(docs, scores) if score > 0]
    positive.sort(key=lambda d: (-d["score"], d["id"]))
    selected = positive[:request.top_k]
    # The cutoff must not hide ties or near-best evidence relevant to routing ambiguity.
    if selected:
        cutoff = selected[-1]["score"]
        best = selected[0]["score"]
        selected.extend(d for d in positive[request.top_k:]
                        if math.isclose(d["score"], cutoff, rel_tol=1e-12, abs_tol=1e-15)
                        or (best - d["score"]) / best <= request.ambiguity_margin + 1e-12)
    return {"method": "injected_embedding_cosine" if embed is not None else "lexical_tfidf_cosine",
            "query": request.ticket.question, "requested_top_k": request.top_k,
            "status": "evidence_found" if selected else "no_evidence",
            "hits": [{k: v for k, v in d.items() if k != "index_text"} for d in selected]}


def highest_priority(values):
    return max(values, key=PRIORITIES.index)


def route(request, findings, retrieval):
    hits = retrieval["hits"]
    if not hits:
        return {"status": "needs_evidence", "category_id": None, "owner": None,
                "priority": request.ticket.urgency, "candidate_category_ids": [],
                "category_scores": {}, "evidence_ids": [], "conflicting_finding_ids": [],
                "reason": "No positive retrieval score; no category guessed."}
    # Max rather than sum prevents repeated/redundant findings from voting an owner in.
    scores = {}
    for hit in hits:
        scores[hit["category_id"]] = max(scores.get(hit["category_id"], 0), hit["score"])
    best = max(scores.values())
    candidates = {category for category, score in scores.items()
                  if (best - score) / best <= request.ambiguity_margin + 1e-12}
    candidate_findings = {h["finding_id"] for h in hits if h["category_id"] in candidates}
    conflicts = [f for f in findings if f["id"] in candidate_findings and f["disagreement"]]
    # All alternatives in a relevant disagreement remain visible even below search cutoff.
    for finding in conflicts:
        candidates.update(a["category_id"] for a in finding["alternatives"])
    ambiguous = len(candidates) != 1 or bool(conflicts)
    category_id = None if ambiguous else next(iter(candidates))
    priority = highest_priority([request.ticket.urgency] +
                                [request.categories[c].priority for c in candidates])
    return {
        "status": "needs_review" if ambiguous else "routed",
        "category_id": category_id,
        "owner": request.categories[category_id].owner if category_id else None,
        "priority": priority, "candidate_category_ids": sorted(candidates),
        "category_scores": dict(sorted(scores.items())),
        "evidence_ids": [h["id"] for h in hits],
        "conflicting_finding_ids": [f["id"] for f in conflicts],
        "reason": ("Competing categories or explicit disagreement require human review."
                   if ambiguous else "Highest retrieved category evidence, configured owner and priority.")}


def onboard(request, routing):
    profile = request.profile
    base = {"profile_id": profile.id, "routing_status": routing["status"],
            "priority": routing["priority"], "preferred_format": profile.preferred_format,
            "experience": profile.experience, "steps": [], "completed_task_ids": [],
            "pace": "expedited" if routing["priority"] in ("high", "critical") else "self_paced"}
    if routing["status"] == "needs_evidence":
        return dict(base, status="needs_evidence")
    needed = set()
    pending = [tid for cid in routing["candidate_category_ids"]
               for tid in request.categories[cid].task_ids]
    while pending:
        tid = pending.pop()
        if tid not in needed:
            needed.add(tid)
            pending.extend(request.tasks[tid].prerequisites)
    completed = set(profile.completed_task_ids)
    base["completed_task_ids"] = sorted(needed & completed)
    ordered = set(needed & completed)
    blocked = set()
    while needed - ordered:
        ready = sorted(tid for tid in needed - ordered
                       if set(request.tasks[tid].prerequisites) <= ordered)
        for tid in ready:
            task = request.tasks[tid]
            reasons = []
            if routing["status"] == "needs_review":
                reasons.append("routing_confirmation_required")
            if tid in profile.blocked_task_ids:
                reasons.append("user_reported_block")
            reasons.extend(f"missing_resource:{r}" for r in task.required_resources
                           if r not in profile.available_resources)
            reasons.extend(f"blocked_prerequisite:{p}" for p in task.prerequisites if p in blocked)
            if reasons:
                blocked.add(tid)
            status = ("blocked" if reasons else
                      "ready" if set(task.prerequisites) <= completed else "waiting")
            detail = {"beginner": "guided_steps", "intermediate": "standard_instructions",
                      "expert": "concise_checklist"}[profile.experience]
            base["steps"].append({
                "task_id": tid, "title": task.title,
                "prerequisites": list(task.prerequisites), "status": status,
                "blocked_by": reasons, "format": profile.preferred_format, "detail": detail})
            ordered.add(tid)
    status = ("needs_review" if routing["status"] == "needs_review" else
              "blocked" if blocked else "planned" if base["steps"] else "complete")
    return dict(base, status=status)


def grounded_wording(retrieval, wording):
    snippets = {hit["id"]: hit["text"] for hit in retrieval["hits"]}
    # A callback can select/reorder exact extracts, not invent paraphrased assertions.
    result = wording({"evidence": dict(snippets)})
    obj(result, ("text", "evidence_ids"), path="wording")
    ids = unique_strings(result["evidence_ids"], "wording.evidence_ids")
    references(ids, snippets, "wording.evidence_ids")
    expected = "\n".join(snippets[eid] for eid in ids)
    if not isinstance(result["text"], str) or result["text"] != expected:
        fail("wording: unsupported text; only exact cited extracts are permitted")
    return {"text": expected, "evidence_ids": list(ids)}


def run_pipeline(raw, embed: Callable | None = None, wording: Callable | None = None):
    request = validate(raw)
    findings = consolidate(request)
    retrieval = retrieve(request, findings, embed)
    routing = route(request, findings, retrieval)
    result = {"schema_version": 1, "ticket_id": request.ticket.id,
              "research": {"method": "explicit_claim_consolidation",
                           "findings": findings},
              "search": retrieval, "triage": routing,
              "onboarding": onboard(request, routing)}
    if wording is not None:
        result["wording"] = grounded_wording(retrieval, wording)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON: duplicate key {key}")
        result[key] = value
    return result


def load_json(value):
    return json.loads(value, object_pairs_hook=unique_object,
                      parse_constant=lambda value: fail(f"JSON: nonfinite constant {value}"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = load_json(args.input.read_text(encoding="utf-8"))
        result = run_pipeline(raw)
    except (ValidationError, ValueError, OSError, RecursionError) as exc:
        print(json.dumps({"error": {"type": "validation_error", "message": str(exc)}}))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
