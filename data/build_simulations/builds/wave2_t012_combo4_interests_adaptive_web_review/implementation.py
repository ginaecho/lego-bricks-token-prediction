"""Offline synthetic reference pipeline. Python standard library; no network I/O.

Run: python -B implementation.py example_input.json
Evidence checks are literal phrase checks, not semantic or certification judgments.
"""

import copy
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(required.split()), f"{path}: unexpected or missing fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")


def strings(value, path):
    require(isinstance(value, list), f"{path}: expected array")
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), f"{path}: duplicate entries")


def records(value, path):
    require(isinstance(value, list), f"{path}: expected array")
    result = {}
    for record in value:
        require(isinstance(record, dict), f"{path}: expected object records")
        text(record.get("id"), f"{path}.id")
        require(record["id"] not in result, f"{path}: duplicate id")
        result[record["id"]] = record
    return result


def allowed_url(url, hosts):
    text(url, "source.url")
    require(not any(c.isspace() or ord(c) < 32 for c in url), "URL contains whitespace/control")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(
        parts.scheme == "https" and parts.hostname in hosts
        and parts.username is None and parts.password is None
        and port in (None, 443) and not parts.fragment and "\\" not in url,
        "Source URL must be HTTPS on an exact allowlisted host without credentials/fragment",
    )


def validate(data, stages=None):
    """Shared validation for input and every successive output boundary."""
    fields(data, "schema_version fixture_label profile catalog steps allowed_hosts sources documents requirements", "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "Unsupported schema version")
    require(data["fixture_label"] == "SYNTHETIC", "Fixture must be labeled SYNTHETIC")
    p = data["profile"]
    fields(p, "interests excluded_topics excluded_items experience preferred_format completed_steps limit", "profile")
    require(isinstance(p["interests"], dict), "interests: expected topic/weight object")
    for topic, weight in p["interests"].items():
        text(topic, "interest topic")
        require(type(weight) is int and 1 <= weight <= 10, "Interest weights must be integers 1..10")
    for key in ("excluded_topics", "excluded_items", "completed_steps"):
        strings(p[key], key)
    require(p["experience"] in ("beginner", "experienced"), "Invalid experience")
    require(p["preferred_format"] in ("text", "video"), "Invalid preferred format")
    require(type(p["limit"]) is int and 0 <= p["limit"] <= 100, "limit must be an integer 0..100")
    catalog = records(data["catalog"], "catalog")
    steps = records(data["steps"], "steps")
    sources = records(data["sources"], "sources")
    documents = records(data["documents"], "documents")
    requirements = records(data["requirements"], "requirements")
    require(set(p["excluded_items"]) <= catalog.keys(), "Unknown excluded item")
    require(set(p["completed_steps"]) <= steps.keys(), "Unknown completed step")
    for item in catalog.values():
        fields(item, "id title topics", "catalog item")
        text(item["title"], "title")
        strings(item["topics"], "topics")
        require(bool(item["topics"]), "Catalog item needs topics")
    for step in steps.values():
        fields(step, "id topic format audience prerequisites queries", "step")
        text(step["topic"], "step.topic")
        require(step["format"] in ("text", "video"), "Invalid step format")
        require(step["audience"] in ("all", "beginner"), "Invalid step audience")
        strings(step["prerequisites"], "prerequisites")
        strings(step["queries"], "queries")
        require(bool(step["queries"]), "Step needs at least one research query")
        require(set(step["prerequisites"]) <= steps.keys(), "Unknown prerequisite")
    visiting, visited = set(), set()

    def visit(sid):
        require(sid not in visiting, "Prerequisite cycle")
        if sid in visited:
            return
        visiting.add(sid)
        for parent in steps[sid]["prerequisites"]:
            visit(parent)
        visiting.remove(sid)
        visited.add(sid)

    for sid in steps:
        visit(sid)
    strings(data["allowed_hosts"], "allowed_hosts")
    for host in data["allowed_hosts"]:
        require(bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)+", host)), "Invalid allowlisted hostname")
    for source in sources.values():
        fields(source, "id url title body", "source")
        allowed_url(source["url"], data["allowed_hosts"])
        text(source["title"], "source.title")
        text(source["body"], "source.body")
    for document in documents.values():
        fields(document, "id title body", "document")
        text(document["title"], "document.title")
        text(document["body"], "document.body")
    for req in requirements.values():
        fields(req, "id step_id phrase document_ids", "requirement")
        require(req["step_id"] in steps, "Requirement references unknown step")
        text(req["phrase"], "requirement.phrase")
        strings(req["document_ids"], "document_ids")
        require(bool(req["document_ids"]) and set(req["document_ids"]) <= documents.keys(), "Requirement needs known documents")
    if stages is not None:
        validate_stages(data, stages, catalog, steps, sources, documents, requirements)
    return data


def validate_stages(data, stages, catalog, steps, sources, documents, requirements):
    order = ["interests", "adaptive", "web", "review"]
    require(isinstance(stages, dict) and list(stages) == order[:len(stages)], "Invalid stage order")
    if "interests" not in stages:
        return
    recs = stages["interests"]
    require(isinstance(recs, list), "Recommendations must be an array")
    seen = set()
    for rec in recs:
        fields(rec, "id score matched_topics explanation", "recommendation")
        require(rec["id"] in catalog and rec["id"] not in seen, "Invalid recommendation id")
        seen.add(rec["id"])
        item = catalog[rec["id"]]
        require(not excluded(data, item), "Excluded recommendation")
        matched = sorted(set(item["topics"]) & data["profile"]["interests"].keys())
        require(rec["matched_topics"] == matched and bool(matched), "Ungrounded matched topics")
        require(type(rec["score"]) is int and rec["score"] == sum(data["profile"]["interests"][t] for t in matched), "Ungrounded recommendation score")
        require(rec["explanation"] == recommendation_explanation(data, matched), "Ungrounded explanation")
    require(recs == sorted(recs, key=lambda r: (-r["score"], r["id"])), "Incorrect ranking")
    require(len(recs) <= data["profile"]["limit"], "Too many recommendations")
    if "adaptive" not in stages:
        return
    plan = stages["adaptive"]
    fields(plan, "steps blocked", "adaptive")
    planned = records(plan["steps"], "planned steps")
    blocked = records(plan["blocked"], "blocked steps")
    permitted = {}
    for rec in recs:
        reachable = set()

        def collect(sid):
            if sid not in reachable:
                reachable.add(sid)
                for parent in steps[sid]["prerequisites"]:
                    collect(parent)

        for step in steps.values():
            if step["topic"] in catalog[rec["id"]]["topics"]:
                if data["profile"]["experience"] != "experienced" or step["audience"] != "beginner":
                    collect(step["id"])
        permitted[rec["id"]] = reachable
    available = set(data["profile"]["completed_steps"])
    for row in planned.values():
        fields(row, "id reason recommendation_ids", "planned step")
        require(row["id"] in steps and row["id"] not in available, "Unknown or repeated step")
        step = steps[row["id"]]
        require(step["topic"] not in data["profile"]["excluded_topics"], "Excluded onboarding topic")
        require(set(step["prerequisites"]) <= available, "Unsatisfied prerequisite")
        strings(row["recommendation_ids"], "recommendation_ids")
        require(bool(row["recommendation_ids"]) and set(row["recommendation_ids"]) <= seen, "Missing recommendation lineage")
        require(all(row["id"] in permitted[rid] for rid in row["recommendation_ids"]), "Ungrounded onboarding lineage")
        text(row["reason"], "step reason")
        available.add(row["id"])
    for row in blocked.values():
        fields(row, "id reason", "blocked step")
        require(row["id"] in steps and row["id"] not in planned, "Invalid blocked step")
        text(row["reason"], "blocked reason")
    if "web" not in stages:
        return
    web = stages["web"]
    fields(web, "retrieval_mode findings unmatched", "web")
    require(web["retrieval_mode"] == "offline_synthetic_url_snapshots", "Invalid retrieval mode")
    findings = records(web["findings"], "findings")
    for finding in findings.values():
        fields(finding, "id step_id source_id url query quote start end", "finding")
        require(finding["step_id"] in planned and finding["source_id"] in sources, "Invalid finding lineage")
        source = sources[finding["source_id"]]
        require(finding["url"] == source["url"], "Changed source provenance")
        require(finding["query"] in steps[finding["step_id"]]["queries"], "Unknown query")
        check_span(finding, source["body"])
        require(re.search(re.escape(finding["query"]), finding["quote"], re.I) is not None, "Query absent from quote")
    require(isinstance(web["unmatched"], list), "unmatched must be an array")
    for row in web["unmatched"]:
        fields(row, "step_id query", "unmatched query")
        require(row["step_id"] in planned and row["query"] in steps[row["step_id"]]["queries"], "Invalid unmatched query")
    if "review" not in stages:
        return
    review = stages["review"]
    fields(review, "disclaimer checks", "review")
    require(review["disclaimer"] == DISCLAIMER, "Missing non-certification disclaimer")
    checks = records(review["checks"], "checks")
    require(checks.keys() == requirements.keys(), "Incomplete review")
    for check in checks.values():
        fields(check, "id step_id status finding_ids document_evidence gaps", "check")
        req = requirements[check["id"]]
        require(check["step_id"] == req["step_id"], "Changed requirement lineage")
        strings(check["finding_ids"], "finding_ids")
        for fid in check["finding_ids"]:
            require(fid in findings and findings[fid]["step_id"] == req["step_id"], "Invalid finding reference")
            require(contains(req["phrase"], findings[fid]["quote"]), "Finding does not support requirement")
        require(isinstance(check["document_evidence"], list), "Invalid document evidence")
        for evidence in check["document_evidence"]:
            fields(evidence, "document_id quote start end", "document evidence")
            require(evidence["document_id"] in req["document_ids"], "Unassigned document")
            check_span(evidence, documents[evidence["document_id"]]["body"])
            require(contains(req["phrase"], evidence["quote"]), "Document does not support requirement")
        expected_gaps = gaps_for(req["step_id"] in planned, check["finding_ids"], check["document_evidence"])
        require(check["gaps"] == expected_gaps, "Incorrect gap explanation")
        expected_status = "out_of_scope" if req["step_id"] not in planned else ("gap" if expected_gaps else "evidence_present")
        require(check["status"] == expected_status, "Incorrect review status")


def check_span(evidence, body):
    start, end = evidence["start"], evidence["end"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(body), "Invalid evidence offsets")
    require(evidence["quote"] == body[start:end], "Evidence does not match source")


def contains(phrase, body):
    return re.search(re.escape(phrase), body, re.I) is not None


def excluded(data, item):
    p = data["profile"]
    return item["id"] in p["excluded_items"] or bool(set(item["topics"]) & set(p["excluded_topics"]))


def recommendation_explanation(data, matched):
    return "Matched declared interests: " + ", ".join(f"{topic} (weight {data['profile']['interests'][topic]})" for topic in matched)


def interests_stage(data):
    result = []
    for item in data["catalog"]:
        matched = sorted(set(item["topics"]) & data["profile"]["interests"].keys())
        if matched and not excluded(data, item):
            result.append({"id": item["id"], "score": sum(data["profile"]["interests"][t] for t in matched),
                           "matched_topics": matched, "explanation": recommendation_explanation(data, matched)})
    return sorted(result, key=lambda row: (-row["score"], row["id"]))[:data["profile"]["limit"]]


def adaptive_stage(data, recommendations):
    p = data["profile"]
    catalog = {item["id"]: item for item in data["catalog"]}
    steps = {step["id"]: step for step in data["steps"]}
    roots = []
    for rec in recommendations:
        candidates = [s for s in steps.values() if s["topic"] in catalog[rec["id"]]["topics"]]
        for step in sorted(candidates, key=lambda s: (s["format"] != p["preferred_format"], s["id"])):
            if p["experience"] == "experienced" and step["audience"] == "beginner":
                continue
            roots.append((step["id"], rec["id"]))
    plan, blocked, completed = {}, {}, set(p["completed_steps"])

    def closure(sid):
        if sid in completed:
            return []
        step = steps[sid]
        if step["topic"] in p["excluded_topics"]:
            raise ValidationError(f"Excluded prerequisite/topic: {sid} ({step['topic']})")
        result = []
        for pid in step["prerequisites"]:
            result.extend(closure(pid))
        return result + [sid]

    for sid, rid in roots:
        try:
            needed = closure(sid)
        except ValidationError as exc:
            blocked[sid] = {"id": sid, "reason": str(exc)}
            continue
        for needed_id in needed:
            if needed_id not in plan:
                step = steps[needed_id]
                reason = ("Relevant recommended topic" if needed_id == sid else f"Required prerequisite for {sid}")
                reason += f"; experience={p['experience']}; format={step['format']}; preferred={p['preferred_format']}"
                plan[needed_id] = {"id": needed_id, "reason": reason, "recommendation_ids": []}
            if rid not in plan[needed_id]["recommendation_ids"]:
                plan[needed_id]["recommendation_ids"].append(rid)
    return {"steps": list(plan.values()), "blocked": list(blocked.values())}


def span(phrase, body):
    match = re.search(re.escape(phrase), body, re.I)
    if match is None:
        return None
    start = max(0, match.start() - 50)
    end = min(len(body), match.end() + 100)
    return {"quote": body[start:end], "start": start, "end": end}


def web_stage(data, onboarding):
    steps = {s["id"]: s for s in data["steps"]}
    findings, unmatched = [], []
    for row in onboarding["steps"]:
        for query in steps[row["id"]]["queries"]:
            found = False
            for source in sorted(data["sources"], key=lambda s: s["id"]):
                evidence = span(query, source["body"])
                if evidence is not None:
                    found = True
                    findings.append({"id": f"finding-{len(findings) + 1}", "step_id": row["id"],
                                     "source_id": source["id"], "url": source["url"], "query": query, **evidence})
            if not found:
                unmatched.append({"step_id": row["id"], "query": query})
    return {"retrieval_mode": "offline_synthetic_url_snapshots", "findings": findings, "unmatched": unmatched}


DISCLAIMER = "Literal evidence checks only; not certification, legal advice, or a compliance determination."


def gaps_for(in_scope, findings, documents):
    if not in_scope:
        return ["Step not selected by validated onboarding; requirement not assessed."]
    return ([] if findings else ["No retrieved research evidence contains the required phrase."]) + (
        [] if documents else ["No assigned document contains the required phrase."])


def review_stage(data, research, onboarding):
    selected = {row["id"] for row in onboarding["steps"]}
    documents = {doc["id"]: doc for doc in data["documents"]}
    checks = []
    for req in data["requirements"]:
        active = req["step_id"] in selected
        fids = [f["id"] for f in research["findings"] if active and f["step_id"] == req["step_id"] and contains(req["phrase"], f["quote"])]
        evidence = []
        if active:
            for did in req["document_ids"]:
                match = span(req["phrase"], documents[did]["body"])
                if match is not None:
                    evidence.append({"document_id": did, **match})
        gaps = gaps_for(active, fids, evidence)
        checks.append({"id": req["id"], "step_id": req["step_id"],
                       "status": "out_of_scope" if not active else ("gap" if gaps else "evidence_present"),
                       "finding_ids": fids, "document_evidence": evidence, "gaps": gaps})
    return {"disclaimer": DISCLAIMER, "checks": checks}


def run_pipeline(data):
    data = copy.deepcopy(validate(data))
    stages = {}
    stages["interests"] = interests_stage(data)
    validate(data, stages)
    stages["adaptive"] = adaptive_stage(data, stages["interests"])
    validate(data, stages)
    stages["web"] = web_stage(data, stages["adaptive"])
    validate(data, stages)
    stages["review"] = review_stage(data, stages["web"], stages["adaptive"])
    validate(data, stages)
    return {"schema_version": 1, "fixture_label": "SYNTHETIC", "status": "ok", "stages": stages}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    try:
        args = sys.argv[1:] if argv is None else argv
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with Path(args[0]).open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
