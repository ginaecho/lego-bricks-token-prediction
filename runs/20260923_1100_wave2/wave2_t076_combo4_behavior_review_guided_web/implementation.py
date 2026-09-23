"""Synthetic, offline reference pipeline. Run: python -B implementation.py INPUT.json.

No compliance/certification conclusions are made. Evidence checks are literal,
case-insensitive term checks, not semantic verification. Web retrieval reads only
the supplied, allowlisted synthetic snapshots; it never contacts the network.
"""

import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import sys
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    pass


class Schema:
    VERSION = 1
    MAX_ITEMS = 1000

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def obj(cls, value, fields, path):
        cls.require(isinstance(value, dict), f"{path}: expected object")
        cls.require(set(value) == set(fields.split()), f"{path}: expected fields {fields}")

    @classmethod
    def text(cls, value, path):
        cls.require(isinstance(value, str) and bool(value.strip()) and len(value) <= 200000,
                    f"{path}: expected nonempty bounded string")
        cls.require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value),
                    f"{path}: invalid Unicode surrogate")

    @classmethod
    def identifier(cls, value, path):
        cls.text(value, path)
        cls.require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is not None,
                    f"{path}: invalid identifier")

    @classmethod
    def array(cls, value, path, nonempty=False):
        cls.require(isinstance(value, list) and len(value) <= cls.MAX_ITEMS,
                    f"{path}: expected bounded array")
        cls.require(not nonempty or bool(value), f"{path}: must not be empty")

    @classmethod
    def strings(cls, values, path, nonempty=False):
        cls.array(values, path, nonempty)
        for value in values:
            cls.text(value, path)
        cls.require(len(set(values)) == len(values), f"{path}: duplicate values")

    @classmethod
    def number(cls, value, path, positive=False):
        cls.require(type(value) in (int, float), f"{path}: expected number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        cls.require(finite and (value > 0 if positive else value >= 0),
                    f"{path}: expected finite {'positive' if positive else 'nonnegative'} number")

    @classmethod
    def timestamp(cls, value, path):
        cls.text(value, path)
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            cls.require(result.tzinfo is not None and result.utcoffset() is not None,
                        f"{path}: timezone required")
            return result
        except (ValueError, OverflowError) as exc:
            raise ValidationError(f"{path}: invalid timestamp") from exc

    @classmethod
    def host(cls, host):
        cls.text(host, "allowlisted host")
        cls.require(host == host.lower() and len(host) <= 253 and "." in host,
                    "allowlisted host: expected lowercase domain")
        cls.require(all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                        for label in host.split(".")), "allowlisted host: invalid domain")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValidationError("allowlisted host: IP literals forbidden")
        cls.require(not host.endswith(".localhost"), "localhost forbidden")

    @classmethod
    def url(cls, value, hosts):
        cls.text(value, "URL")
        cls.require(not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
                    and "\\" not in value, "URL: whitespace/control/backslash forbidden")
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme == "https" and parsed.hostname in hosts
                     and parsed.username is None and parsed.password is None
                     and parsed.port in (None, 443) and not parsed.fragment)
        except ValueError as exc:
            raise ValidationError("URL: invalid authority") from exc
        cls.require(valid, "URL: only allowlisted HTTPS URLs without credentials/fragments allowed")
        return urlunsplit(("https", parsed.hostname, parsed.path or "/", parsed.query, ""))

    @classmethod
    def records(cls, values, fields, path, nonempty=False):
        cls.array(values, path, nonempty)
        result = {}
        for row in values:
            cls.obj(row, fields, path)
            cls.identifier(row["id"], path + ".id")
            cls.require(row["id"] not in result, f"{path}: duplicate id")
            result[row["id"]] = row
        return result

    @classmethod
    def input(cls, data):
        cls.obj(data, "schema_version synthetic as_of candidates behavior requirements documents onboarding research",
                "input")
        cls.require(type(data["schema_version"]) is int and data["schema_version"] == cls.VERSION,
                    "unsupported schema_version")
        cls.require(data["synthetic"] is True, "synthetic must be true")
        now = cls.timestamp(data["as_of"], "as_of")
        candidates = cls.records(data["candidates"], "id title category requirement_ids", "candidates", True)
        requirements = cls.records(data["requirements"], "id description terms source_urls", "requirements", True)
        cls.obj(data["research"], "allowlisted_hosts resources", "research")
        hosts = data["research"]["allowlisted_hosts"]
        cls.strings(hosts, "allowlisted_hosts", True)
        for host in hosts:
            cls.host(host)
        for row in requirements.values():
            cls.text(row["description"], "description")
            cls.strings(row["terms"], "terms", True)
            cls.require(len({x.strip().casefold() for x in row["terms"]}) == len(row["terms"]),
                        "terms: case-insensitive duplicates")
            cls.strings(row["source_urls"], "source_urls")
            canonical = [cls.url(url, hosts) for url in row["source_urls"]]
            cls.require(len(set(canonical)) == len(canonical), "source_urls: duplicate canonical URL")
        used = set()
        for row in candidates.values():
            cls.text(row["title"], "title")
            cls.text(row["category"], "category")
            cls.strings(row["requirement_ids"], "requirement_ids", True)
            cls.require(set(row["requirement_ids"]) <= requirements.keys(), "unknown candidate requirement")
            used.update(row["requirement_ids"])
        cls.require(used == requirements.keys(), "every requirement must belong to a candidate")
        cls.obj(data["behavior"], "half_life_days events", "behavior")
        cls.number(data["behavior"]["half_life_days"], "half_life_days", True)
        events = cls.records(data["behavior"]["events"], "id candidate_id kind at", "events")
        for event in events.values():
            cls.require(isinstance(event["candidate_id"], str) and event["candidate_id"] in candidates,
                        "unknown event candidate")
            cls.require(event["kind"] in ("browse", "purchase"), "unknown event kind")
            cls.require(cls.timestamp(event["at"], "event.at") <= now, "future event")
        documents = cls.records(data["documents"], "id requirement_ids text", "documents")
        for doc in documents.values():
            cls.strings(doc["requirement_ids"], "document.requirement_ids", True)
            cls.require(set(doc["requirement_ids"]) <= requirements.keys(), "unknown document requirement")
            cls.text(doc["text"], "document.text")
        cls.obj(data["onboarding"], "steps", "onboarding")
        steps = cls.records(data["onboarding"]["steps"], "id requirement_id depends_on completed", "steps", True)
        for step in steps.values():
            cls.require(isinstance(step["requirement_id"], str) and step["requirement_id"] in requirements,
                        "unknown step requirement")
            cls.strings(step["depends_on"], "depends_on")
            cls.require(set(step["depends_on"]) <= steps.keys(), "unknown prerequisite")
            cls.require(type(step["completed"]) is bool, "completed must be boolean")
        cls.require({s["requirement_id"] for s in steps.values()} == requirements.keys(),
                    "every requirement needs an onboarding step")
        remaining = set(steps)
        visited = set()
        while remaining:
            ready = {sid for sid in remaining if set(steps[sid]["depends_on"]) <= visited}
            cls.require(bool(ready), "onboarding prerequisite cycle")
            visited.update(ready)
            remaining -= ready
        cls.array(data["research"]["resources"], "resources")
        urls = set()
        for resource in data["research"]["resources"]:
            cls.obj(resource, "url content retrieved_at", "resource")
            url = cls.url(resource["url"], hosts)
            cls.require(url not in urls, "duplicate resource URL")
            urls.add(url)
            cls.text(resource["content"], "resource.content")
            cls.require(cls.timestamp(resource["retrieved_at"], "retrieved_at") <= now,
                        "future resource timestamp")
        return data

    @classmethod
    def stage(cls, name, value, previous=None):
        """One shared boundary validator for every stage's typed envelope."""
        cls.obj(value, "schema_version stage items summary", name)
        cls.require(type(value["schema_version"]) is int and value["schema_version"] == cls.VERSION
                    and value["stage"] == name, "invalid stage envelope")
        cls.array(value["items"], name + ".items")
        rows = value["items"]
        if name == "behavior":
            cls.obj(value["summary"], "cold_start requirement_order", "behavior.summary")
            cls.require(type(value["summary"]["cold_start"]) is bool, "invalid cold_start")
            cls.strings(value["summary"]["requirement_order"], "requirement_order", True)
            ids = []
            order = []
            for rank, row in enumerate(rows, 1):
                cls.obj(row, "candidate_id rank score requirement_ids event_ids", name)
                cls.identifier(row["candidate_id"], "candidate_id")
                ids.append(row["candidate_id"])
                cls.require(type(row["rank"]) is int and row["rank"] == rank, "invalid rank")
                cls.number(row["score"], "score")
                cls.strings(row["requirement_ids"], "requirement_ids", True)
                cls.strings(row["event_ids"], "event_ids")
                order.extend(r for r in row["requirement_ids"] if r not in order)
            cls.require(len(set(ids)) == len(ids), "duplicate ranking candidate")
            cls.require(rows == sorted(rows, key=lambda x: (-x["score"], x["candidate_id"])),
                        "invalid ranking order")
            cls.require(order == value["summary"]["requirement_order"], "invalid requirement order")
        elif name == "review":
            cls.obj(value["summary"], "gap_count disclaimer", "review.summary")
            cls.text(value["summary"]["disclaimer"], "disclaimer")
            cls.require([r.get("requirement_id") for r in rows] == previous["summary"]["requirement_order"],
                        "review lost behavior requirement order")
            for row in rows:
                cls.obj(row, "requirement_id status gap_id candidate_ids evidence", name)
                cls.require(row["status"] in ("covered", "gap"), "invalid review status")
                cls.require(row["gap_id"] == ("gap:" + row["requirement_id"]
                                             if row["status"] == "gap" else None), "invalid gap trace")
                cls.strings(row["candidate_ids"], "candidate_ids", True)
                expected = [c["candidate_id"] for c in previous["items"]
                            if row["requirement_id"] in c["requirement_ids"]]
                cls.require(row["candidate_ids"] == expected, "invalid candidate trace")
                cls.array(row["evidence"], "evidence")
                for evidence in row["evidence"]:
                    cls.obj(evidence, "document_id matched_terms missing_terms", "evidence")
                    cls.identifier(evidence["document_id"], "document_id")
                    cls.strings(evidence["matched_terms"], "matched_terms")
                    cls.strings(evidence["missing_terms"], "missing_terms")
                covered = any(not e["missing_terms"] for e in row["evidence"])
                cls.require((row["status"] == "covered") == covered, "review status contradicts evidence")
            cls.require(type(value["summary"]["gap_count"]) is int and
                        value["summary"]["gap_count"] == sum(r["status"] == "gap" for r in rows),
                        "invalid gap count")
        elif name == "guided":
            cls.obj(value["summary"], "total completed ready blocked completion_fraction", "guided.summary")
            reviews = {r["requirement_id"]: r for r in previous["items"]}
            visited = {}
            for row in rows:
                cls.obj(row, "step_id requirement_id prerequisites state review_status gap_id", name)
                cls.identifier(row["step_id"], "step_id")
                cls.require(row["step_id"] not in visited, "duplicate guided step")
                cls.require(isinstance(row["requirement_id"], str) and row["requirement_id"] in reviews,
                            "unknown guided requirement")
                cls.strings(row["prerequisites"], "prerequisites")
                cls.require(set(row["prerequisites"]) <= visited.keys(), "invalid prerequisite order")
                source = reviews[row["requirement_id"]]
                cls.require(row["review_status"] == source["status"] and row["gap_id"] == source["gap_id"],
                            "guided lost review trace")
                cls.require(row["state"] in ("completed", "ready", "blocked"), "invalid step state")
                satisfied = all(visited[p]["state"] == "completed" for p in row["prerequisites"])
                if row["state"] == "completed":
                    cls.require(satisfied and row["review_status"] == "covered", "invalid completed step")
                else:
                    cls.require(row["state"] == ("ready" if satisfied else "blocked"),
                                "invalid readiness")
                visited[row["step_id"]] = row
            cls.require({r["requirement_id"] for r in rows} == reviews.keys(), "guided lost requirements")
            expected = {"total": len(rows),
                        **{state: sum(r["state"] == state for r in rows)
                           for state in ("completed", "ready", "blocked")}}
            expected["completion_fraction"] = expected["completed"] / len(rows) if rows else 0.0
            cls.require(value["summary"] == expected, "invalid progress")
        elif name == "web":
            cls.obj(value["summary"], "eligible_step_ids finding_count retrieval_mode disclaimer", "web.summary")
            eligible = {s["step_id"]: s for s in previous["items"]
                        if s["state"] == "ready" and s["gap_id"] is not None}
            cls.require(value["summary"]["eligible_step_ids"] == list(eligible),
                        "web must target only actionable gaps")
            cls.require(value["summary"]["retrieval_mode"] == "offline_synthetic_snapshots",
                        "invalid retrieval mode")
            cls.text(value["summary"]["disclaimer"], "web.disclaimer")
            seen = set()
            represented = set()
            for row in rows:
                cls.obj(row, "step_id requirement_id gap_id url status matched_terms missing_terms excerpt provenance",
                        name)
                cls.require(isinstance(row["step_id"], str) and row["step_id"] in eligible,
                            "finding targets ineligible step")
                source = eligible[row["step_id"]]
                cls.require(row["requirement_id"] == source["requirement_id"]
                            and row["gap_id"] == source["gap_id"], "finding lost gap trace")
                cls.require(row["url"] is None or isinstance(row["url"], str), "invalid finding URL")
                pair = (row["step_id"], row["url"])
                cls.require(pair not in seen, "duplicate finding")
                seen.add(pair)
                represented.add(row["step_id"])
                cls.strings(row["matched_terms"], "matched_terms")
                cls.strings(row["missing_terms"], "missing_terms")
                cls.require(row["status"] in ("supported", "insufficient", "unavailable", "no_sources"),
                            "invalid finding status")
                if row["status"] in ("supported", "insufficient"):
                    cls.obj(row["provenance"], "url retrieved_at sha256 source", "provenance")
                    cls.require(row["provenance"]["url"] == row["url"], "provenance URL mismatch")
                    cls.timestamp(row["provenance"]["retrieved_at"], "retrieved_at")
                    cls.require(isinstance(row["provenance"]["sha256"], str) and
                                re.fullmatch("[0-9a-f]{64}", row["provenance"]["sha256"]) is not None,
                                "invalid content digest")
                    cls.require(row["provenance"]["source"] == "synthetic_fixture", "invalid source")
                    cls.text(row["excerpt"], "excerpt")
                    cls.require((row["status"] == "supported") == (not row["missing_terms"]),
                                "finding status contradicts terms")
                else:
                    cls.require(row["provenance"] is None and row["excerpt"] is None
                                and not row["matched_terms"], "fabricated provenance")
                    cls.require((row["status"] == "no_sources") == (row["url"] is None),
                                "invalid absent source")
            cls.require(represented == eligible.keys(), "web lost actionable gaps")
            cls.require(type(value["summary"]["finding_count"]) is int and
                        value["summary"]["finding_count"] == len(rows), "invalid finding count")
        else:
            raise ValidationError("unknown stage")
        return value


def envelope(stage, items, summary):
    return {"schema_version": Schema.VERSION, "stage": stage, "items": items, "summary": summary}


def check_terms(text, terms):
    folded = text.casefold()
    matched = [term for term in terms if term.strip().casefold() in folded]
    return matched, [term for term in terms if term not in matched]


def _behavior(data):
    now = Schema.timestamp(data["as_of"], "as_of")
    candidates = {c["id"]: c for c in data["candidates"]}
    items = []
    for cid, candidate in candidates.items():
        contributions = []
        event_ids = []
        for event in sorted(data["behavior"]["events"], key=lambda e: e["id"]):
            related = candidates[event["candidate_id"]]
            similarity = 1.0 if cid == related["id"] else (0.25 if candidate["category"] == related["category"] else 0.0)
            if similarity:
                age = (now - Schema.timestamp(event["at"], "event.at")).total_seconds() / 86400
                weight = 3.0 if event["kind"] == "purchase" else 1.0
                contributions.append(weight * similarity * 2.0 ** (-age / data["behavior"]["half_life_days"]))
                event_ids.append(event["id"])
        items.append({"candidate_id": cid, "rank": 0, "score": round(math.fsum(contributions), 10),
                      "requirement_ids": list(candidate["requirement_ids"]), "event_ids": event_ids})
    items.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    order = []
    for rank, row in enumerate(items, 1):
        row["rank"] = rank
        order.extend(r for r in row["requirement_ids"] if r not in order)
    return envelope("behavior", items, {"cold_start": not any(r["score"] for r in items),
                                       "requirement_order": order})


def _review(data, behavior):
    requirements = {r["id"]: r for r in data["requirements"]}
    items = []
    for rid in behavior["summary"]["requirement_order"]:
        evidence = []
        for doc in sorted(data["documents"], key=lambda d: d["id"]):
            if rid in doc["requirement_ids"]:
                matched, missing = check_terms(doc["text"], requirements[rid]["terms"])
                evidence.append({"document_id": doc["id"], "matched_terms": matched, "missing_terms": missing})
        covered = any(not e["missing_terms"] for e in evidence)
        items.append({"requirement_id": rid, "status": "covered" if covered else "gap",
                      "gap_id": None if covered else "gap:" + rid,
                      "candidate_ids": [c["candidate_id"] for c in behavior["items"] if rid in c["requirement_ids"]],
                      "evidence": evidence})
    return envelope("review", items, {"gap_count": sum(r["status"] == "gap" for r in items),
                                     "disclaimer": "Literal evidence screening only; not certification or a compliance determination."})


def _guided(data, review):
    reviews = {r["requirement_id"]: r for r in review["items"]}
    priority = {rid: index for index, rid in enumerate(reviews)}
    remaining = {s["id"]: s for s in data["onboarding"]["steps"]}
    done = {}
    while remaining:
        available = [s for s in remaining.values() if set(s["depends_on"]) <= done.keys()]
        Schema.require(bool(available), "onboarding prerequisite cycle")
        step = min(available, key=lambda s: (priority[s["requirement_id"]], s["id"]))
        source = reviews[step["requirement_id"]]
        satisfied = all(done[s]["state"] == "completed" for s in step["depends_on"])
        if step["completed"]:
            Schema.require(source["status"] == "covered" and satisfied,
                           f"step {step['id']}: completion requires covered evidence and completed prerequisites")
        state = "completed" if step["completed"] else ("ready" if satisfied else "blocked")
        done[step["id"]] = {"step_id": step["id"], "requirement_id": step["requirement_id"],
                            "prerequisites": list(step["depends_on"]), "state": state,
                            "review_status": source["status"], "gap_id": source["gap_id"]}
        del remaining[step["id"]]
    items = list(done.values())
    summary = {"total": len(items), **{s: sum(r["state"] == s for r in items)
                                      for s in ("completed", "ready", "blocked")}}
    summary["completion_fraction"] = summary["completed"] / len(items) if items else 0.0
    return envelope("guided", items, summary)


def _web(data, guided):
    hosts = data["research"]["allowlisted_hosts"]
    resources = {Schema.url(r["url"], hosts): r for r in data["research"]["resources"]}
    requirements = {r["id"]: r for r in data["requirements"]}
    eligible = [s for s in guided["items"] if s["state"] == "ready" and s["gap_id"] is not None]
    items = []
    for step in eligible:
        req = requirements[step["requirement_id"]]
        for raw_url in req["source_urls"] or [None]:
            url = Schema.url(raw_url, hosts) if raw_url is not None else None
            resource = resources.get(url)
            row = {"step_id": step["step_id"], "requirement_id": req["id"], "gap_id": step["gap_id"],
                   "url": url, "status": "unavailable" if url else "no_sources",
                   "matched_terms": [], "missing_terms": list(req["terms"]), "excerpt": None, "provenance": None}
            if resource:
                matched, missing = check_terms(resource["content"], req["terms"])
                row.update(status="insufficient" if missing else "supported", matched_terms=matched,
                           missing_terms=missing, excerpt=resource["content"][:500],
                           provenance={"url": url, "retrieved_at": resource["retrieved_at"],
                                       "sha256": hashlib.sha256(resource["content"].encode("utf-8")).hexdigest(),
                                       "source": "synthetic_fixture"})
            items.append(row)
    return envelope("web", items, {"eligible_step_ids": [s["step_id"] for s in eligible],
                                  "finding_count": len(items), "retrieval_mode": "offline_synthetic_snapshots",
                                  "disclaimer": "Synthetic source matches are research leads, not verified facts; review gaps remain open."})


def run_pipeline(data):
    Schema.input(data)
    behavior = Schema.stage("behavior", _behavior(data))
    review = Schema.stage("review", _review(data, behavior), behavior)
    guided = Schema.stage("guided", _guided(data, review), review)
    web = Schema.stage("web", _web(data, guided), guided)
    return {"schema_version": Schema.VERSION, "status": "ok", "synthetic": True, "as_of": data["as_of"],
            "stages": {"behavior": behavior, "review": review, "guided": guided, "web": web}}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        Schema.require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("rb") as source:
            raw = source.read(2_000_001)
        Schema.require(len(raw) <= 2_000_000, "input exceeds 2000000 bytes")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda s: (_ for _ in ()).throw(ValidationError("nonfinite JSON number")))
        result = run_pipeline(data)
        code = 0
    except (ValidationError, OSError, ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        result = {"schema_version": Schema.VERSION, "status": "error",
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
