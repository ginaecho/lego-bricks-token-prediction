"""Offline, deterministic FAQ -> journey -> web reference pipeline.

All documents are caller-supplied synthetic snapshots. Nothing is fetched.
Schema version 1 uses exact fields; unknown fields are rejected.
"""

import hashlib
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


def fields(value, names, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(names.split()), location + " has missing or unknown fields")


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), location + " must be nonempty text")
    require(len(value) <= 20000, location + " exceeds text limit")


def strings(value, location, nonempty=False):
    require(isinstance(value, list), location + " must be a list")
    require(len(value) <= 100, location + " exceeds list limit")
    for item in value:
        text(item, location)
    require(len(set(value)) == len(value), location + " contains duplicates")
    require(not nonempty or bool(value), location + " must not be empty")


STOPWORDS = {"a", "an", "the", "to", "how", "do", "i", "my", "is", "and",
             "for", "of", "in", "can", "what", "with", "on", "it"}


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.lower())) - STOPWORDS


def url_allowed(url, domains):
    text(url, "page.url")
    require(not any(c.isspace() or ord(c) < 32 for c in url), "URL contains whitespace")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname in domains
                 and parsed.username is None and parsed.password is None
                 and parsed.port in (None, 443) and not parsed.fragment
                 and "\\" not in url)
    except ValueError:
        valid = False
    require(valid, "URL must use HTTPS and an exact allowlisted host without credentials or fragment")


class Validator:
    """Shared validation for both external input and every stage handoff."""

    @staticmethod
    def input(data):
        fields(data, "schema_version fixture_label question knowledge_base actions completed_actions web", "input")
        require(type(data["schema_version"]) is int and data["schema_version"] == 1, "unsupported schema_version")
        require(data["fixture_label"] == "synthetic", "fixture_label must be synthetic")
        text(data["question"], "question")
        for key in ("knowledge_base", "actions"):
            require(isinstance(data[key], list) and len(data[key]) <= 100, key + " must be a bounded list")
        ids = set()
        topics = set()
        for doc in data["knowledge_base"]:
            fields(doc, "id title text topics", "knowledge_base item")
            for key in ("id", "title", "text"):
                text(doc[key], "knowledge_base." + key)
            strings(doc["topics"], "knowledge_base.topics", True)
            require(doc["id"] not in ids, "duplicate knowledge_base id")
            ids.add(doc["id"])
            topics.update(doc["topics"])
        actions = {}
        for action in data["actions"]:
            fields(action, "id title required_topics prerequisites research_terms", "action")
            text(action["id"], "action.id")
            text(action["title"], "action.title")
            for key in ("required_topics", "prerequisites", "research_terms"):
                strings(action[key], "action." + key, key != "prerequisites")
            require(set(action["required_topics"]) <= topics, "action references unknown topic")
            require(all(tokens(term) for term in action["research_terms"]), "research term has no searchable tokens")
            require(action["id"] not in actions, "duplicate action id")
            actions[action["id"]] = action
        strings(data["completed_actions"], "completed_actions")
        require(set(data["completed_actions"]) <= actions.keys(), "unknown completed action")
        for action in actions.values():
            require(set(action["prerequisites"]) <= actions.keys(), "unknown prerequisite")
        visiting, visited = set(), set()

        def visit(action_id):
            require(action_id not in visiting, "cyclic prerequisites")
            if action_id in visited:
                return
            visiting.add(action_id)
            for dependency in actions[action_id]["prerequisites"]:
                visit(dependency)
            visiting.remove(action_id)
            visited.add(action_id)

        for action_id in actions:
            visit(action_id)
        fields(data["web"], "allowlist pages", "web")
        strings(data["web"]["allowlist"], "web.allowlist", True)
        for domain in data["web"]["allowlist"]:
            require(bool(re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}", domain)),
                    "allowlist must contain lowercase exact domain names")
        pages = data["web"]["pages"]
        require(isinstance(pages, list) and len(pages) <= 100, "pages must be a bounded list")
        page_ids, urls = set(), set()
        for page in pages:
            fields(page, "id url title text", "page")
            for key in ("id", "title", "text"):
                text(page[key], "page." + key)
            url_allowed(page["url"], data["web"]["allowlist"])
            require(page["id"] not in page_ids, "duplicate page id")
            require(page["url"] not in urls, "duplicate page URL")
            page_ids.add(page["id"])
            urls.add(page["url"])
        return data

    @staticmethod
    def faq(result, data):
        fields(result, "status answer citations topics reason", "faq output")
        require(result["status"] in ("answered", "abstained"), "invalid faq status")
        require(isinstance(result["citations"], list), "citations must be a list")
        strings(result["topics"], "faq topics")
        docs = {doc["id"]: doc for doc in data["knowledge_base"]}
        expected = retrieve_faq(data)
        require(result["citations"] == expected, "FAQ citations do not match retrieved evidence")
        for citation in result["citations"]:
            fields(citation, "source_id quote", "citation")
            require(citation["source_id"] in docs, "unknown FAQ source")
            require(citation["quote"] == docs[citation["source_id"]]["text"], "ungrounded FAQ quote")
        if expected:
            expected_topics = sorted({t for c in expected for t in docs[c["source_id"]]["topics"]})
            require(result["status"] == "answered" and result["reason"] is None, "FAQ status mismatch")
            require(result["answer"] == "\n\n".join(c["quote"] for c in expected), "ungrounded FAQ answer")
            require(result["topics"] == expected_topics, "FAQ topics mismatch")
        else:
            require(result == {"status": "abstained", "answer": None, "citations": [],
                               "topics": [], "reason": "no_matching_knowledge"}, "invalid abstention")
        return result

    @staticmethod
    def journey(result, data, faq):
        Validator.faq(faq, data)
        fields(result, "status faq_source_ids topics steps reason", "journey output")
        require(result["faq_source_ids"] == [c["source_id"] for c in faq["citations"]], "lost FAQ provenance")
        require(result["topics"] == faq["topics"], "lost FAQ topics")
        require(isinstance(result["steps"], list), "steps must be a list")
        plan = choose_plan(data, faq)
        if not plan:
            reason = "faq_abstained" if faq["status"] == "abstained" else "no_valid_two_step_journey"
            require(result["status"] == "blocked" and result["steps"] == [] and result["reason"] == reason,
                    "invalid blocked journey")
            return result
        require(result["status"] == "ready" and result["reason"] is None, "invalid journey status")
        require(len(result["steps"]) == 2, "journey must contain exactly two steps")
        done = set(data["completed_actions"])
        for index, (step, action) in enumerate(zip(result["steps"], plan), 1):
            fields(step, "position action_id title prerequisites research_terms", "journey step")
            require(type(step["position"]) is int and step["position"] == index, "invalid step position")
            require(step["action_id"] == action["id"] and step["action_id"] not in done, "invalid action ordering")
            for key in ("title", "prerequisites", "research_terms"):
                require(step[key] == action[key], "journey action data changed")
            require(set(step["prerequisites"]) <= done, "unmet prerequisite")
            require(set(action["required_topics"]) <= set(faq["topics"]), "unsupported journey topic")
            done.add(step["action_id"])
        return result

    @staticmethod
    def web(result, data, faq, journey):
        Validator.journey(journey, data, faq)
        fields(result, "status action_ids findings unfulfilled_action_ids reason", "web output")
        require(result == research_result(data, journey), "web findings or provenance do not match snapshots")
        return result

    @staticmethod
    def output(result, data):
        fields(result, "schema_version fixture_label status faq journey web", "output")
        require(type(result["schema_version"]) is int and result["schema_version"] == 1, "invalid output version")
        require(result["fixture_label"] == "synthetic" and result["status"] == "ok", "invalid output envelope")
        Validator.web(result["web"], data, result["faq"], result["journey"])
        return result


def retrieve_faq(data):
    query = tokens(data["question"])
    ranked = []
    for doc in data["knowledge_base"]:
        score = len(query & tokens(doc["title"] + " " + doc["text"]))
        if score:
            ranked.append((-score, doc["id"], doc))
    return [{"source_id": doc["id"], "quote": doc["text"]}
            for _, _, doc in sorted(ranked)[:2]]


def answer_faq(data):
    citations = retrieve_faq(data)
    if not citations:
        result = {"status": "abstained", "answer": None, "citations": [],
                  "topics": [], "reason": "no_matching_knowledge"}
    else:
        source_ids = {c["source_id"] for c in citations}
        topics = sorted({t for d in data["knowledge_base"] if d["id"] in source_ids for t in d["topics"]})
        result = {"status": "answered", "answer": "\n\n".join(c["quote"] for c in citations),
                  "citations": citations, "topics": topics, "reason": None}
    return Validator.faq(result, data)


def choose_plan(data, faq):
    if faq["status"] != "answered":
        return []
    done = set(data["completed_actions"])
    candidates = sorted((a for a in data["actions"] if a["id"] not in done
                         and set(a["required_topics"]) <= set(faq["topics"])), key=lambda a: a["id"])
    # Look ahead rather than greedily committing to a dead-end first action.
    for first in candidates:
        if not set(first["prerequisites"]) <= done:
            continue
        for second in candidates:
            if second["id"] != first["id"] and set(second["prerequisites"]) <= done | {first["id"]}:
                return [first, second]
    return []


def recommend_journey(data, faq):
    Validator.faq(faq, data)
    plan = choose_plan(data, faq)
    steps = [{"position": i, "action_id": a["id"], "title": a["title"],
              "prerequisites": list(a["prerequisites"]), "research_terms": list(a["research_terms"])}
             for i, a in enumerate(plan, 1)]
    result = {"status": "ready" if plan else "blocked",
              "faq_source_ids": [c["source_id"] for c in faq["citations"]],
              "topics": list(faq["topics"]), "steps": steps,
              "reason": None if plan else ("faq_abstained" if faq["status"] == "abstained"
                                           else "no_valid_two_step_journey")}
    return Validator.journey(result, data, faq)


def research_result(data, journey):
    action_ids = [s["action_id"] for s in journey["steps"]]
    findings, missing = [], []
    for step in journey["steps"]:
        query = set().union(*(tokens(term) for term in step["research_terms"]))
        ranked = []
        for page in data["web"]["pages"]:
            url_allowed(page["url"], data["web"]["allowlist"])
            score = len(query & tokens(page["text"]))
            if score:
                ranked.append((-score, page["id"], page))
        if not ranked:
            missing.append(step["action_id"])
            continue
        _, _, page = sorted(ranked)[0]
        findings.append({
            "action_id": step["action_id"],
            "research_terms": list(step["research_terms"]),
            "faq_source_ids": list(journey["faq_source_ids"]),
            "source_id": page["id"], "url": page["url"], "title": page["title"],
            "quote": page["text"],
            "sha256": hashlib.sha256(page["text"].encode("utf-8")).hexdigest(),
            "ingestion": "supplied_snapshot",
        })
    status = ("blocked" if journey["status"] == "blocked" else
              "abstained" if not findings else "partial" if missing else "complete")
    reason = ("journey_blocked" if status == "blocked" else
              "no_matching_pages" if status == "abstained" else
              "some_actions_without_evidence" if status == "partial" else None)
    return {"status": status, "action_ids": action_ids, "findings": findings,
            "unfulfilled_action_ids": missing, "reason": reason}


def research_web(data, faq, journey):
    Validator.journey(journey, data, faq)
    return Validator.web(research_result(data, journey), data, faq, journey)


def run_pipeline(data):
    Validator.input(data)
    faq = answer_faq(data)
    journey = recommend_journey(data, faq)
    web = research_web(data, faq, journey)
    return Validator.output({"schema_version": 1, "fixture_label": "synthetic",
                             "status": "ok", "faq": faq, "journey": journey, "web": web}, data)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("invalid JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        path = Path(argv[0])
        require(path.stat().st_size <= 2_000_000, "input file exceeds 2 MB")
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        output = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
