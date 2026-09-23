"""Synthetic support -> cited research -> adaptive onboarding reference CLI.

Python 3 standard library only. Input and all intermediate envelopes use the
same Validator. Character offsets are zero-based, end-exclusive Python string
indices; passage indices are zero-based. No networking or live models are used.
"""

import copy
import json
import re
import sys


VERSION = "1.0"
PRIORITIES = ("urgent", "high", "normal", "low")
EXPERIENCES = ("novice", "intermediate", "expert")
FORMATS = ("reading", "hands_on")
STOPWORDS = frozenset("a an and are as at be by for from i in is it my of on or the to with".split())


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def tokens(text):
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE)) - STOPWORDS


def keyword_hits(text, keywords):
    words = tokens(text)
    return [word for word in keywords if tokens(word) and tokens(word) <= words]


class Validator:
    """One validation boundary for configuration, handoffs, and final output."""

    @staticmethod
    def obj(value, keys, path):
        require(isinstance(value, dict), path + " must be an object")
        require(set(value) == set(keys), path + " has missing or unknown fields")

    @staticmethod
    def text(value, path):
        require(isinstance(value, str) and bool(value.strip()), path + " must be nonblank text")

    @classmethod
    def strings(cls, value, path, nonempty=False):
        require(isinstance(value, list), path + " must be a list")
        require(not nonempty or bool(value), path + " must not be empty")
        for entry in value:
            cls.text(entry, path)
        require(len(value) == len(set(value)), path + " contains duplicates")

    @classmethod
    def input(cls, data):
        cls.obj(data, ("schema_version", "synthetic", "ticket", "triage_config",
                       "corpus", "research_config", "profile", "onboarding_catalog"), "input")
        require(data["schema_version"] == VERSION, "unsupported schema_version")
        require(data["synthetic"] is True, "reference fixtures must be labeled synthetic: true")
        ticket = data["ticket"]
        cls.obj(ticket, ("id", "subject", "body"), "ticket")
        cls.text(ticket["id"], "ticket.id")
        cls.text(ticket["subject"], "ticket.subject")
        require(isinstance(ticket["body"], str), "ticket.body must be text")
        config = data["triage_config"]
        cls.obj(config, ("categories", "default_category", "priority_rules"), "triage_config")
        require(isinstance(config["categories"], list) and bool(config["categories"]),
                "categories must be a nonempty list")
        names = []
        for category in config["categories"]:
            cls.obj(category, ("name", "keywords", "priority", "routing"), "category")
            cls.text(category["name"], "category.name")
            cls.strings(category["keywords"], "category.keywords")
            require(category["priority"] in PRIORITIES, "unknown category priority")
            cls.obj(category["routing"], ("team", "owner"), "routing")
            cls.text(category["routing"]["team"], "routing.team")
            cls.text(category["routing"]["owner"], "routing.owner")
            names.append(category["name"])
        require(len(names) == len(set(names)), "duplicate category")
        require(config["default_category"] in names, "unknown default category")
        require(isinstance(config["priority_rules"], list), "priority_rules must be a list")
        for rule in config["priority_rules"]:
            cls.obj(rule, ("priority", "keywords"), "priority rule")
            require(rule["priority"] in PRIORITIES, "unknown rule priority")
            cls.strings(rule["keywords"], "priority keywords", nonempty=True)
        research = data["research_config"]
        cls.obj(research, ("max_findings",), "research_config")
        require(type(research["max_findings"]) is int and 1 <= research["max_findings"] <= 20,
                "max_findings must be an integer from 1 to 20")
        require(isinstance(data["corpus"], list), "corpus must be a list")
        document_ids = []
        for document in data["corpus"]:
            cls.obj(document, ("id", "title", "passages"), "document")
            cls.text(document["id"], "document.id")
            cls.text(document["title"], "document.title")
            require(isinstance(document["passages"], list), "passages must be a list")
            for passage in document["passages"]:
                cls.text(passage, "passage")
            document_ids.append(document["id"])
        require(len(document_ids) == len(set(document_ids)), "duplicate document id")
        profile = data["profile"]
        cls.obj(profile, ("experience", "preference"), "profile")
        require(profile["experience"] in EXPERIENCES, "unknown experience")
        require(profile["preference"] in FORMATS, "unknown preference")
        catalog = data["onboarding_catalog"]
        require(isinstance(catalog, list), "onboarding_catalog must be a list")
        steps = {}
        for step in catalog:
            cls.obj(step, ("id", "title", "categories", "experience_levels",
                           "evidence_keywords", "prerequisites", "content"), "step")
            cls.text(step["id"], "step.id")
            cls.text(step["title"], "step.title")
            require(step["id"] not in steps, "duplicate step id")
            cls.strings(step["categories"], "step.categories", nonempty=True)
            require(all(c in names or c == "*" for c in step["categories"]), "unknown step category")
            cls.strings(step["experience_levels"], "step.experience_levels", nonempty=True)
            require(all(e in EXPERIENCES for e in step["experience_levels"]), "unknown step experience")
            cls.strings(step["evidence_keywords"], "step.evidence_keywords")
            cls.strings(step["prerequisites"], "step.prerequisites")
            cls.obj(step["content"], FORMATS, "step.content")
            for content in step["content"].values():
                cls.text(content, "step content")
            steps[step["id"]] = step
        for step in catalog:
            require(all(dep in steps for dep in step["prerequisites"]), "unknown prerequisite")
        # Iterative traversal avoids recursion limits on long prerequisite chains.
        pending = set(steps)
        done = set()
        while pending:
            ready = {key for key in pending if set(steps[key]["prerequisites"]) <= done}
            require(bool(ready), "prerequisite cycle")
            pending -= ready
            done |= ready
        return data

    @classmethod
    def citation(cls, citation, data):
        cls.obj(citation, ("document_id", "document_title", "passage_index",
                           "start", "end", "quote"), "citation")
        cls.text(citation["document_id"], "citation.document_id")
        document = next((d for d in data["corpus"] if d["id"] == citation["document_id"]), None)
        require(document is not None, "citation document not found")
        require(citation["document_title"] == document["title"], "citation title mismatch")
        index = citation["passage_index"]
        require(type(index) is int and 0 <= index < len(document["passages"]), "invalid passage index")
        start, end = citation["start"], citation["end"]
        passage = document["passages"][index]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(passage),
                "invalid citation offsets")
        require(citation["quote"] == passage[start:end], "citation quote is not exact")

    @classmethod
    def envelope(cls, output, data, expected_status):
        cls.input(data)
        cls.obj(output, ("schema_version", "synthetic", "status", "ticket_id",
                         "triage", "research", "onboarding"), "output")
        require(expected_status in ("triaged", "researched", "complete"), "invalid expected status")
        require(output["schema_version"] == VERSION and output["synthetic"] is True,
                "output schema mismatch")
        require(output["status"] == expected_status, "invalid stage transition")
        require(output["ticket_id"] == data["ticket"]["id"], "ticket identity changed")
        triage = output["triage"]
        cls.obj(triage, ("category", "priority", "routing", "matched_keywords",
                         "query_terms", "explanation", "classification_method"), "triage")
        category = next((c for c in data["triage_config"]["categories"]
                         if c["name"] == triage["category"]), None)
        require(category is not None, "unknown classified category")
        require(triage["routing"] == category["routing"], "accountable routing mismatch")
        require(triage["priority"] == priority_for(data, category), "priority mismatch")
        cls.text(triage["explanation"], "triage.explanation")
        require(triage["classification_method"] in ("rules", "injected_callable"), "unknown classifier")
        expected_hits = keyword_hits(ticket_text(data), category["keywords"])
        require(triage["matched_keywords"] == expected_hits, "classification evidence mismatch")
        require(triage["query_terms"] == query_for(data, category), "query handoff mismatch")
        if expected_status == "triaged":
            require(output["research"] is None and output["onboarding"] is None,
                    "future stages must be empty")
            return output
        research = output["research"]
        cls.obj(research, ("category", "routing", "query_terms", "findings", "no_evidence",
                           "retrieved_passages_count"), "research")
        for key in ("category", "routing", "query_terms"):
            require(research[key] == triage[key], "triage to research handoff mismatch: " + key)
        require(isinstance(research["findings"], list), "findings must be a list")
        require(len(research["findings"]) <= data["research_config"]["max_findings"], "too many findings")
        require(type(research["no_evidence"]) is bool
                and research["no_evidence"] == (not research["findings"]), "no_evidence mismatch")
        require(type(research["retrieved_passages_count"]) is int
                and research["retrieved_passages_count"] >= len(research["findings"]),
                "invalid retrieved passage count")
        locations = set()
        for index, finding in enumerate(research["findings"], 1):
            cls.obj(finding, ("id", "text", "score", "citation"), "finding")
            require(finding["id"] == "F" + str(index), "invalid finding identity")
            cls.citation(finding["citation"], data)
            require(finding["text"] == finding["citation"]["quote"], "finding is not extractive")
            score = len(tokens(finding["text"]) & set(research["query_terms"]))
            require(type(finding["score"]) is int and finding["score"] == score and score > 0,
                    "finding relevance mismatch")
            location = (finding["citation"]["document_id"], finding["citation"]["passage_index"])
            require(location not in locations, "duplicate finding source")
            locations.add(location)
        if expected_status == "researched":
            require(output["onboarding"] is None, "onboarding must be empty before final stage")
            return output
        onboarding = output["onboarding"]
        cls.obj(onboarding, ("category", "routing", "experience", "preference",
                            "evidence_finding_ids", "steps", "explanation"), "onboarding")
        for key in ("category", "routing"):
            require(onboarding[key] == research[key], "research to onboarding handoff mismatch: " + key)
        for key in ("experience", "preference"):
            require(onboarding[key] == data["profile"][key], "profile adaptation mismatch")
        require(onboarding["evidence_finding_ids"] == [f["id"] for f in research["findings"]],
                "research evidence handoff mismatch")
        cls.text(onboarding["explanation"], "onboarding.explanation")
        require(isinstance(onboarding["steps"], list), "onboarding steps must be a list")
        catalog = {s["id"]: s for s in data["onboarding_catalog"]}
        findings = {f["id"]: f for f in research["findings"]}
        done = set()
        for step in onboarding["steps"]:
            cls.obj(step, ("id", "title", "format", "instruction", "prerequisites",
                           "finding_ids", "citations", "explanation"), "onboarding step")
            cls.text(step["id"], "onboarding step.id")
            require(step["id"] in catalog and step["id"] not in done, "invalid onboarding step identity")
            original = catalog[step["id"]]
            require(step["title"] == original["title"], "step title changed")
            require(step["prerequisites"] == original["prerequisites"], "prerequisites changed")
            require(set(step["prerequisites"]) <= done, "prerequisites not satisfied")
            require(step["format"] == data["profile"]["preference"], "format mismatch")
            require(step["instruction"] == original["content"][step["format"]], "instruction mismatch")
            cls.strings(step["finding_ids"], "step.finding_ids")
            require(all(key in findings for key in step["finding_ids"]), "unknown step finding")
            require(step["citations"] == [findings[key]["citation"] for key in step["finding_ids"]],
                    "onboarding citations mismatch")
            cls.text(step["explanation"], "step.explanation")
            done.add(step["id"])
        return output


def ticket_text(data):
    return data["ticket"]["subject"] + "\n" + data["ticket"]["body"]


def query_for(data, category):
    return sorted(tokens(ticket_text(data)) | tokens(" ".join(category["keywords"])))


def priority_for(data, category):
    choices = [category["priority"]]
    choices.extend(rule["priority"] for rule in data["triage_config"]["priority_rules"]
                   if keyword_hits(ticket_text(data), rule["keywords"]))
    return min(choices, key=PRIORITIES.index)


def triage_ticket(data, classifier=None):
    Validator.input(data)
    categories = data["triage_config"]["categories"]
    text = ticket_text(data)
    method = "rules"
    if classifier is not None:
        require(callable(classifier), "classifier must be callable")
        try:
            name = classifier(copy.deepcopy(data["ticket"]), copy.deepcopy(categories))
        except Exception as exc:
            raise ValidationError("injected classifier failed") from exc
        require(isinstance(name, str) and any(c["name"] == name for c in categories),
                "injected classifier must return a configured category name")
        category = next(c for c in categories if c["name"] == name)
        method = "injected_callable"
    else:
        # Stable configuration order resolves ties.
        category = max(categories, key=lambda c: len(keyword_hits(text, c["keywords"])))
        if not keyword_hits(text, category["keywords"]):
            category = next(c for c in categories if c["name"] == data["triage_config"]["default_category"])
    hits = keyword_hits(text, category["keywords"])
    result = {
        "schema_version": VERSION, "synthetic": True, "status": "triaged",
        "ticket_id": data["ticket"]["id"],
        "triage": {
            "category": category["name"], "priority": priority_for(data, category),
            "routing": copy.deepcopy(category["routing"]), "matched_keywords": hits,
            "query_terms": query_for(data, category), "classification_method": method,
            "explanation": ("Injected category selection; " if method == "injected_callable"
                            else "Keyword-count classification with stable configuration-order ties; ")
                           + ("matched " + ", ".join(hits) if hits else "no keyword matches")
                           + ". Priority uses the most urgent matching rule or category baseline. "
                           + "Configured team and owner remain accountable.",
        },
        "research": None, "onboarding": None,
    }
    return Validator.envelope(result, data, "triaged")


def research_ticket(data, previous):
    Validator.envelope(previous, data, "triaged")
    result = copy.deepcopy(previous)
    triage = result["triage"]
    query = set(triage["query_terms"])
    candidates = []
    for document in data["corpus"]:
        for index, passage in enumerate(document["passages"]):
            score = len(tokens(passage) & query)
            if score:
                candidates.append((score, document["id"], index, document["title"], passage))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    findings = []
    for number, (score, document_id, index, title, passage) in enumerate(
            candidates[:data["research_config"]["max_findings"]], 1):
        # A complete retrieved passage is the extract; no generated claims.
        findings.append({
            "id": "F" + str(number), "text": passage, "score": score,
            "citation": {"document_id": document_id, "document_title": title,
                         "passage_index": index, "start": 0, "end": len(passage), "quote": passage},
        })
    result["research"] = {
        "category": triage["category"], "routing": copy.deepcopy(triage["routing"]),
        "query_terms": list(triage["query_terms"]), "findings": findings,
        "no_evidence": not findings, "retrieved_passages_count": len(candidates),
    }
    result["status"] = "researched"
    return Validator.envelope(result, data, "researched")


def onboard_ticket(data, previous):
    Validator.envelope(previous, data, "researched")
    result = copy.deepcopy(previous)
    research = result["research"]
    profile = data["profile"]
    catalog = {step["id"]: step for step in data["onboarding_catalog"]}
    evidence = {
        step["id"]: [finding for finding in research["findings"]
                     if keyword_hits(finding["text"], step["evidence_keywords"])]
        for step in data["onboarding_catalog"]
    }
    selected = {
        step["id"] for step in data["onboarding_catalog"]
        if (research["category"] in step["categories"] or "*" in step["categories"])
        and profile["experience"] in step["experience_levels"]
        and (not step["evidence_keywords"] or evidence[step["id"]])
    }
    required = set(selected)
    pending = list(selected)
    while pending:
        for dependency in catalog[pending.pop()]["prerequisites"]:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    steps, done = [], set()
    while required - done:
        for step in data["onboarding_catalog"]:
            if step["id"] not in required - done or not set(step["prerequisites"]) <= done:
                continue
            supported = evidence[step["id"]]
            reason = ("Selected for category and experience" if step["id"] in selected
                      else "Required prerequisite; retained even when experience/category filters differ")
            reason += (". Supported by " + ", ".join(f["id"] for f in supported)
                       if supported else ". Configured guidance, not a research-derived claim")
            reason += ". Presented as " + profile["preference"] + " for " + profile["experience"] + "."
            steps.append({
                "id": step["id"], "title": step["title"], "format": profile["preference"],
                "instruction": step["content"][profile["preference"]],
                "prerequisites": list(step["prerequisites"]),
                "finding_ids": [f["id"] for f in supported],
                "citations": [copy.deepcopy(f["citation"]) for f in supported],
                "explanation": reason,
            })
            done.add(step["id"])
    result["onboarding"] = {
        "category": research["category"], "routing": copy.deepcopy(research["routing"]),
        "experience": profile["experience"], "preference": profile["preference"],
        "evidence_finding_ids": [f["id"] for f in research["findings"]], "steps": steps,
        "explanation": ("No retrieved evidence; only eligible configured guidance is offered."
                        if research["no_evidence"] else
                        "Research findings select evidence-dependent guidance; prerequisites are included.")
                       + (" No eligible onboarding steps." if not steps else ""),
    }
    result["status"] = "complete"
    return Validator.envelope(result, data, "complete")


def run_pipeline(data, classifier=None):
    snapshot = copy.deepcopy(Validator.input(data))
    return onboard_ticket(snapshot, research_ticket(snapshot, triage_ticket(snapshot, classifier)))


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(exc)},
                         ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
