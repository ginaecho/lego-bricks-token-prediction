"""Synthetic, deterministic customer-insights pipeline; Python standard library only."""
import datetime as dt
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(names.split()), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def strings(value, path):
    require(isinstance(value, list), path + " must be an array")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), path + " contains duplicates")


def number(value, path, low=0, high=None):
    require(type(value) in (int, float) and math.isfinite(value),
            path + " must be a finite number")
    require(value >= low and (high is None or value <= high), path + " out of range")


def timestamp(value):
    text(value, "timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("invalid timestamp") from exc
    require(result.tzinfo is not None, "timestamp requires a timezone")
    return result


def records(value, path, names):
    require(isinstance(value, list), path + " must be an array")
    ids = set()
    for row in value:
        fields(row, names, path)
        text(row["id"], path + ".id")
        require(row["id"] not in ids, path + " duplicate id")
        ids.add(row["id"])
    return ids


def validate_input(data):
    fields(data, "schema_version synthetic now customer issues sources steps products events", "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "unsupported schema_version")
    require(data["synthetic"] is True, "fixtures must be labeled synthetic")
    now = timestamp(data["now"])
    customer = data["customer"]
    fields(customer, "experience preferred_format interests completed_steps", "customer")
    require(customer["experience"] in ("beginner", "intermediate", "expert"), "invalid experience")
    require(customer["preferred_format"] in ("text", "video"), "invalid format")
    strings(customer["interests"], "interests")
    strings(customer["completed_steps"], "completed_steps")
    records(data["issues"], "issues", "id text severity")
    for row in data["issues"]:
        text(row["text"], "issue text")
        require(row["severity"] in ("low", "medium", "high", "critical"), "invalid severity")
    records(data["sources"], "sources", "id title text topics")
    for row in data["sources"]:
        text(row["title"], "source title")
        text(row["text"], "source text")
        strings(row["topics"], "source topics")
    step_ids = records(data["steps"], "steps", "id title category level format prerequisites")
    for row in data["steps"]:
        for key in ("title", "category"):
            text(row[key], "step " + key)
        require(row["level"] in ("beginner", "intermediate", "expert"), "invalid step level")
        require(row["format"] in ("text", "video"), "invalid step format")
        strings(row["prerequisites"], "prerequisites")
        require(set(row["prerequisites"]) <= step_ids, "unknown prerequisite")
    require(set(customer["completed_steps"]) <= step_ids, "unknown completed step")
    graph = {row["id"]: row["prerequisites"] for row in data["steps"]}
    pending = set(graph)
    done = set()
    while pending:
        ready = {key for key in pending if set(graph[key]) <= done}
        require(bool(ready), "cyclic prerequisites")
        done.update(ready)
        pending.difference_update(ready)
    product_ids = records(data["products"], "products", "id title category popularity")
    for row in data["products"]:
        text(row["title"], "product title")
        text(row["category"], "product category")
        number(row["popularity"], "popularity", high=1)
    records(data["events"], "events", "id product_id kind at")
    for row in data["events"]:
        require(isinstance(row["product_id"], str) and row["product_id"] in product_ids,
                "unknown event product")
        require(row["kind"] in ("browse", "purchase"), "invalid event kind")
        require(timestamp(row["at"]) <= now, "future event")
    return data


POSITIVE = {"good", "great", "love", "excellent", "helpful", "easy", "happy"}
NEGATIVE = {"bad", "broken", "hate", "terrible", "slow", "confusing", "frustrated", "failed"}
STOP = {"a", "an", "the", "is", "are", "i", "my", "to", "and", "it", "of", "for", "with"}
SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}
LEVEL = {"beginner": 0, "intermediate": 1, "expert": 2}


def tokens(value):
    return re.findall(r"[a-z0-9]+", value.lower())


def sentiment(data):
    rows = []
    for issue in data["issues"]:
        words = tokens(issue["text"])
        matches = []
        for index, word in enumerate(words):
            if word in POSITIVE | NEGATIVE:
                negated = any(w in ("not", "no", "never") for w in words[max(0, index - 2):index])
                base = 1 if word in POSITIVE else -1
                matches.append({"token": word, "index": index,
                                "negated": negated, "contribution": -base if negated else base})
        score = sum(m["contribution"] for m in matches) / max(1, len(matches))
        rows.append({"issue_id": issue["id"], "severity": issue["severity"],
                     "score": score, "label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
                     "evidence": matches, "query_terms": sorted(set(words) - STOP)})
    rows.sort(key=lambda row: (-SEVERITY[row["severity"]], row["score"], row["issue_id"]))
    return {"issues": rows, "priority_order": [row["issue_id"] for row in rows]}


def research(data, insights):
    findings = []
    for issue in insights["issues"]:
        candidates = []
        for source in data["sources"]:
            for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", source["text"]):
                raw = match.group()
                quote = raw.strip()
                if not quote:
                    continue
                start = match.start() + len(raw) - len(raw.lstrip())
                overlap = sorted(set(tokens(quote)) & set(issue["query_terms"]))
                if overlap:
                    candidates.append({"issue_id": issue["issue_id"], "source_id": source["id"],
                                       "start": start, "end": start + len(quote), "quote": quote,
                                       "matched_terms": overlap, "relevance": len(overlap),
                                       "topics": source["topics"][:]})
        candidates.sort(key=lambda row: (-row["relevance"], row["source_id"], row["start"]))
        findings.extend(candidates[:2])
    covered = {row["issue_id"] for row in findings}
    return {"findings": findings,
            "unanswered_issue_ids": [key for key in insights["priority_order"] if key not in covered],
            "focus_categories": sorted({topic for row in findings for topic in row["topics"]})}


def onboarding(data, findings):
    customer = data["customer"]
    focus = findings["focus_categories"] or customer["interests"]
    completed = set(customer["completed_steps"])
    steps = {row["id"]: row for row in data["steps"]}
    targets = {key for key, row in steps.items()
               if row["category"] in focus and LEVEL[row["level"]] <= LEVEL[customer["experience"]]
               and key not in completed}
    selected = set(targets)
    pending = list(targets)
    while pending:
        for dependency in steps[pending.pop()]["prerequisites"]:
            if dependency not in completed and dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    plan = []
    satisfied = set(completed)
    while selected:
        ready = [steps[key] for key in selected if set(steps[key]["prerequisites"]) <= satisfied]
        ready.sort(key=lambda row: (row["format"] != customer["preferred_format"],
                                    LEVEL[row["level"]], row["id"]))
        step = ready[0]
        reason = ("Matches researched category and experience" if step["id"] in targets
                  else "Required prerequisite, even if above experience level")
        reason += "; preferred format" if step["format"] == customer["preferred_format"] else "; available format fallback"
        plan.append({"step_id": step["id"], "prerequisites": step["prerequisites"][:],
                     "category": step["category"], "explanation": reason})
        satisfied.add(step["id"])
        selected.remove(step["id"])
    return {"steps": plan, "focus_categories": sorted(set(focus)),
            "preferred_format": customer["preferred_format"],
            "research_issue_ids": sorted({row["issue_id"] for row in findings["findings"]})}


def behavior(data, plan):
    products = {row["id"]: row for row in data["products"]}
    now = timestamp(data["now"])
    affinity = {}
    event_weights = []
    for event in sorted(data["events"], key=lambda row: row["id"]):
        age = (now - timestamp(event["at"])).total_seconds() / 86400
        weight = (3 if event["kind"] == "purchase" else 1) * 2 ** (-age / 30)
        category = products[event["product_id"]]["category"]
        affinity[category] = affinity.get(category, 0) + weight
        event_weights.append({"event_id": event["id"], "age_days": age, "weight": weight})
    ranking = []
    for product in data["products"]:
        category = product["category"]
        components = {"history": affinity.get(category, 0),
                      "onboarding": 0.5 if category in plan["focus_categories"] else 0,
                      "interest": 0.25 if category in data["customer"]["interests"] else 0,
                      "popularity": 0.1 * product["popularity"]}
        ranking.append({"product_id": product["id"], "score": sum(components.values()),
                        "components": components})
    ranking.sort(key=lambda row: (-row["score"], row["product_id"]))
    return {"cold_start": not bool(data["events"]), "event_weights": event_weights,
            "ranking": ranking, "focus_categories": plan["focus_categories"][:]}


def validate_stage(name, output, data, previous=None):
    """One boundary validator for every stage, including exact extractive citations."""
    if name == "sentiment":
        fields(output, "issues priority_order", name)
        expected = {row["id"] for row in data["issues"]}
        require(isinstance(output["issues"], list), "issues must be an array")
        strings(output["priority_order"], "priority_order")
        require(set(output["priority_order"]) == expected, "issue coverage mismatch")
        require([row.get("issue_id") for row in output["issues"]] == output["priority_order"],
                "priority order mismatch")
        originals = {row["id"]: row for row in data["issues"]}
        for row in output["issues"]:
            fields(row, "issue_id severity score label evidence query_terms", name)
            number(row["score"], "sentiment score", -1, 1)
            require(row["severity"] == originals[row["issue_id"]]["severity"], "severity mismatch")
            require(row["label"] == ("negative" if row["score"] < 0 else "positive" if row["score"] > 0 else "neutral"),
                    "sentiment label mismatch")
            strings(row["query_terms"], "query_terms")
            require(isinstance(row["evidence"], list), "invalid sentiment evidence")
            words = tokens(originals[row["issue_id"]]["text"])
            for evidence in row["evidence"]:
                fields(evidence, "token index negated contribution", "evidence")
                index = evidence["index"]
                require(type(index) is int and 0 <= index < len(words), "invalid evidence index")
                require(evidence["token"] == words[index] and evidence["token"] in POSITIVE | NEGATIVE,
                        "invalid evidence token")
                require(type(evidence["negated"]) is bool, "invalid negation flag")
                expected_contribution = (1 if evidence["token"] in POSITIVE else -1) * (-1 if evidence["negated"] else 1)
                require(evidence["contribution"] == expected_contribution, "invalid contribution")
            require(row["score"] == sum(e["contribution"] for e in row["evidence"]) / max(1, len(row["evidence"])),
                    "evidence score mismatch")
    elif name == "research":
        fields(output, "findings unanswered_issue_ids focus_categories", name)
        sources = {row["id"]: row for row in data["sources"]}
        issues = set(previous["priority_order"])
        queries = {row["issue_id"]: set(row["query_terms"]) for row in previous["issues"]}
        require(isinstance(output["findings"], list), "findings must be an array")
        for row in output["findings"]:
            fields(row, "issue_id source_id start end quote matched_terms relevance topics", name)
            require(row["issue_id"] in issues and row["source_id"] in sources, "invalid citation reference")
            source = sources[row["source_id"]]
            require(type(row["start"]) is int and type(row["end"]) is int and
                    0 <= row["start"] < row["end"] <= len(source["text"]), "invalid citation offsets")
            require(source["text"][row["start"]:row["end"]] == row["quote"], "citation is not exact")
            strings(row["matched_terms"], "matched_terms")
            require(row["matched_terms"] == sorted(set(tokens(row["quote"])) & queries[row["issue_id"]]),
                    "retrieval evidence mismatch")
            require(bool(row["matched_terms"]) and row["relevance"] == len(row["matched_terms"]),
                    "invalid relevance")
            require(row["topics"] == source["topics"], "source topics mismatch")
        strings(output["unanswered_issue_ids"], "unanswered_issue_ids")
        covered = {row["issue_id"] for row in output["findings"]}
        require(set(output["unanswered_issue_ids"]) == issues - covered, "unanswered coverage mismatch")
        require(output["focus_categories"] == sorted({t for row in output["findings"] for t in row["topics"]}),
                "research focus mismatch")
    elif name == "onboarding":
        fields(output, "steps focus_categories preferred_format research_issue_ids", name)
        require(output["focus_categories"] == sorted(set(previous["focus_categories"] or data["customer"]["interests"])),
                "onboarding focus mismatch")
        require(output["preferred_format"] == data["customer"]["preferred_format"], "format mismatch")
        require(output["research_issue_ids"] == sorted({r["issue_id"] for r in previous["findings"]}),
                "research provenance mismatch")
        require(isinstance(output["steps"], list), "plan must be an array")
        steps = {row["id"]: row for row in data["steps"]}
        satisfied = set(data["customer"]["completed_steps"])
        for row in output["steps"]:
            fields(row, "step_id prerequisites category explanation", name)
            key = row["step_id"]
            require(key in steps and key not in satisfied, "invalid plan step")
            require(row["prerequisites"] == steps[key]["prerequisites"] and
                    set(row["prerequisites"]) <= satisfied, "prerequisite order mismatch")
            require(row["category"] == steps[key]["category"], "step category mismatch")
            text(row["explanation"], "explanation")
            satisfied.add(key)
    elif name == "behavior":
        fields(output, "cold_start event_weights ranking focus_categories", name)
        require(type(output["cold_start"]) is bool and output["cold_start"] == (not data["events"]),
                "cold start mismatch")
        require(output["focus_categories"] == previous["focus_categories"], "behavior focus mismatch")
        require(isinstance(output["event_weights"], list), "event weights must be an array")
        require([r.get("event_id") for r in output["event_weights"]] == sorted(r["id"] for r in data["events"]),
                "event coverage mismatch")
        for row in output["event_weights"]:
            fields(row, "event_id age_days weight", name)
            number(row["age_days"], "age_days")
            number(row["weight"], "weight", high=3)
        require(isinstance(output["ranking"], list), "ranking must be an array")
        ids = []
        for row in output["ranking"]:
            fields(row, "product_id score components", name)
            ids.append(row["product_id"])
            fields(row["components"], "history onboarding interest popularity", "components")
            for value in row["components"].values():
                number(value, "score component")
            number(row["score"], "score")
            require(row["score"] == sum(row["components"].values()), "ranking score mismatch")
        require(len(ids) == len(set(ids)) and set(ids) == {r["id"] for r in data["products"]},
                "product coverage mismatch")
        require(output["ranking"] == sorted(output["ranking"], key=lambda r: (-r["score"], r["product_id"])),
                "ranking order mismatch")
    else:
        raise ValidationError("unknown stage")
    return output


def run_pipeline(data):
    validate_input(data)
    outputs = {}
    previous = None
    for name, function in (("sentiment", sentiment), ("research", research),
                           ("onboarding", onboarding), ("behavior", behavior)):
        result = function(data) if previous is None else function(data, previous)
        outputs[name] = validate_stage(name, result, data, previous)
        previous = outputs[name]
    return {"schema_version": 1, "synthetic": True, "status": "ok", "stages": outputs}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=reject_duplicates)
        result = run_pipeline(data)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
