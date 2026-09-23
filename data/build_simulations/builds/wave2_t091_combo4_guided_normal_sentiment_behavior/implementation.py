"""Deterministic synthetic onboarding -> research -> insights -> discovery CLI.

Python standard library only. All times must include a timezone. Citation offsets
are Python Unicode character offsets, with an exclusive end. No live providers.
"""
import copy
import datetime as dt
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


class Validator:
    """Shared input and inter-stage contract validation."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def obj(cls, value, keys, path):
        cls.require(isinstance(value, dict), path + " must be an object")
        cls.require(set(value) == set(keys), path + " has missing or unknown keys")

    @classmethod
    def text(cls, value, path):
        cls.require(isinstance(value, str) and bool(value.strip()),
                    path + " must be nonempty text")

    @classmethod
    def integer(cls, value, lo, hi, path):
        cls.require(type(value) is int and lo <= value <= hi,
                    path + " must be an integer in [%s, %s]" % (lo, hi))

    @classmethod
    def array(cls, value, path):
        cls.require(isinstance(value, list), path + " must be an array")

    @classmethod
    def timestamp(cls, value):
        cls.text(value, "timestamp")
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("invalid timestamp") from exc
        cls.require(parsed.tzinfo is not None, "timestamp must include timezone")
        try:
            return parsed.astimezone(dt.timezone.utc)
        except (ValueError, OverflowError) as exc:
            raise ValidationError("timestamp is outside supported UTC range") from exc

    @classmethod
    def unique_records(cls, records, path):
        cls.array(records, path)
        seen = set()
        for record in records:
            cls.require(isinstance(record, dict), path + " entries must be objects")
            cls.text(record.get("id"), path + ".id")
            cls.require(record["id"] not in seen, path + " has duplicate id")
            seen.add(record["id"])
        return seen

    @classmethod
    def input(cls, value):
        cls.obj(value, ["schema_version", "synthetic", "guided", "normal",
                        "sentiment", "behavior"], "input")
        cls.require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                    "unsupported schema_version")
        cls.require(value["synthetic"] is True, "fixtures must be labeled synthetic")
        cls.obj(value["guided"], ["steps"], "guided")
        steps = value["guided"]["steps"]
        cls.array(steps, "steps")
        cls.require(len(steps) <= 3, "too many onboarding steps")
        for index, step in enumerate(steps):
            cls.obj(step, ["id", "data"], "step")
            cls.require(step["id"] == ("profile", "consent", "research")[index],
                        "onboarding steps must follow prerequisites without duplicates")
            data = step["data"]
            if index == 0:
                cls.obj(data, ["name", "email"], "profile")
                cls.text(data["name"], "name")
                cls.text(data["email"], "email")
                cls.require(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", data["email"]) is not None,
                            "invalid email")
            elif index == 1:
                cls.obj(data, ["accepted"], "consent")
                cls.require(data["accepted"] is True, "consent must be accepted")
            else:
                cls.obj(data, ["query"], "research setup")
                cls.text(data["query"], "query")
                cls.require(bool(tokens(data["query"])), "query must contain words")
        normal = value["normal"]
        cls.obj(normal, ["documents", "top_k"], "normal")
        cls.integer(normal["top_k"], 1, 100, "top_k")
        cls.unique_records(normal["documents"], "documents")
        for document in normal["documents"]:
            cls.obj(document, ["id", "text", "severity"], "document")
            cls.text(document["text"], "document text")
            cls.require(isinstance(document["severity"], str)
                        and document["severity"] in SEVERITY, "invalid severity")
        cls.obj(value["sentiment"], ["max_issues"], "sentiment")
        cls.integer(value["sentiment"]["max_issues"], 1, 100, "max_issues")
        behavior = value["behavior"]
        cls.obj(behavior, ["as_of", "catalog", "events", "limit"], "behavior")
        as_of = cls.timestamp(behavior["as_of"])
        cls.integer(behavior["limit"], 1, 100, "limit")
        ids = cls.unique_records(behavior["catalog"], "catalog")
        for item in behavior["catalog"]:
            cls.obj(item, ["id", "title", "tags"], "catalog item")
            cls.text(item["title"], "item title")
            cls.array(item["tags"], "tags")
            for tag in item["tags"]:
                cls.text(tag, "tag")
            cls.require(len(set(item["tags"])) == len(item["tags"]), "duplicate tags")
        cls.array(behavior["events"], "events")
        for event in behavior["events"]:
            cls.obj(event, ["item_id", "kind", "occurred_at"], "event")
            cls.text(event["item_id"], "event item_id")
            cls.require(event["item_id"] in ids, "event references unknown item")
            cls.require(isinstance(event["kind"], str)
                        and event["kind"] in ("browse", "purchase"), "invalid event kind")
            cls.require(cls.timestamp(event["occurred_at"]) <= as_of,
                        "future events are not allowed")
        return value

    @classmethod
    def stage(cls, value, name):
        cls.obj(value, ["schema_version", "stage", "data"], "stage envelope")
        cls.require(type(value["schema_version"]) is int and value["schema_version"] == 1
                    and value["stage"] == name, "invalid upstream stage")
        data = value["data"]
        keys = {
            "guided": ["completed_steps", "next_step", "progress", "ready", "query"],
            "normal": ["query", "findings"],
            "sentiment": ["query", "issues"],
            "behavior": ["cold_start", "ranking", "as_of"],
        }
        cls.obj(data, keys[name], name)
        if name == "guided":
            completed = data["completed_steps"]
            cls.array(completed, "completed_steps")
            cls.require(completed == list(("profile", "consent", "research")[:len(completed)])
                        and len(completed) <= 3, "invalid progress steps")
            n = len(completed)
            cls.require(data["progress"] == n / 3 and data["ready"] is (n == 3),
                        "inconsistent progress")
            cls.require(data["next_step"] == (None if n == 3 else
                                             ("profile", "consent", "research")[n]),
                        "inconsistent next step")
            cls.require((n == 3 and isinstance(data["query"], str)
                         and bool(tokens(data["query"]))) or (n < 3 and data["query"] is None),
                        "invalid guided query")
        elif name in ("normal", "sentiment"):
            cls.text(data["query"], "upstream query")
            records = data["findings" if name == "normal" else "issues"]
            cls.array(records, "stage records")
            for record in records:
                cls.obj(record, ["citation", "severity", "relevance"] +
                        ([] if name == "normal" else
                         ["score", "label", "contributions", "priority"]), "stage record")
                cls.require(isinstance(record["severity"], str)
                            and record["severity"] in SEVERITY, "invalid upstream severity")
                cls.require(type(record["relevance"]) in (float, int)
                            and 0 <= record["relevance"] <= 1, "invalid relevance")
                citation = record["citation"]
                cls.obj(citation, ["source_id", "start", "end", "quote"], "citation")
                cls.text(citation["source_id"], "source_id")
                cls.text(citation["quote"], "quote")
                cls.integer(citation["start"], 0, 10**12, "citation start")
                cls.integer(citation["end"], citation["start"] + 1, 10**12, "citation end")
                cls.require(citation["end"] - citation["start"] == len(citation["quote"]),
                            "citation span does not match quote")
                if name == "sentiment":
                    score, contributions = score_sentiment(citation["quote"])
                    cls.require(record["score"] == score
                                and record["contributions"] == contributions
                                and record["label"] == sentiment_label(score)
                                and record["priority"] == issue_priority(record, score),
                                "inconsistent sentiment calculation")
        else:
            cls.require(type(data["cold_start"]) is bool, "invalid cold_start")
            cls.timestamp(data["as_of"])
            cls.unique_records(data["ranking"], "ranking")
            for item in data["ranking"]:
                cls.obj(item, ["id", "title", "score", "history_score",
                               "issue_score", "matched_sources"], "ranked item")
                cls.text(item["title"], "title")
                for key in ("score", "history_score", "issue_score"):
                    cls.require(type(item[key]) in (float, int) and math.isfinite(item[key])
                                and item[key] >= 0, "invalid ranking score")
                cls.require(abs(item["score"] - item["history_score"] -
                                item["issue_score"]) < 0.000002, "inconsistent ranking score")
                cls.array(item["matched_sources"], "matched_sources")
                for source in item["matched_sources"]:
                    cls.text(source, "matched source")
        return data


SEVERITY = {"low": 10, "medium": 20, "high": 30, "critical": 40}
LEXICON = {"good": 1, "great": 2, "love": 2, "excellent": 2, "helpful": 1,
           "bad": -1, "broken": -2, "slow": -1, "poor": -1, "hate": -2,
           "unsafe": -2, "failed": -2, "terrible": -2}


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def envelope(name, data):
    value = {"schema_version": 1, "stage": name, "data": data}
    Validator.stage(value, name)
    return value


def guided_stage(config):
    steps = config["steps"]
    count = len(steps)
    return envelope("guided", {
        "completed_steps": [step["id"] for step in steps],
        "next_step": None if count == 3 else ("profile", "consent", "research")[count],
        "progress": count / 3, "ready": count == 3,
        "query": steps[2]["data"]["query"] if count == 3 else None,
    })


def normal_stage(upstream, config):
    guided = Validator.stage(upstream, "guided")
    Validator.require(guided["ready"], "research requires complete onboarding")
    query = set(tokens(guided["query"]))
    findings = []
    for document in config["documents"]:
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", document["text"]):
            raw = match.group()
            quote = raw.strip()
            if not quote:
                continue
            relevance = len(query & set(tokens(quote))) / len(query)
            if relevance == 0:
                continue
            start = match.start() + len(raw) - len(raw.lstrip())
            citation = {"source_id": document["id"], "start": start,
                        "end": start + len(quote), "quote": quote}
            Validator.require(document["text"][start:citation["end"]] == quote,
                              "citation does not match source")
            findings.append({"citation": citation, "severity": document["severity"],
                             "relevance": relevance})
    findings.sort(key=lambda f: (-f["relevance"], f["citation"]["source_id"],
                                f["citation"]["start"]))
    return envelope("normal", {"query": guided["query"],
                              "findings": findings[:config["top_k"]]})


def score_sentiment(text):
    words = tokens(text)
    contributions = []
    # Negation is deliberately bounded to the immediately preceding token.
    for i, word in enumerate(words):
        if word in LEXICON:
            negated = i > 0 and words[i - 1] in ("not", "never", "no")
            contributions.append({"token": word, "token_index": i,
                                  "negated": negated,
                                  "value": LEXICON[word] * (-1 if negated else 1)})
    denominator = sum(abs(c["value"]) for c in contributions)
    score = sum(c["value"] for c in contributions) / denominator if denominator else 0.0
    return score, contributions


def sentiment_label(score):
    return "negative" if score < 0 else "positive" if score > 0 else "neutral"


def issue_priority(finding, score):
    # A ten-point severity gap exceeds the full sentiment/relevance adjustment.
    return SEVERITY[finding["severity"]] + 5 * max(0, -score) + finding["relevance"]


def sentiment_stage(upstream, config):
    research = Validator.stage(upstream, "normal")
    issues = []
    for finding in research["findings"]:
        score, contributions = score_sentiment(finding["citation"]["quote"])
        issues.append(dict(copy.deepcopy(finding), score=score,
                           label=sentiment_label(score), contributions=contributions,
                           priority=issue_priority(finding, score)))
    issues.sort(key=lambda f: (-f["priority"], f["citation"]["source_id"],
                              f["citation"]["start"]))
    return envelope("sentiment", {"query": research["query"],
                                 "issues": issues[:config["max_issues"]]})


def behavior_stage(upstream, config):
    insights = Validator.stage(upstream, "sentiment")
    as_of = Validator.timestamp(config["as_of"])
    history = {item["id"]: 0.0 for item in config["catalog"]}
    for event in config["events"]:
        age = (as_of - Validator.timestamp(event["occurred_at"])).total_seconds() / 86400
        weight = 3 if event["kind"] == "purchase" else 1
        history[event["item_id"]] += weight * 2 ** (-age / 30)
    ranking = []
    for item in config["catalog"]:
        terms = set(tokens(" ".join(item["tags"])))
        issue_score = 0.0
        matched_sources = set()
        for issue in insights["issues"]:
            if issue["score"] >= 0:
                continue
            overlap = len(terms & set(tokens(issue["citation"]["quote"])))
            if overlap:
                issue_score += (overlap / len(terms)) * issue["priority"] / 10
                matched_sources.add(issue["citation"]["source_id"])
        ranking.append({
            "id": item["id"], "title": item["title"],
            "score": round(history[item["id"]] + issue_score, 6),
            "history_score": round(history[item["id"]], 6),
            "issue_score": round(issue_score, 6),
            "matched_sources": sorted(matched_sources),
        })
    ranking.sort(key=lambda item: (-item["score"], item["id"]))
    return envelope("behavior", {"cold_start": not config["events"],
                                "ranking": ranking[:config["limit"]],
                                "as_of": config["as_of"]})


def run_pipeline(value):
    value = Validator.input(value)
    guided = guided_stage(value["guided"])
    stages = {"guided": guided, "normal": None, "sentiment": None, "behavior": None}
    if not guided["data"]["ready"]:
        return {"schema_version": 1, "synthetic": True, "status": "blocked", "stages": stages}
    stages["normal"] = normal_stage(guided, value["normal"])
    stages["sentiment"] = sentiment_stage(stages["normal"], value["sentiment"])
    stages["behavior"] = behavior_stage(stages["sentiment"], value["behavior"])
    return {"schema_version": 1, "synthetic": True, "status": "ok", "stages": stages}


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        Validator.require(len(argv) == 1, "usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=strict_object,
                              parse_constant=reject_constant)
        result = run_pipeline(value)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
