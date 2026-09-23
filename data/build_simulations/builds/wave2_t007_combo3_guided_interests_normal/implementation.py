"""Deterministic synthetic onboarding -> discovery -> extractive research."""

import copy
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


STEPS = ("consent", "profile", "preferences")
LABEL = "SYNTHETIC FIXTURE"


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), path="input"):
    require(isinstance(value, dict), path + " must be an object")
    require(set(required) <= set(value), path + " is missing required fields")
    require(set(value) <= set(required) | set(optional), path + " has unknown fields")


def text(value, path, maximum=10000):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonblank text")
    require(len(value) <= maximum, path + " is too long")
    return value


def sequence(value, path, minimum=0, maximum=100):
    require(isinstance(value, list), path + " must be an array")
    require(minimum <= len(value) <= maximum, path + " has invalid length")
    return value


def strings(value, path, minimum=0):
    sequence(value, path, minimum)
    for entry in value:
        text(entry, path, 200)
    require(len(set(value)) == len(value), path + " contains duplicates")
    return value


def integer(value, path, minimum, maximum):
    require(type(value) is int and minimum <= value <= maximum,
            path + " must be an integer in range")


def validate_request(value):
    fields(value, ("schema_version", "fixture_label", "guided", "catalog", "research"))
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "schema_version must be 1")
    require(value["fixture_label"] == LABEL, "fixture_label must identify synthetic data")
    guided = value["guided"]
    fields(guided, ("completed_steps", "responses"), path="guided")
    done = strings(guided["completed_steps"], "completed_steps")
    require(done == list(STEPS[:len(done)]), "steps must follow prerequisite order")
    responses = guided["responses"]
    fields(responses, (), STEPS, "responses")
    require(set(done) <= set(responses), "completed steps need responses")
    if "consent" in responses:
        require(type(responses["consent"]) is bool, "consent must be boolean")
    if "consent" in done:
        require(responses["consent"] is True, "consent must be accepted before progressing")
    if "profile" in responses:
        profile = responses["profile"]
        fields(profile, ("display_name", "region"), path="profile")
        text(profile["display_name"], "display_name", 200)
        text(profile["region"], "region", 200)
    if "preferences" in responses:
        prefs = responses["preferences"]
        fields(prefs, ("interests", "exclude_topics", "exclude_ids", "max_recommendations"),
               path="preferences")
        interests = sequence(prefs["interests"], "interests", 1, 50)
        topics = []
        for interest in interests:
            fields(interest, ("topic", "weight"), path="interest")
            topic = text(interest["topic"], "interest.topic", 200)
            weight = interest["weight"]
            require(type(weight) in (int, float) and 0 < weight <= 100
                    and math.isfinite(weight), "interest weight must be finite and in (0,100]")
            topics.append(topic.casefold())
        require(len(topics) == len(set(topics)), "duplicate interest topics")
        for key in ("exclude_topics", "exclude_ids"):
            strings(prefs[key], key)
            if key == "exclude_topics":
                folded = [entry.casefold() for entry in prefs[key]]
                require(len(folded) == len(set(folded)), "duplicate excluded topics")
        integer(prefs["max_recommendations"], "max_recommendations", 1, 50)
    catalog = sequence(value["catalog"], "catalog")
    ids = set()
    for item in catalog:
        fields(item, ("id", "title", "topics", "passages"), path="catalog item")
        item_id = text(item["id"], "item.id", 200)
        require(item_id not in ids, "duplicate catalog item id")
        ids.add(item_id)
        text(item["title"], "item.title", 500)
        topics = strings(item["topics"], "item.topics", 1)
        require(len({topic.casefold() for topic in topics}) == len(topics),
                "duplicate catalog topics")
        passages = sequence(item["passages"], "passages", 1)
        passage_ids = set()
        for passage in passages:
            fields(passage, ("id", "source", "text"), path="passage")
            passage_id = text(passage["id"], "passage.id", 200)
            require(passage_id not in passage_ids, "duplicate passage id within item")
            passage_ids.add(passage_id)
            source = text(passage["source"], "passage.source", 1000)
            require(source.startswith("synthetic://"), "sources must use synthetic://")
            text(passage["text"], "passage.text")
    research = value["research"]
    fields(research, ("query", "max_findings"), path="research")
    text(research["query"], "research.query", 1000)
    require(bool(tokens(research["query"])), "query must contain searchable words")
    integer(research["max_findings"], "max_findings", 1, 100)
    return copy.deepcopy(value)


def tokens(value):
    return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def guided_setup(request):
    guided = request["guided"]
    done = guided["completed_steps"]
    complete = len(done) == len(STEPS)
    return {
        "completed_steps": list(done),
        "remaining_steps": list(STEPS[len(done):]),
        "next_step": None if complete else STEPS[len(done)],
        "progress": {"completed": len(done), "total": len(STEPS),
                     "fraction": len(done) / len(STEPS)},
        "ready": complete,
        "profile": copy.deepcopy(guided["responses"]["profile"]) if complete else None,
        "preferences": copy.deepcopy(guided["responses"]["preferences"]) if complete else None,
    }


def validate_handoff(request, stage, value, previous=None):
    """All boundaries use this validator, including exact grounded output checks."""
    if stage == "guided":
        require(value == guided_setup(request), "invalid guided handoff")
    elif stage == "interests":
        validate_handoff(request, "guided", previous)
        require(previous["ready"], "recommendations require completed onboarding")
        require(value == interest_recommendations(request, previous), "invalid interests handoff")
    elif stage == "normal":
        guided = guided_setup(request)
        validate_handoff(request, "interests", previous, guided)
        require(value == normal_research(request, previous), "invalid research handoff")
    else:
        raise ValidationError("unknown handoff stage")
    return value


def interest_recommendations(request, guided):
    require(guided["ready"], "recommendations require completed onboarding")
    prefs = guided["preferences"]
    weights = {entry["topic"].casefold(): entry["weight"] for entry in prefs["interests"]}
    excluded_topics = {topic.casefold() for topic in prefs["exclude_topics"]}
    excluded_ids = set(prefs["exclude_ids"])
    recommendations = []
    excluded = []
    for item in request["catalog"]:
        item_topics = {topic.casefold() for topic in item["topics"]}
        reasons = []
        if item["id"] in excluded_ids:
            reasons.append("excluded_id")
        if item_topics & excluded_topics:
            reasons.append("excluded_topic")
        if reasons:
            excluded.append({"item_id": item["id"], "reasons": reasons})
            continue
        matches = sorted(item_topics & set(weights))
        if not matches:
            continue
        score = sum(weights[topic] for topic in matches)
        evidence = [{"topic": topic, "weight": weights[topic]} for topic in matches]
        recommendations.append({
            "item_id": item["id"], "title": item["title"], "score": score,
            "matched_interests": evidence,
            "explanation": "Matched declared interests: " + ", ".join(
                f"{entry['topic']} (weight {entry['weight']})" for entry in evidence),
        })
    recommendations.sort(key=lambda row: (-row["score"], row["item_id"]))
    return {
        "profile": copy.deepcopy(guided["profile"]),
        "preferences": copy.deepcopy(prefs),
        "recommendations": recommendations[:prefs["max_recommendations"]],
        "excluded": sorted(excluded, key=lambda row: row["item_id"]),
    }


def normal_research(request, interests):
    query_tokens = tokens(request["research"]["query"])
    catalog = {item["id"]: item for item in request["catalog"]}
    candidates = []
    eligible = [entry["item_id"] for entry in interests["recommendations"]]
    for rank, item_id in enumerate(eligible):
        for passage in catalog[item_id]["passages"]:
            overlap = sorted(query_tokens & tokens(passage["text"]))
            if overlap:
                finding = {
                    "text": passage["text"],
                    "matched_query_terms": overlap,
                    "retrieval_score": len(overlap),
                    "citation": {
                        "item_id": item_id, "passage_id": passage["id"],
                        "source": passage["source"],
                        "start": 0, "end": len(passage["text"]),
                    },
                }
                candidates.append((-len(overlap), rank, passage["id"], finding))
    candidates.sort(key=lambda row: row[:3])
    findings = [row[3] for row in candidates[:request["research"]["max_findings"]]]
    return {"query": request["research"]["query"], "eligible_item_ids": eligible,
            "findings": findings, "no_evidence": not bool(findings)}


def run_pipeline(raw):
    request = validate_request(raw)
    guided = validate_handoff(request, "guided", guided_setup(request))
    output = {"schema_version": 1, "fixture_label": LABEL,
              "status": "ok" if guided["ready"] else "needs_setup",
              "guided": guided, "interests": None, "normal": None}
    if guided["ready"]:
        interests = validate_handoff(
            request, "interests", interest_recommendations(request, guided), guided)
        output["interests"] = interests
        output["normal"] = validate_handoff(
            request, "normal", normal_research(request, interests), interests)
    return output


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], "r", encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=unique_object)
        output = run_pipeline(raw)
        code = 0
    except (ValueError, OSError, RecursionError) as exc:
        output = {"schema_version": 1, "status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
