"""Deterministic synthetic marketplace pipeline; Python standard library only.

Run: python -B implementation.py example_input.json
Embedding injection: run_pipeline(request, embedder), where embedder accepts a
list of texts and returns equally many finite, nonzero, equal-length vectors.
It is never loaded from JSON or connected to a provider by this implementation.
"""

import copy
import json
import math
import re
import sys
from collections import defaultdict


class ValidationError(ValueError):
    pass


LEVELS = {"beginner": 0, "intermediate": 1, "expert": 2}
LESSONS = {
    "basics": [],
    "search": ["basics"],
    "compare": ["search"],
}
ALIASES = {
    "photography": "camera", "photo": "camera", "photos": "camera",
    "cameras": "camera", "hiking": "outdoor", "camping": "outdoor",
    "outdoors": "outdoor", "lightweight": "portable", "compact": "portable",
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, keys, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(keys), path + " has missing or unknown fields")


def text(value, path, blank=False):
    require(isinstance(value, str), path + " must be text")
    require(blank or bool(value.strip()), path + " must not be blank")
    require(len(value) <= 10000, path + " is too long")


def strings(value, path):
    require(isinstance(value, list), path + " must be an array")
    for entry in value:
        text(entry, path + " entry")
    require(len(set(value)) == len(value), path + " must have unique entries")


def number(value, path, low, high):
    require(type(value) in (int, float) and math.isfinite(value),
            path + " must be a finite number")
    require(low <= value <= high, path + " is out of range")


def tokens(value):
    return [ALIASES.get(word, word) for word in re.findall(r"\w+", value.casefold())]


def validate_request(value):
    obj(value, ("schema_version", "fixture_label", "profile", "query", "catalog",
                "options"), "request")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "schema_version must be 1")
    text(value["fixture_label"], "fixture_label")
    text(value["query"], "query", blank=True)
    profile = value["profile"]
    obj(profile, ("experience", "interests", "preferences", "excluded_ids",
                  "excluded_categories", "completed_lessons"), "profile")
    require(isinstance(profile["experience"], str)
            and profile["experience"] in LEVELS, "invalid experience")
    for field in ("interests", "excluded_ids", "excluded_categories", "completed_lessons"):
        strings(profile[field], "profile." + field)
    require(set(profile["completed_lessons"]) <= set(LESSONS), "unknown completed lesson")
    for lesson in profile["completed_lessons"]:
        require(set(LESSONS[lesson]) <= set(profile["completed_lessons"]),
                "completed lesson is missing a prerequisite")
    require(isinstance(profile["preferences"], dict), "preferences must be an object")
    for tag, weight in profile["preferences"].items():
        text(tag, "preference tag")
        number(weight, "preference weight", 0, 1)
    options = value["options"]
    obj(options, ("limit", "min_score"), "options")
    require(type(options["limit"]) is int and 1 <= options["limit"] <= 100,
            "limit must be an integer from 1 to 100")
    number(options["min_score"], "min_score", 0, 1)
    require(isinstance(value["catalog"], list) and len(value["catalog"]) <= 1000,
            "catalog must be an array with at most 1000 products")
    ids = []
    for product in value["catalog"]:
        obj(product, ("id", "name", "description", "category", "tags",
                      "min_experience", "required_lessons"), "product")
        for field in ("id", "name", "description", "category"):
            text(product[field], "product." + field)
        strings(product["tags"], "product.tags")
        strings(product["required_lessons"], "product.required_lessons")
        require(set(product["required_lessons"]) <= set(LESSONS), "unknown required lesson")
        require(isinstance(product["min_experience"], str)
                and product["min_experience"] in LEVELS, "invalid product experience")
        ids.append(product["id"])
    require(len(ids) == len(set(ids)), "duplicate product id")
    return value


def eligible(product, context):
    profile = context["profile"]
    return (
        product["id"] not in profile["excluded_ids"]
        and product["category"].casefold() not in
        {category.casefold() for category in profile["excluded_categories"]}
        and LEVELS[product["min_experience"]] <= LEVELS[profile["experience"]]
        and set(product["required_lessons"]) <= set(profile["completed_lessons"])
    )


def validate_envelope(value, expected_stage):
    """The same validator is applied on both sides of every stage handoff."""
    obj(value, ("schema_version", "stage", "context", "onboarding", "search",
                "recommendations"), "envelope")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "invalid envelope schema_version")
    require(value["stage"] == expected_stage, "incorrect pipeline stage")
    context = validate_request(value["context"])
    onboarding = value["onboarding"]
    obj(onboarding, ("experience", "search_query", "steps"), "onboarding")
    require(onboarding["experience"] == context["profile"]["experience"],
            "onboarding experience does not match profile")
    text(onboarding["search_query"], "onboarding.search_query", blank=True)
    require(isinstance(onboarding["steps"], list), "steps must be an array")
    seen = set(context["profile"]["completed_lessons"])
    for step in onboarding["steps"]:
        obj(step, ("lesson", "prerequisites", "explanation"), "step")
        require(isinstance(step["lesson"], str) and step["lesson"] in LESSONS,
                "unknown onboarding lesson")
        require(step["lesson"] not in seen, "duplicate onboarding lesson")
        require(step["prerequisites"] == LESSONS[step["lesson"]],
                "incorrect lesson prerequisites")
        require(set(step["prerequisites"]) <= seen, "steps violate prerequisites")
        text(step["explanation"], "step.explanation")
        seen.add(step["lesson"])
    catalog = {p["id"]: p for p in context["catalog"]}
    for field in ("search", "recommendations"):
        require(isinstance(value[field], list), field + " must be an array")
        seen_ids = set()
        for row in value[field]:
            obj(row, ("id", "score", "explanations"), field + " row")
            text(row["id"], "result id")
            require(row["id"] in catalog and row["id"] not in seen_ids,
                    "unknown or duplicate result")
            require(eligible(catalog[row["id"]], context), "ineligible result")
            number(row["score"], "result score", 0, 1)
            strings(row["explanations"], "explanations")
            require(bool(row["explanations"]), "explanations must not be empty")
            seen_ids.add(row["id"])
        require(value[field] == sorted(value[field], key=lambda r: (-r["score"], r["id"])),
                field + " is not ranked deterministically")
    search_ids = {r["id"] for r in value["search"]}
    require(all(r["id"] in search_ids for r in value["recommendations"]),
            "recommendations must originate in search")
    require(len(value["recommendations"]) <= context["options"]["limit"],
            "too many recommendations")
    require(all(r["score"] >= context["options"]["min_score"] for r in value["search"]),
            "search result below threshold")
    if expected_stage == "adaptive":
        require(not value["search"] and not value["recommendations"],
                "adaptive stage cannot contain downstream results")
    elif expected_stage == "semantic":
        require(not value["recommendations"], "semantic stage cannot contain recommendations")
    return value


def adaptive_onboarding(request):
    context = copy.deepcopy(validate_request(request))
    profile = context["profile"]
    # Plans never count as completed lessons and never silently grant eligibility.
    targets = ["basics", "search"] if profile["experience"] == "beginner" else ["search"]
    if profile["preferences"] or profile["experience"] == "expert":
        targets.append("compare")
    steps = []
    scheduled = set(profile["completed_lessons"])

    def schedule(lesson):
        if lesson in scheduled:
            return
        for prerequisite in LESSONS[lesson]:
            schedule(prerequisite)
        reason = ("Compare product attributes against your weighted preferences."
                  if lesson == "compare" and profile["preferences"]
                  else f"{lesson.title()} supports {profile['experience']} discovery.")
        steps.append({"lesson": lesson, "prerequisites": LESSONS[lesson][:],
                      "explanation": reason})
        scheduled.add(lesson)

    for target in targets:
        schedule(target)
    query = context["query"].strip()
    if not query:
        query = " ".join(profile["interests"])
    envelope = {
        "schema_version": 1, "stage": "adaptive", "context": context,
        "onboarding": {"experience": profile["experience"], "search_query": query,
                       "steps": steps},
        "search": [], "recommendations": [],
    }
    return validate_envelope(envelope, "adaptive")


def embedding_scores(embedder, query, documents):
    try:
        vectors = embedder([query] + documents)
    except Exception as exc:
        raise ValidationError("embedding callable failed") from exc
    require(isinstance(vectors, list) and len(vectors) == len(documents) + 1,
            "embedding batch has wrong shape")
    normalized = []
    dimension = None
    for vector in vectors:
        require(isinstance(vector, (list, tuple)) and 1 <= len(vector) <= 4096,
                "embedding vector has wrong shape")
        if dimension is None:
            dimension = len(vector)
        require(len(vector) == dimension, "embedding dimensions differ")
        for entry in vector:
            number(entry, "embedding component", -1e100, 1e100)
        norm = math.hypot(*vector)
        require(norm > 0, "embedding vector must be nonzero")
        normalized.append([entry / norm for entry in vector])
    return [max(0.0, min(1.0, sum(a * b for a, b in zip(normalized[0], vector))))
            for vector in normalized[1:]]


def semantic_search(previous, embedder=None):
    envelope = copy.deepcopy(validate_envelope(previous, "adaptive"))
    context = envelope["context"]
    products = [p for p in context["catalog"] if eligible(p, context)]
    documents = [" ".join([p["name"], p["description"], p["category"]] + p["tags"])
                 for p in products]
    # Inverted index plus smoothed inverse-document-frequency query coverage.
    index = defaultdict(set)
    for position, document in enumerate(documents):
        for term in set(tokens(document)):
            index[term].add(position)
    query = envelope["onboarding"]["search_query"]
    terms = sorted(set(tokens(query)))
    weights = {term: 1 + math.log((1 + len(products)) / (1 + len(index[term])))
               for term in terms}
    denominator = sum(weights.values())
    semantic = embedding_scores(embedder, query, documents) if embedder is not None else None
    rows = []
    for position, product in enumerate(products):
        matched = sorted(term for term in terms if position in index[term])
        lexical = sum(weights[term] for term in matched) / denominator if denominator else 0.0
        score = lexical if semantic is None else 0.5 * lexical + 0.5 * semantic[position]
        score = round(score, 8)
        # A blank query is explicit browsing, not a fabricated textual match.
        if terms and score <= 0:
            continue
        if score < context["options"]["min_score"]:
            continue
        reasons = ["Matched indexed concepts: " + ", ".join(matched)] if matched else []
        if semantic is not None:
            reasons.append(f"Injected embedding cosine similarity: {semantic[position]:.6f}")
        if not reasons:
            reasons.append("Browse candidate; no search concepts were supplied.")
        rows.append({"id": product["id"], "score": score, "explanations": reasons})
    envelope["stage"] = "semantic"
    envelope["search"] = sorted(rows, key=lambda row: (-row["score"], row["id"]))
    return validate_envelope(envelope, "semantic")


def recommend_interests(previous):
    envelope = copy.deepcopy(validate_envelope(previous, "semantic"))
    context = envelope["context"]
    profile = context["profile"]
    catalog = {p["id"]: p for p in context["catalog"]}
    interests = set(tokens(" ".join(profile["interests"])))
    preference_total = sum(profile["preferences"].values())
    rows = []
    for result in envelope["search"]:
        product = catalog[result["id"]]
        require(eligible(product, context), "excluded recommendation")
        attributes = set(tokens(" ".join([product["category"]] + product["tags"])))
        matched_interests = sorted(interests & attributes)
        matched_preferences = sorted(
            key for key, weight in profile["preferences"].items()
            if weight > 0 and set(tokens(key)) and set(tokens(key)) <= attributes
        )
        interest_score = len(matched_interests) / len(interests) if interests else 0
        preference_score = (sum(profile["preferences"][k] for k in matched_preferences)
                            / preference_total if preference_total else 0)
        weighted = [(0.5, result["score"])]
        if interests:
            weighted.append((0.3, interest_score))
        if preference_total:
            weighted.append((0.2, preference_score))
        score = round(sum(weight * value for weight, value in weighted)
                      / sum(weight for weight, _ in weighted), 8)
        reasons = [f"Search relevance: {result['score']:.8f}"]
        if matched_interests:
            reasons.append("Interest concepts in category/tags: " + ", ".join(matched_interests))
        if matched_preferences:
            reasons.append("Preferred category/tag concepts: " + ", ".join(matched_preferences))
        if not matched_interests and not matched_preferences:
            reasons.append("No matching interest or preferred attribute; ranked by search evidence.")
        rows.append({"id": product["id"], "score": score, "explanations": reasons})
    envelope["stage"] = "interests"
    envelope["recommendations"] = sorted(rows, key=lambda r: (-r["score"], r["id"]))[
        :context["options"]["limit"]]
    return validate_envelope(envelope, "interests")


def run_pipeline(request, embedder=None):
    result = recommend_interests(semantic_search(adaptive_onboarding(request), embedder))
    return {"status": "ok", **result}


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
        with open(args[0], encoding="utf-8") as source:
            request = json.load(source, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        output = run_pipeline(request)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
