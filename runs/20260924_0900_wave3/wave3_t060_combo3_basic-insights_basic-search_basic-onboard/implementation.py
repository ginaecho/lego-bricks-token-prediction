"""Deterministic synthetic customer-insights -> search -> onboarding pipeline."""

import copy
import difflib
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


THEMES = {
    "accessibility": ({"accessible", "accessibility", "keyboard", "screenreader"},
                      "Review accessibility barriers and test assistive workflows."),
    "usability": ({"confusing", "difficult", "easy", "simple", "intuitive", "usability"},
                  "Simplify the most frequently reported confusing workflow."),
    "performance": ({"slow", "fast", "speed", "performance", "latency"},
                    "Measure and improve the frequently reported slow operations."),
    "pricing": ({"expensive", "cheap", "price", "pricing", "affordable", "budget"},
                "Review pricing clarity and publish value-focused guidance."),
    "reliability": ({"broken", "crash", "error", "reliable", "reliability"},
                    "Reproduce reported failures and prioritize reliability fixes."),
    "support": ({"help", "support", "documentation", "tutorial", "onboarding"},
                "Improve help content for recurring customer questions."),
}
ALIASES = {
    "cheap": "affordable", "budget": "affordable", "inexpensive": "affordable",
    "quick": "fast", "speedy": "fast", "speed": "fast",
    "simple": "easy", "intuitive": "easy", "beginner": "easy",
    "assistance": "support", "help": "support",
    "dependable": "reliable", "screenreader": "accessible",
}
NEGATIVE = {"slow", "confusing", "difficult", "expensive", "broken", "crash", "error"}
STOP = {"a", "an", "the", "to", "for", "of", "and", "i", "it", "is", "my",
        "want", "need", "with", "please", "me", "find"}
STEPS = {"choose_product", "set_goal", "learn_basics", "configure_product",
         "complete_first_task", "review_progress"}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path, allow_empty=False):
    require(isinstance(value, str), path + " must be a string")
    require(len(value) <= 10000 and (allow_empty or bool(value.strip())),
            path + " must contain nonblank text of at most 10000 characters")


def array(value, path):
    require(isinstance(value, list) and len(value) <= 1000,
            path + " must be an array with at most 1000 entries")


def fields(value, names, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(names), path + " has missing or unknown fields")


def unique_strings(value, path):
    array(value, path)
    for item in value:
        text(item, path + " item")
    require(len(set(value)) == len(value), path + " must be unique")


def validate_request(data):
    fields(data, ("schema_version", "synthetic", "fixture_label", "customer",
                  "feedback", "products", "query", "limit"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "synthetic must be true for this reference")
    text(data["fixture_label"], "fixture_label")
    text(data["query"], "query", allow_empty=True)
    require(type(data["limit"]) is int and 1 <= data["limit"] <= 20,
            "limit must be an integer from 1 to 20")
    customer = data["customer"]
    fields(customer, ("id", "goal", "experience", "completed_steps"), "customer")
    text(customer["id"], "customer.id")
    text(customer["goal"], "customer.goal", allow_empty=True)
    require(customer["experience"] in ("beginner", "experienced"),
            "customer.experience must be beginner or experienced")
    unique_strings(customer["completed_steps"], "customer.completed_steps")
    require(set(customer["completed_steps"]) <= STEPS, "unknown completed step")
    for key in ("feedback", "products"):
        array(data[key], key)
        ids = []
        for item in data[key]:
            expected = ("id", "customer_id", "text") if key == "feedback" else (
                "id", "name", "description", "tags")
            fields(item, expected, key + " item")
            for field in expected:
                if field == "tags":
                    unique_strings(item[field], "product.tags")
                else:
                    text(item[field], key + "." + field,
                         allow_empty=field == "description")
            ids.append(item["id"])
        require(len(ids) == len(set(ids)), key + " IDs must be unique")
    return data


def validate(state, stage="input"):
    """Single validation boundary used by all stages and the CLI."""
    if stage == "input":
        return validate_request(state)
    require(stage in ("insights", "search", "onboarding"), "unknown pipeline stage")
    expected = ["status", "schema_version", "input", "insights"]
    if stage in ("search", "onboarding"):
        expected.append("search")
    if stage == "onboarding":
        expected.append("onboarding")
    fields(state, expected, "pipeline")
    require(state["status"] == "ok", "pipeline status must be ok")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "output schema_version must be 1")
    data = validate_request(state["input"])
    insights = state["insights"]
    fields(insights, ("feedback_count", "themes"), "insights")
    require(type(insights["feedback_count"]) is int and
            insights["feedback_count"] == len(data["feedback"]), "feedback_count mismatch")
    array(insights["themes"], "insights.themes")
    feedback_ids = {item["id"] for item in data["feedback"]}
    seen = set()
    covered = set()
    for theme in insights["themes"]:
        fields(theme, ("id", "count", "negative_count", "feedback_ids", "action"), "theme")
        text(theme["id"], "theme.id")
        require(theme["id"] in set(THEMES) | {"other"} and theme["id"] not in seen,
                "invalid or duplicate theme")
        seen.add(theme["id"])
        unique_strings(theme["feedback_ids"], "theme.feedback_ids")
        require(set(theme["feedback_ids"]) <= feedback_ids, "unknown feedback reference")
        covered.update(theme["feedback_ids"])
        require(type(theme["count"]) is int and theme["count"] > 0 and
                theme["count"] == len(theme["feedback_ids"]), "theme count mismatch")
        require(type(theme["negative_count"]) is int and
                0 <= theme["negative_count"] <= theme["count"], "invalid negative_count")
        text(theme["action"], "theme.action")
    require(covered == feedback_ids, "insights must cover all feedback")
    if stage == "insights":
        return state
    search = state["search"]
    fields(search, ("query_terms", "theme_ids", "results"), "search")
    unique_strings(search["query_terms"], "search.query_terms")
    unique_strings(search["theme_ids"], "search.theme_ids")
    require(set(search["theme_ids"]) == seen, "search must carry all insight themes")
    array(search["results"], "search.results")
    require(len(search["results"]) <= data["limit"], "search exceeds limit")
    product_ids = {product["id"] for product in data["products"]}
    result_ids = []
    ranks = []
    for result in search["results"]:
        fields(result, ("product_id", "score", "matched_terms", "matched_themes"), "result")
        text(result["product_id"], "result.product_id")
        require(result["product_id"] in product_ids, "unknown product reference")
        score = result["score"]
        require(type(score) in (int, float) and math.isfinite(score) and score > 0,
                "score must be finite and positive")
        unique_strings(result["matched_terms"], "result.matched_terms")
        unique_strings(result["matched_themes"], "result.matched_themes")
        require(set(result["matched_terms"]) <= set(search["query_terms"]),
                "unknown query term")
        require(set(result["matched_themes"]) <= seen, "unknown matched theme")
        result_ids.append(result["product_id"])
        ranks.append((-score, result["product_id"]))
    require(len(set(result_ids)) == len(result_ids), "duplicate search result")
    require(ranks == sorted(ranks), "search results must be ranked")
    if stage == "search":
        return state
    onboarding = state["onboarding"]
    fields(onboarding, ("customer_id", "recommended_product_id", "theme_ids",
                        "steps", "next_step", "status"), "onboarding")
    require(onboarding["customer_id"] == data["customer"]["id"], "customer mismatch")
    recommendation = result_ids[0] if result_ids else None
    require(onboarding["recommended_product_id"] == recommendation,
            "onboarding must consume top search result")
    require(onboarding["theme_ids"] == search["theme_ids"], "onboarding themes mismatch")
    array(onboarding["steps"], "onboarding.steps")
    step_ids = []
    for step in onboarding["steps"]:
        fields(step, ("id", "instruction"), "step")
        text(step["id"], "step.id")
        require(step["id"] in STEPS and
                step["id"] not in data["customer"]["completed_steps"], "invalid pending step")
        text(step["instruction"], "step.instruction")
        step_ids.append(step["id"])
    require(len(step_ids) == len(set(step_ids)), "duplicate onboarding step")
    require(onboarding["next_step"] == (onboarding["steps"][0] if step_ids else None),
            "next_step must be first pending step")
    require(onboarding["status"] == ("ready" if step_ids else "complete"),
            "onboarding status mismatch")
    return state


def tokens(value):
    return {word for word in re.findall(r"[^\W_]+", value.casefold())
            if word not in STOP}


def normalized(value):
    return {ALIASES.get(word, word) for word in tokens(value)}


def customer_insights(data):
    validate(data)
    grouped = {}
    for feedback in data["feedback"]:
        words = tokens(feedback["text"])
        labels = [label for label, (vocabulary, _) in THEMES.items() if words & vocabulary]
        for label in labels or ["other"]:
            if label not in grouped:
                grouped[label] = {"id": label, "count": 0, "negative_count": 0,
                                  "feedback_ids": [], "action": THEMES[label][1]
                                  if label in THEMES else
                                  "Review uncategorized feedback and refine the theme taxonomy."}
            theme = grouped[label]
            theme["count"] += 1
            theme["negative_count"] += int(bool(words & NEGATIVE))
            theme["feedback_ids"].append(feedback["id"])
    themes = sorted(grouped.values(),
                    key=lambda item: (-item["negative_count"], -item["count"], item["id"]))
    for theme in themes:
        theme["feedback_ids"].sort()
    state = {"status": "ok", "schema_version": 1, "input": copy.deepcopy(data),
             "insights": {"feedback_count": len(data["feedback"]), "themes": themes}}
    return validate(state, "insights")


def product_search(state):
    validate(state, "insights")
    result = copy.deepcopy(state)
    data = state["input"]
    query_terms = normalized(data["query"])
    if not query_terms:
        query_terms = normalized(data["customer"]["goal"])
    themes = state["insights"]["themes"]
    matches = []
    for product in data["products"]:
        raw_words = tokens(" ".join([product["name"], product["description"]] + product["tags"]))
        words = {ALIASES.get(word, word) for word in raw_words}
        exact = query_terms & words
        fuzzy = {term for term in query_terms - exact if len(term) >= 4 and
                 any(len(word) >= 4 and difflib.SequenceMatcher(None, term, word).ratio() >= .84
                     for word in words)}
        matched_themes = [theme["id"] for theme in themes if theme["id"] in THEMES and
                          raw_words & THEMES[theme["id"]][0]]
        # Aggregate feedback is a bounded tie-breaker, never a substitute for query relevance.
        bonus = min(1.0, sum(min(theme["count"], 3) * .1 for theme in themes
                             if theme["id"] in matched_themes))
        score = len(exact) * 3 + len(fuzzy) * 2 + bonus
        if (query_terms and not (exact or fuzzy)) or (not query_terms and not bonus):
            continue
        matches.append({"product_id": product["id"], "score": round(score, 4),
                        "matched_terms": sorted(exact | fuzzy),
                        "matched_themes": matched_themes})
    matches.sort(key=lambda item: (-item["score"], item["product_id"]))
    result["search"] = {"query_terms": sorted(query_terms),
                        "theme_ids": [theme["id"] for theme in themes],
                        "results": matches[:data["limit"]]}
    return validate(result, "search")


def guided_onboarding(state):
    validate(state, "search")
    result = copy.deepcopy(state)
    customer = state["input"]["customer"]
    matches = state["search"]["results"]
    product_id = matches[0]["product_id"] if matches else None
    product = next((p for p in state["input"]["products"] if p["id"] == product_id), None)
    steps = []
    if product is None:
        steps.append({"id": "choose_product",
                      "instruction": "Refine your search or ask support to identify a suitable product."})
    else:
        if not customer["goal"].strip():
            steps.append({"id": "set_goal", "instruction": "Describe your first desired outcome."})
        if customer["experience"] == "beginner":
            steps.append({"id": "learn_basics",
                          "instruction": "Follow the introductory tutorial for " + product["name"] + "."})
        steps.append({"id": "configure_product",
                      "instruction": "Configure " + product["name"] + " for your workflow."})
        steps.append({"id": "complete_first_task",
                      "instruction": "Complete a first task toward: " +
                      (customer["goal"].strip() or "your chosen outcome") + "."})
        steps.append({"id": "review_progress", "instruction": "Review your results and share feedback."})
    pending = [step for step in steps if step["id"] not in customer["completed_steps"]]
    result["onboarding"] = {
        "customer_id": customer["id"], "recommended_product_id": product_id,
        "theme_ids": list(state["search"]["theme_ids"]), "steps": pending,
        "next_step": pending[0] if pending else None,
        "status": "ready" if pending else "complete"}
    return validate(result, "onboarding")


def run_pipeline(data):
    return guided_onboarding(product_search(customer_insights(data)))


def reject_constant(value):
    raise ValidationError("Non-finite JSON numbers are not supported: " + value)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant,
                             object_pairs_hook=reject_duplicate_keys)
        output = run_pipeline(data)
        code = 0
    except (ValidationError, OSError, ValueError, RecursionError) as exc:
        output = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
