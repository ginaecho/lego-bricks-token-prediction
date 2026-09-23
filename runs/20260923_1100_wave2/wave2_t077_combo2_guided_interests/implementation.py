"""Synthetic, deterministic guided-onboarding -> discovery reference pipeline."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_fields(value, fields, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(fields), f"{path}: expected fields {sorted(fields)}")
    return value


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")
    require(value == value.strip(), f"{path}: surrounding whitespace is not allowed")
    return value


def strings(value, path, nonempty=False):
    require(isinstance(value, list), f"{path}: expected array")
    for entry in value:
        text(entry, path)
    require(len(set(value)) == len(value), f"{path}: duplicate values")
    require(not nonempty or bool(value), f"{path}: must not be empty")
    return value


@dataclass(frozen=True)
class DiscoveryContext:
    """Only completed, consented onboarding can create this handoff."""

    user_id: str
    ranked_interests: tuple
    excluded_categories: tuple
    excluded_item_ids: tuple


STEPS = ("profile", "consent", "preferences")


def validate_input(data):
    object_fields(data, ("schema_version", "fixture_label", "onboarding", "discovery"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version: expected integer 1")
    require(data["fixture_label"] == "synthetic", "fixture_label: expected synthetic")
    onboarding = object_fields(data["onboarding"], ("completed_steps",), "onboarding")
    require(isinstance(onboarding["completed_steps"], list), "completed_steps: expected array")
    require(len(onboarding["completed_steps"]) <= len(STEPS), "completed_steps: too many steps")
    discovery = object_fields(data["discovery"], ("limit", "catalog"), "discovery")
    require(type(discovery["limit"]) is int and 1 <= discovery["limit"] <= 100,
            "discovery.limit: expected integer in 1..100")
    require(isinstance(discovery["catalog"], list), "catalog: expected array")
    seen = set()
    for item in discovery["catalog"]:
        object_fields(item, ("id", "title", "category", "tags"), "catalog item")
        for field in ("id", "title", "category"):
            text(item[field], f"catalog.{field}")
        strings(item["tags"], "catalog.tags")
        require(item["id"] not in seen, "catalog: duplicate item id")
        seen.add(item["id"])
    return data


def guided_setup(onboarding):
    answers = {}
    for index, step in enumerate(onboarding["completed_steps"]):
        object_fields(step, ("step_id", "answers"), "completed step")
        require(step["step_id"] == STEPS[index],
                f"step prerequisite: expected {STEPS[index]}")
        value = step["answers"]
        if step["step_id"] == "profile":
            object_fields(value, ("user_id", "display_name"), "profile")
            text(value["user_id"], "profile.user_id")
            text(value["display_name"], "profile.display_name")
        elif step["step_id"] == "consent":
            object_fields(value, ("personalized_discovery",), "consent")
            require(type(value["personalized_discovery"]) is bool,
                    "consent.personalized_discovery: expected boolean")
        else:
            require(answers["consent"]["personalized_discovery"],
                    "preferences prerequisite: personalized discovery consent required")
            object_fields(value, ("ranked_interests", "excluded_categories", "excluded_item_ids"),
                          "preferences")
            strings(value["ranked_interests"], "ranked_interests", nonempty=True)
            strings(value["excluded_categories"], "excluded_categories")
            strings(value["excluded_item_ids"], "excluded_item_ids")
        answers[step["step_id"]] = value
    count = len(answers)
    consent_denied = count >= 2 and not answers["consent"]["personalized_discovery"]
    complete = count == len(STEPS)
    progress = {
        "status": "complete" if complete else "blocked" if consent_denied else "in_progress",
        "completed_steps": list(answers),
        "completed_count": count,
        "total_steps": len(STEPS),
        "progress_percent": round(100 * count / len(STEPS), 2),
        "next_step": None if complete or consent_denied else STEPS[count],
        "blocked_reason": "consent_required" if consent_denied else None,
        "steps": [
            {
                "step_id": name,
                "prerequisites": [] if index == 0 else [STEPS[index - 1]],
                "status": "completed" if name in answers else
                          "available" if index == count and not consent_denied else "locked",
            }
            for index, name in enumerate(STEPS)
        ],
    }
    context = None
    if complete:
        preferences = answers["preferences"]
        context = DiscoveryContext(
            answers["profile"]["user_id"],
            tuple(preferences["ranked_interests"]),
            tuple(preferences["excluded_categories"]),
            tuple(preferences["excluded_item_ids"]),
        )
    return progress, context


def recommend(context, discovery):
    weights = {interest: len(context.ranked_interests) - index
               for index, interest in enumerate(context.ranked_interests)}
    recommendations = []
    excluded_count = 0
    unmatched_count = 0
    for item in discovery["catalog"]:
        if (item["category"] in context.excluded_categories
                or item["id"] in context.excluded_item_ids):
            excluded_count += 1
            continue
        matches = []
        for interest, weight in weights.items():
            sources = []
            if interest == item["category"]:
                sources.append("category")
            if interest in item["tags"]:
                sources.append("tags")
            if sources:
                matches.append({"interest": interest, "weight": weight, "sources": sources})
        if not matches:
            unmatched_count += 1
            continue
        score = sum(match["weight"] for match in matches)
        recommendations.append({
            "item_id": item["id"], "title": item["title"], "category": item["category"],
            "score": score,
            "explanation": {
                "rule": "Sum each matching ranked interest weight once; highest rank has highest weight.",
                "matches": matches,
            },
        })
    recommendations.sort(key=lambda item: (-item["score"], item["item_id"]))
    return {
        "status": "complete", "user_id": context.user_id,
        "applied_preferences": {
            "ranked_interests": list(context.ranked_interests),
            "excluded_categories": list(context.excluded_categories),
            "excluded_item_ids": list(context.excluded_item_ids),
        },
        "eligible_count": len(recommendations),
        "excluded_count": excluded_count,
        "unmatched_count": unmatched_count,
        "recommendations": recommendations[:discovery["limit"]],
    }


def run_pipeline(data):
    data = validate_input(data)
    progress, context = guided_setup(data["onboarding"])
    discovery = (recommend(context, data["discovery"]) if context else {
        "status": "skipped",
        "reason": progress["blocked_reason"] or "onboarding_incomplete",
        "recommendations": [],
    })
    return {
        "schema_version": 1,
        "fixture_label": "synthetic",
        "status": "ok",
        "onboarding": progress,
        "discovery": discovery,
    }


def reject_constant(value):
    raise ValidationError(f"invalid JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        data = json.loads(Path(argv[0]).read_text(encoding="utf-8"),
                          parse_constant=reject_constant, object_pairs_hook=unique_object)
        result = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
