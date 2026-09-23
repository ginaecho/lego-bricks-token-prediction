"""Synthetic personalized discovery -> prerequisite-driven onboarding CLI.

Run: python -B implementation.py example_input.json
Only Python's standard library is required. All timestamps must be timezone-aware.
"""

import copy
import json
import math
import sys
from datetime import datetime


class ValidationError(ValueError):
    """An input or inter-stage artifact violated the shared schema."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(value) == set(names), f"{path} must contain exactly: {', '.join(names)}")


def identifier(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be a nonempty string")


def number(value, path, minimum=0, maximum=1e12):
    require(
        type(value) in (int, float) and math.isfinite(value)
        and minimum <= value <= maximum,
        f"{path} must be a finite number in [{minimum}, {maximum}]",
    )


def timestamp(value, path):
    require(isinstance(value, str), f"{path} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{path} must be an ISO 8601 timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            f"{path} must have a timezone")
    return parsed


def unique_strings(value, path):
    require(isinstance(value, list), f"{path} must be an array")
    for entry in value:
        identifier(entry, path)
    require(len(value) == len(set(value)), f"{path} contains duplicates")


def prerequisite_order(steps):
    """Iterative topological ordering; lexical ties are independent of input order."""
    pending = {step["id"]: set(step["prerequisites"]) for step in steps}
    ordered = []
    done = set()
    while pending:
        ready = sorted(key for key, prerequisites in pending.items() if prerequisites <= done)
        require(bool(ready), "steps must form an acyclic prerequisite graph")
        ordered.extend(ready)
        done.update(ready)
        for key in ready:
            del pending[key]
    return ordered


class Schema:
    """One validation boundary for input, behavior handoff, and final output."""

    @staticmethod
    def request(data):
        fields(data, ("schema_version", "fixture_kind", "user_id", "now", "settings",
                      "catalog", "steps", "events", "completed_step_ids"), "input")
        require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                "schema_version must be 1")
        require(data["fixture_kind"] == "synthetic", "fixture_kind must be synthetic")
        identifier(data["user_id"], "user_id")
        now = timestamp(data["now"], "now")
        fields(data["settings"], ("top_n", "half_life_days"), "settings")
        top_n = data["settings"]["top_n"]
        require(type(top_n) is int and 1 <= top_n <= 1000,
                "settings.top_n must be an integer from 1 to 1000")
        number(data["settings"]["half_life_days"], "settings.half_life_days",
               minimum=0.000001, maximum=365000)
        for key in ("catalog", "steps", "events"):
            require(isinstance(data[key], list), f"{key} must be an array")
        require(len(data["catalog"]) <= 1000 and len(data["steps"]) <= 1000
                and len(data["events"]) <= 10000, "reference implementation size limits exceeded")
        step_ids = set()
        for step in data["steps"]:
            fields(step, ("id", "title", "prerequisites"), "step")
            identifier(step["id"], "step.id")
            identifier(step["title"], "step.title")
            require(step["id"] not in step_ids, "duplicate step id")
            step_ids.add(step["id"])
            unique_strings(step["prerequisites"], "step.prerequisites")
        for step in data["steps"]:
            require(set(step["prerequisites"]) <= step_ids, "unknown prerequisite step")
        prerequisite_order(data["steps"])
        product_ids = set()
        for product in data["catalog"]:
            fields(product, ("id", "title", "popularity", "setup_step_ids"), "product")
            identifier(product["id"], "product.id")
            identifier(product["title"], "product.title")
            require(product["id"] not in product_ids, "duplicate product id")
            product_ids.add(product["id"])
            number(product["popularity"], "product.popularity")
            unique_strings(product["setup_step_ids"], "product.setup_step_ids")
            require(set(product["setup_step_ids"]) <= step_ids, "unknown product setup step")
        for event in data["events"]:
            fields(event, ("product_id", "action", "at"), "event")
            identifier(event["product_id"], "event.product_id")
            require(event["product_id"] in product_ids, "event references unknown product")
            require(event["action"] in ("browse", "purchase"), "unsupported event action")
            require(timestamp(event["at"], "event.at") <= now, "future events are not allowed")
        unique_strings(data["completed_step_ids"], "completed_step_ids")
        completed = set(data["completed_step_ids"])
        require(completed <= step_ids, "unknown completed step")
        for step in data["steps"]:
            if step["id"] in completed:
                require(set(step["prerequisites"]) <= completed,
                        "completed steps must include their prerequisites")
        return data

    @staticmethod
    def artifact(data, final=False):
        fields(data, ("schema_version", "status", "request", "behavior", "guided"), "artifact")
        require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                "artifact schema_version must be 1")
        require(data["status"] == "ok", "artifact status must be ok")
        request = Schema.request(data["request"])
        require(data["behavior"] == calculate_behavior(request),
                "behavior handoff does not match validated request")
        if final:
            require(data["guided"] == calculate_guided(request, data["behavior"]),
                    "guided output does not match validated behavior handoff")
        else:
            require(data["guided"] is None, "behavior handoff must not already contain guided output")
        return data


def calculate_behavior(request):
    now = timestamp(request["now"], "now")
    half_life = request["settings"]["half_life_days"]
    contributions = {product["id"]: [] for product in request["catalog"]}
    weights = {"browse": 1.0, "purchase": 4.0}
    for event in request["events"]:
        age_days = (now - timestamp(event["at"], "event.at")).total_seconds() / 86400
        contributions[event["product_id"]].append(
            weights[event["action"]] * 2.0 ** (-age_days / half_life)
        )
    cold_start = not request["events"]
    scores = {
        product_id: math.fsum(sorted(values))
        for product_id, values in contributions.items()
    }
    ranked = sorted(
        request["catalog"],
        key=lambda product: (
            -(product["popularity"] if cold_start else scores[product["id"]]),
            -product["popularity"], product["id"],
        ),
    )[:request["settings"]["top_n"]]
    return {
        "cold_start": cold_start,
        "strategy": "popularity" if cold_start else "recency_weighted",
        "recommendations": [
            {
                "rank": index,
                "product_id": product["id"],
                "score": product["popularity"] if cold_start else scores[product["id"]],
                "event_count": len(contributions[product["id"]]),
            }
            for index, product in enumerate(ranked, start=1)
        ],
    }


def calculate_guided(request, behavior):
    catalog = {product["id"]: product for product in request["catalog"]}
    steps = {step["id"]: step for step in request["steps"]}
    recommended = [item["product_id"] for item in behavior["recommendations"]]
    targets = {}
    for product_id in recommended:
        pending = list(catalog[product_id]["setup_step_ids"])
        seen = set()
        while pending:
            step_id = pending.pop()
            if step_id in seen:
                continue
            seen.add(step_id)
            targets.setdefault(step_id, []).append(product_id)
            pending.extend(steps[step_id]["prerequisites"])
    completed = set(request["completed_step_ids"])
    plan = []
    for step_id in prerequisite_order(request["steps"]):
        if step_id not in targets:
            continue
        step = steps[step_id]
        missing = sorted(set(step["prerequisites"]) - completed)
        state = "completed" if step_id in completed else ("blocked" if missing else "available")
        plan.append({
            "step_id": step_id,
            "title": step["title"],
            "product_ids": targets[step_id],
            "prerequisite_ids": sorted(step["prerequisites"]),
            "status": state,
            "missing_prerequisite_ids": missing,
        })
    completed_count = sum(step["status"] == "completed" for step in plan)
    total = len(plan)
    return {
        "source_product_ids": recommended,
        "steps": plan,
        "next_step_ids": [step["step_id"] for step in plan if step["status"] == "available"],
        "progress": {
            "completed": completed_count,
            "total": total,
            "fraction": completed_count / total if total else 1.0,
            "is_complete": completed_count == total,
        },
    }


def behavior_stage(request):
    validated = copy.deepcopy(Schema.request(request))
    artifact = {
        "schema_version": 1, "status": "ok", "request": validated,
        "behavior": calculate_behavior(validated), "guided": None,
    }
    return Schema.artifact(artifact)


def guided_stage(behavior_artifact):
    validated = copy.deepcopy(Schema.artifact(behavior_artifact))
    validated["guided"] = calculate_guided(validated["request"], validated["behavior"])
    return Schema.artifact(validated, final=True)


def run_pipeline(request):
    return guided_stage(behavior_stage(request))


def reject_constant(value):
    raise ValidationError(f"nonstandard JSON number: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            request = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
        result = run_pipeline(request)
        exit_code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        result = {"schema_version": 1, "status": "error", "message": str(exc)}
        exit_code = 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
