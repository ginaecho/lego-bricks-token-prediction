"""Deterministic fictional-onboarding planner; Python standard library only.

CLI: python -B implementation.py example_input.json
Library: evaluate(payload, wording_callback=None).
The callback receives an isolated normalized step list and returns an ID-keyed
mapping of title/description overrides only. It is never called by the CLI.
"""

import copy
import heapq
import json
import sys


class ValidationError(ValueError):
    """Input or callback output violates the planner contract."""


def object_value(value, path, allowed=None):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValidationError(f"{path}: expected an object with string keys")
    if allowed is not None:
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValidationError(f"{path}: unknown keys {unknown}")
    return value


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path}: expected nonblank string")
    return value


def names(value, path):
    if not isinstance(value, list):
        raise ValidationError(f"{path}: expected a list")
    for item in value:
        text(item, path)
    if len(value) != len(set(value)):
        raise ValidationError(f"{path}: duplicate entries")
    return sorted(value)


def present(value):
    return value is not None and value != "" and value != [] and value != {} and (
        not isinstance(value, str) or bool(value.strip())
    )


def normalize_steps(raw):
    if not isinstance(raw, list):
        raise ValidationError("steps: expected a list")
    steps = {}
    for index, entry in enumerate(raw):
        path = f"steps[{index}]"
        object_value(entry, path, {
            "id", "action", "title", "description", "prerequisites",
            "required_fields", "completion_evidence",
        })
        step_id = text(entry.get("id"), f"{path}.id")
        if step_id in steps:
            raise ValidationError(f"steps: duplicate ID {step_id!r}")
        step = {
            "id": step_id,
            "action": text(entry.get("action"), f"{path}.action"),
            "prerequisites": names(entry.get("prerequisites", []), f"{path}.prerequisites"),
            "required_fields": names(entry.get("required_fields", []), f"{path}.required_fields"),
            "completion_evidence": names(
                entry.get("completion_evidence", []), f"{path}.completion_evidence"
            ),
        }
        for key in ("title", "description"):
            if key in entry:
                step[key] = text(entry[key], f"{path}.{key}")
        steps[step_id] = step
    for step_id, step in steps.items():
        unknown = sorted(set(step["prerequisites"]) - set(steps))
        if unknown:
            raise ValidationError(f"{step_id}: unknown prerequisite IDs {unknown}")
    return steps


def topological_order(steps):
    indegree = {key: len(step["prerequisites"]) for key, step in steps.items()}
    children = {key: [] for key in steps}
    for key, step in steps.items():
        for prerequisite in step["prerequisites"]:
            children[prerequisite].append(key)
    ready = [key for key, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        key = heapq.heappop(ready)
        order.append(key)
        for child in sorted(children[key]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(steps):
        remaining = sorted(key for key, degree in indegree.items() if degree)
        raise ValidationError(f"steps: cycle detected; unresolved IDs {remaining}")
    return order


def validate_claims(claims, fields, steps, path):
    object_value(claims, path)
    unknown = sorted(set(claims) - set(steps))
    if unknown:
        raise ValidationError(f"{path}: unknown step IDs {unknown}")
    for key in sorted(claims):
        claim = object_value(claims[key], f"{path}.{key}", {"evidence"})
        evidence = object_value(claim.get("evidence", {}), f"{path}.{key}.evidence")
        missing_prerequisites = sorted(set(steps[key]["prerequisites"]) - set(claims))
        missing_fields = [name for name in steps[key]["required_fields"]
                          if not present(fields.get(name))]
        missing_evidence = [name for name in steps[key]["completion_evidence"]
                            if not present(evidence.get(name))]
        problems = []
        if missing_prerequisites:
            problems.append(f"missing completed prerequisites {missing_prerequisites}")
        if missing_fields:
            problems.append(f"missing required fields {missing_fields}")
        if missing_evidence:
            problems.append(f"missing completion evidence {missing_evidence}")
        if problems:
            raise ValidationError(f"{path}.{key}: invalid completion claim: " + "; ".join(problems))


def evaluate(payload, wording_callback=None):
    """Validate atomically, apply updates, and return a deterministic remaining plan.

    Fields are global literal keys, not dotted paths. Completion records contain
    an evidence object. Nonempty values establish presence, not factual truth.
    Update completions can include a whole prerequisite chain in any input order.
    Existing completion records may be replaced, but never silently removed.
    """
    object_value(payload, "input", {"steps", "progress", "updates"})
    steps = normalize_steps(payload.get("steps"))
    order = topological_order(steps)
    progress = object_value(payload.get("progress", {}), "progress", {"fields", "completed"})
    fields = copy.deepcopy(object_value(progress.get("fields", {}), "progress.fields"))
    completed = copy.deepcopy(object_value(progress.get("completed", {}), "progress.completed"))
    validate_claims(completed, fields, steps, "progress.completed")

    updates = object_value(payload.get("updates", {}), "updates", {"fields", "complete"})
    fields.update(copy.deepcopy(object_value(updates.get("fields", {}), "updates.fields")))
    new_claims = object_value(updates.get("complete", {}), "updates.complete")
    completed.update(copy.deepcopy(new_claims))
    validate_claims(completed, fields, steps, "updated_progress.completed")
    completed = {key: {"evidence": copy.deepcopy(completed[key].get("evidence", {}))}
                 for key in sorted(completed)}

    if wording_callback is not None:
        try:
            overrides = wording_callback(copy.deepcopy([steps[key] for key in order]))
        except Exception as exc:
            raise ValidationError("wording callback failed") from exc
        object_value(overrides, "wording")
        unknown = sorted(set(overrides) - set(steps))
        if unknown:
            raise ValidationError(f"wording: unknown step IDs {unknown}")
        for key in sorted(overrides):
            wording = object_value(overrides[key], f"wording.{key}", {"title", "description"})
            for field, value in wording.items():
                steps[key][field] = text(value, f"wording.{key}.{field}")

    eligible = []
    blocked = {}
    plan = [key for key in order if key not in completed]
    for key in plan:
        reasons = []
        missing = sorted(set(steps[key]["prerequisites"]) - set(completed))
        if missing:
            reasons.append({"code": "missing_prerequisites", "ids": missing})
        missing = [name for name in steps[key]["required_fields"]
                   if not present(fields.get(name))]
        if missing:
            reasons.append({"code": "missing_required_fields", "fields": missing})
        if reasons:
            blocked[key] = reasons
        else:
            eligible.append(key)
    return {
        "eligible_next_steps": eligible,
        "blocked_reasons": blocked,
        "topological_plan": plan,
        "progress": {"fields": fields, "completed": completed},
        "steps": [steps[key] for key in order],
    }


def reject_constant(value):
    raise ValidationError(f"invalid JSON numeric constant {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("usage: python -B implementation.py INPUT.json")
        with open(args[0], encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        result = evaluate(payload)
        output = json.dumps(result, sort_keys=True, allow_nan=False, indent=2)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"error": {"code": "validation_error", "message": str(exc)}},
                         sort_keys=True))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
