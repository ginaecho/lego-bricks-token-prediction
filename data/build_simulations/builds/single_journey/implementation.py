"""Deterministic, local customer-journey recommendations; Python stdlib only."""

import argparse
import copy
import json
import math
import sys


class ValidationError(ValueError):
    pass


def _object(value, label, required, optional=()):
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValidationError(
            f"{label}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}"
        )


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a nonempty string")
    return value


def _strings(value, label):
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a list")
    for item in value:
        _text(item, label)
    if len(set(value)) != len(value):
        raise ValidationError(f"{label} must not contain duplicates")
    return value


def _references(values, valid, label):
    unknown = set(values) - set(valid)
    if unknown:
        raise ValidationError(f"{label}: unknown references {sorted(unknown)}")


def _validate(payload):
    _object(payload, "input", ("journey", "user"))
    journey, user = payload["journey"], payload["user"]
    _object(journey, "journey", ("stages", "actions"))
    _object(
        user, "user", ("completed_stages",),
        ("completed_actions", "preferences", "excluded_actions", "excluded_tags"),
    )
    stages, actions = {}, {}
    for name, collection, required, optional in (
        ("stages", stages, ("id", "name"), ("prerequisites",)),
        ("actions", actions, ("id", "name", "stage_id"),
         ("prerequisites", "tags", "priority")),
    ):
        entries = journey[name]
        if not isinstance(entries, list) or not entries:
            raise ValidationError(f"journey.{name} must be a nonempty list")
        for entry in entries:
            _object(entry, name, required, optional)
            identifier = _text(entry["id"], f"{name}.id")
            _text(entry["name"], f"{name}.name")
            if identifier in collection:
                raise ValidationError(f"duplicate {name} ID: {identifier}")
            _strings(entry.get("prerequisites", []), f"{name}.prerequisites")
            if name == "actions":
                _text(entry["stage_id"], "actions.stage_id")
                _strings(entry.get("tags", []), "actions.tags")
                priority = entry.get("priority", 0)
                if (
                    isinstance(priority, bool)
                    or not isinstance(priority, (int, float))
                    or not -1_000_000 <= priority <= 1_000_000
                    or not math.isfinite(priority)
                ):
                    raise ValidationError("actions.priority must be finite in [-1000000, 1000000]")
            collection[identifier] = entry
    members = {identifier: [] for identifier in stages}
    for identifier, stage in stages.items():
        _references(stage.get("prerequisites", []), stages, f"stage {identifier}")
    for identifier, action in actions.items():
        _references([action["stage_id"]], stages, f"action {identifier} stage")
        _references(action.get("prerequisites", []), actions, f"action {identifier}")
        members[action["stage_id"]].append(identifier)
    for identifier, group in members.items():
        if not group:
            raise ValidationError(f"stage {identifier} must contain at least one action")

    # Completion depends on every stage action. This graph also catches mixed
    # stage/action cycles, such as an early action depending on a later stage.
    graph = {}
    for identifier, stage in stages.items():
        graph[("stage", identifier)] = {
            ("stage", prerequisite) for prerequisite in stage.get("prerequisites", [])
        } | {("action", action_id) for action_id in members[identifier]}
    for identifier, action in actions.items():
        graph[("action", identifier)] = {
            ("action", prerequisite) for prerequisite in action.get("prerequisites", [])
        } | {
            ("stage", prerequisite)
            for prerequisite in stages[action["stage_id"]].get("prerequisites", [])
        }
    dependents = {node: [] for node in graph}
    pending = {node: len(dependencies) for node, dependencies in graph.items()}
    for node, dependencies in graph.items():
        for dependency in dependencies:
            dependents[dependency].append(node)
    queue = [node for node, count in pending.items() if count == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for dependent in dependents[node]:
            pending[dependent] -= 1
            if pending[dependent] == 0:
                queue.append(dependent)
    if visited != len(graph):
        raise ValidationError("cyclic prerequisites prevent a valid journey")

    for name in ("completed_stages", "completed_actions", "preferences",
                 "excluded_actions", "excluded_tags"):
        _strings(user.get(name, []), f"user.{name}")
    _references(user["completed_stages"], stages, "user.completed_stages")
    _references(user.get("completed_actions", []), actions, "user.completed_actions")
    _references(user.get("excluded_actions", []), actions, "user.excluded_actions")
    return stages, actions, members, user


def recommend(payload, explanation_callback=None):
    """Return JSON-compatible recommendations without mutating the input.

    Callback receives a deep-copied result and returns
    {"explanations": [{"action_id": str, "text": str}, ...]}.
    No callback is invoked unless explicitly injected by a Python caller.
    """
    stages, actions, members, user = _validate(payload)
    preferences = set(user.get("preferences", []))
    excluded = set(user.get("excluded_actions", []))
    excluded_tags = set(user.get("excluded_tags", []))

    def normalize(done_stages, done_actions):
        changed = True
        while changed:
            changed = False
            for stage_id, stage in stages.items():
                if stage_id in done_stages:
                    done_actions.update(members[stage_id])
                elif (
                    set(members[stage_id]) <= done_actions
                    and set(stage.get("prerequisites", [])) <= done_stages
                ):
                    done_stages.add(stage_id)
                    changed = True

    def blockers(action_id, done_stages, done_actions):
        action = actions[action_id]
        reasons = []
        if action_id in excluded:
            reasons.append({"kind": "excluded_action", "id": action_id})
        for tag in sorted(set(action.get("tags", [])) & excluded_tags):
            reasons.append({"kind": "excluded_tag", "id": tag})
        for stage_id in sorted(stages[action["stage_id"]].get("prerequisites", [])):
            if stage_id not in done_stages:
                reasons.append({"kind": "required_stage", "id": stage_id})
        for required_id in sorted(action.get("prerequisites", [])):
            if required_id not in done_actions:
                reasons.append({"kind": "required_action", "id": required_id})
        return reasons

    def ranked(done_stages, done_actions):
        candidates = []
        for action_id, action in actions.items():
            if action_id in done_actions or blockers(action_id, done_stages, done_actions):
                continue
            matching = sorted(preferences & set(action.get("tags", [])))
            priority = action.get("priority", 0)
            required_stages = sorted(stages[action["stage_id"]].get("prerequisites", []))
            required_actions = sorted(action.get("prerequisites", []))
            candidates.append({
                "action_id": action_id,
                "name": action["name"],
                "stage_id": action["stage_id"],
                "score": priority + 10 * len(matching),
                "matched_preferences": matching,
                "prerequisite_explanation": {
                    "completed_required_stages": required_stages,
                    "completed_required_actions": required_actions,
                    "text": (
                        "All required stages and actions are complete."
                        if required_stages or required_actions else "No prerequisites."
                    ),
                },
            })
        return sorted(candidates, key=lambda item: (-item["score"], item["action_id"]))

    done_stages = set(user["completed_stages"])
    done_actions = set(user.get("completed_actions", []))
    normalize(done_stages, done_actions)
    next_actions = ranked(done_stages, done_actions)
    plan_stages, plan_actions = set(done_stages), set(done_actions)
    plan = []
    for step in range(1, 3):
        candidates = ranked(plan_stages, plan_actions)
        if not candidates:
            break
        chosen = candidates[0]
        plan.append({
            **chosen, "step": step,
            "assumed_completed_plan_actions": [item["action_id"] for item in plan],
        })
        plan_actions.add(chosen["action_id"])
        normalize(plan_stages, plan_actions)

    reachable_stages, reachable_actions = set(done_stages), set(done_actions)
    while True:
        candidates = ranked(reachable_stages, reachable_actions)
        if not candidates:
            break
        reachable_actions.update(item["action_id"] for item in candidates)
        normalize(reachable_stages, reachable_actions)
    completed = len(done_stages) == len(stages)
    result = {
        "status": "completed" if completed else ("ready" if next_actions else "blocked"),
        "next_actions": next_actions,
        "two_step_plan": plan,
        "completed_stages": sorted(done_stages),
        "progression_possible": len(reachable_stages) == len(stages),
        "unreachable_stages": sorted(set(stages) - reachable_stages),
        "blocked_actions": [
            {"action_id": action_id, "reasons": blockers(action_id, done_stages, done_actions)}
            for action_id in sorted(actions)
            if action_id not in done_actions and blockers(action_id, done_stages, done_actions)
        ],
        "explanation_status": "not_requested",
        "supplemental_explanations": [],
    }
    if explanation_callback is not None:
        try:
            supplement = explanation_callback(copy.deepcopy(result))
            _object(supplement, "callback result", ("explanations",))
            if not isinstance(supplement["explanations"], list):
                raise ValidationError("callback explanations must be a list")
            allowed = {item["action_id"] for item in next_actions + plan}
            seen = set()
            for entry in supplement["explanations"]:
                _object(entry, "callback explanation", ("action_id", "text"))
                action_id = _text(entry["action_id"], "callback action_id")
                _text(entry["text"], "callback text")
                if action_id not in allowed or action_id in seen:
                    raise ValidationError("callback action ID is not recommended or is duplicated")
                seen.add(action_id)
            result["supplemental_explanations"] = copy.deepcopy(supplement["explanations"])
            result["explanation_status"] = "accepted"
        except Exception:
            result["explanation_status"] = "rejected"
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="UTF-8 JSON file containing journey and user")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8-sig") as stream:
            payload = json.load(stream, object_pairs_hook=_unique_object)
        result = recommend(payload)
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
