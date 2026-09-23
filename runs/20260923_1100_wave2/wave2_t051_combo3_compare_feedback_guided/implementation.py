"""Synthetic, deterministic comparison -> feedback -> onboarding reference CLI."""

import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


METRICS = {
    "price": ("USD", {"USD": 1, "cents": 0.01}),
    "weight": ("g", {"g": 1, "kg": 1000}),
    "battery": ("hours", {"hours": 1, "minutes": 1 / 60}),
}
THEMES = {
    "battery": ("battery", "charge", "charging"),
    "setup": ("setup", "install", "installation", "pair", "pairing"),
    "value": ("price", "expensive", "cheap", "value"),
    "portability": ("weight", "heavy", "light", "portable"),
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def shape(value, fields, path):
    require(isinstance(value, dict), f"{path}: expected object")
    require(set(value) == set(fields), f"{path}: expected keys {', '.join(fields)}")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")


def number(value, path, positive=False):
    require(type(value) in (int, float), f"{path}: expected number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and (value > 0 if positive else value >= 0), f"{path}: invalid number")


def string_list(value, path):
    require(isinstance(value, list), f"{path}: expected list")
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), f"{path}: duplicate values")


def records(value, fields, path, nonempty=False):
    require(isinstance(value, list) and (bool(value) or not nonempty), f"{path}: expected list")
    ids = set()
    for row in value:
        shape(row, fields, path)
        text(row["id"], f"{path}.id")
        require(row["id"] not in ids, f"{path}: duplicate id")
        ids.add(row["id"])
    return ids


def validate(value, kind="input"):
    """One validation boundary shared by input and every stage handoff."""
    if kind != "input":
        shape(value, ("schema_version", "stage", "data"), "handoff")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "handoff: unsupported schema version")
        require(value["stage"] == kind, "handoff: wrong stage")
        data = value["data"]
        if kind == "compare":
            shape(data, ("selected_product_id", "columns", "rows", "ranking"), "compare")
            require(data["columns"] == {k: v[0] for k, v in METRICS.items()}, "compare: columns")
            ids = records(data["rows"], ("id", "name", "attributes"), "compare.rows", True)
            for row in data["rows"]:
                text(row["name"], "compare.name")
                shape(row["attributes"], METRICS, "compare.attributes")
                for metric, amount in row["attributes"].items():
                    number(amount, metric)
            ranks = records(data["ranking"], ("id", "score"), "compare.ranking", True)
            require(ranks == ids, "compare: ranking coverage")
            for row in data["ranking"]:
                number(row["score"], "compare.score")
                require(row["score"] <= 1, "compare: score outside range")
            require(data["ranking"] == sorted(data["ranking"], key=lambda r: (-r["score"], r["id"])),
                    "compare: unordered ranking")
            require(data["selected_product_id"] == data["ranking"][0]["id"], "compare: winner mismatch")
        elif kind == "feedback":
            shape(data, ("comparison", "selected_product_id", "source_count", "unique_count",
                         "duplicate_count", "excerpts", "themes"), "feedback")
            validate(data["comparison"], "compare")
            require(data["selected_product_id"] == data["comparison"]["data"]["selected_product_id"],
                    "feedback: selected product changed")
            for key in ("source_count", "unique_count", "duplicate_count"):
                require(type(data[key]) is int and data[key] >= 0, "feedback: invalid count")
            ids = records(data["excerpts"], ("id", "product_id", "text", "source_ids"), "excerpts")
            sources = []
            for row in data["excerpts"]:
                text(row["text"], "excerpt.text")
                string_list(row["source_ids"], "excerpt.source_ids")
                require(row["source_ids"] and row["id"] == row["source_ids"][0], "excerpt: source id")
                require(row["product_id"] == data["selected_product_id"], "excerpt: wrong product")
                sources.extend(row["source_ids"])
            require(len(sources) == len(set(sources)) == data["source_count"], "feedback: source coverage")
            require(len(ids) == data["unique_count"] and
                    data["duplicate_count"] == data["source_count"] - len(ids), "feedback: counts")
            records(data["themes"], ("id", "count", "excerpt_ids"), "themes")
            covered = set()
            for theme in data["themes"]:
                require(theme["id"] in {*THEMES, "general"}, "feedback: unknown theme")
                string_list(theme["excerpt_ids"], "theme.excerpt_ids")
                require(set(theme["excerpt_ids"]) <= ids and theme["excerpt_ids"], "theme: missing excerpt")
                require(type(theme["count"]) is int and theme["count"] == len(theme["excerpt_ids"]),
                        "theme: count mismatch")
                covered.update(theme["excerpt_ids"])
            require(covered == ids, "feedback: unthemed excerpts")
        elif kind == "guided":
            shape(data, ("feedback", "selected_product_id", "steps", "progress"), "guided")
            validate(data["feedback"], "feedback")
            require(data["selected_product_id"] == data["feedback"]["data"]["selected_product_id"],
                    "guided: product mismatch")
            ids = records(data["steps"], ("id", "title", "prerequisites", "state",
                                          "matched_themes", "supporting_excerpt_ids"), "guided.steps")
            completed = {s["id"] for s in data["steps"] if s["state"] == "completed"}
            evidence = {e["id"] for e in data["feedback"]["data"]["excerpts"]}
            themes = {t["id"] for t in data["feedback"]["data"]["themes"]}
            for step in data["steps"]:
                text(step["title"], "guided.title")
                for key in ("prerequisites", "matched_themes", "supporting_excerpt_ids"):
                    string_list(step[key], f"guided.{key}")
                require(set(step["prerequisites"]) <= ids, "guided: missing prerequisite")
                ready = set(step["prerequisites"]) <= completed
                expected = "completed" if step["id"] in completed else "available" if ready else "blocked"
                require(step["state"] == expected and (step["state"] != "completed" or ready),
                        "guided: invalid progress state")
                require(set(step["matched_themes"]) <= themes, "guided: unknown theme")
                require(set(step["supporting_excerpt_ids"]) <= evidence, "guided: unknown excerpt")
            expected_progress = {
                "completed": len(completed), "total": len(ids),
                "percent": round(100 * len(completed) / len(ids), 2) if ids else 100.0,
            }
            require(data["progress"] == expected_progress, "guided: invalid progress")
        else:
            raise ValidationError("Unknown schema kind")
        return value

    shape(value, ("schema_version", "synthetic", "products", "preferences", "feedback", "onboarding"), "input")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1, "Unsupported schema version")
    require(value["synthetic"] is True, "Fixtures must be labeled synthetic")
    product_ids = records(value["products"], ("id", "name", "attributes"), "products", True)
    for product in value["products"]:
        text(product["name"], "product.name")
        shape(product["attributes"], METRICS, "product.attributes")
        for metric, raw in product["attributes"].items():
            shape(raw, ("value", "unit"), metric)
            number(raw["value"], metric)
            require(isinstance(raw["unit"], str) and raw["unit"] in METRICS[metric][1],
                    f"{metric}: unsupported unit")
            number(raw["value"] * METRICS[metric][1][raw["unit"]], f"{metric}: normalized value")
    require(isinstance(value["preferences"], dict) and bool(value["preferences"]), "preferences: empty")
    require(set(value["preferences"]) <= set(METRICS), "preferences: unknown metric")
    for metric, pref in value["preferences"].items():
        shape(pref, ("weight", "direction"), "preference")
        number(pref["weight"], "preference.weight", True)
        require(pref["direction"] in ("min", "max"), "preference: invalid direction")
    number(sum(p["weight"] for p in value["preferences"].values()), "total preference weight", True)
    records(value["feedback"], ("id", "product_id", "text"), "feedback")
    for item in value["feedback"]:
        text(item["product_id"], "feedback.product_id")
        require(item["product_id"] in product_ids, "feedback: unknown product")
        text(item["text"], "feedback.text")
        require(bool(re.findall(r"\w+", item["text"])), "feedback: text has no words")
    shape(value["onboarding"], ("steps", "completed_steps"), "onboarding")
    steps = value["onboarding"]["steps"]
    ids = records(steps, ("id", "title", "prerequisites", "when_themes"), "onboarding.steps")
    for step in steps:
        text(step["title"], "step.title")
        string_list(step["prerequisites"], "step.prerequisites")
        string_list(step["when_themes"], "step.when_themes")
        require(set(step["prerequisites"]) <= ids, "step: unknown prerequisite")
        require(set(step["when_themes"]) <= {*THEMES, "general"}, "step: unknown theme")
    remaining = {s["id"]: set(s["prerequisites"]) for s in steps}
    visited = set()
    while remaining:
        ready = {key for key, deps in remaining.items() if deps <= visited}
        require(ready, "onboarding: cyclic prerequisites")
        visited.update(ready)
        remaining = {key: deps for key, deps in remaining.items() if key not in ready}
    string_list(value["onboarding"]["completed_steps"], "completed_steps")
    require(set(value["onboarding"]["completed_steps"]) <= ids, "Unknown completed step")
    return value


def envelope(stage, data):
    return validate({"schema_version": 1, "stage": stage, "data": data}, stage)


def compare(request):
    validate(request)
    rows = [
        {"id": p["id"], "name": p["name"],
         "attributes": {m: raw["value"] * METRICS[m][1][raw["unit"]]
                        for m, raw in p["attributes"].items()}}
        for p in sorted(request["products"], key=lambda p: p["id"])
    ]
    preferences = request["preferences"]
    total = sum(p["weight"] for p in preferences.values())
    scores = {row["id"]: 0.0 for row in rows}
    for metric, preference in sorted(preferences.items()):
        low = min(row["attributes"][metric] for row in rows)
        high = max(row["attributes"][metric] for row in rows)
        for row in rows:
            utility = 1.0 if high == low else (row["attributes"][metric] - low) / (high - low)
            if high != low and preference["direction"] == "min":
                utility = 1 - utility
            scores[row["id"]] += utility * (preference["weight"] / total)
    ranking = sorted([{"id": key, "score": round(score, 10)} for key, score in scores.items()],
                     key=lambda row: (-row["score"], row["id"]))
    return envelope("compare", {"selected_product_id": ranking[0]["id"],
                                "columns": {k: v[0] for k, v in METRICS.items()},
                                "rows": rows, "ranking": ranking})


def analyze_feedback(request, comparison):
    validate(request)
    validate(comparison, "compare")
    require(comparison == compare(request), "feedback: comparison does not belong to input")
    winner = comparison["data"]["selected_product_id"]
    source = sorted([f for f in request["feedback"] if f["product_id"] == winner], key=lambda f: f["id"])
    unique = {}
    for item in source:
        key = " ".join(re.findall(r"\w+", item["text"].casefold()))
        if key not in unique:
            unique[key] = {"id": item["id"], "product_id": winner,
                           "text": item["text"], "source_ids": []}
        unique[key]["source_ids"].append(item["id"])
    groups = {}
    for key, excerpt in unique.items():
        words = set(key.split())
        matched = [theme for theme, keywords in THEMES.items() if words.intersection(keywords)] or ["general"]
        for theme in matched:
            groups.setdefault(theme, []).append(excerpt["id"])
    return envelope("feedback", {
        "comparison": comparison, "selected_product_id": winner, "source_count": len(source),
        "unique_count": len(unique), "duplicate_count": len(source) - len(unique),
        "excerpts": list(unique.values()),
        "themes": [{"id": theme, "count": len(ids), "excerpt_ids": ids} for theme, ids in sorted(groups.items())],
    })


def guide(request, feedback):
    validate(request)
    validate(feedback, "feedback")
    require(feedback == analyze_feedback(request, feedback["data"]["comparison"]),
            "guided: feedback does not belong to input")
    themes = {t["id"]: t["excerpt_ids"] for t in feedback["data"]["themes"]}
    definitions = {s["id"]: s for s in request["onboarding"]["steps"]}
    active = {key for key, step in definitions.items()
              if not step["when_themes"] or set(step["when_themes"]).intersection(themes)}
    while True:
        expanded = active | {dep for key in active for dep in definitions[key]["prerequisites"]}
        if expanded == active:
            break
        active = expanded
    completed = set(request["onboarding"]["completed_steps"])
    require(completed <= active, "Cannot complete inactive steps")
    require(all(set(definitions[key]["prerequisites"]) <= completed for key in completed),
            "Completed steps require completed prerequisites")
    steps = []
    emitted = set()
    while emitted != active:
        ready = sorted(key for key in active - emitted if set(definitions[key]["prerequisites"]) <= emitted)
        for key in ready:
            step = definitions[key]
            matched = sorted(set(step["when_themes"]).intersection(themes))
            steps.append({
                "id": key, "title": step["title"], "prerequisites": sorted(step["prerequisites"]),
                "state": "completed" if key in completed else
                         "available" if set(step["prerequisites"]) <= completed else "blocked",
                "matched_themes": matched,
                "supporting_excerpt_ids": sorted({ref for theme in matched for ref in themes[theme]}),
            })
        emitted.update(ready)
    return envelope("guided", {
        "feedback": feedback, "selected_product_id": feedback["data"]["selected_product_id"],
        "steps": steps, "progress": {"completed": len(completed), "total": len(active),
                                   "percent": round(100 * len(completed) / len(active), 2) if active else 100.0},
    })


def run_pipeline(request):
    validate(request)
    comparison = compare(request)
    feedback = analyze_feedback(request, comparison)
    return {"schema_version": 1, "synthetic": True, "status": "ok", "result": guide(request, feedback)}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        request = json.loads(Path(args[0]).read_text(encoding="utf-8"),
                             object_pairs_hook=reject_duplicate_keys,
                             parse_constant=lambda token: (_ for _ in ()).throw(
                                 ValidationError(f"Nonfinite JSON constant: {token}")))
        result = run_pipeline(request)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
