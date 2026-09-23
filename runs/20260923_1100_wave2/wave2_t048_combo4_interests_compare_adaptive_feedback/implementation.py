"""Deterministic synthetic marketplace pipeline; Python standard library only."""

import copy
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def number(value, path):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            path + " must be a finite nonnegative number")


def strings(value, path, nonempty=False):
    require(isinstance(value, list), path + " must be a list")
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), path + " has duplicates")
    require(not nonempty or bool(value), path + " must not be empty")


def fields(value, required, optional=()):
    require(isinstance(value, dict), "expected an object")
    require(set(required) <= set(value), "missing fields: " + str(set(required) - set(value)))
    require(set(value) <= set(required) | set(optional), "unknown fields")


def records(value, path):
    require(isinstance(value, list), path + " must be a list")
    seen = set()
    for row in value:
        require(isinstance(row, dict) and "id" in row, path + " needs object records with ids")
        text(row["id"], path + ".id")
        require(row["id"] not in seen, path + " has duplicate ids")
        seen.add(row["id"])
    return seen


def validate(kind, value, context=None):
    """One validation boundary shared by input and every stage handoff."""
    if kind == "input":
        return _validate_input(value)
    elif kind in ("interests", "compare", "adaptive", "feedback"):
        require(context is not None, "stage validation requires context")
        _validate_stage(kind, value, context)
        return value
    else:
        raise ValidationError("unknown schema kind")
    return value


def _validate_input(data):
    # Interests and priorities are maps with dynamic tag/attribute keys.
    fields(data, ("schema_version", "synthetic", "preferences", "products",
                  "onboarding_steps", "feedback"))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(data["synthetic"] is True, "fixtures must be labeled synthetic")
    pref = data["preferences"]
    fields(pref, ("interests", "excluded_ids", "excluded_tags", "priorities", "experience"),
           ("limit",))
    for name in ("interests", "priorities"):
        require(isinstance(pref[name], dict), name + " must be an object")
        for key, weight in pref[name].items():
            text(key, name)
            number(weight, name)
            require(weight <= 100, name + " weights must be <= 100")
    require(set(pref["priorities"]) <= {"price", "weight"}, "unknown comparison priority")
    strings(pref["excluded_ids"], "excluded_ids")
    strings(pref["excluded_tags"], "excluded_tags")
    require(pref["experience"] in ("beginner", "intermediate", "expert"), "invalid experience")
    require(type(pref.get("limit", 3)) is int and 1 <= pref.get("limit", 3) <= 100,
            "limit must be an integer from 1 to 100")
    product_ids = records(data["products"], "products")
    require(set(pref["excluded_ids"]) <= product_ids, "unknown excluded product")
    for product in data["products"]:
        fields(product, ("id", "name", "tags", "attributes"))
        text(product["name"], "product name")
        strings(product["tags"], "product tags")
        fields(product["attributes"], (), ("price", "weight"))
        for key, measurement in product["attributes"].items():
            fields(measurement, ("value", "unit"))
            number(measurement["value"], key)
            units = ("USD", "cents") if key == "price" else ("g", "kg")
            require(measurement["unit"] in units, "unsupported " + key + " unit")
            require(math.isfinite(float(measurement["value"]) *
                                  (1000 if measurement["unit"] == "kg" else 1)),
                    "normalized measurement overflows")
    step_ids = records(data["onboarding_steps"], "onboarding_steps")
    for step in data["onboarding_steps"]:
        fields(step, ("id", "title", "instructions", "tags", "experience_levels", "prerequisites"))
        text(step["title"], "step title")
        text(step["instructions"], "step instructions")
        strings(step["tags"], "step tags")
        strings(step["experience_levels"], "experience_levels", True)
        require(set(step["experience_levels"]) <= {"beginner", "intermediate", "expert"},
                "unknown step experience level")
        strings(step["prerequisites"], "prerequisites")
        require(set(step["prerequisites"]) <= step_ids, "unknown prerequisite")
    # Kahn's algorithm checks cycles without recursion-depth limitations.
    pending = {step["id"]: set(step["prerequisites"]) for step in data["onboarding_steps"]}
    done = set()
    while pending:
        ready = sorted(key for key, deps in pending.items() if deps <= done)
        require(bool(ready), "onboarding prerequisites contain a cycle")
        for key in ready:
            done.add(key)
            del pending[key]
    records(data["feedback"], "feedback")
    for item in data["feedback"]:
        fields(item, ("id", "product_id", "text"), ("step_id",))
        require(isinstance(item["product_id"], str) and item["product_id"] in product_ids,
                "unknown feedback product")
        require(item.get("step_id") is None or
                (isinstance(item["step_id"], str) and item["step_id"] in step_ids),
                "unknown feedback step")
        text(item["text"], "feedback text")
    return data


def _validate_stage(kind, value, context):
    data = context["input"]
    products = {p["id"]: p for p in data["products"]}
    require(isinstance(value, dict), "stage output must be an object")
    if kind == "interests":
        fields(value, ("recommendations", "excluded_ids"))
        ids = records(value["recommendations"], "recommendations")
        require(ids <= set(products), "recommendation references unknown product")
        pref = data["preferences"]
        forbidden = {p["id"] for p in products.values()
                     if p["id"] in pref["excluded_ids"] or
                     set(p["tags"]) & set(pref["excluded_tags"])}
        require(not ids & forbidden, "excluded recommendation")
        require(value["excluded_ids"] == sorted(forbidden), "exclusion audit mismatch")
        require(len(ids) <= pref.get("limit", 3), "recommendation limit exceeded")
        for row in value["recommendations"]:
            fields(row, ("id", "score", "matched_interests", "explanations"))
            number(row["score"], "interest score")
            strings(row["matched_interests"], "matched interests")
            require(set(row["matched_interests"]) <= set(products[row["id"]]["tags"]),
                    "ungrounded interest")
            matches = sorted(tag for tag in products[row["id"]]["tags"]
                             if pref["interests"].get(tag, 0) > 0)
            require(row["matched_interests"] == matches and
                    row["score"] == sum(pref["interests"][tag] for tag in matches),
                    "interest score must be grounded in preferences")
            strings(row["explanations"], "explanations", True)
        require(value["recommendations"] == sorted(value["recommendations"],
                key=lambda row: (-row["score"], row["id"])), "incorrect interest ranking")
    elif kind == "compare":
        fields(value, ("columns", "ranking", "selected_product_id"))
        source = context["interests"]["recommendations"]
        ids = records(value["ranking"], "comparison ranking")
        require(ids == {r["id"] for r in source}, "comparison must preserve candidates")
        require(value["columns"] == [
            {"attribute": "price", "unit": "USD", "preferred": "lower"},
            {"attribute": "weight", "unit": "g", "preferred": "lower"}],
            "invalid comparison columns")
        for row in value["ranking"]:
            fields(row, ("id", "name", "attributes", "interest_score", "attribute_score",
                         "score", "explanations"))
            fields(row["attributes"], ("price", "weight"))
            for measurement in row["attributes"].values():
                if measurement is not None:
                    number(measurement, "normalized attribute")
            for key in ("interest_score", "attribute_score", "score"):
                number(row[key], key)
            source_score = next(r["score"] for r in source if r["id"] == row["id"])
            require(row["interest_score"] == source_score, "interest score handoff mismatch")
            require(row["score"] == row["interest_score"] + row["attribute_score"],
                    "comparison total mismatch")
            require(row["name"] == products[row["id"]]["name"], "ungrounded product name")
            for key in ("price", "weight"):
                measurement = products[row["id"]]["attributes"].get(key)
                expected = (None if measurement is None else measurement["value"] *
                            {"USD": 1, "cents": 0.01, "g": 1, "kg": 1000}[measurement["unit"]])
                require(row["attributes"][key] == expected, "incorrect normalization")
            strings(row["explanations"], "comparison explanations", True)
        require(value["ranking"] == sorted(value["ranking"],
                key=lambda row: (-row["score"], row["id"])), "incorrect comparison ranking")
        require(value["selected_product_id"] ==
                (value["ranking"][0]["id"] if value["ranking"] else None),
                "selected product must lead comparison")
    elif kind == "adaptive":
        fields(value, ("selected_product_id", "experience", "steps"))
        require(value["selected_product_id"] == context["compare"]["selected_product_id"],
                "onboarding product handoff mismatch")
        require(value["experience"] == data["preferences"]["experience"], "experience mismatch")
        known = {s["id"]: s for s in data["onboarding_steps"]}
        ids = records(value["steps"], "adaptive steps")
        require(ids <= set(known), "unknown onboarding step")
        require(value["selected_product_id"] is not None or not ids,
                "cannot onboard without selected product")
        seen = set()
        for row in value["steps"]:
            fields(row, ("id", "title", "instructions", "prerequisites", "reason"))
            original = known[row["id"]]
            require(row["prerequisites"] == original["prerequisites"], "altered prerequisites")
            require(set(row["prerequisites"]) <= seen, "prerequisites must precede steps")
            require(row["title"] == original["title"] and
                    row["instructions"] == original["instructions"], "ungrounded instructions")
            text(row["reason"], "step reason")
            seen.add(row["id"])
    elif kind == "feedback":
        fields(value, ("selected_product_id", "onboarding_step_ids", "unique_count",
                       "duplicate_count", "ignored_ids", "themes"))
        adaptive = context["adaptive"]
        require(value["selected_product_id"] == adaptive["selected_product_id"],
                "feedback product handoff mismatch")
        require(value["onboarding_step_ids"] == [s["id"] for s in adaptive["steps"]],
                "feedback onboarding handoff mismatch")
        originals = {f["id"]: f for f in data["feedback"]}
        strings(value["ignored_ids"], "ignored feedback ids")
        require(set(value["ignored_ids"]) <= set(originals), "unknown ignored feedback")
        records(value["themes"], "themes")
        covered = set()
        groups = set()
        for theme in value["themes"]:
            fields(theme, ("id", "count", "support"))
            require(type(theme["count"]) is int and theme["count"] == len(theme["support"]),
                    "theme count mismatch")
            for excerpt in theme["support"]:
                fields(excerpt, ("source_ids", "text", "step_id"))
                strings(excerpt["source_ids"], "source ids", True)
                require(set(excerpt["source_ids"]) <= set(originals), "unknown evidence")
                for source_id in excerpt["source_ids"]:
                    source = originals[source_id]
                    require(source["product_id"] == adaptive["selected_product_id"] and
                            source.get("step_id") == excerpt["step_id"], "evidence scope mismatch")
                    require(excerpt["step_id"] is None or
                            excerpt["step_id"] in value["onboarding_step_ids"], "unplanned step evidence")
                    require(_normalized_text(source["text"]) == _normalized_text(excerpt["text"]),
                            "unsupported excerpt")
                require(excerpt["text"] == originals[excerpt["source_ids"][0]]["text"],
                        "excerpt must preserve verbatim source")
                covered.update(excerpt["source_ids"])
                groups.add(tuple(excerpt["source_ids"]))
        require(not covered & set(value["ignored_ids"]), "ignored feedback used as evidence")
        require(covered | set(value["ignored_ids"]) == set(originals), "feedback lost")
        require(type(value["unique_count"]) is int and value["unique_count"] == len(groups),
                "unique feedback count mismatch")
        require(type(value["duplicate_count"]) is int and
                value["duplicate_count"] == len(covered) - len(groups), "duplicate count mismatch")


def interests(data):
    validate("input", data)
    pref = data["preferences"]
    recommendations, excluded = [], []
    for product in data["products"]:
        if (product["id"] in pref["excluded_ids"] or
                set(product["tags"]) & set(pref["excluded_tags"])):
            excluded.append(product["id"])
            continue
        matches = sorted(tag for tag in product["tags"] if pref["interests"].get(tag, 0) > 0)
        score = sum(pref["interests"][tag] for tag in matches)
        explanations = [
            f"Product tag '{tag}' matches stated interest (weight {pref['interests'][tag]})."
            for tag in matches]
        recommendations.append({"id": product["id"], "score": score,
                                "matched_interests": matches,
                                "explanations": explanations or [
                                    "No positively weighted interest matches; neutral fallback."]})
    recommendations.sort(key=lambda row: (-row["score"], row["id"]))
    result = {"recommendations": recommendations[:pref.get("limit", 3)],
              "excluded_ids": sorted(excluded)}
    return validate("interests", result, {"input": data})


def compare(data, previous):
    validate("interests", previous, {"input": data})
    products = {p["id"]: p for p in data["products"]}
    rows = []
    for recommendation in previous["recommendations"]:
        product = products[recommendation["id"]]
        attrs = {}
        for key in ("price", "weight"):
            measure = product["attributes"].get(key)
            factor = {"USD": 1, "cents": 0.01, "g": 1, "kg": 1000}
            attrs[key] = None if measure is None else measure["value"] * factor[measure["unit"]]
        rows.append({"id": product["id"], "name": product["name"], "attributes": attrs,
                     "interest_score": recommendation["score"], "attribute_score": 0,
                     "score": recommendation["score"], "explanations": [
                         f"Interest score carried forward: {recommendation['score']}."]})
    for key, priority in sorted(data["preferences"]["priorities"].items()):
        observed = [row["attributes"][key] for row in rows if row["attributes"][key] is not None]
        for row in rows:
            value = row["attributes"][key]
            if value is None:
                row["explanations"].append(f"{key}: missing; contributes 0, never imputed.")
                continue
            low, high = min(observed), max(observed)
            utility = 1 if low == high else (high - value) / (high - low)
            contribution = priority * utility
            row["attribute_score"] += contribution
            row["explanations"].append(
                f"{key}: normalized value {value}; lower preferred; contribution {contribution}.")
    for row in rows:
        row["score"] += row["attribute_score"]
    rows.sort(key=lambda row: (-row["score"], row["id"]))
    result = {"columns": [{"attribute": "price", "unit": "USD", "preferred": "lower"},
                          {"attribute": "weight", "unit": "g", "preferred": "lower"}],
              "ranking": rows, "selected_product_id": rows[0]["id"] if rows else None}
    return validate("compare", result, {"input": data, "interests": previous})


def adaptive(data, previous, recommendation_output):
    context = {"input": data, "interests": recommendation_output, "compare": previous}
    validate("compare", previous, context)
    selected = previous["selected_product_id"]
    experience = data["preferences"]["experience"]
    result = {"selected_product_id": selected, "experience": experience, "steps": []}
    if selected is not None:
        product = next(p for p in data["products"] if p["id"] == selected)
        steps = {s["id"]: s for s in data["onboarding_steps"]}
        roots = {key for key, step in steps.items()
                 if experience in step["experience_levels"] and
                 (not step["tags"] or set(step["tags"]) & set(product["tags"]))}
        included = set(roots)
        queue = list(roots)
        while queue:
            for dep in steps[queue.pop()]["prerequisites"]:
                if dep not in included:
                    included.add(dep)
                    queue.append(dep)
        seen = set()
        while included - seen:
            ready = [key for key in included - seen if set(steps[key]["prerequisites"]) <= seen]
            require(bool(ready), "cannot order prerequisites")
            ready.sort(key=lambda key: (
                -sum(data["preferences"]["interests"].get(tag, 0) for tag in steps[key]["tags"]), key))
            key = ready[0]
            step = steps[key]
            reason = (f"Matches {experience} experience and selected product tags; ready steps "
                      "ordered by stated interests." if key in roots else
                      "Required prerequisite retained despite experience/tag filtering.")
            result["steps"].append({k: copy.deepcopy(step[k]) for k in
                                    ("id", "title", "instructions", "prerequisites")} | {"reason": reason})
            seen.add(key)
    return validate("adaptive", result, context)


def _normalized_text(value):
    return " ".join(value.casefold().split())


def analyze_feedback(data, previous, comparison_output):
    context = {"input": data, "compare": comparison_output, "adaptive": previous}
    validate("adaptive", previous, context)
    selected = previous["selected_product_id"]
    step_ids = [s["id"] for s in previous["steps"]]
    groups, ignored = {}, []
    for item in sorted(data["feedback"], key=lambda row: row["id"]):
        if selected is None or item["product_id"] != selected or (
                item.get("step_id") is not None and item["step_id"] not in step_ids):
            ignored.append(item["id"])
            continue
        key = (item.get("step_id"), _normalized_text(item["text"]))
        if key not in groups:
            groups[key] = {"source_ids": [], "text": item["text"], "step_id": item.get("step_id")}
        groups[key]["source_ids"].append(item["id"])
    themes = {}
    vocabulary = {"usability": {"easy", "confusing", "setup", "difficult"},
                  "quality": {"broken", "durable", "quality", "reliable"},
                  "value": {"price", "expensive", "cheap", "value"}}
    for group in groups.values():
        words = set(re.findall(r"\w+", group["text"].casefold()))
        labels = [label for label, keywords in vocabulary.items() if words & keywords] or ["other"]
        for label in labels:
            themes.setdefault(label, []).append(copy.deepcopy(group))
    result = {"selected_product_id": selected, "onboarding_step_ids": step_ids,
              "unique_count": len(groups),
              "duplicate_count": sum(len(g["source_ids"]) - 1 for g in groups.values()),
              "ignored_ids": ignored,
              "themes": [{"id": label, "count": len(support), "support": support}
                         for label, support in sorted(themes.items())]}
    return validate("feedback", result, context)


def run_pipeline(data):
    data = copy.deepcopy(validate("input", data))
    discovery = interests(data)
    comparison = compare(data, discovery)
    onboarding = adaptive(data, comparison, discovery)
    feedback = analyze_feedback(data, onboarding, comparison)
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "interests": discovery, "compare": comparison, "adaptive": onboarding, "feedback": feedback}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        def reject_constant(value):
            raise ValidationError("nonfinite JSON number: " + value)
        with open(argv[0], encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant)
        output = run_pipeline(data)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
