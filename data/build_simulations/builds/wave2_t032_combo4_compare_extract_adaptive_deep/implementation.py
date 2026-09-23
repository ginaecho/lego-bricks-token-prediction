"""Synthetic, deterministic compare -> extract -> adaptive -> deep pipeline.

Run: python -B implementation.py example_input.json
No providers, dependencies, or network access are used. Each stage accepts only
the preceding validated envelope and carries forward the same request/results.
Extraction recognizes complete ``label: value`` lines, not arbitrary prose.
"""

import copy
import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    """An input or stage handoff violates the shared schema."""


def obj(**fields):
    return {"type": "object", "fields": fields}


def array(items, minimum=0):
    return {"type": "array", "items": items, "minimum": minimum}


def choice(*values):
    return {"type": "string", "enum": values}


TEXT = {"type": "string", "nonempty": True}
STRING = {"type": "string"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
INTEGER = {"type": "integer", "minimum": 0}
SCALAR = {"type": "scalar"}
NULLABLE_NUMBER = {"type": "nullable", "inner": NUMBER}
NULLABLE_SCALAR = {"type": "nullable", "inner": SCALAR}
ATTRIBUTES = ("price_usd", "weight_g", "battery_hours")
ALIASES = {
    "price": "price_usd", "cost": "price_usd", "price_usd": "price_usd",
    "weight": "weight_g", "mass": "weight_g", "weight_g": "weight_g",
    "battery": "battery_hours", "battery_life": "battery_hours",
    "battery_hours": "battery_hours",
}
UNITS = {
    "price_usd": {"usd": 1},
    "weight_g": {"g": 1, "kg": 1000, "lb": 453.59237},
    "battery_hours": {"h": 1, "hr": 1, "hours": 1, "min": 1 / 60},
}
LEVELS = ("beginner", "intermediate", "expert")
MODES = ("visual", "hands_on", "concise")
MEASUREMENT = obj(value=NUMBER, unit=TEXT)
FIELD = obj(name=TEXT, type=choice("string", "number", "boolean"),
            labels=array(TEXT, 1), required=BOOLEAN)
TASK = obj(id=TEXT, title=TEXT, min_experience=choice(*LEVELS),
           modes=array(choice(*MODES), 1), prerequisites=array(TEXT),
           requires_fields=array(TEXT), explanation=TEXT)
DOCUMENT = obj(id=TEXT, product_ids=array(TEXT), text=STRING)
QUESTION = obj(id=TEXT, field=TEXT, text=TEXT)
REQUEST = obj(
    schema_version={"type": "integer", "enum": (1,)},
    synthetic=BOOLEAN,
    products=array(obj(id=TEXT, name=TEXT,
                       attributes={"type": "mapping", "values": MEASUREMENT}), 1),
    comparison=obj(criteria=array(obj(attribute=choice(*ATTRIBUTES),
                                      direction=choice("min", "max"),
                                      weight=NUMBER), 1)),
    documents=array(DOCUMENT),
    extraction_schema=obj(fields=array(FIELD, 1)),
    onboarding=obj(experience=choice(*LEVELS), preferences=array(choice(*MODES)),
                   completed=array(TEXT), tasks=array(TASK)),
    research=obj(questions=array(QUESTION, 1)),
)
NORMALIZED = obj(**{name: NULLABLE_NUMBER for name in ATTRIBUTES})
COMPARISON = obj(
    columns=array(choice(*ATTRIBUTES), 1),
    rows=array(obj(product_id=TEXT, name=TEXT, attributes=NORMALIZED,
                   missing_attributes=array(choice(*ATTRIBUTES))), 1),
    ranking=array(obj(product_id=TEXT, score=NUMBER,
                      contributions=obj(**{name: NULLABLE_NUMBER
                                           for name in ATTRIBUTES})), 1),
    selected_product_id=TEXT, selected_document_ids=array(TEXT),
    rationale=TEXT,
)
SPAN = obj(start=INTEGER, end=INTEGER)
SOURCE = obj(document_id=TEXT, span=SPAN, raw=TEXT, value=SCALAR)
EXTRACTED_FIELD = obj(name=TEXT, type=choice("string", "number", "boolean"),
                      required=BOOLEAN, value=NULLABLE_SCALAR,
                      sources=array(SOURCE), conflict=BOOLEAN)
EXTRACTION = obj(
    product_id=TEXT, document_ids=array(TEXT), fields=array(EXTRACTED_FIELD, 1),
    missing_fields=array(TEXT), missing_required_fields=array(TEXT),
    conflicting_fields=array(TEXT),
    issues=array(obj(field=TEXT, document_id=TEXT, span=SPAN,
                     raw=STRING, reason=TEXT)),
)
ONBOARDING = obj(
    product_id=TEXT, experience=choice(*LEVELS),
    preferences=array(choice(*MODES)),
    tasks=array(obj(id=TEXT, title=TEXT,
                    state=choice("completed", "ready", "waiting", "blocked"),
                    mode=choice(*MODES), explanation=TEXT, reasons=array(TEXT))),
    ready_task_ids=array(TEXT), completed_task_ids=array(TEXT),
    unresolved_fields=array(TEXT), research_focus_fields=array(TEXT),
)
ANSWER = obj(
    question_id=TEXT, field=TEXT, question=TEXT,
    status=choice("supported", "disputed", "unresolved"),
    groups=array(obj(value=SCALAR, document_ids=array(TEXT, 1),
                     citations=array(SOURCE, 1))),
    distinct_document_count=INTEGER, summary=TEXT,
)
RESEARCH = obj(
    product_id=TEXT, focus_fields=array(TEXT), answers=array(ANSWER, 1),
    disagreements=array(TEXT), unresolved_questions=array(TEXT),
    onboarding_blockers=array(TEXT), synthesis=TEXT,
)
STAGES = ("compare", "extract", "adaptive", "deep")
RESULT_NAMES = ("comparison", "extraction", "onboarding", "research")
RESULT_SCHEMAS = (COMPARISON, EXTRACTION, ONBOARDING, RESEARCH)


def validate(value, schema, path="$"):
    """The shared structural validator used for inputs and all stage outputs."""
    kind = schema["type"]
    if kind == "nullable":
        if value is not None:
            validate(value, schema["inner"], path)
        return
    if kind == "object":
        if not isinstance(value, dict):
            raise ValidationError(f"{path}: expected object")
        fields = schema["fields"]
        missing, extra = fields.keys() - value.keys(), value.keys() - fields.keys()
        if missing or extra:
            raise ValidationError(f"{path}: missing={sorted(missing)}, extra={sorted(extra)}")
        for name, subschema in fields.items():
            validate(value[name], subschema, f"{path}.{name}")
        return
    if kind == "mapping":
        if not isinstance(value, dict) or not value:
            raise ValidationError(f"{path}: expected nonempty mapping")
        for name, item in value.items():
            validate(name, TEXT, path)
            validate(item, schema["values"], f"{path}.{name}")
        return
    if kind == "array":
        if not isinstance(value, list) or len(value) < schema["minimum"]:
            raise ValidationError(f"{path}: expected array of at least {schema['minimum']} items")
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")
        return
    if kind == "string":
        valid = isinstance(value, str) and (not schema.get("nonempty") or bool(value.strip()))
    elif kind in ("number", "integer"):
        valid = type(value) in (int, float) and finite(value)
        if kind == "integer":
            valid = type(value) is int and valid
        valid = valid and value >= schema.get("minimum", -math.inf)
    elif kind == "boolean":
        valid = type(value) is bool
    elif kind == "scalar":
        valid = isinstance(value, str) or type(value) is bool or (
            type(value) in (int, float) and finite(value))
    else:
        raise ValidationError(f"{path}: unknown schema type {kind}")
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise ValidationError(f"{path}: invalid {kind}")


def finite(value):
    try:
        return math.isfinite(value)
    except (TypeError, OverflowError):
        return False


def ensure(condition, message):
    if not condition:
        raise ValidationError(message)


def unique(items, description):
    ensure(len(set(items)) == len(items), f"{description}: duplicates are not allowed")


def attribute_name(raw):
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    ensure(key in ALIASES, f"Unknown product attribute: {raw}")
    return ALIASES[key]


def normalize(product):
    attributes = dict.fromkeys(ATTRIBUTES)
    for raw, measurement in product["attributes"].items():
        name = attribute_name(raw)
        ensure(attributes[name] is None, f"Duplicate normalized attribute: {name}")
        unit = measurement["unit"].strip().lower()
        ensure(unit in UNITS[name], f"Unsupported unit for {name}: {unit}")
        ensure(measurement["value"] >= 0, f"{name}: negative measurement")
        value = measurement["value"] * UNITS[name][unit]
        ensure(finite(value), f"{name}: normalized measurement is not finite")
        attributes[name] = value
    return attributes


def task_order(tasks, preferences):
    """Stable topological order, with preference matching among available tasks."""
    remaining = {task["id"]: task for task in tasks}
    ordered, visited = [], set()
    while remaining:
        available = [task for task in remaining.values()
                     if set(task["prerequisites"]) <= visited]
        ensure(bool(available), "Onboarding prerequisites contain a cycle")
        def priority(task):
            matches = [preferences.index(mode) for mode in task["modes"]
                       if mode in preferences]
            return (min(matches, default=len(preferences)), task["id"])
        task = min(available, key=priority)
        ordered.append(task)
        visited.add(task["id"])
        del remaining[task["id"]]
    return ordered


def validate_request(request):
    validate(request, REQUEST)
    ensure(request["synthetic"] is True, "This reference fixture must be labeled synthetic")
    product_ids = [product["id"] for product in request["products"]]
    unique(product_ids, "Product ids")
    for product in request["products"]:
        normalize(product)
    criteria = request["comparison"]["criteria"]
    unique([criterion["attribute"] for criterion in criteria], "Comparison criteria")
    for criterion in criteria:
        ensure(criterion["weight"] > 0, "Criterion weights must be positive")
    document_ids = [document["id"] for document in request["documents"]]
    unique(document_ids, "Document ids")
    for document in request["documents"]:
        unique(document["product_ids"], "Document product references")
        ensure(set(document["product_ids"]) <= set(product_ids), "Unknown document product")
    fields = request["extraction_schema"]["fields"]
    field_names = [field["name"] for field in fields]
    unique(field_names, "Field names")
    labels = []
    for field in fields:
        for label in field["labels"]:
            ensure("\n" not in label and "\r" not in label and ":" not in label,
                   "Extraction labels must be single-line and contain no colon")
            labels.append(label.strip().casefold())
    unique(labels, "Extraction labels")
    onboarding = request["onboarding"]
    tasks = onboarding["tasks"]
    task_ids = [task["id"] for task in tasks]
    unique(task_ids, "Task ids")
    unique(onboarding["preferences"], "Onboarding preferences")
    unique(onboarding["completed"], "Completed tasks")
    ensure(set(onboarding["completed"]) <= set(task_ids), "Unknown completed task")
    for task in tasks:
        for key in ("prerequisites", "requires_fields", "modes"):
            unique(task[key], f"Task {task['id']} {key}")
        ensure(set(task["prerequisites"]) <= set(task_ids), "Unknown prerequisite task")
        ensure(set(task["requires_fields"]) <= set(field_names), "Unknown required field")
        if task["id"] in onboarding["completed"]:
            ensure(set(task["prerequisites"]) <= set(onboarding["completed"]),
                   "Completed tasks must have completed prerequisites")
    task_order(tasks, onboarding["preferences"])
    questions = request["research"]["questions"]
    unique([question["id"] for question in questions], "Research question ids")
    for question in questions:
        ensure(question["field"] in field_names, "Research question references unknown field")
    return request


def selected_documents(request, product_id):
    return [document for document in request["documents"]
            if not document["product_ids"] or product_id in document["product_ids"]]


def scalar_key(value):
    # Keep booleans distinct from numbers, but treat 2 and 2.0 as equal.
    if type(value) is bool:
        return ("boolean", value)
    if type(value) in (int, float):
        return ("number", value)
    return ("string", value)


def typed_value(raw, kind):
    if kind == "string":
        ensure(bool(raw), "empty value")
        return raw
    if kind == "boolean":
        ensure(raw.casefold() in ("true", "false", "yes", "no"), "expected boolean")
        return raw.casefold() in ("true", "yes")
    ensure(bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw)),
           "expected number")
    value = float(raw)
    ensure(finite(value), "number must be finite")
    return value


def validate_envelope(envelope, expected_stage):
    ensure(expected_stage in STAGES, "Unknown pipeline stage")
    index = STAGES.index(expected_stage)
    schema = obj(
        schema_version={"type": "integer", "enum": (1,)},
        status=choice("ok"), stage=choice(expected_stage), synthetic=BOOLEAN,
        request=REQUEST,
        results=obj(**dict(zip(RESULT_NAMES[:index + 1], RESULT_SCHEMAS[:index + 1]))),
    )
    validate(envelope, schema)
    request = validate_request(envelope["request"])
    ensure(envelope["synthetic"] is True, "Output must be labeled synthetic")
    results = envelope["results"]
    comparison = results["comparison"]
    product_ids = [product["id"] for product in request["products"]]
    ensure([row["product_id"] for row in comparison["rows"]] == product_ids,
           "Comparison rows must preserve product ids and input order")
    ensure(comparison["columns"] == list(ATTRIBUTES), "Incorrect comparison columns")
    for row, product in zip(comparison["rows"], request["products"]):
        attributes = normalize(product)
        ensure(row["name"] == product["name"] and row["attributes"] == attributes,
               "Comparison row differs from normalized input")
        ensure(row["missing_attributes"] == [name for name in ATTRIBUTES
                                            if attributes[name] is None],
               "Incorrect missing attributes")
    ranking = comparison["ranking"]
    ensure(len(ranking) == len(product_ids)
           and {row["product_id"] for row in ranking} == set(product_ids),
           "Ranking must contain every product exactly once")
    ensure(ranking == sorted(ranking, key=lambda row: (-row["score"], row["product_id"])),
           "Ranking must be sorted by score then id")
    for row in ranking:
        ensure(0 <= row["score"] <= 1, "Comparison score outside [0, 1]")
    selected = comparison["selected_product_id"]
    ensure(selected == ranking[0]["product_id"], "Selection differs from first-ranked product")
    documents = selected_documents(request, selected)
    document_ids = [document["id"] for document in documents]
    ensure(comparison["selected_document_ids"] == document_ids,
           "Selected documents do not match selected product")
    if index >= 1:
        extraction = results["extraction"]
        ensure(extraction["product_id"] == selected
               and extraction["document_ids"] == document_ids, "Extraction handoff mismatch")
        declared = request["extraction_schema"]["fields"]
        extracted = extraction["fields"]
        ensure([field["name"] for field in extracted] ==
               [field["name"] for field in declared], "Extracted schema fields mismatch")
        by_document = {document["id"]: document for document in documents}
        def validate_span(source):
            ensure(source["document_id"] in by_document, "Source outside selected documents")
            text = by_document[source["document_id"]]["text"]
            start, end = source["span"]["start"], source["span"]["end"]
            ensure(start <= end <= len(text) and text[start:end] == source["raw"],
                   "Source span does not match original text")
        for field, spec in zip(extracted, declared):
            ensure(field["type"] == spec["type"] and field["required"] == spec["required"],
                   "Extracted field definition mismatch")
            for source in field["sources"]:
                validate_span(source)
                ensure(scalar_key(typed_value(source["raw"], spec["type"])) ==
                       scalar_key(source["value"]), "Source value does not match its text")
            expected = field["sources"][0]["value"] if field["sources"] else None
            ensure(scalar_key(field["value"]) == scalar_key(expected),
                   "Field value does not match its first source")
            ensure(field["conflict"] == (len({scalar_key(source["value"])
                                             for source in field["sources"]}) > 1),
                   "Incorrect conflict reporting")
        missing = [field["name"] for field in extracted if not field["sources"]]
        required = [field["name"] for field in extracted
                    if not field["sources"] and field["required"]]
        conflicts = [field["name"] for field in extracted if field["conflict"]]
        ensure(extraction["missing_fields"] == missing
               and extraction["missing_required_fields"] == required
               and extraction["conflicting_fields"] == conflicts,
               "Incorrect extraction missing/conflict reporting")
        for issue in extraction["issues"]:
            ensure(issue["field"] in [field["name"] for field in declared],
                   "Unknown extraction issue field")
            validate_span(issue)
    if index >= 2:
        onboarding = results["onboarding"]
        ensure(onboarding["product_id"] == selected, "Onboarding product handoff mismatch")
        ensure(onboarding["experience"] == request["onboarding"]["experience"]
               and onboarding["preferences"] == request["onboarding"]["preferences"],
               "Onboarding profile handoff mismatch")
        ensure({task["id"] for task in onboarding["tasks"]} ==
               {task["id"] for task in request["onboarding"]["tasks"]}
               and len(onboarding["tasks"]) == len(request["onboarding"]["tasks"]),
               "Onboarding task handoff mismatch")
        for key, state in (("ready_task_ids", "ready"), ("completed_task_ids", "completed")):
            ensure(onboarding[key] == [task["id"] for task in onboarding["tasks"]
                                       if task["state"] == state], "Incorrect task state index")
        expected_unresolved = sorted(set(results["extraction"]["missing_fields"] +
                                         results["extraction"]["conflicting_fields"]))
        ensure(onboarding["unresolved_fields"] == expected_unresolved,
               "Unresolved fields not propagated")
        ensure(set(onboarding["research_focus_fields"]) <=
               {field["name"] for field in results["extraction"]["fields"]},
               "Unknown research focus field")
    if index >= 3:
        research = results["research"]
        ensure(research["product_id"] == selected
               and research["focus_fields"] == results["onboarding"]["research_focus_fields"],
               "Research handoff mismatch")
        ensure({answer["question_id"] for answer in research["answers"]} ==
               {question["id"] for question in request["research"]["questions"]}
               and len(research["answers"]) == len(request["research"]["questions"]),
               "Research questions mismatch")
        for answer in research["answers"]:
            for group in answer["groups"]:
                for citation in group["citations"]:
                    validate_span(citation)
        ensure(research["disagreements"] == [answer["question_id"]
                                             for answer in research["answers"]
                                             if answer["status"] == "disputed"]
               and research["unresolved_questions"] == [answer["question_id"]
                                                        for answer in research["answers"]
                                                        if answer["status"] != "supported"],
               "Incorrect research status indexes")
    return envelope


def finish(envelope, stage, result):
    envelope["stage"] = stage
    envelope["results"][RESULT_NAMES[STAGES.index(stage)]] = result
    return validate_envelope(envelope, stage)


def compare(request):
    request = copy.deepcopy(validate_request(request))
    criteria = request["comparison"]["criteria"]
    rows = []
    for product in request["products"]:
        values = normalize(product)
        rows.append({"product_id": product["id"], "name": product["name"],
                     "attributes": values, "missing_attributes":
                     [name for name in ATTRIBUTES if values[name] is None]})
    # Scale first to avoid overflow when users provide large finite weights.
    largest = max(criterion["weight"] for criterion in criteria)
    weights = [criterion["weight"] / largest for criterion in criteria]
    total = sum(weights)
    ranking = []
    for row in rows:
        contributions = dict.fromkeys(ATTRIBUTES)
        for criterion, weight in zip(criteria, weights):
            name = criterion["attribute"]
            available = [item["attributes"][name] for item in rows
                         if item["attributes"][name] is not None]
            value = row["attributes"][name]
            if value is None:
                utility = 0
            elif max(available) == min(available):
                utility = 1
            else:
                utility = (value - min(available)) / (max(available) - min(available))
                if criterion["direction"] == "min":
                    utility = 1 - utility
            contributions[name] = utility * weight / total
        ranking.append({"product_id": row["product_id"],
                        "score": min(1.0, max(0.0, sum(value for value in contributions.values()
                                                     if value is not None))),
                        "contributions": contributions})
    ranking.sort(key=lambda item: (-item["score"], item["product_id"]))
    selected = ranking[0]["product_id"]
    envelope = {"schema_version": 1, "status": "ok", "stage": "compare",
                "synthetic": True, "request": request, "results": {}}
    return finish(envelope, "compare", {
        "columns": list(ATTRIBUTES), "rows": rows, "ranking": ranking,
        "selected_product_id": selected,
        "selected_document_ids": [document["id"]
                                  for document in selected_documents(request, selected)],
        "rationale": "Weighted min-max utility; missing attributes score zero. "
                     "Equal observed values score one. Ties break by product id.",
    })


def extract(previous):
    envelope = copy.deepcopy(validate_envelope(previous, "compare"))
    request = envelope["request"]
    selected = envelope["results"]["comparison"]["selected_product_id"]
    documents = selected_documents(request, selected)
    fields, issues = [], []
    for spec in request["extraction_schema"]["fields"]:
        labels = "|".join(re.escape(label.strip()) for label in spec["labels"])
        pattern = re.compile(r"^[ \t]*(?:" + labels + r")[ \t]*:[ \t]*(?P<raw>[^\r\n]*)",
                             re.IGNORECASE | re.MULTILINE)
        sources = []
        for document in documents:
            for match in pattern.finditer(document["text"]):
                raw = match.group("raw").strip()
                start = match.start("raw") + len(match.group("raw")) - len(
                    match.group("raw").lstrip())
                span = {"start": start, "end": start + len(raw)}
                try:
                    value = typed_value(raw, spec["type"])
                except ValidationError as error:
                    issues.append({"field": spec["name"], "document_id": document["id"],
                                   "span": span, "raw": raw, "reason": str(error)})
                    continue
                sources.append({"document_id": document["id"], "span": span,
                                "raw": raw, "value": value})
        fields.append({
            "name": spec["name"], "type": spec["type"], "required": spec["required"],
            "value": sources[0]["value"] if sources else None, "sources": sources,
            "conflict": len({scalar_key(source["value"]) for source in sources}) > 1,
        })
    return finish(envelope, "extract", {
        "product_id": selected, "document_ids": [document["id"] for document in documents],
        "fields": fields, "missing_fields": [field["name"] for field in fields
                                             if not field["sources"]],
        "missing_required_fields": [field["name"] for field in fields
                                    if not field["sources"] and field["required"]],
        "conflicting_fields": [field["name"] for field in fields if field["conflict"]],
        "issues": issues,
    })


def adaptive(previous):
    envelope = copy.deepcopy(validate_envelope(previous, "extract"))
    profile = envelope["request"]["onboarding"]
    extraction = envelope["results"]["extraction"]
    unresolved = sorted(set(extraction["missing_fields"] + extraction["conflicting_fields"]))
    completed, states, tasks = set(profile["completed"]), {}, []
    focus = set(extraction["missing_required_fields"] + extraction["conflicting_fields"])
    for task in task_order(profile["tasks"], profile["preferences"]):
        reasons = []
        missing = sorted(set(task["requires_fields"]) & set(unresolved))
        if task["id"] in completed:
            state = "completed"
            reasons.append("Previously completed; no action needed.")
        else:
            if missing:
                reasons.append("Missing or disputed fields: " + ", ".join(missing))
                focus.update(missing)
            if LEVELS.index(profile["experience"]) < LEVELS.index(task["min_experience"]):
                reasons.append("Requires experience: " + task["min_experience"])
            blocked = [dependency for dependency in task["prerequisites"]
                       if states[dependency] == "blocked"]
            if blocked:
                reasons.append("Blocked prerequisites: " + ", ".join(blocked))
            pending = [dependency for dependency in task["prerequisites"]
                       if dependency not in completed]
            state = "blocked" if reasons else ("waiting" if pending else "ready")
            if pending:
                reasons.append("Complete prerequisites: " + ", ".join(pending))
        states[task["id"]] = state
        mode = next((mode for mode in profile["preferences"] if mode in task["modes"]),
                    task["modes"][0])
        guidance = {
            "beginner": "Follow one step at a time and verify the result.",
            "intermediate": "Use the checklist and verify prerequisites.",
            "expert": "Use the concise verification path.",
        }[profile["experience"]]
        explanation = f"{task['explanation']} {guidance} Delivery mode: {mode}."
        tasks.append({"id": task["id"], "title": task["title"], "state": state,
                      "mode": mode, "explanation": explanation, "reasons": reasons})
    return finish(envelope, "adaptive", {
        "product_id": extraction["product_id"], "experience": profile["experience"],
        "preferences": profile["preferences"], "tasks": tasks,
        "ready_task_ids": [task["id"] for task in tasks if task["state"] == "ready"],
        "completed_task_ids": [task["id"] for task in tasks if task["state"] == "completed"],
        "unresolved_fields": unresolved, "research_focus_fields": sorted(focus),
    })


def deep(previous):
    envelope = copy.deepcopy(validate_envelope(previous, "adaptive"))
    results = envelope["results"]
    onboarding = results["onboarding"]
    fields = {field["name"]: field for field in results["extraction"]["fields"]}
    focus = onboarding["research_focus_fields"]
    questions = sorted(envelope["request"]["research"]["questions"],
                       key=lambda question: (question["field"] not in focus, question["id"]))
    answers = []
    for question in questions:
        field = fields[question["field"]]
        grouped = {}
        for source in field["sources"]:
            key = scalar_key(source["value"])
            if key not in grouped:
                grouped[key] = {"value": source["value"], "document_ids": [], "citations": []}
            group = grouped[key]
            group["citations"].append(copy.deepcopy(source))
            if source["document_id"] not in group["document_ids"]:
                group["document_ids"].append(source["document_id"])
        groups = list(grouped.values())
        count = len({source["document_id"] for source in field["sources"]})
        if not groups:
            status, summary = "unresolved", "No valid extracted evidence; obtain a source."
        elif len(groups) > 1:
            status = "disputed"
            summary = (f"{len(groups)} incompatible values across {count} document(s); "
                       "do not choose a value without resolving the disagreement.")
        else:
            status = "supported"
            summary = (f"Evidence supports {json.dumps(groups[0]['value'], ensure_ascii=True)} "
                       f"across {count} document(s). "
                       + ("Single-source evidence, not independently corroborated."
                          if count == 1 else
                          "Multiple documents agree; source independence is not established."))
        answers.append({"question_id": question["id"], "field": question["field"],
                        "question": question["text"], "status": status, "groups": groups,
                        "distinct_document_count": count, "summary": summary})
    disagreements = [answer["question_id"] for answer in answers
                     if answer["status"] == "disputed"]
    unresolved = [answer["question_id"] for answer in answers
                  if answer["status"] != "supported"]
    blockers = [task["id"] for task in onboarding["tasks"] if task["state"] == "blocked"]
    return finish(envelope, "deep", {
        "product_id": onboarding["product_id"], "focus_fields": list(focus),
        "answers": answers, "disagreements": disagreements, "unresolved_questions": unresolved,
        "onboarding_blockers": blockers,
        "synthesis": f"Synthetic evidence for {onboarding['product_id']}: "
                     f"{len(answers) - len(unresolved)} supported question(s), "
                     f"{len(disagreements)} disagreement(s), "
                     f"{len(unresolved)} unresolved question(s), "
                     f"{len(blockers)} blocked onboarding task(s). "
                     "No external research or unsupported inference was performed.",
    })


def run_pipeline(request):
    return deep(adaptive(extract(compare(request))))


def reject_constant(value):
    raise ValidationError(f"Non-finite JSON constant: {value}")


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        ensure(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        ensure(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with Path(args[0]).open("r", encoding="utf-8") as handle:
            request = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=reject_duplicate_keys)
        output = run_pipeline(request)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as error:
        print(json.dumps({"schema_version": 1, "status": "error",
                          "error": {"type": type(error).__name__, "message": str(error)}},
                         ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
