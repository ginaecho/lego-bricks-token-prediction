"""Deterministic synthetic research -> documents -> insights -> onboarding CLI."""

import copy
import json
import re
import sys
from collections import Counter


STAGES = ("input", "research", "documents", "insights", "onboard")
STOP_WORDS = set("a an and are can do does for how in is of on or the to what with".split())
THEMES = {
    "setup": ({"setup", "install", "installation", "configure", "onboarding"},
              "Complete a guided setup checklist and verify the first successful run."),
    "usability": ({"interface", "navigation", "usability", "layout", "buttons"},
                  "Walk through the core workflow with a worked example."),
    "reliability": ({"crash", "crashes", "broken", "error", "errors", "slow", "failure"},
                    "Run a health check and collect reproducible error details."),
    "pricing": ({"price", "pricing", "cost", "expensive", "billing"},
               "Review the plan, expected usage, and billing options."),
    "support": ({"support", "help", "response", "ticket"},
               "Introduce the support channel and agree on the next follow-up."),
}
POSITIVE = set("good great easy helpful excellent love fast clear success".split())
NEGATIVE = set("bad confusing difficult broken slow expensive poor crash crashes error errors failure".split())


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected.split()), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 20000,
            path + " must be a nonempty string of at most 20000 characters")


def sequence(value, path):
    require(isinstance(value, list) and len(value) <= 10000, path + " must be a bounded array")


def strings(value, path):
    sequence(value, path)
    for item in value:
        text(item, path)
    require(len(value) == len(set(value)), path + " must not contain duplicates")


def records(value, expected, path):
    sequence(value, path)
    ids = set()
    for item in value:
        fields(item, expected, path)
        text(item["id"], path + ".id")
        require(item["id"] not in ids, path + " contains duplicate IDs")
        ids.add(item["id"])
    return {item["id"]: item for item in value}


def count(value, path):
    require(type(value) is int and value >= 0, path + " must be a nonnegative integer")


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def validate(data, expected_stage=None):
    """The same validator guards external input and every internal handoff."""
    fields(data, "schema_version fixture_label questions sources customers stage results", "envelope")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    text(data["fixture_label"], "fixture_label")
    require(data["fixture_label"].startswith("SYNTHETIC"), "fixture_label must start with SYNTHETIC")
    require(isinstance(data["stage"], str) and data["stage"] in STAGES, "unknown stage")
    require(expected_stage is None or data["stage"] == expected_stage, "unexpected pipeline stage")
    questions = records(data["questions"], "id text keywords", "questions")
    require(bool(questions), "at least one question is required")
    for question in questions.values():
        text(question["text"], "question.text")
        strings(question["keywords"], "question.keywords")
        require(bool(query_terms(question)), "question needs searchable words or keywords")
    customers = records(data["customers"], "id name goal experience", "customers")
    for customer in customers.values():
        text(customer["name"], "customer.name")
        text(customer["goal"], "customer.goal")
        require(customer["experience"] in ("beginner", "experienced"), "invalid customer experience")
    sources = records(data["sources"], "id title text kind customer_id", "sources")
    for source in sources.values():
        text(source["title"], "source.title")
        text(source["text"], "source.text")
        require(source["kind"] in ("reference", "feedback"), "invalid source kind")
        require(source["customer_id"] is None or
                (isinstance(source["customer_id"], str) and source["customer_id"] in customers),
                "source references an unknown customer")
        require(source["kind"] != "reference" or source["customer_id"] is None,
                "reference sources cannot belong to a customer")
    rank = STAGES.index(data["stage"])
    fields(data["results"], " ".join(STAGES[1:rank + 1]), "results")
    if rank == 0:
        return data
    research = data["results"]["research"]
    fields(research, "evidence unanswered_question_ids", "research")
    evidence = records(research["evidence"], "id question_id source_id quote matched_terms",
                       "research.evidence")
    answered = set()
    for item in evidence.values():
        require(isinstance(item["question_id"], str) and item["question_id"] in questions,
                "evidence question is unknown")
        require(isinstance(item["source_id"], str) and item["source_id"] in sources,
                "evidence source is unknown")
        text(item["quote"], "evidence.quote")
        require(item["quote"] in sources[item["source_id"]]["text"], "evidence quote is not in source")
        strings(item["matched_terms"], "evidence.matched_terms")
        require(bool(item["matched_terms"]) and set(item["matched_terms"]) ==
                tokens(item["quote"]) & query_terms(questions[item["question_id"]]),
                "evidence matched_terms are inconsistent")
        answered.add(item["question_id"])
    strings(research["unanswered_question_ids"], "unanswered_question_ids")
    require(set(research["unanswered_question_ids"]) == set(questions) - answered,
            "unanswered question list is inconsistent")
    if rank == 1:
        return data
    docs = data["results"]["documents"]
    fields(docs, "rows checks", "documents")
    rows = records(docs["rows"], "id source_id source_title kind customer_id text word_count evidence_ids question_ids",
                   "documents.rows")
    seen = []
    for row in rows.values():
        require(isinstance(row["source_id"], str) and row["source_id"] in sources, "unknown row source")
        source = sources[row["source_id"]]
        for key in ("kind", "customer_id"):
            require(row[key] == source[key], "row source metadata mismatch")
        require(row["source_title"] == source["title"], "row source title mismatch")
        text(row["text"], "row.text")
        count(row["word_count"], "row.word_count")
        require(row["word_count"] == len(row["text"].split()), "row word count mismatch")
        strings(row["evidence_ids"], "row.evidence_ids")
        strings(row["question_ids"], "row.question_ids")
        require(bool(row["evidence_ids"]) and set(row["evidence_ids"]) <= set(evidence), "invalid row evidence")
        linked = [evidence[eid] for eid in row["evidence_ids"]]
        require(all(e["source_id"] == row["source_id"] and e["quote"] == row["text"]
                    for e in linked), "row evidence lineage mismatch")
        require(set(row["question_ids"]) == {e["question_id"] for e in linked}, "row question lineage mismatch")
        seen.extend(row["evidence_ids"])
    require(Counter(seen) == Counter(evidence.keys()), "documents must preserve each evidence item once")
    fields(docs["checks"], "evidence_count row_count duplicates_collapsed", "documents.checks")
    expected = {"evidence_count": len(evidence), "row_count": len(rows),
                "duplicates_collapsed": len(evidence) - len(rows)}
    for key, value in expected.items():
        count(docs["checks"][key], key)
        require(docs["checks"][key] == value, "document checks mismatch")
    if rank == 2:
        return data
    insights = data["results"]["insights"]
    fields(insights, "themes feedback_row_count", "insights")
    themes = records(insights["themes"], "id row_ids customer_ids sentiment_counts recommendation", "insights.themes")
    feedback_rows = {rid for rid, row in rows.items() if row["kind"] == "feedback"}
    count(insights["feedback_row_count"], "feedback_row_count")
    require(insights["feedback_row_count"] == len(feedback_rows), "feedback count mismatch")
    themed = []
    for theme in themes.values():
        require(theme["id"] in (*THEMES, "other"), "unknown theme")
        strings(theme["row_ids"], "theme.row_ids")
        require(bool(theme["row_ids"]) and set(theme["row_ids"]) <= feedback_rows, "invalid theme rows")
        strings(theme["customer_ids"], "theme.customer_ids")
        require(set(theme["customer_ids"]) ==
                {rows[rid]["customer_id"] for rid in theme["row_ids"] if rows[rid]["customer_id"] is not None},
                "theme customer lineage mismatch")
        expected_counts = dict.fromkeys(("positive", "negative", "mixed", "neutral"), 0)
        for rid in theme["row_ids"]:
            require(classify_theme(rows[rid]["text"]) == theme["id"], "theme classification mismatch")
            expected_counts[sentiment(rows[rid]["text"])] += 1
        fields(theme["sentiment_counts"], "positive negative mixed neutral", "sentiment_counts")
        for key, value in expected_counts.items():
            count(theme["sentiment_counts"][key], key)
            require(theme["sentiment_counts"][key] == value, "sentiment count mismatch")
        text(theme["recommendation"], "theme.recommendation")
        themed.extend(theme["row_ids"])
    require(Counter(themed) == Counter(feedback_rows), "each feedback row must occur in one theme")
    if rank == 3:
        return data
    onboard = data["results"]["onboard"]
    fields(onboard, "plans", "onboard")
    plans = records(onboard["plans"], "id customer_id customer_name goal experience basis theme_ids evidence_row_ids next_steps",
                    "onboard.plans")
    require(Counter(p["customer_id"] for p in plans.values()
                    if isinstance(p["customer_id"], str)) == Counter(customers.keys()),
            "one onboarding plan per customer is required")
    for plan in plans.values():
        require(isinstance(plan["customer_id"], str) and plan["customer_id"] in customers, "unknown plan customer")
        customer = customers[plan["customer_id"]]
        require(plan["customer_name"] == customer["name"] and plan["goal"] == customer["goal"]
                and plan["experience"] == customer["experience"], "plan personalization mismatch")
        require(plan["basis"] in ("personal_feedback", "shared_feedback", "goal_only"), "invalid plan basis")
        strings(plan["theme_ids"], "plan.theme_ids")
        require(set(plan["theme_ids"]) <= set(themes), "plan references unknown themes")
        strings(plan["evidence_row_ids"], "plan.evidence_row_ids")
        selected_rows = {rid for tid in plan["theme_ids"] for rid in themes[tid]["row_ids"]}
        if plan["basis"] == "personal_feedback":
            selected_rows = {rid for rid in selected_rows if rows[rid]["customer_id"] == customer["id"]}
        require(set(plan["evidence_row_ids"]) == selected_rows, "plan evidence lineage mismatch")
        require((plan["basis"] == "goal_only") == (not plan["theme_ids"]), "plan basis mismatch")
        require(plan["basis"] != "personal_feedback" or bool(selected_rows), "personal plan needs personal evidence")
        strings(plan["next_steps"], "plan.next_steps")
        require(bool(plan["next_steps"]), "plan needs a next step")
    return data


def query_terms(question):
    if question["keywords"]:
        return set().union(*(tokens(term) for term in question["keywords"]))
    return tokens(question["text"]) - STOP_WORDS


def advance(data, previous, stage, payload):
    validate(data, previous)
    result = copy.deepcopy(data)
    result["stage"] = stage
    result["results"][stage] = payload
    return validate(result, stage)


def research(data):
    validate(data, "input")
    evidence, unanswered = [], []
    for question in data["questions"]:
        before = len(evidence)
        for source in data["sources"]:
            for quote in re.split(r"(?<=[.!?])\s+|\n+", source["text"]):
                quote = quote.strip()
                matched = sorted(tokens(quote) & query_terms(question))
                if matched:
                    evidence.append({"id": "e" + str(len(evidence) + 1),
                                     "question_id": question["id"], "source_id": source["id"],
                                     "quote": quote, "matched_terms": matched})
        if len(evidence) == before:
            unanswered.append(question["id"])
    return advance(data, "input", "research", {"evidence": evidence, "unanswered_question_ids": unanswered})


def documents(data):
    validate(data, "research")
    sources = {s["id"]: s for s in data["sources"]}
    grouped = {}
    for item in data["results"]["research"]["evidence"]:
        key = (item["source_id"], item["quote"])
        if key not in grouped:
            source = sources[item["source_id"]]
            grouped[key] = {"id": "r" + str(len(grouped) + 1), "source_id": source["id"],
                            "source_title": source["title"], "kind": source["kind"],
                            "customer_id": source["customer_id"], "text": item["quote"],
                            "word_count": len(item["quote"].split()), "evidence_ids": [], "question_ids": []}
        row = grouped[key]
        row["evidence_ids"].append(item["id"])
        if item["question_id"] not in row["question_ids"]:
            row["question_ids"].append(item["question_id"])
    size = len(data["results"]["research"]["evidence"])
    return advance(data, "research", "documents", {"rows": list(grouped.values()),
                   "checks": {"evidence_count": size, "row_count": len(grouped),
                              "duplicates_collapsed": size - len(grouped)}})


def classify_theme(value):
    words = tokens(value)
    scores = [(len(words & lexicon), name) for name, (lexicon, _) in THEMES.items()]
    score, name = sorted(scores, key=lambda pair: (-pair[0], pair[1]))[0]
    return name if score else "other"


def sentiment(value):
    words = tokens(value)
    positive, negative = bool(words & POSITIVE), bool(words & NEGATIVE)
    return "mixed" if positive and negative else "positive" if positive else "negative" if negative else "neutral"


def insights(data):
    validate(data, "documents")
    grouped = {}
    feedback = [row for row in data["results"]["documents"]["rows"] if row["kind"] == "feedback"]
    for row in feedback:
        name = classify_theme(row["text"])
        if name not in grouped:
            grouped[name] = {"id": name, "row_ids": [], "customer_ids": [],
                             "sentiment_counts": dict.fromkeys(("positive", "negative", "mixed", "neutral"), 0),
                             "recommendation": THEMES[name][1] if name in THEMES else
                             "Ask a focused follow-up question and agree on one measurable outcome."}
        theme = grouped[name]
        theme["row_ids"].append(row["id"])
        if row["customer_id"] is not None and row["customer_id"] not in theme["customer_ids"]:
            theme["customer_ids"].append(row["customer_id"])
        theme["sentiment_counts"][sentiment(row["text"])] += 1
    ordered = sorted(grouped.values(), key=lambda theme: (
        -(theme["sentiment_counts"]["negative"] + theme["sentiment_counts"]["mixed"]),
        -len(theme["row_ids"]), theme["id"]))
    return advance(data, "documents", "insights", {"themes": ordered, "feedback_row_count": len(feedback)})


def onboard(data):
    validate(data, "insights")
    themes = data["results"]["insights"]["themes"]
    rows = {r["id"]: r for r in data["results"]["documents"]["rows"]}
    plans = []
    for customer in data["customers"]:
        personal = [theme for theme in themes if customer["id"] in theme["customer_ids"]]
        selected = (personal or themes)[:2]
        basis = "personal_feedback" if personal else "shared_feedback" if selected else "goal_only"
        evidence_rows = sorted({rid for theme in selected for rid in theme["row_ids"]
                                if not personal or rows[rid]["customer_id"] == customer["id"]})
        first = ("Start with a guided introduction" if customer["experience"] == "beginner"
                 else "Configure a focused pilot")
        steps = [f"{first} for {customer['name']}; target outcome: {customer['goal']}."]
        steps.extend(theme["recommendation"] for theme in selected)
        steps.append("Confirm the first milestone and schedule a progress check.")
        plans.append({"id": "p" + str(len(plans) + 1), "customer_id": customer["id"],
                      "customer_name": customer["name"], "goal": customer["goal"],
                      "experience": customer["experience"], "basis": basis,
                      "theme_ids": [theme["id"] for theme in selected], "evidence_row_ids": evidence_rows,
                      "next_steps": steps})
    return advance(data, "insights", "onboard", {"plans": plans})


def run_pipeline(data):
    state = validate(copy.deepcopy(data), "input")
    for stage in (research, documents, insights, onboard):
        state = stage(state)
    return {"status": "ok", "data": state}


def reject_constant(value):
    raise ValidationError("Nonfinite JSON numbers are not supported: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8-sig") as handle:
            data = json.load(handle, parse_constant=reject_constant)
        result = run_pipeline(data)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
