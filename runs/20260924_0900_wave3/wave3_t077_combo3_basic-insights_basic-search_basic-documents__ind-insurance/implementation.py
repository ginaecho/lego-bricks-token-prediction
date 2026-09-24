"""Synthetic insurance reference pipeline. Standard library; no network or providers."""
import copy
import datetime as dt
import json
import math
import re
import sys
import xml.etree.ElementTree as ET


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected object")
    require(set(required) <= set(value), "Missing fields: " + ", ".join(sorted(set(required) - set(value))))
    require(set(value) <= set(required) | set(optional), "Unknown fields are not allowed (data minimization)")


def text(value, label):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 4000, label + " must be nonempty bounded text")
    return value.strip()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"SYN-[A-Z0-9-]{1,60}", value), "IDs must be synthetic SYN- references")
    return value


def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and minimum <= value <= 1e9,
            label + " must be a finite nonnegative number <= 1e9")
    return value


def date(value):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), "Date must be ISO YYYY-MM-DD")
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError("Invalid calendar date") from exc
    return value


def sequence(value, label):
    require(isinstance(value, list) and len(value) <= 1000, label + " must be a bounded list")
    return value


def unique(records, key):
    require(len({r[key] for r in records}) == len(records), "Duplicate " + key)


THEMES = {
    "water": {"water", "flood", "flooding", "leak", "leaking", "burst"},
    "fire": {"fire", "smoke", "burn", "burning"},
    "theft": {"theft", "stolen", "burglary", "robbery"},
    "motor": {"motor", "car", "auto", "vehicle", "collision"},
    "speed": {"slow", "delay", "delayed", "waiting", "quick", "fast"},
    "clarity": {"unclear", "confusing", "explain", "explanation", "clarity"},
}
FACTORS = {"property_type", "construction", "year_built"}
STAGES = ("normalized", "insights", "search", "documents")
CORE = ("schema_version", "synthetic", "stage", "policies", "claims", "submissions",
        "feedback", "queries", "insights", "search", "documents")


def tokens(value):
    words = set(re.findall(r"[a-z0-9]+", value.lower()))
    return words | {theme for theme, synonyms in THEMES.items() if words & synonyms}


def parse_claim(raw):
    text(raw, "Claim form")
    result = {}
    names = {"Claim-ID": "claim_id", "Policy-ID": "policy_id",
             "Loss-Date": "loss_date", "Loss-Amount": "loss_amount", "Peril": "peril"}
    for line in raw.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        require(sep and key.strip() in names, "Unsupported claim-form field")
        key = names[key.strip()]
        require(key not in result, "Duplicate claim-form field")
        result[key] = value.strip()
    fields(result, names.values())
    try:
        result["loss_amount"] = float(result["loss_amount"])
    except ValueError as exc:
        raise ValidationError("Invalid loss amount") from exc
    return result


def parse_submission(raw):
    fields(raw, ("format", "data"))
    if raw["format"] == "json":
        value = copy.deepcopy(raw["data"])
    else:
        require(raw["format"] == "xml", "Submission format must be json or xml")
        payload = text(raw["data"], "XML submission")
        require("<!DOCTYPE" not in payload.upper() and "<!ENTITY" not in payload.upper(),
                "XML declarations/entities are forbidden")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ValidationError("Malformed submission XML") from exc
        require(root.tag == "ACORDSubmission" and not root.attrib, "Expected ACORDSubmission root")
        value = {"factors": []}
        for child in root:
            require(child.tag in {"SubmissionID", "PolicyID", "Factor", "FixtureVIN", "FixtureAddress"}, "Unknown XML field")
            require(not list(child), "Nested XML fields are unsupported")
            if child.tag == "Factor":
                require(set(child.attrib) == {"name", "disclosed"}, "Invalid factor attributes")
                require(child.attrib["disclosed"] in {"true", "false"}, "Invalid disclosure boolean")
                value["factors"].append({"name": child.attrib["name"], "value": child.text or "",
                                         "disclosed": child.attrib["disclosed"] == "true"})
            else:
                require(not child.attrib, "Unexpected XML attributes")
                key = {"SubmissionID": "submission_id", "PolicyID": "policy_id",
                       "FixtureVIN": "fixture_vin", "FixtureAddress": "fixture_address"}[child.tag]
                require(key not in value, "Duplicate XML field")
                value[key] = child.text or ""
    fields(value, ("submission_id", "policy_id", "factors"), ("fixture_address", "fixture_vin"))
    # Fixture-only location/vehicle identifiers never enter the shared envelope.
    for key in ("fixture_address", "fixture_vin"):
        if key in value:
            require(text(value[key], key).startswith("SYNTHETIC-"), "Fixture identifiers must be labeled")
            del value[key]
    return value


def validate(envelope, expected_stage=None):
    """One validator checks the shared envelope at every handoff."""
    fields(envelope, CORE)
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1, "Unsupported schema version")
    require(envelope["synthetic"] is True, "Only explicitly synthetic data is supported")
    stage = envelope["stage"]
    require(stage in STAGES and (expected_stage is None or stage == expected_stage), "Invalid pipeline stage")
    for key in CORE[3:]:
        sequence(envelope[key], key)
    policies = envelope["policies"]
    for p in policies:
        fields(p, ("policy_id", "holder_ref", "name", "description", "perils", "start_date", "end_date", "limit"))
        identifier(p["policy_id"])
        identifier(p["holder_ref"])
        text(p["name"], "Policy name")
        text(p["description"], "Policy description")
        sequence(p["perils"], "perils")
        require(p["perils"] and len(set(p["perils"])) == len(p["perils"]), "Perils must be nonempty and unique")
        for peril in p["perils"]:
            text(peril, "Peril")
            require(peril == peril.lower().strip(), "Perils must be lowercase canonical terms")
        require(date(p["start_date"]) <= date(p["end_date"]), "Policy dates reversed")
        number(p["limit"], "Policy limit")
    unique(policies, "policy_id")
    policy_ids = {p["policy_id"] for p in policies}
    for c in envelope["claims"]:
        fields(c, ("claim_id", "policy_id", "loss_date", "loss_amount", "peril"))
        identifier(c["claim_id"])
        require(c["policy_id"] in policy_ids, "Unknown claim policy")
        date(c["loss_date"])
        number(c["loss_amount"], "Loss amount")
        text(c["peril"], "Claim peril")
        require(c["peril"] == c["peril"].strip().lower(), "Claim peril must be canonical lowercase")
    unique(envelope["claims"], "claim_id")
    claim_ids = {c["claim_id"] for c in envelope["claims"]}
    for s in envelope["submissions"]:
        fields(s, ("submission_id", "policy_id", "factors"))
        identifier(s["submission_id"])
        require(s["policy_id"] in policy_ids, "Unknown submission policy")
        sequence(s["factors"], "Factors")
        require(s["factors"], "At least one disclosed underwriting factor is required")
        for factor in s["factors"]:
            fields(factor, ("name", "value", "disclosed"))
            require(factor["name"] in FACTORS and factor["disclosed"] is True,
                    "Underwriting factors must be permitted and explicitly disclosed")
            text(factor["value"], "Factor value")
        unique(s["factors"], "name")
    unique(envelope["submissions"], "submission_id")
    unique(envelope["submissions"], "policy_id")
    require({s["policy_id"] for s in envelope["submissions"]} == policy_ids, "Every policy requires its disclosed submission")
    for f in envelope["feedback"]:
        fields(f, ("feedback_id", "claim_id", "signals"))
        identifier(f["feedback_id"])
        require(f["claim_id"] in claim_ids, "Unknown feedback claim")
        sequence(f["signals"], "signals")
        require(all(v in THEMES for v in f["signals"]), "Unknown signal")
    unique(envelope["feedback"], "feedback_id")
    for q in envelope["queries"]:
        fields(q, ("claim_id", "terms"))
        require(q["claim_id"] in claim_ids, "Unknown query claim")
        sequence(q["terms"], "query terms")
        require(all(isinstance(t, str) and re.fullmatch("[a-z0-9]+", t) for t in q["terms"]), "Invalid query terms")
    unique(envelope["queries"], "claim_id")
    require({q["claim_id"] for q in envelope["queries"]} == claim_ids, "Exactly one query per claim is required")
    level = STAGES.index(stage)
    for key, threshold in (("insights", 1), ("search", 2), ("documents", 3)):
        if level < threshold:
            require(not envelope[key], "Premature stage output")
    if level >= 1:
        require(envelope["insights"] == derive_insights(envelope), "Invalid insights or provenance")
    if level >= 2:
        require(envelope["search"] == derive_search(envelope), "Invalid search or provenance")
    if level >= 3:
        require(envelope["documents"] == derive_documents(envelope), "Invalid documents or decision reasons")
    return envelope


def normalize(raw):
    fields(raw, ("schema_version", "synthetic", "policies", "claim_forms", "submissions", "feedback", "queries"))
    for key in ("policies", "claim_forms", "submissions", "feedback", "queries"):
        sequence(raw[key], key)
    e = {key: [] for key in CORE}
    e.update(schema_version=raw["schema_version"], synthetic=raw["synthetic"], stage="normalized",
             policies=copy.deepcopy(raw["policies"]),
             claims=[parse_claim(c) for c in raw["claim_forms"]],
             submissions=[parse_submission(s) for s in raw["submissions"]])
    for f in raw["feedback"]:
        fields(f, ("feedback_id", "claim_id", "text"))
        found = tokens(text(f["text"], "Feedback"))
        e["feedback"].append({"feedback_id": f["feedback_id"], "claim_id": f["claim_id"],
                              "signals": sorted(set(THEMES) & found)})
    for q in raw["queries"]:
        fields(q, ("claim_id", "text"))
        found = tokens(text(q["text"], "Query"))
        # Keep only useful catalog vocabulary; discard names, emails and arbitrary free text.
        vocabulary = set(THEMES)
        for p in raw["policies"]:
            fields(p, ("policy_id", "holder_ref", "name", "description", "perils", "start_date", "end_date", "limit"))
            vocabulary |= tokens(text(p["name"], "Policy name") + " " + text(p["description"], "Description"))
        e["queries"].append({"claim_id": q["claim_id"], "terms": sorted(found & vocabulary)})
    return validate(e, "normalized")


def derive_insights(e):
    result = []
    for claim in e["claims"]:
        entries = [f for f in e["feedback"] if f["claim_id"] == claim["claim_id"]]
        themes = []
        for theme in sorted(THEMES):
            evidence = sorted(f["feedback_id"] for f in entries if theme in f["signals"])
            if evidence:
                themes.append({"theme": theme, "count": len(evidence), "evidence": evidence,
                               "action": "Review " + theme + " feedback and provide a clear response."})
        result.append({"claim_id": claim["claim_id"], "themes": themes})
    return result


def derive_search(e):
    results = []
    insight_map = {i["claim_id"]: i for i in e["insights"]}
    for query in e["queries"]:
        themes = [t["theme"] for t in insight_map[query["claim_id"]]["themes"]]
        intent = set(query["terms"]) | set(themes)
        matches = []
        for p in e["policies"]:
            matched = sorted(intent & tokens(p["name"] + " " + p["description"] + " " + " ".join(p["perils"])))
            if matched:
                matches.append({"policy_id": p["policy_id"], "score": len(matched),
                                "matched_terms": matched, "reason": "Matches disclosed product terms: " + ", ".join(matched)})
        matches.sort(key=lambda m: (-m["score"], m["policy_id"]))
        results.append({"claim_id": query["claim_id"], "insight_themes": themes,
                        "intent_terms": sorted(intent), "matches": matches[:3]})
    return results


def derive_documents(e):
    policies = {p["policy_id"]: p for p in e["policies"]}
    submissions = {s["policy_id"]: s for s in e["submissions"]}
    searches = {s["claim_id"]: s for s in e["search"]}
    documents = []
    for c in e["claims"]:
        p = policies[c["policy_id"]]
        reasons = []
        if c["peril"] not in p["perils"]:
            reasons.append("Reported peril is not listed; human review of wording and evidence is required.")
        if not p["start_date"] <= c["loss_date"] <= p["end_date"]:
            reasons.append("Loss date is outside the supplied policy period; human review is required.")
        if c["loss_amount"] > p["limit"]:
            reasons.append("Reported amount exceeds the supplied limit; human review is required.")
        status = "human_review" if reasons else "preliminary_match"
        if not reasons:
            reasons = ["Reported peril, date and amount match the supplied policy terms; this is not approval."]
        search = searches[c["claim_id"]]
        submission = submissions[c["policy_id"]]
        documents.append({
            "document_id": "SYN-DOC-" + c["claim_id"][4:], "claim_id": c["claim_id"],
            "policy_id": c["policy_id"], "holder_ref": p["holder_ref"],
            "loss_date": c["loss_date"], "reported_loss_amount": c["loss_amount"],
            "assessment": status, "reasons": reasons, "human_decision_required": True,
            "submission_id": submission["submission_id"],
            "disclosed_underwriting_factors": copy.deepcopy(submission["factors"]),
            "customer_themes": search["insight_themes"],
            "suggested_policy_ids": [m["policy_id"] for m in search["matches"]],
            "search_note": "Suggestions do not replace the actual policy or determine claim outcomes.",
        })
    return documents


def advance(envelope, previous, current, key, derive):
    validate(envelope, previous)
    result = copy.deepcopy(envelope)
    result[key] = derive(result)
    result["stage"] = current
    return validate(result, current)


def customer_insights(e):
    return advance(e, "normalized", "insights", "insights", derive_insights)


def smart_search(e):
    return advance(e, "insights", "search", "search", derive_search)


def automate_documents(e):
    return advance(e, "search", "documents", "documents", derive_documents)


def run(raw):
    return automate_documents(smart_search(customer_insights(normalize(raw))))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "Usage: python -B implementation.py example_input.json")
        with open(args[0], encoding="utf-8") as stream:
            raw = json.load(stream)
        output = {"status": "ok", "data": run(raw)}
        code = 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        output = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
