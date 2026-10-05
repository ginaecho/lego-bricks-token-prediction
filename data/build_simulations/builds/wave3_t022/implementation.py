"""Synthetic, deterministic research -> personalized discovery reference CLI.

Run: python -B implementation.py example_input.json
Research accepts explicit claims grounded by exact source quotations; it does
not extract facts from prose or equate source agreement with objective truth.
"""

import copy
import json
import math
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, path):
    require(isinstance(value, dict), path + " must be an object")
    require(set(value) == set(expected), path + " has missing or unknown fields")


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonblank text")


def unit(value, path, positive=False):
    require(type(value) in (int, float) and
            (type(value) is int or math.isfinite(value)),
            path + " must be a finite number")
    require((0 < value <= 1) if positive else (0 <= value <= 1),
            path + " must be in " + ("(0, 1]" if positive else "[0, 1]"))


def array(value, path):
    require(isinstance(value, list), path + " must be an array")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate(state, stage="input"):
    """One validation layer for input, research handoff and completed output."""
    require(stage in ("input", "research", "output"), "Unknown validation stage")
    root = {"schema_version", "synthetic", "documents", "products", "profile", "limit"}
    if stage != "input":
        root |= {"status", "research"}
    if stage == "output":
        root.add("recommendations")
    fields(state, root, "pipeline")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1,
            "schema_version must be integer 1")
    require(state["synthetic"] is True, "This reference requires synthetic=true")
    require(type(state["limit"]) is int and 1 <= state["limit"] <= 100,
            "limit must be an integer from 1 to 100")
    for name in ("documents", "products"):
        array(state[name], name)
    seen = set()
    for doc in state["documents"]:
        fields(doc, {"id", "title", "text", "reliability", "claims"}, "document")
        for key in ("id", "title", "text"):
            text(doc[key], "document." + key)
        require(doc["id"] not in seen, "Duplicate document id")
        seen.add(doc["id"])
        unit(doc["reliability"], "document.reliability")
        array(doc["claims"], "document.claims")
        claims = set()
        for claim in doc["claims"]:
            fields(claim, {"topic", "value", "stance", "confidence", "quote"}, "claim")
            for key in ("topic", "value", "quote"):
                text(claim[key], "claim." + key)
            require(claim["stance"] in ("support", "oppose"), "Invalid claim stance")
            unit(claim["confidence"], "claim.confidence")
            require(claim["quote"] in doc["text"], "Claim quote is not in source text")
            key = (claim["topic"], claim["value"], claim["stance"])
            require(key not in claims, "Duplicate claim within document")
            claims.add(key)
    seen = set()
    for product in state["products"]:
        fields(product, {"id", "name", "attributes"}, "product")
        for key in ("id", "name"):
            text(product[key], "product." + key)
        require(product["id"] not in seen, "Duplicate product id")
        seen.add(product["id"])
        require(isinstance(product["attributes"], dict), "attributes must be an object")
        for topic, value in product["attributes"].items():
            text(topic, "attribute topic")
            text(value, "attribute value")
    fields(state["profile"], {"preferences", "exclude_product_ids"}, "profile")
    array(state["profile"]["preferences"], "profile.preferences")
    array(state["profile"]["exclude_product_ids"], "profile.exclude_product_ids")
    topics = set()
    for preference in state["profile"]["preferences"]:
        fields(preference, {"topic", "value", "weight"}, "preference")
        text(preference["topic"], "preference.topic")
        text(preference["value"], "preference.value")
        unit(preference["weight"], "preference.weight", positive=True)
        require(preference["topic"] not in topics, "Duplicate preference topic")
        topics.add(preference["topic"])
    exclusions = state["profile"]["exclude_product_ids"]
    for value in exclusions:
        text(value, "excluded product id")
    require(len(set(exclusions)) == len(exclusions), "Duplicate product exclusion")
    require(set(exclusions) <= seen, "Excluded product id is not in catalog")
    if stage != "input":
        require(state["status"] == ("researched" if stage == "research" else "ok"),
                "Invalid stage status")
        require(canonical(state["research"]) == canonical(_research(state)),
                "Research handoff is inconsistent with validated source evidence")
    if stage == "output":
        require(canonical(state["recommendations"]) == canonical(_rank(state)),
                "Recommendations are inconsistent with research handoff")
    return state


def _research(state):
    grouped = {}
    for doc in sorted(state["documents"], key=lambda d: d["id"]):
        for claim in doc["claims"]:
            key = (claim["topic"], claim["value"])
            grouped.setdefault(key, []).append({
                "document_id": doc["id"],
                "quote": claim["quote"],
                "stance": claim["stance"],
                "weight": doc["reliability"] * claim["confidence"],
            })
    findings = []
    questions = []
    for (topic, value), citations in sorted(grouped.items()):
        citations.sort(key=lambda c: (c["document_id"], c["stance"]))
        support = math.fsum(c["weight"] for c in citations if c["stance"] == "support")
        oppose = math.fsum(c["weight"] for c in citations if c["stance"] == "oppose")
        total = support + oppose
        assessment = ("unresolved" if total == 0 else
                      "disputed" if support > 0 and oppose > 0 else
                      "supported" if support > 0 else "opposed")
        finding = {
            "topic": topic, "value": value,
            "support_weight": support, "oppose_weight": oppose,
            "evidence_balance": (support - oppose) / total if total else 0.0,
            "assessment": assessment,
            "source_count": len({c["document_id"] for c in citations}),
            "citations": citations,
        }
        findings.append(finding)
        if assessment in ("unresolved", "disputed"):
            questions.append({
                "topic": topic, "value": value, "reason": assessment,
                "question": "What additional independent evidence resolves "
                            + topic + "=" + value + "?",
            })
    for pref in sorted(state["profile"]["preferences"], key=lambda p: p["topic"]):
        if (pref["topic"], pref["value"]) not in grouped:
            questions.append({
                "topic": pref["topic"], "value": pref["value"], "reason": "no_evidence",
                "question": "What evidence supports " + pref["topic"] + "=" + pref["value"] + "?",
            })
    return {
        "document_count": len(state["documents"]),
        "findings": findings,
        "disagreements": [
            {"topic": f["topic"], "value": f["value"],
             "supporting_document_ids": sorted({c["document_id"] for c in f["citations"]
                                                if c["stance"] == "support" and c["weight"] > 0}),
             "opposing_document_ids": sorted({c["document_id"] for c in f["citations"]
                                              if c["stance"] == "oppose" and c["weight"] > 0})}
            for f in findings if f["assessment"] == "disputed"
        ],
        "unresolved_questions": questions,
        "limitations": [
            "Synthetic fixture claims, not independently verified facts.",
            "Exact topic/value matching; source independence is not verified.",
            "Evidence balance measures weighted agreement, not truth probability.",
        ],
    }


def _rank(state):
    evidence = {(f["topic"], f["value"]): f for f in state["research"]["findings"]}
    preferences = sorted(state["profile"]["preferences"], key=lambda p: p["topic"])
    total_weight = math.fsum(p["weight"] for p in preferences)
    results = []
    for product in state["products"]:
        if product["id"] in state["profile"]["exclude_product_ids"]:
            continue
        reasons = []
        contributions = []
        # Cold-start discovery uses catalog attributes, without inventing preferences.
        targets = preferences or [
            {"topic": k, "value": v, "weight": 1.0}
            for k, v in sorted(product["attributes"].items())
        ]
        for pref in targets:
            matched = product["attributes"].get(pref["topic"]) == pref["value"]
            finding = evidence.get((pref["topic"], pref["value"])) if matched else None
            balance = finding["evidence_balance"] if finding else 0.0
            if preferences:
                contribution = pref["weight"] * (0.75 + 0.25 * balance) if matched else 0.0
            else:
                contribution = max(0.0, balance) if finding else 0.0
            contributions.append(contribution)
            reasons.append({
                "topic": pref["topic"], "desired_value": pref["value"],
                "actual_value": product["attributes"].get(pref["topic"]),
                "matched": matched,
                "assessment": finding["assessment"] if finding else
                              ("no_evidence" if matched else "not_matched"),
                "evidence_balance": balance,
                "document_ids": sorted({c["document_id"] for c in finding["citations"]})
                                if finding else [],
                "weighted_contribution": contribution,
            })
        denominator = total_weight if preferences else len(targets)
        score = math.fsum(contributions) / denominator if denominator else 0.0
        results.append({
            "product_id": product["id"], "name": product["name"],
            "score": score,
            "mode": "personalized" if preferences else "cold_start",
            "reasons": reasons,
            "unresolved_questions": [
                copy.deepcopy(q) for q in state["research"]["unresolved_questions"]
                if any(q["topic"] == r["topic"] and q["value"] == r["desired_value"]
                       and r["matched"] for r in reasons)
            ],
        })
    results.sort(key=lambda r: (-r["score"], r["product_id"]))
    return results[:state["limit"]]


def synthesize(state):
    validate(state, "input")
    result = copy.deepcopy(state)
    result["status"] = "researched"
    result["research"] = _research(result)
    return validate(result, "research")


def recommend(research_state):
    validate(research_state, "research")
    result = copy.deepcopy(research_state)
    result["status"] = "ok"
    result["recommendations"] = _rank(result)
    return validate(result, "output")


def run(state):
    return recommend(synthesize(state))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON constant: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as stream:
            state = json.load(stream, object_pairs_hook=unique_object,
                              parse_constant=reject_constant)
        result = run(state)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
