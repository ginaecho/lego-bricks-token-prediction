"""Deterministic, offline research briefs. Run: python -B implementation.py INPUT."""

import json
import math
import re
import sys
from pathlib import Path


SCHEMA_VERSION = "1.0"
MAX_BYTES = 1_000_000
STOP_WORDS = set(
    "a an and are as at be by can did do does for from how i in is it of on or "
    "should that the their this to was we what when where which who will with".split()
)


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def record(value, required, optional, path):
    require(isinstance(value, dict), f"{path} must be an object")
    require(set(required) <= value.keys(), f"{path} missing required fields")
    require(value.keys() <= set(required) | set(optional), f"{path} has unknown fields")


def text(value, path, limit=20000):
    require(isinstance(value, str) and bool(value.strip()), f"{path} must be nonempty text")
    require(len(value) <= limit, f"{path} exceeds {limit} characters")


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)) - STOP_WORDS


def validate(data):
    """Single validation boundary shared by the Python API and CLI."""
    record(data, ["schema_version", "fixture_label", "questions", "sources"], [], "input")
    require(data["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    text(data["fixture_label"], "fixture_label", 200)
    require(isinstance(data["questions"], list) and 1 <= len(data["questions"]) <= 20,
            "questions must contain 1..20 items")
    require(isinstance(data["sources"], list) and len(data["sources"]) <= 100,
            "sources must contain 0..100 items")
    question_ids = set()
    for i, question in enumerate(data["questions"]):
        path = f"questions[{i}]"
        record(question, ["id", "text"], ["keywords"], path)
        text(question["id"], path + ".id", 80)
        text(question["text"], path + ".text", 1000)
        require(question["id"] not in question_ids, "duplicate question id")
        question_ids.add(question["id"])
        if "keywords" in question:
            require(isinstance(question["keywords"], list) and
                    1 <= len(question["keywords"]) <= 30, path + ".keywords must have 1..30 items")
            for keyword in question["keywords"]:
                text(keyword, path + ".keywords[]", 80)
        query = tokens(" ".join(question.get("keywords", [question["text"]])))
        require(bool(query), path + " has no searchable terms")
    source_ids = set()
    for i, source in enumerate(data["sources"]):
        path = f"sources[{i}]"
        record(source, ["id", "title", "text", "reliability"],
               ["origin", "locator", "positions"], path)
        for field, limit in (("id", 80), ("title", 300), ("text", 20000)):
            text(source[field], path + "." + field, limit)
        require(source["id"] not in source_ids, "duplicate source id")
        source_ids.add(source["id"])
        quality = source["reliability"]
        require(type(quality) in (int, float) and 0 <= quality <= 1 and
                math.isfinite(quality), path + ".reliability must be a finite number in [0,1]")
        for field in ("origin", "locator"):
            if field in source:
                text(source[field], path + "." + field, 500)
        positions = source.get("positions", {})
        require(isinstance(positions, dict), path + ".positions must be an object")
        require(positions.keys() <= question_ids, path + ".positions references unknown question")
        require(all(isinstance(v, str) and v in ("support", "oppose", "neutral")
                    for v in positions.values()), path + ".positions has invalid stance")
    return data


def research(data):
    """Extract source quotes; never invent answers or infer stance from prose.

    Positions are caller-supplied annotations, not verified truth. Reliability is
    also caller supplied. Equal origin identifiers prevent syndicated evidence
    from accumulating additional directional weight.
    """
    validate(data)
    briefs = []
    for question in data["questions"]:
        query = tokens(" ".join(question.get("keywords", [question["text"]])))
        evidence = []
        matched_sources = set()
        origin_weights = {}
        for source in data["sources"]:
            candidates = []
            for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", source["text"])):
                quote = sentence.strip()
                matches = sorted(tokens(quote) & query)
                if matches:
                    relevance = len(matches) / len(query)
                    candidates.append((relevance, index, quote, matches))
            candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
            if not candidates:
                continue
            matched_sources.add(source["id"])
            stance = source.get("positions", {}).get(question["id"], "neutral")
            origin = source.get("origin", source["id"])
            weight = candidates[0][0] * source["reliability"]
            group = origin_weights.setdefault(origin, {"support": 0.0, "oppose": 0.0, "neutral": 0.0})
            group[stance] = max(group[stance], weight)
            for relevance, index, quote, matches in candidates[:3]:
                evidence.append({
                    "source_id": source["id"], "source_title": source["title"],
                    "origin": origin, "locator": source.get("locator"),
                    "passage_index": index, "quote": quote, "matched_terms": matches,
                    "relevance": round(relevance, 6), "reliability": source["reliability"],
                    "stance": stance, "weighted_relevance": round(relevance * source["reliability"], 6)
                })
        evidence.sort(key=lambda item: (-item["weighted_relevance"],
                                       item["source_id"], item["passage_index"]))
        totals = {stance: sum(group[stance] for group in origin_weights.values())
                  for stance in ("support", "oppose", "neutral")}
        positive_origins = sum(max(group.values()) > 0 for group in origin_weights.values())
        if not positive_origins:
            assessment = "insufficient_evidence"
        elif totals["support"] > 0 and totals["oppose"] > 0:
            assessment = "mixed"
        elif totals["support"] > 0:
            assessment = "support"
        elif totals["oppose"] > 0:
            assessment = "oppose"
        else:
            assessment = "informational"
        limitations = [
            "Lexical matching can miss paraphrases and retrieve irrelevant mentions.",
            "Reliability, stance and origin annotations are supplied by the caller, not independently verified.",
            "Scores measure weighted term coverage, not probability or factual confidence."
        ]
        if len(matched_sources) > len(origin_weights):
            limitations.append("Sources sharing an origin are deduplicated for directional weighting.")
        if positive_origins < 2:
            limitations.append("Fewer than two positive-weight independent origins; corroboration is limited.")
        if not evidence:
            limitations.append("No source passages matched the question terms.")
        if assessment == "mixed":
            next_step = "Resolve conflicting source positions before deciding; inspect the cited passages."
        elif assessment == "insufficient_evidence":
            next_step = "Gather relevant, credible source material before making a decision."
        elif assessment == "informational":
            next_step = "Review the cited facts and annotate decision positions if a directional answer is needed."
        else:
            next_step = "The annotated evidence leans " + assessment + "; verify applicability and contrary evidence before acting."
        briefs.append({
            "question_id": question["id"], "question": question["text"],
            "search_terms": sorted(query), "assessment": assessment,
            "evidence": evidence, "source_count": len(matched_sources),
            "independent_origin_count": len(origin_weights),
            "positive_weight_origin_count": positive_origins,
            "directional_weights": {key: round(value, 6) for key, value in totals.items()},
            "limitations": limitations, "recommended_next_step": next_step
        })
    return {
        "schema_version": SCHEMA_VERSION, "status": "ok", "fixture_label": data["fixture_label"],
        "briefs": briefs, "method": "deterministic lexical retrieval with caller-annotated positions"
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "input file exceeds 1000000 bytes")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        result = research(data)
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                          "error": str(error)}, ensure_ascii=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
