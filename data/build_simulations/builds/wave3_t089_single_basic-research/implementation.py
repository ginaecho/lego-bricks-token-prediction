"""Deterministic research synthesis; source statements are evidence, not verified facts."""

import json
import re
import sys
from datetime import date


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label, limit=20000):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    require(len(value) <= limit, label + " exceeds length limit")
    return value


def fields(value, required, optional, label):
    require(isinstance(value, dict), label + " must be an object")
    require(required <= value.keys(), label + " is missing required fields")
    require(value.keys() <= required | optional, label + " has unknown fields")


def validate(data):
    """The shared boundary for the CLI and direct Python callers."""
    fields(data, {"schema_version", "questions", "sources"},
           {"synthetic_fixture", "min_sources", "max_evidence"}, "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be 1")
    require(type(data.get("synthetic_fixture", False)) is bool,
            "synthetic_fixture must be boolean")
    for name, default, upper in (("min_sources", 2, 20), ("max_evidence", 5, 50)):
        value = data.get(name, default)
        require(type(value) is int and 1 <= value <= upper,
                name + " must be an integer between 1 and " + str(upper))
    questions, sources = data["questions"], data["sources"]
    require(isinstance(questions, list) and 1 <= len(questions) <= 50,
            "questions must contain 1 to 50 entries")
    require(isinstance(sources, list) and len(sources) <= 200,
            "sources must contain at most 200 entries")
    question_ids, source_ids = set(), set()
    for q in questions:
        fields(q, {"id", "text"}, set(), "question")
        text(q["id"], "question.id", 100)
        text(q["text"], "question.text", 2000)
        require(q["id"] not in question_ids, "duplicate question id")
        question_ids.add(q["id"])
    for source in sources:
        fields(source, {"id", "title", "content"},
               {"reliability", "published", "claims", "origin"}, "source")
        for name in ("id", "title", "content"):
            text(source[name], "source." + name, 100 if name == "id" else 20000)
        require(source["id"] not in source_ids, "duplicate source id")
        source_ids.add(source["id"])
        quality = source.get("reliability", 0.5)
        require(type(quality) in (int, float) and 0 <= quality <= 1,
                "source.reliability must be a finite number from 0 to 1")
        if "origin" in source:
            text(source["origin"], "source.origin", 200)
        if "published" in source:
            published = text(source["published"], "source.published", 10)
            require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", published)),
                    "source.published must be YYYY-MM-DD")
            try:
                date.fromisoformat(published)
            except ValueError as exc:
                raise ValidationError("source.published is not a valid date") from exc
        claims = source.get("claims", [])
        require(isinstance(claims, list) and len(claims) <= 100,
                "source.claims must contain at most 100 entries")
        for claim in claims:
            fields(claim, {"question_id", "stance", "quote"}, set(), "claim")
            text(claim["question_id"], "claim.question_id", 100)
            require(claim["question_id"] in question_ids, "claim references unknown question")
            require(isinstance(claim["stance"], str) and
                    claim["stance"] in {"supports", "opposes", "neutral"}, "invalid claim stance")
            text(claim["quote"], "claim.quote")
            require(claim["quote"] in source["content"], "claim.quote must occur verbatim in source")
    return data


STOP_WORDS = frozenset(
    "a an and are as at be by can could did do does for from how i in is it of on or "
    "should that the their this to was we were what when where which who why will with".split()
)


def tokens(value):
    return set(re.findall(r"[^\W_]+", value.casefold())) - STOP_WORDS


def synthesize(data):
    data = validate(data)
    results = []
    minimum = data.get("min_sources", 2)
    for question in data["questions"]:
        terms = tokens(question["text"])
        evidence = []
        for source in data["sources"]:
            claims = [c for c in source.get("claims", [])
                      if c["question_id"] == question["id"]]
            candidates = [(c["quote"], c["stance"], "user_annotation") for c in claims]
            if not candidates:
                sentences = re.split(r"(?<=[.!?])\s+|\n+", source["content"])
                candidates = [(s.strip(), "neutral", "lexical_retrieval")
                              for s in sentences if s.strip() and terms & tokens(s)]
            seen = set()
            for quote, stance, method in candidates:
                if (quote, stance) in seen:
                    continue
                seen.add((quote, stance))
                relevance = len(terms & tokens(quote)) / max(1, len(terms))
                quality = source.get("reliability", 0.5)
                evidence.append({
                    "source_id": source["id"], "title": source["title"],
                    "origin": source.get("origin", source["id"]),
                    "published": source.get("published"), "quote": quote,
                    "start_offset": source["content"].index(quote),
                    "stance": stance, "method": method,
                    "relevance": round(relevance, 4), "reliability": quality,
                    "rank_score": round(0.7 * relevance + 0.3 * quality, 4),
                })
        evidence.sort(key=lambda e: (-e["rank_score"], e["source_id"], e["quote"], e["stance"]))
        # Count origins, not passages; identical supplied texts are also one evidence family.
        content_origins = {}
        parents = {}

        def root(origin):
            parents.setdefault(origin, origin)
            while parents[origin] != origin:
                origin = parents[origin]
            return origin

        for source in sorted(data["sources"], key=lambda s: s["id"]):
            normalized = " ".join(source["content"].casefold().split())
            origin = source.get("origin", source["id"])
            previous = content_origins.setdefault(normalized, origin)
            parents[root(origin)] = root(previous)
        canonical_origins = {
            source["id"]: root(source.get("origin", source["id"])) for source in data["sources"]
        }
        qualified = [e for e in evidence if e["reliability"] >= 0.5]
        support = {canonical_origins[e["source_id"]] for e in qualified if e["stance"] == "supports"}
        oppose = {canonical_origins[e["source_id"]] for e in qualified if e["stance"] == "opposes"}
        all_origins = {canonical_origins[e["source_id"]] for e in qualified}
        if support and oppose:
            direction = "mixed"
            action = "Resolve conflicting source statements before deciding."
        elif len(support) >= minimum:
            direction = "supports"
            action = "Evidence supports the proposition; verify source claims before acting."
        elif len(oppose) >= minimum:
            direction = "opposes"
            action = "Evidence opposes the proposition; verify source claims before acting."
        else:
            direction = "insufficient"
            action = "Gather additional independent, directional evidence before deciding."
        limitations = ["No independent fact checking; reliability and stances are supplied annotations.",
                       "Lexical retrieval does not interpret negation, causality, or semantic equivalence.",
                       "Publication dates are reported but freshness is not assessed."]
        if not evidence:
            limitations.append("No matching evidence was found.")
        if len(all_origins) < minimum:
            limitations.append("Qualified evidence origins are below the requested minimum.")
        if evidence and not support and not oppose:
            limitations.append("Relevant passages alone do not establish a directional conclusion.")
        cap = data.get("max_evidence", 5)
        if len(evidence) > cap:
            limitations.append("Displayed evidence is capped; conclusion uses all matched evidence.")
        results.append({
            "question_id": question["id"], "question": question["text"],
            "direction": direction, "recommended_next_step": action,
            "evidence_strength": "corroborated" if direction in {"supports", "opposes"} else "limited",
            "counts": {"matched_passages": len(evidence), "qualified_origins": len(all_origins),
                       "supporting_origins": len(support), "opposing_origins": len(oppose)},
            "evidence": evidence[:cap], "limitations": limitations,
        })
    return {"schema_version": 1, "status": "ok",
            "synthetic_fixture": data.get("synthetic_fixture", False), "results": results}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("non-finite JSON number: " + value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json (or - for stdin)")
        if argv[0] == "-":
            raw = sys.stdin.read()
        else:
            with open(argv[0], encoding="utf-8") as stream:
                raw = stream.read()
        data = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        output = synthesize(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
