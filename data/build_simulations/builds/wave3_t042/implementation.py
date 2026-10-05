"""Deterministic, extractive FAQ answering; Python standard library only."""

import json
import math
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


STOP_WORDS = frozenset(
    "a an the is are was were do does can could i my me you your how what "
    "when where please tell about to of for in on and with it".split()
)


def tokens(text):
    return set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)) - STOP_WORDS


def validate(value, kind, evidence=None):
    """One boundary validator for requests and optional answerer results."""
    if not isinstance(value, dict):
        raise ValidationError(f"{kind} must be an object")
    if kind == "request":
        required = {"question", "knowledge_base"}
        allowed = required | {"min_score", "max_results"}
        if not required <= value.keys() or value.keys() - allowed:
            raise ValidationError("request has missing or unknown fields")
        question = value["question"]
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValidationError("question must contain 1-2000 characters")
        articles = value["knowledge_base"]
        if not isinstance(articles, list) or len(articles) > 100:
            raise ValidationError("knowledge_base must be an array of at most 100 articles")
        seen = set()
        for article in articles:
            if not isinstance(article, dict) or set(article) != {"id", "title", "content"}:
                raise ValidationError("articles require exactly id, title and content")
            for key, limit in (("id", 100), ("title", 500), ("content", 10000)):
                item = article[key]
                if not isinstance(item, str) or not item.strip() or len(item) > limit:
                    raise ValidationError(f"article {key} must contain 1-{limit} characters")
            if article["id"] in seen:
                raise ValidationError("article ids must be unique")
            seen.add(article["id"])
        threshold = value.get("min_score", 0.5)
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 < threshold <= 1):
            raise ValidationError("min_score must be a finite number in (0, 1]")
        limit = value.get("max_results", 3)
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValidationError("max_results must be an integer in [1, 10]")
        return {
            "question": question.strip(),
            "knowledge_base": [dict(article) for article in articles],
            "min_score": threshold,
            "max_results": limit,
        }
    if kind == "answer":
        if set(value) != {"answer", "citations"}:
            raise ValidationError("answerer must return exactly answer and citations")
        citations = value["citations"]
        if (not isinstance(citations, list) or not citations
                or any(not isinstance(item, str) for item in citations)
                or len(set(citations)) != len(citations)):
            raise ValidationError("citations must be a nonempty array of unique ids")
        available = {item["id"]: item["quote"] for item in evidence}
        if any(item not in available for item in citations):
            raise ValidationError("citations must reference retrieved evidence")
        expected = "\n\n".join(available[item] for item in citations)
        if value["answer"] != expected:
            raise ValidationError("answer must exactly reproduce the cited evidence")
        return {"answer": expected, "citations": list(citations)}
    raise ValidationError("unknown validation boundary")


def error_result(message):
    return {
        "status": "error", "question": None, "answer": None,
        "citations": [], "evidence": [], "reason": "validation_or_execution_error",
        "error": message,
    }


def answer_question(request, answerer=None):
    """An injected callable receives copies of evidence, never a live provider."""
    try:
        request = validate(request, "request")
        query = tokens(request["question"])
        evidence = []
        for article in request["knowledge_base"]:
            # Content-only scoring prevents title-only matches from grounding an answer.
            matched = query & tokens(article["content"])
            score = len(matched) / len(query) if query else 0
            if matched and score >= request["min_score"]:
                evidence.append({
                    "id": article["id"], "title": article["title"],
                    "quote": article["content"], "score": score,
                    "matched_terms": sorted(matched),
                })
        evidence.sort(key=lambda item: (-item["score"], item["id"]))
        evidence = evidence[:request["max_results"]]
        result = {
            "status": "abstained", "question": request["question"],
            "answer": None, "citations": [], "evidence": evidence,
            "reason": "no_supported_match", "error": None,
        }
        if not evidence:
            return result
        if answerer is None:
            candidate = {
                "answer": "\n\n".join(item["quote"] for item in evidence),
                "citations": [item["id"] for item in evidence],
            }
        else:
            # Deep copies prevent an injected callable from rewriting grounding evidence.
            candidate = answerer(request["question"], json.loads(json.dumps(evidence)))
        grounded = validate(candidate, "answer", evidence)
        result.update(grounded)
        result.update(status="answered", reason=None)
        return result
    except ValidationError as exc:
        return error_result(str(exc))
    except Exception:
        return error_result("answerer or processing failed")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON field")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonfinite JSON number")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        result = error_result("usage: python -B implementation.py input.json")
    else:
        try:
            text = Path(args[0]).read_text(encoding="utf-8")
            request = json.loads(
                text, object_pairs_hook=unique_object, parse_constant=reject_constant
            )
            result = answer_question(request)
        except (OSError, UnicodeError, ValueError, RecursionError):
            result = error_result("could not read a valid UTF-8 JSON input file")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
