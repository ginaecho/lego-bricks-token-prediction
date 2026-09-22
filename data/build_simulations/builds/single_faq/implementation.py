"""Offline, standard-library FAQ retrieval. Run with --help for the file CLI."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys


STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it me my of "
    "on or please the to was what when where which who why with you your".split()
)
MAX_PASSAGE_CHARS = 1200


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Passage:
    article_id: str
    passage_index: int
    quote: str
    tokens: frozenset


@dataclass(frozen=True)
class Index:
    articles: tuple
    passages: tuple
    postings: dict


def tokens(text):
    return frozenset(
        token for token in re.findall(r"[^\W_]+", text.casefold())
        if token not in STOPWORDS
    )


def nonempty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def confidence_threshold(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("threshold must be a finite number between 0 and 1")
    return float(value)


def build_index(articles):
    """Validate articles, split exact source passages, and build token postings.

    IDs must be unique (case-sensitive). Repeated content under different IDs
    remains independently citable. Canonical ordering does not depend on input.
    """
    if not isinstance(articles, list):
        raise ValueError("articles must be an array")
    normalized = []
    seen = set()
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("each article must be an object")
        if not {"id", "text"} <= article.keys() or article.keys() - {"id", "title", "text"}:
            raise ValueError("article requires id and text; only title is optional")
        article_id = nonempty_string(article["id"], "article id")
        if article_id != article_id.strip():
            raise ValueError("article id cannot have surrounding whitespace")
        if article_id in seen:
            raise ValueError(f"duplicate article id: {article_id}")
        seen.add(article_id)
        text = nonempty_string(article["text"], "article text")
        title = article.get("title", "")
        if not isinstance(title, str):
            raise ValueError("article title must be a string")
        normalized.append(Article(article_id, title, text))
    normalized.sort(key=lambda article: article.id)
    passages = []
    postings = {}
    for article in normalized:
        ordinal = 0
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", article.text):
            for offset in range(0, len(sentence), MAX_PASSAGE_CHARS):
                quote = sentence[offset:offset + MAX_PASSAGE_CHARS].strip()
                if not quote:
                    continue
                passage = Passage(article.id, ordinal, quote, tokens(quote))
                ordinal += 1
                position = len(passages)
                passages.append(passage)
                for token in sorted(passage.tokens):
                    postings.setdefault(token, []).append(position)
    return Index(tuple(normalized), tuple(passages),
                 {token: tuple(positions) for token, positions in sorted(postings.items())})


def index_document(index):
    return {
        "schema_version": 1,
        "articles": [
            {"id": a.id, "title": a.title, "text": a.text} for a in index.articles
        ],
        "passages": [
            {"article_id": p.article_id, "passage_index": p.passage_index,
             "quote": p.quote, "tokens": sorted(p.tokens)}
            for p in index.passages
        ],
        "postings": {token: list(positions) for token, positions in index.postings.items()},
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON numeric constant: {value}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object,
                         parse_constant=_invalid_constant)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def load_index(path):
    document = read_json(path)
    if (not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1):
        raise ValueError("index must have schema_version 1")
    index = build_index(document.get("articles"))
    # Compare serialized JSON as well as values: True must not equal integer 1.
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
    if canonical(document) != canonical(index_document(index)):
        raise ValueError("index passages or postings do not match its source articles")
    return index


def retrieve(index, question, limit=3):
    """Rank by unique query-token coverage; break ties by ID then passage index."""
    nonempty_string(question, "question")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    query_tokens = tokens(question)
    positions = set()
    for token in query_tokens:
        positions.update(index.postings.get(token, ()))
    ranked = []
    for position in positions:
        passage = index.passages[position]
        score = len(query_tokens & passage.tokens) / len(query_tokens)
        ranked.append({
            "article_id": passage.article_id,
            "passage_index": passage.passage_index,
            "quote": passage.quote,
            "score": score,
        })
    ranked.sort(key=lambda p: (-p["score"], p["article_id"], p["passage_index"]))
    return ranked[:limit]


def _callback_answer(value, candidates):
    if not isinstance(value, dict) or set(value) != {"answer", "citations"}:
        raise ValueError("callback must return answer and citations")
    citations = value["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValueError("callback must cite evidence")
    allowed = {
        (p["article_id"], p["passage_index"], p["quote"]) for p in candidates
    }
    seen = set()
    verified = []
    for citation in citations:
        if (not isinstance(citation, dict)
                or set(citation) != {"article_id", "passage_index", "quote"}
                or not isinstance(citation["article_id"], str)
                or type(citation["passage_index"]) is not int
                or not isinstance(citation["quote"], str)):
            raise ValueError("invalid callback citation")
        key = (citation["article_id"], citation["passage_index"], citation["quote"])
        if key not in allowed or key in seen:
            raise ValueError("callback citation must uniquely match retrieved evidence")
        seen.add(key)
        verified.append(dict(citation))
    answer = "\n".join(c["quote"] for c in verified)
    if value["answer"] != answer:
        raise ValueError("callback answer must consist only of exact cited quotes")
    return answer, verified


def answer_question(index, question, threshold=0.5, answer_callback=None):
    """Callback signature: callback(question, tuple_of_candidate_dicts).

    Callback answers must be newline-joined complete retrieved quotes. This
    deliberately forbids paraphrase so every returned claim is source text.
    """
    threshold = confidence_threshold(threshold)
    ranked = retrieve(index, question)
    confidence = ranked[0]["score"] if ranked else 0.0
    result = {
        "question": question, "status": "abstained", "answer": None,
        "citations": [], "confidence": confidence, "threshold": threshold,
        "reason": "insufficient_evidence",
    }
    if not index.passages:
        result["reason"] = "empty_corpus"
        return result
    if not ranked:
        result["reason"] = "no_matching_evidence"
        return result
    if confidence < threshold:
        return result
    candidates = [dict(p) for p in ranked if p["score"] >= threshold]
    if answer_callback is None:
        best = candidates[0]
        answer = best["quote"]
        citations = [{k: best[k] for k in ("article_id", "passage_index", "quote")}]
    else:
        try:
            # Do not expose the validation evidence to mutation by the callback.
            value = answer_callback(question, tuple(dict(p) for p in candidates))
            answer, citations = _callback_answer(value, candidates)
        except Exception:
            result["reason"] = "invalid_callback_output"
            return result
    result.update(status="answered", answer=answer, citations=citations, reason=None)
    return result


def read_request(path):
    request = read_json(path)
    if (not isinstance(request, dict) or "articles" not in request
            or request.keys() - {"articles", "questions", "threshold"}):
        raise ValueError("request requires articles; questions and threshold are optional")
    index = build_index(request["articles"])
    questions = request.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questions must be an array")
    for question in questions:
        nonempty_string(question, "question")
    threshold = confidence_threshold(request.get("threshold", 0.5))
    return index, questions, threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="create a persisted JSON passage index")
    index_parser.add_argument("--input", required=True)
    index_parser.add_argument("--output", required=True)
    ask_parser = subparsers.add_parser("ask", help="query a persisted JSON index")
    ask_parser.add_argument("--index", required=True)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--threshold", type=float, default=0.5)
    run_parser = subparsers.add_parser("run", help="index and answer a JSON request")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            if Path(args.input).resolve() == Path(args.output).resolve():
                raise ValueError("index output must not overwrite the input")
            index, _, _ = read_request(args.input)
            write_json(args.output, index_document(index))
            result = {"status": "indexed", "articles": len(index.articles),
                      "passages": len(index.passages)}
        elif args.command == "ask":
            result = answer_question(load_index(args.index), args.question, args.threshold)
        else:
            index, questions, threshold = read_request(args.input)
            if args.threshold is not None:
                threshold = confidence_threshold(args.threshold)
            result = {"results": [
                answer_question(index, question, threshold) for question in questions
            ]}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, UnicodeError, RecursionError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
