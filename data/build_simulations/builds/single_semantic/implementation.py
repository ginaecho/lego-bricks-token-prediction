"""Offline text retrieval: smoothed TF-IDF or injected embedding cosine.

Run: python implementation.py example_input.json
API: TextIndex(documents, embedding_callback=None).search(query, top_k=5,
min_score=0.0, mode="auto"). Documents are {"id": str, "text": str}.
The callback receives a list of document dictionaries and returns an exact
ID-to-vector mapping. It is called once at construction and once per embedding
query, whose ID is "__query__". All vectors must have the same positive dimension.
"""

import argparse
from collections import Counter
from collections.abc import Mapping
import html
import json
import math
from pathlib import Path
import re
import sys


TOKEN = re.compile(r"\w+", re.UNICODE)
QUERY_ID = "__query__"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _tokens(text):
    return [match.group().casefold() for match in TOKEN.finditer(text)]


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _unit_vector(values):
    # Scaling first avoids overflow/underflow for finite, extreme coordinates.
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0:
        return tuple(0.0 for _ in values)
    scaled = [value / scale for value in values]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def _validate_vectors(raw, expected_ids, dimensions=None):
    if not isinstance(raw, Mapping) or set(raw) != set(expected_ids):
        raise ValueError("embedding IDs must exactly match requested IDs")
    result = {}
    for identifier in expected_ids:
        vector = raw[identifier]
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("each embedding must be a nonempty list or tuple")
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions:
            raise ValueError("embedding dimensionality mismatch")
        result[identifier] = _unit_vector(
            [_number(value, "embedding coordinate") for value in vector]
        )
    return result, dimensions


def _evidence(text, query_tokens):
    spans = [
        {"start": match.start(), "end": match.end(), "text": match.group()}
        for match in TOKEN.finditer(text)
        if match.group().casefold() in query_tokens
    ]
    pieces = []
    position = 0
    for span in spans:
        pieces.append(html.escape(text[position:span["start"]]))
        pieces.append("<mark>" + html.escape(span["text"]) + "</mark>")
        position = span["end"]
    pieces.append(html.escape(text[position:]))
    return spans, "".join(pieces)


class TextIndex:
    """Deterministic in-memory index; ties break by ascending document ID."""

    def __init__(self, documents, embedding_callback=None):
        if not isinstance(documents, (list, tuple)) or not documents:
            raise ValueError("documents must be a nonempty list or tuple")
        if embedding_callback is not None and not callable(embedding_callback):
            raise ValueError("embedding_callback must be callable")
        copied = {}
        for document in documents:
            if not isinstance(document, Mapping) or set(document) != {"id", "text"}:
                raise ValueError("each document must contain exactly id and text")
            identifier = _text(document["id"], "document ID")
            text = _text(document["text"], "document text")
            if not _tokens(text):
                raise ValueError("document text must contain a word token")
            if identifier in copied:
                raise ValueError(f"duplicate document ID: {identifier}")
            copied[identifier] = text
        self._documents = dict(sorted(copied.items()))
        counts = {key: Counter(_tokens(text)) for key, text in self._documents.items()}
        frequencies = Counter(token for count in counts.values() for token in count)
        self._idf = {
            token: math.log((1 + len(counts)) / (1 + frequency)) + 1
            for token, frequency in frequencies.items()
        }
        self._lexical = {key: self._tfidf(count) for key, count in counts.items()}
        self._callback = embedding_callback
        self._embeddings = None
        self._dimensions = None
        if embedding_callback is not None:
            batch = [{"id": key, "text": text} for key, text in self._documents.items()]
            self._embeddings, self._dimensions = _validate_vectors(
                embedding_callback(batch), self._documents
            )

    def _tfidf(self, counts):
        weighted = {
            token: count * self._idf[token]
            for token, count in counts.items()
            if token in self._idf
        }
        # Raw count TF and relative-frequency TF are equivalent after L2 normalization.
        keys = sorted(weighted)
        return dict(zip(keys, _unit_vector([weighted[key] for key in keys])))

    @property
    def metadata(self):
        return {
            "document_count": len(self._documents),
            "vocabulary_size": len(self._idf),
            "embedding_dimensions": self._dimensions,
        }

    def search(self, query, top_k=5, min_score=0.0, mode="auto"):
        query = _text(query, "query")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        min_score = _number(min_score, "min_score")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between zero and one")
        if mode not in ("auto", "lexical", "embedding"):
            raise ValueError("mode must be auto, lexical, or embedding")
        use_embedding = mode == "embedding" or (
            mode == "auto" and self._callback is not None
        )
        if use_embedding and self._callback is None:
            raise ValueError("embedding mode requires an embedding callback")
        tokens = set(_tokens(query))
        if use_embedding:
            vectors, _ = _validate_vectors(
                self._callback([{"id": QUERY_ID, "text": query}]),
                [QUERY_ID],
                self._dimensions,
            )
            query_vector = vectors[QUERY_ID]
            zero_query = not any(query_vector)
            scores = {
                key: math.fsum(a * b for a, b in zip(query_vector, vector))
                for key, vector in self._embeddings.items()
            }
        else:
            query_vector = self._tfidf(Counter(_tokens(query)))
            zero_query = not any(query_vector.values())
            scores = {
                key: math.fsum(query_vector.get(token, 0) * weight
                               for token, weight in vector.items())
                for key, vector in self._lexical.items()
            }
        ranked = []
        for identifier, score in scores.items():
            score = max(-1.0, min(1.0, score))
            if score > 0 and score >= min_score:
                ranked.append((identifier, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        results = []
        for identifier, score in ranked[:top_k]:
            text = self._documents[identifier]
            evidence, highlighted = _evidence(text, tokens)
            results.append({
                "id": identifier,
                "score": score,
                "text": text,
                "evidence": evidence,
                "highlighted_text": highlighted,
            })
        return {
            "mode": "embedding_cosine" if use_embedding else "lexical_tfidf",
            "interpretation": (
                "Cosine similarity of caller-supplied embeddings; quality is unverified."
                if use_embedding else
                "Lexical TF-IDF matching, not neural semantic understanding."
            ),
            "fallback_reason": (
                "No embedding callback supplied; using lexical retrieval."
                if mode == "auto" and not use_embedding else None
            ),
            "evidence_kind": "Exact source spans of Unicode-casefolded query token matches; "
                             "not an explanation of embedding similarity.",
            "reason": "zero_query_vector" if zero_query else (
                "matched" if results else "no_match_above_threshold"
            ),
            "results": results,
        }


def run_request(request):
    allowed = {"documents", "query", "top_k", "min_score", "mode", "embeddings"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("input must be an object with only supported fields")
    if not {"documents", "query"} <= set(request):
        raise ValueError("documents and query are required")
    callback = None
    if "embeddings" in request:
        embeddings = request["embeddings"]
        if not isinstance(embeddings, dict) or set(embeddings) != {"documents", "query"}:
            raise ValueError("embeddings must contain exactly documents and query")
        # Index construction and query are separate calls, even if a document ID
        # happens to equal the reserved query-batch ID.
        first_call = True

        def callback(batch):
            nonlocal first_call
            if first_call:
                first_call = False
                return embeddings["documents"]
            return {QUERY_ID: embeddings["query"]}

    index = TextIndex(request["documents"], callback)
    if "embeddings" in request:
        _validate_vectors(
            {QUERY_ID: request["embeddings"]["query"]}, [QUERY_ID], index._dimensions
        )
    return {
        "index": index.metadata,
        "search": index.search(
            request["query"],
            request.get("top_k", 5),
            request.get("min_score", 0.0),
            request.get("mode", "auto"),
        ),
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSON request file")
    args = parser.parse_args(argv)
    try:
        request = json.loads(
            args.input.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        response = run_request(request)
    except (ValueError, OSError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps(response, ensure_ascii=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
