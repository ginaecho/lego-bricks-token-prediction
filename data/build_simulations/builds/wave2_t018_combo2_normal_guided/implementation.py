"""Synthetic, deterministic research -> guided onboarding reference pipeline.

Run: python -B implementation.py example_input.json
All inputs use schema_version 1. Citations use zero-based, end-exclusive
Python Unicode character offsets into the supplied source passage.
No network access, external dependencies, or model provider is used.
"""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    """An input or cross-stage contract is invalid."""


class Schema:
    """Shared validation primitives for inputs, evidence, and progress."""

    @staticmethod
    def fail(path, message):
        raise ValidationError(f"{path}: {message}")

    @classmethod
    def obj(cls, value, path, keys=None):
        if not isinstance(value, dict):
            cls.fail(path, "expected an object")
        if keys is not None and set(value) != set(keys):
            cls.fail(path, f"expected exactly keys {sorted(keys)}")
        return value

    @classmethod
    def array(cls, value, path, minimum=0, maximum=100):
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            cls.fail(path, f"expected an array with {minimum}..{maximum} items")
        return value

    @classmethod
    def text(cls, value, path, maximum=5000):
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            cls.fail(path, f"expected nonblank text of at most {maximum} characters")
        # Lone surrogate code points are not valid Unicode source text.
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            cls.fail(path, "text contains a lone Unicode surrogate")
        return value

    @classmethod
    def integer(cls, value, path, minimum, maximum):
        if type(value) is not int or not minimum <= value <= maximum:
            cls.fail(path, f"expected integer {minimum}..{maximum}")
        return value

    @classmethod
    def strings(cls, value, path, minimum=0, maximum=100, fold=False):
        cls.array(value, path, minimum, maximum)
        for item in value:
            cls.text(item, path, 200)
        comparable = [item.casefold() if fold else item for item in value]
        if len(set(comparable)) != len(value):
            cls.fail(path, "duplicate values")
        return value


def validate_input(data):
    Schema.obj(data, "input", {"schema_version", "fixture_label", "research", "guided"})
    Schema.integer(data["schema_version"], "schema_version", 1, 1)
    Schema.text(data["fixture_label"], "fixture_label", 200)
    if not data["fixture_label"].startswith("SYNTHETIC:"):
        Schema.fail("fixture_label", "must start with SYNTHETIC:")
    research = Schema.obj(data["research"], "research", {"keywords", "max_findings", "sources"})
    Schema.strings(research["keywords"], "research.keywords", 1, 30, fold=True)
    Schema.integer(research["max_findings"], "research.max_findings", 1, 100)
    sources = Schema.array(research["sources"], "research.sources", 0, 30)
    source_ids = set()
    for source in sources:
        Schema.obj(source, "source", {"id", "title", "passages"})
        Schema.text(source["id"], "source.id", 200)
        Schema.text(source["title"], "source.title", 200)
        if source["id"] in source_ids:
            Schema.fail("source.id", "duplicate source identifier")
        source_ids.add(source["id"])
        passage_ids = set()
        for passage in Schema.array(source["passages"], "source.passages", 1, 100):
            Schema.obj(passage, "passage", {"id", "text"})
            Schema.text(passage["id"], "passage.id", 200)
            Schema.text(passage["text"], "passage.text")
            if passage["id"] in passage_ids:
                Schema.fail("passage.id", "duplicate identifier within a source")
            passage_ids.add(passage["id"])

    guided = Schema.obj(data["guided"], "guided", {"steps", "progress"})
    steps = Schema.array(guided["steps"], "guided.steps", 1, 100)
    by_id = {}
    for step in steps:
        Schema.obj(step, "step", {"id", "title", "prerequisites", "evidence_keywords", "required_fields"})
        Schema.text(step["id"], "step.id", 200)
        Schema.text(step["title"], "step.title", 200)
        if step["id"] in by_id:
            Schema.fail("step.id", "duplicate identifier")
        by_id[step["id"]] = step
        Schema.strings(step["prerequisites"], "step.prerequisites")
        Schema.strings(step["evidence_keywords"], "step.evidence_keywords", 1, 30, fold=True)
        Schema.strings(step["required_fields"], "step.required_fields", 0, 30)
    for step in steps:
        if any(prerequisite not in by_id for prerequisite in step["prerequisites"]):
            Schema.fail("step.prerequisites", "unknown prerequisite")
    visiting, visited = set(), set()

    def visit(step_id):
        if step_id in visiting:
            Schema.fail("step.prerequisites", "dependency cycle")
        if step_id in visited:
            return
        visiting.add(step_id)
        for prerequisite in by_id[step_id]["prerequisites"]:
            visit(prerequisite)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in by_id:
        visit(step_id)
    progress = Schema.obj(guided["progress"], "guided.progress", {"completed_step_ids", "responses"})
    completed = Schema.strings(progress["completed_step_ids"], "progress.completed_step_ids")
    if not set(completed) <= by_id.keys():
        Schema.fail("progress.completed_step_ids", "unknown step")
    responses = Schema.obj(progress["responses"], "progress.responses")
    for step_id, response in responses.items():
        if step_id not in by_id:
            Schema.fail("progress.responses", "unknown step")
        Schema.obj(response, f"responses.{step_id}")
        if not response.keys() <= set(by_id[step_id]["required_fields"]):
            Schema.fail(f"responses.{step_id}", "unknown response field")
        for field, value in response.items():
            Schema.text(value, f"responses.{step_id}.{field}", 1000)
    return data


def matches(keyword, text):
    """Literal, case-insensitive phrase matching at Unicode word boundaries."""
    return re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", text, re.IGNORECASE) is not None


def _extract_findings(data):
    candidates = []
    for source in data["research"]["sources"]:
        for passage in source["passages"]:
            matched = [
                keyword for keyword in data["research"]["keywords"]
                if matches(keyword, passage["text"])
            ]
            if matched:
                candidates.append({
                    "quote": passage["text"],
                    "matched_keywords": matched,
                    "score": len(matched),
                    "citation": {
                        "source_id": source["id"],
                        "source_title": source["title"],
                        "passage_id": passage["id"],
                        "start": 0,
                        "end": len(passage["text"]),
                    },
                })
    candidates.sort(key=lambda item: (
        -item["score"], item["citation"]["source_id"], item["citation"]["passage_id"]
    ))
    return [
        {"id": f"finding-{index}", **item}
        for index, item in enumerate(candidates[:data["research"]["max_findings"]], 1)
    ]


def validate_research(data, result):
    """Reject fabricated, stale, incomplete, or reordered cross-stage evidence."""
    Schema.obj(result, "research_result", {"findings", "retrieved_count"})
    findings = Schema.array(result["findings"], "research_result.findings")
    Schema.integer(result["retrieved_count"], "research_result.retrieved_count", 0, 100)
    lookup = {
        (source["id"], passage["id"]): (source["title"], passage["text"])
        for source in data["research"]["sources"] for passage in source["passages"]
    }
    for finding in findings:
        Schema.obj(finding, "finding", {"id", "quote", "matched_keywords", "score", "citation"})
        Schema.text(finding["id"], "finding.id", 200)
        Schema.text(finding["quote"], "finding.quote")
        Schema.strings(finding["matched_keywords"], "finding.matched_keywords", 1, 30, fold=True)
        Schema.integer(finding["score"], "finding.score", 1, 30)
        citation = Schema.obj(
            finding["citation"], "citation",
            {"source_id", "source_title", "passage_id", "start", "end"},
        )
        for key in ("source_id", "source_title", "passage_id"):
            Schema.text(citation[key], f"citation.{key}", 200)
        key = citation["source_id"], citation["passage_id"]
        if key not in lookup:
            Schema.fail("citation", "unknown source passage")
        title, text = lookup[key]
        Schema.integer(citation["start"], "citation.start", 0, len(text))
        Schema.integer(citation["end"], "citation.end", citation["start"] + 1, len(text))
        if citation["source_title"] != title or text[citation["start"]:citation["end"]] != finding["quote"]:
            Schema.fail("citation", "quote or title does not exactly match source")
    expected = _extract_findings(data)
    if findings != expected or result["retrieved_count"] != len(expected):
        Schema.fail("research_result", "does not match deterministic retrieval contract")
    return result


def run_research(data):
    validate_input(data)
    findings = _extract_findings(data)
    return validate_research(data, {"findings": findings, "retrieved_count": len(findings)})


def run_guided(data, research_result):
    validate_input(data)
    validate_research(data, research_result)
    progress = data["guided"]["progress"]
    completed = set(progress["completed_step_ids"])
    results = []
    for step in data["guided"]["steps"]:
        evidence_ids = []
        covered = set()
        for finding in research_result["findings"]:
            supported = {
                keyword for keyword in step["evidence_keywords"]
                if matches(keyword, finding["quote"])
            }
            if supported:
                evidence_ids.append(finding["id"])
                covered.update(supported)
        missing_evidence = [
            keyword for keyword in step["evidence_keywords"] if keyword not in covered
        ]
        unmet_prerequisites = [
            prerequisite for prerequisite in step["prerequisites"] if prerequisite not in completed
        ]
        responses = progress["responses"].get(step["id"], {})
        missing_fields = [field for field in step["required_fields"] if field not in responses]
        if step["id"] in completed:
            if missing_evidence or unmet_prerequisites or missing_fields:
                Schema.fail(f"progress.{step['id']}", "completed step lacks evidence, prerequisites, or responses")
            status = "completed"
        elif missing_evidence or unmet_prerequisites:
            status = "blocked"
        elif missing_fields:
            status = "awaiting_input"
        else:
            status = "ready"
        results.append({
            "id": step["id"],
            "title": step["title"],
            "status": status,
            "evidence_finding_ids": evidence_ids,
            "missing_evidence_keywords": missing_evidence,
            "unmet_prerequisites": unmet_prerequisites,
            "missing_fields": missing_fields,
            "responses": dict(responses),
        })
    total = len(results)
    done = sum(step["status"] == "completed" for step in results)
    return {
        "steps": results,
        "progress": {
            "completed_step_ids": [step["id"] for step in results if step["status"] == "completed"],
            "completed_count": done,
            "total_steps": total,
            "completion_percent": round(100 * done / total, 2),
        },
        "next_step_ids": [step["id"] for step in results if step["status"] in ("ready", "awaiting_input")],
    }


def run_pipeline(data):
    validate_input(data)
    research = run_research(data)
    guided = run_guided(data, research)
    return {
        "schema_version": 1,
        "fixture_label": data["fixture_label"],
        "status": "ok",
        "research": research,
        "guided": guided,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            Schema.fail("json", f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    Schema.fail("json", f"non-finite number {value} is not allowed")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            Schema.fail("arguments", "usage: python -B implementation.py INPUT.json")
        text = Path(args[0]).read_text(encoding="utf-8")
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        output = run_pipeline(data)
        code = 0
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as error:
        output = {"schema_version": 1, "status": "error", "error": str(error)}
        code = 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
