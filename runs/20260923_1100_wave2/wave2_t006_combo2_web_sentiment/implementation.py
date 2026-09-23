"""Offline, synthetic web-research -> customer-insight reference pipeline.

Run: python -B implementation.py example_input.json
Only exact allowlisted HTTPS hosts (port 443 or omitted) are accepted.
Retrieval uses supplied fixtures, never network access. Each nonblank content
line becomes a verbatim finding. This is extraction, not verified truth.
"""

import copy
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


class Schema:
    """Shared validation for input, retrieval, findings and final output."""

    VERSION = "1.0"

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def keys(cls, value, expected, name):
        cls.require(isinstance(value, dict), name + " must be an object")
        cls.require(set(value) == set(expected), name + " has missing or unknown fields")

    @classmethod
    def text(cls, value, name, limit=20000):
        cls.require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty text")
        cls.require(len(value) <= limit, name + " exceeds size limit")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValidationError(name + " must be valid UTF-8 text") from exc

    @classmethod
    def hosts(cls, hosts):
        cls.require(isinstance(hosts, list) and 1 <= len(hosts) <= 20, "allowlist requires 1..20 hosts")
        for host in hosts:
            cls.text(host, "host", 253)
            cls.require(host == host.lower(), "hosts must be lowercase")
            labels = host.split(".")
            cls.require(len(labels) >= 2 and all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in labels
            ), "invalid exact hostname")
        cls.require(len(set(hosts)) == len(hosts), "duplicate allowlisted host")

    @classmethod
    def url(cls, value, hosts):
        cls.text(value, "URL", 2048)
        cls.require(not any(c.isspace() or ord(c) < 32 for c in value)
                    and "\\" not in value, "invalid URL characters")
        try:
            parsed = urlsplit(value)
            cls.require(parsed.scheme == "https" and parsed.hostname in hosts,
                        "URL must use HTTPS and an exact allowlisted host")
            cls.require(parsed.port in (None, 443), "URL port must be 443")
            cls.require(parsed.username is None and parsed.password is None,
                        "URL credentials forbidden")
            cls.require(not parsed.fragment, "URL fragments forbidden")
        except ValueError as exc:
            raise ValidationError("invalid or disallowed URL: " + str(exc)) from exc

    @classmethod
    def timestamp(cls, value):
        cls.text(value, "retrieved_at", 64)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("retrieved_at must be ISO 8601") from exc
        cls.require(parsed.tzinfo is not None, "retrieved_at needs a timezone")

    @classmethod
    def response(cls, response, hosts):
        cls.keys(response, ("final_url", "content", "retrieved_at"), "retrieval response")
        cls.url(response["final_url"], hosts)
        cls.text(response["content"], "content")
        cls.require(len(response["content"].splitlines()) <= 200, "too many content lines")
        cls.timestamp(response["retrieved_at"])

    @classmethod
    def input(cls, data):
        cls.keys(data, ("schema_version", "synthetic", "allowlisted_hosts", "urls",
                        "retrieval_fixtures"), "input")
        cls.require(data["schema_version"] == cls.VERSION, "unsupported schema_version")
        cls.require(data["synthetic"] is True, "input must be labeled synthetic")
        cls.hosts(data["allowlisted_hosts"])
        cls.require(isinstance(data["urls"], list) and 1 <= len(data["urls"]) <= 20,
                    "urls requires 1..20 entries")
        for url in data["urls"]:
            cls.url(url, data["allowlisted_hosts"])
        cls.require(len(set(data["urls"])) == len(data["urls"]), "duplicate URL")
        fixtures = data["retrieval_fixtures"]
        cls.require(isinstance(fixtures, dict) and set(fixtures) == set(data["urls"]),
                    "fixtures must match requested URLs exactly")
        for response in fixtures.values():
            cls.response(response, data["allowlisted_hosts"])
        return data

    @classmethod
    def research(cls, data):
        cls.keys(data, ("schema_version", "synthetic", "allowlisted_hosts", "sources",
                        "findings"), "research")
        cls.require(data["schema_version"] == cls.VERSION and data["synthetic"] is True,
                    "invalid research envelope")
        cls.hosts(data["allowlisted_hosts"])
        cls.require(isinstance(data["sources"], list) and 1 <= len(data["sources"]) <= 20,
                    "invalid sources")
        sources = {}
        expected_findings = []
        for index, source in enumerate(data["sources"], 1):
            cls.keys(source, ("source_id", "requested_url", "final_url", "content",
                              "retrieved_at", "content_sha256"), "source")
            cls.require(source["source_id"] == f"S{index:03d}", "invalid source ID")
            cls.url(source["requested_url"], data["allowlisted_hosts"])
            cls.response({key: source[key] for key in ("final_url", "content", "retrieved_at")},
                         data["allowlisted_hosts"])
            digest = hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
            cls.require(source["content_sha256"] == digest, "source digest mismatch")
            sources[source["source_id"]] = source
            for line_number, line in enumerate(source["content"].splitlines(), 1):
                if line.strip():
                    expected_findings.append({
                        "finding_id": f"F{len(expected_findings) + 1:04d}",
                        "source_id": source["source_id"],
                        "source_url": source["final_url"],
                        "retrieved_at": source["retrieved_at"],
                        "content_sha256": digest,
                        "line_number": line_number,
                        "text": line,
                    })
        cls.require(len({s["requested_url"] for s in sources.values()}) == len(sources),
                    "duplicate research URL")
        cls.require(data["findings"] == expected_findings, "findings do not match provenance")
        return data

    @classmethod
    def output(cls, data):
        cls.keys(data, ("status", "research", "insights", "priority_order", "method"), "output")
        cls.require(data["status"] == "ok", "invalid output status")
        cls.research(data["research"])
        expected = [score_finding(finding) for finding in data["research"]["findings"]]
        cls.require(data["insights"] == expected, "insight scores or provenance mismatch")
        cls.require(data["priority_order"] == prioritize(expected), "priority order mismatch")
        cls.require(data["method"] == METHOD, "invalid scoring method")
        return data


LEXICON = {
    "excellent": 2, "great": 2, "love": 2, "helpful": 1, "good": 1, "fast": 1,
    "bad": -1, "slow": -1, "broken": -2, "crash": -2, "crashes": -2,
    "terrible": -2, "hate": -2, "unsafe": -3, "lost": -2, "failed": -2,
}
SEVERITY_TERMS = {
    "critical": ["unsafe", "data loss", "security breach"],
    "high": ["broken", "crash", "crashes", "failed", "cannot access"],
    "medium": ["slow", "delay", "confusing"],
}
SEVERITY_WEIGHT = {"critical": 300, "high": 200, "medium": 100, "low": 0}
METHOD = {
    "name": "deterministic-lexicon-v1",
    "lexicon": LEXICON,
    "negation": "Flip a lexicon weight when its immediately preceding word is not, never or no.",
    "sentiment": "Sum adjusted weights; positive if >0, negative if <0, otherwise neutral.",
    "severity": "Highest matched severity term, regardless of sentiment or negation; otherwise low.",
    "severity_terms": SEVERITY_TERMS,
    "priority": "Severity base (critical 300, high 200, medium 100, low 0) plus min(99, max(0, -score)).",
    "ties": "Original finding order.",
    "limitations": "English lexical heuristic; no sarcasm, truth verification or contextual risk assessment.",
}


def research(data, retriever=None):
    """Validate input, retrieve offline fixtures, preserve line-level evidence."""
    Schema.input(data)
    sources, findings = [], []
    for index, url in enumerate(data["urls"], 1):
        Schema.url(url, data["allowlisted_hosts"])
        if retriever is None:
            response = data["retrieval_fixtures"][url]
        else:
            try:
                response = retriever(url)
            except Exception as exc:
                raise ValidationError("injected retrieval failed") from exc
        Schema.response(response, data["allowlisted_hosts"])
        source = {
            "source_id": f"S{index:03d}",
            "requested_url": url,
            **response,
            "content_sha256": hashlib.sha256(response["content"].encode("utf-8")).hexdigest(),
        }
        sources.append(source)
        for line_number, line in enumerate(source["content"].splitlines(), 1):
            if line.strip():
                findings.append({
                    "finding_id": f"F{len(findings) + 1:04d}",
                    "source_id": source["source_id"],
                    "source_url": source["final_url"],
                    "retrieved_at": source["retrieved_at"],
                    "content_sha256": source["content_sha256"],
                    "line_number": line_number,
                    "text": line,
                })
    return Schema.research({
        "schema_version": Schema.VERSION, "synthetic": True,
        "allowlisted_hosts": list(data["allowlisted_hosts"]),
        "sources": sources, "findings": findings,
    })


def score_finding(finding):
    text = finding["text"]
    tokens = list(re.finditer(r"\w+|[^\w\s]", text))
    evidence = []
    for index, token in enumerate(tokens):
        word = token.group().lower()
        if word in LEXICON:
            negated = index > 0 and tokens[index - 1].group().lower() in {"not", "never", "no"}
            evidence.append({
                "token": word, "offset": token.start(), "base_weight": LEXICON[word],
                "negated": negated, "weight": -LEXICON[word] if negated else LEXICON[word],
            })
    score = sum(item["weight"] for item in evidence)
    severity_evidence = []
    for severity, terms in SEVERITY_TERMS.items():
        for term in terms:
            for match in re.finditer(r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b",
                                     text, re.IGNORECASE):
                severity_evidence.append({"severity": severity, "term": term, "offset": match.start()})
    severity = max((e["severity"] for e in severity_evidence),
                   key=SEVERITY_WEIGHT.get, default="low")
    return {
        **finding,
        "sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral",
        "sentiment_score": score,
        "sentiment_evidence": evidence,
        "severity": severity,
        "severity_evidence": severity_evidence,
        "priority_score": SEVERITY_WEIGHT[severity] + min(99, max(0, -score)),
        "requires_attention": severity != "low" or score < 0,
    }


def prioritize(insights):
    return [item["finding_id"] for item in sorted(
        insights, key=lambda item: -item["priority_score"]
    ) if item["requires_attention"]]


def sentiment(research_output):
    """Consumes only a validated research envelope, never raw input text."""
    Schema.research(research_output)
    insights = [score_finding(finding) for finding in research_output["findings"]]
    return Schema.output({
        "status": "ok", "research": research_output, "insights": insights,
        "priority_order": prioritize(insights), "method": copy.deepcopy(METHOD),
    })


def run_pipeline(data, retriever=None):
    return sentiment(research(data, retriever))


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        Schema.require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        path = Path(argv[0])
        with path.open("rb") as stream:
            raw = stream.read(1_000_001)
        Schema.require(len(raw) <= 1_000_000, "input file exceeds 1 MB")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValidationError("non-finite JSON value: " + value)))
        result = run_pipeline(data)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
