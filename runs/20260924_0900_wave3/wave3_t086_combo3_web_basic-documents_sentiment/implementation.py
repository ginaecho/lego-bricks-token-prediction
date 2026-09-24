"""Offline, synthetic web -> document -> sentiment reference pipeline.

Run: python -B implementation.py example_input.json
Retrieval is deliberately fixture-backed: no network or model calls are made.
"""

import csv
import hashlib
import io
import json
import re
import sys
from urllib.parse import urlsplit


VERSION = "1.0"
LEVELS = {"low": 1, "medium": 2, "high": 3, "critical": 4}
LEXICON = {
    "good": 1, "great": 2, "excellent": 2, "helpful": 1,
    "love": 2, "happy": 1, "thanks": 1, "resolved": 1,
    "bad": -1, "broken": -2, "terrible": -2, "hate": -2,
    "slow": -1, "failed": -2, "unhappy": -1, "frustrating": -1,
}
ESCALATIONS = {"data loss": "critical", "security breach": "critical",
               "injury": "critical", "outage": "high"}
DEFAULT_MAPPING = {"issue_id": "issue_id", "text": "text",
                   "customer": "customer", "severity": "severity"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, name, limit=100000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            name + " must be a nonempty bounded string")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValidationError("non-finite JSON constant: " + value)))


def check_url(url, hosts):
    text(url, "URL", 2048)
    try:
        parsed = urlsplit(url)
        port = parsed.port
        require(parsed.scheme == "https" and parsed.hostname in hosts,
                "URL must use HTTPS and an exactly allowlisted host")
        require(parsed.username is None and parsed.password is None
                and port in (None, 443) and not parsed.fragment,
                "URL credentials, non-HTTPS ports and fragments are forbidden")
        require(not re.search(r"[\s\\]", url), "URL contains unsafe characters")
    except ValueError as exc:
        raise ValidationError("invalid URL: " + str(exc)) from exc


def digest(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate(stage, value):
    """One validation boundary shared by all stages and their handoffs."""
    require(isinstance(value, dict), stage + " must be an object")
    require(value.get("schema_version") == VERSION, "unsupported schema_version")
    require(value.get("synthetic") is True, "synthetic must be true")
    if stage == "input":
        hosts = value.get("allowed_hosts")
        require(isinstance(hosts, list) and 0 < len(hosts) <= 50,
                "allowed_hosts must be a nonempty bounded list")
        for host in hosts:
            text(host, "host", 253)
            require(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) is not None,
                    "hosts must be lowercase hostnames without ports or wildcards")
        urls = value.get("urls")
        require(isinstance(urls, list) and 0 < len(urls) <= 50,
                "urls must contain 1..50 entries")
        for url in urls:
            check_url(url, hosts)
        require(len(urls) == len(set(urls)), "duplicate URLs")
        pages = value.get("fixture_pages")
        require(isinstance(pages, dict) and set(pages) == set(urls),
                "fixture_pages must match requested URLs exactly")
        for page in pages.values():
            require(isinstance(page, dict), "fixture page must be an object")
            text(page.get("title"), "page title", 500)
            text(page.get("body"), "page body", 1000000)
            require(page.get("media_type") in ("application/json", "text/csv"),
                    "unsupported fixture media_type")
        rules = value.get("document_rules", {})
        require(isinstance(rules, dict) and set(rules) <= {"field_mapping"},
                "unknown document rule")
        mapping = rules.get("field_mapping", DEFAULT_MAPPING)
        require(isinstance(mapping, dict)
                and {"issue_id", "text"} <= set(mapping)
                and set(mapping) <= set(DEFAULT_MAPPING), "invalid field_mapping")
        for source in mapping.values():
            text(source, "mapped source field", 100)
        require(len(set(mapping.values())) == len(mapping), "ambiguous field mapping")
    elif stage == "research":
        findings = value.get("findings")
        require(isinstance(findings, list) and 0 < len(findings) <= 50,
                "invalid findings")
        hosts = value.get("allowed_hosts")
        require(isinstance(hosts, list) and all(isinstance(h, str) for h in hosts),
                "missing research allowlist")
        ids = set()
        for finding in findings:
            require(isinstance(finding, dict), "finding must be an object")
            sid = text(finding.get("source_id"), "source_id")
            require(sid not in ids, "duplicate source_id")
            ids.add(sid)
            check_url(finding.get("url"), hosts)
            text(finding.get("title"), "title")
            body = text(finding.get("body"), "body", 1000000)
            require(finding.get("sha256") == digest(body), "source hash mismatch")
            require(finding.get("media_type") in ("application/json", "text/csv"),
                    "invalid finding media_type")
    elif stage in ("documents", "sentiment"):
        validate("research", value.get("research"))
        records = value.get("records")
        require(isinstance(records, list) and len(records) <= 10000,
                "records must be a bounded list")
        sources = {f["source_id"]: f for f in value["research"]["findings"]}
        ids = set()
        for record in records:
            require(isinstance(record, dict), "record must be an object")
            rid = text(record.get("issue_id"), "issue_id", 200)
            require(rid not in ids, "duplicate issue_id: " + rid)
            ids.add(rid)
            text(record.get("text"), "record text")
            require(isinstance(record.get("customer"), str), "invalid customer")
            require(record.get("severity") in LEVELS, "invalid severity")
            provenance = record.get("provenance")
            require(isinstance(provenance, dict), "missing provenance")
            source = sources.get(provenance.get("source_id"))
            require(source is not None, "unknown provenance source")
            require(provenance.get("url") == source["url"]
                    and provenance.get("sha256") == source["sha256"],
                    "provenance mismatch")
            require(type(provenance.get("row")) is int and provenance["row"] > 0,
                    "invalid provenance row")
        if stage == "sentiment":
            issues = value.get("issues")
            require(isinstance(issues, list) and len(issues) == len(records),
                    "issues must correspond to all documents")
            require(all(isinstance(i, dict) for i in issues), "invalid issue")
            require({i.get("issue_id") for i in issues} == ids,
                    "sentiment issue IDs mismatch")
            by_id = {r["issue_id"]: r for r in records}
            for issue in issues:
                record = by_id[issue["issue_id"]]
                require(all(issue.get(k) == v for k, v in record.items()),
                        "sentiment changed source record")
                expected = score(record["text"], record["severity"])
                require(all(issue.get(k) == v for k, v in expected.items()),
                        "invalid sentiment calculation")
            require(issues == sorted(issues, key=priority_key), "issues not prioritized")
    else:
        raise ValidationError("unknown validation stage")
    return value


def envelope(**values):
    return {"schema_version": VERSION, "synthetic": True, **values}


def research(data):
    validate("input", data)
    findings = []
    for index, url in enumerate(data["urls"], 1):
        page = data["fixture_pages"][url]
        findings.append({"source_id": "source-" + str(index), "url": url,
                         "title": page["title"], "body": page["body"],
                         "media_type": page["media_type"],
                         "sha256": digest(page["body"])})
    return validate("research", envelope(allowed_hosts=data["allowed_hosts"],
                                         findings=findings))


def extract(finding):
    if finding["media_type"] == "application/json":
        rows = load_json(finding["body"])
        require(isinstance(rows, list), "JSON document must contain an array")
    else:
        reader = csv.reader(io.StringIO(finding["body"], newline=""), strict=True)
        header = next(reader, None)
        require(header and all(h.strip() for h in header)
                and len(header) == len(set(header)), "invalid CSV header")
        rows = []
        for values in reader:
            if not values:
                continue
            require(len(values) == len(header), "CSV row width mismatch")
            rows.append(dict(zip(header, values)))
    require(len(rows) <= 10000 and all(isinstance(r, dict) for r in rows),
            "document rows must be a bounded list of objects")
    return rows


def documents(found, rules=None):
    validate("research", found)
    rules = {} if rules is None else rules
    require(isinstance(rules, dict) and set(rules) <= {"field_mapping"},
            "invalid document rules")
    mapping = rules.get("field_mapping", DEFAULT_MAPPING)
    require(isinstance(mapping, dict) and {"issue_id", "text"} <= set(mapping)
            and set(mapping) <= set(DEFAULT_MAPPING), "invalid field_mapping")
    for field in mapping.values():
        text(field, "mapped field")
    require(len(set(mapping.values())) == len(mapping), "ambiguous field mapping")
    records = []
    for finding in found["findings"]:
        for row_number, row in enumerate(extract(finding), 1):
            record = {}
            for target, source in mapping.items():
                item = row.get(source, "low" if target == "severity" else "")
                require(isinstance(item, str), "document fields must be strings")
                record[target] = item.strip()
            record.setdefault("customer", "")
            record.setdefault("severity", "low")
            record["severity"] = record["severity"].lower()
            record["provenance"] = {
                "source_id": finding["source_id"], "url": finding["url"],
                "sha256": finding["sha256"], "row": row_number}
            records.append(record)
    return validate("documents", envelope(research=found, records=records))


def score(content, severity):
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?;,]", content.lower())
    evidence = []
    recent = []
    for index, token in enumerate(tokens):
        if token in ".!?;,":
            recent = []
            continue
        if token in LEXICON:
            negated = sum(t in {"not", "never", "no", "don't", "isn't", "wasn't"}
                          for t in recent[-3:]) % 2 == 1
            contribution = LEXICON[token] * (-1 if negated else 1)
            evidence.append({"token": token, "token_index": index,
                             "negated": negated, "contribution": contribution})
        recent.append(token)
    raw = sum(e["contribution"] for e in evidence)
    denominator = sum(abs(e["contribution"]) for e in evidence)
    triggers = [phrase for phrase in ESCALATIONS
                if re.search(r"\b" + re.escape(phrase) + r"\b", content.lower())]
    effective = max([severity] + [ESCALATIONS[p] for p in triggers],
                    key=LEVELS.get)
    return {"sentiment": "positive" if raw > 0 else "negative" if raw < 0 else "neutral",
            "raw_score": raw, "normalized_score": round(raw / denominator, 4)
            if denominator else 0.0, "evidence": evidence,
            "effective_severity": effective, "severity_triggers": triggers,
            "priority": "P" + str(4 - LEVELS[effective]),
            "priority_reason": "Severity first; ascending sentiment score then issue_id."}


def priority_key(issue):
    return (-LEVELS[issue["effective_severity"]], issue["raw_score"], issue["issue_id"])


def sentiment(document):
    validate("documents", document)
    issues = [{**r, **score(r["text"], r["severity"])} for r in document["records"]]
    issues.sort(key=priority_key)
    return validate("sentiment", envelope(research=document["research"],
                                          records=document["records"], issues=issues))


def run(data):
    found = research(data)
    reshaped = documents(found, data.get("document_rules"))
    result = sentiment(reshaped)
    return {"schema_version": VERSION, "status": "ok", "synthetic": True,
            "research": found, "documents": reshaped, "sentiment": result}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: implementation.py INPUT.json")
        with open(argv[0], "r", encoding="utf-8") as handle:
            raw = handle.read(2000001)
        require(len(raw) <= 2000000, "input exceeds 2,000,000 characters")
        result = run(load_json(raw))
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError,
            csv.Error, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
