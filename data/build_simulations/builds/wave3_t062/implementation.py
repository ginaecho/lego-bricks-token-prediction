"""Offline, synthetic web -> documents -> guided-setup reference pipeline.

Usage: python -B implementation.py example_input.json
Retrieval uses exact-URL fixture responses, never the network. Document fields
select source keys, convert types, apply text transforms, and enforce checks.
Guided steps consume validated document rows and retain their provenance.
"""

import hashlib
import ipaddress
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, label):
    require(isinstance(value, dict), label + " must be an object")
    return value


def array(value, label, nonempty=False):
    require(isinstance(value, list), label + " must be an array")
    require(not nonempty or bool(value), label + " must not be empty")
    return value


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def scalar(value, label):
    require(value is None or type(value) in (str, int, float, bool),
            label + " must be a JSON scalar")
    require(type(value) is not float or math.isfinite(value), label + " must be finite")
    return value


def unique_strings(value, label, nonempty=False):
    values = array(value, label, nonempty)
    for item in values:
        text(item, label + " item")
    require(len(set(values)) == len(values), label + " contains duplicates")
    return values


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValidationError("nonfinite JSON constant: " + value)

    try:
        return json.loads(body, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, RecursionError) as exc:
        raise ValidationError("invalid JSON: " + str(exc)) from exc


def validate_url(url, allowed_hosts):
    text(url, "URL")
    require(not any(c.isspace() or ord(c) < 32 for c in url) and "\\" not in url,
            "URL contains whitespace or invalid characters")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise ValidationError("invalid URL") from exc
    require(parsed.scheme == "https" and host is not None, "URL must use HTTPS")
    require(parsed.username is None and parsed.password is None, "URL credentials forbidden")
    require(not parsed.fragment and port in (None, 443), "URL fragment or port forbidden")
    require(host in allowed_hosts, "URL host is not allowlisted: " + host)
    return url


def validate_provenance(value):
    obj(value, "provenance")
    text(value.get("url"), "provenance.url")
    require(isinstance(value.get("sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "invalid provenance digest")
    require(type(value.get("record_index")) is int and value["record_index"] >= 0,
            "invalid provenance record index")
    require(value.get("retrieval") == "synthetic_fixture", "invalid retrieval provenance")


def validate_records(records, label):
    array(records, label, True)
    seen = set()
    for row in records:
        obj(row, label + " row")
        identifier = text(row.get("id"), label + ".id")
        require(identifier not in seen, label + " contains duplicate IDs")
        seen.add(identifier)
        data = obj(row.get("data"), label + ".data")
        require(bool(data), label + " data must not be empty")
        for key, value in data.items():
            text(key, label + " field name")
            scalar(value, label + "." + key)
        validate_provenance(row.get("provenance"))
    return records


def research(config):
    obj(config, "research")
    hosts = unique_strings(config.get("allowed_hosts"), "allowed_hosts", True)
    for host in hosts:
        require(host == host.lower() and len(host) <= 253 and
                re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) and "." in host,
                "allowlist requires lowercase DNS host names")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValidationError("IP literals are not allowed")
    urls = unique_strings(config.get("urls"), "urls", True)
    require(len(urls) <= 50, "at most 50 URLs are supported")
    fixtures = obj(config.get("fixtures"), "fixtures")
    for url in fixtures:
        validate_url(url, hosts)
    findings = []
    sources = []
    for source_index, url in enumerate(urls):
        validate_url(url, hosts)
        require(url in fixtures, "no synthetic fixture for URL: " + url)
        response = obj(fixtures[url], "fixture response")
        body = text(response.get("body"), "fixture body")
        require(len(body) <= 100000, "fixture body exceeds 100000 characters")
        media = response.get("content_type")
        require(media in ("application/json", "text/plain"), "unsupported content_type")
        if media == "application/json":
            records = array(strict_json(body), "retrieved JSON records", True)
        else:
            records = []
            record = {}
            for line in body.splitlines() + [""]:
                if not line.strip():
                    if record:
                        records.append(record)
                        record = {}
                    continue
                require("=" in line, "text records require key=value lines")
                key, value = line.split("=", 1)
                key = text(key.strip(), "text field")
                require(key not in record, "duplicate text field: " + key)
                record[key] = value.strip()
        require(bool(records), "retrieved records must not be empty")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        sources.append({"url": url, "sha256": digest, "content_type": media,
                        "record_count": len(records), "retrieval": "synthetic_fixture"})
        for index, data in enumerate(records):
            findings.append({"id": "source_%d_record_%d" % (source_index + 1, index + 1),
                             "data": data,
                             "provenance": {"url": url, "sha256": digest,
                                            "record_index": index,
                                            "retrieval": "synthetic_fixture"}})
        require(len(findings) <= 1000, "at most 1000 records are supported")
    return {"sources": sources, "findings": validate_records(findings, "findings")}


def validate_check(check, fields):
    obj(check, "check")
    field = text(check.get("field"), "check.field")
    require(field in fields, "check references unknown field: " + field)
    operation = check.get("op")
    require(operation in ("nonempty", "equals", "min", "max", "regex", "unique"),
            "unsupported check operation")
    if operation in ("equals", "min", "max", "regex"):
        require("value" in check, "check requires value")
        scalar(check["value"], "check.value")
    if operation in ("min", "max"):
        require(type(check["value"]) in (int, float), "numeric check requires number")
    if operation == "regex":
        text(check["value"], "regex")
        require(len(check["value"]) <= 200, "regex too long")
        try:
            re.compile(check["value"])
        except re.error as exc:
            raise ValidationError("invalid regex: " + str(exc)) from exc
    return check


def check_rows(check, rows):
    values = [row["data"][check["field"]] for row in rows]
    operation = check["op"]
    if operation == "unique":
        return len({(type(value), value) for value in values}) == len(values)
    for value in values:
        if operation == "nonempty":
            passed = value is not None and (not isinstance(value, str) or bool(value.strip()))
        elif operation == "equals":
            passed = type(value) is type(check["value"]) and value == check["value"]
        elif operation in ("min", "max"):
            passed = type(value) in (int, float) and (
                value >= check["value"] if operation == "min" else value <= check["value"])
        else:
            passed = isinstance(value, str) and re.fullmatch(check["value"], value) is not None
        if not passed:
            return False
    return True


def convert(value, spec):
    if value is None:
        require(not spec.get("required", True), "required field missing: " + spec["name"])
        return None
    kind = spec["type"]
    try:
        if kind == "string":
            require(type(value) is str, "string field requires text: " + spec["name"])
        elif kind == "integer":
            require(type(value) is int or
                    (type(value) is str and re.fullmatch(r"[+-]?\d+", value.strip())),
                    "invalid integer: " + spec["name"])
            value = int(value)
        elif kind == "number":
            require(type(value) in (int, float, str), "invalid number: " + spec["name"])
            value = float(value)
            require(math.isfinite(value), "number must be finite")
        else:
            if type(value) is str:
                require(value.strip().lower() in ("true", "false"), "invalid boolean")
                value = value.strip().lower() == "true"
            require(type(value) is bool, "invalid boolean")
    except (ValueError, OverflowError) as exc:
        raise ValidationError("conversion failed for " + spec["name"] + ": " + str(exc)) from exc
    for transform in spec.get("transforms", []):
        value = getattr(value, transform)()
    require(not spec.get("required", True) or not isinstance(value, str) or bool(value.strip()),
            "required field is empty: " + spec["name"])
    return value


def documents(config, research_output):
    obj(config, "documents")
    findings = validate_records(obj(research_output, "research output").get("findings"), "findings")
    specs = array(config.get("fields"), "document fields", True)
    names = []
    for spec in specs:
        obj(spec, "field specification")
        name = text(spec.get("name"), "field.name")
        require(name not in names, "duplicate document field: " + name)
        names.append(name)
        text(spec.get("source"), "field.source")
        require(spec.get("type") in ("string", "integer", "number", "boolean"),
                "unsupported field type")
        require(type(spec.get("required", True)) is bool, "required must be boolean")
        transforms = array(spec.get("transforms", []), "transforms")
        require(all(t in ("strip", "lower", "upper", "title") for t in transforms),
                "unsupported transform")
        require(not transforms or spec["type"] == "string", "transforms require string fields")
        if "default" in spec:
            scalar(spec["default"], "default")
            convert(spec["default"], spec)
    checks = [validate_check(c, names) for c in array(config.get("checks", []), "checks")]
    rows = []
    for finding in findings:
        data = {spec["name"]: convert(finding["data"].get(spec["source"], spec.get("default")), spec)
                for spec in specs}
        rows.append({"id": finding["id"], "data": data,
                     "provenance": dict(finding["provenance"])})
    validate_records(rows, "document rows")
    for check in checks:
        require(check_rows(check, rows), "document check failed: " + check["field"] + "/" + check["op"])
    return {"fields": names, "rows": rows, "checks_passed": len(checks)}


def guided(config, document_output):
    obj(config, "guided")
    obj(document_output, "document output")
    rows = validate_records(document_output.get("rows"), "document rows")
    fields = unique_strings(document_output.get("fields"), "document output fields", True)
    require(all(set(row["data"]) == set(fields) for row in rows), "document row schema mismatch")
    specs = array(config.get("steps"), "steps", True)
    by_id = {}
    for step in specs:
        obj(step, "step")
        identifier = text(step.get("id"), "step.id")
        require(identifier not in by_id, "duplicate step ID")
        text(step.get("title"), "step.title")
        unique_strings(step.get("requires", []), "step.requires")
        for check in array(step.get("checks", []), "step.checks"):
            validate_check(check, fields)
        by_id[identifier] = step
    for step in specs:
        require(all(dep in by_id and dep != step["id"] for dep in step.get("requires", [])),
                "unknown or self prerequisite")
    requested = unique_strings(config.get("completed_steps", []), "completed_steps")
    require(set(requested) <= set(by_id), "unknown completed step")
    pending = list(by_id)
    statuses = {}
    while pending:
        candidates = [identifier for identifier in pending
                      if all(dep in statuses for dep in by_id[identifier].get("requires", []))]
        require(bool(candidates), "prerequisite cycle")
        for identifier in candidates:
            step = by_id[identifier]
            blockers = ["prerequisite:" + dep for dep in step.get("requires", [])
                        if statuses[dep]["status"] != "completed"]
            blockers += ["data:" + check["field"] + "/" + check["op"]
                         for check in step.get("checks", []) if not check_rows(check, rows)]
            require(identifier not in requested or not blockers,
                    "cannot complete blocked step: " + identifier)
            statuses[identifier] = {
                "id": identifier, "title": step["title"],
                "status": "blocked" if blockers else ("completed" if identifier in requested else "ready"),
                "blockers": blockers,
                "evidence": [{"record_id": row["id"], "provenance": dict(row["provenance"])}
                             for row in rows],
            }
            pending.remove(identifier)
    done = sum(step["status"] == "completed" for step in statuses.values())
    return {"steps": [statuses[step["id"]] for step in specs],
            "progress": {"completed": done, "total": len(specs),
                         "fraction": done / len(specs)}}


def run_pipeline(payload):
    obj(payload, "input")
    require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1,
            "schema_version must be 1")
    require(payload.get("synthetic") is True, "synthetic must be true")
    researched = research(payload.get("research"))
    reshaped = documents(payload.get("documents"), researched)
    onboarding = guided(payload.get("guided"), reshaped)
    return {"schema_version": 1, "synthetic": True, "status": "ok",
            "research": researched, "documents": reshaped, "guided": onboarding}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        payload = strict_json(Path(args[0]).read_text(encoding="utf-8"))
        output = run_pipeline(payload)
    except (ValidationError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
