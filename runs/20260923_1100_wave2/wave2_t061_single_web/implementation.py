"""Offline, deterministic web-research reference. No network access is implemented."""

import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, expected, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == set(expected), f"{label} keys must be {sorted(expected)}")


def text(value, label, limit=100000):
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
    require(len(value) <= limit, f"{label} exceeds {limit} characters")
    return value


def canonical_url(value):
    text(value, "URL", 2048)
    require(not re.search(r"[\s\\\x00-\x1f\x7f]", value), "URL contains forbidden characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise ValidationError("Malformed URL") from exc
    require(parsed.scheme == "https", "Only HTTPS URLs are accepted")
    require(host is not None and re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host),
            "URL must have an ASCII hostname")
    require(parsed.username is None and parsed.password is None, "URL credentials are forbidden")
    require(port in (None, 443), "Only the default HTTPS port is allowed")
    require(not parsed.netloc.endswith(":"), "Empty URL port is forbidden")
    return urlunsplit(("https", host, parsed.path or "/", parsed.query, ""))


def validate_input(data):
    fields(data, ("schema_version", "query", "allowlisted_hosts", "urls", "fixtures"), "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    text(data["query"], "query", 1000)
    hosts = data["allowlisted_hosts"]
    require(isinstance(hosts, list) and 1 <= len(hosts) <= 100, "allowlisted_hosts must contain 1-100 hosts")
    for host in hosts:
        text(host, "allowlisted host", 253)
        require(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) is not None,
                "Allowlist entries must be lowercase ASCII hostnames, without wildcards")
    require(len(set(hosts)) == len(hosts), "Duplicate allowlisted hosts")
    urls = data["urls"]
    require(isinstance(urls, list) and 1 <= len(urls) <= 100, "urls must contain 1-100 URLs")
    canonical = []
    for raw in urls:
        url = canonical_url(raw)
        require(urlsplit(url).hostname in hosts, "Requested URL host is not allowlisted")
        if url not in canonical:
            canonical.append(url)
    fixtures = data["fixtures"]
    require(isinstance(fixtures, list) and len(fixtures) <= 100, "fixtures must be a list of at most 100")
    index = {}
    for fixture in fixtures:
        fields(fixture, ("url", "title", "content_type", "body", "synthetic"), "fixture")
        require(fixture["synthetic"] is True, "All fixture data must be labeled synthetic")
        url = canonical_url(fixture["url"])
        require(urlsplit(url).hostname in hosts, "Fixture host is not allowlisted")
        require(url not in index, "Duplicate canonical fixture URL")
        text(fixture["title"], "fixture title", 500)
        require(fixture["content_type"] in ("text/plain", "text/html"), "Unsupported content_type")
        require(isinstance(fixture["body"], str) and len(fixture["body"]) <= 100000,
                "Fixture body must be text of at most 100000 characters")
        index[url] = fixture
    return canonical, index


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "template"):
            self.hidden.append(tag)
        if not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        if not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def visible_text(fixture):
    body = fixture["body"]
    if fixture["content_type"] == "text/html":
        parser = VisibleText()
        parser.feed(body)
        parser.close()
        body = "".join(parser.parts)
    return " ".join(body.split())


def validate_output(result):
    fields(result, ("schema_version", "status", "query", "synthetic", "sources", "findings", "warnings"),
           "output")
    require(result["status"] == "ok" and result["schema_version"] == 1 and result["synthetic"] is True,
            "Invalid output envelope")
    for finding in result["findings"]:
        source = result["sources"][finding["source_index"]]
        start, end = finding["start"], finding["end"]
        require(0 <= start < end <= len(source["text"]), "Invalid evidence offsets")
        require(source["text"][start:end] == finding["quote"], "Evidence does not match source")
        require(finding["url"] == source["url"] and finding["sha256"] == source["sha256"],
                "Evidence provenance mismatch")
    return result


def research(data):
    urls, fixtures = validate_input(data)
    result = {"schema_version": 1, "status": "ok", "query": data["query"],
              "synthetic": True, "sources": [], "findings": [], "warnings": []}
    terms = set(re.findall(r"\w+", data["query"].casefold()))
    require(bool(terms), "query must contain a word")
    for url in urls:
        fixture = fixtures.get(url)
        if fixture is None:
            result["warnings"].append({"url": url, "code": "fixture_missing"})
            continue
        content = visible_text(fixture)
        source = {"url": url, "title": fixture["title"], "text": content,
                  "content_type": fixture["content_type"], "retrieval": "synthetic_fixture",
                  "sha256": hashlib.sha256(fixture["body"].encode("utf-8")).hexdigest()}
        index = len(result["sources"])
        result["sources"].append(source)
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", content):
            quote = match.group().strip()
            if not quote:
                continue
            matched = sorted(terms & set(re.findall(r"\w+", quote.casefold())))
            if matched:
                start = match.start() + len(match.group()) - len(match.group().lstrip())
                result["findings"].append({
                    "source_index": index, "url": url, "title": fixture["title"],
                    "sha256": source["sha256"], "quote": quote, "start": start,
                    "end": start + len(quote), "matched_terms": matched,
                })
    return validate_output(result)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py input.json")
        raw = Path(argv[0]).read_bytes()
        require(len(raw) <= 12000000, "Input exceeds 12 MB")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        result = research(data)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
