"""Offline web-research reference. All retrieval uses explicitly synthetic fixtures.

Run: python -B implementation.py example_input.json
No sockets, providers, runtime dependencies, or arbitrary file retrieval are used.
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ValidationError(ValueError):
    """A rejected request, retrieval, or output contract."""


class Validator:
    """Shared validation boundary for ingestion, retrieval, and findings."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValidationError(message)

    @classmethod
    def text(cls, value, name, maximum=100_000):
        cls.require(isinstance(value, str) and 0 < len(value) <= maximum,
                    f"{name} must be a nonempty string of at most {maximum} characters")
        return value

    @classmethod
    def fields(cls, value, required, optional=()):
        cls.require(isinstance(value, dict), "Expected a JSON object")
        cls.require(set(required) <= value.keys(), "Missing required fields")
        cls.require(value.keys() <= set(required) | set(optional), "Unknown fields")

    @classmethod
    def host(cls, value):
        cls.text(value, "allowlisted host", 253)
        host = value.lower()
        labels = host.split(".")
        cls.require(len(labels) >= 2 and all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in labels), "Hosts must be exact ASCII DNS names, not wildcards")
        cls.require(not all(label.isdigit() for label in labels),
                    "IP addresses are not allowed")
        return host

    @classmethod
    def url(cls, value, hosts):
        cls.text(value, "URL", 2048)
        cls.require(not any(c.isspace() or ord(c) < 32 or ord(c) == 127
                            for c in value) and "\\" not in value,
                    "URL contains whitespace, controls, or backslashes")
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError as exc:
            raise ValidationError("Malformed URL") from exc
        cls.require(parts.scheme == "https" and parts.hostname is not None,
                    "Only absolute HTTPS URLs are allowed")
        cls.require(parts.username is None and parts.password is None,
                    "URL credentials are forbidden")
        cls.require(port in (None, 443), "Only HTTPS port 443 is allowed")
        cls.require(not parts.fragment and not parts.netloc.endswith(":"),
                    "Fragments and empty ports are forbidden")
        host = cls.host(parts.hostname)
        cls.require(host in hosts, f"URL host is not allowlisted: {host}")
        return urlunsplit(("https", host, parts.path or "/", parts.query, ""))

    @classmethod
    def timestamp(cls, value):
        cls.text(value, "retrieved_at", 64)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("retrieved_at must be ISO 8601") from exc
        cls.require("T" in value and parsed.utcoffset() is not None,
                    "retrieved_at must include time and timezone")

    @classmethod
    def request(cls, value):
        cls.fields(value, ("schema_version", "synthetic", "query", "allowed_hosts",
                           "urls", "fixtures"), ("max_findings",))
        cls.require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                    "schema_version must be 1")
        cls.require(value["synthetic"] is True, "synthetic must be true")
        cls.text(value["query"], "query", 500)
        cls.require(bool(re.findall(r"\w+", value["query"])), "query needs a word")
        for key, maximum in (("allowed_hosts", 20), ("urls", 20), ("fixtures", 50)):
            cls.require(isinstance(value[key], list) and 1 <= len(value[key]) <= maximum,
                        f"{key} must contain 1 to {maximum} entries")
        hosts = {cls.host(host) for host in value["allowed_hosts"]}
        urls = list(dict.fromkeys(cls.url(url, hosts) for url in value["urls"]))
        limit = value.get("max_findings", 10)
        cls.require(type(limit) is int and 1 <= limit <= 100,
                    "max_findings must be an integer from 1 to 100")
        fixtures = {}
        for fixture in value["fixtures"]:
            cls.fields(fixture, ("url", "retrieved_at"),
                       ("body", "content_type", "redirect_to"))
            url = cls.url(fixture["url"], hosts)
            cls.require(url not in fixtures, "Duplicate canonical fixture URL")
            cls.timestamp(fixture["retrieved_at"])
            if "redirect_to" in fixture:
                cls.require("body" not in fixture and "content_type" not in fixture,
                            "Redirect fixtures cannot also contain a response body")
                normalized = dict(fixture, redirect_to=cls.url(fixture["redirect_to"], hosts))
            else:
                cls.require("body" in fixture and "content_type" in fixture,
                            "Response fixtures require body and content_type")
                cls.require(isinstance(fixture["body"], str)
                            and len(fixture["body"]) <= 100_000,
                            "body must be a string of at most 100000 characters")
                cls.require(fixture["content_type"] in ("text/plain", "text/html"),
                            "Unsupported content_type")
                normalized = dict(fixture)
            fixtures[url] = normalized
        return hosts, urls, fixtures, limit

    @classmethod
    def response(cls, result):
        cls.fields(result, ("schema_version", "synthetic", "status", "query",
                            "sources", "findings", "warnings"))
        sources = {source["source_id"]: source for source in result["sources"]}
        cls.require(len(sources) == len(result["sources"]), "Duplicate source IDs")
        for finding in result["findings"]:
            cls.require(finding["source_id"] in sources, "Unresolved citation")
            source = sources[finding["source_id"]]
            cls.require(source["text"][finding["start"]:finding["end"]] == finding["quote"],
                        "Finding offsets do not match source text")
            cls.require(finding["url"] == source["final_url"]
                        and finding["content_sha256"] == source["content_sha256"]
                        and finding["retrieved_at"] == source["retrieved_at"],
                        "Finding provenance differs from source")
        return result


class TextExtractor(HTMLParser):
    BLOCKS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section"}
    HIDDEN = {"script", "style", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self.HIDDEN:
            self.hidden.append(tag)
        if not self.hidden and tag in self.BLOCKS:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif tag in self.BLOCKS:
            self.chunks.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.chunks.append(data)


def extract_text(body, content_type):
    if content_type == "text/html":
        parser = TextExtractor()
        parser.feed(body)
        parser.close()
        body = "".join(parser.chunks)
    return "\n".join(" ".join(line.split()) for line in body.splitlines() if line.strip())


def retrieve(url, hosts, fixtures):
    chain = []
    current = url
    while True:
        current = Validator.url(current, hosts)
        Validator.require(current not in chain, "Redirect cycle detected")
        chain.append(current)
        Validator.require(current in fixtures, f"No synthetic fixture for URL: {current}")
        response = fixtures[current]
        if "redirect_to" not in response:
            return current, chain, response
        Validator.require(len(chain) <= 5, "Redirect limit of five exceeded")
        current = response["redirect_to"]


def research(request):
    """Atomic deterministic research; retrieved pages are data, never instructions."""
    hosts, urls, fixtures, limit = Validator.request(request)
    terms = set(re.findall(r"\w+", request["query"].casefold()))
    result = dict(schema_version=1, synthetic=True, status="ok", query=request["query"],
                  sources=[], findings=[], warnings=[])
    candidates = []
    for index, requested in enumerate(urls, 1):
        final, chain, response = retrieve(requested, hosts, fixtures)
        text = extract_text(response["body"], response["content_type"])
        source = dict(source_id=f"source-{index:03d}", requested_url=requested,
                      final_url=final, redirect_chain=chain,
                      retrieved_at=response["retrieved_at"],
                      content_type=response["content_type"], text=text,
                      content_sha256=hashlib.sha256(response["body"].encode("utf-8")).hexdigest(),
                      text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      synthetic=True)
        result["sources"].append(source)
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", text):
            raw = match.group()
            quote = raw.strip()
            if not quote:
                continue
            matched = sorted(terms & set(re.findall(r"\w+", quote.casefold())))
            if not matched:
                continue
            start = match.start() + len(raw) - len(raw.lstrip())
            candidates.append(dict(source_id=source["source_id"], url=final,
                                   retrieved_at=source["retrieved_at"],
                                   content_sha256=source["content_sha256"],
                                   quote=quote, start=start, end=start + len(quote),
                                   matched_terms=matched, synthetic=True))
    for index, finding in enumerate(candidates[:limit], 1):
        result["findings"].append(dict(finding_id=f"finding-{index:03d}", **finding))
    if not candidates:
        result["warnings"].append("No query terms matched the retrieved text.")
    if len(candidates) > limit:
        result["warnings"].append(f"Findings truncated from {len(candidates)} to {limit}.")
    return Validator.response(result)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        Validator.require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Nonstandard JSON constant: {value}")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        Validator.require(len(args) == 1, "Usage: python -B implementation.py input.json")
        with Path(args[0]).open("rb") as handle:
            raw = handle.read(2_000_001)
        Validator.require(len(raw) <= 2_000_000, "Input file exceeds 2000000 bytes")
        request = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        result = research(request)
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError,
            RecursionError, OverflowError) as exc:
        print(json.dumps(dict(schema_version=1, status="error", error=str(exc)),
                         ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
