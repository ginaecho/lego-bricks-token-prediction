"""Stdlib web passage retrieval. Run: python implementation.py example_input.json.

Offline captured fixtures are the default. --live explicitly enables HTTPS.
Injected fetchers receive (validated_url, remaining_seconds, max_bytes) and must
honor those bounds. Injected synthesis is optional and never invoked by the CLI.
"""

import argparse
import codecs
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ResearchError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fail(code, message):
    raise ResearchError(code, message)


def hostname(value):
    if not isinstance(value, str) or not value or value != value.strip():
        fail("unsafe_url", "Allowlist entries must be bare hostnames")
    try:
        result = value.encode("idna").decode("ascii").lower()
    except UnicodeError:
        fail("unsafe_url", "Invalid hostname")
    if len(result) > 253 or not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+",
        result,
    ) or any(len(label) > 63 for label in result.split(".")):
        fail("unsafe_url", "Invalid public DNS hostname")
    try:
        ipaddress.ip_address(result)
    except ValueError:
        return result
    fail("unsafe_url", "IP literals are not allowed")


class URLPolicy:
    def __init__(self, allowed_hosts):
        if not isinstance(allowed_hosts, list) or not allowed_hosts:
            fail("invalid_input", "A nonempty explicit hostname allowlist is required")
        self.hosts = {hostname(host) for host in allowed_hosts}

    def validate(self, url):
        if not isinstance(url, str) or not url or len(url) > 8192:
            fail("unsafe_url", "Invalid URL")
        if "\\" in url or any(ord(char) <= 32 or ord(char) == 127 for char in url):
            fail("unsafe_url", "URL contains whitespace, controls, or backslashes")
        try:
            parsed = urllib.parse.urlsplit(url)
            port = parsed.port
            host = hostname(parsed.hostname or "")
        except (ValueError, UnicodeError):
            fail("unsafe_url", "Malformed URL")
        if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
            fail("unsafe_url", "Only credential-free HTTPS URLs are allowed")
        if port not in (None, 443) or host not in self.hosts or parsed.fragment:
            fail("unsafe_url", "Hostname, port, or fragment violates URL policy")
        if "%" in parsed.netloc or parsed.netloc.endswith(":"):
            fail("unsafe_url", "Ambiguous authority is forbidden")
        return urllib.parse.urlunsplit(("https", host, parsed.path or "/", parsed.query, ""))


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict
    body: bytes


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _public_addresses(host, timeout):
    # Bound DNS waiting as well as socket I/O; no connection uses a second lookup.
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(socket.getaddrinfo, host, 443, 0, socket.SOCK_STREAM)
    try:
        infos = future.result(timeout=timeout)
    except FutureTimeout:
        fail("timeout", "DNS lookup timed out")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    addresses = []
    for family, socktype, proto, _, address in infos:
        ip = ipaddress.ip_address(address[0])
        if not ip.is_global:
            fail("unsafe_url", "DNS resolved to a non-public address")
        if (family, socktype, proto, address) not in addresses:
            addresses.append((family, socktype, proto, address))
    if not addresses:
        fail("network_error", "Hostname has no usable addresses")
    return addresses


class _PinnedHTTPS(http.client.HTTPSConnection):
    def connect(self):
        deadline = time.monotonic() + self.timeout
        addresses = _public_addresses(self.host, self.timeout)
        last_error = None
        for family, socktype, proto, address in addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Connection deadline expired")
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(remaining)
                sock.connect(address)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("TLS deadline expired")
                sock.settimeout(remaining)
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except OSError as exc:
                last_error = exc
                sock.close()
        raise last_error or OSError("No usable address")


class _PinnedHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PinnedHTTPS, req, context=self._context)


class HTTPSFetcher:
    """Actual urllib HTTPS transport; proxy-free, pinned public DNS and TLS verified."""

    def __init__(self, policy):
        self.policy = policy
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
            _PinnedHandler(context=ssl.create_default_context()),
        )

    def __call__(self, url, timeout, max_bytes):
        url = self.policy.validate(url)
        deadline = time.monotonic() + timeout
        req = urllib.request.Request(url, headers={
            "User-Agent": "MeasuredResearchCLI/1.0",
            "Accept": "text/html, application/xhtml+xml",
            "Accept-Encoding": "identity",
        })
        response = None
        try:
            try:
                response = self.opener.open(req, timeout=timeout)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                headers = dict(response.headers.items())
                # Redirect/error bodies are not useful and need not be downloaded.
                if response.status != 200:
                    return Response(response.status, headers, b"")
                chunks, count = [], 0
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        fail("timeout", "Download deadline expired")
                    # urllib exposes the HTTPResponse socket through this chain.
                    response.fp.raw._sock.settimeout(remaining)
                    chunk = response.read1(min(65536, max_bytes + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > max_bytes:
                        fail("content_too_large", "Response exceeds byte limit")
                    chunks.append(chunk)
                    if response.isclosed():
                        break
                return Response(response.status, headers, b"".join(chunks))
        except (TimeoutError, socket.timeout):
            fail("timeout", "HTTPS request timed out")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                fail("timeout", "HTTPS request timed out")
            fail("network_error", str(exc.reason))
        except OSError as exc:
            fail("network_error", str(exc))


class FixtureFetcher:
    def __init__(self, fixtures, policy):
        if not isinstance(fixtures, dict):
            fail("invalid_input", "Offline mode requires a fixtures object")
        self.fixtures = {}
        for url, fixture in fixtures.items():
            canonical = policy.validate(url)
            if canonical in self.fixtures:
                fail("invalid_input", "Duplicate canonical fixture URL")
            self.fixtures[canonical] = fixture

    def __call__(self, url, timeout, max_bytes):
        if url not in self.fixtures:
            fail("missing_fixture", "No captured fixture for " + url)
        fixture = self.fixtures[url]
        if not isinstance(fixture, dict) or not isinstance(fixture.get("body", ""), str):
            fail("invalid_input", "Fixture must have a string body")
        return Response(fixture.get("status", 200), fixture.get("headers", {
            "Content-Type": "text/html; charset=utf-8",
        }), fixture.get("body", "").encode("utf-8"))


class TextExtractor(HTMLParser):
    BLOCKS = {"p", "div", "article", "section", "main", "li", "br", "hr",
              "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote"}
    HIDDEN = {"script", "style", "noscript", "template", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.HIDDEN:
            self.hidden.append(tag)
        if not self.hidden and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def passages(self):
        result = []
        for block in "".join(self.parts).splitlines():
            words = block.split()
            current = []
            length = 0
            for word in words:
                if current and length + len(word) + 1 > 700:
                    result.append(" ".join(current))
                    current, length = [], 0
                # Long unbroken input must also respect the passage bound.
                while len(word) > 700:
                    if current:
                        result.append(" ".join(current))
                        current, length = [], 0
                    result.append(word[:700])
                    word = word[700:]
                current.append(word)
                length += len(word) + 1
            if current:
                result.append(" ".join(current))
        return result


def extract(response, max_bytes):
    if not isinstance(response, Response) or type(response.status) is not int:
        fail("invalid_response", "Fetcher must return Response with an integer status")
    if not isinstance(response.headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in response.headers.items()
    ):
        fail("invalid_response", "Response headers must be strings")
    if not isinstance(response.body, bytes):
        fail("invalid_response", "Response body must be bytes")
    if len(response.body) > max_bytes:
        fail("content_too_large", "Response exceeds byte limit")
    headers = {}
    for key, value in response.headers.items():
        if key.lower() in headers:
            fail("invalid_response", "Duplicate case-insensitive response header")
        headers[key.lower()] = value
    return headers


def html_passages(response, headers):
    if headers.get("content-encoding", "identity").lower() != "identity":
        fail("unsupported_content", "Compressed responses are not supported")
    message = Message()
    message["content-type"] = headers.get("content-type", "")
    if message.get_content_type() not in {"text/html", "application/xhtml+xml"}:
        fail("unsupported_content", "Expected an HTML Content-Type")
    charset = message.get_content_charset() or "utf-8"
    try:
        codecs.lookup(charset)
        text = response.body.decode(charset, errors="strict")
    except (LookupError, UnicodeError):
        fail("unsupported_content", "Unsupported charset or invalid encoded HTML")
    parser = TextExtractor()
    parser.feed(text)
    parser.close()
    return parser.passages()


def tokens(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def research(payload, fetcher=None, synthesizer=None, live=False):
    if not isinstance(payload, dict):
        fail("invalid_input", "Input must be a JSON object")
    policy = URLPolicy(payload.get("allowed_hosts"))
    query = payload.get("query")
    if not isinstance(query, str) or not tokens(query):
        fail("invalid_input", "Query must contain searchable words")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources or len(sources) > 50:
        fail("invalid_input", "Provide between 1 and 50 sources")
    limits = payload.get("limits", {})
    if not isinstance(limits, dict) or set(limits) - {"timeout_seconds", "max_bytes", "max_redirects", "top_k"}:
        fail("invalid_input", "Unknown limits")
    values = {}
    for key, default, minimum, maximum in [
        ("timeout_seconds", 10, 0.01, 60),
        ("max_bytes", 1000000, 1, 5000000),
        ("max_redirects", 3, 0, 10),
        ("top_k", 5, 1, 50),
    ]:
        value = limits.get(key, default)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not minimum <= value <= maximum
                or (key != "timeout_seconds" and type(value) is not int)):
            fail("invalid_input", "Invalid limit: " + key)
        values[key] = value
    checked, ids = [], set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"].strip():
            fail("invalid_input", "Every source requires a nonempty string id")
        if source["id"] in ids:
            fail("duplicate_id", "Duplicate source id: " + source["id"])
        ids.add(source["id"])
        checked.append((source["id"], policy.validate(source.get("url"))))
    mode = "injected_fetcher" if fetcher is not None else ("live_https" if live else "offline_fixtures")
    if fetcher is None:
        fetcher = HTTPSFetcher(policy) if live else FixtureFetcher(payload.get("fixtures"), policy)
    deadline = time.monotonic() + values["timeout_seconds"]
    candidates = []
    source_results = []
    for source_id, initial_url in checked:
        url, seen = initial_url, set()
        for hop in range(values["max_redirects"] + 1):
            if url in seen:
                fail("redirect_loop", "Redirect loop for " + source_id)
            seen.add(url)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fail("timeout", "Research deadline expired")
            try:
                response = fetcher(url, remaining, values["max_bytes"])
            except (TimeoutError, socket.timeout):
                fail("timeout", "Fetcher timed out")
            if time.monotonic() >= deadline:
                fail("timeout", "Fetcher exceeded research deadline")
            headers = extract(response, values["max_bytes"])
            if response.status in (301, 302, 303, 307, 308):
                location = headers.get("location")
                if not location:
                    fail("invalid_response", "Redirect is missing Location")
                if any(ord(char) <= 32 or ord(char) == 127 for char in location) or "\\" in location:
                    fail("unsafe_url", "Unsafe redirect location")
                target = policy.validate(urllib.parse.urljoin(url, location))
                if hop == values["max_redirects"]:
                    fail("redirect_limit", "Redirect limit exceeded")
                url = target
                continue
            if response.status != 200:
                fail("http_error", "HTTP status " + str(response.status))
            passages = html_passages(response, headers)
            source_results.append({"id": source_id, "requested_url": initial_url, "url": url})
            candidates.extend({"source_id": source_id, "url": url, "quote": passage}
                              for passage in passages)
            break
    query_terms = set(tokens(query))
    counts = [Counter(tokens(item["quote"])) for item in candidates]
    document_frequency = Counter(term for count in counts for term in query_terms if term in count)
    ranked = []
    for item, count in zip(candidates, counts):
        score = sum(
            (1 + math.log(count[term])) * (1 + math.log((1 + len(counts)) / (1 + document_frequency[term])))
            for term in query_terms if count[term]
        )
        if score:
            ranked.append({**item, "score": round(score / (1 + 0.001 * sum(count.values())), 6)})
    ranked.sort(key=lambda item: (-item["score"], item["source_id"], item["quote"]))
    if not ranked:
        fail("no_match", "No passage matches any query term")
    selected = ranked[:values["top_k"]]
    result = {"mode": mode, "query": query, "sources": source_results, "passages": selected}
    if time.monotonic() >= deadline:
        fail("timeout", "Research deadline expired during extraction or ranking")
    if synthesizer is not None:
        # Supply copies: a callback cannot alter the evidence used for validation.
        synthesis = synthesizer(query, [dict(item) for item in selected])
        if not isinstance(synthesis, dict) or not isinstance(synthesis.get("answer"), str) or not synthesis["answer"].strip():
            fail("invalid_synthesis", "Synthesis requires a nonempty answer")
        citations = synthesis.get("citations")
        if not isinstance(citations, list) or not citations:
            fail("invalid_synthesis", "Synthesis requires citations")
        for citation in citations:
            if not isinstance(citation, dict):
                fail("invalid_synthesis", "Invalid citation")
            quote, url = citation.get("quote"), citation.get("url")
            if not isinstance(quote, str) or not quote.strip() or not isinstance(url, str):
                fail("invalid_synthesis", "Citation requires a source URL and quote")
            if not any(url == item["url"] and quote in item["quote"] for item in selected):
                fail("invalid_synthesis", "Citation URL or verbatim quote is not retrieved evidence")
        if time.monotonic() >= deadline:
            fail("timeout", "Synthesis exceeded research deadline")
        result["synthesis"] = synthesis
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a JSON request")
    parser.add_argument("--live", action="store_true", help="Explicitly enable real HTTPS instead of fixtures")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as handle:
            payload = json.load(handle)
        result = research(payload, live=args.live)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        error = ResearchError("invalid_input", str(exc))
    except ResearchError as exc:
        error = exc
    print(json.dumps({"error": {"code": error.code, "message": str(error)}}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
