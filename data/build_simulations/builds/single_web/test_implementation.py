import contextlib
import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import Mock, patch

import implementation as web


BASE = "https://research.example.org/"


def payload():
    return {
        "query": "solar storage",
        "allowed_hosts": ["research.example.org"],
        "sources": [{"id": "a", "url": BASE}],
        "fixtures": {BASE: {"body": "<p>Solar energy storage uses batteries.</p><p>Flowers grow.</p>"}},
    }


def response(body="<p>Solar storage works.</p>", status=200, headers=None):
    return web.Response(status, headers or {"Content-Type": "text/html"}, body.encode())


class ResearchTests(unittest.TestCase):
    def error(self, code, operation):
        with self.assertRaises(web.ResearchError) as caught:
            operation()
        self.assertEqual(code, caught.exception.code)

    def test_offline_ranks_and_cites_real_text(self):
        result = web.research(payload())
        self.assertEqual(result["mode"], "offline_fixtures")
        self.assertEqual(len(result["passages"]), 1)
        self.assertEqual(result["passages"][0]["url"], BASE)
        self.assertIn("batteries", result["passages"][0]["quote"])

    def test_query_changes_retrieval(self):
        data = payload()
        data["query"] = "flowers"
        self.assertEqual(web.research(data)["passages"][0]["quote"], "Flowers grow.")

    def test_ranking_rewards_query_coverage(self):
        data = payload()
        data["fixtures"][BASE]["body"] = "<p>Solar panels work.</p><p>Solar storage works.</p>"
        self.assertEqual(web.research(data)["passages"][0]["quote"], "Solar storage works.")

    def test_entities_inline_tags_hidden_elements_and_chunking(self):
        parser = web.TextExtractor()
        parser.feed("<head>hidden</head><p>Solar <b>storage</b> &amp; energy.</p>"
                    "<script>fake</script><style>fake</style><template>fake</template>"
                    "<p>" + "word " * 200 + "</p><p>" + "x" * 1401 + "</p>")
        parts = parser.passages()
        self.assertEqual(parts[0], "Solar storage & energy.")
        self.assertNotIn("fake", " ".join(parts))
        self.assertTrue(all(len(part) <= 700 for part in parts))

    def test_unsafe_urls_before_fetch(self):
        policy = web.URLPolicy(["research.example.org"])
        for url in [
            "http://research.example.org/", "https://evil.example.org/",
            "https://research.example.org.evil.org/", "https://user@research.example.org/",
            "https://research.example.org:444/", "https://research.example.org/#fragment",
            "https://127.0.0.1/", "https://[::1]/", "https://research.example.org\\@evil.org/",
            " https://research.example.org/", "https://research.example.org/\n",
            "https://research.example.org:/", "file:///etc/passwd",
            "https://research%2eexample.org/", "https://research.example.org:bad/",
        ]:
            with self.subTest(url=url):
                self.error("unsafe_url", lambda: policy.validate(url))

    def test_canonicalization(self):
        self.assertEqual(web.URLPolicy(["RESEARCH.example.org"]).validate(
            "https://RESEARCH.example.org:443"), BASE)

    def test_no_implicit_subdomain_allowance(self):
        self.error("unsafe_url", lambda: web.URLPolicy(["example.org"]).validate(BASE))

    def test_missing_allowlist(self):
        data = payload()
        del data["allowed_hosts"]
        self.error("invalid_input", lambda: web.research(data))

    def test_duplicate_ids_preflight(self):
        data = payload()
        data["sources"].append(dict(data["sources"][0]))
        fetch = Mock()
        self.error("duplicate_id", lambda: web.research(data, fetch))
        fetch.assert_not_called()

    def test_all_sources_validated_before_fetch(self):
        data = payload()
        data["sources"].append({"id": "b", "url": "https://evil.org/"})
        fetch = Mock()
        self.error("unsafe_url", lambda: web.research(data, fetch))
        fetch.assert_not_called()

    def test_relative_redirect_and_final_citation(self):
        fetch = Mock(side_effect=[
            web.Response(302, {"Location": "/final"}, b""),
            response(),
        ])
        result = web.research(payload(), fetch)
        self.assertEqual(result["passages"][0]["url"], BASE + "final")
        self.assertEqual(fetch.call_args_list[1].args[0], BASE + "final")

    def test_unsafe_redirects_never_fetched(self):
        for target in ["http://research.example.org/", "//evil.org/", " /bad", "/x\n"]:
            fetch = Mock(return_value=web.Response(302, {"Location": target}, b""))
            with self.subTest(target=target):
                self.error("unsafe_url", lambda: web.research(payload(), fetch))
                self.assertEqual(fetch.call_count, 1)

    def test_redirect_loop(self):
        self.error("redirect_loop", lambda: web.research(
            payload(), lambda *args: web.Response(302, {"Location": "/"}, b"")))

    def test_redirect_limit(self):
        data = payload()
        data["limits"] = {"max_redirects": 0}
        self.error("redirect_limit", lambda: web.research(
            data, lambda *args: web.Response(302, {"Location": "/next"}, b"")))

    def test_redirect_missing_location(self):
        self.error("invalid_response", lambda: web.research(
            payload(), lambda *args: web.Response(302, {}, b"")))

    def test_unsupported_content_and_encoding(self):
        for headers in [
            {}, {"Content-Type": "application/json"},
            {"Content-Type": "text/html", "Content-Encoding": "gzip"},
            {"Content-Type": "text/html; charset=unknown-charset"},
        ]:
            with self.subTest(headers=headers):
                self.error("unsupported_content", lambda: web.research(
                    payload(), lambda *args: web.Response(200, headers, b"solar")))

    def test_invalid_utf8(self):
        self.error("unsupported_content", lambda: web.research(
            payload(), lambda *args: web.Response(200, {"Content-Type": "text/html"}, b"\xff")))

    def test_content_limit(self):
        data = payload()
        data["limits"] = {"max_bytes": 5}
        self.error("content_too_large", lambda: web.research(data))

    def test_timeout_exception(self):
        fetch = Mock(side_effect=TimeoutError())
        self.error("timeout", lambda: web.research(payload(), fetch))

    def test_over_budget_injected_fetcher(self):
        with patch.object(web.time, "monotonic", side_effect=[0, 0, 11]):
            self.error("timeout", lambda: web.research(payload(), lambda *args: response()))

    def test_http_error(self):
        self.error("http_error", lambda: web.research(
            payload(), lambda *args: response(status=404)))

    def test_no_match(self):
        data = payload()
        data["query"] = "octopus"
        self.error("no_match", lambda: web.research(data))

    def test_missing_fixture(self):
        data = payload()
        data["fixtures"] = {}
        self.error("missing_fixture", lambda: web.research(data))

    def test_invalid_input_and_limits(self):
        for update in [
            {"query": "..."}, {"sources": []}, {"limits": {"max_bytes": True}},
            {"limits": {"timeout_seconds": float("nan")}}, {"limits": {"top_k": 1.5}},
            {"limits": {"unexpected": 3}},
        ]:
            data = payload()
            data.update(update)
            with self.subTest(update=update):
                self.error("invalid_input", lambda: web.research(data))

    def test_invalid_fetcher_response(self):
        for value in [None, web.Response(200, {}, "not bytes"),
                      web.Response(200, {"Content-Type": "text/html", "content-type": "text/plain"}, b"")]:
            self.error("invalid_response", lambda: web.research(payload(), lambda *args: value))

    def test_valid_synthesis(self):
        def synth(query, passages):
            return {"answer": "The captured text discusses batteries.",
                    "citations": [{"url": passages[0]["url"], "quote": "storage uses batteries"}]}
        self.assertIn("synthesis", web.research(payload(), synthesizer=synth))

    def test_invalid_synthesis(self):
        for citation in [
            {"url": "https://evil.org/", "quote": "Solar"},
            {"url": BASE, "quote": "invented evidence"},
            {"url": BASE, "quote": ""},
            {"url": BASE, "quote": "Flowers grow."},
        ]:
            with self.subTest(citation=citation):
                self.error("invalid_synthesis", lambda: web.research(
                    payload(), synthesizer=lambda *args: {"answer": "Answer", "citations": [citation]}))

    def test_synth_cannot_mutate_evidence(self):
        def synth(query, passages):
            passages[0]["quote"] = "Fabrication"
            return {"answer": "Answer", "citations": [{"url": BASE, "quote": "Fabrication"}]}
        self.error("invalid_synthesis", lambda: web.research(payload(), synthesizer=synth))

    def test_dns_private_and_mixed_answers_rejected(self):
        for addresses in [["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"],
                          ["8.8.8.8", "192.168.0.1"]]:
            records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in addresses]
            with patch.object(web.socket, "getaddrinfo", return_value=records):
                self.error("unsafe_url", lambda: web._public_addresses("research.example.org", 1))

    def test_public_dns_addresses_preserved(self):
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        with patch.object(web.socket, "getaddrinfo", return_value=records):
            self.assertEqual(web._public_addresses("research.example.org", 1)[0][-1], ("8.8.8.8", 443))

    def test_pinned_connection_uses_public_address_and_original_tls_name(self):
        context = Mock()
        connection = web._PinnedHTTPS("research.example.org", timeout=1, context=context)
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, ("8.8.8.8", 443))]
        sock = Mock()
        with patch.object(web, "_public_addresses", return_value=addresses), \
                patch.object(web.socket, "socket", return_value=sock):
            connection.connect()
        sock.connect.assert_called_once_with(("8.8.8.8", 443))
        context.wrap_socket.assert_called_once_with(sock, server_hostname="research.example.org")

    def test_urllib_adapter_without_network(self):
        fetcher = web.HTTPSFetcher(web.URLPolicy(["research.example.org"]))
        network_response = Mock()
        network_response.status = 200
        network_response.headers = {"Content-Type": "text/html"}
        network_response.__enter__ = Mock(return_value=network_response)
        network_response.__exit__ = Mock(return_value=False)
        network_response.isclosed.return_value = False
        network_response.read1.side_effect = [b"<p>Solar storage.</p>", b""]
        fetcher.opener = Mock()
        fetcher.opener.open.return_value = network_response
        self.assertIn(b"Solar", fetcher(BASE, 1, 100).body)
        request = fetcher.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, BASE)
        self.assertEqual(request.get_header("Accept-encoding"), "identity")

    def test_urllib_adapter_size_timeout_and_redirect(self):
        fetcher = web.HTTPSFetcher(web.URLPolicy(["research.example.org"]))
        fake = Mock()
        fake.status = 200
        fake.headers = {"Content-Type": "text/html"}
        fake.__enter__ = Mock(return_value=fake)
        fake.__exit__ = Mock(return_value=False)
        fake.read1.return_value = b"123456"
        fetcher.opener = Mock()
        fetcher.opener.open.return_value = fake
        self.error("content_too_large", lambda: fetcher(BASE, 1, 5))
        fetcher.opener.open.side_effect = web.urllib.error.URLError(socket.timeout())
        self.error("timeout", lambda: fetcher(BASE, 1, 5))
        fetcher.opener.open.side_effect = None
        fake.status = 302
        fake.headers = {"Location": "/next"}
        self.assertEqual(fetcher(BASE, 1, 5).status, 302)

    def test_urllib_redirect_handler_does_not_follow(self):
        self.assertIsNone(web._NoRedirect().redirect_request(None, None, 302, "", {}, BASE))

    def test_actual_http_response_content_length_eof_without_network(self):
        body = b"<p>Solar storage.</p>"

        class Wire(io.BytesIO):
            @property
            def raw(self):
                return self

        wire = Wire(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: "
                    + str(len(body)).encode() + b"\r\n\r\n" + body)
        wire._sock = Mock()
        sock = Mock()
        sock.makefile.return_value = wire
        raw_response = web.http.client.HTTPResponse(sock)
        raw_response.begin()
        fetcher = web.HTTPSFetcher(web.URLPolicy(["research.example.org"]))
        fetcher.opener = Mock()
        fetcher.opener.open.return_value = raw_response
        self.assertEqual(fetcher(BASE, 1, 100).body, body)
        self.assertTrue(raw_response.isclosed())

    def test_cli_example_and_json_error(self):
        example = str(Path(__file__).with_name("example_input.json"))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(web.main([example]), 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "offline_fixtures")
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(web.main([str(Path(__file__).with_name("nonexistent.json"))]), 2)
        self.assertEqual(json.loads(error.getvalue())["error"]["code"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
