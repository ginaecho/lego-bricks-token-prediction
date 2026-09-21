"""Serve the local-only Customer Outcomes Studio and its scratch SQLite store.

Run: python -m examples.customer_outcomes_server --port 8793
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from examples.customer_outcomes_demo import seed_demo
from token_yield.customer_outcomes import OutcomeStore, canonical
from token_yield.delivery_feedback import DeliveryFeedback
from token_yield.brick_learning import BrickLearning


LOGGER = logging.getLogger(__name__)
PAGE = Path(__file__).with_name("customer-outcomes-prototype.html")
FEEDBACK_PAGE = Path(__file__).with_name("delivered-project-feedback.html")
LEARNING_PAGE = Path(__file__).with_name("brick-learning-prototype.html")
BRICK_CATALOG = Path(__file__).with_name("marketplace_feedback_catalog.json")


def _invalid_constant(value: str):
    raise ValueError(f"nonfinite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def make_server(store: OutcomeStore, port: int) -> ThreadingHTTPServer:
    deliveries = DeliveryFeedback(store)
    bricks = BrickLearning(deliveries)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            LOGGER.info(format, *args)

        def local_request(self) -> bool:
            hosts = {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return ((origin is None or origin in {f"http://{host}" for host in hosts})
                    and self.headers.get("Sec-Fetch-Site") != "cross-site")

        def respond(self, code: int, body: object, *, html: bool = False) -> None:
            if html:
                if not isinstance(body, str):
                    raise TypeError("HTML response must be text")
                content = body.encode("utf-8")
            else:
                content = canonical(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else
                             "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' "
                             "'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                             "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
                             "base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if not self.local_request():
                self.respond(403, {"error": "local same-origin requests only"})
                return
            url = urlsplit(self.path)
            try:
                if url.path in ("/", "/customer-outcomes-prototype.html", "/feedback"):
                    self.respond(200, FEEDBACK_PAGE.read_text(encoding="utf-8"), html=True)
                elif url.path == "/admin":
                    self.respond(200, PAGE.read_text(encoding="utf-8"), html=True)
                elif url.path == "/learning":
                    self.respond(200, LEARNING_PAGE.read_text(encoding="utf-8"), html=True)
                elif url.path == "/api/brick-catalog":
                    self.respond(200, json.loads(BRICK_CATALOG.read_text(encoding="utf-8")))
                elif url.path == "/api/deliveries":
                    receipt = parse_qs(url.query).get("receipt", [None])[0]
                    self.respond(200, deliveries.get(receipt) if receipt else deliveries.listing())
                elif url.path == "/api/delivery-learning":
                    source = parse_qs(url.query).get("source", ["real"])[0]
                    self.respond(200, deliveries.learning(source))
                elif url.path == "/api/state":
                    source = parse_qs(url.query).get("source", ["real"])[0]
                    self.respond(200, store.state(source))
                elif url.path == "/api/health":
                    self.respond(200, {"status": "ok", "prototype": True})
                else:
                    self.respond(404, {"error": "unknown route"})
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except (OSError, sqlite3.Error):
                LOGGER.exception("Unable to load local studio")
                self.respond(500, {"error": "Local store/page unavailable; inspect the server log."})

        def do_POST(self):
            if not self.local_request():
                self.respond(403, {"error": "local same-origin requests only"})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "application/json is required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("JSON body must be 1 to 1000000 bytes")
                payload = json.loads(self.rfile.read(length), parse_constant=_invalid_constant,
                                     object_pairs_hook=_unique_object)
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                route = urlsplit(self.path).path
                if route == "/api/deliveries/feedback":
                    result = deliveries.submit(payload)
                elif route == "/api/brick-rankings":
                    result = bricks.rankings(**payload)
                elif route == "/api/brick-recommendations":
                    result = bricks.recommend(**payload)
                elif route == "/api/brick-learning/demo":
                    result = bricks.seed_demo(**payload)
                elif route == "/api/delivery-policy/suggest":
                    if set(payload) != {"source", "feature_id", "options"}:
                        raise ValueError("suggest requires source, feature_id, options")
                    result = deliveries.suggest(**payload)
                elif route == "/api/delivery-policy/promote":
                    result = deliveries.promote(**payload)
                elif route == "/api/delivery-policy/predict":
                    result = deliveries.predict(**payload)
                elif route == "/api/projects":
                    result = store.create_project(payload)
                elif route == "/api/recommend":
                    if not {"project_id", "function_id"} <= set(payload) <= {
                            "project_id", "function_id", "split"}:
                        raise ValueError("recommend accepts project_id, function_id, and optional split only")
                    result = store.recommend(**payload)
                elif route == "/api/approve":
                    result = store.approve(**payload)
                elif route == "/api/feedback":
                    result = store.feedback(payload)
                elif route == "/api/review":
                    result = store.review(**payload)
                elif route == "/api/train":
                    result = store.train(**payload)
                elif route == "/api/evaluate":
                    result = store.evaluate(**payload)
                elif route == "/api/promote":
                    result = store.promote(**payload)
                elif route == "/api/predict":
                    result = store.predict(**payload)
                elif route == "/api/demo":
                    if payload:
                        raise ValueError("demo accepts an empty object only")
                    result = seed_demo(store)
                else:
                    self.respond(404, {"error": "unknown route"})
                    return
                self.respond(200, result)
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                LOGGER.warning("Rejected outcomes request: %s", exc)
                self.respond(400, {"error": str(exc)})
            except sqlite3.Error:
                LOGGER.exception("Outcomes transaction failed")
                self.respond(500, {"error": "Local database transaction failed; inspect server log."})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--db", type=Path,
                        default=Path(".outcomes-prototype") / "PROTOTYPE-customer-outcomes.sqlite3")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not 0 <= args.port <= 65535:
        LOGGER.error("port must be between 0 and 65535")
        return 2
    try:
        store = OutcomeStore(args.db)
        with make_server(store, args.port) as server:
            LOGGER.info("Local prototype: http://127.0.0.1:%s (database: %s)",
                        server.server_port, args.db.resolve())
            server.serve_forever()
    except KeyboardInterrupt:
        return 130
    except (OSError, sqlite3.Error, ValueError):
        LOGGER.exception("Could not start Customer Outcomes Studio")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
