"""A small stdlib HTTP server for the assay studio.

Deliberately dependency-free. The rest of the project runs on a fresh clone with
no credentials and no package installs; an npm toolchain for the sake of a demo
page would contradict that, and would make the one claim the studio exists to
support -- that you can check this yourself -- harder to act on.

The server holds a single :class:`~assay.studio.demo.DemoState`. The browser
polls ``/api/state``; there is no websocket and no server-sent events, because a
poll is easier to reason about when the interesting failure mode is "the run
died halfway".
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..pricing import PROVISIONAL
from ..quote import make_quote
from .demo import DemoState, start

APP_HTML = Path(__file__).with_name("app.html")


def _quote(state: DemoState, arm: str, units: dict[str, float], context_bytes: int) -> dict[str, Any]:
    """Price a request the visitor typed, using whatever that arm actually earned.

    The null arm is the point of this endpoint. It has no per-brick prices, so it
    answers with a size-only band -- visibly wider, and visibly indifferent to
    which bricks you asked for. That contrast is the demonstration.
    """
    priced = state.priced.get(arm)
    if priced is None:
        return {"error": f"the {arm} arm has not been fitted yet"}
    q = make_quote(
        f"studio-{arm}",
        units,
        context_bytes,
        model=priced["model"],
        conformal=priced["conformal"],
        pricing=PROVISIONAL,
        show_per_brick=priced["show_per_brick"],
    )
    out = q.as_dict()
    out["arm"] = arm
    out["show_per_brick"] = priced["show_per_brick"]
    return out


class _Handler(BaseHTTPRequestHandler):
    state: DemoState
    server_version = "assay-studio"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102 - quiet by default
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, APP_HTML.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self._json(self.state.snapshot())
        elif url.path == "/api/quote":
            q = parse_qs(url.query)
            arm = (q.get("arm") or ["brick"])[0]
            try:
                context_bytes = int((q.get("bytes") or ["4000"])[0])
            except ValueError:
                self._json({"error": "bytes must be a whole number"}, 400)
                return
            units: dict[str, float] = {}
            for key, values in q.items():
                if not key.startswith("u."):
                    continue
                try:
                    units[key[2:]] = float(values[0])
                except ValueError:
                    self._json({"error": f"{key[2:]} must be a number"}, 400)
                    return
            self._json(_quote(self.state, arm, units, context_bytes))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if urlparse(self.path).path == "/api/run":
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            # start() hands back a Thread, which is not JSON-serialisable; the
            # browser only needs to know whether its click was accepted.
            started = start(self.state) is not None
            self._json({"started": started, "status": self.state.status})
        else:
            self._json({"error": "not found"}, 404)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    autorun: bool = True,
    state: DemoState | None = None,
) -> ThreadingHTTPServer:
    """Start the studio. Returns the server; call ``shutdown()`` to stop it."""
    state = state or DemoState()
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    if autorun:
        start(state)
    url = f"http://{host}:{httpd.server_address[1]}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    return httpd


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="ty studio", description="Run the assay studio.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    httpd = serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"assay studio on {url}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
