"""Serve the bounded offline marketplace demo on loopback only.

Usage: python -m examples.marketplace_demo_server --port 8765 --run-dir .demo-runs
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from token_yield.marketplace_demo import run_pipeline, validate_request, write_json

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROUTES = frozenset({
    "/marketplace-sales-demo.html", "/marketplace-operations-demo.html",
    "/marketplace-prototype.html",
})
MAX_BODY = 32768


class RunLimitError(Exception):
    """The local demo's work or retention bound has been reached."""


class RunStore:
    """Lock-protected snapshots with append-only worker events."""

    def __init__(self, run_dir: Path, *, max_running: int = 2, max_runs: int = 32):
        self.run_dir = Path(run_dir)
        self.max_running = max_running
        self.max_runs = max_runs
        self.lock = threading.RLock()
        self.runs: dict[str, dict[str, Any]] = {}
        self.previous_run_count = (
            sum(1 for _ in self.run_dir.glob("*/request.json")) if self.run_dir.exists() else 0
        )

    def submit(self, request: dict[str, Any]) -> str:
        request = validate_request(request)
        with self.lock:
            active = sum(run["status"] in ("queued", "running") for run in self.runs.values())
            if active >= self.max_running or self.previous_run_count + len(self.runs) >= self.max_runs:
                raise RunLimitError("Demo run limit reached; wait for active work or restart with a new run directory.")
            run_id = uuid.uuid4().hex
            self.runs[run_id] = {"id": run_id, "status": "queued", "request": request,
                                 "events": [], "result": None, "error": None}
            threading.Thread(target=self._worker, args=(run_id,), daemon=True).start()
            return run_id

    def _event(self, run_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            self.runs[run_id]["events"].append(copy.deepcopy(event))
            print(json.dumps({"run_id": run_id, **event}, ensure_ascii=True), flush=True)

    def _worker(self, run_id: str) -> None:
        try:
            with self.lock:
                self.runs[run_id]["status"] = "running"
                request = self.runs[run_id]["request"].copy()
            result = run_pipeline(request, self.run_dir, run_id=run_id,
                                  on_event=lambda event: self._event(run_id, event))
            with self.lock:
                snapshot = copy.deepcopy(self.runs[run_id])
                snapshot.update(status="completed", result=result)
                write_json(self.run_dir / run_id / "run.json", snapshot)
                self.runs[run_id].update(status="completed", result=result)
        except Exception as exc:
            with self.lock:
                run = self.runs[run_id]
                run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                event = {"seq": len(run["events"]) + 1,
                         "time": datetime.now(timezone.utc).isoformat(), "stage": "failed",
                         "message": "Offline run failed; no fallback result was generated.",
                         "data": {"error": run["error"]}}
                self._event(run_id, event)
                try:
                    folder = self.run_dir / run_id
                    folder.mkdir(parents=True, exist_ok=True)
                    with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event) + "\n")
                    write_json(folder / "run.json", run)
                except OSError as persist_error:
                    run["error"] += f"; persistence failed: {persist_error}"

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            return copy.deepcopy(self.runs.get(run_id))

    def listing(self) -> dict[str, Any]:
        with self.lock:
            return {"runs": [{"id": run["id"], "status": run["status"],
                              "description": run["request"]["description"]}
                             for run in reversed(self.runs.values())]}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON fields")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def make_server(port: int = 8765, *, run_dir: Path | None = None,
                root: Path = ROOT, store: RunStore | None = None) -> ThreadingHTTPServer:
    """Build (but do not start) a loopback server; port 0 supports isolated tests."""
    root = Path(root).resolve()
    store = store or RunStore(run_dir if run_dir is not None else ROOT / ".demo-runs")

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format: str, *args: Any) -> None:
            # Do not log attacker-controlled URLs or duplicate JSON event output.
            pass

        def _reply(self, status: int, payload: Any, *, html: bool = False) -> None:
            body = payload if html else json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(body)

        def _trusted(self) -> bool:
            hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            host_headers = self.headers.get_all("Host", [])
            origins = self.headers.get_all("Origin", [])
            if len(host_headers) != 1 or host_headers[0] not in hosts:
                self._reply(403, {"error": "untrusted Host"})
                return False
            if origins and (len(origins) != 1 or origins[0] != f"http://{host_headers[0]}"):
                self._reply(403, {"error": "untrusted Origin"})
                return False
            if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
                self._reply(403, {"error": "cross-site request rejected"})
                return False
            if not self.path.startswith("/") or self.path.startswith("//"):
                self._reply(400, {"error": "origin-form request target required"})
                return False
            return True

        def do_GET(self) -> None:
            if not self._trusted():
                return
            path = urlsplit(self.path).path
            if path == "/api/health":
                self._reply(200, {"status": "ok"})
            elif path == "/api/runs":
                self._reply(200, store.listing())
            elif re.fullmatch(r"/api/runs/[0-9a-f]{32}", path):
                run = store.get(path.rsplit("/", 1)[-1])
                self._reply(200 if run else 404, run or {"error": "run not found"})
            elif path in STATIC_ROUTES:
                file = root / path[1:]
                if file.resolve().parent != root or not file.is_file():
                    self._reply(404, {"error": "page not found"})
                    return
                self._reply(200, file.read_bytes(), html=True)
            else:
                self._reply(404, {"error": "route not found"})

        def do_POST(self) -> None:
            if not self._trusted():
                return
            if self.path != "/api/runs":
                self._reply(404, {"error": "route not found"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
                self._reply(400, {"error": "one Content-Length and no Transfer-Encoding required"})
                return
            if not re.fullmatch(r"[0-9]+", lengths[0]) or len(lengths[0]) > 8:
                self._reply(400, {"error": "invalid Content-Length"})
                return
            length = int(lengths[0])
            if length > MAX_BODY:
                self._reply(413, {"error": "JSON body too large"})
                return
            if self.headers.get_content_type() != "application/json":
                self._reply(415, {"error": "Content-Type must be application/json"})
                return
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                request = validate_request(json.loads(body.decode("utf-8"),
                                           object_pairs_hook=_unique_object,
                                           parse_constant=_invalid_constant))
            except (ValueError, UnicodeError, RecursionError):
                self._reply(400, {"error": "Invalid JSON or request fields: description 20..6000, valid model_id, integer runs_per_month 0..1000000, optional new_function <=120."})
                return
            except TimeoutError:
                self._reply(408, {"error": "request body timed out"})
                return
            try:
                run_id = store.submit(request)
            except RunLimitError as exc:
                self._reply(429, {"error": str(exc)})
                return
            self._reply(202, {"id": run_id, "operations_url": f"/marketplace-operations-demo.html?run={run_id}"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".demo-runs")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if not 1 <= args.port <= 65535:
        print("--port must be in 1..65535", file=sys.stderr)
        return 2
    try:
        with make_server(args.port, run_dir=args.run_dir) as server:
            print(json.dumps({"stage": "server", "url": f"http://127.0.0.1:{args.port}/marketplace-sales-demo.html",
                              "source": "synthetic", "offline": True}), flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 130
    except (OSError, BrokenPipeError) as exc:
        print(f"Server error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
