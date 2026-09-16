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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from token_yield.marketplace_demo import PipelineCancelled, run_pipeline, validate_request, write_json

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROUTES = frozenset({
    "/marketplace-sales-demo.html", "/marketplace-operations-demo.html",
    "/marketplace-prototype.html",
})
MAX_BODY = 32768


class RunLimitError(Exception):
    """The local demo's work or retention bound has been reached."""


class StageConflictError(Exception):
    """A stale, duplicate, or terminal-run control request was rejected."""


class RunStore:
    """Lock-protected snapshots with append-only worker events."""

    def __init__(self, run_dir: Path, *, max_running: int = 2, max_runs: int = 32,
                 stage_timeout: float = 1800):
        self.run_dir = Path(run_dir)
        self.max_running = max_running
        self.max_runs = max_runs
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stage_timeout = stage_timeout
        self.runs: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()
        self._approved: dict[str, str] = {}
        self._cancelled: set[str] = set()
        self._finalizing: set[str] = set()
        self.previous_run_count = (
            sum(1 for _ in self.run_dir.glob("*/request.json")) if self.run_dir.exists() else 0
        )

    def submit(self, request: dict[str, Any]) -> str:
        request = validate_request(request)
        with self.lock:
            if len(self._active) >= self.max_running or self.previous_run_count + len(self.runs) >= self.max_runs:
                raise RunLimitError("Demo run limit reached; wait for active work or restart with a new run directory.")
            run_id = uuid.uuid4().hex
            self.runs[run_id] = {"id": run_id, "status": "queued", "request": request,
                                 "events": [], "result": None, "error": None, "next_stage": None}
            self._active.add(run_id)
            threading.Thread(target=self._worker, args=(run_id,), daemon=True).start()
            return run_id

    def approve(self, run_id: str, stage: str) -> None:
        """Consume exactly the current named gate; approval never carries forward."""
        with self.condition:
            run = self.runs[run_id]
            if (run["status"] != "waiting" or run["next_stage"] != stage
                    or run_id in self._cancelled):
                raise StageConflictError("Stale or duplicate approval; refresh next_stage.")
            self._approved[run_id] = stage
            run.update(status="running", next_stage=None)
            self.condition.notify_all()

    def cancel(self, run_id: str) -> None:
        """Cancel cooperatively at a stage boundary, before finalization starts."""
        with self.condition:
            run = self.runs[run_id]
            if run["status"] not in ("queued", "waiting", "running") or run_id in self._finalizing:
                raise StageConflictError("Run is terminal or finalization has already started.")
            self._cancelled.add(run_id)
            self._approved.pop(run_id, None)
            run.update(status="cancelled", next_stage=None)
            self.condition.notify_all()

    def _before_stage(self, run_id: str, stage: str) -> None:
        with self.condition:
            if run_id in self._cancelled:
                raise PipelineCancelled("Cancelled by user before the next stage.")
            run = self.runs[run_id]
            if run["request"]["execution_mode"] == "step":
                run.update(status="waiting", next_stage=stage)
                write_json(self.run_dir / run_id / "run.json", run)
                self.condition.notify_all()
                approved = self.condition.wait_for(
                    lambda: run_id in self._cancelled or self._approved.get(run_id) == stage,
                    timeout=self.stage_timeout,
                )
                if run_id in self._cancelled:
                    raise PipelineCancelled("Cancelled by user while waiting for stage approval.")
                if not approved:
                    run.update(status="failed", next_stage=None)
                    raise TimeoutError(f"Stage approval timed out after {self.stage_timeout:g}s: {stage}")
                del self._approved[run_id]
            run.update(status="running", next_stage=None)
            if stage == "complete":
                # Final artifact publication is indivisible from the user's perspective.
                self._finalizing.add(run_id)

    def _event(self, run_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            self.runs[run_id]["events"].append(copy.deepcopy(event))
            print(json.dumps({"run_id": run_id, **event}, ensure_ascii=True), flush=True)

    def _worker(self, run_id: str) -> None:
        try:
            with self.lock:
                if run_id not in self._cancelled:
                    self.runs[run_id]["status"] = "running"
                request = self.runs[run_id]["request"].copy()
            result = run_pipeline(request, self.run_dir, run_id=run_id,
                                  on_event=lambda event: self._event(run_id, event),
                                  before_stage=lambda stage: self._before_stage(run_id, stage))
            with self.lock:
                snapshot = copy.deepcopy(self.runs[run_id])
                snapshot.update(status="completed", result=result, next_stage=None)
                write_json(self.run_dir / run_id / "run.json", snapshot)
                self.runs[run_id].update(status="completed", result=result, next_stage=None)
        except Exception as exc:
            with self.lock:
                run = self.runs[run_id]
                cancelled = isinstance(exc, PipelineCancelled)
                status = "cancelled" if cancelled else "failed"
                run.update(status=status, next_stage=None, result=None,
                           error=None if cancelled else f"{type(exc).__name__}: {exc}")
                # Terminal diagnostics are not completed work stages.
                print(json.dumps({"run_id": run_id, "status": status, "error": run["error"]}),
                      flush=True)
                try:
                    folder = self.run_dir / run_id
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder / "run.json", run)
                except OSError as persist_error:
                    run["error"] = f"{run['error'] or status}; persistence failed: {persist_error}"
        finally:
            with self.condition:
                self._active.discard(run_id)
                self._approved.pop(run_id, None)
                self._finalizing.discard(run_id)
                self.condition.notify_all()

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
            control = re.fullmatch(r"/api/runs/([0-9a-f]{32})/(next|cancel)", self.path)
            if self.path != "/api/runs" and not control:
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
                request = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object,
                                     parse_constant=_invalid_constant)
                if control:
                    if not isinstance(request, dict):
                        raise ValueError("control body must be an object")
                    if control[2] == "next":
                        if set(request) != {"stage"} or not isinstance(request["stage"], str):
                            raise ValueError("next requires exactly one string stage")
                    elif request:
                        raise ValueError("cancel requires an empty object")
                else:
                    request = validate_request(request)
            except (ValueError, UnicodeError, RecursionError):
                self._reply(400, {"error": "Invalid JSON or request fields. Runs require description, model_id, integer runs_per_month; optional new_function and execution_mode step|automatic. Next requires {stage:string}; cancel requires {}."})
                return
            except TimeoutError:
                self._reply(408, {"error": "request body timed out"})
                return
            try:
                if control:
                    run_id = control[1]
                    if control[2] == "next":
                        store.approve(run_id, request["stage"])
                    else:
                        store.cancel(run_id)
                else:
                    run_id = store.submit(request)
            except KeyError:
                self._reply(404, {"error": "run not found"})
                return
            except StageConflictError as exc:
                self._reply(409, {"error": str(exc)})
                return
            except RunLimitError as exc:
                self._reply(429, {"error": str(exc)})
                return
            self._reply(202, {"id": run_id} if control else {
                "id": run_id, "operations_url": f"/marketplace-operations-demo.html?run={run_id}"})

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
