"""Serve the bounded marketplace demo on loopback, offline unless explicitly enabled.

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
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from token_yield.marketplace_demo import STAGES, PipelineCancelled, run_pipeline, validate_request, write_json

if TYPE_CHECKING:
    from token_yield.marketplace_agents import AgentRuntime

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROUTES = frozenset({
    "/marketplace-sales-demo.html", "/marketplace-operations-demo.html",
    "/marketplace-prototype.html",
    "/marketplace-console.html",
})
MAX_BODY = 32768


def _now_server() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLimitError(Exception):
    """The local demo's work or retention bound has been reached."""


class StageConflictError(Exception):
    """A stale, duplicate, or terminal-run control request was rejected."""


class RuntimeUnavailableError(Exception):
    """Paid execution was requested without a configured, approved runtime."""


def validate_run_request(value: Any) -> dict[str, Any]:
    """Keep the offline contract unchanged while explicitly selecting a runtime."""
    if not isinstance(value, dict):
        raise ValueError("body must be a JSON object")
    original = dict(value)
    runtime = original.pop("runtime", "offline")
    if not isinstance(runtime, str) or runtime not in ("offline", "foundry"):
        raise ValueError("runtime must be offline or foundry")
    result = validate_request(original)
    if runtime == "foundry" and result["model_id"] != "gpt":
        raise ValueError("Foundry runs require model_id gpt, the pinned deployment alias")
    if "runtime" in value:
        result["runtime"] = runtime
    return result


class RunStore:
    """Lock-protected snapshots with append-only worker events."""

    def __init__(self, run_dir: Path, *, max_running: int = 2, max_runs: int = 32,
                 stage_timeout: float = 1800, agent_runtime: AgentRuntime | None = None):
        self.run_dir = Path(run_dir)
        self.max_running = max_running
        self.max_runs = max_runs
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stage_timeout = stage_timeout
        self.agent_runtime = agent_runtime
        self.runs: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()
        self._approved: dict[str, str] = {}
        self._establishment_decisions: dict[str, dict[str, Any]] = {}
        self._cancelled: set[str] = set()
        self._finalizing: set[str] = set()
        self.previous_run_count = (
            sum(1 for _ in self.run_dir.glob("*/request.json")) if self.run_dir.exists() else 0
        )

    def submit(self, request: dict[str, Any]) -> str:
        request = validate_run_request(request)
        with self.lock:
            live = request.get("runtime") == "foundry"
            if live and self.agent_runtime is None:
                raise RuntimeUnavailableError("Foundry execution is disabled. Start the server with an approved campaign budget.")
            if live and any(self.runs[active]["request"].get("runtime") == "foundry"
                            for active in self._active):
                raise RunLimitError("A Foundry run is already active. Finish or cancel it before starting another.")
            if len(self._active) >= self.max_running or self.previous_run_count + len(self.runs) >= self.max_runs:
                raise RunLimitError("Demo run limit reached; wait for active work or restart with a new run directory.")
            if live:
                from token_yield.marketplace_agents import AGENT_STAGES

                stages = AGENT_STAGES
            else:
                stages = STAGES
            run_id = uuid.uuid4().hex
            self.runs[run_id] = {"id": run_id, "status": "queued", "request": request,
                                 "events": [], "result": None, "error": None, "next_stage": None,
                                 "stages": list(stages)}
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

    def decide_establishment(self, run_id: str, decision: dict[str, Any]) -> None:
        """Approve or reject exactly the pending new-brick contract hash."""
        with self.condition:
            run = self.runs[run_id]
            pending = run.get("establishment_request")
            if (run["status"] != "waiting" or run.get("next_stage") != "human_establishment_approval"
                    or not isinstance(pending, dict)):
                raise StageConflictError("No pending establishment approval for this run.")
            if (not isinstance(decision, dict)
                    or set(decision) - {"decision", "contract_hash", "actor", "reason"}
                    or decision.get("decision") not in ("approve", "reject")
                    or decision.get("contract_hash") != pending.get("contract_hash")
                    or not isinstance(decision.get("actor"), str)
                    or not decision["actor"].strip()
                    or not isinstance(decision.get("reason", ""), str)):
                raise ValueError("establishment decision requires approve/reject, actor, and exact contract_hash")
            record = {
                "decision": decision["decision"], "contract_hash": decision["contract_hash"],
                "actor": decision["actor"].strip(), "reason": decision.get("reason", "").strip(),
                "timestamp": _now_server(),
            }
            self._establishment_decisions[run_id] = record
            run.update(status="running", next_stage=None, establishment_decision=record)
            write_json(self.run_dir / run_id / "run.json", run)
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

    def _establishment_decision(self, run_id: str, request: dict[str, Any]) -> dict[str, Any]:
        with self.condition:
            if run_id in self._cancelled:
                raise PipelineCancelled("Cancelled before human establishment approval.")
            run = self.runs[run_id]
            run.update(status="waiting", next_stage="human_establishment_approval",
                       establishment_request=copy.deepcopy(request))
            write_json(self.run_dir / run_id / "run.json", run)
            self.condition.notify_all()
            approved = self.condition.wait_for(
                lambda: run_id in self._cancelled or run_id in self._establishment_decisions,
                timeout=self.stage_timeout,
            )
            if run_id in self._cancelled:
                raise PipelineCancelled("Cancelled while waiting for human establishment approval.")
            if not approved:
                run.update(status="failed", next_stage=None)
                raise TimeoutError("Human establishment approval timed out.")
            return self._establishment_decisions.pop(run_id)

    def _event(self, run_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            self.runs[run_id]["events"].append(copy.deepcopy(event))
            print(json.dumps({"run_id": run_id, **event}, ensure_ascii=True), flush=True)

    def _check_cancel(self, run_id: str) -> None:
        with self.lock:
            if run_id in self._cancelled:
                raise PipelineCancelled("Cancelled before the next paid request or tool.")

    def _worker(self, run_id: str) -> None:
        try:
            with self.lock:
                if run_id not in self._cancelled:
                    self.runs[run_id]["status"] = "running"
                request = self.runs[run_id]["request"].copy()
            callbacks = {
                "run_id": run_id,
                "on_event": lambda event: self._event(run_id, event),
                "before_stage": lambda stage: self._before_stage(run_id, stage),
            }
            if request.get("runtime") == "foundry":
                if self.agent_runtime is None:
                    raise RuntimeUnavailableError("Foundry runtime is no longer available")
                result = self.agent_runtime.run_pipeline(
                    request, self.run_dir, **callbacks,
                    check_cancel=lambda: self._check_cancel(run_id),
                    establishment_decision=lambda pending: self._establishment_decision(run_id, pending),
                )
            else:
                request.pop("runtime", None)
                result = run_pipeline(request, self.run_dir, **callbacks)
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
            current = self.runs.get(run_id)
            if current is not None:
                return {**copy.deepcopy(current), "replay": False}
            if not re.fullmatch(r"[0-9a-f]{32}", run_id):
                return None
            saved = self.run_dir / run_id / "run.json"
            if not saved.is_file():
                return None
            run = json.loads(saved.read_text(encoding="utf-8"))
            if run.get("id") != run_id:
                raise ValueError("Persisted run identity does not match its directory")
            if run["status"] in ("queued", "waiting", "running"):
                run.update(status="failed", next_stage=None, result=None,
                           error="Server restarted during execution. This historical run cannot resume; any uncertain charges remain reserved.")
            return {**run, "replay": True}

    def runtime_status(self) -> dict[str, Any]:
        if self.agent_runtime is None:
            return {"enabled": False, "source": "synthetic",
                    "message": "Offline only. No paid calls are enabled."}
        return self.agent_runtime.public_status()

    def catalog(self) -> dict[str, Any]:
        if self.agent_runtime is None:
            return {"items": [], "enabled": False,
                    "message": "No Foundry measurement campaign is enabled."}
        return self.agent_runtime.catalog()

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
            elif path == "/api/runtime":
                self._reply(200, store.runtime_status())
            elif path == "/api/catalog":
                self._reply(200, store.catalog())
            elif path == "/api/runs":
                self._reply(200, store.listing())
            elif path == "/api/scenarios":
                from token_yield.marketplace_scenarios import SCENARIOS

                fixture = store.agent_runtime.source_fixture if store.agent_runtime else None
                self._reply(200, {"source": "fictional-policy-fixture" if fixture else "invented-public-metadata-inspired",
                                  "scenarios": [fixture] if fixture else SCENARIOS})
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
            control = re.fullmatch(r"/api/runs/([0-9a-f]{32})/(next|cancel|establishment)", self.path)
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
                    elif control[2] == "establishment":
                        if (set(request) - {"decision", "contract_hash", "actor", "reason"}
                                or request.get("decision") not in ("approve", "reject")
                                or not isinstance(request.get("contract_hash"), str)
                                or not isinstance(request.get("actor"), str)
                                or not request["actor"].strip()
                                or not isinstance(request.get("reason", ""), str)):
                            raise ValueError("establishment requires decision, contract_hash and actor")
                    elif request:
                        raise ValueError("cancel requires an empty object")
                else:
                    request = validate_run_request(request)
            except (ValueError, UnicodeError, RecursionError):
                self._reply(400, {"error": "Invalid JSON or request fields. Runs require description, model_id, integer runs_per_month; optional new_function, execution_mode step|automatic and runtime offline|foundry. Foundry uses the pinned gpt alias. Next requires {stage:string}; cancel requires {}."})
                return
            except TimeoutError:
                self._reply(408, {"error": "request body timed out"})
                return
            try:
                if control:
                    run_id = control[1]
                    if control[2] == "next":
                        store.approve(run_id, request["stage"])
                    elif control[2] == "establishment":
                        store.decide_establishment(run_id, request)
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
            except RuntimeUnavailableError as exc:
                self._reply(409, {"error": str(exc)})
                return
            except ValueError as exc:
                self._reply(400, {"error": str(exc)})
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
    parser.add_argument("--enable-foundry", action="store_true",
                        help="Enable paid calls using the existing pinned Azure deployment.")
    parser.add_argument("--agent-config", type=Path,
                        default=ROOT / "experiments" / "customer_requests" / "pilot.json")
    parser.add_argument("--agent-state-dir", type=Path,
                        help="Durable campaign budget and registry directory; reuse it on restart.")
    parser.add_argument("--agent-budget-usd", type=float,
                        help="New approved total campaign cap (maximum USD 25), not a per-run cap.")
    parser.add_argument("--agent-approval-id",
                        help="Explicit new campaign approval identifier; previous pilot approval is not reused.")
    parser.add_argument("--campaign-file", type=Path,
                        help="Reviewed campaign JSON; execution_enabled must be true after independent verification.")
    parser.add_argument("--mock-agents", action="store_true",
                        help="Use explicit synthetic provider fixtures; no authentication or network calls.")
    parser.add_argument("--source-fixture", choices=["archive-exceptions-v2"],
                        help="Fictional source version; mock-only unless a separately approved --scope-file is supplied.")
    parser.add_argument("--scope-file", type=Path,
                        help="Separately reviewed and user-approved v2 scope under the existing campaign ledger.")
    parser.add_argument("--measurement-policy", action="store_true",
                        help="Offline reward-based measurement bandit; requires --mock-agents.")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if not 1 <= args.port <= 65535:
        print("--port must be in 1..65535", file=sys.stderr)
        return 2
    if args.mock_agents and (args.enable_foundry or args.campaign_file):
        print("--mock-agents cannot be combined with paid campaign flags", file=sys.stderr)
        return 2
    if args.scope_file and (args.mock_agents or not args.enable_foundry or not args.campaign_file
                           or not args.source_fixture or not args.agent_state_dir):
        print("--scope-file requires real v2 mode, original campaign and explicit existing agent state",
              file=sys.stderr)
        return 2
    if args.source_fixture and not args.mock_agents and not args.scope_file:
        print("--source-fixture requires --mock-agents; no paid execution is approved", file=sys.stderr)
        return 2
    if args.measurement_policy and not args.mock_agents:
        print("--measurement-policy requires --mock-agents; no paid policy approval exists", file=sys.stderr)
        return 2
    if args.campaign_file and not args.enable_foundry:
        print("--campaign-file requires explicit --enable-foundry", file=sys.stderr)
        return 2
    if args.enable_foundry and not args.campaign_file and (
            args.agent_budget_usd is None or not 0 < args.agent_budget_usd <= 25
            or not args.agent_approval_id or not args.agent_approval_id.strip()):
        print("--enable-foundry requires --agent-budget-usd in (0,25] and --agent-approval-id",
              file=sys.stderr)
        return 2
    runtime = None
    try:
        if args.enable_foundry:
            from token_yield.marketplace_agents import AgentRuntime

            runtime = AgentRuntime(
                args.agent_state_dir or args.run_dir / "agent-state",
                args.agent_config, cap_usd=args.agent_budget_usd or 25,
                approval_id=args.agent_approval_id or "marketplace-new-25usd-pilot",
                campaign=json.loads(args.campaign_file.read_text(encoding="utf-8")) if args.campaign_file else None,
                source_fixture=args.source_fixture,
                scope_file=args.scope_file,
            )
        elif args.mock_agents:
            from token_yield.marketplace_agents import APPROVED_ENDPOINT, MODEL_ID, AgentRuntime
            from token_yield.marketplace_mock import MockProvider

            args.run_dir.mkdir(parents=True, exist_ok=True)
            config = args.run_dir / "mock-connection.json"
            write_json(config, {
                "endpoint": APPROVED_ENDPOINT, "deployment": MODEL_ID, "expected_response_model": MODEL_ID,
                "deployment_version": "2026-03-05", "deployment_sku": "GlobalStandard",
                "reasoning_effort": "none", "text_verbosity": "low",
                "pricing": {"input_per_million": 2.5, "cached_input_per_million": .25, "output_per_million": 15.0},
            })
            runtime = AgentRuntime(args.run_dir / "mock-state", config,
                                   dispatch=MockProvider(delay=.015),
                                   source_fixture=args.source_fixture,
                                   measurement_policy=args.measurement_policy,
                                   token_provider=lambda: (_ for _ in ()).throw(AssertionError("No mock authentication")))
        store = RunStore(args.run_dir, agent_runtime=runtime)
        with make_server(args.port, store=store) as server:
            print(json.dumps({"stage": "server", "url": f"http://127.0.0.1:{args.port}/marketplace-sales-demo.html",
                              "foundry_enabled": runtime is not None,
                              "offline_available": True}), flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Server error: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
