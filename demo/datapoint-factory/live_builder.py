"""Live builder agent on Azure AI Foundry: one real build becomes one data point.

The agent receives main's exact builder brief (``build_waves_v3.instructions``)
and acts only through four tools. Every chat request's reported usage is
recorded as a token event. The app then re-runs tests and CLI itself before
accepting the build.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from token_yield import build_waves_v3 as v3

import foundry
from sandbox import FILES, Workspace

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite one deliverable file.",
        "parameters": {"type": "object", "required": ["name", "content"], "properties": {
            "name": {"type": "string", "enum": list(FILES)}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read one deliverable file.",
        "parameters": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string", "enum": list(FILES)}}}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run python -B -m unittest discover -s . -p test_implementation.py",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_cli", "description": "Run python -B implementation.py example_input.json",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Finish with the short final report once tests and CLI pass.",
        "parameters": {"type": "object", "required": ["report"], "properties": {"report": {"type": "string"}}}}},
]
PREAMBLE = ("You act only through the provided tools: write_file, read_file, run_tests, run_cli and finish. "
            "The working directory already exists; tool calls operate inside it. Call finish when done.")
NUDGE = "Continue using the tools. Call finish only after run_tests and run_cli both pass."
MAX_REQUESTS = 30
MAX_TOKENS = 900_000


def _tail(result: dict) -> str:
    return json.dumps({"exit_code": result["exit_code"], "output": (result["stdout"] + result["stderr"])[-3000:]})


class LiveBuild:
    def __init__(self, parts, industry, deployment, config, root: Path, finalize):
        self.id = uuid.uuid4().hex[:12]
        self.parts, self.industry, self.deployment, self.config = parts, industry, deployment, config
        self.spec = v3.build_spec_v3(parts, industry)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.point_id = f"live_{stamp}_{self.spec['id']}"
        self.workspace = Workspace(root / self.point_id)
        self.brief = v3.instructions({"parts": parts, "industry": industry, "trial_id": self.point_id},
                                     str(self.workspace.directory))
        self.finalize = finalize
        self.events: list[dict] = []
        self.usage: list[dict] = []
        self.status, self.result, self.error = "running", None, None

    def emit(self, kind: str, **data) -> None:
        self.events.append({"type": kind, "at": time.time(), **data})

    def call_tool(self, name: str, arguments: dict) -> tuple[str, str]:
        if name == "write_file":
            text = self.workspace.write(arguments.get("name", ""), arguments.get("content", ""))
            return text, text
        if name == "read_file":
            return self.workspace.read(arguments.get("name", "")), f"read {arguments.get('name')}"
        if name in ("run_tests", "run_cli"):
            result = getattr(self.workspace, name)()
            lines = (result["stdout"] + result["stderr"]).strip().splitlines()
            return _tail(result), f"exit {result['exit_code']} · {lines[-1][:80] if lines else ''}"
        if name == "finish":
            return "finished", arguments.get("report", "")[:300]
        return f"error: unknown tool {name}", "unknown tool"

    def loop(self) -> str | None:
        api = foundry.client(self.config)
        messages = [{"role": "system", "content": PREAMBLE}, {"role": "user", "content": self.brief}]
        for index in range(1, MAX_REQUESTS + 1):
            message, usage = foundry.complete(api, self.deployment, messages, TOOLS)
            self.usage.append(usage)
            calls = message.tool_calls or []
            self.emit("request", index=index, usage=usage, tools=[c.function.name for c in calls])
            messages.append(message.model_dump(exclude_none=True))
            if not calls:
                messages.append({"role": "user", "content": NUDGE})
                continue
            report = None
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                content, summary = self.call_tool(call.function.name, arguments)
                self.emit("tool", index=index, name=call.function.name, summary=summary)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                if call.function.name == "finish":
                    report = arguments.get("report", "")
            if report is not None:
                return report
            if sum(u["input_tokens"] + u["output_tokens"] for u in self.usage) > MAX_TOKENS:
                self.emit("limit", reason=f"stopped at the {MAX_TOKENS:,}-token safety cap")
                return None
        self.emit("limit", reason=f"stopped after {MAX_REQUESTS} requests")
        return None

    def point(self, report, evidence) -> dict:
        totals = {k: sum(u[k] for u in self.usage)
                  for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        return {
            **self.spec, "id": self.point_id, "origin": "live_foundry_demo_build",
            "status": "measured_build_passed" if evidence["accepted"] else "live_build_failed_verification",
            "split": None,
            "builder_instructions_version": v3.BUILDER_TEMPLATE_VERSION,
            "builder_instructions_sha256": hashlib.sha256(self.brief.encode("utf-8")).hexdigest(),
            "build_token_usage": {
                "source": "azure_foundry_chat_completions_usage", "endpoint": self.config.openai_endpoint,
                "project_endpoint": self.config.project_endpoint, "builder_agent_id": self.id,
                "builder_models": sorted({u["model"] for u in self.usage}) or [self.deployment],
                "deployment": self.deployment, "provider_call_count": len(self.usage), **totals,
                "total_tokens": totals["input_tokens"] + totals["output_tokens"],
                "input_includes_cache_read_and_write": True,
                "measurement_scope": "Entire live builder conversation: brief, tool results, code, tests, repairs and final report.",
                "events": self.usage,
            },
            "build_evidence": {**evidence, "builder_report": report},
            "training_use": "Not pooled with the corpus: different builder model and agent harness.",
        }

    def run(self) -> None:
        try:
            self.emit("status", text=f"{self.deployment} builder started")
            report = self.loop()
            self.emit("status", text="independent runtime verification")
            evidence = self.workspace.verify()
            point = self.point(report, evidence)
            (self.workspace.directory / "data_point.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
            self.result = self.finalize(self.parts, self.industry, point, self.workspace.files(), self.brief)
            self.status = "done"
        except Exception as error:  # surfaced to the browser; the build is not silently replaced
            self.error, self.status = f"{type(error).__name__}: {error}", "failed"


class LiveJobs:
    """At most one live build at a time to bound spend."""

    def __init__(self, config, root: Path, finalize):
        self.config, self.root, self.finalize = config, root, finalize
        self.jobs: dict[str, LiveBuild] = {}
        self.lock = threading.Lock()

    def start(self, parts, industry, deployment) -> LiveBuild:
        if deployment not in self.config.deployments:
            raise ValueError(f"Unknown deployment {deployment}")
        with self.lock:
            if any(job.status == "running" for job in self.jobs.values()):
                raise ValueError("A live build is already running")
            job = LiveBuild(parts, industry, deployment, self.config, self.root, self.finalize)
            self.jobs[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        return job

    def poll(self, job_id: str, since: int) -> dict:
        job = self.jobs[job_id]
        return {"status": job.status, "events": job.events[since:], "next": len(job.events),
                "result": job.result, "error": job.error, "point_id": job.point_id}
