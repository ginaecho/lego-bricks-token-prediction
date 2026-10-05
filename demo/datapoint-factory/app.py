"""Datapoint Factory: local demo of how one brick composition becomes a training data point.

Run from the repository root:
    python demo/datapoint-factory/app.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import source_tree  # noqa: E402
import studio  # noqa: E402
from foundry import OPENAI_ENDPOINT, PROJECT_ENDPOINT, SUBSCRIPTION_ID, TENANT_ID, FoundryConfig  # noqa: E402
from studio import StudioConfig  # noqa: E402

STATIC = HERE / "static"
LIVE_RUNS = HERE / "live_runs"
STUDIO_STATE = HERE / ".studio-runs"
MAX_BODY = 10_000


class State:
    corpus = None
    pipeline = None
    trainer = None
    intents = None
    live = None
    studio = None
    foundry = FoundryConfig()
    provenance = ""


def load(ref: str, source_root: Path | None, cache: Path, config: FoundryConfig) -> None:
    root, State.provenance = source_tree.prepare(ref, source_root, cache)
    from corpus import Corpus
    from intents import brick_intents
    from live_builder import LiveJobs
    from pipeline import Pipeline
    from training import Trainer
    State.foundry = config
    State.corpus = Corpus(root)
    State.intents = brick_intents(State.corpus)
    State.trainer = Trainer(root)
    State.pipeline = Pipeline(root, State.corpus, State.trainer)
    State.live = LiveJobs(config, LIVE_RUNS, lambda parts, industry, point, files, brief: State.pipeline.run(
        parts, industry, live=point, live_files=files, brief=brief))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, *_):
        pass

    def send_json(self, value, status=200):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        routes = {
            "/api/catalog": lambda: {**State.corpus.catalog(), "intents": State.intents,
                                     "provenance": State.provenance, "foundry": State.foundry.public(),
                                     "studio": {"url": State.studio.url, "foundry": State.studio.enable_foundry,
                                                "budget_usd": State.studio.budget_usd} if State.studio else None,
                                     "measured_builds": len(State.corpus.measured)},
            "/api/corpus": lambda: {"builds": State.corpus.overview()},
            "/api/training": lambda: State.trainer.summary(),
        }
        if url.path in routes:
            return self.send_json(routes[url.path]())
        if url.path.startswith("/api/live/"):
            since = int(parse_qs(url.query).get("since", ["0"])[0])
            try:
                return self.send_json(State.live.poll(url.path.rsplit("/", 1)[1], since))
            except KeyError:
                return self.send_json({"error": "unknown live build"}, 404)
        return super().do_GET()

    def do_POST(self):
        try:
            request = self.read_json()
            parts, industry = list(request["parts"]), request.get("industry") or None
            if self.path == "/api/run":
                return self.send_json(State.pipeline.run(parts, industry, request.get("build_id")))
            if self.path == "/api/brief":
                return self.send_json({"prompt": State.pipeline.brief(parts, industry)})
            if self.path == "/api/live":
                job = State.live.start(parts, industry, request["deployment"])
                return self.send_json({"job_id": job.id, "point_id": job.point_id})
        except (KeyError, TypeError, ValueError) as error:
            return self.send_json({"error": str(error)}, 400)
        return self.send_json({"error": "not found"}, 404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="main", help="git ref holding data/build_simulations (default: main)")
    parser.add_argument("--source-root", type=Path, help="use an existing checkout instead of exporting --ref")
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    parser.add_argument("--foundry-project-endpoint", default=PROJECT_ENDPOINT)
    parser.add_argument("--azure-openai-endpoint", default=OPENAI_ENDPOINT)
    parser.add_argument("--deployments", default="gpt-5-mini,gpt-5.4,gpt-5.6-sol",
                        help="comma-separated Foundry deployments offered for live builds (first is default)")
    parser.add_argument("--azure-tenant-id", default=TENANT_ID)
    parser.add_argument("--azure-subscription-id", default=SUBSCRIPTION_ID)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-studio", action="store_true", help="do not launch the Token Yield Studio marketplace")
    parser.add_argument("--studio-port", type=int, default=StudioConfig.port)
    parser.add_argument("--studio-offline", action="store_true", help="launch the Studio without paid Foundry calls")
    parser.add_argument("--studio-budget-usd", type=float, default=10.0,
                        help="Studio campaign cap in USD (max 25), shared across restarts")
    parser.add_argument("--studio-approval-id", default=StudioConfig.approval_id)
    args = parser.parse_args()
    config = FoundryConfig(args.foundry_project_endpoint, args.azure_openai_endpoint,
                           tuple(name.strip() for name in args.deployments.split(",") if name.strip()),
                           args.azure_tenant_id, args.azure_subscription_id)
    account = studio.ensure_az_account(config.subscription_id, config.tenant_id)
    print(f"Azure account: {account}", flush=True)
    print("Loading measured corpus and fitting the model (first start can take ~30 s)...", flush=True)
    load(args.ref, args.source_root, args.cache, config)
    print(f"Loaded {len(State.corpus.measured)} measured builds from {State.provenance}", flush=True)
    print(f"Live builds: {config.openai_endpoint} ({', '.join(config.deployments)})", flush=True)
    process = None
    if not args.no_studio:
        State.studio = studio.StudioConfig(args.studio_port, args.studio_budget_usd, args.studio_approval_id,
                                           not args.studio_offline)
        root = studio.export_main(args.ref, args.cache)
        process = studio.start(root, State.studio, STUDIO_STATE, STUDIO_STATE / "studio.log")
        mode = f"Foundry enabled, cap ${State.studio.budget_usd:g}" if State.studio.enable_foundry else "offline"
        print(f"Token Yield Studio at {State.studio.url} ({mode}; log {STUDIO_STATE / 'studio.log'})", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Datapoint Factory at {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    finally:
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    main()
