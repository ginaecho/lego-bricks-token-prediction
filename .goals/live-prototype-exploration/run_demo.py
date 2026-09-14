"""Run the prototype's sustainability-report demo request live, end to end.

This script is an *exploration artifact*: it lives under `.goals/` and only
imports existing, unmodified `token_yield` methods. It performs the exact
chain the frozen prototype snapshot promises:

    plain-English request
        -> quote  (decompose + compose, pre-run estimate)
        -> run    (FoundryDispatcher, live Azure Foundry call)
        -> invoice (measured usage vs. quote, reconciliation)

Credentials are never written to disk: the bearer token is acquired in memory
via `token_yield.foundry_dispatch.acquire_entra_token` (Azure CLI / Entra
environment) and is discarded after the process exits. Only auditable
metadata (hashes, token counts, latency, status) is persisted as evidence.

Usage:
    python .goals/live-prototype-exploration/run_demo.py --endpoint <url> --deployment gpt-5-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from token_yield.compose import default_runs_path, load_runs, select_model  # noqa: E402
from token_yield.decompose import (  # noqa: E402
    explain,
    heuristic_decompose,
    price,
)
from token_yield.foundry_dispatch import (  # noqa: E402
    AuthenticationError,
    FoundryDispatchError,
    FoundryDispatcher,
    acquire_entra_token,
)
from token_yield.tasks import ORDER  # noqa: E402

DEMO_REQUEST = (
    "Compare the latest sustainability reports from three suppliers, extract "
    "their Scope 1 emissions and renewable-energy targets, validate the "
    "figures against the reporting checklist, and draft a recommendation "
    "for procurement."
)

# Manual encoding of the demo request into the fixed brick vocabulary. The
# prototype's UI performs this step with an agent encoder; this script uses
# the same vocabulary as `token_yield.tasks.ORDER` but supplies the encoding
# directly, so the live run is not gated on an additional agent call. This is
# recorded as `source: "manual"`, distinct from `"agent"` or `"heuristic"`.
DEMO_ENCODING = {"review": 1, "extract": 2, "validate": 1, "draft": 1}
DEMO_CONTEXT_BYTES = 31_800  # matches the prototype's "31.8 KB" source hint


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_quote(evidence_dir: Path) -> dict:
    """Pre-run quote: existing compose/decompose methods, no network call."""
    runs = load_runs(default_runs_path())
    model = select_model(runs)
    counts = {s: 0 for s in ORDER}
    counts.update(DEMO_ENCODING)
    from token_yield.decompose import Decomposition

    decomposition = Decomposition(
        counts,
        rationale="manual encoding for live-prototype-exploration demo request",
        context_bytes=DEMO_CONTEXT_BYTES,
        source="manual",
    )
    predicted_tokens = price(decomposition, model)
    heuristic = heuristic_decompose(DEMO_REQUEST, DEMO_CONTEXT_BYTES)
    quote = {
        "evidence_class": "pipeline_only",
        "generated_at": utc_now(),
        "request": DEMO_REQUEST,
        "request_sha256": sha256_text(DEMO_REQUEST),
        "encoding": decomposition.counts,
        "encoding_source": decomposition.source,
        "encoding_notation": decomposition.notation(),
        "heuristic_encoding": heuristic.counts,
        "heuristic_notation": heuristic.notation(),
        "context_bytes": DEMO_CONTEXT_BYTES,
        "model_form": model.form,
        "model_equation": model.equation(),
        "model_fitted_on_runs": model.n,
        "model_loo_mape": model.loo_mape,
        "predicted_tokens": predicted_tokens,
        "explain": explain(decomposition, model),
    }
    (evidence_dir / "quote.json").write_text(
        json.dumps(quote, indent=2), encoding="utf-8"
    )
    return quote


def run_live(
    endpoint: str, deployment: str, evidence_dir: Path
) -> dict:
    """Live run through the existing FoundryDispatcher, or an explicit failure record."""
    record: dict = {
        "evidence_class": "live_feasibility_attempt",
        "generated_at": utc_now(),
        "endpoint": endpoint,
        "deployment": deployment,
    }
    try:
        # Acquire the bearer token in memory only; never persisted to disk.
        token = acquire_entra_token()
    except AuthenticationError as exc:
        record.update(
            {
                "status": "failed",
                "failure_stage": "authentication",
                "failure_reason": str(exc),
            }
        )
        (evidence_dir / "run.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        return record

    dispatcher = FoundryDispatcher(
        endpoint=endpoint,
        model=deployment,
        token_provider=lambda: token,
        max_output_tokens=512,
        reasoning_effort="minimal",
        text_verbosity="low",
    )
    prompt = (
        "Read this request and reply with a one-sentence acknowledgement "
        "only, no analysis: " + DEMO_REQUEST
    )
    started = time.perf_counter()
    try:
        result = dispatcher.dispatch(prompt, target="live_prototype_demo")
    except FoundryDispatchError as exc:
        record.update(
            {
                "status": "failed",
                "failure_stage": "dispatch",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        )
        (evidence_dir / "run.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        return record

    record.update(
        {
            "status": "completed",
            "response_id": result.response_id,
            "response_status": result.status,
            "model_echoed": result.model,
            "elapsed_ms": result.elapsed_ms,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "reasoning_tokens": result.usage.reasoning_tokens,
                "cached_tokens": result.usage.cached_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "response_calls": [
                {
                    "target": c.target,
                    "response_id": c.response_id,
                    "status": c.status,
                    "model": c.model,
                    "elapsed_ms": c.elapsed_ms,
                    "request_sha256": c.request_sha256,
                    "response_sha256": c.response_sha256,
                }
                for c in result.response_calls
            ],
            "output_text_sha256": sha256_text(result.output),
        }
    )
    (evidence_dir / "run.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def build_invoice(quote: dict, run: dict, evidence_dir: Path) -> dict:
    """Reconcile predicted vs. measured usage, or record why it can't reconcile."""
    invoice: dict = {
        "generated_at": utc_now(),
        "predicted_tokens": quote["predicted_tokens"],
    }
    if run.get("status") != "completed":
        invoice.update(
            {
                "evidence_class": "failure_exclusion",
                "reconciled": False,
                "exclusion_reason": (
                    f"live run failed at stage '{run.get('failure_stage')}': "
                    f"{run.get('failure_reason')}"
                ),
                "note": (
                    "No actual usage is available. This invoice is NOT a "
                    "measured feasibility result and must not be treated as one."
                ),
            }
        )
    else:
        actual = run["usage"]["total_tokens"]
        predicted = quote["predicted_tokens"]
        error = abs(actual - predicted) / actual if actual else None
        invoice.update(
            {
                "evidence_class": "live_feasibility_result",
                "reconciled": True,
                "actual_tokens": actual,
                "actual_usage_breakdown": run["usage"],
                "reconstruction_error": error,
                "note": (
                    "NOTE: the acknowledgement-only probe prompt does not "
                    "perform the demo's full review/extract/validate/draft "
                    "work, so this reconciliation is between the quote (for "
                    "the full task) and usage for a minimal live probe of "
                    "the same runtime/deployment/auth path. It demonstrates "
                    "live feasibility, not that the quote priced this exact "
                    "call."
                ),
            }
        )
    (evidence_dir / "invoice.json").write_text(
        json.dumps(invoice, indent=2), encoding="utf-8"
    )
    return invoice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", default="gpt-5-mini")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / ".goals" / "live-prototype-exploration" / "evidence",
    )
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    quote = build_quote(args.evidence_dir)
    run = run_live(args.endpoint, args.deployment, args.evidence_dir)
    invoice = build_invoice(quote, run, args.evidence_dir)

    print(json.dumps({"quote": quote, "run": run, "invoice": invoice}, indent=2))


if __name__ == "__main__":
    main()
