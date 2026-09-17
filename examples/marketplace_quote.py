"""Save an offline marketplace quote under explicit recorded-runtime assumptions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from token_yield.customer_decomposition import SCOPING_BRICKS, canonical_json, content_hash
from token_yield.marketplace import quote_selection, rate_quote, service_catalog


ROOT = Path(__file__).resolve().parents[1]


def create_parser() -> argparse.ArgumentParser:
    """Require explicit service selection and acknowledgment of historical assumptions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path,
                        default=ROOT / "runs" / "20260914_customer_scoping_v2")
    parser.add_argument("--model-run", type=Path,
                        default=ROOT / "runs" / "20260916_marketplace_tokens_v1")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--services", nargs="+", choices=SCOPING_BRICKS, required=True)
    parser.add_argument("--execution-mode", choices=("separate", "batched"), required=True)
    parser.add_argument("--recorded-runtime", action="store_true", required=True,
                        help="Acknowledge this is conditional on the saved runtime, not a live check")
    parser.add_argument("--historical-rates", action="store_true", required=True,
                        help="Use recorded as-of rates, not a current price lookup")
    parser.add_argument("--tool-cost-usd", type=float)
    parser.add_argument("--human-review-cost-usd", type=float)
    parser.add_argument("--gross-margin-fraction", type=float)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    destination = args.run_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"quote output already exists: {destination}")
    for source in (args.source_run.resolve(), args.model_run.resolve()):
        if source in destination.parents:
            raise ValueError("quote output must be outside frozen source and model runs")
    artifact = json.loads((args.model_run / "models.json").read_text(encoding="utf-8"))
    raw = (args.source_run / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["provenance"]["source_file_sha256"]["protocol.json"]:
        raise ValueError("source protocol differs from model provenance")
    protocol = json.loads(raw)
    matches = [project for project in protocol["catalog"]["projects"] if project["id"] == args.project_id]
    if len(matches) != 1:
        raise ValueError("project-id must identify one project in the frozen source protocol")
    project = matches[0]
    template = protocol["template"]
    catalog = service_catalog(template, artifact)
    versions = {service["id"]: service["version"] for service in catalog["services"]}
    selections = [
        {"service_id": slug, "version": versions[slug],
         "quantity": 1 if slug == "report" else len(project["requirements"])}
        for slug in args.services
    ]
    quote = quote_selection(project, template, artifact, artifact["runtime"], selections,
                            execution_mode=args.execution_mode)
    config = protocol["config"]
    card = {
        "id": "recorded-scoping-retail-rates",
        "version": config["pricing_source"]["retrieved_on"],
        "currency": "USD", "runtime_sha256": content_hash(artifact["runtime"]),
        "as_of": config["pricing_source"]["retrieved_on"],
        "rates": config["pricing"], "source": config["pricing_source"],
    }
    rated = rate_quote(quote, card, tool_cost_usd=args.tool_cost_usd,
                       human_review_cost_usd=args.human_review_cost_usd,
                       gross_margin_fraction=args.gross_margin_fraction)
    summary = {
        "status": rated["status"], "project_id": project["id"],
        "runtime_basis": "conditional_on_recorded_execution_not_live_verified",
        "pricing_basis": "historical_as_of_rates_not_current_price_lookup",
        "quote_sha256": quote["quote_sha256"], "rated_quote_sha256": rated["rated_quote_sha256"],
        "point_estimate": quote["point_estimate"], "api_cost_usd": rated["api_cost_usd"],
        "proposed_selling_price_usd": rated["proposed_selling_price_usd"],
        "reasons": rated["reasons"], "spent_usd": 0,
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("catalog.json", catalog), ("selections.json", selections),
                        ("quote.json", quote), ("rated_quote.json", rated), ("analysis.json", summary)):
        (destination / name).write_text(canonical_json(value) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
