"""Prepare public-request scoping plans; execute only with explicit --execute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_yield.customer_pilot import (
    execute_campaign, prepare_campaign, prepare_prospective_campaign, train_project_forecast,
)


ROOT = Path(__file__).resolve().parents[1]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=ROOT / "experiments" / "customer_requests")
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="Spend API credits within the frozen approved cap")
    mode.add_argument("--train-from-run", type=Path,
                      help="Train project-level forecasts offline from a completed measured run")
    parser.add_argument("--test-model", type=Path,
                        help="Frozen project model run; never refit or select on new outcomes")
    parser.add_argument("--catalog", type=Path,
                        help="New all-holdout customer-requests-v1 catalog for --test-model")
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    if bool(args.test_model) != bool(args.catalog):
        parser.error("--test-model and --catalog must be supplied together")
    if args.test_model is not None and args.train_from_run is not None:
        parser.error("--test-model cannot be combined with --train-from-run")
    if args.train_from_run is not None:
        result = train_project_forecast(args.train_from_run, args.run_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    campaign = (prepare_prospective_campaign(args.test_model, args.catalog)
                if args.test_model is not None else prepare_campaign(args.experiment_dir))
    if args.execute:
        result = execute_campaign(campaign, args.run_dir)
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "preview.json").write_text(
            json.dumps(campaign, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result = {
            "status": "offline_preview", "projects": len(campaign["plans"]),
            "planned_calls": len(campaign["calls"]),
            "train_calls": sum(call["split"] == "train" for call in campaign["calls"]),
            "holdout_calls": sum(call["split"] == "holdout" for call in campaign["calls"]),
            "candidate_layouts_per_project": len(campaign["plans"][0]["candidates"]),
            "preview": str(args.run_dir / "preview.json"), "spent_usd": 0,
        }
        if args.test_model is not None:
            result.update(
                model_sha256=campaign["prospective"]["model_sha256"],
                catalog_sha256=campaign["catalog_sha256"],
                planned_observations=len(campaign["prospective"]["predictions"]),
                cap_usd=campaign["config"]["cap_usd"], stop_usd=campaign["config"]["stop_usd"],
                runtime_compatibility=campaign["prospective"]["compatibility"],
            )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
