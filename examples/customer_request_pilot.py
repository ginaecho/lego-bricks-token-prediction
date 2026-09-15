"""Prepare public-request scoping plans; execute only with explicit --execute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_yield.customer_pilot import execute_campaign, prepare_campaign


ROOT = Path(__file__).resolve().parents[1]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=ROOT / "experiments" / "customer_requests")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Spend API credits within the frozen approved cap")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    campaign = prepare_campaign(args.experiment_dir)
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
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
