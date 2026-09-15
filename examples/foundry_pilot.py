"""Execute or inspect the preregistered 30-session Foundry wave-2 pilot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from token_yield.pilot_runner import run_pilot


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "foundry_wave2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-cases", type=int)
    parser.add_argument("--continue-after-errors", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir or (
        ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M}_wave2"
    )
    result = run_pilot(
        EXPERIMENT,
        run_dir,
        args.endpoint,
        dry_run=args.dry_run,
        max_new_cases=args.max_new_cases,
        continue_after_errors=args.continue_after_errors,
    )
    print(json.dumps({"run_dir": str(run_dir), **result}, indent=2, default=str))


if __name__ == "__main__":
    main()
