"""Train shared token predictors offline from an existing measured scoping run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_yield.customer_pilot import train_token_forecast


def create_parser() -> argparse.ArgumentParser:
    """Create an offline-only command with explicit source and destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    result = train_token_forecast(args.source_run, args.run_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
