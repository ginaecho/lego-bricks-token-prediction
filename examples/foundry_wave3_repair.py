"""Run or inspect the frozen Microsoft Foundry wave-3 repair block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_yield.wave3_repair_runner import run_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-cases", type=int)
    parser.add_argument("--continue-after-errors", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_repair(
        root / "experiments" / "foundry_wave3",
        args.run_dir,
        args.endpoint,
        dry_run=args.dry_run,
        max_new_cases=args.max_new_cases,
        continue_after_errors=args.continue_after_errors,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
