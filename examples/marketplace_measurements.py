"""Preview offline by default; --execute spends the separately approved budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from token_yield.marketplace_measurements import execute_campaign, prepare_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Enable paid, single-use execution")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "20260916_marketplace_measurements_v1")
    parser.add_argument("--variant", help="Unsupported variants fail closed; no matrix subsetting")
    args = parser.parse_args()
    campaign = prepare_campaign(ROOT, variant=args.variant)
    if args.execute:
        result = execute_campaign(campaign, ROOT, args.run_dir, execute=True)
    else:
        manifest = campaign["manifest"]
        result = {
            "status": "offline_prepared", "spent_usd": 0,
            "manifest_sha256": campaign["manifest_sha256"],
            "preview": manifest["preview"], "authorization": manifest["authorization"],
            "grouping": manifest["grouping"],
            "unsupported_variants": manifest["variation_matrix"]["unsupported"],
            "live_preflight": "Existing Entra/CLI credentials and pinned deployment required. "
                              "No credentials or deployment probes used during preview.",
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
