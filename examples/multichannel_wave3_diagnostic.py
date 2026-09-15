"""Print the offline Wave 3 repair-block multichannel diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from token_yield.multichannel_workflow import fit_wave3_repair_diagnostic


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    diagnostic = fit_wave3_repair_diagnostic(
        root / "runs" / "20260827_1152_wave3",
        root / "experiments" / "foundry_wave3",
    )
    print(json.dumps(diagnostic.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
