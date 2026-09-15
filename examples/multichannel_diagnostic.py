"""Run the leak-free, historical-only Wave 2 channel diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from token_yield.multichannel_workflow import (
    fit_wave2_historical_diagnostic,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    diagnostic = fit_wave2_historical_diagnostic(
        root / "runs" / "20260826_1627_wave2",
        root / "experiments" / "foundry_wave2",
    )
    print(json.dumps(diagnostic.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
