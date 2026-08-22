"""Token Yield demo — predict token budget for a sample business project.

Run with: python -m examples.token_yield_demo
"""

from token_yield import (
    CalibrationStore,
    CalibrationRecord,
    ComplexityTier,
    ProjectSpec,
    ProjectForecaster,
    TokenPredictor,
)
from token_yield.report import text_report


def main() -> None:
    # ── Step 1: Calibrate from measured baselines ────────────────────────
    store = CalibrationStore()

    # Simulated measurements: "we ran 5 bug-fix tasks and recorded token usage"
    for tokens, dur in [(12_400, 180), (14_200, 210), (11_800, 165),
                        (13_600, 195), (15_000, 220)]:
        store.add(CalibrationRecord("bug_fix", tokens, duration_seconds=dur,
                                    harness_tokens=1200))

    for tokens, dur in [(28_000, 420), (32_000, 480), (25_500, 390),
                        (30_000, 450)]:
        store.add(CalibrationRecord("feature", tokens, duration_seconds=dur,
                                    harness_tokens=2400))

    for tokens, dur in [(18_000, 300), (20_000, 330), (19_000, 315)]:
        store.add(CalibrationRecord("refactor", tokens, duration_seconds=dur,
                                    harness_tokens=1800))

    for tokens, dur in [(4_500, 60), (5_000, 75), (4_200, 55),
                        (4_800, 65), (5_200, 80), (4_600, 62)]:
        store.add(CalibrationRecord("docs", tokens, duration_seconds=dur,
                                    harness_tokens=600))

    for tokens, dur in [(22_000, 360), (25_000, 400), (20_000, 340)]:
        store.add(CalibrationRecord("data_analysis", tokens, duration_seconds=dur,
                                    harness_tokens=2000))

    print("Calibration complete:")
    print(f"  {store.total_records()} records across {len(store.task_types)} task types")
    print()

    # ── Step 2: Explore predictions at different complexities ────────────
    predictor = TokenPredictor(store)

    print("Scaled predictions for 'bug_fix':")
    print(f"  {'Tier':<12} {'Tokens':>10} {'Confidence':>20}")
    for tier, pred in predictor.predict_scaled("bug_fix").items():
        print(f"  {tier.value:<12} {pred.predicted_tokens:>10,} "
              f"  [{pred.confidence_low:,} – {pred.confidence_high:,}]")
    print()

    # ── Step 3: Compare scenarios ────────────────────────────────────────
    scenarios = {
        "MVP (just bug fixes)": [
            ("bug_fix", ComplexityTier.BASE, 5),
        ],
        "Standard release": [
            ("bug_fix", ComplexityTier.BASE, 5),
            ("feature", ComplexityTier.BASE, 3),
            ("docs", ComplexityTier.BASE, 3),
        ],
        "Major release": [
            ("bug_fix", ComplexityTier.PLUS, 8),
            ("feature", ComplexityTier.PLUS, 5),
            ("refactor", ComplexityTier.BASE, 3),
            ("data_analysis", ComplexityTier.BASE, 2),
            ("docs", ComplexityTier.PLUS, 5),
        ],
    }

    print("Scenario comparison:")
    totals = predictor.compare_scenarios(scenarios)
    for name, tokens in sorted(totals.items(), key=lambda x: x[1]):
        print(f"  {name:<30} {tokens:>12,} tokens")
    print()

    # ── Step 4: Full project forecast ────────────────────────────────────
    spec = (ProjectSpec("Q3 Platform Upgrade", interaction_overhead=0.15)
            .add("bug_fix", ComplexityTier.PLUS, count=8)
            .add("feature", ComplexityTier.PLUS, count=5)
            .add("refactor", ComplexityTier.BASE, count=3)
            .add("data_analysis", ComplexityTier.BASE, count=2)
            .add("docs", ComplexityTier.PLUS, count=5))

    forecaster = ProjectForecaster(store)
    forecast = forecaster.forecast(spec)

    print(text_report(forecast, dollars_per_million_tokens=3.0))

    # ── Step 5: Cost dict for programmatic use ───────────────────────────
    cost_dict = forecaster.forecast_with_cost(spec, dollars_per_million_tokens=3.0)
    print("Programmatic output (dict):")
    print(f"  Total tokens:     {cost_dict['total_tokens']:,}")
    print(f"  Estimated cost:   ${cost_dict['estimated_cost']['estimated']}")
    print(f"  Cost range:       ${cost_dict['estimated_cost']['range_low']} – "
          f"${cost_dict['estimated_cost']['range_high']}")
    print(f"  Estimated hours:  {cost_dict['estimated_hours']}")


if __name__ == "__main__":
    main()
