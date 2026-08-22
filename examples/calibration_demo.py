"""Where the calibration data comes from, and how the model learns from it.

Run with: python -m examples.calibration_demo

Walks the whole loop:
  1. the measured probe suite — real subagent runs, not invented numbers
  2. fitting — the model *form* is chosen by cross-validation, not decreed
  3. validation — cross-validated error against the irreducible noise floor
  4. falsification — what the first version's hardcoded constants claimed
  5. out-of-sample — predicting the held-out composition experiment
  6. tune-back — new runs arrive, drift is detected, the model refits
"""

from token_yield.backtest import backtest, learning_curve, noise_floor
from token_yield.learn import LearningStore, seeded_store
from token_yield.plan import PlanForecaster, WorkPlan
from token_yield.probes import (
    COMPOSITION_MEASURED, MEASURED, PROBE_SUITE, composition_evidence,
    replicate_spread,
)
from token_yield.taxonomy import Provenance, ScopedRecord


def rule(title: str) -> None:
    print()
    print(title)
    print("=" * 74)


def main() -> None:
    # ── 1. the measurements ──────────────────────────────────────────────
    rule("1. THE PROBE SUITE — real runs, dispatched to be measured")
    print(f"{len(PROBE_SUITE)} probe specs, {len(MEASURED)} measured runs, "
          f"all provenance={Provenance.PROBE.value}")
    print()
    print(f"  {'kind':<16}{'scope':>6}{'tokens':>10}{'tools':>7}{'sec':>8}   label")
    for r in MEASURED:
        print(f"  {r.kind:<16}{r.scope:>6g}{r.tokens:>10,}{r.tool_uses:>7}"
              f"{r.duration_seconds:>8.1f}   {r.label}")

    rule("   replicates — the noise floor")
    for kind, scope in (("comprehension", 1), ("comprehension", 3), ("code_write", 3)):
        n, mean, sd = replicate_spread(kind, scope)
        print(f"  {kind:<16} scope {scope}: n={n}  mean={mean:>9,.0f}  "
              f"sd={sd:>7,.0f}  cv={sd / mean:.1%}")
    floor = noise_floor(MEASURED)
    print(f"\n  pooled noise floor: {floor:.1%} — no model can beat this")

    # ── 2. fitting ───────────────────────────────────────────────────────
    store = seeded_store()
    rule("2. FITTING — the data picks the shape")
    print(store.report())

    # ── 3. validation ────────────────────────────────────────────────────
    rule("3. VALIDATION — cross-validated error vs the noise floor")
    for kind, rep in backtest(MEASURED).items():
        print(f"  {rep.summary()}")
    print()
    print("  learning curve (is more measurement still worth it?)")
    for kind in ("comprehension", "code_write"):
        pts = learning_curve(kind, MEASURED)
        print(f"    {kind:<16}" + "  ".join(f"n={n}:{s:.1%}" for n, s in pts))

    # ── 4. falsification ─────────────────────────────────────────────────
    rule("4. WHAT THE HARDCODED CONSTANTS CLAIMED")
    print("  The first version asserted A+ = 2x and A++ = 4x. On measured data:")
    print()
    print(f"  {'kind':<16}{'scope':>6}{'asserted':>12}{'fitted':>12}{'ratio':>9}")
    for kind in store.kinds():
        m = store.model_for(kind)
        base = m.predict(1)
        for scope in (2, 4, 8):
            asserted = base * scope
            fitted = m.predict(scope)
            print(f"  {kind:<16}{scope:>6}{asserted:>12,.0f}{fitted:>12,.0f}"
                  f"{asserted / fitted:>8.1f}x")
    print()
    print("  The 'proportional' form IS that assumption. Its cross-validated error:")
    for kind in store.kinds():
        sel = store.selection_for(kind)
        print(f"    {kind:<16} proportional {sel.scores['proportional']:>6.1%}   "
              f"vs chosen {sel.form} {sel.scores[sel.form]:>6.1%}")

    # ── 5. out-of-sample ─────────────────────────────────────────────────
    rule("5. OUT-OF-SAMPLE — predicting the held-out composition experiment")
    ev = composition_evidence()
    plan = WorkPlan("replica").add("comprehension", 3).add("code_write", 3)
    pred = PlanForecaster(store).compare_batching(plan)
    print(f"  {len(COMPOSITION_MEASURED)} batched runs, never fitted on.")
    print()
    print(f"  {'':<26}{'predicted':>12}{'measured':>12}{'error':>9}")
    for label, p, m in (("two separate agents", pred["separate_agents"], ev["separate_sum"]),
                        ("one batched agent", pred["batched_single_agent"], ev["batched_mean"])):
        print(f"  {label:<26}{p:>12,.0f}{m:>12,.0f}{abs(p - m) / m:>8.1%}")
    print()
    print(f"  boot cost {pred['boot_cost']:,.0f}, paid "
          f"{pred['boot_paid_times_if_separate']}x when run separately")
    print(f"  measured saving from batching: {ev['saving']:.1%}")
    print(f"  the old model's +15% interaction SURCHARGE predicted "
          f"{ev['separate_sum'] * 1.15:,.0f} — "
          f"{ev['separate_sum'] * 1.15 / ev['batched_mean']:.1f}x the truth, wrong sign")

    # ── 6. tune-back ─────────────────────────────────────────────────────
    rule("6. TUNE-BACK — new runs arrive and the model answers for itself")
    live = LearningStore()
    live.observe_many([r for r in MEASURED if r.kind == "comprehension"])
    before = live.model_for("comprehension")
    print(f"  standing model: {before.equation()}   (n={before.n})")

    print("\n  (a) new runs that agree with it")
    agreeing = [ScopedRecord("comprehension", 4, 47_000, provenance=Provenance.PRODUCTION),
                ScopedRecord("comprehension", 6, 51_500, provenance=Provenance.PRODUCTION)]
    for kind, d in live.observe_many(agreeing).items():
        print(f"      {d.summary()}")

    print("\n  (b) new runs from a world that got more expensive")
    shifted = [ScopedRecord("comprehension", 3, 88_000, provenance=Provenance.PRODUCTION),
               ScopedRecord("comprehension", 5, 96_000, provenance=Provenance.PRODUCTION),
               ScopedRecord("comprehension", 8, 112_000, provenance=Provenance.PRODUCTION)]
    for kind, d in live.observe_many(shifted).items():
        print(f"      {d.summary()}")

    after = live.model_for("comprehension")
    print(f"\n  refitted model: {after.equation()}   (n={after.n})")
    print(f"  evidence now: {live.evidence('comprehension')}")
    print()
    print("  The loop's whole discipline is in step (b): a model that silently")
    print("  absorbed those runs would look healthy forever. This one said so.")


if __name__ == "__main__":
    main()
