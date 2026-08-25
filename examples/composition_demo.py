"""Price a business request that has never been run, from measured base tasks.

Run with: python -m examples.composition_demo

The chain is:

    plain-English request
        -> decomposed into named base tasks   (the encoder)
        -> recomposed through the fitted model (the decoder)
        -> a token budget, itemised

Every number comes from ``experiments/train_runs.jsonl``: 39 real agent runs
against real SEC filings and earnings transcripts. Nothing here is asserted.
"""

import json
import os

from token_yield.compose import (
    batching_saving, default_runs_path, load_runs, noise_floor, select_model,
)
from token_yield.decompose import (
    Decomposition, explain, heuristic_decompose, price, reconstruction_error,
)
from token_yield.tasks import ORDER, PRIMITIVES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rule(title: str) -> None:
    print()
    print(title)
    print("=" * 74)


def main() -> None:
    runs = load_runs(default_runs_path())
    model = select_model(runs)

    rule("1. THE BASE TASKS — the vocabulary everything is priced in")
    print(f"  {'task':<12}{'category':<14}{'marginal':>10}   where it is bought")
    floor = noise_floor(runs) * min(r.tokens for r in runs if r.total_units == 0)
    for slug in ORDER:
        p = PRIMITIVES[slug]
        m = model.marginals().get(slug, 0.0)
        shown = "~0" if m < floor else f"{m:,.0f}"
        print(f"  {p.name:<12}{p.category:<14}{shown:>10}   {p.industry}")
    print()
    print("  '~0' means the extra unit costs less than run-to-run noise: the work")
    print("  is real, but its cost is swallowed by the price of starting an agent.")

    rule("2. WHAT THE MODEL LEARNED")
    print(f"  {model.equation()}")
    print()
    print(f"  chosen by cross-validation from {len(model.scores)} candidate forms:")
    for name, s in sorted(model.scores.items(), key=lambda kv: kv[1]):
        mark = "  <- selected" if name == model.form else ""
        print(f"    {name:<22}{s:>7.2%}{mark}")
    print()
    print(f"  fitted on {model.n} measured runs")
    print(f"  cross-validated error : {model.loo_mape:.2%}")
    print(f"  repeat-run noise floor: {noise_floor(runs):.2%}")

    rule("3. HELD OUT — compositions the model was never fitted on")
    held = [r for r in runs if r.held_out]
    print(f"  {'composition':<44}{'actual':>9}{'predicted':>11}{'error':>8}")
    errs = []
    for r in held:
        pred = model.predict_run(r)
        e = abs(pred - r.tokens) / r.tokens
        errs.append(e)
        print(f"  {r.notation:<44}{r.tokens:>9,}{pred:>11,.0f}{e:>7.1%}")
    print(f"  {'':<44}{'':>9}{'mean':>11}{sum(errs) / len(errs):>7.1%}")

    rule("4. PRICING A REQUEST NOBODY HAS RUN")
    cases_path = os.path.join(ROOT, "experiments", "decompose_cases.jsonl")
    cases = [json.loads(l) for l in open(cases_path, encoding="utf-8")
             if l.strip()]
    first = cases[0]
    counts = {s: 0 for s in ORDER}
    counts.update(first["encoded"])
    dec = Decomposition(counts, first["rationale"], first["context_bytes"])
    print(f'  Request: "{first["request"]}"')
    print()
    for line in explain(dec, model).splitlines():
        print("  " + line)
    print()
    print(f"  actually cost when run: {first['actual_tokens']:,}")
    print(f"  reconstruction error  : "
          f"{reconstruction_error(dec, model, first['actual_tokens']):.1%}")

    rule("5. THE LEVER THAT ACTUALLY SAVES MONEY")
    batched, separate, saving = batching_saving(model, dec.counts,
                                                dec.context_bytes)
    print(f"  one agent doing all parts : {batched:>10,.0f}")
    print(f"  one agent per part        : {separate:>10,.0f}")
    print(f"  saving from batching      : {saving:>10.1%}")
    print()
    print("  The start-up cost is paid once per agent, not once per task. Any")
    print("  model that just adds the parts up cannot see this, and it is the")
    print("  largest single decision a buyer makes.")

    rule("6. WITHOUT AN AGENT TO DECOMPOSE")
    h = heuristic_decompose(first["request"], first["context_bytes"])
    print(f"  keyword encoder says  : {h.notation()}")
    print(f"  agent encoder said    : {dec.notation()}")
    print(f"  keyword estimate      : {price(h, model):,.0f}")
    print(f"  agent estimate        : {price(dec, model):,.0f}")
    print(f"  actual                : {first['actual_tokens']:,}")
    print()
    print("  The fallback keeps a pipeline running; it cannot count units, so it")
    print("  is a weaker estimate and the Decomposition records which was used.")


if __name__ == "__main__":
    main()
