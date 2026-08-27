"""The product prototype end to end: token budget, price, risk, staffing, time.

Run with: python -m examples.project_yield_demo

``examples/composition_demo.py`` shows the token model: bricks in, tokens out,
fitted on 39 real agent runs. This shows what happens when the same encode /
decode structure is pointed at the other four questions a Microsoft PM has to
answer before they can decide anything — what the client will pay, whether it
will land, who it takes, and how long it runs.

The token numbers here are the real ones. Everything else is fitted on a
synthetic engagement corpus (``experiments/engagements.jsonl``), which exists so
the machinery can be demonstrated and tested before it is wired to real delivery
data. Every screen below says so, because a demo that is coy about which of its
numbers are measured teaches people to trust the wrong ones.
"""

from __future__ import annotations

from project_yield.encode import heuristic_encode
from project_yield.outcomes import ORDER, OUTCOMES
from project_yield.predict import Predictor
from project_yield.report import (forecast_card, model_card, portfolio_table,
                                  rule, warnings_block)
from project_yield.usecase import UseCase


#: Three use cases a PM might scope in one afternoon, the third a continuation
#: of the first — which is the case the token model alone cannot price.
CASES = [
    UseCase(
        id="NEW-1", title="Northwind invoice intake",
        description=("Automate supplier invoice intake for a manufacturing "
                     "client: read each invoice, extract the header and line "
                     "fields, classify for routing, validate totals against "
                     "the purchase order and flag exceptions for a human."),
        industry="manufacturing", goal="cost_reduction",
        counts={"review": 2, "extract": 9, "classify": 4, "validate": 3,
                "remediate": 2},
        context_bytes=22000, monthly_runs=24000, encoder="manual",
    ),
    UseCase(
        id="NEW-2", title="Woodgrove disclosure review",
        description=("Review quarterly disclosures against the regulator's "
                     "checklist, reconcile the figures to the filed accounts, "
                     "and produce a reviewer's note listing every exception."),
        industry="financial_services", goal="compliance_risk",
        counts={"review": 6, "reconcile": 5, "validate": 7, "extract": 3,
                "report": 2},
        context_bytes=48000, monthly_runs=120, encoder="manual",
    ),
    UseCase(
        id="NEW-3", title="Northwind claims intake (phase 2)",
        description=("Extend the invoice pipeline to supplier warranty claims: "
                     "same extraction and routing, plus a reconciliation "
                     "against the warranty terms."),
        industry="manufacturing", goal="cost_reduction",
        counts={"review": 2, "extract": 8, "classify": 4, "validate": 3,
                "reconcile": 3},
        context_bytes=26000, monthly_runs=9000,
        parent_id="NEW-1", encoder="manual",
    ),
]


def main() -> None:
    predictor = Predictor.from_defaults()
    # The use cases being scoped now are part of the lineage graph too: a PM
    # linking today's use case to one they scoped an hour ago should get the
    # same treatment as one that closed last year.
    for case in CASES:
        predictor.index.add(case)

    print(rule("1. WHAT THIS ADDS TO THE TOKEN MODEL"))
    print("  The token model answers one question. A scoping decision needs "
          "five.")
    print()
    print(f"  {'head':<20}{'unit':<14}{'link':<10}what it is for")
    print(f"  {'tokens':<20}{'tokens':<14}{'linear':<10}"
          "what one run costs to execute")
    for slug in ORDER:
        o = OUTCOMES[slug]
        print(f"  {o.name:<20}{o.unit:<14}{o.link.name:<10}{o.question}")
    print()
    print("  Six heads, not one model with six outputs. Money multiplies, a "
          "win is")
    print("  bounded at both ends, and elapsed time has a floor no headcount "
          "moves.")

    print(model_card(predictor.heads, predictor.evaluate_holdout(),
                     title="2. THE MODEL"))

    print(rule("3. A USE CASE, PRICED"))
    print(forecast_card(predictor.forecast(CASES[0])))

    print()
    print(rule("4. THE SAME WORK, AS A CONTINUATION"))
    print("  NEW-3 is NEW-1's pipeline pointed at a second document type. It "
          "is")
    print("  nearly the same size — and the lineage features are what stop it")
    print("  being priced as though nobody had built it before.")
    greenfield = UseCase.from_dict(dict(CASES[2].to_dict(), id="NEW-3-alone",
                                        parent_id=None))
    linked = predictor.forecast(CASES[2])
    alone = predictor.forecast(greenfield)
    print()
    print(f"  {'':<24}{'as greenfield':>16}{'as continuation':>18}"
          f"{'difference':>13}")
    rows = [("brick units", float(greenfield.total_units),
             float(CASES[2].total_units))]
    rows += [(OUTCOMES[s].name, alone.value(s), linked.value(s)) for s in ORDER]
    for name, a, b in rows:
        delta = (b - a) / a if a else 0.0
        print(f"  {name:<24}{a:>16,.1f}{b:>18,.1f}{delta:>12.0%}")
    print()
    print(f"  inherited brick share: {linked.lineage.inherited_fraction:.0%} "
          f"({linked.lineage.explain()})")
    print("  No coefficient above was set by hand. Reuse depth, sibling count "
          "and")
    print("  inherited brick share are three features among twenty-two, and "
          "every")
    print("  head was free to price them at zero.")

    print()
    print(rule("5. THE PORTFOLIO VIEW"))
    forecasts = [predictor.forecast(c) for c in CASES]
    print(portfolio_table(forecasts))

    print()
    print(rule("6. FROM A DESCRIPTION, WITH NO STRUCTURED INPUT"))
    typed = heuristic_encode(
        "We want to triage inbound service tickets for a retail client and "
        "draft a first response, around 900 tickets per week, to cut handling "
        "cost.", uid="TYPED-1", title="Retail ticket triage")
    print(f"  typed:    {typed.description[:66]}...")
    print(f"  encoded:  {typed.notation()}")
    print(f"  read as:  {typed.industry} · {typed.goal} · "
          f"{typed.monthly_runs:,} runs/month")
    print(f"  {typed.rationale}")
    f = predictor.forecast(typed)
    print()
    for slug in ORDER:
        print(f"    {OUTCOMES[slug].name:<22}{f.estimates[slug].format()}")
    print(f"    {'tokens, one run':<22}{f.tokens.formatted}")
    print()
    print(warnings_block(f))
    print()
    print("  This is the fallback encoder, and it is genuinely worse than a "
          "model")
    print("  reading the same paragraph. The estimate carries that fact with "
          "it —")
    print("  see the warnings on the card — rather than looking like any other "
          "one.")


if __name__ == "__main__":
    main()
