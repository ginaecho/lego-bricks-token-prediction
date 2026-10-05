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
    print("  The token model answers one question. A scoping conversation "
          "needs five.")
    print()
    print(f"  {'head':<22}{'unit':<14}{'link':<10}what it is for")
    print(f"  {'tokens':<22}{'tokens':<14}{'linear':<10}"
          "what one run costs to execute")
    for slug in ORDER:
        o = OUTCOMES[slug]
        print(f"  {o.name:<22}{o.unit:<14}{o.link.name:<10}{o.question}")
    print(f"  {'annual impact':<22}{'USD':<14}{'derived':<10}"
          "what the client gets once it runs")
    print()
    print("  Plus two heads for every role on the roster — is this role needed")
    print("  at all, and how many days if it is:")
    print()
    for role in predictor.roster:
        fitted = "fitted" if role.days_outcome in predictor.heads else "NO DATA"
        print(f"    {role.name:<24}${role.day_rate:>7,.0f}/day   {fitted}")
    print()
    print("  The roster is a JSON file, not a list in the code. Add a role and")
    print("  — given a column in the corpus — it is fitted, priced and "
          "reported")
    print("  with no code change. Nothing is one model with many outputs: "
          "money")
    print("  multiplies, a win is bounded at both ends, elapsed time has a "
          "floor")
    print("  no headcount moves, and a data scientist is not on every job.")

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
    rows += [("total staff days", alone.economics.total_staff_days,
              linked.economics.total_staff_days),
             ("delivery cost", alone.economics.delivery_cost,
              linked.economics.delivery_cost)]
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
    print(rule("5. WHAT THE CLIENT GETS"))
    print("  Everything above prices the engagement. A build costing $60,000 "
          "is")
    print("  expensive or cheap depending entirely on what it displaces, and "
          "no")
    print("  delivery estimate can tell you which.")
    print()
    print(f"  {'use case':<34}{'min/run':>9}{'runs/mo':>10}"
          f"{'impact/yr':>13}{'payback':>10}")
    for case in CASES:
        f = predictor.forecast(case)
        i = f.impact
        payback = (f"{i.payback_months:.1f} mo" if i.payback_months
                   else ("—" if not i.quoted else "never"))
        print(f"  {case.title[:33]:<34}{i.minutes_per_run:>9.0f}"
              f"{case.monthly_runs:>10,}{i.annual_net_benefit:>13,.0f}"
              f"{payback:>10}")
    print()
    print("  Same three use cases, same bricks. The one that runs 24,000 times")
    print("  a month is a different business case from the one that runs 120")
    print("  times, and the token model is what makes that difference visible.")

    print()
    print(rule("6. THE PORTFOLIO VIEW"))
    forecasts = [predictor.forecast(c) for c in CASES]
    print(portfolio_table(forecasts))

    print()
    print(rule("7. FROM A DESCRIPTION, WITH NO STRUCTURED INPUT"))
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
    print(f"    {'team':<22}"
          + ", ".join(r.role.name for r in f.staffing))
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
