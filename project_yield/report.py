"""Rendering a forecast as something a person can argue with.

The obligation here is the one :func:`token_yield.decompose.explain` sets for
the token model: every number shows where it came from. A price with no visible
terms is a guess with a decimal point, and the first time a delivery lead cannot
reconstruct one, the tool stops being used.

So each card prints the band, not just the point; the warnings before the
numbers rather than in a footnote; and the comparable engagements the estimate
is standing on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .multihead import MultiHeadModel
from .outcomes import OUTCOMES
from .predict import Forecast

WIDTH = 74


def rule(title: str = "") -> str:
    return f"\n{title}\n" + "=" * WIDTH if title else "-" * WIDTH


def forecast_card(f: Forecast) -> str:
    """The single-use-case card, in the order a sponsor asks the questions.

    Impact first. Everything else here prices the engagement, and a build that
    costs sixty thousand dollars is expensive or cheap depending entirely on
    whether it displaces forty thousand a year or four million — so leading
    with the cost is leading with the half of the case that cannot settle it.
    """
    uc, e, i = f.usecase, f.economics, f.impact
    out: List[str] = []
    out.append(rule(f"{uc.title.upper()}"))
    out.append(f"  {uc.industry.replace('_', ' ')} · "
               f"{uc.goal.replace('_', ' ')} · encoded by {uc.encoder}")
    out.append(f"  scope   : {uc.notation()}")
    out.append(f"  context : {uc.context_bytes:,} bytes"
               + (f" · {uc.monthly_runs:,} runs/month in production"
                  if uc.monthly_runs else " · no production run rate given"))
    out.append(f"  lineage : {f.lineage.explain()}")
    if uc.rationale:
        out.append(f"  encoder : {uc.rationale}")

    out.append("")
    out.append("  WHAT IT IS WORTH TO THE CLIENT")
    if i.quoted:
        out.append(f"    {'handling displaced':<28}"
                   f"{f'{i.hours_saved:,.0f} hours/year':>34}")
        out.append(f"    {'  which is':<28}"
                   f"{f'{i.fte_equivalent:.1f} full-time people':>34}")
        out.append(f"    {'annual benefit':<28}"
                   f"{'$' + format(i.annual_benefit, ',.0f'):>34}")
        out.append(f"    {'less running it':<28}"
                   f"{'-$' + format(e.annual_token_cost, ',.0f'):>34}")
        out.append(f"    {'net impact, per year':<28}"
                   f"{'$' + format(i.annual_net_benefit, ',.0f'):>34}")
        out.append(f"    {'first-year return':<28}"
                   f"{'$' + format(i.first_year_return, ',.0f'):>34}")
        out.append(f"    {'':<28}{i.verdict:>34}")
    else:
        out.append(f"    {'annual impact':<28}{i.verdict:>34}")
    out.append(f"    assumes {i.minutes_per_run:.0f} min/run by hand, "
               f"{i.assumptions.deflection:.0%} removed — {i.assumptions.source}")

    out.append("")
    out.append("  WHAT IT IS WORTH TO US")
    for slug in ("contract_value", "win_probability"):
        out.append(f"    {OUTCOMES[slug].name:<28}"
                   f"{f.estimates[slug].format():>34}")
    out.append(f"    {'gross margin if it lands':<28}"
               f"{'$' + format(e.gross_margin, ',.0f') + f'  ({e.gross_margin_pct:.0%})':>34}")
    out.append(f"    {'expected margin':<28}"
               f"{'$' + format(e.expected_margin, ',.0f'):>34}")
    out.append(f"    {'breaks even at a win rate of':<28}"
               f"{e.breakeven_win_rate:>34.0%}")

    out.append("")
    out.append("  WHO IT TAKES")
    if not f.staffing:
        out.append("    (no roles on the plan)")
    for role in f.staffing:
        out.append(f"    {role.role.name:<28}{role.summary():>34}")
    out.append(f"    {'total expected staff days':<28}"
               f"{e.total_staff_days:>34,.1f}")
    out.append(f"    {'delivery cost':<28}"
               f"{'$' + format(e.delivery_cost, ',.0f'):>34}")
    out.append(f"    {'working days to finish':<28}"
               f"{f.estimates['calendar_days'].format():>34}")
    sourced = sorted({r.source for r in f.staffing})
    if sourced and sourced != ["predicted"]:
        out.append(f"    {'team decided by':<28}{', '.join(sourced):>34}")
    out.append(f"    rates: {e.rates.source}")

    out.append("")
    out.append("  WHAT IT COSTS TO RUN")
    out.append(f"    {'tokens, one run':<28}{f.tokens.formatted:>34}")
    out.append(f"    {'batching saving vs split':<28}"
               f"{f.tokens.batching_saving:>33.1%}")
    if e.has_run_rate:
        out.append(f"    {'inference, first year':<28}"
                   f"{'$' + format(e.annual_token_cost, ',.0f'):>34}")

    if f.warnings:
        out.append("")
        out.append("  READ THIS FIRST")
        for w in f.warnings:
            out.extend("  ! " + line for line in _wrap(w, WIDTH - 4))

    if f.neighbours:
        out.append("")
        out.append("  COMPARABLE ENGAGEMENTS IT IS STANDING ON")
        out.append(f"    {'brick':>5}  {'id':<6}{'engagement':<32}"
                   f"{'also matches':<22}")
        for uc2, sim, why in f.neighbours:
            out.append(f"    {sim:>5.2f}  {uc2.id:<6}{uc2.title[:31]:<32}"
                       f"{(', '.join(why) or '—'):<22}")
    return "\n".join(out)


def model_card(model: MultiHeadModel,
               holdout: Optional[Dict[str, Tuple[float, int]]] = None,
               title: str = "THE MODEL") -> str:
    """What each head is, how it was chosen, and how well it does.

    ``holdout`` is the output of :meth:`~project_yield.predict.Predictor.
    evaluate_holdout` — the score on engagements no head was fitted on. It is
    printed next to the cross-validated score because agreement between the two
    is the evidence that neither is an artefact of the selection.
    """
    out = [rule(title), model.report()]
    if holdout:
        out.append("")
        out.append("  Scored on the most recent engagements, held out entirely.")
        out.append("  'base' is the do-nothing model on the same rows — a score")
        out.append("  means nothing without it.")
        out.append(f"    {'head':<24}{'cross-val':>10}{'held out':>10}"
                   f"{'base':>8}{'n':>5}  ")
        for slug in (model.order or tuple(model.heads)):
            head = model.heads.get(slug)
            score = holdout.get(slug)
            if head is None or score is None:
                continue
            flag = "" if score.beats_baseline else "  <- loses to base rate"
            out.append(f"    {head.outcome.name[:23]:<24}{head.loo_score:>10.3f}"
                       f"{score.score:>10.3f}{score.baseline:>8.3f}"
                       f"{score.n:>5}{flag}")
        lost = [model.heads[s].outcome.name for s in holdout
                if s in model.heads and not holdout[s].beats_baseline]
        out.append("")
        if lost:
            out.append("  Cross-validation and the forward hold-out disagreeing is")
            out.append("  the most useful thing this table can show, and it is")
            out.append("  showing it for: " + ", ".join(sorted(set(lost))) + ".")
            out.append("  Those heads interpolate within the history and do not")
            out.append("  yet predict the next engagement.")
        else:
            out.append("  The columns agreeing is the point. A gap would mean the")
            out.append("  form selection had fitted the corpus rather than the")
            out.append("  process that generated it.")
    out.append("")
    out.append("  Each head chose its own form by cross-validation over the "
               "same seven")
    out.append("  candidates. Where two heads chose differently, the data said "
               "so.")
    return "\n".join(out)


def portfolio_table(forecasts: Sequence[Forecast]) -> str:
    """Rank a set of use cases by the thing that is actually scarce: people."""
    rows = sorted(forecasts, key=lambda f: -f.impact.first_year_return)
    out = [rule("PORTFOLIO — ranked by first-year client return")]
    out.append(f"  {'use case':<30}{'impact/yr':>13}{'value':>10}{'win':>6}"
               f"{'days':>6}{'payback':>9}")
    out.append("  " + "-" * (WIDTH - 4))
    for f in rows:
        e, i = f.economics, f.impact
        payback = (f"{i.payback_months:.0f} mo" if i.payback_months
                   else ("—" if not i.quoted else "never"))
        out.append(f"  {f.usecase.title[:29]:<30}"
                   f"{i.annual_net_benefit:>13,.0f}"
                   f"{e.contract_value:>10,.0f}"
                   f"{e.win_probability:>6.0%}"
                   f"{e.total_staff_days:>6,.0f}"
                   f"{payback:>9}")
    out.append("  " + "-" * (WIDTH - 4))
    out.append(f"  {'all of it':<30}"
               f"{sum(f.impact.annual_net_benefit for f in rows):>13,.0f}"
               f"{sum(f.economics.contract_value for f in rows):>10,.0f}"
               f"{'':>6}"
               f"{sum(f.economics.total_staff_days for f in rows):>6,.0f}")
    out.append("")
    out.append("  Ranked by what the client gets, because that is what decides")
    out.append("  whether a use case is funded at all. Ranking by our own margin")
    out.append("  picks the deals we like; ranking by payback picks the ones that")
    out.append("  get a second phase.")
    return "\n".join(out)


def warnings_block(f: Forecast, indent: str = "  ") -> str:
    """The caveats on their own, for surfaces that show numbers without a card."""
    if not f.warnings:
        return f"{indent}(no caveats)"
    return "\n".join(f"{indent}! " + line
                      for w in f.warnings for line in _wrap(w, WIDTH - 4))


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
