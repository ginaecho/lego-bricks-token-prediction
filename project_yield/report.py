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
from .outcomes import ORDER, OUTCOMES, STAFF_OUTCOMES
from .predict import Forecast

WIDTH = 74


def rule(title: str = "") -> str:
    return f"\n{title}\n" + "=" * WIDTH if title else "-" * WIDTH


def forecast_card(f: Forecast) -> str:
    """The single-use-case card: what it costs, what it earns, what it risks."""
    uc, e = f.usecase, f.economics
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

    if f.warnings:
        out.append("")
        out.append("  READ THIS FIRST")
        for w in f.warnings:
            out.extend("  ! " + line for line in _wrap(w, WIDTH - 4))

    out.append("")
    out.append("  WHAT IT COSTS TO RUN")
    out.append(f"    {'tokens, one run':<28}{f.tokens.formatted:>34}")
    out.append(f"    {'batching saving vs split':<28}"
               f"{f.tokens.batching_saving:>33.1%}")
    if e.has_run_rate:
        out.append(f"    {'inference, first year':<28}"
                   f"{'$' + format(e.annual_token_cost, ',.0f'):>34}")

    out.append("")
    out.append("  WHAT IT IS WORTH")
    for slug in ("contract_value", "win_probability"):
        out.append(f"    {OUTCOMES[slug].name:<28}"
                   f"{f.estimates[slug].format():>34}")

    out.append("")
    out.append("  WHAT IT TAKES")
    for slug in STAFF_OUTCOMES:
        out.append(f"    {OUTCOMES[slug].name:<28}"
                   f"{f.estimates[slug].format():>34}")
    out.append(f"    {'total staff days':<28}{e.total_staff_days:>34,.1f}")
    out.append(f"    {OUTCOMES['calendar_days'].name:<28}"
               f"{f.estimates['calendar_days'].format():>34}")

    out.append("")
    out.append("  THE DECISION")
    out.append(f"    {'delivery cost':<28}"
               f"{'$' + format(e.delivery_cost, ',.0f'):>34}")
    out.append(f"    {'gross margin if it lands':<28}"
               f"{'$' + format(e.gross_margin, ',.0f') + f'  ({e.gross_margin_pct:.0%})':>34}")
    out.append(f"    {'expected margin':<28}"
               f"{'$' + format(e.expected_margin, ',.0f'):>34}")
    out.append(f"    {'breaks even at a win rate of':<28}"
               f"{e.breakeven_win_rate:>34.0%}")
    verdict = ("worth staffing" if e.is_worth_doing else
               "NOT worth staffing at this price")
    out.append(f"    {'verdict':<28}{verdict:>34}")
    out.append(f"    rates: {e.rates.source}")

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
        out.append("  Scored on the most recent engagements, held out entirely:")
        out.append(f"    {'head':<20}{'cross-validated':>17}{'held out':>12}"
                   f"{'n':>5}")
        for slug in ORDER:
            head = model.heads.get(slug)
            if head is None or slug not in holdout:
                continue
            score, n = holdout[slug]
            out.append(f"    {OUTCOMES[slug].name:<20}{head.loo_score:>17.3f}"
                       f"{score:>12.3f}{n:>5}")
        out.append("")
        out.append("  The two columns agreeing is the point. A gap between them")
        out.append("  would mean the form selection had fitted the corpus rather")
        out.append("  than the process that generated it.")
    out.append("")
    out.append("  Each head chose its own form by cross-validation over the "
               "same seven")
    out.append("  candidates. Where two heads chose differently, the data said "
               "so.")
    return "\n".join(out)


def portfolio_table(forecasts: Sequence[Forecast]) -> str:
    """Rank a set of use cases by the thing that is actually scarce: people."""
    rows = sorted(forecasts, key=lambda f: -f.economics.margin_per_staff_day)
    out = [rule("PORTFOLIO — ranked by expected margin per staff day")]
    out.append(f"  {'use case':<32}{'value':>10}{'win':>6}{'days':>6}"
               f"{'exp. margin':>13}{'$/day':>7}")
    out.append("  " + "-" * (WIDTH - 4))
    for f in rows:
        e = f.economics
        out.append(f"  {f.usecase.title[:31]:<32}"
                   f"{e.contract_value:>10,.0f}"
                   f"{e.win_probability:>6.0%}"
                   f"{e.total_staff_days:>6,.0f}"
                   f"{e.expected_margin:>13,.0f}"
                   f"{e.margin_per_staff_day:>7,.0f}")
    total = sum(f.economics.expected_margin for f in rows)
    days = sum(f.economics.total_staff_days for f in rows)
    out.append("  " + "-" * (WIDTH - 4))
    out.append(f"  {'all of it':<32}{'':>10}{'':>6}{days:>6,.0f}{total:>13,.0f}")
    out.append("")
    out.append("  Ranking on expected margin alone would pick the biggest deals.")
    out.append("  Ranking per staff day picks the ones you can actually deliver")
    out.append("  more of, which is the constraint that binds in practice.")
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
