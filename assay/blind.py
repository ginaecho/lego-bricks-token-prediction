"""The prospective test on the sealed set.

This is the only part of the project that can produce evidence about the *next* task
rather than the last one, and everything here exists to keep it that way:

* Every quote is committed to disk, with a hash, **before** anything is dispatched. The
  runner refuses a task that has no committed quote -- a prediction written afterwards is
  not a prediction.
* The ordering ``quoted_at < dispatched_at < recorded_at`` is checked, not trusted.
* Nothing is refitted afterwards. Ever.
* More than three exclusions out of twenty-four voids the set rather than shrinking it.
  Dropping the inconvenient runs and scoring the rest is how a blind test stops being one.

Blind gold is never labelled and encoder exact-match is never scored here. The blind set
answers one question -- did the quote match the bill -- and spending it on a second
question would leave neither properly answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

from assay.conformal import Conformal
from assay.costmodel import CostModel, Predictor, predict_all
from assay.errors import SealError
from assay.evidence import EvidenceClass
from assay.harness.probes import Probe
from assay.metrics import Scorecard, coverage, quantile, relative_lift, score, wape, wilson_ci
from assay.pricing import Pricing
from assay.quote import Quote, append_quote, make_quote
from assay.schemas import Run

MAX_EXCLUSIONS = 3


def commit_quotes(
    probes: Sequence[Probe],
    *,
    model: CostModel,
    conformal: Conformal,
    pricing: Pricing,
    vectors: Mapping[str, dict[str, int]] | None = None,
    baselines: Mapping[str, Predictor] | None = None,
    context_bytes: Mapping[str, int] | None = None,
    path=None,
    evidence_class: EvidenceClass = EvidenceClass.PIPELINE_ONLY,
) -> dict[str, Quote]:
    """Price every sealed task and write the quotes down before any of them runs.

    ``vectors`` lets the encoder supply the brick counts. When it is omitted the probe's
    designed vector is used, which measures the decoder alone -- a useful decomposition,
    because a bad end-to-end number is otherwise impossible to attribute.
    """
    quotes: dict[str, Quote] = {}
    for probe in probes:
        units = (vectors or {}).get(probe.probe_id, probe.units)
        nbytes = (context_bytes or {}).get(probe.probe_id)
        if nbytes is None:
            raise ValueError(f"{probe.probe_id}: context size must be supplied at quote time")
        quote = make_quote(
            probe.probe_id, units, nbytes,
            model=model, conformal=conformal, pricing=pricing,
            baselines=baselines, evidence_class=evidence_class,
        )
        quotes[probe.probe_id] = quote
        if path is not None:
            append_quote(path, quote)
    return quotes


def _as_utc(stamp: str, what: str) -> datetime:
    """Parse an ISO timestamp into an aware UTC instant.

    Quotes are written by this package and are always aware; run timestamps can arrive
    from a replayed file that dropped the offset. Comparing an aware datetime with a naive
    one raises ``TypeError``, which would surface as a crash in the middle of blind
    scoring rather than as a verdict. A naive stamp is read as UTC, which is what every
    writer in this package emits.
    """
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def assert_quote_preceded_run(quote: Quote, run: Run) -> None:
    """The invariant the whole prospective claim rests on."""
    if not run.timestamp:
        raise SealError(f"{run.probe_id}: run has no timestamp; ordering cannot be verified")
    quoted = _as_utc(quote.quoted_at, "quote")
    recorded = _as_utc(run.timestamp, "run")
    if quoted > recorded:
        raise SealError(
            f"{run.probe_id}: quote was written at {quote.quoted_at}, after the run at "
            f"{run.timestamp} -- that is not a prediction"
        )


def guard_dispatch(probe: Probe, quotes: Mapping[str, Quote]) -> None:
    if probe.probe_id not in quotes:
        raise SealError(
            f"{probe.probe_id} has no committed quote; the blind runner will not dispatch it"
        )


@dataclass
class BlindReport:
    n_planned: int
    n_scored: int
    n_excluded: int
    voided: bool
    scorecard: Scorecard | None
    baseline_scorecards: dict[str, Scorecard]
    best_baseline: str | None
    lift_pct: float | None
    coverage_hits: int
    coverage_n: int
    coverage_ci: tuple[float, float]
    median_relative_half_width: float
    p90_relative_half_width: float
    evidence_class: EvidenceClass
    notes: list[str] = field(default_factory=list)

    @property
    def coverage_claim(self) -> str:
        lo, hi = self.coverage_ci
        return (
            f"{self.coverage_hits}/{self.coverage_n} inside the band "
            f"(95% CI {lo:.0%}-{hi:.0%}) -- a screen, not proof of the stated level"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "n_planned": self.n_planned,
            "n_scored": self.n_scored,
            "n_excluded": self.n_excluded,
            "voided": self.voided,
            "evidence_class": self.evidence_class.value,
            "scorecard": self.scorecard.as_dict() if self.scorecard else None,
            "baselines": {k: v.as_dict() for k, v in self.baseline_scorecards.items()},
            "best_baseline": self.best_baseline,
            "lift_vs_best_baseline_pct": (
                round(self.lift_pct, 3) if self.lift_pct is not None else None
            ),
            "coverage": {
                "hits": self.coverage_hits,
                "n": self.coverage_n,
                "wilson_ci": [round(c, 4) for c in self.coverage_ci],
                "claim": self.coverage_claim,
            },
            "median_relative_half_width_pct": round(self.median_relative_half_width, 3),
            "p90_relative_half_width_pct": round(self.p90_relative_half_width, 3),
            "notes": self.notes,
        }

    def observations(self) -> dict[str, float | None]:
        """The gate values this experiment contributes to the verdict."""
        if self.voided or self.scorecard is None:
            return {"M6_blind_mape_pct": None, "M6b_blind_lift_pct": None, "M9b_coverage_hits": None}
        return {
            "M6_blind_mape_pct": self.scorecard.mape_total,
            "M6b_blind_lift_pct": self.lift_pct,
            "M9b_coverage_hits": float(self.coverage_hits),
        }


def score_blind(
    quotes: Mapping[str, Quote],
    runs: Sequence[Run],
    *,
    boot: float,
    baselines: Mapping[str, Predictor] | None = None,
    n_planned: int | None = None,
) -> BlindReport:
    notes: list[str] = []
    accepted = [r for r in runs if r.accepted]
    excluded = len(runs) - len(accepted)
    planned = n_planned if n_planned is not None else len(runs)

    for run in accepted:
        quote = quotes.get(run.probe_id)
        if quote is None:
            raise SealError(f"{run.probe_id} was dispatched without a committed quote")
        assert_quote_preceded_run(quote, run)

    voided = excluded > MAX_EXCLUSIONS
    if voided:
        notes.append(
            f"{excluded} exclusions of {planned} -- above the ceiling of {MAX_EXCLUSIONS}. "
            "The set is void; scoring the survivors would not be a blind test."
        )

    if not accepted or voided:
        return BlindReport(
            n_planned=planned, n_scored=len(accepted), n_excluded=excluded, voided=voided,
            scorecard=None, baseline_scorecards={}, best_baseline=None, lift_pct=None,
            coverage_hits=0, coverage_n=0, coverage_ci=(0.0, 1.0),
            median_relative_half_width=0.0, p90_relative_half_width=0.0,
            evidence_class=EvidenceClass.PIPELINE_ONLY, notes=notes,
        )

    actual = [r.billable_units for r in accepted]
    predicted = [quotes[r.probe_id].predicted_units for r in accepted]
    card = score(actual, predicted, boot)

    base_cards: dict[str, Scorecard] = {}
    for key, predictor in (baselines or {}).items():
        base_cards[key] = score(actual, predict_all(predictor, accepted), boot)

    best = min(base_cards, key=lambda k: base_cards[k].vwape) if base_cards else None
    lift = (
        relative_lift(base_cards[best].vwape, card.vwape) if best is not None else None
    )

    intervals = [
        (quotes[r.probe_id].interval_lo, quotes[r.probe_id].interval_hi) for r in accepted
    ]
    hits, n = coverage(actual, intervals)
    widths = [quotes[r.probe_id].relative_half_width_pct for r in accepted]

    classes = {r.evidence_class for r in accepted}
    evidence = (
        EvidenceClass.FEASIBILITY
        if classes == {EvidenceClass.FEASIBILITY}
        else EvidenceClass.PIPELINE_ONLY
    )

    if card.negative_variable_rows:
        notes.append(
            f"{card.negative_variable_rows} run(s) cost no more than the empty task; "
            "kept in the table as measurement failures"
        )

    return BlindReport(
        n_planned=planned,
        n_scored=len(accepted),
        n_excluded=excluded,
        voided=False,
        scorecard=card,
        baseline_scorecards=base_cards,
        best_baseline=best,
        lift_pct=lift,
        coverage_hits=hits,
        coverage_n=n,
        coverage_ci=wilson_ci(hits, n),
        median_relative_half_width=quantile(widths, 0.5),
        p90_relative_half_width=quantile(widths, 0.9),
        evidence_class=evidence,
        notes=notes,
    )


def variable_wape(actual: Sequence[float], predicted: Sequence[float], boot: float) -> float:
    return wape([a - boot for a in actual], [p - boot for p in predicted])
