"""The whole loop, end to end: a use case in, a decision out.

    description
        -> encode              (bricks + industry + goal + scope + lineage)
        -> decode, seven ways  (tokens, price, win rate, 3 staffing lines, time)
        -> economics           (margin, expected margin, breakeven win rate)

The token head is not refitted here. It is
:class:`token_yield.compose.CompositionModel`, fitted on the 39 real measured
agent runs, used unchanged — because that is the one part of this that rests on
measurement, and re-deriving it from a synthetic corpus would quietly throw away
the only real evidence in the system.

Everything this returns carries its own provenance and its own caveats. A
forecast that cannot say how much to trust it is not a forecast, and the
warnings on a :class:`Forecast` are not decoration: an estimate produced by the
keyword encoder, or extrapolated past anything in the corpus, is displayed
differently by every surface that renders one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import economics as econ
from . import multihead
from .corpus import Engagement, load_engagements
from .features import FeatureRow, row_of
from .lineage import LineageFeatures, LineageIndex, index_from_engagements
from .multihead import Estimate, MultiHeadModel
from .outcomes import ORDER, OUTCOMES, STAFF_OUTCOMES
from .usecase import BRICKS, UseCase
from token_yield.tasks import PRIMITIVES
from token_yield.compose import (CompositionModel, batching_saving,
                                 default_runs_path, load_runs, select_model)

#: Below this cosine similarity to the nearest engagement, a use case's brick
#: mix is unlike anything the heads were fitted on. The interval machinery
#: cannot see this — leave-one-out residuals describe the corpus, not the
#: distance from it — so it is reported separately.
MIX_FLOOR = 0.60

#: Slack allowed before a brick's share counts as outside the fitted range.
MIX_TOLERANCE = 0.05


@dataclass(frozen=True)
class TokenEstimate:
    """What one run of the finished pipeline costs in tokens."""

    tokens: float
    low: float
    high: float
    #: Cross-validated error of the token model, which sets the band.
    loo_mape: float
    batched: float
    separate: float
    batching_saving: float

    @property
    def formatted(self) -> str:
        return f"{self.tokens:,.0f} ({self.low:,.0f} – {self.high:,.0f})"


@dataclass
class Forecast:
    """Everything the prototype has to say about one use case."""

    usecase: UseCase
    lineage: LineageFeatures
    tokens: TokenEstimate
    estimates: Dict[str, Estimate]
    economics: econ.Economics
    neighbours: List[Tuple[UseCase, float, Tuple[str, ...]]] = field(
        default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def value(self, slug: str) -> float:
        return self.estimates[slug].value

    @property
    def staff_days(self) -> Dict[str, float]:
        return {s: self.estimates[s].value for s in STAFF_OUTCOMES}

    def to_dict(self) -> Dict[str, object]:
        """A JSON-safe view, for the web prototype and for any caller."""
        e = self.economics
        return {
            "usecase": self.usecase.to_dict(),
            "notation": self.usecase.notation(),
            "lineage": {
                "reuse_depth": self.lineage.reuse_depth,
                "sibling_count": self.lineage.sibling_count,
                "inherited_fraction": round(self.lineage.inherited_fraction, 4),
                "ancestors": list(self.lineage.ancestor_ids),
                "explanation": self.lineage.explain(),
            },
            "tokens": {
                "per_run": round(self.tokens.tokens),
                "low": round(self.tokens.low), "high": round(self.tokens.high),
                "batching_saving": round(self.tokens.batching_saving, 4),
            },
            "outcomes": {
                slug: {
                    "name": OUTCOMES[slug].name,
                    "value": round(est.value, 4),
                    "low": round(est.low, 4), "high": round(est.high, 4),
                    "unit": est.unit,
                    "formatted": est.format(),
                } for slug, est in self.estimates.items()
            },
            "economics": {
                "delivery_cost": round(e.delivery_cost),
                "labour_cost": round(e.labour_cost),
                "build_token_cost": round(e.token_cost),
                "annual_token_cost": round(e.annual_token_cost),
                "has_run_rate": e.has_run_rate,
                "run_rate_dominates": e.run_rate_dominates,
                "gross_margin": round(e.gross_margin),
                "gross_margin_pct": round(e.gross_margin_pct, 4),
                "risk_adjusted_value": round(e.risk_adjusted_value),
                "expected_margin": round(e.expected_margin),
                "breakeven_win_rate": round(e.breakeven_win_rate, 4),
                "margin_per_staff_day": round(e.margin_per_staff_day),
                "total_staff_days": round(e.total_staff_days, 1),
                "is_worth_doing": e.is_worth_doing,
                "rates_source": e.rates.source,
            },
            "neighbours": [
                {"id": u.id, "title": u.title, "similarity": round(sim, 3),
                 "also_matches": list(why), "notation": u.notation()}
                for u, sim, why in self.neighbours
            ],
            "warnings": list(self.warnings),
        }


def _brick_similarity(a: UseCase, b: UseCase) -> float:
    from .lineage import _cosine
    return _cosine(a.counts, b.counts)


def _shares(usecase: UseCase) -> Dict[str, float]:
    total = usecase.total_units
    if not total:
        return {}
    return {s: usecase.counts.get(s, 0) / total for s in BRICKS}


class Predictor:
    """Holds the fitted models and answers questions about new use cases."""

    def __init__(self, heads: MultiHeadModel, token_model: CompositionModel,
                 index: LineageIndex, corpus: Sequence[Engagement],
                 rates: econ.Rates = econ.DEFAULT_RATES) -> None:
        self.heads = heads
        self.token_model = token_model
        self.index = index
        self.corpus = list(corpus)
        self.rates = rates
        fitted = [e for e in self.corpus if not e.held_out]
        self._unit_range = (min(e.total_units for e in fitted),
                            max(e.total_units for e in fitted))
        self._byte_range = (min(e.context_bytes for e in fitted),
                            max(e.context_bytes for e in fitted))
        # The mix block feeds each head a *share* per brick. A share outside
        # anything the corpus contains is extrapolation in shape rather than in
        # size, and the leave-one-out intervals cannot see it: they describe the
        # corpus, not the distance from it.
        self._share_max = {
            slug: max((e.counts.get(slug, 0) / e.total_units)
                      for e in fitted if e.total_units)
            for slug in BRICKS}
        self._synthetic = any(e.provenance == "synthetic" for e in self.corpus)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_defaults(cls, corpus_path: Optional[str] = None,
                      runs_path: Optional[str] = None,
                      rates: econ.Rates = econ.DEFAULT_RATES) -> "Predictor":
        """Fit everything from the data committed with the package."""
        corpus = load_engagements(corpus_path)
        token_model = select_model(load_runs(runs_path or default_runs_path()))
        index = index_from_engagements(corpus)

        train = [e for e in corpus if not e.held_out]
        rows = [row_of(e, index.features_for(e.as_usecase())) for e in train]
        observations = {slug: [e.observed(slug) for e in train] for slug in ORDER}
        heads = multihead.fit(rows, observations)
        return cls(heads, token_model, index, corpus, rates)

    # -- prediction ------------------------------------------------------

    def feature_row(self, usecase: UseCase) -> FeatureRow:
        return row_of(usecase, self.index.features_for(usecase))

    def forecast(self, usecase: UseCase, neighbours: int = 5) -> Forecast:
        row = self.feature_row(usecase)
        lineage = row.lineage

        tokens = self._token_estimate(usecase)
        estimates = self.heads.estimate_all(row)

        economics = econ.compute(
            contract_value=estimates["contract_value"].value,
            win_probability=estimates["win_probability"].value,
            staff_days={s: estimates[s].value for s in STAFF_OUTCOMES},
            tokens=tokens.tokens, rates=self.rates,
            monthly_runs=usecase.monthly_runs,
        )
        return Forecast(
            usecase=usecase, lineage=lineage, tokens=tokens,
            estimates=estimates, economics=economics,
            neighbours=self.index.nearest(usecase, neighbours),
            warnings=self._warnings(usecase, row),
        )

    def _token_estimate(self, usecase: UseCase) -> TokenEstimate:
        model = self.token_model
        point = model.predict(usecase.counts, usecase.context_bytes)
        band = model.loo_mape
        batched, separate, saving = batching_saving(
            model, usecase.counts, usecase.context_bytes)
        return TokenEstimate(
            tokens=point, low=point * (1.0 - band), high=point * (1.0 + band),
            loo_mape=band, batched=batched, separate=separate,
            batching_saving=saving,
        )

    # -- honesty ---------------------------------------------------------

    def _warnings(self, usecase: UseCase, row: FeatureRow) -> List[str]:
        out: List[str] = []
        if self._synthetic:
            out.append(
                "The value and impact heads are fitted on a SYNTHETIC "
                "engagement corpus. Treat every figure except the token budget "
                "as a demonstration of the machinery, not as evidence.")
        if usecase.encoder == "heuristic":
            out.append(
                "Encoded by keyword match, not by a model. The brick counts "
                "are a guess at scale rather than a reading of intent.")
        lo, hi = self._unit_range
        if usecase.total_units > hi:
            out.append(
                f"Scope of {usecase.total_units} brick units is larger than "
                f"anything in the corpus (max {hi}). Every head is "
                f"extrapolating.")
        elif usecase.total_units < lo:
            out.append(
                f"Scope of {usecase.total_units} brick units is smaller than "
                f"anything in the corpus (min {lo}).")
        blo, bhi = self._byte_range
        if usecase.context_bytes > bhi:
            out.append(
                f"Context of {usecase.context_bytes:,} bytes exceeds the "
                f"corpus maximum ({bhi:,}); the token model is also outside "
                f"its measured range.")
        for slug, share in _shares(usecase).items():
            ceiling = self._share_max.get(slug, 1.0)
            if share > ceiling + MIX_TOLERANCE:
                out.append(
                    f"{PRIMITIVES[slug].name} is {share:.0%} of this scope; no "
                    f"engagement in the corpus goes above {ceiling:.0%}. The "
                    f"heads are extrapolating in shape as well as size, which "
                    f"the stated intervals do not cover.")
        best_cosine = max((sim for _, sim, _ in self.index.nearest(usecase, 8)),
                          default=0.0)
        if best_cosine < MIX_FLOOR:
            out.append(
                f"No engagement in the corpus has a brick mix close to this one "
                f"(best match {best_cosine:.2f} of 1.00), so the comparables "
                f"panel is not really comparable.")
        if usecase.parent_id and usecase.parent_id not in self.index:
            out.append(
                f"Parent use case {usecase.parent_id!r} is not in the library, "
                f"so this is being priced as greenfield. Any reuse benefit is "
                f"absent from these numbers.")
        for slug in ORDER:
            head = self.heads.heads.get(slug)
            if head is not None and not head.beats_baseline:
                out.append(
                    f"The {OUTCOMES[slug].name.lower()} head does not beat its "
                    f"own baseline on held-out data; read it as the corpus "
                    f"base rate, not as a prediction about this use case.")
        return out

    # -- evidence --------------------------------------------------------

    def evaluate_holdout(self) -> Dict[str, Tuple[float, int]]:
        """Score each head on engagements it was never fitted on.

        The hold-out is the most recent slice of the corpus, not a random
        subset: the question a PM is asking is about the next engagement, and
        predicting forward in time is a strictly harder test than interpolating
        within a history you have already seen.
        """
        held = [e for e in self.corpus if e.held_out]
        if not held:
            return {}
        out: Dict[str, Tuple[float, int]] = {}
        for slug in ORDER:
            head = self.heads.heads.get(slug)
            if head is None:
                continue
            errs = []
            for eng in held:
                row = row_of(eng, self.index.features_for(eng.as_usecase()))
                pred = head.predict(row)
                actual = eng.observed(slug)
                if OUTCOMES[slug].binary:
                    errs.append((pred - actual) ** 2)        # Brier
                elif actual:
                    errs.append(abs(pred - actual) / actual)  # MAPE
            if errs:
                out[slug] = (sum(errs) / len(errs), len(errs))
        return out
