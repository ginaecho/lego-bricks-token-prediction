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
from . import impact as imp
from . import multihead
from .corpus import Engagement, load_engagements
from .features import FeatureRow, row_of
from .lineage import LineageFeatures, LineageIndex, index_from_engagements
from .multihead import Estimate, MultiHeadModel
from .outcomes import ORDER, OUTCOMES, build_order, build_outcomes
from .roles import DEFAULT_ROSTER, Role, Roster
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

#: Below this probability a role is left off the plan entirely rather than
#: shown as a fraction of a day. A staffing plan with nine roles on it, seven
#: of them at 4%, is not a plan.
ROLE_FLOOR = 0.10


@dataclass(frozen=True)
class HoldoutScore:
    """One head's score on engagements it was never fitted on.

    Carries the baseline on the *same* rows, because a bare score is not
    interpretable: a Brier of 0.32 is good on a corpus of coin flips and bad on
    one where nine in ten engagements land. The comparison is the whole content.
    """

    outcome: str
    score: float
    baseline: float
    n: int
    metric: str = "mape"

    @property
    def beats_baseline(self) -> bool:
        return self.score < self.baseline

    @property
    def skill(self) -> float:
        return ((self.baseline - self.score) / self.baseline
                if self.baseline else 0.0)


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


@dataclass(frozen=True)
class RoleEstimate:
    """One line of the staffing plan.

    Three numbers rather than one, because they answer different questions and
    collapsing them loses the one a PM actually acts on:

    ``probability``  how often comparable work needed this role at all
    ``days_when_needed``  how many days on the jobs that did use it
    ``expected_days``  the product, which is what the cost is built from

    A role the PM has named is certain by definition, so its probability is 1
    and the two day figures coincide. That is the point of letting them name
    it: their knowledge beats a base rate.
    """

    role: Role
    probability: float
    days_when_needed: float
    low: float
    high: float
    source: str = "predicted"

    @property
    def expected_days(self) -> float:
        return self.probability * self.days_when_needed

    @property
    def cost(self) -> float:
        return self.expected_days * self.role.day_rate

    @property
    def is_certain(self) -> bool:
        return self.probability >= 0.999

    def summary(self) -> str:
        if self.is_certain:
            return f"{self.days_when_needed:,.1f} days"
        return (f"{self.expected_days:,.1f} days expected · "
                f"{self.probability:.0%} likely, {self.days_when_needed:,.1f} "
                f"days when needed")


@dataclass
class Forecast:
    """Everything the prototype has to say about one use case."""

    usecase: UseCase
    lineage: LineageFeatures
    tokens: TokenEstimate
    estimates: Dict[str, Estimate]
    #: The staffing plan, in roster order, excluding roles nobody needs.
    staffing: List[RoleEstimate]
    economics: econ.Economics
    #: What the client gets once it is running — the other half of the case.
    impact: imp.Impact
    neighbours: List[Tuple[UseCase, float, Tuple[str, ...]]] = field(
        default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def value(self, slug: str) -> float:
        return self.estimates[slug].value

    @property
    def staff_days(self) -> Dict[str, float]:
        """Expected days by role slug — what the cost is built from."""
        return {r.role.slug: r.expected_days for r in self.staffing}

    def role(self, slug: str) -> Optional[RoleEstimate]:
        return next((r for r in self.staffing if r.role.slug == slug), None)

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
            "staffing": [
                {"role": r.role.slug, "name": r.role.name,
                 "probability": round(r.probability, 4),
                 "days_when_needed": round(r.days_when_needed, 2),
                 "expected_days": round(r.expected_days, 2),
                 "low": round(r.low, 2), "high": round(r.high, 2),
                 "day_rate": r.role.day_rate, "cost": round(r.cost),
                 "certain": r.is_certain, "source": r.source,
                 "summary": r.summary()}
                for r in self.staffing
            ],
            # Only the engagement-level outcomes. The per-role heads are the
            # staffing plan above, assembled and overridable, rather than
            # eighteen raw numbers a caller would have to pair up themselves.
            "outcomes": {
                slug: {
                    "name": est.name,
                    "value": round(est.value, 4),
                    "low": round(est.low, 4), "high": round(est.high, 4),
                    "unit": est.unit,
                    "formatted": est.format(),
                } for slug, est in self.estimates.items() if slug in ORDER
            },
            "impact": {
                "quoted": self.impact.quoted,
                "minutes_per_run": round(self.impact.minutes_per_run, 1),
                "hours_saved": round(self.impact.hours_saved),
                "fte_equivalent": round(self.impact.fte_equivalent, 1),
                "annual_benefit": round(self.impact.annual_benefit),
                "annual_net_benefit": round(self.impact.annual_net_benefit),
                "first_year_return": round(self.impact.first_year_return),
                "payback_months": (round(self.impact.payback_months, 1)
                                   if self.impact.payback_months else None),
                "is_positive": self.impact.is_positive,
                "verdict": self.impact.verdict,
                "assumptions": self.impact.assumptions.source,
                "deflection": self.impact.assumptions.deflection,
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
                 rates: econ.Rates = econ.DEFAULT_RATES,
                 roster: Roster = DEFAULT_ROSTER,
                 assumptions: imp.ImpactAssumptions = imp.DEFAULT_ASSUMPTIONS
                 ) -> None:
        self.heads = heads
        self.roster = roster
        self.assumptions = assumptions
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
        # Computed once: it is the same for every use case, and it is the
        # check most likely to change how a number should be read.
        self._holdout = self.evaluate_holdout()

    # -- construction ----------------------------------------------------

    @classmethod
    def from_defaults(cls, corpus_path: Optional[str] = None,
                      runs_path: Optional[str] = None,
                      rates: Optional[econ.Rates] = None,
                      roster: Optional[Roster] = None) -> "Predictor":
        """Fit everything from the data committed with the package.

        The roster decides which staffing heads exist, and a role the corpus
        has no column for simply gets no head — recorded in
        :attr:`MultiHeadModel.unfitted` rather than silently absent, because a
        missing role looks exactly like a role nobody needs.
        """
        from .roles import load_roster

        roster = roster or load_roster()
        rates = rates or econ.Rates(roster=roster)
        corpus = load_engagements(corpus_path)
        token_model = select_model(load_runs(runs_path or default_runs_path()))
        index = index_from_engagements(corpus)

        train = [e for e in corpus if not e.held_out]
        rows = [row_of(e, index.features_for(e.as_usecase())) for e in train]
        outcomes = build_outcomes(roster)
        order = build_order(roster)

        observations = {slug: [e.observed(slug) for e in train]
                        for slug in order}
        # The days head for a role sees only the engagements that used it.
        subsets = {role.days_outcome: [e.used(role.slug) for e in train]
                   for role in roster}
        heads = multihead.fit(rows, observations, outcomes, order, subsets)
        return cls(heads, token_model, index, corpus, rates, roster)

    # -- prediction ------------------------------------------------------

    def feature_row(self, usecase: UseCase) -> FeatureRow:
        return row_of(usecase, self.index.features_for(usecase))

    def forecast(self, usecase: UseCase, neighbours: int = 5) -> Forecast:
        row = self.feature_row(usecase)
        lineage = row.lineage

        tokens = self._token_estimate(usecase)
        estimates = self.heads.estimate_all(row)
        staffing = self._staffing(usecase, row)

        economics = econ.compute(
            contract_value=estimates["contract_value"].value,
            win_probability=estimates["win_probability"].value,
            staff_days={r.role.slug: r.expected_days for r in staffing},
            tokens=tokens.tokens, rates=self.rates,
            monthly_runs=usecase.monthly_runs,
        )
        impact = imp.compute(
            counts=usecase.counts, monthly_runs=usecase.monthly_runs,
            annual_token_cost=economics.annual_token_cost,
            delivery_cost=economics.delivery_cost,
            assumptions=self.assumptions,
        )
        return Forecast(
            usecase=usecase, lineage=lineage, tokens=tokens,
            estimates=estimates, staffing=staffing, economics=economics,
            impact=impact,
            neighbours=self.index.nearest(usecase, neighbours),
            warnings=self._warnings(usecase, row),
        )

    def _staffing(self, usecase: UseCase, row: FeatureRow) -> List[RoleEstimate]:
        """The staffing plan, with the PM's knowledge taking precedence.

        Three sources, in strict order:

        1. **Hand-entered days** win outright. A negotiated number is not an
           estimate and should not be averaged with one.
        2. **A named roster** for this use case makes those roles certain and
           every other role absent. The PM knowing a data scientist is needed
           is better evidence than the rate at which comparable work used one.
        3. **The history**, otherwise: presence from the ``_used`` head, days
           from the ``_days`` head.
        """
        named = None
        if usecase.roles is not None:
            named = set(self.roster.validate(usecase.roles))
        out: List[RoleEstimate] = []

        for role in self.roster:
            if role.slug in usecase.role_days:
                days = float(usecase.role_days[role.slug])
                if days <= 0:
                    continue
                out.append(RoleEstimate(role, 1.0, days, days, days,
                                        source="entered"))
                continue

            days_head = self.heads.heads.get(role.days_outcome)
            if days_head is None:
                # No history for the role. If the PM named it we must say we
                # cannot price it rather than quietly leaving it off the bill.
                if named is not None and role.slug in named:
                    out.append(RoleEstimate(role, 1.0, 0.0, 0.0, 0.0,
                                            source="no history"))
                continue

            days = days_head.predict(row)
            low, high = days_head.interval(row)

            if named is not None:
                if role.slug not in named:
                    continue
                probability, source = 1.0, "named"
            else:
                probability, source = self._presence(role, row)
                if probability < ROLE_FLOOR:
                    continue
            out.append(RoleEstimate(role, probability, days, low, high, source))
        return out

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

    def _presence(self, role: Role, row: FeatureRow) -> Tuple[float, str]:
        """How likely this role is, and where that number came from.

        A presence head that cannot beat its own base rate on held-out data is
        *replaced* by that base rate rather than merely flagged. For a role
        staffed on 97% of engagements there is nothing to predict and the base
        rate is the better answer; for one in the ambiguous middle it is an
        admission, and the plan says which of the two it is rather than
        dressing both up as a prediction.
        """
        head = self.heads.heads.get(role.used_outcome)
        if head is None:
            return 1.0, "assumed"
        if head.beats_baseline:
            return head.predict(row), "predicted"
        return head.baseline_value, "base rate"

    # -- honesty ---------------------------------------------------------

    def _warnings(self, usecase: UseCase, row: FeatureRow) -> List[str]:
        out: List[str] = []
        if usecase.monthly_runs:
            out.append(
                f"The annual impact assumes a person currently spends "
                f"{imp.manual_minutes(usecase.counts, self.assumptions):.0f} "
                f"minutes on one of these by hand and that the pipeline "
                f"removes {self.assumptions.deflection:.0%} of that. Both are "
                f"placeholders — the benefit case moves proportionally with "
                f"them, so replace them before quoting it.")
        if self._synthetic:
            out.append(
                "The value and impact heads are fitted on a SYNTHETIC "
                "engagement corpus. Treat every figure except the token budget "
                "as a demonstration of the machinery, not as evidence.")
        if usecase.encoder == "heuristic":
            out.append(
                "Encoded by keyword match, not by a model. The brick counts "
                "are a guess at scale rather than a reading of intent.")
        out.extend(usecase.assumptions)
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
            if head is None:
                continue
            holdout = self._holdout.get(slug)
            if not head.beats_baseline:
                out.append(
                    f"The {OUTCOMES[slug].name.lower()} head does not beat its "
                    f"own baseline in cross-validation; read it as the corpus "
                    f"base rate, not as a prediction about this use case.")
            elif holdout is not None and not holdout.beats_baseline:
                out.append(
                    f"The {OUTCOMES[slug].name.lower()} head has skill in "
                    f"cross-validation but loses to its own base rate on the "
                    f"{holdout.n} most recent engagements "
                    f"({holdout.score:.3f} against {holdout.baseline:.3f}). "
                    f"It may not be generalising forward — treat the point "
                    f"estimate as weaker than its interval suggests.")

        named = set(usecase.roles or ())
        for slug in sorted(named):
            role = self.roster.get(slug)
            if role is not None and role.days_outcome in self.heads.unfitted:
                out.append(
                    f"{role.name} is named on this use case but the corpus has "
                    f"no history for it ({self.heads.unfitted[role.days_outcome]}), "
                    f"so it is on the plan at zero days and zero cost. Enter "
                    f"the days by hand or the estimate is missing that person.")
        if usecase.roles is not None:
            out.append(
                f"The team was specified rather than predicted: "
                f"{len(named)} role(s) named, so their presence is taken as "
                f"certain and every other role on the roster is excluded.")
        return out

    # -- evidence --------------------------------------------------------

    def evaluate_holdout(self) -> Dict[str, HoldoutScore]:
        """Score each head on engagements it was never fitted on.

        The hold-out is the most recent slice of the corpus, not a random
        subset: the question a PM is asking is about the next engagement, and
        predicting forward in time is a strictly harder test than interpolating
        within a history you have already seen.

        The head's own baseline is scored on the same rows. Cross-validation
        can say a head has skill and the forward hold-out can disagree — which
        is the single most useful thing this function can find, and it is only
        visible with both numbers side by side.
        """
        held = [e for e in self.corpus if e.held_out]
        if not held:
            return {}
        out: Dict[str, HoldoutScore] = {}
        for slug, head in self.heads.heads.items():
            binary = head.outcome.binary
            rows, errs, base = [], [], []
            for eng in held:
                if slug.endswith("_days") and slug != "calendar_days":
                    # days-when-needed is only defined where the role was used
                    if not eng.used(slug[:-len("_days")]):
                        continue
                actual = eng.observed(slug)
                if not binary and not actual:
                    continue
                row = row_of(eng, self.index.features_for(eng.as_usecase()))
                rows.append(row)
                pred = head.predict(row)
                if binary:
                    errs.append((pred - actual) ** 2)
                    base.append((head.baseline_value - actual) ** 2)
                else:
                    errs.append(abs(pred - actual) / actual)
                    base.append(abs(head.baseline_value - actual) / actual)
            if errs:
                out[slug] = HoldoutScore(
                    slug, sum(errs) / len(errs), sum(base) / len(base),
                    len(errs), "brier" if binary else "mape")
        return out
