"""Scoring an encoder -- in vectors, and then in money.

Exact-match accuracy treats every confusion as equal, and they are emphatically not.
Reading a Retrieve as a Classify is, on the reference marginals, a five-thousand-token
mistake; reading a Draft as a Review costs essentially nothing. So the encoder is scored
twice: once on the vector, and once through the frozen cost model, which is the number
that actually decides whether quoting can be automated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from assay.costmodel import CostModel
from assay.decompose.parser import Decomposition
from assay.vocabulary import Vocabulary


@dataclass
class BrickScore:
    brick: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    abs_error: int = 0
    n: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mae(self) -> float:
        return self.abs_error / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "brick": self.brick,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mae": round(self.mae, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
        }


@dataclass
class EncoderReport:
    source: str
    n: int
    exact_match_pct: float
    macro_f1: float
    mae: float
    parse_failure_pct: float
    per_brick: dict[str, BrickScore]
    confusions: dict[tuple[str, str], int] = field(default_factory=dict)
    priced_error_pct: float | None = None
    priced_error_note: str = ""

    def worst_brick(self) -> BrickScore:
        return min(self.per_brick.values(), key=lambda s: s.f1)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "n": self.n,
            "exact_match_pct": round(self.exact_match_pct, 2),
            "macro_f1": round(self.macro_f1, 4),
            "mae": round(self.mae, 4),
            "parse_failure_pct": round(self.parse_failure_pct, 2),
            "priced_error_pct": (
                round(self.priced_error_pct, 3) if self.priced_error_pct is not None else None
            ),
            "priced_error_note": self.priced_error_note,
            "per_brick": {k: v.as_dict() for k, v in self.per_brick.items()},
            "confusions": {f"{a}->{b}": n for (a, b), n in sorted(self.confusions.items())},
        }

    def summary(self) -> str:
        worst = self.worst_brick()
        priced = (
            "not priced (no model supplied)"
            if self.priced_error_pct is None
            else f"{self.priced_error_pct:.1f}%"
        )
        return (
            f"{self.source}: exact {self.exact_match_pct:.1f}%  "
            f"macro F1 {self.macro_f1:.2f}  MAE {self.mae:.2f}  "
            f"priced error {priced}  worst brick {worst.brick} (F1 {worst.f1:.2f})"
        )


def priced_error_units(
    gold: Mapping[str, int], predicted: Mapping[str, int], model: CostModel
) -> float:
    """Absolute cost of a wrong vector, in billable units.

    This is the number that compares two *different* mistakes: reading a Retrieve as a
    Classify costs thousands of units; reading a Draft as a Review costs almost nothing,
    and exact-match accuracy scores them identically.
    """
    gold_var = sum(model.marginals.get(b, 0.0) * n for b, n in gold.items())
    pred_var = sum(model.marginals.get(b, 0.0) * n for b, n in predicted.items())
    return abs(pred_var - gold_var)


def priced_error(
    gold: Mapping[str, int],
    predicted: Mapping[str, int],
    model: CostModel,
    *,
    context_bytes: int = 0,
) -> float:
    """The same error as a share of the gold variable cost.

    Measured on the variable portion only: the start-up toll is paid whatever the encoder
    says, so including it would flatter every encoder equally and hide the differences.

    Beware this figure on a *single* cheap request -- if the gold vector is all
    below-noise bricks the denominator approaches zero and the percentage explodes. Across
    a set, use the aggregate in :func:`evaluate`, which divides summed error by summed
    cost rather than averaging per-request percentages.
    """
    gold_var = sum(model.marginals.get(b, 0.0) * n for b, n in gold.items())
    error = priced_error_units(gold, predicted, model)
    if abs(gold_var) < 1e-9:
        return 0.0 if error < 1e-9 else 100.0
    return error / abs(gold_var) * 100.0


def evaluate(
    gold: Sequence[Mapping[str, int]],
    predicted: Sequence[Decomposition | None],
    vocab: Vocabulary,
    *,
    source: str = "llm",
    model: CostModel | None = None,
) -> EncoderReport:
    """``None`` in ``predicted`` means the encoder failed to produce a usable vector.

    Failures are counted, never dropped and never replaced. An encoder that parses 80% of
    the time and is perfect on those is not an 80%-accurate encoder in any sense that
    matters to someone waiting for a quote.
    """
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted differ in length")

    per_brick = {name: BrickScore(name) for name in vocab.names}
    confusions: dict[tuple[str, str], int] = {}
    exact = 0
    abs_total = 0
    failures = 0
    priced_abs = 0.0
    priced_gold = 0.0

    for g, p in zip(gold, predicted):
        if p is None:
            failures += 1
            for name in vocab.names:
                score = per_brick[name]
                score.n += 1
                if g.get(name, 0):
                    score.fn += 1
                    score.abs_error += g[name]
                abs_total += g.get(name, 0)
            if model is not None:
                # A failure produces no quote at all, so it is charged the whole job.
                gold_var = sum(model.marginals.get(b, 0.0) * n for b, n in g.items())
                priced_abs += abs(gold_var)
                priced_gold += abs(gold_var)
            continue

        if all(p.units.get(n, 0) == g.get(n, 0) for n in vocab.names):
            exact += 1

        missed: list[str] = []
        invented: list[str] = []
        for name in vocab.names:
            score = per_brick[name]
            score.n += 1
            gv, pv = g.get(name, 0), p.units.get(name, 0)
            score.abs_error += abs(pv - gv)
            abs_total += abs(pv - gv)
            if gv and pv:
                score.tp += 1
            elif pv and not gv:
                score.fp += 1
                invented.append(name)
            elif gv and not pv:
                score.fn += 1
                missed.append(name)

        for a in missed:
            for b in invented:
                confusions[(a, b)] = confusions.get((a, b), 0) + 1

        if model is not None:
            priced_abs += priced_error_units(g, p.units, model)
            priced_gold += abs(sum(model.marginals.get(b, 0.0) * n for b, n in g.items()))

    n = len(gold)
    return EncoderReport(
        source=source,
        n=n,
        exact_match_pct=exact / n * 100.0 if n else 0.0,
        macro_f1=sum(s.f1 for s in per_brick.values()) / len(per_brick),
        mae=abs_total / (n * len(vocab.names)) if n else 0.0,
        parse_failure_pct=failures / n * 100.0 if n else 0.0,
        per_brick=per_brick,
        confusions=confusions,
        priced_error_pct=(
            priced_abs / priced_gold * 100.0 if model is not None and priced_gold > 1e-9 else None
        ),
        priced_error_note=(
            "" if model else "supply the frozen cost model to price encoder mistakes"
        ),
    )


def stability(runs: Sequence[Decomposition], vocab: Vocabulary) -> float:
    """Share of repeat encodings of the same request that agree with the first.

    A model at temperature 0 that still moves between runs cannot support a quote anyone
    can rely on, and paraphrase drift is the same defect in a different coat.
    """
    if len(runs) < 2:
        return 100.0
    first = runs[0]
    agree = sum(
        1 for r in runs[1:] if all(r.units.get(n, 0) == first.units.get(n, 0) for n in vocab.names)
    )
    return agree / (len(runs) - 1) * 100.0
