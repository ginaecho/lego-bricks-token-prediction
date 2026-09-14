"""Which side of the bill is the model actually predicting?

``billable_units = prompt_tokens + ratio * completion_tokens`` fuses two processes that
have almost nothing to do with each other. Prompt tokens are mostly *material*: how much
context was mounted, plus a fixed wrapper. Completion tokens are the *answer*: how much
the model chose to write, which is what the request asked for. On the reference pricing an
output token costs five input tokens, so the smaller count carries disproportionate money.

Reporting one blended error hides the question that matters. A decoder can look mediocre
overall while predicting the answer side very well and the material side not at all --
which is not a failure of brick decomposition, it is a reason to quote the two separately.
The opposite is worse: a decoder can look excellent purely because prompt tokens are
nearly deterministic in ``context_bytes``, and ``bytes`` alone would have done the same
job. The blend cannot distinguish those, and both are things a reader needs to know.

Nothing here fits a new model or introduces a new form. It re-runs the *same* grouped
cross-validation over the same ladder against a retargeted ``y``, so the two sides are
evaluated by exactly the code path that evaluates the whole. This is a diagnostic: it
moves no gate and appears in no policy.

One limitation, stated because it is easy to miss. :class:`~assay.harness.mock_adapter.MockAdapter`
derives completion tokens as a fixed share of the same generated total, so under the
simulator the two sides are one signal scaled and they score alike whatever form is used.
A flat table from mock runs is therefore evidence about the simulator, not about whether
the split matters. Only a real dispatch can populate this table meaningfully.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Sequence

from assay.pricing import Pricing
from assay.schemas import Run, Tier
from assay.select import CVResult, cross_validate
from assay.vocabulary import Vocabulary


class Side(str, Enum):
    PROMPT = "prompt"
    """Material sent. Mostly context bytes plus a fixed harness wrapper."""

    COMPLETION = "completion"
    """The answer written back, priced in input-token-equivalents so the two sides add up."""

    COMBINED = "combined"
    """``prompt + ratio * completion`` -- the billable target everything else uses."""


def side_units(run: Run, side: Side, pricing: Pricing) -> float:
    if side is Side.PROMPT:
        return float(run.prompt_tokens)
    if side is Side.COMPLETION:
        return pricing.ratio * run.completion_tokens
    return pricing.billable_units(run.prompt_tokens, run.completion_tokens)


def retarget(runs: Sequence[Run], side: Side, pricing: Pricing) -> list[Run]:
    """Clone runs with ``billable_units`` replaced by one side of the bill.

    Cloning rather than parameterising the scorer is deliberate: the side then travels
    through `cross_validate`, `score` and the lift bar untouched, so a side cannot be
    scored more gently than the whole.
    """
    out: list[Run] = []
    for run in runs:
        clone = copy.deepcopy(run)
        clone.billable_units = side_units(run, side, pricing)
        out.append(clone)
    return out


def side_boot(runs: Sequence[Run], side: Side, pricing: Pricing) -> float:
    """The empty-task cost on one side, from the null probes and nowhere else."""
    nulls = [
        side_units(r, side, pricing)
        for r in runs
        if r.tier is Tier.NULL and r.accepted
    ]
    if not nulls:
        raise ValueError("no accepted null probe; the start-up toll cannot be measured")
    return median(nulls)


@dataclass
class SideReport:
    """One form, scored three ways. The three are reported together or not at all."""

    form: str
    prompt: CVResult
    completion: CVResult
    combined: CVResult
    boot_prompt: float
    boot_completion: float
    completion_share_of_bill_pct: float
    """How much of the average bill the answer side accounts for, after the price ratio.

    Without it the two error columns are not comparable: 40% error on a side worth 3% of
    the bill is a rounding difference, and 12% on a side worth half of it is not.
    """

    @property
    def money_weighted_note(self) -> str:
        if self.completion.pooled.vwape < self.prompt.pooled.vwape:
            harder, easier = "material", "answer"
            gap = self.prompt.pooled.vwape - self.completion.pooled.vwape
        else:
            harder, easier = "answer", "material"
            gap = self.completion.pooled.vwape - self.prompt.pooled.vwape
        return (
            f"the {easier} side is predicted {gap:.1f} points better than the {harder} "
            f"side; the answer side carries {self.completion_share_of_bill_pct:.1f}% of "
            "the bill"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "form": self.form,
            "boot_prompt": round(self.boot_prompt, 2),
            "boot_completion": round(self.boot_completion, 2),
            "completion_share_of_bill_pct": round(self.completion_share_of_bill_pct, 3),
            "prompt": self.prompt.as_dict(),
            "completion": self.completion.as_dict(),
            "combined": self.combined.as_dict(),
            "note": self.money_weighted_note,
        }

    def render(self) -> str:
        rows = [
            ("material (prompt)", self.prompt, self.boot_prompt),
            ("answer (completion)", self.completion, self.boot_completion),
            ("combined (billable)", self.combined, self.boot_prompt + self.boot_completion),
        ]
        lines = [
            f"SIDES OF THE BILL  ({self.form}, grouped CV)",
            "",
            f"  {'side':<22}{'toll':>12}{'MAPE (total)':>16}{'WAPE (variable)':>18}",
            f"  {'-' * 68}",
        ]
        lines.extend(
            f"  {label:<22}{boot:>12,.0f}{cv.pooled.mape_total:>15.2f}%"
            f"{cv.pooled.vwape:>17.2f}%"
            for label, cv, boot in rows
        )
        lines += [
            "",
            f"  answer side is {self.completion_share_of_bill_pct:.1f}% of the average bill",
            f"  {self.money_weighted_note}",
            "",
            "  Diagnostic only. No gate reads this table.",
        ]
        return "\n".join(lines)


def score_sides(
    runs: Sequence[Run], form_key: str, vocab: Vocabulary, pricing: Pricing
) -> SideReport:
    """Cross-validate one form against the material side, the answer side, and the blend."""
    scored: dict[Side, CVResult] = {}
    boots: dict[Side, float] = {}
    for side in (Side.PROMPT, Side.COMPLETION, Side.COMBINED):
        boot = side_boot(runs, side, pricing)
        boots[side] = boot
        scored[side] = cross_validate(retarget(runs, side, pricing), form_key, vocab, boot)

    total = sum(side_units(r, Side.COMBINED, pricing) for r in runs)
    answer = sum(side_units(r, Side.COMPLETION, pricing) for r in runs)

    return SideReport(
        form=form_key,
        prompt=scored[Side.PROMPT],
        completion=scored[Side.COMPLETION],
        combined=scored[Side.COMBINED],
        boot_prompt=boots[Side.PROMPT],
        boot_completion=boots[Side.COMPLETION],
        completion_share_of_bill_pct=answer / total * 100.0 if total else 0.0,
    )
