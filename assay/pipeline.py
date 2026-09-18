"""The whole loop in one place: design, run, fit, calibrate, quote, invoice.

Every piece of this already existed as a library call, but only the test suite knew the
order to call them in. That is a real gap in an evidence package: a reader who cannot
reproduce the loop without reading `conftest.py` cannot check the claim, and a loop that
lives only in tests drifts from the one the README describes.

Two things are enforced here rather than left to the caller:

* **Calibration never sees its own fit.** The conformal band is calibrated on the
  out-of-fold predictions of the chosen form, not on refitted in-sample residuals. An
  in-sample band is narrower and wrong, and it is wrong in the flattering direction.
* **Per-brick detail follows the verdict, not the fit.** When selection does not choose a
  decoder, `show_per_brick` is false all the way through quote and invoice. The estimator
  may still ship; the per-brick prices do not, because nothing established them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

from assay.conformal import Conformal, calibrate
from assay.corpus import SealedCorpus
from assay.costmodel import CostModel
from assay.evidence import EvidenceClass
from assay.pricing import Pricing
from assay.schemas import Run, Tier, read_runs
from assay.select import Selection, select_form
from assay.vocabulary import Vocabulary


def boot_estimate(runs: Sequence[Run]) -> float:
    """The start-up toll: the median of the accepted null probes, and nothing else.

    This is the most load-bearing scalar in the project -- every variable-portion metric
    is measured against it -- so it has exactly one definition. Deriving it from anything
    other than a null probe (the minimum observed cost, say) would let a cheap task stand
    in for the empty task, and the variable portion would then be measured against a
    number that already contains work.
    """
    nulls = [r.billable_units for r in runs if r.tier is Tier.NULL and r.accepted]
    if not nulls:
        raise ValueError(
            "no accepted null probe: the start-up toll was never measured, so the "
            "variable portion of every other run is undefined"
        )
    return median(nulls)


@dataclass
class Fitted:
    """A fitted estimator and everything needed to defend it."""

    model: CostModel
    conformal: Conformal
    selection: Selection
    boot: float
    evidence_class: EvidenceClass

    @property
    def show_per_brick(self) -> bool:
        """Per-brick prices ship only if the ladder actually chose a decoder."""
        return self.selection.bricks_helped

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "model": d / "model.json",
            "conformal": d / "conformal.json",
            "fit": d / "fit.json",
        }
        self.model.save(paths["model"])
        self.conformal.save(paths["conformal"])
        paths["fit"].write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return paths

    def as_dict(self) -> dict[str, object]:
        lift = self.selection.decoder_lift()
        return {
            "evidence_class": self.evidence_class.value,
            "boot": round(self.boot, 1),
            "chosen_form": self.selection.chosen,
            "chose_a_decoder": self.selection.bricks_helped,
            "best_baseline": self.selection.best_baseline,
            "show_per_brick": self.show_per_brick,
            "bar_pct": self.selection.bar_pct,
            "lift_over_baseline": lift.as_dict() if lift else None,
            "notes": list(self.selection.notes),
            "model": self.model.to_dict(),
            "conformal": self.conformal.as_dict(),
        }

    def render(self) -> str:
        lift = self.selection.decoder_lift()
        head = [
            f"FIT  (evidence class: {self.evidence_class.value})",
            "",
            f"  start-up toll (null median) : {self.boot:,.0f} units",
            f"  rows fitted                 : {self.model.n_fitted}",
            f"  best baseline               : {self.selection.best_baseline}",
            f"  chosen form                 : {self.selection.chosen}"
            f"{'  (decoder)' if self.selection.bricks_helped else '  (baseline)'}",
        ]
        if lift is not None:
            head.append(
                f"  lift over baseline          : {lift.lift_pct:.1f}%  "
                f"(bar {self.selection.bar_pct:.0f}%, "
                f"bootstrap lo {lift.bootstrap_lo:.1f}%) -> "
                f"{'passes' if lift.passed else 'does not pass'}"
            )
        if not self.selection.bricks_helped:
            head.append(
                "\n  Bricks did not clear the bar. The estimator still quotes; the "
                "per-brick prices are withheld, because nothing established them."
            )
        head.append(
            f"\n  {self.conformal.level:.0%} band: {self.conformal.method.value}, "
            f"q={self.conformal.q:,.4g}, calibrated on {self.conformal.n_calibration} "
            "out-of-fold rows"
        )
        for note in self.selection.notes:
            head.append(f"  note: {note}")
        if self.evidence_class is not EvidenceClass.FEASIBILITY:
            head.append(
                "\n  This fit cannot move a gate: it is not feasibility evidence."
            )
        return "\n".join(head)


def fit_pipeline(
    runs: Sequence[Run],
    vocab: Vocabulary,
    *,
    bar_pct: float = 15.0,
    level: float = 0.90,
    iters: int = 2000,
    seed: int = 1337,
) -> Fitted:
    """Select a form, fit it, and calibrate its band on out-of-fold predictions."""
    accepted = [r for r in runs if r.accepted]
    if not accepted:
        raise ValueError("every run was excluded; there is nothing to fit")

    classes = {r.evidence_class for r in accepted}
    evidence = (
        EvidenceClass.FEASIBILITY
        if classes == {EvidenceClass.FEASIBILITY}
        else EvidenceClass.PIPELINE_ONLY
    )

    boot = boot_estimate(runs)
    selection = select_form(runs, vocab, boot, bar_pct=bar_pct, iters=iters, seed=seed)
    model = CostModel.fit(runs, selection.chosen, vocab, boot)

    held = selection.results[selection.chosen]
    conformal = calibrate(held.actual, held.predicted, level=level, floor=boot)

    return Fitted(
        model=model,
        conformal=conformal,
        selection=selection,
        boot=boot,
        evidence_class=evidence,
    )


def load_runs(path: str | Path, pricing: Pricing) -> list[Run]:
    return read_runs(path, pricing)


@dataclass(frozen=True)
class Priced:
    """A fit reloaded from disk, with the verdict that governs how much of it may ship.

    ``show_per_brick`` is deliberately read back from ``fit.json`` rather than inferred
    from the model file. The coefficients exist either way; whether anyone is entitled to
    quote them is a conclusion of the selection ladder, and a reloaded model that
    forgot the conclusion would silently start publishing per-brick prices the evidence
    never supported.
    """

    model: CostModel
    conformal: Conformal
    boot: float
    show_per_brick: bool
    evidence_class: EvidenceClass
    chosen_form: str

    @classmethod
    def load(cls, out_dir: str | Path) -> "Priced":
        d = Path(out_dir)
        fit_path = d / "fit.json"
        if not fit_path.exists():
            raise FileNotFoundError(
                f"{fit_path} not found -- run `ty fit --out-dir {d}` first; a model file "
                "on its own does not say whether per-brick prices may be shown"
            )
        meta = json.loads(fit_path.read_text(encoding="utf-8"))
        return cls(
            model=CostModel.load(d / "model.json"),
            conformal=Conformal.load(d / "conformal.json"),
            boot=float(meta["boot"]),
            show_per_brick=bool(meta["show_per_brick"]),
            evidence_class=EvidenceClass(meta["evidence_class"]),
            chosen_form=str(meta["chosen_form"]),
        )


def simulate_campaign(
    vocab: Vocabulary,
    corpus: SealedCorpus,
    out_path: str | Path,
    *,
    pricing: Pricing,
    mode: str = "brick",
    seed: int = 1337,
    campaign_id: str | None = None,
):
    """Design and execute a fit campaign against the simulator.

    Imported lazily so the fit/quote/invoice half of this module stays usable without
    dragging the harness in. The return is the campaign report; the runs are on disk.
    """
    from assay.harness.campaign import design_campaign
    from assay.harness.mock_adapter import MockAdapter, MockMode
    from assay.harness.runner import run_campaign

    m = MockMode(mode)
    plan = design_campaign(
        vocab, corpus, campaign_id=campaign_id or f"sim_{m.value}", seed=seed
    )
    adapter = MockAdapter(m, seed=seed)
    report = run_campaign(
        plan.probes,
        adapter,
        out_path,
        campaign_id=plan.campaign_id,
        pricing=pricing,
        corpus=corpus,
        runtime_hash=f"simulated-{m.value}",
        expected_model=adapter.model,
    )
    return plan, report
