"""Batching: is one combined request cheaper than N separate ones?

The saving is always split into two parts, and the split is the whole point:

``toll_component``
    The start-up cost paid once instead of *N* times. This is arithmetic, predictable
    before any dispatch, and it is not a discovery.

``residual_component``
    Whatever is saved beyond that -- genuine sub-additivity in the variable work.

The reference reported savings of 64-71% on a runtime whose empty task cost 29,821 tokens.
On a lean harness the same *mechanism* can produce a far smaller percentage simply because
the toll being amortised is smaller. Without the decomposition, "we missed the 40% bar"
and "there is no saving here" look identical in the data; with it, the first is a Narrow
verdict and the second is a Stop.

Cost and quality are computed separately and combined only at the gate, because a cheaper
worse answer must never be able to pass on price alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from assay.grading import QualityReport
from assay.metrics import BootstrapResult, bootstrap
from assay.schemas import Run, Split


@dataclass(frozen=True)
class Bundle:
    bundle_id: str
    separate_units: float
    batched_units: float
    n_tasks: int

    @property
    def saving_pct(self) -> float:
        if self.separate_units <= 0:
            return 0.0
        return (self.separate_units - self.batched_units) / self.separate_units * 100.0

    def toll_component(self, boot: float) -> float:
        """The saving that is just the start-up toll, paid once instead of n times."""
        if self.separate_units <= 0:
            return 0.0
        return (self.n_tasks - 1) * boot / self.separate_units * 100.0

    def residual_component(self, boot: float) -> float:
        """Measured saving minus the predictable part. This is the interesting number."""
        return self.saving_pct - self.toll_component(boot)

    def as_dict(self, boot: float) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "n_tasks": self.n_tasks,
            "separate_units": round(self.separate_units, 2),
            "batched_units": round(self.batched_units, 2),
            "saving_pct": round(self.saving_pct, 3),
            "toll_component_pct": round(self.toll_component(boot), 3),
            "residual_component_pct": round(self.residual_component(boot), 3),
        }


def collect_bundles(runs: Sequence[Run]) -> list[Bundle]:
    """Pair the two arms of every bundle. A bundle missing an arm is dropped, not patched."""
    arms: dict[str, dict[str, list[Run]]] = {}
    for run in runs:
        if run.split is not Split.BATCHING or not run.accepted or not run.bundle_id:
            continue
        arm = "batched" if run.probe_id.endswith("-batch") else "separate"
        arms.setdefault(run.bundle_id, {}).setdefault(arm, []).append(run)

    bundles: list[Bundle] = []
    for bundle_id, sides in sorted(arms.items()):
        separate = sides.get("separate", [])
        batched = sides.get("batched", [])
        if not separate or len(batched) != 1:
            continue
        bundles.append(
            Bundle(
                bundle_id=bundle_id,
                separate_units=sum(r.billable_units for r in separate),
                batched_units=batched[0].billable_units,
                n_tasks=len(separate),
            )
        )
    return bundles


@dataclass
class BatchingReport:
    bundles: list[Bundle]
    boot: float
    aggregate_saving_pct: float
    two_task_saving_pct: float | None
    bootstrap: BootstrapResult | None
    n_positive: int
    mean_toll_component: float
    mean_residual_component: float
    quality: QualityReport | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def tested(self) -> bool:
        return len(self.bundles) >= 8

    @property
    def cost_gate_passes(self) -> bool:
        if not self.tested or self.bootstrap is None:
            return False
        return (
            self.aggregate_saving_pct >= 40.0
            and self.bootstrap.lo >= 20.0
            and self.n_positive >= max(7, round(0.83 * len(self.bundles)))
        )

    def interpretation(self) -> str:
        if not self.tested:
            return (
                f"untested -- {len(self.bundles)} bundles is below the floor of 8; "
                "recorded as unproven rather than as an absence of saving"
            )
        if self.aggregate_saving_pct <= 0:
            return "no saving: batching did not reduce cost"
        if self.cost_gate_passes:
            return "saving clears the bar"
        return (
            f"saving is real but below the bar: {self.mean_toll_component:.1f} points of it "
            f"is the start-up toll amortised and {self.mean_residual_component:+.1f} points "
            "is genuine sub-additivity. A small toll can miss a 40% bar with the mechanism "
            "fully intact -- this is a Narrow result, not an absent effect."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "n_bundles": len(self.bundles),
            "tested": self.tested,
            "aggregate_saving_pct": round(self.aggregate_saving_pct, 3),
            "two_task_saving_pct": (
                round(self.two_task_saving_pct, 3) if self.two_task_saving_pct is not None else None
            ),
            "bootstrap": self.bootstrap.as_dict() if self.bootstrap else None,
            "n_positive": self.n_positive,
            "mean_toll_component_pct": round(self.mean_toll_component, 3),
            "mean_residual_component_pct": round(self.mean_residual_component, 3),
            "cost_gate_passes": self.cost_gate_passes,
            "interpretation": self.interpretation(),
            "quality": self.quality.as_dict() if self.quality else None,
            "bundles": [b.as_dict(self.boot) for b in self.bundles],
            "notes": self.notes,
        }

    def observations(self) -> dict[str, float | None]:
        if not self.tested:
            return {"M11_batching_saving_pct": None, "M12_batching_quality_lb": None}
        return {
            "M11_batching_saving_pct": self.aggregate_saving_pct,
            "M12_batching_quality_lb": (
                self.quality.delta_lower_90
                if self.quality and self.quality.conclusive
                else None
            ),
        }


def analyse(
    bundles: Sequence[Bundle],
    *,
    boot: float,
    quality: QualityReport | None = None,
    iters: int = 10_000,
    seed: int = 1337,
) -> BatchingReport:
    bundles = list(bundles)
    notes: list[str] = []

    if not bundles:
        return BatchingReport(
            bundles=[], boot=boot, aggregate_saving_pct=0.0, two_task_saving_pct=None,
            bootstrap=None, n_positive=0, mean_toll_component=0.0, mean_residual_component=0.0,
            quality=quality, notes=["no complete bundles"],
        )

    total_sep = sum(b.separate_units for b in bundles)
    total_bat = sum(b.batched_units for b in bundles)
    aggregate = (total_sep - total_bat) / total_sep * 100.0 if total_sep else 0.0

    two = [b for b in bundles if b.n_tasks == 2]
    two_saving = None
    if two:
        s = sum(b.separate_units for b in two)
        two_saving = (s - sum(b.batched_units for b in two)) / s * 100.0 if s else 0.0

    # The bundle is the unit of resampling. Tasks inside a bundle shared one dispatch and
    # are not independent, so resampling tasks would understate the interval.
    res = bootstrap(
        bundles,
        lambda sample: (
            (sum(b.separate_units for b in sample) - sum(b.batched_units for b in sample))  # type: ignore[union-attr]
            / sum(b.separate_units for b in sample) * 100.0  # type: ignore[union-attr]
        ),
        iters=iters,
        level=0.95,
        seed=seed,
    )

    if quality is not None and not quality.conclusive:
        notes.append("cost saving stands, but the quality arm was inconclusive")

    return BatchingReport(
        bundles=bundles,
        boot=boot,
        aggregate_saving_pct=aggregate,
        two_task_saving_pct=two_saving,
        bootstrap=res,
        n_positive=sum(1 for b in bundles if b.saving_pct > 0),
        mean_toll_component=sum(b.toll_component(boot) for b in bundles) / len(bundles),
        mean_residual_component=sum(b.residual_component(boot) for b in bundles) / len(bundles),
        quality=quality,
        notes=notes,
    )
