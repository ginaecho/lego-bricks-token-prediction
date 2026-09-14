"""The mechanical verdict.

Every gate is a declarative comparison against a bar that is written down *before* any
data exists, and the outcome is composed by rule. Nothing in this module knows what the
observed numbers mean or hopes for a particular answer -- which is the point. The policy
is hashed into the release manifest before the blind set is unsealed, so the bars cannot
move once the results are visible.

Three outcomes, all of them finished work:

``feasible``
    Bricks carry real predictive power over the baselines, intervals are usable, batching
    saves at equal quality.
``narrow``
    Start-up, context and total unit count are predictable, but per-brick structure adds
    little -- or intervals are unusable, or batching is unproven. Ship the smaller true
    thing: a harness-overhead estimator, with per-brick prices hidden.
``stop``
    No skill over baselines, unusable encoder, drift, or integrity failure. Publish the
    negative result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from assay.evidence import EvidenceClass


class Outcome(str, Enum):
    FEASIBLE = "feasible"
    NARROW = "narrow"
    STOP = "stop"

    @property
    def rank(self) -> int:
        return {"feasible": 0, "narrow": 1, "stop": 2}[self.value]


class Direction(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class GateSpec:
    id: str
    label: str
    direction: Direction
    pass_bar: float
    narrow_bar: float | None = None
    """Where the narrow band ends. ``None`` means there is no narrow band: miss the pass
    bar and the gate stops the project."""
    on_missing: Outcome = Outcome.NARROW
    """An unmeasured gate is not a passed gate. Untested batching is Narrow, not Feasible."""
    caps_only: bool = False
    """A failing cap holds the verdict down to Narrow but never Stops on its own.

    Which gates are caps is a design decision, not a detail. Stop is reserved for two
    situations: *nothing* predicts cost usefully, or the campaign's integrity failed.
    Everything else -- bricks adding nothing over a size baseline, intervals too wide to
    quote a range, batching unproven, the *encoder* being unusable -- still leaves a
    shippable, smaller, true thing, so those gates cap at Narrow. Without this, the "ship
    a runtime estimator" outcome the plan describes would be unreachable: any lift failure
    would Stop the project even when the baseline predicts cost perfectly well.

    Note what is deliberately *not* a cap. ``M13_alpha_macro`` stops, because if trained
    humans cannot agree what a brick is then the vocabulary is not a measurable construct
    and every unit count in the study is uninterpretable -- that is an integrity failure,
    not a smaller result. ``M7``/``M9`` by contrast only say the *automatic* encoder is
    unusable; the decoder underneath is untouched and still quotes from a hand-written
    vector.
    """

    def evaluate(self, value: float | None) -> "GateResult":
        if value is None:
            return GateResult(self, None, self.on_missing, "not measured")
        if self.direction is Direction.HIGHER_IS_BETTER:
            passed = value >= self.pass_bar
            narrow = self.narrow_bar is not None and value > self.narrow_bar
        else:
            passed = value <= self.pass_bar
            narrow = self.narrow_bar is not None and value <= self.narrow_bar

        if passed:
            return GateResult(self, value, Outcome.FEASIBLE, "meets the bar")
        if narrow:
            return GateResult(self, value, Outcome.NARROW, "inside the narrow band")
        outcome = Outcome.NARROW if self.caps_only else Outcome.STOP
        return GateResult(self, value, outcome, "misses the bar")


@dataclass(frozen=True)
class GateResult:
    spec: GateSpec
    value: float | None
    outcome: Outcome
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.spec.id,
            "label": self.spec.label,
            "value": self.value,
            "pass_bar": self.spec.pass_bar,
            "narrow_bar": self.spec.narrow_bar,
            "direction": self.spec.direction.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }

    def line(self) -> str:
        shown = "not measured" if self.value is None else f"{self.value:g}"
        arrow = "<=" if self.spec.direction is Direction.LOWER_IS_BETTER else ">="
        return (
            f"  [{self.outcome.value.upper():8}] {self.spec.id:22} "
            f"{shown:>14}  (pass {arrow} {self.spec.pass_bar:g})  {self.spec.label}"
        )


DEFAULT_POLICY: tuple[GateSpec, ...] = (
    GateSpec(
        "M1_noise_floor_pct", "replicate CV; above this, cheap bricks are unmeasurable",
        Direction.LOWER_IS_BETTER, 1.0, 3.0,
    ),
    GateSpec(
        "M2_drift_pct", "null-probe movement between opening and closing brackets",
        Direction.LOWER_IS_BETTER, 2.0, 5.0,
    ),
    GateSpec(
        "M5b_cv_lift_pct", "variable-portion lift of the chosen form over the best baseline (CV)",
        Direction.HIGHER_IS_BETTER, 20.0, 0.0, caps_only=True,
    ),
    GateSpec(
        "M6_blind_mape_pct", "total error on the sealed blind set",
        Direction.LOWER_IS_BETTER, 15.0, 30.0,
    ),
    GateSpec(
        "M6b_blind_lift_pct", "variable-portion lift over the best baseline, on blind",
        Direction.HIGHER_IS_BETTER, 15.0, 0.0, caps_only=True,
    ),
    GateSpec(
        "M7_encoder_exact_pct", "encoder exact-vector match",
        Direction.HIGHER_IS_BETTER, 85.0, 70.0, caps_only=True,
    ),
    GateSpec(
        "M9_e2e_mape_pct", "end-to-end quote error, encoder vector through the model",
        Direction.LOWER_IS_BETTER, 20.0, 35.0, caps_only=True,
    ),
    GateSpec(
        "M9b_coverage_hits", "90% interval hits out of 24 (a screen, reported with a Wilson CI)",
        Direction.HIGHER_IS_BETTER, 20.0, 17.0, caps_only=True,
    ),
    GateSpec(
        "M11_batching_saving_pct", "measured cost saving, batched versus separate",
        Direction.HIGHER_IS_BETTER, 40.0, 0.0, caps_only=True,
    ),
    GateSpec(
        "M12_batching_quality_lb", "lower 90% bound on quality change; a cheaper worse answer is not a saving",
        Direction.HIGHER_IS_BETTER, -5.0, None, caps_only=True,
    ),
    GateSpec(
        "M13_alpha_macro", "inter-rater agreement on brick vectors",
        Direction.HIGHER_IS_BETTER, 0.80, None,
    ),
    GateSpec(
        "M14_exclusion_pct", "non-transport exclusions",
        Direction.LOWER_IS_BETTER, 10.0, None,
    ),
    GateSpec(
        "M15_identifiable", "design supports a coefficient per brick (1 = yes)",
        Direction.HIGHER_IS_BETTER, 1.0, None, caps_only=True,
    ),
)


@dataclass
class Verdict:
    outcome: Outcome
    gates: list[GateResult]
    evidence_class: EvidenceClass
    policy_sha256: str
    notes: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """Non-zero on Stop, so a pipeline cannot report a dead project as success."""
        return 1 if self.outcome is Outcome.STOP else 0

    def stopping_gates(self) -> list[GateResult]:
        return [g for g in self.gates if g.outcome is Outcome.STOP]

    def narrowing_gates(self) -> list[GateResult]:
        return [g for g in self.gates if g.outcome is Outcome.NARROW]

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "evidence_class": self.evidence_class.value,
            "policy_sha256": self.policy_sha256,
            "gates": [g.as_dict() for g in self.gates],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"VERDICT: {self.outcome.value.upper()}",
            f"evidence: {self.evidence_class.value}   policy: {self.policy_sha256[:12]}",
            "",
        ]
        lines.extend(g.line() for g in self.gates)
        if self.notes:
            lines.append("")
            lines.extend(f"  note: {n}" for n in self.notes)
        if self.evidence_class is not EvidenceClass.FEASIBILITY:
            lines.append("")
            lines.append(
                "  THIS IS NOT A RESULT. Evidence class is "
                f"{self.evidence_class.value}; only real dispatches can decide anything."
            )
        return "\n".join(lines)


def policy_hash(policy: tuple[GateSpec, ...] = DEFAULT_POLICY) -> str:
    payload = json.dumps(
        [
            {**asdict(g), "direction": g.direction.value, "on_missing": g.on_missing.value}
            for g in policy
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_policy(path: str | Path, policy: tuple[GateSpec, ...] = DEFAULT_POLICY) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_sha256": policy_hash(policy),
        "gates": [
            {**asdict(g), "direction": g.direction.value, "on_missing": g.on_missing.value}
            for g in policy
        ],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(
    observations: dict[str, float | None],
    *,
    policy: tuple[GateSpec, ...] = DEFAULT_POLICY,
    evidence_class: EvidenceClass = EvidenceClass.PIPELINE_ONLY,
) -> Verdict:
    unknown = sorted(set(observations) - {g.id for g in policy})
    if unknown:
        raise ValueError(f"observations name gates that are not in the policy: {unknown}")

    results = [spec.evaluate(observations.get(spec.id)) for spec in policy]
    worst = max((r.outcome for r in results), key=lambda o: o.rank)

    notes: list[str] = []
    caps = [r for r in results if r.spec.caps_only and r.outcome is not Outcome.FEASIBLE]
    if caps and worst is not Outcome.STOP:
        worst = max(worst, Outcome.NARROW, key=lambda o: o.rank)
        notes.append("capped at narrow by: " + ", ".join(r.spec.id for r in caps))
    if evidence_class is not EvidenceClass.FEASIBILITY and worst is Outcome.FEASIBLE:
        notes.append(
            "feasible on non-feasibility evidence -- this is a pipeline check, not a finding"
        )

    return Verdict(
        outcome=worst,
        gates=results,
        evidence_class=evidence_class,
        policy_sha256=policy_hash(policy),
        notes=notes,
    )
