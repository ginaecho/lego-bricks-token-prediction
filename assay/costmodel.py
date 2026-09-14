"""Fitting and predicting.

Three predictor kinds share one interface so that a baseline and a decoder can be scored
by exactly the same code path -- there is no route by which the model under test gets a
gentler evaluation than the thing it must beat.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from assay.errors import IdentifiabilityError
from assay.evidence import EvidenceClass
from assay.features import FORMS, cell_key, design_matrix, feature_row
from assay.linalg import condition_number, nnls, solve_ols, spearman, vif
from assay.schemas import Run
from assay.vocabulary import Vocabulary


class Predictor(Protocol):
    key: str

    def predict(self, units: dict[str, int], context_bytes: int) -> float: ...


@dataclass(frozen=True)
class Identifiability:
    """Can this design support a coefficient per brick at all?

    A model can fit beautifully and still be meaningless if two columns move together --
    the coefficients are then arbitrary split of one effect. Checked before the numbers
    are believed, not after they are quoted.
    """

    condition_number: float
    max_vif: float
    spearman_units_bytes: float
    n_rows: int
    n_params: int

    @property
    def ok(self) -> bool:
        return (
            self.condition_number < 30.0
            and self.max_vif < 5.0
            and abs(self.spearman_units_bytes) <= 0.20
            and self.n_rows > self.n_params
        )

    def problems(self) -> list[str]:
        out = []
        if self.condition_number >= 30.0:
            out.append(f"condition number {self.condition_number:.1f} >= 30")
        if self.max_vif >= 5.0:
            out.append(f"max VIF {self.max_vif:.1f} >= 5")
        if abs(self.spearman_units_bytes) > 0.20:
            out.append(
                f"units and bytes are correlated (Spearman {self.spearman_units_bytes:+.2f})"
            )
        if self.n_rows <= self.n_params:
            out.append(f"{self.n_rows} rows cannot support {self.n_params} parameters")
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "condition_number": round(self.condition_number, 3),
            "max_vif": round(self.max_vif, 3),
            "spearman_units_bytes": round(self.spearman_units_bytes, 4),
            "n_rows": self.n_rows,
            "n_params": self.n_params,
            "ok": self.ok,
            "problems": self.problems(),
        }


@dataclass
class BootModel:
    """B0. Predicts the measured empty-task cost and nothing else."""

    boot: float
    key: str = "boot"

    def predict(self, units: dict[str, int], context_bytes: int) -> float:
        return self.boot


@dataclass
class CellMeanModel:
    """B5. The mean of each observed design cell; a ceiling, not a competitor."""

    cells: dict[tuple[str, ...], float]
    fallback: float
    vocab: Vocabulary
    key: str = "cell_mean"

    def predict_run(self, run: Run) -> float:
        return self.cells.get(cell_key(run, self.vocab), self.fallback)

    def predict(self, units: dict[str, int], context_bytes: int) -> float:
        band = "0" if context_bytes == 0 else str(len(str(context_bytes)))
        present = tuple(f"{n}x{units[n]}" for n in self.vocab.names if units.get(n))
        return self.cells.get(present + (f"band{band}",), self.fallback)


@dataclass
class CostModel:
    """A fitted linear form.

    ``marginals_ols_unclamped`` exists only for the NNLS form: it carries the sign pattern
    the constraint hid, so a negative marginal stays visible in the report even when the
    shipped model refuses to use it.
    """

    form: str
    columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    boot: float
    brick_names: tuple[str, ...]
    n_fitted: int
    data_sha256: str
    evidence_class: EvidenceClass
    identifiability: Identifiability | None = None
    marginals_ols_unclamped: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.form

    @property
    def intercept(self) -> float:
        return self.coefficients[0] if "intercept" in self.columns else 0.0

    @property
    def bytes_coef(self) -> float:
        return self._coef("context_bytes")

    def _coef(self, name: str) -> float:
        if name not in self.columns:
            return 0.0
        return self.coefficients[self.columns.index(name)]

    @property
    def marginals(self) -> dict[str, float]:
        return {n: self._coef(n) for n in self.brick_names if n in self.columns}

    def predict(self, units: dict[str, int], context_bytes: int) -> float:
        row = feature_row(
            self.form, units, context_bytes, _StubVocab(self.brick_names)  # type: ignore[arg-type]
        )
        return sum(c * x for c, x in zip(self.coefficients, row))

    def predict_run(self, run: Run) -> float:
        return self.predict(run.units, run.context_bytes)

    # -- fitting ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        runs: Sequence[Run],
        form_key: str,
        vocab: Vocabulary,
        boot: float,
        *,
        check_identifiability: bool = True,
    ) -> "CostModel":
        form = FORMS[form_key]
        if form.solver in {"fixed", "cell"}:
            raise ValueError(f"{form_key} is not a linear form; construct it directly")
        if not runs:
            raise ValueError("cannot fit on zero runs")

        X, y = design_matrix(runs, form_key, vocab)
        ident = None
        if check_identifiability and form.uses_per_brick:
            ident = assess_identifiability(runs, form_key, vocab)
            if not ident.ok and ident.n_rows <= ident.n_params:
                raise IdentifiabilityError(
                    f"{form_key}: " + "; ".join(ident.problems())
                )

        unclamped: dict[str, float] = {}
        if form.solver == "nnls":
            beta = nnls(X, y)
            ols = solve_ols(X, y)
            cols = form.columns(vocab)
            unclamped = {
                name: ols[cols.index(name)] for name in vocab.names if name in cols
            }
        else:
            beta = solve_ols(X, y)

        evidence = {r.evidence_class for r in runs}
        return cls(
            form=form_key,
            columns=form.columns(vocab),
            coefficients=tuple(beta),
            boot=boot,
            brick_names=tuple(vocab.names),
            n_fitted=len(runs),
            data_sha256=data_hash(runs),
            evidence_class=(
                EvidenceClass.FEASIBILITY
                if evidence == {EvidenceClass.FEASIBILITY}
                else EvidenceClass.PIPELINE_ONLY
                if EvidenceClass.PIPELINE_ONLY in evidence
                else EvidenceClass.REPLAY
            ),
            identifiability=ident,
            marginals_ols_unclamped=unclamped,
        )

    # -- serialisation ------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "form": self.form,
            "columns": list(self.columns),
            "coefficients": [round(c, 8) for c in self.coefficients],
            "intercept": round(self.intercept, 6),
            "bytes_coef": round(self.bytes_coef, 8),
            "boot_hat": round(self.boot, 4),
            "marginals": {k: round(v, 4) for k, v in self.marginals.items()},
            "marginals_ols_unclamped": {
                k: round(v, 4) for k, v in self.marginals_ols_unclamped.items()
            },
            "brick_names": list(self.brick_names),
            "n_fitted": self.n_fitted,
            "data_sha256": self.data_sha256,
            "evidence_class": self.evidence_class.value,
            "identifiability": self.identifiability.as_dict() if self.identifiability else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostModel":
        return cls(
            form=data["form"],
            columns=tuple(data["columns"]),
            coefficients=tuple(float(c) for c in data["coefficients"]),
            boot=float(data["boot_hat"]),
            brick_names=tuple(data["brick_names"]),
            n_fitted=int(data["n_fitted"]),
            data_sha256=data["data_sha256"],
            evidence_class=EvidenceClass(data["evidence_class"]),
            marginals_ols_unclamped=dict(data.get("marginals_ols_unclamped", {})),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


class _StubVocab:
    """Lets a deserialised model rebuild feature rows without reloading bricks.toml."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)


def data_hash(runs: Sequence[Run]) -> str:
    payload = "|".join(sorted(f"{r.run_id}:{r.billable_units:.6f}" for r in runs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assess_identifiability(
    runs: Sequence[Run], form_key: str, vocab: Vocabulary
) -> Identifiability:
    X, _ = design_matrix(runs, form_key, vocab)
    # Rank correlation is measured over the rows that actually vary. Null probes sit at
    # the origin of both axes -- zero units, zero bytes -- so including them manufactures
    # a positive correlation out of a single replicated design point, and would flag a
    # perfectly balanced grid as confounded. They stay in the design matrix, where they
    # pin the intercept and where the condition number and VIF still account for them.
    varying = [r for r in runs if r.context_bytes > 0] or list(runs)
    factors = vif(X)
    return Identifiability(
        condition_number=condition_number(X),
        max_vif=max(factors),
        spearman_units_bytes=spearman(
            [float(sum(r.units.values())) for r in varying],
            [float(r.context_bytes) for r in varying],
        ),
        n_rows=len(runs),
        n_params=len(X[0]),
    )


def fit_cell_means(runs: Sequence[Run], vocab: Vocabulary, boot: float) -> CellMeanModel:
    buckets: dict[tuple[str, ...], list[float]] = {}
    for r in runs:
        buckets.setdefault(cell_key(r, vocab), []).append(r.billable_units)
    cells = {k: sum(v) / len(v) for k, v in buckets.items()}
    fallback = sum(r.billable_units for r in runs) / len(runs) if runs else boot
    return CellMeanModel(cells=cells, fallback=fallback, vocab=vocab)


def predict_all(model: Predictor, runs: Sequence[Run]) -> list[float]:
    predict_run = getattr(model, "predict_run", None)
    if predict_run is not None:
        return [predict_run(r) for r in runs]
    return [model.predict(r.units, r.context_bytes) for r in runs]
