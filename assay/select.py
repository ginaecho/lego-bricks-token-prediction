"""Model selection: climb the ladder only on evidence.

Two rules, both aimed at the same failure -- picking a richer model because it fitted the
training data slightly better, and then reporting that as though bricks had been shown to
matter.

1. **Grouped cross-validation.** Folds are whole source documents. A model is never scored
   on material it was fitted on.
2. **A real lift bar.** A richer form replaces a simpler one only if it removes at least
   ``min_lift_pct`` of the simpler form's *variable-portion* error, and a bootstrap lower
   bound on that lift stays above zero. The reference's rule -- "strictly beats" -- would
   accept a 0.3 pp win on total error, which is noise wearing a result's clothes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from assay.costmodel import (
    BootModel,
    CostModel,
    Predictor,
    fit_cell_means,
    predict_all,
)
from assay.errors import IdentifiabilityError, MisalignedFoldsError
from assay.features import FORMS, LADDER, run_group
from assay.metrics import Scorecard, bootstrap, relative_lift, score, wape
from assay.schemas import Run
from assay.vocabulary import Vocabulary

NO_GROUP = "_nocontext"


def grouped_folds(runs: Sequence[Run]) -> list[tuple[list[Run], list[Run]]]:
    """Leave-one-group-out. Runs with no source material stay in training every time."""
    groups: dict[str, list[Run]] = {}
    for r in runs:
        groups.setdefault(run_group(r), []).append(r)

    testable = sorted(k for k in groups if k != NO_GROUP)
    if len(testable) < 2:
        raise ValueError(
            f"need at least 2 source groups to cross-validate, found {len(testable)}"
        )

    folds = []
    for held in testable:
        test = groups[held]
        train = [r for k, rows in groups.items() if k != held for r in rows]
        folds.append((train, test))
    return folds


def _build(form_key: str, train: Sequence[Run], vocab: Vocabulary, boot: float) -> Predictor:
    form = FORMS[form_key]
    if form.solver == "fixed":
        return BootModel(boot)
    if form.solver == "cell":
        return fit_cell_means(train, vocab, boot)
    return CostModel.fit(train, form_key, vocab, boot, check_identifiability=False)


@dataclass
class CVResult:
    form: str
    role: str
    n_params: int
    n_scored: int
    pooled: Scorecard
    mean_fold_vwape: float
    actual: list[float] = field(default_factory=list, repr=False)
    predicted: list[float] = field(default_factory=list, repr=False)
    groups: list[str] = field(default_factory=list, repr=False)
    """The held-out group behind each scored row, in order.

    Carried so that a lift test can *verify* two forms were scored on the same rows rather
    than assume it. A form whose fit failed on one fold produces a shorter list, and
    zipping it against a complete one would pair predictions with the wrong documents --
    silently, and in whichever direction the arithmetic happened to fall.
    """
    failed_folds: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "form": self.form,
            "role": self.role,
            "n_params": self.n_params,
            "n_scored": self.n_scored,
            "failed_folds": self.failed_folds,
            "mean_fold_vwape_pct": round(self.mean_fold_vwape, 4),
            **self.pooled.as_dict(),
        }


def cross_validate(
    runs: Sequence[Run], form_key: str, vocab: Vocabulary, boot: float
) -> CVResult:
    """Out-of-fold predictions pooled once, then scored once.

    Pooling beats averaging per-fold percentages here: folds are small and a single
    near-zero variable actual inside one fold would otherwise dominate the mean. The mean
    of fold vWAPEs is reported alongside so the two can be compared.
    """
    folds = grouped_folds(runs)
    actual: list[float] = []
    predicted: list[float] = []
    groups: list[str] = []
    fold_scores: list[float] = []
    failed = 0

    for train, test in folds:
        try:
            model = _build(form_key, train, vocab, boot)
        except (ZeroDivisionError, IdentifiabilityError, ValueError):
            failed += 1
            continue
        preds = predict_all(model, test)
        acts = [r.billable_units for r in test]
        actual.extend(acts)
        predicted.extend(preds)
        groups.extend(run_group(r) for r in test)
        fold_scores.append(
            wape([a - boot for a in acts], [p - boot for p in preds])
        )

    if not actual:
        raise ValueError(f"{form_key}: every fold failed to fit")

    finite = [s for s in fold_scores if s != float("inf")]
    return CVResult(
        form=form_key,
        role=FORMS[form_key].role,
        n_params=FORMS[form_key].n_params(vocab),
        n_scored=len(actual),
        pooled=score(actual, predicted, boot),
        mean_fold_vwape=sum(finite) / len(finite) if finite else float("inf"),
        actual=actual,
        predicted=predicted,
        groups=groups,
        failed_folds=failed,
    )


@dataclass
class LiftTest:
    challenger: str
    incumbent: str
    lift_pct: float
    bootstrap_lo: float
    bootstrap_hi: float
    bar_pct: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "challenger": self.challenger,
            "incumbent": self.incumbent,
            "lift_pct": round(self.lift_pct, 3),
            "bootstrap_lo": round(self.bootstrap_lo, 3),
            "bootstrap_hi": round(self.bootstrap_hi, 3),
            "bar_pct": self.bar_pct,
            "passed": self.passed,
        }


def lift_test(
    incumbent: CVResult,
    challenger: CVResult,
    boot: float,
    *,
    bar_pct: float = 15.0,
    iters: int = 2000,
    seed: int = 1337,
) -> LiftTest:
    """Paired bootstrap on the variable-portion lift of challenger over incumbent.

    Raises :class:`MisalignedFoldsError` when the two forms were not scored on the same
    rows. That happens whenever one of them failed to fit on a fold the other survived,
    and it is not recoverable here: pairing row *i* of one form with row *i* of the other
    would compare a prediction about one document against an actual from a different one.
    """
    if incumbent.groups != challenger.groups:
        raise MisalignedFoldsError(
            f"{challenger.form} was scored on {len(challenger.groups)} rows and "
            f"{incumbent.form} on {len(incumbent.groups)} -- "
            f"{challenger.failed_folds} and {incumbent.failed_folds} folds failed to fit "
            "respectively, so the two are not row-comparable"
        )
    paired = list(
        zip(incumbent.actual, incumbent.predicted, challenger.predicted)
    )

    def statistic(sample: Sequence[object]) -> float:
        acts = [row[0] - boot for row in sample]  # type: ignore[index]
        inc = [row[1] - boot for row in sample]  # type: ignore[index]
        cha = [row[2] - boot for row in sample]  # type: ignore[index]
        return relative_lift(wape(acts, inc), wape(acts, cha))

    res = bootstrap(paired, statistic, iters=iters, seed=seed)
    return LiftTest(
        challenger=challenger.form,
        incumbent=incumbent.form,
        lift_pct=res.point,
        bootstrap_lo=res.lo,
        bootstrap_hi=res.hi,
        bar_pct=bar_pct,
        passed=res.point >= bar_pct and res.lo > 0.0,
    )


@dataclass
class Selection:
    chosen: str
    results: dict[str, CVResult]
    tests: list[LiftTest]
    best_baseline: str
    bar_pct: float
    notes: list[str] = field(default_factory=list)

    @property
    def bricks_helped(self) -> bool:
        return FORMS[self.chosen].role == "decoder"

    def decoder_lift(self) -> LiftTest | None:
        """The lift of the chosen form over the best baseline -- the headline number."""
        for t in reversed(self.tests):
            if t.challenger == self.chosen and t.incumbent == self.best_baseline:
                return t
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "chosen": self.chosen,
            "chose_a_decoder": self.bricks_helped,
            "best_baseline": self.best_baseline,
            "bar_pct": self.bar_pct,
            "scores": {k: v.as_dict() for k, v in self.results.items()},
            "lift_tests": [t.as_dict() for t in self.tests],
            "notes": self.notes,
        }


def select_form(
    runs: Sequence[Run],
    vocab: Vocabulary,
    boot: float,
    *,
    bar_pct: float = 15.0,
    iters: int = 2000,
    seed: int = 1337,
    include_diagnostic: bool = True,
) -> Selection:
    keys = list(LADDER) + (["cell_mean"] if include_diagnostic else [])
    results: dict[str, CVResult] = {}
    for key in keys:
        try:
            results[key] = cross_validate(runs, key, vocab, boot)
        except (ValueError, ZeroDivisionError):
            continue

    if not results:
        raise ValueError("no form could be cross-validated")

    baselines = [k for k in LADDER if FORMS[k].role == "baseline" and k in results]
    best_baseline = min(baselines, key=lambda k: results[k].pooled.vwape)

    incumbent = baselines[0]
    tests: list[LiftTest] = []
    notes: list[str] = []
    for key in LADDER:
        if key == incumbent or key not in results:
            continue
        try:
            test = lift_test(
                results[incumbent], results[key], boot, bar_pct=bar_pct, iters=iters, seed=seed
            )
        except MisalignedFoldsError as exc:
            # A challenger that cannot be compared has not beaten anything. Climbing on an
            # untestable comparison is exactly the move the lift bar exists to prevent.
            notes.append(f"{key} was not promoted: {exc}")
            continue
        tests.append(test)
        if test.passed:
            incumbent = key

    if FORMS[incumbent].role == "decoder" and incumbent != best_baseline:
        try:
            final = lift_test(
                results[best_baseline], results[incumbent], boot,
                bar_pct=bar_pct, iters=iters, seed=seed,
            )
        except MisalignedFoldsError as exc:
            notes.append(f"{incumbent} fell back to {best_baseline}: {exc}")
            incumbent = best_baseline
        else:
            tests.append(final)
            if not final.passed:
                incumbent = best_baseline

    return Selection(
        chosen=incumbent,
        results=results,
        tests=tests,
        best_baseline=best_baseline,
        bar_pct=bar_pct,
        notes=notes,
    )


def shuffle_target(runs: Sequence[Run], seed: int = 1337) -> list[Run]:
    """Break the link between features and cost, for the negative control.

    Under shuffled targets no form can beat the mean. A suite in which a decoder still
    wins here is measuring its own wiring.
    """
    import copy

    rng = random.Random(seed)
    values = [r.billable_units for r in runs]
    rng.shuffle(values)
    out = []
    for r, v in zip(runs, values):
        clone = copy.deepcopy(r)
        clone.billable_units = v
        out.append(clone)
    return out
