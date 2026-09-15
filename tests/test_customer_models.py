"""Constructed synthetic TEST DATA only; no customer measurements or API calls."""

from __future__ import annotations

import copy
import json

import pytest

from token_yield.customer_models import evaluate_predictions, fit_models, predict_cost
from token_yield.robust import ConstantModel, Record, RidgeLinearModel


FORMS = ("constant", "size", "size+units", "lego")


def synthetic_test_rows():
    """Unequal synthetic project sizes expose accidental call-weighted CV."""
    rows = []
    for project, replicates in enumerate((1, 2, 3, 5)):
        for replicate in range(replicates):
            rows.append({
                "call_id": f"test-{project}-{replicate}",
                "project_id": f"synthetic-test-project-{project}",
                "split": "train",
                "quote": {
                    "context_bytes": 100 * project,
                    "prompt_bytes": 200 + 600 * project + 40 * replicate,
                    "planned_output_tokens": 100 + 30 * replicate,
                    "counts": {
                        "extract": project + 1,
                        "classify": replicate,
                        "plan": project % 2,
                        "report": 1,
                    },
                },
                "rated_cost_usd": 0.3 + 0.7 * project + 0.02 * replicate,
                "status": "completed",
                "usage": {
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                    "cached_tokens": 5,
                    "reasoning_tokens": 2,
                },
                "quality_accepted": False,
            })
    return rows


def independent_features(row, form):
    quote = row["quote"]
    if form == "constant":
        return (0.0,)
    if form == "size":
        return (quote["prompt_bytes"],)
    if form == "size+units":
        return (
            quote["prompt_bytes"], sum(quote["counts"].values()), quote["planned_output_tokens"]
        )
    return (
        quote["prompt_bytes"], quote["planned_output_tokens"],
        *(quote["counts"][name] for name in ("extract", "classify", "plan", "report")),
    )


def test_group_cv_matches_independent_leave_one_project_out_recomputation():
    rows = synthetic_test_rows()
    artifact = fit_models(rows)
    projects = {row["project_id"] for row in rows}
    for form in FORMS:
        project_errors = {}
        for held_out in projects:
            training = [
                Record(independent_features(row, form), row["rated_cost_usd"], row["project_id"])
                for row in sorted(rows, key=lambda row: row["call_id"])
                if row["project_id"] != held_out
            ]
            model = (
                ConstantModel.fit(training) if form == "constant"
                else RidgeLinearModel.fit(training, alpha=10.0)
            )
            errors = []
            for row in rows:
                if row["project_id"] == held_out:
                    prediction = max(0.0, model.predict(independent_features(row, form)))
                    assert artifact["forms"][form]["cv_predictions"][row["call_id"]] == pytest.approx(
                        prediction
                    )
                    errors.append(abs(row["rated_cost_usd"] - prediction))
            project_errors[held_out] = sum(errors) / len(errors)
        expected = sum(project_errors.values()) / len(project_errors)
        assert artifact["cv_mae"][form] == pytest.approx(expected)
        assert artifact["forms"][form]["per_project_mae"] == pytest.approx(project_errors)
        for fold in artifact["forms"][form]["cv_folds"]:
            assert fold["validation_project_id"] not in fold["training_project_ids"]
            assert set(fold["training_project_ids"]) | {fold["validation_project_id"]} == projects
    assert artifact["selected_form"] == min(FORMS, key=lambda form: artifact["cv_mae"][form])
    assert artifact["training_call_ids"] == sorted(row["call_id"] for row in rows)


def test_all_forms_persist_and_round_trip_without_refitting(monkeypatch):
    rows = synthetic_test_rows()
    artifact = fit_models(rows)
    saved = json.loads(json.dumps(artifact, allow_nan=False))
    assert set(saved["forms"]) == set(FORMS)
    assert saved["calibrated_interval"] is None
    assert saved["calibration_status"] == "insufficient_independent_groups_shared_template"
    assert "not_coverage" in saved["residual_summary"]["interpretation"]
    expected = {form: predict_cost(artifact, rows[0]["quote"], form) for form in FORMS}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("prediction must never fit")

    monkeypatch.setattr(ConstantModel, "fit", forbidden_fit)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden_fit)
    for form in FORMS:
        assert predict_cost(saved, rows[0]["quote"], form) == expected[form]
        if form != "constant":
            assert saved["forms"][form]["model"]["alpha"] == 10.0
    assert predict_cost(saved, rows[0]["quote"]) == expected[saved["selected_form"]]


def test_constant_target_ties_choose_simple_form_and_are_deterministic():
    rows = synthetic_test_rows()
    for row in rows:
        row["rated_cost_usd"] = 1.0
    artifact = fit_models(rows)
    assert artifact["selected_form"] == "constant"
    assert artifact["cv_mae"] == dict.fromkeys(FORMS, 0.0)
    assert artifact == fit_models(rows) == fit_models(list(reversed(rows)))


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_holdout_rows_are_always_rejected(status):
    rows = synthetic_test_rows()
    rows[-1].update(split="holdout", status=status)
    with pytest.raises(ValueError, match="holdout"):
        fit_models(rows)


@pytest.mark.parametrize("target", [None, True, "0.5", float("nan"), float("inf"), -1.0])
def test_invalid_completed_targets_are_not_silently_zeroed(target):
    rows = synthetic_test_rows()
    rows[0]["rated_cost_usd"] = target
    with pytest.raises((TypeError, ValueError), match="rated_cost_usd"):
        fit_models(rows)


@pytest.mark.parametrize("field", ["call_id", "project_id", "rated_cost_usd", "quote", "usage", "status"])
def test_missing_row_fields_rejected(field):
    rows = synthetic_test_rows()
    del rows[0][field]
    with pytest.raises(ValueError, match=field):
        fit_models(rows)


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1, "output_tokens": 1},
                                  {"input_tokens": 1, "output_tokens": 1, "total_tokens": 9}])
def test_incomplete_usage_rejected(usage):
    rows = synthetic_test_rows()
    rows[0]["usage"] = usage
    with pytest.raises((TypeError, ValueError)):
        fit_models(rows)


def test_noncompleted_runs_explicitly_excluded_not_quality_rejected_completions():
    rows = synthetic_test_rows()
    failed = copy.deepcopy(rows[0])
    failed.update(call_id="failed-test-call", status="failed", usage=None, rated_cost_usd=None)
    artifact = fit_models(rows + [failed])
    assert artifact["excluded_call_ids"] == ["failed-test-call"]
    assert artifact["training_call_ids"] == sorted(row["call_id"] for row in rows)
    assert artifact["forms"] == fit_models(rows)["forms"]
    with pytest.raises(ValueError, match="3"):
        fit_models(rows[:3])
    with pytest.raises(ValueError, match="duplicate"):
        fit_models(rows + [rows[0]])
    with pytest.raises(ValueError):
        fit_models([])
    with pytest.raises(TypeError):
        fit_models({})


def test_clipping_and_artifact_validation():
    rows = synthetic_test_rows()
    artifact = fit_models(rows)
    artifact["forms"]["constant"]["model"]["value"] = -2.0
    assert predict_cost(artifact, rows[0]["quote"], "constant") == 0.0
    with pytest.raises(ValueError, match="unknown form"):
        predict_cost(artifact, rows[0]["quote"], "unknown")
    artifact["forms"]["size"]["model"]["scales"] = [0.0]
    with pytest.raises(ValueError, match="scales"):
        predict_cost(artifact, rows[0]["quote"], "size")
    artifact["forms"]["constant"]["model"]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        predict_cost(artifact, rows[0]["quote"], "constant")


def test_negative_extrapolation_clipped_in_cross_validation_and_inference():
    rows = [copy.deepcopy(synthetic_test_rows()[0]) for _ in range(4)]
    for index, (row, size, cost) in enumerate(zip(rows, (0, 1, 2, 100), (3.0, 2.0, 1.0, 0.0))):
        row.update(call_id=f"clip-test-{index}", project_id=f"clip-project-{index}", rated_cost_usd=cost)
        row["quote"]["prompt_bytes"] = size
    raw = RidgeLinearModel.fit(
        [Record((row["quote"]["prompt_bytes"],), row["rated_cost_usd"], row["project_id"])
         for row in rows[:3]], alpha=10.0
    )
    assert raw.predict((100,)) < 0
    artifact = fit_models(rows)
    assert artifact["forms"]["size"]["cv_predictions"]["clip-test-3"] == 0.0
    quote = copy.deepcopy(rows[-1]["quote"])
    quote["prompt_bytes"] = 100000
    assert predict_cost(artifact, quote, "size") == 0.0


@pytest.mark.parametrize("channel", ["input_tokens", "output_tokens", "total_tokens",
                                    "cached_tokens", "reasoning_tokens"])
def test_every_measured_usage_channel_required(channel):
    rows = synthetic_test_rows()
    del rows[0]["usage"][channel]
    with pytest.raises(ValueError, match=channel):
        fit_models(rows)


@pytest.mark.parametrize("value", [None, True, -1, 2.5, "3", float("inf")])
def test_invalid_quote_counts_rejected(value):
    rows = synthetic_test_rows()
    artifact = fit_models(rows)
    rows[0]["quote"]["counts"]["extract"] = value
    with pytest.raises((TypeError, ValueError)):
        predict_cost(artifact, rows[0]["quote"])
    with pytest.raises((TypeError, ValueError)):
        fit_models(rows)


def frozen_test_predictions(rows):
    return [{
        "call_id": row["call_id"],
        "predictions": {form: row["rated_cost_usd"] + index for index, form in enumerate(FORMS)},
    } for row in rows]


def test_evaluation_matches_by_id_without_fitting_or_selection(monkeypatch):
    rows = synthetic_test_rows()
    for row in rows:
        row["split"] = "holdout"

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("evaluation must never fit")

    monkeypatch.setattr(ConstantModel, "fit", forbidden_fit)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden_fit)
    predictions = frozen_test_predictions(rows)
    result = evaluate_predictions(rows, list(reversed(predictions)))
    assert result["mae"] == pytest.approx(dict(zip(FORMS, (0.0, 1.0, 2.0, 3.0))))
    assert "selected_form" not in result
    assert result["n_calls"] == len(rows)
    assert result["n_projects"] == 4
    predictions[0]["predictions"] = dict.fromkeys(FORMS, rows[0]["rated_cost_usd"] + 4.0)
    result = evaluate_predictions(rows, predictions)
    assert result["mae"]["constant"] == pytest.approx(1.0)
    assert result["forms"]["constant"]["call_weighted_mae"] == pytest.approx(4.0 / len(rows))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "missing-id", "forms", "nan"])
def test_evaluation_rejects_mismatched_or_invalid_predictions(mutation):
    rows = synthetic_test_rows()
    predictions = frozen_test_predictions(rows)
    if mutation == "missing":
        predictions.pop()
    elif mutation == "extra":
        predictions.append({"call_id": "unknown-test-call", "predictions": dict.fromkeys(FORMS, 0.0)})
    elif mutation == "duplicate":
        predictions.append(predictions[0])
    elif mutation == "missing-id":
        del predictions[0]["call_id"]
    elif mutation == "forms":
        del predictions[0]["predictions"]["lego"]
    else:
        predictions[0]["predictions"]["lego"] = float("nan")
    with pytest.raises(ValueError):
        evaluate_predictions(rows, predictions)


def test_evaluation_rejects_unmeasured_completed_target():
    rows = synthetic_test_rows()
    predictions = frozen_test_predictions(rows)
    rows[0]["rated_cost_usd"] = None
    with pytest.raises(TypeError, match="rated_cost_usd"):
        evaluate_predictions(rows, predictions)
