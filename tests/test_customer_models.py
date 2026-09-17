"""Constructed synthetic TEST DATA only; no customer measurements or API calls."""

from __future__ import annotations

import copy
import json

import pytest

from token_yield import customer_models
from token_yield.customer_models import evaluate_predictions, fit_models, predict_cost
from token_yield.robust import ConstantModel, Record, RidgeLinearModel


FORMS = ("constant", "size", "size+units", "lego")


def test_token_channels_use_measured_targets_and_grouped_selection():
    rows = synthetic_test_rows()
    for index, row in enumerate(rows):
        row["usage"].update(input_tokens=100 + index * 10, output_tokens=30 + index,
                            total_tokens=130 + index * 11)
    artifact = customer_models.fit_token_models(rows, {"template": "test-v1"})
    for target, channel in artifact["channels"].items():
        assert channel["target"] == target
        assert channel["selection_metric"] == f"equal_project_weight_mae_{target}"
        assert "prediction_floor_usd" not in channel
        assert channel["selected_form"] == min(FORMS, key=lambda f: channel["cv_mae"][f])
        for form in FORMS:
            errors = {}
            for project in artifact["training_project_ids"]:
                training = [
                    Record(independent_features(row, form), row["usage"][target], row["project_id"])
                    for row in sorted(rows, key=lambda row: row["call_id"])
                    if row["project_id"] != project
                ]
                model = (ConstantModel.fit(training) if form == "constant"
                         else RidgeLinearModel.fit(training, alpha=10.0))
                residuals = []
                for row in rows:
                    if row["project_id"] == project:
                        prediction = max(0.0, model.predict(independent_features(row, form)))
                        assert channel["forms"][form]["cv_predictions"][row["call_id"]] == pytest.approx(
                            prediction)
                        residuals.append(abs(row["usage"][target] - prediction))
                errors[project] = sum(residuals) / len(residuals)
            assert channel["cv_mae"][form] == pytest.approx(sum(errors.values()) / len(errors))
    changed = copy.deepcopy(rows)
    for row in changed:
        row["rated_cost_usd"] *= 100
        row["quality_accepted"] = True
    assert customer_models.fit_token_models(changed, artifact["runtime"]) == artifact
    assert customer_models.fit_token_models(list(reversed(rows)), artifact["runtime"]) == artifact


def test_token_reload_forecast_and_evaluation_never_refit(monkeypatch):
    rows = synthetic_test_rows()
    runtime = {"template": "test-v1", "model": "test-model"}
    artifact = json.loads(json.dumps(customer_models.fit_token_models(rows, runtime)))
    expected = customer_models.forecast_tokens(artifact, rows[0]["quote"], runtime)
    assert expected["point_estimate"] == {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
    assert expected["support"]["status"] == "within_observed_ranges"
    assert expected["calibrated_interval"] is None and expected["upper_budget_bound"] is None
    holdout = copy.deepcopy(rows)
    for row in holdout:
        row.update(split="holdout", project_id="held-out-" + row["project_id"])

    def forbidden(*args, **kwargs):
        pytest.fail("forecast and evaluation must never fit")

    monkeypatch.setattr(ConstantModel, "fit", forbidden)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden)
    assert customer_models.forecast_tokens(artifact, rows[0]["quote"], runtime) == expected
    evaluation = customer_models.evaluate_token_models(artifact, holdout)
    assert evaluation["selected_forms"] == {"input_tokens": "constant", "output_tokens": "constant"}
    assert all(score["mae"] == 0 for forms in evaluation["scores"].values() for score in forms.values())
    assert "retrospective" in evaluation["method"]
    with pytest.raises(ValueError, match="schema_version"):
        predict_cost(artifact["channels"]["input_tokens"], rows[0]["quote"])


def test_token_forecast_flags_unsupported_conditions():
    rows = synthetic_test_rows()
    artifact = customer_models.fit_token_models(rows, {"template": "v1"})
    quote = copy.deepcopy(rows[0]["quote"])
    mismatch = customer_models.forecast_tokens(artifact, quote, {"template": "v2"})
    assert mismatch["point_estimate"] is None
    assert mismatch["support"]["reasons"] == ["runtime_contract_mismatch"]
    quote["counts"] = {"extract": 0, "classify": 1, "plan": 0, "report": 0}
    quote["prompt_bytes"] = 100000
    forecast = customer_models.forecast_tokens(artifact, quote, artifact["runtime"])
    assert forecast["support"]["status"] == "extrapolation"
    assert "unmeasured_operation_combination" in forecast["support"]["reasons"]
    assert "outside_training_range:prompt_bytes" in forecast["support"]["reasons"]


@pytest.mark.parametrize("mutation", ["empty", "unknown", "negative", "boolean", "extra", "nan"])
def test_token_quotes_reject_invalid_inputs_during_fit_and_forecast(mutation):
    rows = synthetic_test_rows()
    artifact = customer_models.fit_token_models(rows, {"template": "v1"})
    quote = rows[0]["quote"]
    if mutation == "empty":
        quote["counts"] = dict.fromkeys(quote["counts"], 0)
    elif mutation == "unknown":
        quote["counts"]["retrieve"] = 1
    elif mutation == "extra":
        quote["actual_output_tokens"] = 99
    else:
        quote["prompt_bytes"] = {"negative": -1, "boolean": True, "nan": float("nan")}[mutation]
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_token_models(rows, artifact["runtime"])
    with pytest.raises((TypeError, ValueError)):
        customer_models.forecast_tokens(artifact, quote, artifact["runtime"])


def test_token_training_and_evaluation_enforce_project_boundaries():
    rows = synthetic_test_rows()
    artifact = customer_models.fit_token_models(rows, {"template": "v1"})
    with pytest.raises(ValueError, match="holdout"):
        customer_models.evaluate_token_models(artifact, rows)
    rows[0]["split"] = "holdout"
    with pytest.raises(ValueError, match="holdout"):
        customer_models.fit_token_models(rows, artifact["runtime"])
    for row in rows:
        row["split"] = "holdout"
    with pytest.raises(ValueError, match="disjoint"):
        customer_models.evaluate_token_models(artifact, rows)


def test_token_artifact_rejects_swapped_channels_and_malformed_support():
    rows = synthetic_test_rows()
    artifact = customer_models.fit_token_models(rows, {"template": "v1"})
    altered = copy.deepcopy(artifact)
    altered["channels"]["input_tokens"] = altered["channels"]["output_tokens"]
    with pytest.raises(ValueError, match="target mismatch"):
        customer_models.forecast_tokens(altered, rows[0]["quote"], artifact["runtime"])
    artifact["support"]["feature_ranges"]["prompt_bytes"] = [1, 0]
    with pytest.raises(ValueError, match="lower bound"):
        customer_models.forecast_tokens(artifact, rows[0]["quote"], artifact["runtime"])


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


PROJECT_FORMS = (*FORMS, "workflow")
PROJECT_CHANNELS = ("input_tokens", "output_tokens", "rated_cost_usd")
PROJECT_RUNTIME = {"model": "test-model", "harness": "test-v1"}


def synthetic_project_test_rows():
    """Complete synthetic project/arm/replicate totals, never public measurements."""
    seed = synthetic_test_rows()[0]
    rows = []
    for project, replicates in enumerate((1, 2, 3, 1, 2, 3)):
        for replicate in range(replicates):
            for arm, calls, maximum in (("single", 4, 1), ("batch", 2, 3)):
                row = copy.deepcopy(seed)
                units = 3 * (project + 1) + 1
                row.update(
                    call_id=f"project-test-{project}-{arm}-{replicate}",
                    project_id=f"project-test-{project}",
                    arm_id=arm,
                    replicate=replicate,
                    rated_cost_usd=0.1 + 0.02 * project + 0.03 * calls + 0.001 * replicate,
                )
                row["quote"].update(
                    context_bytes=100 * project,
                    prompt_bytes=200 + 600 * project + 40 * replicate + 50 * calls,
                    planned_output_tokens=100 * units,
                    counts=dict(extract=project + 1, classify=project + 1,
                                plan=project + 1, report=1),
                    planned_calls=calls,
                    max_operations_per_call=maximum,
                )
                input_tokens = 80 + 100 * project + 20 * calls + replicate
                output_tokens = 20 + 15 * units + 2 * replicate
                row["usage"].update(input_tokens=input_tokens, output_tokens=output_tokens,
                                    total_tokens=input_tokens + output_tokens)
                rows.append(row)
    return rows


def independent_project_features(row, form):
    if form == "workflow":
        return (*independent_features(row, "lego"), row["quote"]["planned_calls"],
                row["quote"]["max_operations_per_call"])
    return independent_features(row, form)


def project_target(row, channel):
    return row[channel] if channel == "rated_cost_usd" else row["usage"][channel]


def assert_uncalibrated_project_forecast(result):
    assert result["calibrated_interval"] is None
    assert result["upper_budget_bound"] is None
    assert result["calibration_status"] == "insufficient_independent_groups_shared_template"
    assert isinstance(result["support"]["reasons"], list)


def test_project_channels_match_independent_grouped_cv_and_selection():
    rows = synthetic_project_test_rows()
    artifact = customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    assert artifact["schema_version"] == "customer-project-model-v1"
    assert set(artifact["channels"]) == set(PROJECT_CHANNELS)
    projects = {row["project_id"] for row in rows}
    assert len(projects) == 6
    for channel in PROJECT_CHANNELS:
        fitted = artifact["channels"][channel]
        assert set(fitted["forms"]) == set(PROJECT_FORMS)
        for form in PROJECT_FORMS:
            project_errors = {}
            for held_out in sorted(projects):
                records = [
                    Record(independent_project_features(row, form),
                           project_target(row, channel), row["project_id"])
                    for row in sorted(rows, key=lambda row: row["call_id"])
                    if row["project_id"] != held_out
                ]
                model = (ConstantModel.fit(records) if form == "constant"
                         else RidgeLinearModel.fit(records, alpha=10.0))
                errors = []
                for row in rows:
                    if row["project_id"] == held_out:
                        prediction = max(0.0, model.predict(independent_project_features(row, form)))
                        assert fitted["forms"][form]["cv_predictions"][row["call_id"]] == (
                            pytest.approx(prediction)
                        )
                        errors.append(abs(project_target(row, channel) - prediction))
                project_errors[held_out] = sum(errors) / len(errors)
            assert fitted["forms"][form]["per_project_mae"] == pytest.approx(project_errors)
            assert fitted["cv_mae"][form] == pytest.approx(
                sum(project_errors.values()) / len(project_errors)
            )
            folds = fitted["forms"][form]["cv_folds"]
            assert {fold["validation_project_id"] for fold in folds} == projects
            for fold in folds:
                assert set(fold["training_project_ids"]) == (
                    projects - {fold["validation_project_id"]}
                )
        assert fitted["selected_form"] == min(PROJECT_FORMS, key=fitted["cv_mae"].get)
    assert artifact == customer_models.fit_project_models(list(reversed(rows)), PROJECT_RUNTIME)


def test_project_artifact_json_roundtrip_forecasts_without_refitting(monkeypatch):
    rows = synthetic_project_test_rows()
    artifact = customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    restored = json.loads(json.dumps(artifact, allow_nan=False))
    quote = rows[0]["quote"]
    expected = customer_models.forecast_project(artifact, quote, PROJECT_RUNTIME)
    assert expected["support"]["status"] == "within_observed_ranges"
    assert_uncalibrated_project_forecast(expected)
    point = expected["point_estimate"]
    assert set(point) == {*PROJECT_CHANNELS, "total_tokens"}
    assert point["total_tokens"] == pytest.approx(point["input_tokens"] + point["output_tokens"])
    assert all(value >= 0 for value in point.values())
    for channel in PROJECT_CHANNELS:
        for form in PROJECT_FORMS:
            if form != "constant":
                assert restored["channels"][channel]["forms"][form]["model"]["alpha"] == 10.0
        selected = artifact["channels"][channel]["selected_form"]
        records = [
            Record(independent_project_features(row, selected), project_target(row, channel),
                   row["project_id"])
            for row in sorted(rows, key=lambda row: row["call_id"])
        ]
        model = (ConstantModel.fit(records) if selected == "constant"
                 else RidgeLinearModel.fit(records, alpha=10.0))
        assert point[channel] == pytest.approx(
            max(0.0, model.predict(independent_project_features(rows[0], selected)))
        )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("project forecasting must never refit")

    monkeypatch.setattr(ConstantModel, "fit", forbidden_fit)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden_fit)
    assert customer_models.forecast_project(restored, quote, PROJECT_RUNTIME) == expected


def test_project_runtime_is_canonically_compared_and_frozen():
    rows = synthetic_project_test_rows()
    runtime = {**PROJECT_RUNTIME, "settings": {"temperature": 0, "seed": 42}}
    artifact = customer_models.fit_project_models(rows, runtime)
    reordered = {"settings": {"seed": 42, "temperature": 0},
                 "harness": "test-v1", "model": "test-model"}
    assert customer_models.forecast_project(artifact, rows[0]["quote"], reordered)[
        "support"
    ]["status"] == "within_observed_ranges"
    runtime["settings"]["seed"] = 43
    mismatch = customer_models.forecast_project(artifact, rows[0]["quote"], runtime)
    assert mismatch["point_estimate"] is None
    assert mismatch["support"]["status"] == "unsupported"
    assert mismatch["support"]["reasons"]
    assert_uncalibrated_project_forecast(mismatch)
    assert customer_models.forecast_project(artifact, rows[0]["quote"], reordered)[
        "point_estimate"
    ] is not None


@pytest.mark.parametrize("field", [
    "context_bytes", "prompt_bytes", "planned_output_tokens",
])
def test_project_quote_outside_training_range_warns_without_calibration(field):
    rows = synthetic_project_test_rows()
    artifact = customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    quote = copy.deepcopy(rows[0]["quote"])
    quote[field] = 100 * max(row["quote"][field] for row in rows) + 1
    result = customer_models.forecast_project(artifact, quote, PROJECT_RUNTIME)
    assert result["support"]["status"] == "extrapolation"
    assert result["support"]["reasons"]
    assert_uncalibrated_project_forecast(result)


def test_project_valid_unmeasured_layout_warns_without_calibration():
    rows = synthetic_project_test_rows()
    artifact = customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    quote = copy.deepcopy(rows[0]["quote"])
    quote.update(planned_calls=2, max_operations_per_call=2)
    result = customer_models.forecast_project(artifact, quote, PROJECT_RUNTIME)
    assert result["support"]["status"] == "extrapolation"
    assert "unmeasured_call_layout" in result["support"]["reasons"]
    assert result["point_estimate"] is not None
    assert_uncalibrated_project_forecast(result)


@pytest.mark.parametrize("calls,maximum", [
    (5, 1), (1, 5), (100, 1), (1, 100),
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 4),
    (3, 1), (3, 3), (3, 4),
    (4, 2), (4, 3), (4, 4),
])
def test_project_impossible_or_unbounded_layouts_rejected(calls, maximum):
    rows = synthetic_project_test_rows()
    artifact = customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    rows[0]["quote"].update(planned_calls=calls, max_operations_per_call=maximum)
    with pytest.raises(ValueError):
        customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    with pytest.raises(ValueError):
        customer_models.forecast_project(artifact, rows[0]["quote"], PROJECT_RUNTIME)


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_project_holdout_labels_rejected_before_any_fitting(monkeypatch, status):
    rows = synthetic_project_test_rows()
    held_out = copy.deepcopy(rows[0])
    held_out.update(call_id="unseen-holdout-row", split="holdout", status=status)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("holdout labels must be rejected before any fitting")

    monkeypatch.setattr(ConstantModel, "fit", forbidden_fit)
    monkeypatch.setattr(RidgeLinearModel, "fit", forbidden_fit)
    for project_id in (rows[0]["project_id"], "independent-holdout-project"):
        held_out["project_id"] = project_id
        for label in (0.0, 1e6):
            held_out["rated_cost_usd"] = label
            with pytest.raises(ValueError):
                customer_models.fit_project_models(rows + [held_out], PROJECT_RUNTIME)


def test_project_duplicate_rows_and_fewer_than_three_groups_rejected():
    rows = synthetic_project_test_rows()
    with pytest.raises(ValueError):
        customer_models.fit_project_models(rows + [copy.deepcopy(rows[0])], PROJECT_RUNTIME)
    for group_count in (0, 1, 2):
        projects = {f"project-test-{index}" for index in range(group_count)}
        with pytest.raises(ValueError):
            customer_models.fit_project_models(
                [row for row in rows if row["project_id"] in projects], PROJECT_RUNTIME
            )


@pytest.mark.parametrize("field", ["quote", "usage", "rated_cost_usd"])
def test_project_missing_measurements_rejected(field):
    rows = synthetic_project_test_rows()
    del rows[0][field]
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_project_models(rows, PROJECT_RUNTIME)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1])
@pytest.mark.parametrize("field", [*PROJECT_CHANNELS, "total_tokens"])
def test_project_nonfinite_or_missing_labels_rejected(field, value):
    rows = synthetic_project_test_rows()
    target = rows[0] if field == "rated_cost_usd" else rows[0]["usage"]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_project_models(rows, PROJECT_RUNTIME)


@pytest.mark.parametrize("field", ["planned_calls", "max_operations_per_call"])
@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "2", float("nan"), float("inf")])
def test_project_workflow_counts_must_be_positive_integers(field, value):
    rows = synthetic_project_test_rows()
    rows[0]["quote"][field] = value
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_project_models(rows, PROJECT_RUNTIME)


@pytest.mark.parametrize("field", ["planned_calls", "max_operations_per_call"])
def test_project_workflow_counts_are_required(field):
    rows = synthetic_project_test_rows()
    del rows[0]["quote"][field]
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_project_models(rows, PROJECT_RUNTIME)


@pytest.mark.parametrize("counts", [
    dict(extract=0, classify=0, plan=0, report=1),
    dict(extract=2, classify=1, plan=2, report=1),
    dict(extract=2, classify=2, plan=1, report=1),
    dict(extract=1, classify=1, plan=1, report=0),
    dict(extract=1, classify=1, plan=1, report=2),
    dict(extract=1, classify=1, report=1),
    dict(extract=1, classify=1, plan=1, report=1, unknown=1),
])
def test_project_incomplete_or_unknown_operation_counts_rejected(counts):
    rows = synthetic_project_test_rows()
    artifact = customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    rows[0]["quote"]["counts"] = counts
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_project_models(rows, PROJECT_RUNTIME)
    try:
        forecast = customer_models.forecast_project(artifact, rows[0]["quote"], PROJECT_RUNTIME)
    except ValueError:
        return
    assert forecast["support"]["status"] == "unsupported"
    assert forecast["point_estimate"] is None
    assert forecast["support"]["reasons"]
    assert_uncalibrated_project_forecast(forecast)


@pytest.mark.parametrize("runtime", [None, {}, [], {"model": float("nan")},
                                     {"model": float("inf")}, {"model": object()}])
def test_project_runtime_must_be_nonempty_finite_json_mapping(runtime):
    with pytest.raises((TypeError, ValueError)):
        customer_models.fit_project_models(synthetic_project_test_rows(), runtime)
