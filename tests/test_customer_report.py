"""Offline evidence rendering and project-weighted metric regression tests."""

from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

import pytest

from token_yield.report import customer_error_metrics, customer_html_report, main


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = ROOT / "runs" / "20260915_customer_project_forecast_v1"
CATALOG = ROOT / "experiments" / "customer_requests" / "catalog.json"


class EvidenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_evidence = False
        self.evidence = ""
        self.ids = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs["href"])
        if tag == "script" and attrs.get("id") == "model-evidence":
            self.in_evidence = True

    def handle_data(self, data):
        if self.in_evidence:
            self.evidence += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_evidence = False


def test_report_matches_saved_metrics_and_preserves_artifacts():
    before = {path.name: sha256(path.read_bytes()).hexdigest() for path in MODEL_RUN.glob("*.json")}
    html = customer_html_report(MODEL_RUN, CATALOG)
    parser = EvidenceParser()
    parser.feed(html)
    evidence = json.loads(parser.evidence)
    assert evidence["historical_metrics"]["rated_cost_usd"]["mae"] == pytest.approx(0.000886915729293822)
    assert evidence["historical_metrics"]["input_tokens"]["mae"] == pytest.approx(186.84837170795268)
    assert evidence["historical_metrics"]["output_tokens"]["mae"] == pytest.approx(48.827610742254336)
    assert evidence["prospective_status"] == "not_executed"
    assert len(evidence["historical_observations"]) == 8
    assert len(parser.ids) == len(set(parser.ids))
    assert all(link[1:] in parser.ids for link in parser.links if link.startswith("#"))
    assert "42 hand-annotated requirements" in html
    assert "8/8 historical holdout predictions are extrapolations" in html
    assert "Prediction intervals are not calibrated" in html
    assert "label preservation, not autonomous categorization" in html
    assert "experiment-building steps, NOT the four GPT tasks" in html
    assert "We did not fine-tune GPT" in html
    assert "$0.000887" in html
    assert before == {path.name: sha256(path.read_bytes()).hexdigest() for path in MODEL_RUN.glob("*.json")}


def test_metrics_weight_projects_not_repeat_counts():
    rows = [
        {"observation_id": str(i), "project_id": "a" if i < 3 else "b",
         "actual": {"input_tokens": 10}, "predicted": {"input_tokens": 10 if i < 3 else 30}}
        for i in range(4)
    ]
    result = customer_error_metrics(rows, "input_tokens")
    assert result["mae"] == 10
    assert result["rmse"] == pytest.approx(200 ** 0.5)
    assert result["mape_pct"] == 100
    assert result["bias"] == 10
    rows[0]["actual"]["input_tokens"] = 0
    assert customer_error_metrics(rows, "input_tokens")["mape_pct"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        customer_error_metrics(rows + [rows[0]], "input_tokens")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "10"])
def test_metrics_reject_invalid_measurements(bad):
    with pytest.raises(ValueError, match="finite nonnegative"):
        customer_error_metrics([
            {"observation_id": "a", "project_id": "a",
             "actual": {"input_tokens": bad}, "predicted": {"input_tokens": 1}}
        ], "input_tokens")


def test_report_escapes_source_metadata_and_rejects_unsafe_links(tmp_path):
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    data["projects"][0]["buyer"] = '</script><img src=x onerror="alert(1)">'
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    html = customer_html_report(MODEL_RUN, path)
    assert '<img src=x' not in html
    assert "&lt;img src=x" in html
    data["projects"][0]["source_url"] = "javascript:alert(1)"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="HTTP"):
        customer_html_report(MODEL_RUN, path)


def test_report_rejects_mismatched_artifacts(tmp_path):
    run = tmp_path / "model"
    shutil.copytree(MODEL_RUN, run)
    path = run / "predictions.json"
    predictions = json.loads(path.read_text(encoding="utf-8"))
    predictions[0]["point_estimate"]["rated_cost_usd"] += 1
    path.write_text(json.dumps(predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="MAE"):
        customer_html_report(run, CATALOG)


def test_report_cli_writes_self_contained_html(tmp_path):
    output = tmp_path / "nested" / "report.html"
    assert main(["--model-run", str(MODEL_RUN), "--catalog", str(CATALOG),
                 "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_completed_prospective_report_requires_frozen_model_and_catalog(tmp_path):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["projects"] = [deepcopy(catalog["projects"][0])]
    catalog["projects"][0]["id"] = "new-project"
    catalog_path = tmp_path / "new.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    model = json.loads((MODEL_RUN / "models.json").read_text(encoding="utf-8"))
    selected = {target: channel["selected_form"] for target, channel in model["channels"].items()}
    values = {"input_tokens": 100, "output_tokens": 100, "total_tokens": 200, "rated_cost_usd": 0.01}
    evaluation = {
        "status": "completed", "model_sha256": sha256((MODEL_RUN / "models.json").read_bytes()).hexdigest(),
        "catalog_sha256": sha256(catalog_path.read_bytes()).hexdigest(),
        "catalog_file_sha256": sha256(catalog_path.read_bytes()).hexdigest(),
        "selected_forms": selected, "n_projects": 1, "n_observations": 1, "limitations": ["Scoping only."],
        "scores": {target: {form: {"mae": 0} for form in channel["forms"]}
                   for target, channel in model["channels"].items()},
        "observations": [
            {"observation_id": "new-0-batched", "project_id": "new-project", "arm": "batched", "replicate": 0,
             "actual": values, "predicted": values, "support": {"status": "extrapolation", "reasons": ["new"]},
             "contract_passed": False}
        ],
        "totals": {"rated_cost_usd": 0.01},
        "source_model_run": str(tmp_path / "private-model-location"),
        "private_metadata": {"access_token": "TEST-ONLY-NOT-A-CREDENTIAL"},
    }
    path = tmp_path / "prospective_evaluation.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    html = customer_html_report(MODEL_RUN, CATALOG, new_catalog_path=catalog_path, prospective_dir=tmp_path)
    assert "New measurements, using the unchanged prediction formula" in html
    assert "$0.010000" in html
    assert "private-model-location" not in html
    assert "TEST-ONLY-NOT-A-CREDENTIAL" not in html
    assert "lego-model-training.md" in html
    assert json.loads(path.read_text())["private_metadata"] == evaluation["private_metadata"]
    evaluation["model_sha256"] = "wrong"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="identical frozen model"):
        customer_html_report(MODEL_RUN, CATALOG, new_catalog_path=catalog_path, prospective_dir=tmp_path)
