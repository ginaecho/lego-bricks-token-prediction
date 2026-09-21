import copy
import json
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.marketplace_demo_server import validate_run_request
from token_yield.marketplace_agent_contracts import (
    contracts, fingerprint, schema_from_example, source_documents, validate_message,
)
from token_yield.marketplace_generalization import (
    FEATURES, dataset_plan, evaluate, features_for, fit_training, forecast,
    load_published, prompt_for, selected_contract, PublishedMarketplace,
)


def fixture_rows():
    """Explicit synthetic labels for offline regression tests, never live evidence."""
    plan = dataset_plan()
    rows = {"train": [], "holdout": []}

    def add(brick, ids, split, index):
        docs = source_documents(f"mock-{split}-{index}", index)
        vector = features_for(brick, docs)
        rows[split].append({
            "id": f"mock-{brick['id']}-{split}-{index}", "ids": ids, "group": brick["id"],
            "split": split, "x": vector, "source": "mocked-test-provider",
            "input": round(len(prompt_for(brick, docs)) / 4),
            "output": 70 + 19 * sum(brick["atoms"].values()),
        })
    for brick in contracts():
        for index in range(4):
            add(brick, [brick["id"]], "train", index)
    for split, entries in plan["groups"].items():
        for item in entries:
            for index in ((0, 3) if split == "train" else (4, 5)):
                add(item["contract"], item["ids"], split, index)
    return rows


def test_design_covers_all_subtypes_and_unseen_memberships():
    plan = dataset_plan()
    assert plan == dataset_plan()
    wanted = {item["id"] for item in contracts()}
    sets = {}
    for split, entries in plan["groups"].items():
        assert {member for item in entries for member in item["ids"]} == wanted
        assert all(len(item["contract"]["steps"]) <= 40 for item in entries)
        sets[split] = {tuple(sorted(item["ids"])) for item in entries}
    assert not sets["train"] & sets["holdout"]
    assert len(sets["train"]) == 12
    assert len(sets["holdout"]) == 8
    next_plan = dataset_plan(plan)
    assert next_plan["groups"]["train"] == plan["groups"]["train"]
    assert next_plan["gates"] == plan["gates"]
    assert next_plan["holdout_repeats"] == 1
    assert {member for item in next_plan["groups"]["holdout"] for member in item["ids"]} == wanted
    assert not {tuple(sorted(item["ids"])) for item in next_plan["groups"]["holdout"]} & (sets["train"] | sets["holdout"])


def test_actual_train_test_seam_rejects_leakage_and_bad_model():
    rows = fixture_rows()
    model = fit_training(rows["train"])
    assert len(model["features"]) == len(FEATURES)
    assert model["holdout_used_for_selection"] is False
    result = evaluate(model, rows["train"], rows["holdout"])
    assert result["accepted"], result
    assert result["subtypes_tested"] == 16
    assert result["unseen_combinations"] == 8
    assert set(result["per_subtype"]) == {item["id"] for item in contracts()}
    with pytest.raises(ValueError, match="holdouts cannot enter"):
        fit_training(rows["train"] + rows["holdout"])
    leaked = copy.deepcopy(rows["holdout"])
    leaked[0]["ids"] = rows["train"][0]["ids"]
    with pytest.raises(ValueError, match="leaked"):
        evaluate(model, rows["train"], leaked)
    poor = copy.deepcopy(model)
    for target in poor["models"].values():
        target["coefficients"] = [0] * len(FEATURES)
        target["intercept"] = 1
    assert not evaluate(poor, rows["train"], rows["holdout"])["accepted"]


def test_order_changes_features_and_compact_protocol():
    a = selected_contract(["behavior", "compare"])
    b = selected_contract(["compare", "behavior"])
    docs = source_documents("test", 0)
    assert features_for(a, docs) != features_for(b, docs)
    payload = json.loads(prompt_for(a, docs))
    assert all("8 words" in result for result in payload["output_contract"]["atom_results"].values())
    assert len(payload["steps"]) == sum(a["atoms"].values())
    schema = schema_from_example(payload["output_contract"])
    assert set(schema["properties"]["atom_results"]["required"]) == {step["id"] for step in a["steps"]}
    output = {"answer": "Fixture output.", "evidence": [{"document_id": docs[0]["id"], "quote": docs[0]["text"]}],
              "limitations": [], "atom_results": {step["id"]: "Missing evidence." for step in a["steps"]}}
    validate_message("workload", output, docs, [{**a, "result_format": "keyed-steps-v1"}])
    output["atom_results"].pop(a["steps"][0]["id"])
    with pytest.raises(ValueError, match="named result"):
        validate_message("workload", output, docs, [{**a, "result_format": "keyed-steps-v1"}])
    with pytest.raises(ValueError, match="distinct"):
        selected_contract(["behavior", "behavior"])
    with pytest.raises(ValueError, match="known"):
        selected_contract(["behavior", "invented"])
    with pytest.raises(ValueError, match="bounds"):
        selected_contract([item["id"] for item in contracts()])


def test_training_scope_is_explicit_and_foundry_only():
    request = {"description": "Measure all marketplace combinations for generalization.",
               "model_id": "gpt", "runs_per_month": 1000,
               "runtime": "foundry", "training_scope": "marketplace-generalization"}
    assert validate_run_request(request)["training_scope"] == "marketplace-generalization"
    assert validate_run_request({**request, "training_parent_run": "a"*32})["training_parent_run"] == "a"*32
    with pytest.raises(ValueError):
        validate_run_request({**request, "training_parent_run": ".."})
    with pytest.raises(ValueError):
        validate_run_request({**request, "runtime": "offline"})
    with pytest.raises(ValueError):
        validate_run_request({**request, "training_scope": "unlimited"})


def test_published_inference_is_read_only_and_provenance_checked(tmp_path, monkeypatch):
    rows = fixture_rows()
    model = fit_training(rows["train"])
    model.update(version="a" * 32, compatibility="mock-compatibility",
                 catalog_hash=fingerprint(contracts()), source="mocked-test-provider",
                 evaluation=evaluate(model, rows["train"], rows["holdout"]))
    version_dir = tmp_path / "composition-versions"
    version_dir.mkdir()
    model_path = version_dir / f"{model['version']}.json"
    model_path.write_text(json.dumps(model))
    pointer = tmp_path / "composition-current.json"
    pointer.write_text(json.dumps({"version": model["version"], "sha256": fingerprint(model)}))
    runtime = SimpleNamespace(state_dir=tmp_path, _compatibility="mock-compatibility", _dispatch_hook=True,
                              _cost=lambda i, o: (i * 2.5 + o * 15) / 1e6)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = forecast(runtime, ["journey", "faq", "extract"])
    assert result["supported"]
    assert result["forecast_mode"] == "measured-composition-model"
    assert result["source"] == "mocked-test-provider"
    assert result["standalone_forecasts_summed"] is False
    assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with monkeypatch.context() as patch:
        patch.setattr("token_yield.marketplace_generalization._predict", lambda model, vector: 1800)
        bounded = forecast(runtime, ["journey", "faq", "extract"])
        assert bounded["output_tokens"] == 1536
        assert bounded["uncapped_output_tokens"] == 1800
        assert bounded["output_ceiling_applied"] is True
        assert bounded["total_tokens"] == 1800 + 1536
    runtime._compatibility = "changed"
    assert load_published(runtime) is None
    runtime._compatibility = "mock-compatibility"
    model["models"]["output"]["intercept"] += 1
    model_path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="fingerprint"):
        load_published(runtime)


def test_shipped_model_predicts_all_pairs_without_training():
    root = Path(__file__).resolve().parents[1] / "examples" / "data" / "marketplace-trained"
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    predictor = PublishedMarketplace(root)
    catalog = predictor.catalog()
    singles = [item for item in catalog["items"] if item["feature_id"] != "workflow"]
    assert len(singles) == 16
    assert all(item["supported"] and item["source"] == "measured-foundry" for item in singles)
    assert not hasattr(predictor, "run_pipeline")
    for pair in permutations([item["id"] for item in singles], 2):
        result = predictor.forecast_selection(list(pair))
        assert result["supported"] and result["total_tokens"] > 0
        assert result["usd_per_run"] > 0
    assert before == {path: path.read_bytes() for path in root.rglob("*.json")}
