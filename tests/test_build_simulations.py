import copy
import json
import sqlite3
import hashlib
from collections import Counter
from pathlib import Path, PureWindowsPath

import pytest

from token_yield.build_simulations import (
    REWARD_WEIGHTS, build_spec, combination_design, complete_point, measured_usage,
    planned_point, price_usage, pseudo_outcomes, read_usage, reward,
)


def event(identifier=1, agent="builder"):
    return {
        "id": identifier, "agent_id": agent, "model": "mock-test-builder",
        "input_tokens": 200, "output_tokens": 20,
        "cache_read_tokens": 150, "cache_write_tokens": 20, "reasoning_tokens": 10,
        "duration_ms": 100, "created_at": "2026-09-21T00:00:00Z",
        "token_details_json": json.dumps([
            {"tokenType": "input", "tokenCount": 30},
            {"tokenType": "cache_read", "tokenCount": 150},
            {"tokenType": "cache_write", "tokenCount": 20},
            {"tokenType": "output", "tokenCount": 20},
        ]),
    }


def card():
    return {
        "model": "mock-rate-scenario", "status": "test_fixture", "source_url": "https://example.invalid",
        "short_context_max_input_tokens": 272000,
        "short_context": {"input_per_million": 2.5, "cached_input_per_million": .25,
                          "output_per_million": 15.},
        "long_context": {"input_per_million": 5., "cached_input_per_million": .5,
                         "output_per_million": 22.5},
    }


def test_independent_design_covers_types_and_distinct_families():
    design = combination_design()
    assert Counter(len(item["types"]) for item in design) == {1: 16, 2: 109, 3: 410, 4: 920}
    assert len({item["id"] for item in design}) == 1455
    assert all(item["status"] == "planned_unmeasured" for item in design)
    point = planned_point(["interests", "compare"])
    assert point["source_case_ids"] == []
    assert point["build_token_usage"] is None
    assert point["input_features"]["staff_data_scientist"] == .25
    assert not any("tokens" in name for name in point["input_features"])
    reverse = build_spec(["compare", "interests"])
    assert reverse["split_group"] == point["split_group"]
    assert reverse["split"] == point["split"]
    with pytest.raises(ValueError, match="distinct basic"):
        build_spec(["interests", "behavior"])


def test_real_counters_no_cache_or_reasoning_double_counting():
    usage = measured_usage([event(), event(2)], "builder")
    assert usage["input_tokens"] == 400
    assert usage["output_tokens"] == 40
    assert usage["total_tokens"] == 440
    assert usage["cache_read_tokens"] == 300
    assert usage["actual_billed_usd"] is None
    invalid = event()
    invalid["input_tokens"] += 1
    with pytest.raises(ValueError, match="disagree"):
        measured_usage([invalid], "builder")
    with pytest.raises(ValueError, match="another agent"):
        measured_usage([event(agent="other")], "builder")
    with pytest.raises(ValueError, match="Distinct"):
        measured_usage([event(), event()], "builder")


def test_price_is_counterfactual_and_tier_is_per_request():
    usage = measured_usage([event()], "builder")
    quote = price_usage(usage, card())
    assert quote["no_cache_estimate_usd"] == pytest.approx(.0008)
    assert quote["observed_cache_profile_estimate_usd"] == pytest.approx(.0004625)
    assert "Counterfactual" in quote["basis"]
    large = {"events": [{"input_tokens": 200000, "output_tokens": 0, "cache_read_tokens": 0}] * 2}
    assert price_usage(large, card())["no_cache_estimate_usd"] == 1.
    large["events"] = [{"input_tokens": 300000, "output_tokens": 0, "cache_read_tokens": 0}]
    assert price_usage(large, card())["no_cache_estimate_usd"] == 1.5
    limited = card()
    limited["long_context"] = None
    with pytest.raises(ValueError, match="verified rate"):
        price_usage(large, limited)


def test_pseudo_feedback_is_bounded_separate_and_reproducible():
    assert reward({name: 0 for name in REWARD_WEIGHTS}) == 0
    assert reward({name: 5 for name in REWARD_WEIGHTS}) == pytest.approx(1)
    outcome = pseudo_outcomes("single_interests")
    assert outcome == pseudo_outcomes("single_interests")
    assert outcome["is_pseudo"] is True
    assert outcome["verified_client_roi_pct"] is None
    assert outcome["policy_promoted"] is False
    assert 0 <= outcome["reward_0_to_1"] <= 1
    for bad in (True, float("nan"), -1, 6):
        with pytest.raises(ValueError, match="scores"):
            reward({**{name: 3 for name in REWARD_WEIGHTS}, "client_satisfaction": bad})


def test_complete_point_requires_working_artifacts_and_unchanged_features():
    point = planned_point(["interests"])
    with pytest.raises(ValueError, match="tested implementation"):
        complete_point(point, [event()], "builder", [card()], {"tests_passed": False})
    evidence = {"tests_passed": True, "artifact_sha256": {"implementation.py": "mock-hash"}}
    complete = complete_point(point, [event()], "builder", [card()], evidence)
    assert complete["status"] == "measured_build_passed"
    assert complete["build_token_usage"]["total_tokens"] == 220
    changed = copy.deepcopy(point)
    changed["input_features"]["staff_data_scientist"] = 7
    with pytest.raises(ValueError, match="record changed"):
        complete_point(changed, [event()], "builder", [card()], evidence)


def test_usage_reader_is_read_only_and_isolates_session_and_agent(tmp_path):
    path = tmp_path / "usage.db"
    row = event()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE assistant_usage_events (session_id TEXT, id INTEGER, agent_id TEXT, "
                   "model TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, "
                   "cache_write_tokens INTEGER, reasoning_tokens INTEGER, duration_ms INTEGER, "
                   "created_at TEXT, token_details_json TEXT)")
        columns = ["session_id", *row]
        for session, agent in (("wanted", "builder"), ("other", "builder"), ("wanted", "other")):
            values = {"session_id": session, **row, "agent_id": agent}
            db.execute(f"INSERT INTO assistant_usage_events ({','.join(columns)}) "
                       f"VALUES ({','.join('?' for _ in columns)})", [values[name] for name in columns])
    before = path.read_bytes()
    assert read_usage(path, "wanted", "builder") == [row]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="No authoritative"):
        read_usage(path, "missing", "builder")


def test_persisted_wave_has_traceable_separate_build_labels():
    repository = Path(__file__).resolve().parents[1]
    root = repository / "data" / "build_simulations"
    points = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "data_points").glob("*.json")]
    measured = [point for point in points if point["status"] == "measured_build_passed"]
    sizes = Counter(point["input_features"]["functionality_count"] for point in measured)
    assert sizes[1] >= 16 and all(sizes[size] >= 1 for size in (2, 3, 4))
    assert len({tuple(point["input_features"]["types"]) for point in measured
                if point["input_features"]["functionality_count"] == 1}) == 16
    assert len({point["build_token_usage"]["builder_agent_id"] for point in measured}) == len(measured)
    event_ids = []
    for point in points:
        assert point["pseudo_outcomes"]["is_pseudo"] is True
        assert point["pseudo_outcomes"]["verified_client_roi_pct"] is None
        if point["origin"] == "real_use_case_reference":
            assert point["build_token_usage"] is None
            assert point["azure_cost_scenarios"] == []
    for point in measured:
        usage = point["build_token_usage"]
        assert point["source_case_ids"] == []
        assert usage["source"] == "copilot_runtime_assistant_usage_events"
        assert usage["input_tokens"] == sum(event["input_tokens"] for event in usage["events"])
        assert usage["output_tokens"] == sum(event["output_tokens"] for event in usage["events"])
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        assert usage["actual_billed_usd"] is None
        event_ids.extend(event["id"] for event in usage["events"])
        evidence = point["build_evidence"]
        directory = repository.joinpath(*PureWindowsPath(evidence["artifact_directory"]).parts)
        assert directory.resolve().is_relative_to(root.resolve())
        for filename, expected in evidence["artifact_sha256"].items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
    assert len(set(event_ids)) == len(event_ids), "Do not reuse the same request as labels for multiple builds"
