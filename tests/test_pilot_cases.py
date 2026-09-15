"""Tests for the frozen Foundry pilot case matrix."""

import json
from pathlib import Path

from token_yield.pilot_cases import evaluate_pilot_output, load_pilot_cases

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "foundry_wave2"


def test_preregistered_matrix_has_exact_allocation_and_groups():
    cases = load_pilot_cases(EXPERIMENT)
    assert len(cases) == len({case.case_id for case in cases}) == 30
    assert {family: sum(c.family == family for c in cases) for family in {
        "summarise", "transform", "fetch", "control"
    }} == {"summarise": 9, "transform": 9, "fetch": 6, "control": 6}
    assert len({case.group_id for case in cases if case.family == "fetch"}) == 3


def test_context_and_unit_ladders_are_replicated_three_times():
    cases = load_pilot_cases(EXPERIMENT)
    summary = [case for case in cases if case.family == "summarise"]
    transform = [case for case in cases if case.family == "transform"]
    assert sorted(case.context_bytes for case in summary) == [
        1024, 1024, 1024, 10240, 10240, 10240, 102400, 102400, 102400
    ]
    assert sorted(case.units for case in transform) == [
        2, 2, 2, 8, 8, 8, 32, 32, 32
    ]


def test_transform_oracle_binds_values_not_just_field_names():
    case = next(
        item for item in load_pilot_cases(EXPERIMENT)
        if item.case_id == "transform-2-r1"
    )
    assert evaluate_pilot_output(case, json.dumps(case.expected))["accepted"]
    wrong = {key: "wrong" for key in case.expected}
    assert not evaluate_pilot_output(case, json.dumps(wrong))["accepted"]


def test_fetch_pairs_have_same_oracle_and_live_requires_manifest():
    cases = load_pilot_cases(EXPERIMENT)
    snapshot = next(c for c in cases if c.case_id == "fetch-2019-24499-snapshot")
    live = next(c for c in cases if c.case_id == "fetch-2019-24499-live")
    assert snapshot.expected == live.expected
    assert snapshot.manifest_id is None
    assert live.manifest_id == "federal-register-2019-24499"


def test_snapshot_hashes_match_allowlist():
    manifests = json.loads(
        (EXPERIMENT / "api_manifests.json").read_text(encoding="utf-8")
    )
    for spec in manifests.values():
        path = EXPERIMENT / spec["snapshot_path"]
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec[
            "snapshot_sha256"
        ]
