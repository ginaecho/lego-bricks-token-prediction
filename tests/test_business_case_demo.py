"""Focused tests for the business-case demo's evidence boundary."""

import json

import pytest

from examples.business_case_demo import (
    _print_proofs,
    load_proof_results,
    partition_proof_records,
)
from token_yield.economics import Pricing, WorkOutcome


def test_partition_separates_acceptance_only_from_positive_measurements():
    measured, acceptance_only = partition_proof_records([
        {"case_id": "verdict", "accepted": True},
        {"case_id": "zero", "accepted": False, "total_tokens": 0},
        {
            "case_id": "measured",
            "accepted": True,
            "input_tokens": 80,
            "output_tokens": 20,
            "cost_center": "finance",
            "provenance": "measured",
        },
    ])

    assert [item.case_id for item in measured] == ["measured"]
    assert measured[0].tokens() == 100
    assert [item.case_id for item in acceptance_only] == ["verdict", "zero"]


@pytest.mark.parametrize("field,value", [
    ("total_tokens", -1),
    ("input_tokens", 1.5),
    ("output_tokens", True),
])
def test_partition_rejects_non_measurement_token_values(field, value):
    with pytest.raises(ValueError, match=field):
        partition_proof_records([
            {
                "case_id": "bad",
                "accepted": True,
                "provenance": "measured",
                field: value,
            },
        ])


def test_acceptance_only_record_rejects_accidental_token_usage():
    with pytest.raises(ValueError, match="acceptance-only"):
        partition_proof_records([{
            "case_id": "bad",
            "accepted": True,
            "proof_type": "acceptance-only",
            "total_tokens": 1,
        }])


def test_unprovenanced_token_usage_is_not_treated_as_measured():
    with pytest.raises(ValueError, match="provenance"):
        partition_proof_records([{
            "case_id": "bad",
            "accepted": True,
            "total_tokens": 1,
        }])


def test_loader_reports_invalid_json_line(tmp_path):
    path = tmp_path / "proof_results.jsonl"
    path.write_text(
        json.dumps({"case_id": "ok", "accepted": True}) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"proof_results\.jsonl:2"):
        load_proof_results(path)


def test_economics_excludes_acceptance_only_proofs(capsys):
    measured = [
        WorkOutcome("a", True, total_tokens=100, cost_center="finance"),
        WorkOutcome("b", True, total_tokens=200, cost_center="finance"),
    ]
    acceptance_only = [WorkOutcome("unmeasured", True)]

    _print_proofs(
        measured,
        acceptance_only,
        Pricing(
            input_per_million=10,
            output_per_million=10,
            blended_per_million=10,
        ),
    )

    output = capsys.readouterr().out
    assert "acceptance-only (missing token measurements): 1" in output
    assert "fully measured outcomes: 2" in output
    assert "p50/p90/p95/p99 tokens: 150" in output
    assert "accepted-work cost: $0.001500" in output
    assert "finance: $0.003000 actual" in output
