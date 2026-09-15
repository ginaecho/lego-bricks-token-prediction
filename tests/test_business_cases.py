"""Tests for source-backed cases, acceptance gates, and unit economics."""

import json
from pathlib import Path

import pytest

from token_yield.business_cases import (
    AcceptanceRule,
    BusinessCase,
    catalog_summary,
    evaluate_output,
    load_cases,
    to_run,
    validate_case,
)
from token_yield.runner import AgentResult, dry_run, run_case

REPO_ROOT = Path(__file__).resolve().parents[1]
from token_yield.economics import (
    Pricing,
    WorkOutcome,
    chargeback,
    percentile,
    summarize,
)


def case(**overrides):
    values = {
        "case_id": "invoice-extract",
        "title": "Extract invoice fields",
        "tier": "base",
        "business_domain": "finance",
        "provenance": "source-backed",
        "source_urls": ("https://example.com/invoice",),
        "source_note": "Public example invoice.",
        "input_driver": "output_units",
        "input_size": 3,
        "context_bytes": 100,
        "counts": {"extract": 3},
        "prompt": "Return invoice_number, supplier, and total as JSON.",
        "acceptance": (AcceptanceRule(
            "valid_json_fields", ["invoice_number", "supplier", "total"]),),
    }
    values.update(overrides)
    return BusinessCase(**values)


def test_base_case_is_a_single_primitive_with_public_evidence():
    item = case()
    validate_case(item)
    assert item.arity == 1
    assert item.total_units == 3
    assert item.notation() == "3xExtract"


def test_composite_case_requires_multiple_primitives():
    with pytest.raises(ValueError, match="arity"):
        validate_case(case(tier="composite"))


def test_case_rejects_non_public_source_links():
    with pytest.raises(ValueError, match="HTTPS"):
        validate_case(case(source_urls=("file:///private/invoice.pdf",)))


def test_acceptance_gate_reports_each_failed_rule():
    item = case(acceptance=(
        AcceptanceRule("contains_all", ["approved", "owner"]),
        AcceptanceRule("max_words", 3),
    ))
    result = evaluate_output(item, "approved but no named person is the owner")
    assert not result.accepted
    assert result.checks[0][1]
    assert not result.checks[1][1]


def test_json_acceptance_is_machine_checkable():
    result = evaluate_output(
        case(), json.dumps({"invoice_number": "1", "supplier": "ACME", "total": 9})
    )
    assert result.accepted


def test_json_acceptance_allows_fenced_agent_output():
    result = evaluate_output(
        case(),
        'Result:\n```json\n{"invoice_number":"1","supplier":"ACME","total":9}\n```',
    )
    assert result.accepted


def test_json_equals_binds_expected_values_to_fields():
    item = case(acceptance=(AcceptanceRule(
        "json_equals", {"invoice_number": "1", "supplier": "ACME", "total": 9}
    ),))
    assert evaluate_output(
        item, '{"invoice_number":"1","supplier":"ACME","total":9}'
    ).accepted


def test_json_types_checks_structure_without_accepting_booleans_as_numbers():
    item = case(acceptance=(AcceptanceRule(
        "json_types", {"records": "array", "amount": "number"}
    ),))
    assert evaluate_output(item, '{"records":[],"amount":9}').accepted
    assert not evaluate_output(item, '{"records":{},"amount":true}').accepted
    assert not evaluate_output(
        item,
        '{"invoice_number":"WRONG","supplier":"WRONG","total":0,'
        '"note":"1 ACME 9"}',
    ).accepted


def test_catalog_loader_and_coverage(tmp_path):
    data = {
        "case_id": "x",
        "title": "X",
        "tier": "base",
        "business_domain": "support",
        "provenance": "created",
        "source_urls": ["https://example.com/spec"],
        "source_note": "Public specification.",
        "input_driver": "output_units",
        "input_size": 2,
        "context_bytes": 50,
        "counts": {"classify": 2},
        "prompt": "Classify two tickets.",
        "acceptance": [{"kind": "min_lines", "value": 2}],
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    cases = load_cases(str(path))
    summary = catalog_summary(cases)
    assert summary["cases"] == 1
    assert summary["primitives_covered"] == 1
    assert "classify" not in summary["missing_primitives"]


def test_expanded_catalog_covers_all_new_bricks_with_public_evidence():
    items = load_cases(str(
        REPO_ROOT / "experiments" / "business_cases" / "expanded_cases.jsonl"
    ))
    covered = {
        slug for item in items for slug, count in item.counts.items() if count
    }
    assert covered == {
        "fetch", "score", "summarise", "monitor",
        "plan", "notify", "approve", "transform",
    }
    assert all(item.source_urls for item in items)


def test_percentile_interpolates_empirical_distribution():
    assert percentile([100, 200, 300], 0.50) == 200
    assert percentile([100, 200], 0.95) == pytest.approx(195)


def test_failed_attempt_spend_is_in_cost_per_accepted_outcome():
    outcomes = [
        WorkOutcome("a", True, total_tokens=100, cost_center="finance"),
        WorkOutcome("a", False, total_tokens=300, cost_center="finance"),
        WorkOutcome("a", True, total_tokens=200, cost_center="finance"),
    ]
    result = summarize(outcomes, Pricing(blended_per_million=10))
    assert result.acceptance_rate == pytest.approx(2 / 3)
    assert result.total_cost == pytest.approx(0.006)
    assert result.cost_per_accepted_outcome == pytest.approx(0.003)
    assert result.attempts_per_accepted_outcome == pytest.approx(1.5)
    assert result.tokens_per_accepted_outcome == pytest.approx(300)
    quote = result.quote(95)
    assert quote["risk_reserve_tokens_per_attempt"] > 0
    assert quote["budget_tokens_per_accepted_outcome"] > quote[
        "budget_tokens_per_attempt"
    ]


def test_split_input_output_pricing_and_chargeback():
    outcomes = [
        WorkOutcome("a", True, input_tokens=1_000_000, output_tokens=100_000,
                    cost_center="legal"),
        WorkOutcome("b", False, input_tokens=500_000, output_tokens=50_000,
                    cost_center="support"),
    ]
    rows = chargeback(outcomes, Pricing(input_per_million=2, output_per_million=8))
    assert rows[0]["actual_cost"] == pytest.approx(2.8)
    assert rows[0]["cost_per_accepted_outcome"] == pytest.approx(2.8)
    assert rows[1]["actual_cost"] == pytest.approx(1.4)
    assert rows[1]["cost_per_accepted_outcome"] == float("inf")


def test_cached_input_is_discounted_and_reasoning_is_not_double_counted():
    outcome = WorkOutcome(
        "reasoning-task",
        accepted=True,
        input_tokens=1_000,
        cached_tokens=400,
        output_tokens=300,
        reasoning_tokens=200,
        total_tokens=1_300,
    )
    pricing = Pricing(
        input_per_million=10,
        cached_input_per_million=2,
        output_per_million=50,
    )

    assert outcome.cost(pricing) == pytest.approx(
        (600 * 10 + 400 * 2 + 300 * 50) / 1_000_000
    )
    assert pricing.attempt_cost(
        input_tokens=1_000,
        cached_input_tokens=400,
        output_tokens=300,
    ) == outcome.cost(pricing)


def test_outcome_tail_quote_uses_explicit_geometric_retry_model():
    outcomes = [
        WorkOutcome(str(i), i == 0, total_tokens=100)
        for i in range(5)
    ]
    quote = summarize(outcomes, Pricing()).quote(95)
    assert quote["retry_model"] == "iid_geometric"
    assert quote["retry_attempts_at_percentile"] == 14
    assert quote["budget_tokens_per_accepted_outcome"] == 1400


def test_case_result_bridges_into_composition_training():
    row = to_run(case(held_out=True), {
        "total_tokens": 321,
        "tool_uses": 2,
        "provenance": "measured",
    })
    assert row.counts["extract"] == 3
    assert row.context_bytes == 100
    assert row.tokens == 321
    assert row.held_out


def test_runner_dry_run_and_live_record_bind_cost_to_acceptance():
    item = case()
    plan = dry_run(item)
    assert plan["prompt_sha256"]
    assert plan["source_manifest_sha256"]

    def dispatch(prompt):
        assert prompt == item.prompt
        return AgentResult(
            output='{"invoice_number":"1","supplier":"ACME","total":9}',
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            tool_uses=0,
            duration_ms=5,
            model="test-model",
            model_version="1",
        )

    record = run_case(item, dispatch)
    assert record.accepted
    assert record.total_tokens == 120
    assert record.prompt_sha256 == plan["prompt_sha256"]
    row = to_run(item, record)
    assert row.tokens == 120


def test_acceptance_only_proof_cannot_enter_training():
    with pytest.raises(ValueError, match="measured"):
        to_run(case(), {
            "total_tokens": 1,
            "proof_type": "acceptance-only",
        })


def test_committed_proofs_match_frozen_acceptance_rules():
    cases = {
        item.case_id: item
        for item in load_cases(str(
            REPO_ROOT / "experiments" / "business_cases" / "cases.jsonl"
        ))
    }
    proof_path = (
        REPO_ROOT / "experiments" / "business_cases" / "proof_results.jsonl"
    )
    proofs = [
        json.loads(line) for line in proof_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {proof["case_id"] for proof in proofs} == set(cases)
    assert all(proof["proof_type"] == "acceptance-only" for proof in proofs)
    for proof in proofs:
        verdict = evaluate_output(cases[proof["case_id"]], proof["output"])
        assert proof["accepted"] == verdict.accepted
    assert sum(proof["accepted"] for proof in proofs) == 11
