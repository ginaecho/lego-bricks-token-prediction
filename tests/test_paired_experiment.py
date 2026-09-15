"""Tests for metered dispatch parsing and paired experiment partitions."""

import json
import os

import pytest

from examples.paired_experiment import held_out_cases, training_cases
from token_yield.copilot_dispatch import parse_usage
from token_yield.copilot_dispatch import CopilotDispatcher


def test_usage_parser_uses_provider_reported_input_and_output():
    data = {
        "modelMetrics": {
            "claude-haiku-4.5": {
                "usage": {"inputTokens": 100, "outputTokens": 25}
            }
        }
    }
    assert parse_usage(data, "claude-haiku-4.5") == {
        "input_tokens": 100,
        "output_tokens": 25,
        "total_tokens": 125,
    }


def test_usage_parser_rejects_wrong_model():
    with pytest.raises(ValueError, match="model metrics"):
        parse_usage({"modelMetrics": {}}, "claude-haiku-4.5")


def test_dispatcher_avoids_windows_batch_shim_for_literal_prompts(tmp_path):
    dispatcher = CopilotDispatcher(tmp_path, "x")
    if os.name == "nt":
        assert dispatcher.command_prefix[0] == "node"
        assert dispatcher.command_prefix[1].endswith("npm-loader.js")
    else:
        assert dispatcher.command_prefix == ["copilot"]


def test_train_and_held_out_partitions_do_not_overlap():
    train = training_cases()
    held = held_out_cases()
    train_ids = {case.case_id for case in train}
    held_ids = {case.case_id for case in held}
    assert not train_ids & held_ids
    assert len(train) >= 20
    assert len(held) >= 4
    assert all(not case.held_out for case in train)
    assert all(case.held_out for case in held)


def test_training_has_two_sizes_for_every_primitive():
    sizes = {}
    for case in training_cases():
        if case.tier != "base":
            continue
        for slug, count in case.counts.items():
            if count:
                sizes.setdefault(slug, set()).add((count, case.context_bytes))
    assert all(len(shapes) >= 2 for shapes in sizes.values())
    assert len(sizes) == 9


def test_real_cases_are_source_backed_and_held_out():
    real = [case for case in held_out_cases() if case.case_id.startswith("real-")]
    assert len(real) >= 3
    assert all(case.provenance == "source-backed" for case in real)
