"""Tests for the frozen wave-3 repair case matrix."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from token_yield.wave3_repair_cases import load_repair_cases

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "foundry_wave3"


def test_repair_matrix_has_frozen_allocation_and_unique_ids():
    cases = load_repair_cases(EXPERIMENT)
    assert len(cases) == len({case.case_id for case in cases}) == 36
    assert Counter(case.family for case in cases) == {
        "summarise": 12,
        "fetch": 18,
        "control": 4,
        "tokenizer_confirmation": 2,
    }


def test_warm_replicates_follow_a_seed_in_every_group():
    cases = load_repair_cases(EXPERIMENT)
    positions = defaultdict(list)
    for index, case in enumerate(cases):
        positions[case.group_id].append((index, case.cache_warm))
    for values in positions.values():
        cold = [index for index, warm in values if not warm]
        warm = [index for index, warm in values if warm]
        assert cold
        assert not warm or min(warm) > min(cold)


def test_fetch_arms_have_three_replicates_and_declared_bytes():
    cases = [case for case in load_repair_cases(EXPERIMENT)
             if case.family == "fetch"]
    groups = Counter(case.group_id for case in cases)
    assert set(groups.values()) == {3}
    assert all(case.declared_fetch_bytes > 0 for case in cases)
    assert all(case.k_declared == (2 if case.arm == "live" else 1)
               for case in cases)


def test_source_snapshot_hashes_are_frozen():
    registry = json.loads(
        (EXPERIMENT / "source_registry.json").read_text(encoding="utf-8")
    )
    manifests = json.loads(
        (EXPERIMENT / "api_manifests.json").read_text(encoding="utf-8")
    )
    entries = [*registry["summary_sources"].values(), *manifests.values()]
    for entry in entries:
        path = EXPERIMENT / entry["snapshot_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[
            "snapshot_sha256"
        ]


def test_order_is_deterministic_but_seed_sensitive():
    first = [case.case_id for case in load_repair_cases(EXPERIMENT, 7)]
    assert first == [case.case_id for case in load_repair_cases(EXPERIMENT, 7)]
    assert first != [case.case_id for case in load_repair_cases(EXPERIMENT, 8)]
