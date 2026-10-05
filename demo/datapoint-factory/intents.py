"""Default intent of every brick: the exact task the builder agent receives.

Rendered with main's ``build_waves_v3.instructions`` for a single-part build,
so the text shown is the same text an isolated builder is given.
"""

from __future__ import annotations

import re

from token_yield import build_waves_v3 as v3
from token_yield.marketplace_agent_contracts import ATOM_DESCRIPTIONS, contracts

TASK = re.compile(r"^\s+1\. (.*)$", re.M)
ACCEPTANCE = re.compile(r"at least (\d+) test cases covering\s+(.*?);", re.S)


def _operations(atoms: dict) -> list[dict]:
    return [{"operation": name, "count": count, "meaning": ATOM_DESCRIPTIONS[name]}
            for name, count in atoms.items() if count]


def intent(part: str, atoms: dict, corpus) -> dict:
    text = v3.instructions({"parts": [part], "industry": None, "trial_id": "preview"}, "<staging>")
    tests, coverage = ACCEPTANCE.search(text).groups()
    measured = [p["build_token_usage"]["total_tokens"]
                for p in corpus.by_group.get(v3.build_spec_v3([part], None)["split_group"], [])]
    return {
        "task": TASK.search(text).group(1),
        "prompt": text,
        "operations": _operations(atoms),
        "acceptance": f"At least {tests} unit tests covering {' '.join(coverage.split())}.",
        "deliverables": "implementation.py (CLI) · test_implementation.py · example_input.json · build_manifest.json",
        "measured_builds": len(measured),
        "mean_tokens": sum(measured) / len(measured) if measured else None,
    }


def brick_intents(corpus) -> dict:
    """{part id: intent} for every generic brick and every catalog variant."""
    result = {f"basic:{family}": intent(f"basic:{family}", v3.generic_atoms(family), corpus) for family in v3.BASICS}
    result.update({item["id"]: intent(item["id"], item["atoms"], corpus) for item in contracts()})
    return result
