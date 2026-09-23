"""Frozen, append-only measurement waves for the construction-token corpus."""

from __future__ import annotations

import random
from collections import Counter

from .build_simulations import BUILD_SCOPE, FAMILIES, build_spec, combination_design, point_id, pseudo_outcomes
from .marketplace_agent_contracts import contracts, fingerprint

BUILDER_TEMPLATE_VERSION = "builder-instructions-v2"
BUILDER_TEMPLATE = """You are an isolated software builder for a token-measurement experiment.
Your entire session is measured as ONE construction label, so work normally and completely,
but do not do unrelated exploration.

Build ID: {trial_id}
Working directory (create it if missing; write ONLY here): {directory}

Isolation rules (mandatory):
- Start from an empty directory. Do not read, list, search or copy any other file or folder
  in the repository or on this machine. Do not look at other builds.
- Do not spawn sub-agents, do not use git, do not call networks, web tools or model providers.
- Use Python 3 standard library only.

Build ONE integrated, runnable Python CLI reference implementation of this composition
({count} basic functionalit{plural}, executed in this order):
{steps}
{integration}
Scope: bounded reference implementation with one shared input/output schema and one shared
validation layer; deterministic logic; any LLM step is an optional injected Python callable
validated with test fixtures (never a live provider). Not enterprise production delivery.
Use clearly labeled synthetic fixture data.

Deliverables, exactly these four files in the working directory:
1. implementation.py - `python -B implementation.py example_input.json` must print one JSON
   object to stdout and exit 0. Validation/file errors print JSON with status "error", exit 2.
2. test_implementation.py - unittest suite with at least {acceptance} test cases covering
   {coverage}; `python -B -m unittest discover -s . -p test_implementation.py` must pass.
3. example_input.json - synthetic example input accepted by the CLI.
4. build_manifest.json - JSON with keys: build_id, composition_label, construction
   (started_with_empty_directory, implementation_origin, runtime, external_dependencies,
   networks_or_providers_called, agents_spawned, git_used), deliverables, commands,
   shared_schema, cross_stage_handoffs, behavior, scope (included/excluded), test_outcomes.
   Do not include token counts, cost, staffing or client scores.

Run the tests and the CLI yourself and repair until both pass. Finish with a short report
(under 120 words) stating the file list and test result.
"""


def instructions(trial: dict, directory: str) -> str:
    registry = {item["id"]: item for item in contracts()}
    types = trial["types"]
    steps = "\n".join(
        f"  {index}. {registry[item]['name']} ({FAMILIES[registry[item]['feature_id']]}): {BUILD_SCOPE[item]}"
        for index, item in enumerate(types, 1))
    if len(types) == 1:
        integration = ""
        coverage = "normal, edge, invalid-input and CLI behavior"
    else:
        edges = ", ".join(f"{left} -> {right}" for left, right in zip(types, types[1:]))
        integration = (f"Integration: data handoffs {edges}. Build one integrated pipeline where each stage "
                       "consumes the previous stage's validated output; share schemas and validation. "
                       "Do not build separate standalone programs and concatenate them.\n")
        coverage = "each stage, cross-stage propagation, edge, invalid-input and CLI behavior"
    return BUILDER_TEMPLATE.format(
        trial_id=trial["trial_id"], directory=directory, count=len(types),
        plural="y" if len(types) == 1 else "ies", steps=steps, integration=integration,
        acceptance=build_spec(types)["input_features"]["planned_acceptance_case_count"], coverage=coverage)


def _pick(candidates: list[dict], quota: int, counts: Counter, rng: random.Random) -> list[dict]:
    pool, chosen = list(candidates), []
    rng.shuffle(pool)
    for _ in range(min(quota, len(pool))):
        best = min(pool, key=lambda item: (sum(counts[name] for name in item["types"]), rng.random()))
        pool.remove(best)
        chosen.append(best)
        counts.update(best["types"])
    if len(chosen) != quota:
        raise ValueError("Not enough eligible memberships for the requested quota")
    return chosen


def select_wave(measured: list[dict], quotas: dict, seed: int, wave: str) -> list[dict]:
    """Choose trials without looking at any token labels; only memberships and split groups."""
    rng = random.Random(seed)
    measured_groups = {item["split_group"] for item in measured}
    design = [item for item in combination_design() if item["split_group"] not in measured_groups]
    counts: Counter = Counter()
    trials = []
    for item in sorted(measured, key=lambda point: point["id"]):
        trials.append({"types": item["types"], "kind": "repeat_of_wave1", "repeat_of": item["id"]})
    counts.update(name for item in trials for name in item["types"])
    new = []
    for size in (2, 3, 4):
        for split in ("train", "validation", "test"):
            quota = quotas.get(f"{size}_{split}", 0)
            eligible = [item for item in design if len(item["types"]) == size and item["split"] == split]
            for item in _pick(eligible, quota, counts, rng):
                order = list(item["types"])
                rng.shuffle(order)
                new.append({"types": order, "kind": "new_membership", "repeat_of": None})
    trials += new
    for size in (2, 3, 4):
        for split in ("train", "validation"):
            options = [item for item in new if len(item["types"]) == size
                       and build_spec(item["types"])["split"] == split]
            for item in rng.sample(options, quotas.get(f"repeat_{size}_{split}", 0)):
                trials.append({"types": item["types"], "kind": "repeat_of_wave2", "repeat_of": point_id(item["types"])})
    strata: dict = {}
    for item in trials:
        strata.setdefault((item["kind"] != "repeat_of_wave2", len(item["types"])), []).append(item)
    ordered = []
    primary = [strata[key] for key in sorted(strata) if key[0]]
    while any(primary):
        for group in primary:
            if group:
                ordered.append(group.pop(0))
    ordered += [item for key in sorted(strata) if not key[0] for item in strata[key]]
    seen: Counter = Counter()
    for index, item in enumerate(ordered, 1):
        spec = build_spec(item["types"])
        seen[spec["split_group"]] += 1
        item.update({
            "trial_id": f"{wave}_t{index:03d}_{spec['id']}", "membership_point_id": spec["id"],
            "split": spec["split"], "split_group": spec["split_group"],
            "repeat_index_in_wave": seen[spec["split_group"]],
        })
    return ordered


def trial_point(trial: dict, wave: str, instruction_sha256: str) -> dict:
    spec = {**build_spec(trial["types"]), "id": trial["trial_id"]}
    return {
        **spec, "spec_sha256": fingerprint(spec), "status": "planned_unmeasured",
        "wave": wave, "trial_id": trial["trial_id"], "membership_point_id": trial["membership_point_id"],
        "trial_kind": trial["kind"], "repeat_of": trial["repeat_of"],
        "builder_instructions_version": BUILDER_TEMPLATE_VERSION,
        "builder_instructions_sha256": instruction_sha256,
        "build_token_usage": None, "azure_cost_scenarios": [],
        "pseudo_outcomes": pseudo_outcomes(trial["trial_id"]), "build_evidence": None,
    }
