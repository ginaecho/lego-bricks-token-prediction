"""Wave 3: basic (B), basic + type (BT) and basic + type + industry (BTI) compositions.

A composition is an ordered list of 1-4 *parts* from distinct basic functionalities.
A part is either a generic Studio basic functionality (``basic:<family>``, level B) or
one of its catalog types (level BT). An optional industry context turns the build into
BTI. Parts and industries are chosen from the catalog and curated industry profiles;
real-use-case references never drive the selection.
"""

from __future__ import annotations

import random
from collections import Counter

from .build_simulations import (
    BUILD_SCOPE, FAMILIES, PROTOCOL, STAFF, build_spec, pseudo_outcomes,
)
from .marketplace_agent_contracts import ATOMS, contracts, fingerprint

SPEC_VERSION = "composition-levels-v3"

# Mirrors the Studio CATALOG in marketplace-sales-demo.html (basic functionality cards).
BASICS = {
    "recommend": {"category": "Customer experience", "description": "Make the next product feel like the right product."},
    "research": {"category": "Intelligence", "description": "Turn questions and source material into decision-ready evidence."},
    "support": {"category": "Customer experience", "description": "Helpful, grounded answers. Even when your team is offline."},
    "search": {"category": "Customer experience", "description": "Help customers find what they mean, not just what they type."},
    "insights": {"category": "Intelligence", "description": "Turn scattered feedback into themes your team can act on."},
    "documents": {"category": "Operations", "description": "Extract, reshape, and check data without repetitive work."},
    "onboard": {"category": "Operations", "description": "Give every new customer a clear, personalized next step."},
}

# Curated from cited industry research (data/build_simulations/industry_use_cases.json).
# Every profile has exactly three entities, three constraints and two formats so that
# the instruction length is comparable across industries.
INDUSTRIES = {
    "finance": {
        "name": "Financial services",
        "entities": ["customer/KYC profile", "transaction", "loan application"],
        "constraints": ["KYC/AML screening flags", "PCI DSS masking of card numbers", "model-risk explainability of every score"],
        "formats": ["JSON/CSV transaction ledgers", "ISO 20022-style payment payloads"],
        "guidance": "fabricated customers, fake account numbers/IBANs and randomized transaction histories",
    },
    "insurance": {
        "name": "Insurance",
        "entities": ["policy", "claim", "underwriting submission"],
        "constraints": ["fair claims-handling decisions with stated reasons", "GDPR minimization of policyholder data",
                        "underwriting factors must be disclosed, not hidden"],
        "formats": ["ACORD-style XML/JSON submissions", "claim-form text"],
        "guidance": "invented policyholders, fabricated VINs/property addresses and randomized loss amounts and dates",
    },
    "healthcare": {
        "name": "Healthcare",
        "entities": ["patient record", "clinical note", "prior authorization request"],
        "constraints": ["HIPAA de-identification of patient identifiers", "no autonomous clinical decisions; human review flag",
                        "audit trail for every record change"],
        "formats": ["HL7 FHIR-style JSON resources", "clinical note text"],
        "guidance": "fictitious Synthea-style patients, diagnoses and lab values with no real MRNs, names or birth dates",
    },
    "manufacturing": {
        "name": "Manufacturing",
        "entities": ["work order", "sensor/telemetry reading", "quality inspection record"],
        "constraints": ["ISO 9001 traceability of quality decisions", "safety-critical alerts escalate to a human",
                        "units and tolerances validated on input"],
        "formats": ["time-series sensor CSV", "ERP/MES work-order JSON"],
        "guidance": "physically plausible randomized telemetry, fictitious machine/asset IDs and invented maintenance logs",
    },
    "retail": {
        "name": "Retail",
        "entities": ["product catalog SKU", "customer profile", "order/basket"],
        "constraints": ["GDPR/CCPA consent before personalization", "no fabricated reviews or endorsements",
                        "price and stock shown must match the catalog"],
        "formats": ["product catalog JSON feed", "clickstream event log"],
        "guidance": "fictitious catalog with invented SKUs, prices and stock plus synthetic shopper personas",
    },
    "energy": {
        "name": "Energy and utilities",
        "entities": ["meter reading", "grid asset", "outage report"],
        "constraints": ["NERC CIP-style protection of critical asset identifiers", "outage priority must follow declared safety rules",
                        "emissions figures carry units and source"],
        "formats": ["smart-meter interval CSV", "SCADA-style telemetry JSON"],
        "guidance": "synthetic grid topology, fictitious substation/device IDs and plausible seasonal consumption",
    },
    "public_sector": {
        "name": "Public sector",
        "entities": ["citizen service request", "benefits application", "policy document"],
        "constraints": ["Privacy Act/GDPR protection of citizen PII", "decisions explainable for FOIA/open-records review",
                        "accessible plain-language outputs (Section 508 spirit)"],
        "formats": ["government form JSON", "policy/regulation text"],
        "guidance": "fabricated citizen personas, invented case numbers and non-traceable addresses",
    },
    "telecom": {
        "name": "Telecommunications",
        "entities": ["subscriber account", "network fault ticket", "call/data usage record"],
        "constraints": ["GDPR/CCPA protection of subscriber data", "data-residency tag respected on every record",
                        "billing adjustments require a stated reason"],
        "formats": ["call detail records (CSV)", "support chat transcripts"],
        "guidance": "fictitious phone numbers/IMEIs, randomized usage volumes and invented support tickets",
    },
}

LEVELS = ("B", "BT", "BI", "BTI")
BUILDER_TEMPLATE_VERSION = "builder-instructions-v3"
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
- Write no files other than the four deliverables below (no prompt logs, notes or docs folders),
  and never write outside the working directory.

Build ONE integrated, runnable Python CLI reference implementation of this composition
({count} basic functionalit{plural}, executed in this order):
{steps}
{integration}{industry}Scope: bounded reference implementation with one shared input/output schema and one shared
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


def _registry() -> dict:
    return {item["id"]: item for item in contracts()}


def family_of(part: str) -> str:
    return part.split(":", 1)[1] if part.startswith("basic:") else _registry()[part]["feature_id"]


def generic_atoms(family: str) -> dict:
    """Core operations shared by every catalog type of a family (element-wise minimum)."""
    variants = [item["atoms"] for item in contracts() if item["feature_id"] == family]
    return {atom: min(item[atom] for item in variants) for atom in ATOMS}


def membership_id(parts: list[str], industry: str | None) -> str:
    names = "_".join(part.replace("basic:", "basic-") for part in parts)
    prefix = "single_" if len(parts) == 1 else f"combo{len(parts)}_"
    return prefix + names + (f"__ind-{industry}" if industry else "")


def split_group(parts: list[str], industry: str | None) -> str:
    if industry is None and not any(part.startswith("basic:") for part in parts):
        return build_spec(parts)["split_group"]  # same group as wave-1/2 builds of this membership
    return fingerprint({"parts": sorted(parts), "industry": industry})


def level_of(parts: list[str], industry: str | None) -> str:
    generic = all(part.startswith("basic:") for part in parts)
    if industry:
        return "BI" if generic else "BTI"
    return "B" if generic else "BT"


def build_spec_v3(parts: list[str], industry: str | None) -> dict:
    registry = _registry()
    if not 1 <= len(parts) <= 4 or len(set(parts)) != len(parts):
        raise ValueError("Choose one to four distinct parts")
    for part in parts:
        if part.startswith("basic:") and part[6:] not in BASICS or not part.startswith("basic:") and part not in registry:
            raise ValueError(f"Unknown part {part}")
    families = [family_of(part) for part in parts]
    if len(set(families)) != len(parts):
        raise ValueError("This design combines distinct basic functionalities")
    if industry is not None and industry not in INDUSTRIES:
        raise ValueError(f"Unknown industry {industry}")
    types = [part for part in parts if not part.startswith("basic:")]
    count = len(parts)
    atoms = [generic_atoms(family) if part.startswith("basic:") else registry[part]["atoms"]
             for part, family in zip(parts, families)]
    features = {
        "basic_functionality_ids": families,
        "basic_functionalities": [FAMILIES[family] for family in families],
        "parts": parts, "types": types, "industry": industry,
        "composition_level": level_of(parts, industry),
        "part_levels": ["B" if part.startswith("basic:") else "BT" for part in parts],
        "functionality_count": count, "integration_edge_count": count - 1,
        "shared_schema_count": 1, "shared_validation_layer_count": 1,
        "planned_acceptance_case_count": 6 + 3 * (count - 1) + (2 if industry else 0),
        "generic_basic_part_count": sum(part.startswith("basic:") for part in parts),
        "industry_context": int(industry is not None),
        "industry_constraint_count": len(INDUSTRIES[industry]["constraints"]) if industry else 0,
        "artifact_kind": "python_cli_reference_implementation",
        "llm_integration": "injected_callable_validated_with_test_fixtures",
        "estimated_months_to_finish": .5 + .25 * (count - 1),
        **{name: round(value * (1 + .25 * (count - 1)), 3) for name, value in STAFF.items()},
        **{f"has_type_{item}": int(item in types) for item in registry},
        **{f"has_generic_{family}": int(f"basic:{family}" in parts) for family in BASICS},
        **{f"industry_{name}": int(industry == name) for name in INDUSTRIES},
        **{f"planned_operation_{atom}": sum(item[atom] for item in atoms) for atom in ATOMS},
    }
    group = split_group(parts, industry)
    bucket = int(group[:8], 16) % 10
    return {
        "id": membership_id(parts, industry), "protocol": PROTOCOL, "spec_version": SPEC_VERSION,
        "origin": "catalog_designed_simulation", "catalog_origin": "independent_functionality_taxonomy",
        "industry_origin": "curated_industry_profile_not_a_specific_case" if industry else None,
        "source_case_ids": [], "input_features": features,
        "requirements": [BASICS[part[6:]]["description"] if part.startswith("basic:") else BUILD_SCOPE[part]
                         for part in parts],
        "integration": {
            "ordered_parts": parts,
            "edges": [{"from": left, "to": right, "kind": "data_handoff"} for left, right in zip(parts, parts[1:])],
            "instruction": "Build one integrated implementation, not a sum or concatenation of standalone token estimates.",
        },
        "planning_assumptions": {
            "source": "declared_scenario_not_observed_staffing", "staff_unit": "FTE",
            "causal_status": "These staffing/duration values were not experimental interventions on the builders.",
        },
        "split": "test" if bucket >= 8 else "validation" if bucket == 7 else "train",
        "split_group": group,
        "split_rule": "Keep repeated/reordered builds of the same parts and industry in one partition.",
    }


def instructions(trial: dict, directory: str) -> str:
    registry = _registry()
    parts, industry = trial["parts"], trial["industry"]
    lines = []
    for index, part in enumerate(parts, 1):
        if part.startswith("basic:"):
            family = part[6:]
            lines.append(f"  {index}. {FAMILIES[family]} (basic functionality, {BASICS[family]['category']}; "
                         f"no variant prescribed): {BASICS[family]['description']} Implement a general-purpose version.")
        else:
            lines.append(f"  {index}. {registry[part]['name']} ({FAMILIES[registry[part]['feature_id']]}): {BUILD_SCOPE[part]}")
    if len(parts) == 1:
        integration, coverage = "", "normal, edge, invalid-input and CLI behavior"
    else:
        edges = ", ".join(f"{left} -> {right}" for left, right in zip(parts, parts[1:]))
        integration = (f"Integration: data handoffs {edges}. Build one integrated pipeline where each stage "
                       "consumes the previous stage's validated output; share schemas and validation. "
                       "Do not build separate standalone programs and concatenate them.\n")
        coverage = "each stage, cross-stage propagation, edge, invalid-input and CLI behavior"
    if industry:
        profile = INDUSTRIES[industry]
        block = (f"Industry context: {profile['name']}. Model inputs and outputs on these entities: "
                 f"{'; '.join(profile['entities'])}. Enforce these constraints as demonstrative validation rules "
                 f"(no compliance certification claims): {'; '.join(profile['constraints'])}. "
                 f"Mimic these data formats: {'; '.join(profile['formats'])}. Synthetic fixtures: {profile['guidance']}.\n")
        coverage = coverage.replace(" and CLI", ", industry-constraint and CLI")
    else:
        block = ""
    return BUILDER_TEMPLATE.format(
        trial_id=trial["trial_id"], directory=directory, count=len(parts),
        plural="y" if len(parts) == 1 else "ies", steps="\n".join(lines), integration=integration,
        industry=block, acceptance=build_spec_v3(parts, industry)["input_features"]["planned_acceptance_case_count"],
        coverage=coverage)


# Balanced 160-build plan.
WAVE3_PLAN = {
    "B_single": 14, "BT_single_repeat": 16, "BTI_single": 48,
    "combo_B": 17, "combo_BI": 17, "combo_mixed": 16, "combo_mixedI": 16, "combo_BTI": 16,
}
SPLIT_SHARE = {"test": .20, "validation": .15}


def _split_quota(total: int) -> dict:
    test, validation = round(total * SPLIT_SHARE["test"]), round(total * SPLIT_SHARE["validation"])
    return {"test": test, "validation": validation, "train": total - test - validation}


def select_wave3(measured_groups: set[str], seed: int, wave: str) -> list[dict]:
    """Selection uses only catalog memberships, industries and split groups; never token labels."""
    rng = random.Random(seed)
    types_by_family: dict = {}
    for item in contracts():
        types_by_family.setdefault(item["feature_id"], []).append(item["id"])
    families, industries = sorted(BASICS), sorted(INDUSTRIES)
    used = set(measured_groups)
    use: Counter = Counter()
    groups: dict = {}

    def load(parts, industry):
        return sum(use[p] for p in parts) + sum(use[f] for f in map(family_of, parts)) + (use[industry] if industry else 0)

    def accept(category, parts, industry):
        use.update(parts)
        use.update(map(family_of, parts))
        if industry:
            use[industry] += 1
        groups.setdefault(category, []).append({"parts": parts, "industry": industry, "category": category})
        used.add(split_group(parts, industry))

    for family in families * 2:
        accept("B_single", [f"basic:{family}"], None)
    for item in sorted(contracts(), key=lambda entry: entry["id"]):
        accept("BT_single_repeat", [item["id"]], None)

    def fill(category, candidates, amount):
        quota = _split_quota(amount)
        pool = [c for c in candidates if split_group(*c) not in used]
        rng.shuffle(pool)
        splits = {id(c): build_spec_v3(*c)["split"] for c in pool}
        keys = {id(c): split_group(*c) for c in pool}
        while sum(quota.values()):
            pool = [c for c in pool if keys[id(c)] not in used and quota[splits[id(c)]] > 0]
            if not pool:
                raise ValueError(f"Not enough eligible memberships for {category}")
            best = min(pool, key=lambda c: (load(*c), rng.random()))
            quota[splits[id(best)]] -= 1
            accept(category, *best)

    fill("BTI_single", [([t], ind) for t in sorted(_registry()) for ind in industries], WAVE3_PLAN["BTI_single"])

    def combos(mode, with_industry, count):
        result = []
        for _ in range(count):
            size = rng.choice((2, 3, 4))
            chosen = rng.sample(families, size)
            if mode == "B":
                parts = [f"basic:{f}" for f in chosen]
            elif mode == "BT":
                parts = [rng.choice(types_by_family[f]) for f in chosen]
            else:
                typed = set(rng.sample(range(size), rng.randint(1, size - 1)))
                parts = [rng.choice(types_by_family[f]) if i in typed else f"basic:{f}" for i, f in enumerate(chosen)]
            result.append((parts, rng.choice(industries) if with_industry else None))
        return result

    for category, mode, with_industry in (("combo_B", "B", False), ("combo_BI", "B", True),
                                          ("combo_mixed", "mixed", False), ("combo_mixedI", "mixed", True),
                                          ("combo_BTI", "BT", True)):
        need = WAVE3_PLAN[category]
        per_size = {2: need // 3 + (need % 3 > 0), 3: need // 3 + (need % 3 > 1), 4: need // 3}
        candidates = combos(mode, with_industry, 4000)
        for size, amount in per_size.items():
            fill(category, [c for c in candidates if len(c[0]) == size], amount)

    ordered, queues = [], [list(groups[name]) for name in WAVE3_PLAN]
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    seen: Counter = Counter()
    for index, trial in enumerate(ordered, 1):
        spec = build_spec_v3(trial["parts"], trial["industry"])
        seen[spec["split_group"]] += 1
        trial.update({
            "trial_id": f"{wave}_t{index:03d}_{spec['id']}", "membership_point_id": spec["id"],
            "level": spec["input_features"]["composition_level"],
            "kind": "repeat_of_earlier_wave" if spec["split_group"] in measured_groups
            else "repeat_within_wave" if seen[spec["split_group"]] > 1 else "new_membership",
            "split": spec["split"], "split_group": spec["split_group"],
            "repeat_index_in_wave": seen[spec["split_group"]],
        })
    return ordered


def trial_point(trial: dict, wave: str, instruction_sha256: str) -> dict:
    spec = {**build_spec_v3(trial["parts"], trial["industry"]), "id": trial["trial_id"]}
    return {
        **spec, "spec_sha256": fingerprint(spec), "status": "planned_unmeasured",
        "wave": wave, "trial_id": trial["trial_id"], "membership_point_id": trial["membership_point_id"],
        "trial_kind": trial["kind"], "trial_category": trial["category"],
        "builder_instructions_version": BUILDER_TEMPLATE_VERSION,
        "builder_instructions_sha256": instruction_sha256,
        "build_token_usage": None, "azure_cost_scenarios": [],
        "pseudo_outcomes": pseudo_outcomes(trial["trial_id"]), "build_evidence": None,
    }
