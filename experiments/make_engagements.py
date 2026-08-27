"""Generate the SYNTHETIC engagement corpus the value and impact heads train on.

    python -m experiments.make_engagements > experiments/engagements.jsonl

Read this before you read any number the prototype produces.

**Nothing here is a measurement.** The token model in ``token_yield`` is fitted
on 39 real agent runs against real filings; this file is not that. It is a
generator that invents a plausible delivery history so the multi-head machinery
can be demonstrated, tested and reviewed end to end before anyone wires it to
real data. No pricing, staffing or win-rate conclusion drawn from it means
anything about any real client.

It is committed, and seeded, for three reasons: the demo is reproducible, the
tests have a fixture, and — most importantly — the latent process below is
written down. When the heads recover these relationships you know the fitting
code works; when they fail to, you know it is the code and not the data. That
check is impossible with a corpus of real engagements, where a bad fit and a
weak signal look identical.

The latent process
------------------
Effort follows a power law in weighted scope (the shape effort estimation has
used since COCOMO), reuse discounts it, governance-heavy industries inflate the
coordination and the calendar, and price is anchored on what the industry pays
for the *goal* — only loosely on what the work costs, which is why margin varies
at all. Success is a Bernoulli draw from a logistic latent in which inherited
work is the dominant positive term and raw scope the dominant negative one.

Staffing is generated in two stages per role, matching how it is predicted: a
Bernoulli draw for whether the engagement used the role at all, then a
power-law draw for how many days if it did. Which roles are *likely* depends on
what the work is — a data scientist appears where classification and validation
dominate, a data engineer where retrieval and large corpora do, a change
manager where the work displaces an existing team — so a retail copy-generation
job and a healthcare coding audit come out with genuinely different teams
rather than the same three lines at different sizes.

Replacing this file with a Fabric extract of real delivery records is the whole
of the productionisation task for the value heads. See
``docs/product-prototype.md``.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import Dict, List, Optional

# Runnable as `python -m experiments.make_engagements` from the repo root, and
# as a plain script from anywhere — the second is how anyone regenerating the
# corpus on a Fabric or Azure ML compute will actually invoke it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_yield.usecase import GOALS, INDUSTRIES        # noqa: E402
from token_yield.tasks import ORDER as BRICKS              # noqa: E402

SEED = 20260827
N_ENGAGEMENTS = 150
HELD_OUT_FRACTION = 0.15

# ── industry character ───────────────────────────────────────────────────
# value_rate    : dollars a unit of weighted scope is worth to this industry.
#                 Calibrated so the corpus lands on a median gross margin of
#                 roughly 45% against the placeholder day rates in
#                 the placeholder day rates on the roster in
#                 project_yield.roles — a plausible services book, not a
#                 measured one. Price is anchored here rather than on realised
#                 effort on purpose: if it were derived from effort, margin
#                 would be constant by construction and there would be nothing
#                 for a scoping decision to discover.
# governance    : coordination and calendar multiplier
# integration   : how hard the systems are to build against
# win_adj       : log-odds shift on delivery success
INDUSTRY = {
    "financial_services": dict(value_rate=10400, governance=1.25,
                               integration=1.20, win_adj=+0.10),
    "healthcare":         dict(value_rate=8950, governance=1.45,
                               integration=1.30, win_adj=-0.25),
    "manufacturing":      dict(value_rate=6450, governance=0.95,
                               integration=1.00, win_adj=+0.30),
    "retail":             dict(value_rate=5500, governance=0.85,
                               integration=0.90, win_adj=+0.35),
    "public_sector":      dict(value_rate=7200, governance=1.70,
                               integration=1.25, win_adj=-0.45),
    "energy":             dict(value_rate=8200, governance=1.10,
                               integration=1.15, win_adj=0.00),
}

# ── what each goal is bought as, and which bricks it is made of ──────────
GOAL = {
    "cost_reduction": dict(
        value_mult=1.00, win_adj=+0.15,
        profile={"extract": 6.0, "classify": 8.0, "validate": 2.0,
                 "review": 1.5, "remediate": 1.0}),
    "revenue_growth": dict(
        value_mult=1.40, win_adj=-0.35,
        profile={"retrieve": 3.0, "draft": 4.0, "report": 3.0,
                 "review": 1.5, "extract": 1.0}),
    "compliance_risk": dict(
        value_mult=1.15, win_adj=+0.25,
        profile={"review": 4.0, "validate": 5.0, "reconcile": 3.5,
                 "extract": 2.0, "report": 1.5}),
    "customer_experience": dict(
        value_mult=0.90, win_adj=-0.10,
        profile={"classify": 6.0, "retrieve": 3.0, "draft": 2.5,
                 "remediate": 2.5, "report": 1.0}),
}

# ── who gets staffed, and how much ──────────────────────────────────────
#
# For each role: the log-odds of being used at all, and the days if used.
#   base      : log-odds intercept for presence
#   drivers   : per-brick-share pull on presence, as log-odds
#   goals     : per-goal pull on presence
#   scale     : days multiplier, applied to weighted scope ** exponent
#   exponent  : how days grow with scope
#   governance: how much the industry's governance load inflates the days
#   reuse     : how much of the role's work a continuation avoids
ROLE_PROCESS = {
    "solution_architect": dict(
        base=3.2, drivers={}, goals={}, scale=1.30, exponent=0.42,
        governance=0.35, reuse=0.52),
    "software_engineer": dict(
        base=3.6, drivers={}, goals={}, scale=1.85, exponent=0.78,
        governance=0.15, reuse=0.38),
    "project_manager": dict(
        base=3.0, drivers={}, goals={}, scale=0.75, exponent=0.55,
        governance=1.00, reuse=0.10),
    "data_scientist": dict(
        base=-1.1,
        drivers={"classify": 2.6, "validate": 1.9, "reconcile": 0.8},
        goals={"cost_reduction": 0.4, "compliance_risk": 0.3},
        scale=1.05, exponent=0.55, governance=0.20, reuse=0.45),
    "data_engineer": dict(
        base=-0.7,
        drivers={"retrieve": 3.0, "reconcile": 1.6, "extract": 1.0},
        goals={},
        scale=0.95, exponent=0.62, governance=0.10, reuse=0.55),
    "security_expert": dict(
        base=-1.0, drivers={"retrieve": 1.8, "extract": 1.2},
        goals={"compliance_risk": 1.6, "customer_experience": 0.5},
        scale=0.60, exponent=0.44, governance=1.10, reuse=0.60),
    "consultant": dict(
        base=-0.9, drivers={"review": 1.4, "report": 1.2},
        goals={"compliance_risk": 1.5, "revenue_growth": 0.7},
        scale=0.80, exponent=0.50, governance=0.90, reuse=0.30),
    "change_manager": dict(
        base=-1.4, drivers={"classify": 0.9, "remediate": 1.1},
        goals={"cost_reduction": 1.4, "customer_experience": 1.2},
        scale=0.55, exponent=0.45, governance=0.80, reuse=0.25),
}

# How much engineering one unit of each brick implies, relative to Classify.
BRICK_WEIGHT = {
    "review": 1.0, "extract": 1.2, "classify": 0.6, "retrieve": 2.4,
    "reconcile": 2.2, "draft": 1.1, "remediate": 1.8, "validate": 1.3,
    "report": 1.0,
}

CLIENT_NAMES = [
    "Northwind", "Contoso", "Fabrikam", "Tailspin", "Litware", "Adventure",
    "Proseware", "Wingtip", "Woodgrove", "Lamna", "Relecloud", "Trey",
    "Alpine", "Blue Yonder", "Coho", "Fourth Coffee", "Graphic Design",
    "Humongous", "Margie", "Nod Publishers", "Consolidated", "VanArsdel",
]

TITLE_STEM = {
    "cost_reduction": ["invoice intake", "claims intake", "document triage",
                       "back-office automation", "statement processing"],
    "revenue_growth": ["proposal drafting", "opportunity research",
                       "account briefing", "market scan", "bid support"],
    "compliance_risk": ["regulatory review", "control testing", "audit prep",
                        "policy reconciliation", "disclosure checking"],
    "customer_experience": ["case triage", "service response drafting",
                            "knowledge assist", "complaint handling",
                            "self-service answers"],
}


def _lognormal(rng: random.Random, sigma: float) -> float:
    """Multiplicative noise with a median of exactly 1."""
    return math.exp(rng.gauss(0.0, sigma) - sigma * sigma / 2.0)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def draw_counts(rng: random.Random, goal: str, scale: float) -> Dict[str, int]:
    """A brick multiset for a use case bought against ``goal``, at ``scale``."""
    profile = GOAL[goal]["profile"]
    counts = {b: 0 for b in BRICKS}
    for brick, weight in profile.items():
        drawn = rng.gauss(weight * scale, weight * scale * 0.35)
        counts[brick] = max(0, int(round(drawn)))
    if not sum(counts.values()):                    # never emit an empty scope
        counts[max(profile, key=profile.get)] = 1
    return counts


def inherit_counts(rng: random.Random, parent: Dict[str, int],
                   goal: str, scale: float) -> Dict[str, int]:
    """A continuation: most of the parent's shape, plus genuinely new work."""
    kept = rng.uniform(0.45, 0.85)
    counts = {b: int(round(parent[b] * kept)) for b in BRICKS}
    fresh = draw_counts(rng, goal, scale * rng.uniform(0.30, 0.70))
    for b in BRICKS:
        counts[b] += fresh[b]
    return counts


def weighted_scope(counts: Dict[str, int]) -> float:
    return sum(BRICK_WEIGHT[b] * n for b, n in counts.items())


def inherited_fraction(counts: Dict[str, int],
                       ancestors: List[Dict[str, int]]) -> float:
    total = sum(counts.values())
    if not total or not ancestors:
        return 0.0
    upstream = {b: max(a.get(b, 0) for a in ancestors) for b in BRICKS}
    return sum(min(counts[b], upstream[b]) for b in BRICKS) / total


def generate() -> List[dict]:
    rng = random.Random(SEED)
    records: List[dict] = []
    by_id: Dict[str, dict] = {}
    by_client: Dict[str, List[str]] = {}

    clients = []
    for i, name in enumerate(CLIENT_NAMES):
        clients.append(dict(name=name, industry=INDUSTRIES[i % len(INDUSTRIES)]))

    day = 0
    for n in range(N_ENGAGEMENTS):
        day += rng.randint(3, 14)
        client = rng.choice(clients)
        industry = client["industry"]
        ind = INDUSTRY[industry]

        # -- lineage: roughly two in five continue something already sold ----
        prior = by_client.get(client["name"], [])
        parent_id: Optional[str] = None
        if prior and rng.random() < 0.42:
            parent_id = rng.choice(prior[-4:])

        goal = (by_id[parent_id]["goal"] if parent_id and rng.random() < 0.6
                else rng.choice(GOALS))
        # Wide, and deliberately heavy at the bottom. A services book is not
        # made of medium-sized projects: the most common enterprise shape is a
        # per-item pipeline — one invoice, one ticket, one claim — run tens of
        # thousands of times a month, whose *per-run* scope is one or two
        # bricks. A corpus without those makes the tool warn "smaller than
        # anything I have seen" at exactly the use case it should handle best.
        scale = math.exp(rng.gauss(-0.15, 0.95))

        if parent_id:
            counts = inherit_counts(rng, by_id[parent_id]["counts"], goal, scale)
        else:
            counts = draw_counts(rng, goal, scale)

        # ancestors, for the reuse signal
        ancestors, cur, depth = [], parent_id, 0
        while cur and depth < 4:
            ancestors.append(by_id[cur]["counts"])
            cur = by_id[cur]["parent_id"]
            depth += 1
        inherited = inherited_fraction(counts, ancestors)
        siblings = [r["id"] for r in records
                    if parent_id and r["parent_id"] == parent_id]

        units = sum(counts.values())
        wscope = weighted_scope(counts)
        distinct = sum(1 for b in BRICKS if counts[b])
        context_bytes = int(max(0, rng.gauss(1800, 500))
                            * (counts["review"] + counts["extract"]
                               + counts["reconcile"] + counts["retrieve"]))

        # -- who is staffed, and for how long ------------------------------
        shares = {b: (counts[b] / units if units else 0.0) for b in BRICKS}
        role_days = {}
        for slug, proc in ROLE_PROCESS.items():
            z = (proc["base"]
                 + sum(w * shares.get(b, 0.0)
                       for b, w in proc["drivers"].items())
                 + proc["goals"].get(goal, 0.0)
                 + 0.35 * (math.log1p(units) - 2.2)
                 - 0.30 * inherited
                 + (0.9 if slug == "security_expert"
                    and industry in ("financial_services", "healthcare",
                                     "public_sector") else 0.0))
            if rng.random() >= _sigmoid(z):
                continue                     # this engagement did not use them
            days = (proc["scale"] * (wscope ** proc["exponent"])
                    * (1.0 + proc["governance"] * (ind["governance"] - 1.0))
                    * (1.0 + 0.14 * len(siblings) if slug == "project_manager"
                       else 1.0)
                    * (1.0 + 0.35 * distinct / 9.0
                       if slug == "solution_architect" else 1.0)
                    * ind["integration"] ** (0.6 if slug in
                                             ("software_engineer",
                                              "solution_architect",
                                              "data_engineer") else 0.0)
                    * (1.0 - proc["reuse"] * inherited)
                    * _lognormal(rng, 0.25))
            role_days[slug] = round(max(days, 0.25), 2)

        # The calendar is bounded by the build, not by the total headcount —
        # which is exactly why it is a separate head and not staff-days over a
        # team size.
        build = role_days.get("software_engineer", 1.0)
        calendar = (16.0 + 1.55 * (build ** 0.86) * ind["governance"]
                    * (1.0 - 0.24 * inherited)) * _lognormal(rng, 0.19)

        # -- price: anchored on the value of the goal to the industry ----
        value = (ind["value_rate"] * GOAL[goal]["value_mult"]
                 * (wscope ** 0.82) * (1.0 - 0.16 * inherited)
                 * _lognormal(rng, 0.33))

        # -- success: a Bernoulli draw from a logistic latent -------------
        z = (0.55
             + 1.45 * inherited
             + 0.32 * min(depth, 2)
             - 0.46 * (math.log1p(units) - 2.6)
             - 0.12 * len(siblings)
             + ind["win_adj"] + GOAL[goal]["win_adj"])
        won = rng.random() < _sigmoid(z)

        stem = rng.choice(TITLE_STEM[goal])
        uid = f"E{n + 1:03d}"
        rec = dict(
            id=uid,
            title=(f"{client['name']} {stem}"
                   + (" (phase 2)" if parent_id else "")),
            client=client["name"], industry=industry, goal=goal,
            counts=counts, context_bytes=context_bytes,
            contract_value=round(value, 2), won=won,
            role_days=role_days,
            calendar_days=round(calendar, 2),
            parent_id=parent_id, sibling_ids=siblings,
            started=f"day+{day}", provenance="synthetic",
            held_out=False,
        )
        records.append(rec)
        by_id[uid] = rec
        by_client.setdefault(client["name"], []).append(uid)

    # The most recent slice is held out — the honest test is predicting
    # forward in time, not predicting a random subset of the past.
    for rec in records[-int(N_ENGAGEMENTS * HELD_OUT_FRACTION):]:
        rec["held_out"] = True
    return records


def main() -> None:
    for rec in generate():
        print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
