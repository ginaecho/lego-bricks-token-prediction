"""The encoder: a paragraph of use-case description becomes a feature vector.

This is the same autoencoder move :mod:`token_yield.decompose` makes, widened.
There, a request encodes to a multiset of task bricks and decodes to tokens.
Here it encodes to bricks *plus the commercial context* — industry, business
goal, how much material is in scope, how often it will run — and decodes
through six heads to price, risk, staffing and time.

The vocabulary is fixed and the encoder chooses within it. That constraint is
what makes the whole thing work: a free-form summary cannot be a feature
vector, and an encoder allowed to invent categories produces a model whose
features drift under it.

Two encoders are provided, and every encoding records which produced it:

* :func:`encode_prompt` / :func:`parse_encoding` — an agent reads the
  description and exercises judgement. This is the real one. In a deployment
  it is a model deployed on Azure AI Foundry; see :mod:`project_yield.azure`.
* :func:`heuristic_encode` — keywords and a quantity regex, for when no model
  is reachable. It is genuinely worse: it sees words rather than intent. It
  exists so a pipeline degrades instead of stopping, and every surface that
  displays an estimate also displays which encoder produced it.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .usecase import BRICKS, GOALS, INDUSTRIES, UseCase, normalise_counts
from token_yield.tasks import PRIMITIVES


# ── the agent encoder ────────────────────────────────────────────────────

def encode_prompt(description: str) -> str:
    """The prompt that asks a model to encode a use case.

    Every vocabulary the model must choose from is supplied in full, so it is
    selecting among defined terms rather than inventing a scheme of its own —
    and the counting rule is spelled out, because "one unit" is the single
    place where two encoders most easily disagree.
    """
    bricks = "\n".join(
        f"- {p.name} ({p.slug}) — {p.blurb} Typical use: {p.industry}."
        for p in (PRIMITIVES[s] for s in BRICKS))
    return (
        "You are scoping an AI use case for a delivery team. Encode it into a "
        "fixed vocabulary so that its cost, price, risk, staffing and duration "
        "can be estimated from comparable past engagements.\n\n"
        f"TASK BRICKS\n{bricks}\n\n"
        "INDUSTRIES\n- " + "\n- ".join(INDUSTRIES) + "\n\n"
        "BUSINESS GOALS\n- " + "\n- ".join(GOALS) + "\n\n"
        f"THE USE CASE\n{description}\n\n"
        "Count bricks for ONE end-to-end run of the finished pipeline, not for "
        "the whole production backlog. One unit means one document reviewed, "
        "one field extracted, one item classified, one fact retrieved, one "
        "comparison, one piece drafted, one error corrected, one check written, "
        "or one aspect reported. Count only work the use case actually asks "
        "for.\n\n"
        "Also estimate:\n"
        "- context_bytes: total size in bytes of the source material one run "
        "reads. Zero if it reads nothing.\n"
        "- monthly_runs: how many times per month the finished pipeline runs "
        "in production. Zero if the description does not say or imply it.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"counts": {"<slug>": <int>, ...}, "industry": "<industry>", '
        '"goal": "<goal>", "context_bytes": <int>, "monthly_runs": <int>, '
        '"rationale": "<one sentence>"}'
    )


def parse_encoding(text: str, uid: str = "new", title: str = "",
                   description: str = "") -> UseCase:
    """Read a model's reply into a :class:`UseCase`.

    Tolerates the usual wrappers — fenced code blocks, a sentence before the
    JSON — because failing on formatting rather than on substance helps nobody.
    Unknown brick slugs are dropped and unknown industries or goals fall back
    to the reference category, since silently inventing a feature level would
    corrupt the vector the heads were fitted on.
    """
    blob = re.search(r"\{.*\}", text, re.S)
    if not blob:
        raise ValueError("no JSON object in the encoder reply")
    data = json.loads(blob.group(0))

    industry = str(data.get("industry", "")).strip().lower()
    goal = str(data.get("goal", "")).strip().lower()
    return UseCase(
        id=uid, title=title or str(data.get("title", "") or "Untitled use case"),
        description=description,
        industry=industry if industry in INDUSTRIES else INDUSTRIES[0],
        goal=goal if goal in GOALS else GOALS[0],
        counts=normalise_counts(data.get("counts") or {}),
        context_bytes=_as_int(data.get("context_bytes")),
        monthly_runs=_as_int(data.get("monthly_runs")),
        encoder="agent",
        rationale=str(data.get("rationale", "")).strip(),
    )


def _as_int(value) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


# ── the keyword fallback ─────────────────────────────────────────────────

_BRICK_WORDS: Dict[str, Tuple[str, ...]] = {
    "review": ("read", "review", "go through", "assess", "summarise",
               "summarize", "understand", "ingest"),
    "extract": ("extract", "pull", "capture", "populate", "field", "structured",
                "parse", "into a table", "line item"),
    "classify": ("classify", "categorise", "categorize", "triage", "route",
                 "tag", "sort", "prioritise", "prioritize"),
    "retrieve": ("find", "locate", "search", "look up", "which document",
                 "knowledge base", "retrieval", "rag"),
    "reconcile": ("reconcile", "compare", "cross-check", "tie out", "match "
                  "against", "discrepan", "disagree", "variance"),
    "draft": ("draft", "write a", "compose", "generate a", "prepare a",
              "respond", "reply"),
    "remediate": ("fix", "correct", "remediate", "resolve", "clean up",
                  "exception"),
    "validate": ("validate", "check", "verify", "control test", "audit",
                 "quality"),
    "report": ("report", "board note", "summary for", "write up", "memo",
               "dashboard"),
}

_INDUSTRY_WORDS: Dict[str, Tuple[str, ...]] = {
    "financial_services": ("bank", "insur", "financial", "trading", "loan",
                           "credit", "kyc", "aml", "claims", "underwrit"),
    "healthcare": ("health", "clinic", "patient", "hospital", "medical",
                   "pharma", "care provider"),
    "manufacturing": ("manufactur", "factory", "plant", "supply chain",
                      "production line", "supplier", "bill of materials"),
    "retail": ("retail", "store", "merchand", "ecommerce", "e-commerce",
               "shopper", "sku", "catalogue", "catalog"),
    "public_sector": ("government", "public sector", "council", "ministry",
                      "agency", "citizen", "municipal", "federal"),
    "energy": ("energy", "utility", "grid", "oil", "gas", "renewable",
               "power station", "drilling"),
}

_GOAL_WORDS: Dict[str, Tuple[str, ...]] = {
    "cost_reduction": ("cost", "efficien", "manual effort", "headcount",
                       "automate", "throughput", "backlog", "productivity"),
    "revenue_growth": ("revenue", "sales", "pipeline", "win rate", "upsell",
                       "bid", "proposal", "growth", "opportunit"),
    "compliance_risk": ("complian", "regulat", "audit", "risk", "control",
                        "policy", "obligation", "governance", "disclosure"),
    "customer_experience": ("customer", "citizen experience", "satisfaction",
                            "response time", "service", "complaint", "csat",
                            "self-service"),
}

#: "12 invoices", "around 300 claims a day" — a number attached to a noun.
_QUANTITY = re.compile(r"(\d[\d,]*)\s*(?:\w+\s+){0,2}?"
                       r"(document|documents|file|files|invoice|invoices|"
                       r"claim|claims|record|records|contract|contracts|"
                       r"ticket|tickets|case|cases|filing|filings|report|"
                       r"reports|field|fields|email|emails|page|pages)",
                       re.I)

_RUN_RATE = re.compile(r"(\d[\d,]*)\s*(?:\w+\s+){0,3}?"
                       r"(?:per|a|each|/)\s*(day|week|month|year)", re.I)

_PER_MONTH = {"day": 30.0, "week": 4.33, "month": 1.0, "year": 1.0 / 12.0}


def _score_vocab(text: str, words: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    hits = {key: sum(text.count(w) for w in ws) for key, ws in words.items()}
    best = max(hits, key=lambda k: hits[k])
    return best if hits[best] else None


def heuristic_encode(description: str, uid: str = "new",
                     title: str = "") -> UseCase:
    """Keyword fallback. Weaker by construction, and it says so.

    It can do one thing the token model's fallback cannot: read a quantity out
    of the text, so "triage 400 tickets" is not scored the same as "triage a
    ticket". That is a guess at scale rather than a reading of intent, and it
    is still the difference between a useful estimate and a meaningless one.
    """
    low = description.lower()

    quantities: List[int] = [
        int(m.group(1).replace(",", "")) for m in _QUANTITY.finditer(low)]

    monthly = 0
    throughput = 0
    rate = _RUN_RATE.search(low)
    if rate:
        throughput = int(rate.group(1).replace(",", ""))
        monthly = int(throughput * _PER_MONTH[rate.group(2).lower()])

    # "400 claims a day" is throughput, not the size of one run. Reading it as
    # both — 400 units per run AND 12,000 runs a month — overstates the scope
    # by the volume itself, which is the most expensive mistake this fallback
    # could make. When the same number carries the rate, one run handles one item.
    if quantities and throughput and quantities[0] == throughput:
        default_units = 1
    else:
        default_units = max(1, min(quantities[0], 500)) if quantities else 1

    counts = {slug: 0 for slug in BRICKS}
    for slug, words in _BRICK_WORDS.items():
        if any(w in low for w in words):
            counts[slug] = default_units if slug in (
                "review", "extract", "classify", "validate") else 1
    if not sum(counts.values()):
        counts["review"] = 1

    return UseCase(
        id=uid, title=title or (description.strip().split("\n")[0][:60]
                                or "Untitled use case"),
        description=description,
        industry=_score_vocab(low, _INDUSTRY_WORDS) or INDUSTRIES[0],
        goal=_score_vocab(low, _GOAL_WORDS) or GOALS[0],
        counts=counts,
        # 2 kB per document read is the order of magnitude of the filings the
        # token model was measured on; it is a stand-in, not an observation.
        context_bytes=2000 * (counts["review"] + counts["extract"]
                              + counts["reconcile"] + counts["retrieve"]),
        monthly_runs=monthly,
        encoder="heuristic",
        rationale=("keyword match; no model used"
                   + (f"; read {throughput:,} as throughput -> {monthly:,} "
                      f"runs/month, one item per run"
                      if throughput and quantities
                      and quantities[0] == throughput
                      else f"; scale read from \"{quantities[0]:,}\" in the text"
                      if quantities else "; no quantity found, assumed 1")),
    )
