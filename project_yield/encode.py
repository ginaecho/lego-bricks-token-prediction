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
from typing import Dict, Optional, Tuple

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

#: Deliberately high precision and low recall. Every loose term tried here —
#: "claims", "statutory", "returns", "consultation" — produced a *confident
#: wrong* industry on some use case, and a confident wrong answer is worse than
#: a flagged default: the default is recorded as an assumption and shown, the
#: wrong match is silent. So only unmistakable sector markers are listed, and
#: anything else falls through to the default and says so.
_INDUSTRY_WORDS: Dict[str, Tuple[str, ...]] = {
    "financial_services": ("bank", "insurer", "insurance", "financial services",
                           "kyc", "aml", "underwrit", "policyholder",
                           "mortgage", "credit committee"),
    "healthcare": ("health", "clinic", "patient", "hospital", "medical",
                   "pharma"),
    "manufacturing": ("manufactur", "factory", "plant", "production line",
                      "bill of materials", "supplier quality", "warranty claim",
                      "non-conformance"),
    "retail": ("retail", "merchand", "ecommerce", "e-commerce", "shopper",
               "sku", "storefront"),
    "public_sector": ("government", "public sector", "council", "ministry",
                      "the agency", "citizen", "municipal", "federal",
                      "the department", "freedom of information"),
    "energy": ("energy", "utility", "the grid", "oil", "gas", "renewable",
               "drilling", "outage"),
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

#: Numbers written as words. Scoping notes are prose, and prose spells out
#: small numbers — "extract nine header fields", "the twelve checks the
#: checklist implies". An encoder that only sees digits reads every one of
#: those as a single unit, which flattens every use case to the same size.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

_NUM = r"(?:\d[\d,]*|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"

#: The countable things a use case is scoped in. A quantity only counts when it
#: is attached to one of these, so "48 hours" and "three years" are not scope.
_UNIT_NOUN = (
    r"(?:documents?|files?|invoices?|claims?|records?|contracts?|tickets?|"
    r"cases?|filings?|reports?|fields?|emails?|messages?|pages?|checks?|"
    r"questions?|sections?|stages?|types?|covenants?|criteri(?:on|a)|"
    r"points?|responses?|summar(?:y|ies)|memos?|items?|attributes?|"
    r"categor(?:y|ies)|clauses?|episodes?|applications?|skus?|tenders?|"
    r"bullets?|entries?|customers?|suppliers?|submissions?|packs?)"
)

#: A quantity attached to a countable noun: "12 invoices", "nine header fields".
#: Matched against :func:`_scannable`, which has already folded the note onto
#: one line, so the separators are spaces and tabs by construction.
_QUANTITY = re.compile(r"\b" + _NUM + r"[ \t]+(?:[\w-]+[ \t-]+){0,3}?"
                       + _UNIT_NOUN + r"\b", re.I)

#: A production rate. Digits only, because "two senior people about a week" is
#: a sentence about effort, not a throughput — and the digit requirement is what
#: keeps the two apart without parsing English.
_RUN_RATE = re.compile(r"(\d[\d,]*)\s*(?:[\w-]+\s+){0,3}?"
                       r"(?:per|a|an|each|/)\s*(day|week|month|year|quarter)",
                       re.I)

#: "runs four times a year" — unambiguous enough to allow a spelled-out number.
_TIMES_RATE = re.compile(_NUM + r"\s+times\s+(?:per|a|an|each)\s*"
                         r"(day|week|month|year|quarter)", re.I)

_PER_MONTH = {"day": 30.0, "week": 4.33, "month": 1.0, "quarter": 1.0 / 3.0,
              "year": 1.0 / 12.0}

#: How far a quantity may sit from a brick keyword and still be read as its
#: scope. Wide enough for "extract the nine header and line fields", narrow
#: enough that a number in the next sentence belongs to the next sentence.
_ATTACH_WINDOW = 90

#: Nothing in one run of a pipeline is bigger than this. A four-digit count
#: attached to a brick is almost always a volume the rate patterns missed.
_MAX_UNITS = 500


#: Prefixes on the assumptions the encoder records, so a caller can tell which
#: field an assumption is about without parsing the sentence. They are prefixes
#: rather than a structured field because the assumptions are shown to people
#: verbatim, and a sentence that reads well is worth more here than a schema.
ASSUMED_INDUSTRY = "No industry is named"
ASSUMED_GOAL = "No business goal is stated"
ASSUMED_SCOPE = "No countable scope was found"


def _as_number(text: str) -> int:
    text = text.strip().lower().replace(",", "")
    if text.isdigit():
        return int(text)
    return _NUMBER_WORDS.get(text, 1)


def _leading_number(match_text: str) -> int:
    head = re.match(_NUM, match_text, re.I)
    return _as_number(head.group(0)) if head else 1


def _score_vocab(text: str, words: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    """The vocabulary level with the most keyword hits, or ``None`` for a tie
    at zero — in which case the caller falls back to the reference category."""
    hits = {key: sum(text.count(w) for w in ws) for key, ws in words.items()}
    best = max(hits, key=lambda k: hits[k])
    return best if hits[best] else None


def heuristic_encode(description: str, uid: str = "new",
                     title: str = "") -> UseCase:
    """Keyword fallback. Weaker by construction, and it says so.

    Two things it does that the token model's fallback cannot, both of which
    are the difference between a useful estimate and a meaningless one:

    *Quantities attach to the nearest brick.* "read the invoice, extract nine
    header fields, run three checks" is 1 Review, 9 Extract, 3 Validate — not
    nine of everything, and not one of everything. A single global count read
    off the first number in the text made every use case the same size.

    *Throughput is volume, not scope.* "400 claims a day" is 12,000 runs a
    month of a one-claim pipeline, not a 400-claim pipeline. Reading it as both
    would multiply the scope by the volume, which is the most expensive mistake
    this encoder could make.

    It still sees words rather than intent, cannot tell a hard requirement from
    an aside, and guesses context size from a constant. Use it to keep a
    pipeline running, not to make a decision — ``UseCase.encoder`` records
    which encoder produced any given estimate, and every surface prints it.
    """
    low = _scannable(description)

    # -- production volume, and the spans that carry it ------------------
    monthly, rate_spans, throughput = 0, [], 0
    for pattern in (_RUN_RATE, _TIMES_RATE):
        for m in pattern.finditer(low):
            # The period is the last group in both patterns; the number is read
            # off the front of the match, so a spelled-out one still parses.
            period = m.groups()[-1].lower()
            value = _leading_number(m.group(0))
            rate_spans.append((m.start(), m.end()))
            per_month = int(value * _PER_MONTH[period])
            if per_month > monthly:
                monthly, throughput = per_month, value

    # -- quantities that are scope rather than throughput ----------------
    quantities, spans = [], []
    for m in _QUANTITY.finditer(low):
        if any(lo <= m.start() < hi for lo, hi in rate_spans):
            continue
        quantities.append((m.start(), min(_leading_number(m.group(0)),
                                          _MAX_UNITS)))
        spans.append((m.start(), m.end()))

    # -- where each brick is mentioned -----------------------------------
    # Every occurrence, not just the first: a note that says "extract" twice is
    # describing two different things and the second one may carry the count.
    # Each hit records which quantity phrase it sits inside, if any. A keyword
    # inside a phrase is that phrase's own noun — the "checks" in "three
    # validation checks" is what makes it Validate — so it identifies the brick
    # for *that* quantity and is invisible to every other one. Without the
    # second half, one phrase's noun reaches over and claims the next
    # sentence's number.
    hits = []
    for slug, words in _BRICK_WORDS.items():
        for word in words:
            for m in re.finditer(re.escape(word), low):
                inside = next((i for i, (lo, hi) in enumerate(spans)
                               if lo <= m.start() < hi), None)
                hits.append((m.start(), slug, inside))
    mentioned = {slug for _, slug, _ in hits}

    counts = {slug: 0 for slug in BRICKS}
    # Each quantity is claimed by the single nearest eligible mention, rather
    # than every brick taking the first number it can see. One number, one
    # brick. Ties break toward the keyword on the left, because English puts
    # the verb before its object: "extract nine fields", "write three checks".
    for index, (pos, value) in enumerate(quantities):
        near = [(abs(pos - hp), hp > pos, slug)
                for hp, slug, inside in hits
                if inside in (None, index) and abs(pos - hp) <= _ATTACH_WINDOW]
        if near:
            slug = min(near)[2]
            counts[slug] = max(counts[slug], value)
    for slug in mentioned:
        counts[slug] = counts[slug] or 1
    if not sum(counts.values()):
        counts["review"] = 1

    reading = []
    if throughput:
        reading.append(f"read {throughput:,} as throughput -> {monthly:,} "
                       f"runs/month")
    sized = [f"{n}x{s}" for s, n in counts.items() if n > 1]
    reading.append(", ".join(sized) if sized
                   else "no scope quantities found, assumed one of each")

    assumptions = []
    industry = _score_vocab(low, _INDUSTRY_WORDS)
    if industry is None:
        industry = INDUSTRIES[0]
        assumptions.append(
            f"{ASSUMED_INDUSTRY} in the description, so it was set to "
            f"{industry.replace('_', ' ')}. Industry is a feature in every "
            f"head — set it by hand if it is wrong.")
    goal = _score_vocab(low, _GOAL_WORDS)
    if goal is None:
        goal = GOALS[0]
        assumptions.append(
            f"{ASSUMED_GOAL}, so it was set to {goal.replace('_', ' ')}.")
    if not quantities:
        assumptions.append(
            f"{ASSUMED_SCOPE} in the text, so every task named was counted "
            f"once. The scope is a floor, not an estimate.")

    return UseCase(
        id=uid, title=title or _derive_title(description),
        description=description,
        industry=industry,
        goal=goal,
        counts=counts,
        # 2 kB per document read is the order of magnitude of the filings the
        # token model was measured on; it is a stand-in, not an observation.
        context_bytes=2000 * (counts["review"] + counts["extract"]
                              + counts["reconcile"] + counts["retrieve"]),
        monthly_runs=monthly,
        encoder="heuristic",
        rationale="keyword match; no model used; " + "; ".join(reading),
        assumptions=assumptions,
    )


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")


def _scannable(description: str) -> str:
    """The text the encoder actually reads: heading dropped, lines folded.

    Two small things, both of which change the answer:

    *The heading is a title, not scope.* A note headed ``# One Northwind
    pipeline`` above a body of ``Read 4 contracts`` was read as one contract,
    because the heading's "One" is a number attached to the body's first noun.
    A title is a label; it is not describing work.

    *Wrapped prose is still one sentence.* Scoping notes arrive hard-wrapped at
    eighty columns, so "extract nine header and line fields" routinely straddles
    a line break. Folding the lines is what stops the wrap width of somebody's
    editor from changing the estimate.
    """
    lines = description.strip().split("\n")
    if lines and _HEADING.match(lines[0]):
        lines = lines[1:]
    return " ".join(" ".join(lines).split()).lower()


def _derive_title(description: str) -> str:
    """First line, minus any Markdown heading marks."""
    first = description.strip().split("\n")[0].lstrip("#").strip()
    return first[:70] or "Untitled use case"
