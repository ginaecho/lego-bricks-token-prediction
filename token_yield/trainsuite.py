"""The measurement campaign: what gets run, at what size, and what is held back.

This is the experiment design for the compositional model. It produces
:class:`~token_yield.tasks.TaskSpec` objects in four tiers:

**Null.** No work at all. What it costs is what an agent costs before any work —
the constant every other task also pays, and the single most important number in
the model.

**Base.** Each primitive alone, at more than one size. For ``Review`` the size
knob is the *context* — the same instruction pointed at 0.9 KB or at 30 KB of
real filings — which is how the model learns what adding and removing context
costs. For the producing primitives the knob is how many units are asked for.

**Composite.** Several primitives in one invocation: ``A+A``, ``A+B``,
``A+B+C``. These are what the compositional model is fitted against.

**Held out.** Compositions deliberately excluded from fitting, including one
arity the fit never sees (a four-way mix). They exist to answer the only
question that matters: can a task be priced *before* it is ever run?

The corpus is real. Every document is an SEC filing or an earnings-call
transcript retrieved from Bigdata.com (https://bigdata.com), spanning
pharmaceuticals, retail, semiconductors, consumer goods, industrials and
fintech. Nothing here hardcodes a path: the suite is built against whatever
corpus directory you pass in.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .tasks import TaskSpec

# Fields a document-automation agent is actually asked to pull.
EXTRACT_FIELDS = ("company name", "reporting period", "total revenue",
                  "operating income", "number of operating segments",
                  "total assets")

# Facts to go looking for across the corpus.
RETRIEVE_FACTS = ("which company reported a 26% decline in manufacturing revenue",
                  "which company reported three geographic operating segments",
                  "which company discussed supplier concentration risk")

# Aspects a management report is asked to cover.
REPORT_ASPECTS = ("revenue performance", "segment profitability",
                  "cost drivers", "outlook risks")


def _docs(corpus: str):
    """Every document in the corpus, largest first, so size rungs are stable."""
    names = sorted((n for n in os.listdir(corpus) if n.endswith(".md")),
                   key=lambda n: (-os.path.getsize(os.path.join(corpus, n)), n))
    return [os.path.join(corpus, n) for n in names]


def _figures_anchor(paths):
    """The most figure-dense document in the corpus.

    Extract, Validate and Report are only meaningful against a document that
    actually carries numbers: pointed at narrative risk-factor prose they
    correctly answer "there are no figures here", which measures the wrong
    thing. Picking by digit density keeps the probe honest on any corpus
    instead of hardcoding a filename.
    """
    def density(p):
        text = open(p, encoding="utf-8", errors="replace").read()
        return sum(c.isdigit() for c in text) / max(len(text), 1)
    return max(paths, key=density)


def build_suite(corpus: str) -> List[TaskSpec]:
    """Build the full campaign against a real document corpus."""
    d = _docs(corpus)
    if len(d) < 12:
        raise ValueError(f"corpus at {corpus} is too small to build the suite")

    # Context ladder: the same Review instruction over growing real filings.
    tiny = (d[-1],)                 # smallest single document
    small = (d[2],)                 # a mid-size filing
    medium = (d[0], d[1])           # the two largest
    large = tuple(d[:6])            # a six-document pack
    xlarge = tuple(d[:14])          # a due-diligence pack

    one = (_figures_anchor(d),)     # the most figure-dense document
    pair = (one[0], d[5])           # two documents to reconcile

    S = TaskSpec
    suite: List[TaskSpec] = []

    # ── tier 0: the null task ────────────────────────────────────────────
    suite += [S("null", (), (), ()), S("null_r2", (), (), ())]

    # ── tier 1: base primitives ──────────────────────────────────────────
    # Review: identical ask, five context sizes (the ablation).
    suite += [
        S("review_tiny", (("review", 1),), tiny),
        S("review_small", (("review", 1),), small),
        S("review_medium", (("review", 2),), medium),
        S("review_large", (("review", 6),), large),
        S("review_xlarge", (("review", 14),), xlarge),
        S("review_small_r2", (("review", 1),), small),
    ]
    # Unit ladders for the producing primitives.
    for n in (1, 3):
        suite += [
            S(f"extract_{n}", (("extract", n),), one, EXTRACT_FIELDS),
            S(f"draft_{n}", (("draft", n),), ()),
            S(f"remediate_{n}", (("remediate", n),), ()),
            S(f"validate_{n}", (("validate", n),), one),
            S(f"report_{n}", (("report", n),), one, REPORT_ASPECTS),
        ]
    suite += [S("draft_3_r2", (("draft", 3),), ())]
    # Classify scales with how many documents are in the queue.
    suite += [
        S("classify_2", (("classify", 2),), medium),
        S("classify_6", (("classify", 6),), large),
    ]
    # Retrieve searches the corpus rather than any one document.
    suite += [
        S("retrieve_1", (("retrieve", 1),), (), RETRIEVE_FACTS, corpus),
        S("retrieve_3", (("retrieve", 3),), (), RETRIEVE_FACTS, corpus),
    ]
    # Reconcile reads two documents and answers once.
    suite += [S("reconcile_2", (("reconcile", 1),), pair)]

    # ── tier 2: composites, fitted ───────────────────────────────────────
    suite += [
        S("rr", (("review", 1), ("review", 1)), medium),
        S("ee", (("extract", 2),), one, EXTRACT_FIELDS),
        S("rev_ext", (("review", 1), ("extract", 3)), one, EXTRACT_FIELDS),
        S("rev_dra", (("review", 1), ("draft", 2)), small),
        S("ext_rec", (("extract", 3), ("reconcile", 1)), pair, EXTRACT_FIELDS),
        S("rem_val", (("remediate", 2), ("validate", 2)), one),
        S("dra_val", (("draft", 2), ("validate", 2)), one),
        S("rev_rep", (("review", 1), ("report", 2)), one, REPORT_ASPECTS),
        S("ret_rem", (("retrieve", 1), ("remediate", 2)), (), RETRIEVE_FACTS,
          corpus),
        S("rev_rem_val", (("review", 1), ("remediate", 2), ("validate", 2)),
          one),
        S("ext_cla_rep", (("extract", 2), ("classify", 2), ("report", 1)),
          medium, (("extract", EXTRACT_FIELDS), ("report", REPORT_ASPECTS))),
    ]

    # ── tier 3: held out of the fit entirely ─────────────────────────────
    suite += [
        S("H_rev_val", (("review", 1), ("validate", 2)), one, held_out=True),
        S("H_dra_rep", (("draft", 2), ("report", 1)), one, REPORT_ASPECTS,
          held_out=True),
        S("H_rev6_dra", (("review", 6), ("draft", 1)), large, held_out=True),
        # A four-way mix: an arity the fitted model never saw.
        S("H_quad", (("retrieve", 1), ("review", 1), ("remediate", 1),
                     ("validate", 1)), one, RETRIEVE_FACTS, corpus,
          held_out=True),
    ]
    return suite
