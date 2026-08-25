"""Pricing a task nobody has ever run, by writing it in terms of ones we have.

:mod:`token_yield.compose` can price any *combination* of base tasks. That is
only useful if an arbitrary request can be turned into such a combination. This
module does that turning.

The shape is an autoencoder over tasks:

    request (free text)  --encode-->  primitive counts  --decode-->  tokens

**Encode.** Read the request and express it as a multiset of base tasks:
"pull the segment figures out of these three filings, check them against the
prior quarter, and write me a board note" becomes
``3xExtract + Reconcile + Report``.

**Decode.** Recompose those parts through the fitted cost model.

The encoder is a judgement call, so it is made by an agent rather than a regex,
and the agent is asked to show its reasoning. A keyword-based fallback is
provided so the library still works with no agent available — it is weaker, and
:func:`heuristic_decompose` says so.

Reconstruction error
--------------------
An autoencoder is judged by how well the round trip preserves the original. Here
the original is *what the task actually cost*, so the round trip can be checked
the moment the task is finally run: :func:`reconstruction_error` compares the
recomposed prediction against the measured truth. That number, not the
plausibility of the decomposition, is what says whether the vocabulary is
adequate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .compose import CompositionModel
from .tasks import ORDER, PRIMITIVES


# ── the encoded form ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Decomposition:
    """A request expressed in base tasks."""

    counts: Dict[str, int]
    rationale: str = ""
    context_bytes: int = 0
    source: str = "agent"

    @property
    def parts(self) -> List[Tuple[str, int]]:
        return [(p, self.counts[p]) for p in ORDER if self.counts.get(p)]

    def notation(self) -> str:
        if not self.parts:
            return "(nothing)"
        return " + ".join(f"{n}x{PRIMITIVES[p].name}" if n > 1
                          else PRIMITIVES[p].name for p, n in self.parts)

    def is_empty(self) -> bool:
        return not self.parts


# ── encode: free text -> base tasks ──────────────────────────────────────

def decompose_prompt(request: str) -> str:
    """The prompt that asks an agent to express a request in base tasks.

    The vocabulary is supplied in full, with the industry each primitive comes
    from, so the agent is choosing among defined terms rather than inventing a
    scheme of its own.
    """
    menu = "\n".join(
        f"- {p.name} ({p.slug}) — {p.blurb} Typical use: {p.industry}."
        for p in (PRIMITIVES[s] for s in ORDER))
    return (
        "You are decomposing a work request into a fixed vocabulary of base "
        "tasks, so that its cost can be estimated from measurements of those "
        "base tasks.\n\n"
        f"THE BASE TASKS\n{menu}\n\n"
        f"THE REQUEST\n{request}\n\n"
        "Express the request as counts of base tasks. One unit means one "
        "document reviewed, one field extracted, one item classified, one fact "
        "retrieved, one comparison, one piece drafted, one error corrected, one "
        "check written, or one aspect reported. Count only work the request "
        "actually asks for.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{\"counts\": {\"<slug>\": <int>, ...}, \"rationale\": \"<one sentence>\"}'
    )


def parse_decomposition(text: str, context_bytes: int = 0) -> Decomposition:
    """Read an agent's reply into a :class:`Decomposition`.

    Tolerates the usual wrappers — fenced code blocks, a stray sentence before
    the JSON — because the alternative is failing on formatting rather than on
    substance. Unknown slugs are dropped: the vocabulary is fixed, and silently
    inventing a primitive would corrupt the model's feature vector.
    """
    blob = re.search(r"\{.*\}", text, re.S)
    if not blob:
        raise ValueError("no JSON object in decomposition reply")
    data = json.loads(blob.group(0))
    raw = data.get("counts", {}) or {}
    counts = {s: 0 for s in ORDER}
    for key, value in raw.items():
        slug = str(key).strip().lower()
        if slug in counts:
            try:
                counts[slug] += max(0, int(value))
            except (TypeError, ValueError):
                continue
    return Decomposition(counts, str(data.get("rationale", "")).strip(),
                         context_bytes, source="agent")


_KEYWORDS = {
    "review": ("read", "review", "look at", "go through", "summarise",
               "summarize", "understand"),
    "extract": ("extract", "pull", "capture", "populate", "field", "into a "
                "table", "structured"),
    "classify": ("classify", "categorise", "categorize", "triage", "route",
                 "tag", "sort"),
    "retrieve": ("find", "locate", "search", "where is", "which document",
                 "look up"),
    "reconcile": ("reconcile", "compare", "cross-check", "tie out", "against "
                  "the", "discrepan", "disagree"),
    "draft": ("draft", "write a", "compose", "prepare a", "produce a note"),
    "remediate": ("fix", "correct", "remediate", "resolve", "clean up"),
    "validate": ("validate", "check", "verify", "control", "test", "audit"),
    "report": ("report", "board note", "summary for", "write up", "memo"),
}


def heuristic_decompose(request: str, context_bytes: int = 0) -> Decomposition:
    """Keyword fallback for when no agent is available.

    Deliberately crude: it sees words, not intent, and cannot count units. Use
    it to keep a pipeline running, not to make a decision — the ``source``
    field records which encoder produced any given estimate.
    """
    low = request.lower()
    counts = {s: 0 for s in ORDER}
    for slug, words in _KEYWORDS.items():
        if any(w in low for w in words):
            counts[slug] = 1
    return Decomposition(counts, "keyword match; no agent used", context_bytes,
                         source="heuristic")


# ── decode: base tasks -> tokens ─────────────────────────────────────────

def price(decomp: Decomposition, model: CompositionModel) -> float:
    """Recompose a decomposition through the fitted model."""
    return model.predict(decomp.counts, decomp.context_bytes)


def reconstruction_error(decomp: Decomposition, model: CompositionModel,
                         actual_tokens: int) -> float:
    """How far the round trip landed from the truth, once the task is run."""
    return abs(price(decomp, model) - actual_tokens) / actual_tokens


def explain(decomp: Decomposition, model: CompositionModel) -> str:
    """A line-by-line account of where a predicted number comes from.

    A forecast a buyer cannot interrogate is a guess with a decimal point, so
    every term is shown: the fixed cost of starting an agent, what the context
    adds, and what each base task contributes.
    """
    lines = [f"Request decomposes to: {decomp.notation()}",
             f"  (encoder: {decomp.source})"]
    if decomp.rationale:
        lines.append(f"  {decomp.rationale}")
    lines.append("")
    lines.append(f"  {'agent start-up (paid once)':<34}{model.coef[0]:>10,.0f}")
    if model.byte_slope():
        ctx = model.byte_slope() * decomp.context_bytes
        lines.append(f"  {'context: %s bytes' % f'{decomp.context_bytes:,}':<34}"
                     f"{ctx:>10,.0f}")
    marg = model.marginals()
    for slug, n in decomp.parts:
        add = marg.get(slug, 0.0) * n
        lines.append(f"  {f'{n}x {PRIMITIVES[slug].name}':<34}{add:>10,.0f}")
    lines.append(f"  {'':-<34}{'':->10}")
    lines.append(f"  {'predicted tokens':<34}{price(decomp, model):>10,.0f}")
    return "\n".join(lines)


def batching_advice(decomp: Decomposition, model: CompositionModel) -> str:
    """What it would cost to split this work across separate agents instead."""
    from .compose import batching_saving
    batched, separate, saving = batching_saving(
        model, decomp.counts, decomp.context_bytes)
    return (f"  one agent, all parts : {batched:>10,.0f}\n"
            f"  one agent per part   : {separate:>10,.0f}\n"
            f"  saving from batching : {saving:>10.1%}")
