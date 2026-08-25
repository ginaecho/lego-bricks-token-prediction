"""The base task primitives, and the algebra for building larger tasks from them.

Everything Token Yield can price is expressed as a combination of a small set of
**named base tasks**. This module defines that set and the rules for combining
them; :mod:`token_yield.compose` fits what the combinations cost.

Why a fixed set at all
----------------------
A cost model needs a unit of work. "One task" is not a unit — a task can be
anything. What *is* reasonably stable across a business is the small vocabulary
of things an agent actually does to a body of documents: read them, pull fields
out of them, sort them, find things in them, check them against each other,
write new material, correct what is wrong, test that it holds, and report on it.
Those are the primitives here.

They are not invented. Each one names a task type that enterprises already buy
agents to do — invoice and claims intake, ticket triage, knowledge discovery,
financial close and audit, drafting, exception handling, control testing,
management reporting — and each maps onto the maintenance taxonomy that the
software engineering literature has used since Swanson (1976): corrective,
adaptive and perfective work.

Composition
-----------
A real request is rarely one primitive. "Pull the segment figures out of these
filings and tell me where they disagree" is ``Extract + Reconcile``; "read this
and write me the board note" is ``Review + Report``. A :class:`TaskSpec` is
exactly that: a multiset of primitives, each with a count, evaluated against a
concrete document context.

The point of naming them is that composition becomes arithmetic. If a new task
can be written as ``2xReview + 1xReconcile``, and each of those has been
measured, the new task can be priced without ever having been run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


# ── the base task set ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Primitive:
    """One base task type: a named unit of agent work.

    ``industry`` records where the task type is actually bought, so the
    vocabulary stays anchored to real work rather than to this framework.

    ``driver`` records what we *expect* to move its cost — how much is read, how
    much is produced, or neither. It is a hypothesis, not a setting: the fitted
    model in :mod:`token_yield.compose` selects the signal from data and is free
    to disagree.
    """

    name: str
    slug: str
    category: str
    driver: str
    industry: str
    blurb: str


PRIMITIVES: Dict[str, Primitive] = {
    p.slug: p for p in (
        Primitive(
            "Review", "review", "cross-cutting", "input_bytes",
            "contract and compliance review",
            "Read source material and state what it covers. Pure intake: "
            "nothing is produced but an assessment.",
        ),
        Primitive(
            "Extract", "extract", "cross-cutting", "output_units",
            "invoice and claims intake, KYC",
            "Pull named fields out of an unstructured document into a "
            "structured record. The workhorse of document automation.",
        ),
        Primitive(
            "Classify", "classify", "cross-cutting", "output_units",
            "ticket and email triage, routing",
            "Sort items into categories. Cheap per item, and usually the first "
            "step of any queue an agent is put in front of.",
        ),
        Primitive(
            "Retrieve", "retrieve", "cross-cutting", "corpus_bytes",
            "knowledge discovery, e-discovery",
            "Find where something is stated across a body of documents. Cost "
            "is driven by searching, not by any one document.",
        ),
        Primitive(
            "Reconcile", "reconcile", "corrective", "input_bytes",
            "financial close, audit, dispute resolution",
            "Compare two sources and report where they disagree. Reads twice "
            "and answers once.",
        ),
        Primitive(
            "Draft", "draft", "adaptive", "output_units",
            "proposals, memos, marketing copy",
            "Produce new material to a specification. Pure output: no source "
            "document needs to be read.",
        ),
        Primitive(
            "Remediate", "remediate", "corrective", "mixed",
            "exception handling, error correction",
            "Find what is wrong in existing material and correct it. Reads to "
            "diagnose, then writes — the only primitive that does both.",
        ),
        Primitive(
            "Validate", "validate", "preventive", "output_units",
            "control testing, quality assurance",
            "Write the checks that would confirm a document holds up. Reads a "
            "fixed source, produces one check per unit.",
        ),
        Primitive(
            "Report", "report", "perfective", "output_units",
            "management and board reporting",
            "Turn source material into a written reference. Same shape as "
            "Validate, different product.",
        ),
    )
}

#: Stable ordering for tables, figures and feature vectors.
ORDER: Tuple[str, ...] = ("review", "extract", "classify", "retrieve",
                          "reconcile", "draft", "remediate", "validate",
                          "report")


# ── material for the source-free primitives ──────────────────────────────
#
# Draft and Remediate are self-contained by design: they need no corpus, so
# their material is fixed here. The other seven are corpus-specific and take
# their targets from the TaskSpec. Keeping the two apart is what stops a
# composite task from handing one primitive another primitive's material.

DRAFT_SPECS: Tuple[str, ...] = (
    "A two-sentence investor summary of a company whose quarterly revenue grew "
    "1% while segment margin fell 6 points.",
    "A two-sentence risk note on supplier concentration in a single "
    "manufacturing region.",
    "A two-sentence collections email for an invoice 45 days past due.",
    "A two-sentence internal escalation note for a reconciliation break of "
    "$1.2m found during month-end close.",
)

REMEDIATE_ITEMS: Tuple[str, ...] = (
    '"Revenue rose from $585.5m to $622.9m, an increase of 12%." '
    "(check the arithmetic)",
    '"Operating income grew in all three segments year over year." '
    "(check against: Americas 81.6 -> 63.6, EMEA 69.0 -> 58.2, APAC 43.6 -> 44.5)",
    '"The company reports four operating segments." '
    "(check against: Americas, EMEA, Asia Pacific)",
    '"Total assets increased between August 2025 and May 2026." '
    "(check against: 4,304,272 -> 4,191,997)",
)


# ── instructions per primitive ───────────────────────────────────────────

def _files_block(paths: Sequence[str]) -> str:
    return "\n   ".join(paths)


def _instruction(slug: str, units: int, ctx: Sequence[str],
                 targets: Sequence[str], corpus: str) -> str:
    """The imperative text for one primitive at a given size.

    ``ctx`` are documents to read; ``corpus`` is a directory to search. They are
    kept apart because Retrieve needs a whole corpus while Extract and Validate
    need one named document.
    """
    if slug == "review":
        return (f"Read these {len(ctx)} document(s):\n   {_files_block(ctx)}\n"
                f"Then state, in at most 3 sentences, what this material "
                f"covers.")
    if slug == "extract":
        return (f"Read this document:\n   {ctx[0]}\n"
                f"Then extract these {units} field(s) as a JSON object: "
                f"{', '.join(targets[:units])}.")
    if slug == "classify":
        return (f"Classify each of these {len(ctx)} document(s) by industry "
                f"sector, and by whether the tone is positive, negative or "
                f"neutral:\n   {_files_block(ctx)}\n"
                f"Answer with one line per document.")
    if slug == "retrieve":
        return (f"Search the document corpus at {corpus} to find where these "
                f"{units} fact(s) are stated: {', '.join(targets[:units])}.\n"
                f"Report the file name for each. Do not read whole documents "
                f"you do not need.")
    if slug == "reconcile":
        return (f"Compare these {len(ctx)} document(s):\n   {_files_block(ctx)}\n"
                f"List every figure that appears in both and disagrees, and "
                f"every figure present in one but missing from the other.")
    if slug == "draft":
        listed = "\n".join(f"   {i + 1}. {t}"
                           for i, t in enumerate(DRAFT_SPECS[:units]))
        return (f"Write {units} short business item(s). Do not read any "
                f"document.\n{listed}")
    if slug == "remediate":
        listed = "\n".join(f"   {i + 1}. {t}"
                           for i, t in enumerate(REMEDIATE_ITEMS[:units]))
        return (f"Each statement below contains one error. For each, name the "
                f"error and give the corrected statement.\n{listed}")
    if slug == "validate":
        return (f"Read this document:\n   {ctx[0]}\n"
                f"Then write {units} check(s) an auditor could run to confirm "
                f"the figures in it are internally consistent.")
    if slug == "report":
        return (f"Read this document:\n   {ctx[0]}\n"
                f"Then write a reference summary (Markdown) covering these "
                f"{units} aspect(s): {', '.join(targets[:units])}.")
    raise KeyError(slug)


# ── a task: one or more primitives, run in a single agent invocation ─────

@dataclass(frozen=True)
class TaskSpec:
    """A unit of work to dispatch: a multiset of primitives over a context.

    ``parts`` is the composition — ``(("review", 2), ("reconcile", 1))`` means
    two review units and one reconcile unit, done by **one** agent in one
    invocation. That single-invocation detail is the whole reason composition is
    not additive: the fixed cost of starting an agent is paid once no matter how
    many parts there are.
    """

    label: str
    parts: Tuple[Tuple[str, int], ...]
    context: Tuple[str, ...] = ()
    #: ``((slug, (target, ...)), ...)`` — each primitive gets its own material.
    #: A bare tuple of strings is accepted and applies to every primitive that
    #: asks for targets, which is what a single-primitive task wants.
    targets: Tuple = ()
    corpus: str = ""
    held_out: bool = False

    def targets_for(self, slug: str) -> Tuple[str, ...]:
        """The material for one primitive, never another primitive's."""
        if self.targets and isinstance(self.targets[0], tuple):
            for key, values in self.targets:
                if key == slug:
                    return values
            return ()
        return self.targets

    @property
    def slugs(self) -> Tuple[str, ...]:
        return tuple(s for s, _ in self.parts)

    @property
    def total_units(self) -> int:
        return sum(n for _, n in self.parts)

    @property
    def arity(self) -> int:
        """How many distinct primitives this task mixes."""
        return len(set(self.slugs))

    def counts(self) -> Dict[str, int]:
        """Primitive -> units, the feature vector the cost model is fitted on."""
        out = {s: 0 for s in ORDER}
        for slug, n in self.parts:
            out[slug] += n
        return out

    def context_bytes(self) -> int:
        """Total size of the documents this task is pointed at."""
        return sum(os.path.getsize(p) for p in self.context
                   if os.path.isfile(p))

    def notation(self) -> str:
        """Human-readable composition, e.g. ``2xReview + Reconcile``."""
        bits = []
        for slug, n in self.parts:
            nm = PRIMITIVES[slug].name
            bits.append(f"{n}x{nm}" if n > 1 else nm)
        return " + ".join(bits)

    def prompt(self) -> str:
        """The exact text dispatched to a fresh agent."""
        if not self.parts:
            # The null probe: no work at all, so what it costs is what an agent
            # costs before any work — the constant every other task also pays.
            return ("MEASUREMENT PROBE — null.\n\n"
                    "Do exactly this, nothing more. Do not read any files. "
                    "Do not run any tools.\n\nReply with exactly the word: DONE")
        steps = [_instruction(slug, n, self.context, self.targets_for(slug),
                              self.corpus)
                 for slug, n in self.parts]
        numbered = "\n\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        return (f"MEASUREMENT PROBE — {self.notation()}.\n\n"
                f"Do exactly what is listed below, nothing more. Do not modify "
                f"any file.\n\n{numbered}\n\nStop when the last item is done.")
