"""The keyword encoder: a deterministic control, and possibly the shipped product.

This exists to answer a question the reference project never asked: *does the LLM encoder
beat a regex?* If it does not -- if a few dozen ordered patterns match the model's accuracy
within confidence intervals -- then the right thing to ship is this, because it is free,
instant, deterministic and cannot hallucinate.

It is written from the vocabulary card **before** validation gold is opened, and
:data:`RULES_SHA256` is asserted in CI so it cannot be quietly tuned afterwards. A control
that gets adjusted until it agrees is not a control.

It is never a fallback. When the LLM encoder fails to parse, that is recorded as a
failure; substituting this one would hide the failure rate that the encoder gate measures.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from assay.decompose.parser import Decomposition
from assay.vocabulary import Vocabulary

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Nouns that can actually be counted as brick units. Restricting counting to this list is
# what stops "the 26% decline" being read as twenty-six of something. Note what is *not*
# here: documents and filings. "Compare these two filings" is one comparison over two
# sources, not two comparisons -- the counting rule is requested targets, not material.
COUNTABLE = (
    r"fields?|values?|checks?|questions?|items?|comparisons?|targets?|facts?|"
    r"reports?|summaries|summary|drafts?|notes?|corrections?|fixes|categories|"
    r"sections?|assertions?|rules?|figures?|metrics?"
)

COUNT_RE = re.compile(
    rf"\b(?P<n>\d+|{'|'.join(NUMBER_WORDS)})\s+(?:\w+\s+){{0,2}}(?P<noun>{COUNTABLE})\b",
    re.IGNORECASE,
)

NEGATION_RE = re.compile(r"\b(do not|don't|without|no need to|skip|ignore)\b", re.IGNORECASE)

# Ordered. First match wins for a clause, so the earlier entries settle the confusions the
# vocabulary card calls out: "write two checks" is Validate before it is Draft, and
# "find which filing" is Retrieve before it is Extract.
#
# Stems carry an explicit `\w*` rather than relying on a trailing `\b`. A pattern like
# `\bclassif\b` matches nothing at all -- "classify" continues with a word character, so
# the boundary never fires -- and the rule would silently never trigger.
RULES: tuple[tuple[str, str], ...] = (
    ("Reconcile", r"\b(?:reconcil\w*|compar\w*|discrepanc\w*|versus|vs\.?|differences? between|agree with)\b"),
    ("Validate", r"\b(?:validat\w*|verif\w*|check\w*|confirm\w*|assert\w*|audit|test that|make sure|tie out)\b"),
    ("Classify", r"\b(?:classif\w*|categor\w*|label\w*|triage|rout(?:e|es|ing)|bucket\w*|tag each)\b"),
    ("Remediate", r"\b(?:remediat\w*|fix\w*|correct\w*|repair\w*|amend\w*|clean up)\b"),
    ("Retrieve", r"\b(?:retriev\w*|search\w*|locat\w*|look up|find\w*|which (?:document|filing|file|company|report)|where (?:is|does))\b"),
    ("Report", r"\b(?:report\w*|summar\w*|board note|management note)\b"),
    ("Extract", r"\b(?:extract\w*|pull\w*|list the|get the|captur\w*|populat\w*|fields?|values? of)\b"),
    ("Draft", r"\b(?:draft\w*|writ\w*|compos\w*|author)\b"),
    ("Review", r"\b(?:review\w*|read\w*|assess\w*|judg\w*|consider\w*|opinion on)\b"),
)

CLAUSE_SPLIT = re.compile(r"[;\n,]|\.\s+|(?:^|\s)\d+\.\s+")


def _rules_hash() -> str:
    return hashlib.sha256(repr(RULES).encode("utf-8")).hexdigest()


RULES_SHA256 = _rules_hash()


def split_clauses(request: str) -> list[str]:
    parts = [c.strip(" ,.\t") for c in CLAUSE_SPLIT.split(request) if c and c.strip(" ,.\t")]
    return parts or [request.strip()]


def clause_count(clause: str) -> int:
    match = COUNT_RE.search(clause)
    if not match:
        return 1
    token = match.group("n").lower()
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def classify_clause(clause: str) -> str | None:
    if NEGATION_RE.search(clause):
        return None
    for brick, pattern in RULES:
        if re.search(pattern, clause, re.IGNORECASE):
            return brick
    return None


@dataclass
class KeywordEncoder:
    """Deterministic, offline, and always labelled as itself."""

    vocab: Vocabulary
    name: str = "keyword"

    @property
    def rules_sha256(self) -> str:
        return RULES_SHA256

    def decompose(self, request: str, context_files: tuple[str, ...] = ()) -> Decomposition:
        units = self.vocab.zero_units()
        matched = 0
        for clause in split_clauses(request):
            brick = classify_clause(clause)
            if brick is None:
                continue
            units[brick] += clause_count(clause)
            matched += 1

        total = sum(units.values())
        return Decomposition(
            units=units,
            context_files=context_files,
            rationale=f"{matched} clause(s) matched by rule set {RULES_SHA256[:8]}",
            confidence=0.0 if total == 0 else min(1.0, 0.4 + 0.15 * matched),
            source="keyword",
            out_of_scope=total == 0,
        )
