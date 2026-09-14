"""The encoder prompt.

The vocabulary sheet handed to the model is the *same string* handed to human labellers
(:meth:`assay.vocabulary.Vocabulary.render_for_prompt`). If they read different
text, then encoder accuracy measured against human gold is partly measuring the gap
between two documents, and there is no way afterwards to say how much.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from assay.vocabulary import Vocabulary

OUTPUT_CONTRACT = """\
Return STRICT JSON and nothing else. No prose, no code fence, no explanation.

{
  "units": {<one integer for every brick named above, zero where absent>},
  "context_files": [<file names from the list above, or []>],
  "rationale": "<one short sentence>",
  "confidence": <number between 0 and 1>,
  "needs_clarification": <true only if the request cannot be decomposed as written>,
  "out_of_scope": <true only if no brick applies>
}

If you set needs_clarification or out_of_scope to true, every unit count must be 0.
"""

WORKED_EXAMPLES = [
    (
        "read the filing, pull three fields, write two auditor checks",
        {"Review": 1, "Extract": 3, "Validate": 2},
        "'checks' is Validate, not Draft: each one is independently scorable.",
    ),
    (
        "find which company reported the 26% decline, then draft a risk note",
        {"Retrieve": 1, "Draft": 1},
        "The source is not named, so locating it is Retrieve. '26%' is not a count.",
    ),
    (
        "compare these two filings, then write a board summary",
        {"Reconcile": 1, "Report": 1},
        "Two documents, one comparison. Presenting findings already made is Report.",
    ),
    (
        "what do you think about our pricing?",
        {},
        "No brick applies to an open-ended opinion; out_of_scope with an all-zero vector.",
    ),
]


def build_prompt(vocab: Vocabulary, request: str, corpus_files: Sequence[str] = ()) -> str:
    lines = [vocab.render_for_prompt(), ""]

    if corpus_files:
        lines.append("AVAILABLE DOCUMENTS")
        lines.extend(f"  {name}" for name in corpus_files)
        lines.append("")

    lines.append("WORKED EXAMPLES")
    for text, units, why in WORKED_EXAMPLES:
        full = vocab.zero_units()
        full.update(units)
        shown = ", ".join(f'"{k}": {v}' for k, v in full.items())
        lines.append(f'  request: "{text}"')
        lines.append(f"  units:   {{{shown}}}")
        lines.append(f"  why:     {why}")
        lines.append("")

    lines.append(OUTPUT_CONTRACT)
    lines.append("REQUEST")
    lines.append(request.strip())
    return "\n".join(lines)


def prompt_hash(vocab: Vocabulary) -> str:
    """Changes whenever the vocabulary sheet, contract or examples change.

    The encoder prompt is frozen before validation gold is opened; this hash is what makes
    that freeze checkable rather than a claim.
    """
    payload = vocab.render_for_prompt() + OUTPUT_CONTRACT + repr(WORKED_EXAMPLES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
