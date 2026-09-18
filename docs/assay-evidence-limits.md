# Reference data — provenance

Copied verbatim from the reference repository. **Read-only. Never fitted. Never mixed into
a Token Yield campaign.**

| Field | Value |
| --- | --- |
| Repository | `ginaecho/lego-bricks-token-prediction` |
| Commit | `c63702d` (`main`) |
| Licence | MIT |
| Retrieved | 2026-09-04 |
| Files | `train_runs.jsonl` (39 runs), `decompose_cases.jsonl` (3 encoder cases) |

## What these rows can and cannot support

They are the author's measurements on the author's runtime — a Claude Code subagent over
`claude-haiku-4-5`, whose empty task cost 29,821 tokens. Every number derived from them is
stamped `replay` and cannot move a gate.

Three limits worth stating plainly, because they are easy to miss:

1. **No input/output split.** Each row carries a single scalar `tokens` field. Output tokens
   cost several times what input tokens cost, so no dollar figure can be recovered from
   these rows at any level of care.
2. **The intercept is a foreign harness.** 29,821 tokens is a property of Claude Code's
   subagent wrapper, not of the model. Our own boot will differ by an order of magnitude,
   and that is expected — a runtime artefact, not a failed replication.
3. **The batching savings were never measured.** The 64–71% figures are model-derived
   counterfactuals, not a live batched-versus-separate comparison. They are withdrawn as
   priors and quarantined in `docs/excluded-priors.md`.

## What `ty audit` does with them

Reproduces the published leave-one-out table — a transcription check — and prints the same
six forms scored on the variable portion beside it, so nobody reads 2.55% without seeing
what those models know once the start-up toll is removed from both sides.

It authorises nothing. A different runtime is a different experiment.
