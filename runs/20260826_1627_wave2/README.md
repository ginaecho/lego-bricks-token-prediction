# Foundry wave 2 — 30-session pilot

This folder is the durable evidence for the preregistered direct-Foundry pilot.
Execution stopped after exactly 30 logical sessions and did not open the
66-session expansion or 24-session blind reserve.

## Runtime and spend boundary

- Endpoint stratum: Microsoft Foundry Responses API
- Deployment: `gpt-5-mini`
- Reasoning: minimal
- Text verbosity: low
- Sessions: 30
- Metered model calls: 32, plus one unmetered incomplete attempt
- Controlled public-API tool calls: 3
- Known measured tokens: 94,351
  - input: 89,864
  - cached input (subset of input): 27,136
  - output: 4,487
  - reasoning (subset of output): 0
- Conservative settled safety ledger: **USD 5.05074**
- Hard cap: USD 200; operational stop: USD 190

The safety ledger is not an Azure invoice. It uses intentionally conservative
authorization rates (USD 20/M input and USD 200/M output), and would charge
reasoning detail again even though it is a subset of output. One incomplete
100 KB Summarise attempt lost provider telemetry because the original adapter
validated status before metering; its full USD 2.35606 pre-request reservation
is retained in the safety ledger rather than represented as zero usage.

## Frozen gate outcome

| gate | result | evidence |
|---|---|---|
| telemetry complete | **fail** | one incomplete attempt has unknown usage |
| frozen acceptance ≥80% per shape | **fail** | the literal Summarise oracle rejected normalized equivalents |
| Summarise signal | **pass** | medians 478 → 2,738 → 25,303.5; 1,838.9× pooled MAD |
| Transform signal | **pass** | medians 110 → 278 → 950; zero observed within-shape MAD |
| Fetch signal | **undetermined; conservative implementation failed** | planned branching occurred; requested within-shape MAD is unidentifiable with one row per arm/shape |
| mechanism expansion | **fail** | all mechanism gates were required |

Before dispatch, the implementation conservatively substituted pooled
between-document, within-arm MAD (801.5), under which the 328-token median
increment fails at 0.41×. That statistic is dominated by document size and is
not the requested within-shape noise. A post-hoc paired-delta diagnostic has
MAD 23 and ratio 14.26×, but it cannot replace the frozen implementation gate.

The frozen acceptance result is not overwritten. A labeled post-hoc normalized
semantic regrade accepts 29/30 outputs: all eight completed Summarise outputs
were faithful, while the incomplete 100 KB attempt remains rejected. Because
that shape is 2/3, even the normalized view still fails the ≥80%-per-shape
quality rule.

## Grouped model result

Replicates were grouped by shape/source. Ridge alpha was selected inside every
outer fold, so no replicate leaked into validation.

| metric | result |
|---|---:|
| constant grouped-CV MAE | 5,574.5 tokens |
| ridge grouped-CV MAE | 1,722.4 tokens |
| relative improvement | 69.1% |
| grouped-bootstrap 95% CI | 48.7% to 89.5% |
| 90% interval coverage | 87.0% |
| 95% interval coverage | 87.0% |
| standardized design condition number | 2.88 |
| matrix rank / columns | 6 / 6 |

The preregistered **total-token model-promotion gate passes**, but that does not
override the failed mechanism-expansion gate. Input, cached-input, model-call,
and tool-call targets also beat their constants; output and reasoning did not.
Model/tool-call prediction is mechanically easy because the live Fetch arm
declares the branch before dispatch, so it is not evidence of general token
prediction skill.

The pilot feature matrix records live Fetch context bytes as zero even though a
frozen response-size proxy was available before dispatch. Consequently the
`live_fetch` coefficient absorbs both continuation overhead and source-size
variation. A future preregistration should include declared fetch bytes as its
own pre-run feature rather than interpreting that coefficient causally.

## Files

- `frozen_inputs/` — exact preregistration, allowlist, and source snapshots
- `run_metadata.json` — endpoint, deployment, and frozen-input hashes
- `records.jsonl` — outputs, acceptance, raw usage channels, response IDs,
  request/response hashes, tool evidence, and errors
- `budget.json` — final hard-cap safety ledger
- `analysis.json` — frozen gates, semantic regrade, grouped models, coefficient
  intervals, influence, rank, condition number, and target-specific results

The decision is **pause for approval**. No additional paid sessions are
authorized by this run.
