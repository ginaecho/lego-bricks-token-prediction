# Foundry wave-2 pilot: preregistered protocol

Wave 2 tests whether LEGO-like task features become predictive after removing
the large Copilot harness constant and deliberately varying task mechanisms.
It is a separate runtime stratum from wave 1 and must never be pooled with it.

## Fixed runtime and measurement

- Microsoft Foundry Responses API, deployment `gpt-5-mini`
- Entra authentication; bearer tokens and client secrets are never persisted
- `reasoning.effort=minimal`, `text.verbosity=low`
- Raw response ID, status, model, request/response hashes, and latency retained
- Input, cached input, output, reasoning, total, model-call, and tool-call
  channels retained separately
- Usage includes all attempts; acceptance is a separate outcome

The target is a multi-target ledger, not one ambiguous total. Monetary cost is
unreconciled until an Azure bill or explicit rate card is attached.

## Hard spending boundary

The campaign has a **USD 200 hard cap** and a USD 190 operational stop, leaving
headroom for delayed metering. Before every request, the runner reserves a
worst-case amount using one UTF-8 byte as at most one input token and the
declared output-token limit. Authorization uses deliberately conservative
safety rates of USD 20/M input and USD 200/M output. These are safeguards, not
claims about Azure pricing.

If a reservation would cross the operational stop, the request is not sent.
If Azure supplies an actual charge, settlement uses the greater of that charge
and the safety estimate. The pilot stops after exactly 30 sessions regardless
of the remaining budget.

## Frozen pilot matrix

`experiments/foundry_wave2/preregistration.json` fixes the order before the
first dispatch:

| mechanism | sessions | crossed factor |
|---|---:|---|
| Summarise | 9 | 1,024 / 10,240 / 102,400 context bytes, 3 replicates |
| Transform | 9 | 2 / 8 / 32 field mappings, 3 replicates |
| Fetch | 6 | 3 Federal Register records, paired snapshot/live arms |
| null/drift control | 6 | one randomized into every five-session block |

Replicates share a `group_id`; no grouped-validation fold may split them.
Fetch snapshot and live arms ask for the same exact fields. The snapshot arm
embeds the frozen API response. The live arm can invoke only
`fetch_public_api(manifest_id)`, and the controller accepts only the three
exact HTTPS GET requests in `api_manifests.json`. Redirects, oversized
responses, schema failures, unknown manifests, unknown tools, and extra tool
turns fail explicitly.

## Frozen gates

Expansion requires all of:

1. Complete telemetry and hashes for every response.
2. Summarise and Transform high-to-low median effects at least three times
   pooled within-shape MAD, with monotonic medians.
3. Every live Fetch session follows the planned tool/model-call branch and its
   incremental usage exceeds three times within-arm MAD.
4. Semantic acceptance is at least 80% per shape.

A failed mechanism is dropped rather than rescued by adding rows.

A ridge model may replace the constant only if grouped CV by shape/source
improves excess-over-baseline error by at least 20% and a 95% grouped-bootstrap
confidence interval excludes zero. Validation interval coverage at 90% and
95% must be within 10 percentage points of nominal. The autoencoder remains
diagnostic-only. The 24-session blind reserve is opened once and never used
for retuning.

After the pilot, execution stops and reports the gates. Any move toward the
120-session campaign ceiling requires new approval.

## Completed pilot result

The pilot completed at
[`runs/20260826_1627_wave2`](../runs/20260826_1627_wave2/) and stopped after
30 sessions. Known provider usage was 94,351 tokens across 32 metered model
calls, one unmetered incomplete attempt, and three controlled API calls. The
conservative safety ledger settled at
USD 5.05074, including the full reservation for one attempt with missing
telemetry; this is not represented as an Azure invoice.

Summarise and Transform passed their monotonic 3×-MAD signal gates. Fetch
followed the planned two-call branch, but its requested within-shape MAD is
unidentifiable with one row per arm/shape. The pre-run implementation used a
conservative between-document, within-arm MAD and failed; a labeled post-hoc
paired-delta diagnostic would pass and is not substituted for the frozen gate.
One 100 KB Summarise response was incomplete, and the frozen
literal oracle rejected otherwise faithful normalized titles, agency names,
and dates. A labeled semantic regrade accepted 29/30 attempts but still left
the 100 KB shape at 2/3, below the 80% threshold. The mechanism-expansion gate
therefore failed.

The grouped ridge total-token model nevertheless passed its separate promotion
gate: grouped-CV MAE 1,722 versus 5,575 for the constant, a 69.1% improvement
with grouped-bootstrap 95% CI 48.7–89.5%. The standardized design was full rank
with condition number 2.88. This is evidence that pre-run task features carry
signal in the direct-API runtime; it is not authorization to expand because
the quality, telemetry, and Fetch mechanism gates remain failed.

Model-call and tool-call targets are mechanically predictable because the live
Fetch branch is declared before dispatch; their target-specific gains are not
evidence of general token-prediction skill. Interval coverage uses each
group's prediction against a radius calibrated from other groups' out-of-fold
residuals, rather than calibrating and scoring on the same residual.

One remaining feature limitation is explicit in the run artifact: live Fetch
rows used zero context bytes, so the `live_fetch` coefficient absorbs both
continuation and response-size effects. A subsequent design must preregister
the frozen response-size proxy as a separate pre-run feature.
