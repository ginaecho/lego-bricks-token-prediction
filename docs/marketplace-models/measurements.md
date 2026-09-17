# Marketplace execution-variant measurements

The [frozen manifest](../../experiments/marketplace/manifest.json) binds the six
previously approved local paraphrased briefs, original scoping contracts,
runtime, request payloads, execution order, and additional budget approval.
Preparation is offline; live execution requires an explicit flag.

## Scope

The matrix contains 120 calls: eight contiguous batching layouts for each of
six briefs, once per layout. Each layout covers Extract, Classify, Plan, and
Report exactly once with the original quantities.

All calls independently consume the original brief. Serial scheduling does not
mean sequential handoffs: no generated result becomes another call's input.
Sequential handoffs require a separate validated contract and are blocked.
Automatic retries, tools, new source downloads, and customer delivery are
excluded.

The four training projects and two historical holdout projects keep their
original identities across all layouts. There are zero independent calibration
or fresh-test projects. Layout order rotates deterministically across projects;
this is not randomized causal evidence or repeated-measurement replication.

## Preview without spending

```powershell
.\.venv\Scripts\python.exe -m examples.marketplace_measurements
```

The [runner](../../examples/marketplace_measurements.py) verifies that the
checked-in manifest matches the local source and code hashes. It neither
authenticates nor calls the provider during preview.

The matrix's historical uncached retail bound is USD 6.71023. Its sum of
worst-case safety reservations is USD 139.69784, but reservations are settled
one call at a time. Neither number is an expected bill. The operational USD 48
stop takes precedence over completing the matrix.

## Explicitly authorized execution

The user approved a new additional USD 50 cap for these six briefs. This is
separate from historical pilot spending. The execution command for that
single-use approval is:

```powershell
.\.venv\Scripts\python.exe -m examples.marketplace_measurements `
  --execute `
  --run-dir runs\20260916_marketplace_measurements_v1
```

Execution requires existing Entra credentials and read access to verify the
pinned deployment. No resources are provisioned. Deployment metadata brackets
successful execution; response aliases do not independently attest an immutable
backend version on each call.

The runner exclusively claims the approval before execution. Changing the run
directory does not reset the budget or authorize another campaign. Do not
delete the claim to retry. After a failure, reconcile evidence and obtain new
authorization before another attempt.

## Accounting and acceptance

Before dispatch, the runner persists a conservative reservation, exact request,
and attempt identity. Responses retain raw evidence and hashes without
authentication headers. Five measured usage channels are required. Failed or
incomplete responses retain their charges; unknown usage retains the full
reservation rather than becoming a zero-cost success.

A contract failure, uncertain request, runtime mismatch, or reservation breach
halts the campaign without automatic retry or resume. Budget exhaustion may
leave the matrix incomplete. Inspect the saved status, attempt records, active
reservations, and failure evidence before interpreting any cost total.

Recorded retail costs use the September 14 snapshot and are not reconciled
invoices. Safety-rate accounting is a separate conservative execution guard,
not provider pricing or a provider-enforced billing limit.

Passing structural and grounding checks is not human quality approval.
Incomplete workflows are not complete deliverables. These measurements do not
establish quality noninferiority, calibrated intervals, production readiness, or
a universal batching discount.

## Recorded campaign outcome

The [September 16 run](../../runs/20260916_marketplace_measurements_v1) halted
after 36 calls, with 35 contract passes. The last response omitted the required
outer `report` key. No retry, output repair, or training on partial results was
performed.

Usage and raw evidence for all 36 attempts were verified. Historical-rate
calculated cost is USD 0.21738; safety-accounting cost is USD 2.62992. No usage
is unknown and no reservations remain active. The campaign is incomplete and
the approval is already claimed; the execution command above must not be used
to reset or bypass that state.
