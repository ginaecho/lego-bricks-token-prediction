# Marketplace model requirements

Date: 2026-09-16. Scope agreed in the marketplace planning conversation.

## Goal

Estimate the token consumption of explicitly selected, versioned AI services.
Start with Extract, Classify, Plan, and Report for public-request scoping.
Train numerical predictors, not new language models. The first milestone is an
offline research model, not a production marketplace or a customer price guarantee.

## User stories and requirements

### US1: Train a reproducible shared predictor

* REQ-001: Retain the existing four scoping operations and their definitions.
  Extract, Classify, and Plan count requirements; Report counts projects.
  Existing classification copies supplied annotations; it is not unlabelled
  classification. Existing operations independently read the same brief.
* REQ-002: Use measured input and output tokens as separate targets. Share
  learning across bricks through task-count features. Compare constant, size,
  size-and-units, and LEGO ridge baselines before selecting each channel.
* REQ-003: Use only quote-time features. Reject holdout rows in training and
  keep all repetitions of a project together. Select by equal-project-weight
  leave-one-project-out MAE. Historical holdout scores are retrospective.
* REQ-004: Save reloadable parameters, source hashes, runtime/template identity,
  feature definitions, training IDs, support information, and limitations.
  Reject incompatible runtime settings and malformed inputs explicitly.
* REQ-005: Reuse existing authorized measurements first. New source acquisition
  requires recorded permissions, provenance, deduplication, and a frozen split.
  New paid execution requires an explicit budget and input authorization.

### US2: Quote selected marketplace services

* REQ-006: Define versioned service contracts, input/output schemas, units,
  dependencies, quality checks, and supported ranges. Explicit selections
  bypass natural-language decomposition. Do not silently substitute services.
* REQ-007: Convert predicted tokens to API cost using a separately versioned
  rate card. Distinguish uncached/cached input, output, tool cost, human review,
  and selling-price margin. Do not infer cache savings without evidence.
* REQ-008: Estimate composed workflows using measured batching and handoff
  effects, not unconditional addition or a fixed historical saving percentage.
  Sequential dependencies, retries, and tools require new measurements.

### US3: Promote only evidence-backed models

* REQ-009: Report quality separately from incurred cost; retain metered
  quality-rejected completions. Unknown usage is not zero. Collect independent
  calibration and test projects before claiming prediction-range coverage.
  Abstain from production recommendations without the required evidence.
* REQ-010: Preserve existing APIs and experiments. Validate deterministic fits,
  artifact reload, grouping, input rejection, and backward compatibility.
  Consider task-family or dedicated predictors only after independently
  measured improvement over the shared baseline.

## Acceptance boundaries

The first milestone succeeds when two token channels are fitted offline,
saved, reloaded without refitting, and scored on the historical project holdout,
with targeted regression checks passing. No accuracy threshold is claimed in
advance for this small corpus.

Before production promotion, preregister acceptable absolute/relative error,
coverage, interval width, quality, latency, and independent-project sample-size
targets. These business thresholds are deliberately deferred to that gate;
they do not block the offline milestone.

Marketplace UI, payments, vendor onboarding, arbitrary project execution,
LLM fine-tuning, new cloud deployments, and automatic retraining are not part
of the first implementation.
