# Marketplace model constraints

These constraints apply to the marketplace model work, not unrelated modules.

* Preserve current scoping and cost APIs, historical runs, and user changes.
* Reuse the existing numerical estimators and pytest runner. Add no dependency
  without a demonstrated need.
* Keep measured labels distinct from synthetic test fixtures and estimates.
* Keep project groups and historical holdout assignments intact.
* Persist enough provenance to reproduce fits. Never persist credentials.
* Do not access new sources, spend API credits, or deploy resources in the
  offline milestone.
* Surface invalid data, unsupported settings, and missing evidence explicitly.
* Separate token prediction, API rating, quality acceptance, and selling price.
* Do not claim calibrated intervals or full-delivery capability from the
  existing four-operation scoping experiment.
