"""Token Yield -- predict agent token cost before the work runs, reconcile it after.

Built to the plan in ``docs/TOKEN_YIELD_PLAN.md`` of the reference repository. Two rules
shape every module here:

1. **Score twice.** A start-up toll dominates small agent tasks, so total error is easy and
   uninformative. Every accuracy number is reported beside its variable-portion twin
   (:func:`assay.metrics.vwape`), and only the second one decides anything.
2. **Only real dispatches decide.** Fixtures and the mock adapter are stamped
   ``pipeline_only`` (:mod:`assay.evidence`) and cannot move a gate.
"""

from __future__ import annotations

__version__ = "0.1.0"
