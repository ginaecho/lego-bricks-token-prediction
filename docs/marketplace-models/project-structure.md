# Relevant project structure

The repository is a Python library with CLI/example entry points and experimental
measurement runs. This is a focused inventory for the marketplace model change.

| Area | Existing implementation | Role |
|---|---|---|
| Contracts | [customer_decomposition.py](../../token_yield/customer_decomposition.py) | Four scoping operations, template validation, quote features, hashes |
| Fitting | [customer_models.py](../../token_yield/customer_models.py) | Call-level USD models and whole-project token/USD models |
| Estimators | [robust.py](../../token_yield/robust.py) | Constant/ridge models and project-grouped folds |
| Measurement | [customer_pilot.py](../../token_yield/customer_pilot.py) | Frozen plans, raw response verification, measured runs |
| Pricing | [economics.py](../../token_yield/economics.py) | Existing rate-card calculations |
| Inputs | [customer_requests](../../experiments/customer_requests) | Paraphrased briefs, template, pilot configuration |
| Source run | [20260914_customer_scoping_v2](../../runs/20260914_customer_scoping_v2) | Recorded standalone and batched calls |
| Checks | [tests](../../tests) | Existing pytest regression suite |

There is no marketplace service registry or purchasing API in this scope.
The existing project forecaster requires all four operations; it cannot serve
as an arbitrary selection API. The call-level feature encoder already supports
individual operations and measured batches.
