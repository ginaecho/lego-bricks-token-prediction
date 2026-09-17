# Existing technology stack

* Python package requirement: 3.9 or newer, declared in
  [pyproject.toml](../../pyproject.toml).
* Setuptools build backend; existing runtime dependency is tiktoken.
* Pytest is the existing development test runner.
* Constant and standardized ridge regressions are implemented in Python,
  without a separate machine-learning dependency.
* Experiments and fitted artifacts use JSON files; no database is required.
* Live measurement code exists, but offline fitting requires neither credentials
  nor an active cloud deployment.
* Preserve the package's Python compatibility; do not raise its minimum version
  as part of this work.
