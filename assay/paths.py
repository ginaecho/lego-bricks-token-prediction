"""Where things live, resolved from the package rather than the working directory.

`ty` is meant to be runnable from anywhere in the repo, and the studio server runs with a
working directory nobody controls. Anchoring the defaults to the installed package removes
a whole class of "works on my machine" path bugs.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DATA = PACKAGE_ROOT / "data"
BRICKS = DATA / "bricks.toml"

FROZEN = REPO_ROOT / "frozen"
VERDICT_POLICY = FROZEN / "verdict-policy.json"
SPLIT_SEAL = FROZEN / "split-seal.json"

EXPERIMENTS = REPO_ROOT / "experiments"
"""The reference evidence. These are the repository's own committed measurements -- the
very rows the assay exists to re-examine -- so the audit reads them in place rather than
keeping a second copy that could drift away from the thing it is auditing."""

REFERENCE_RUNS = EXPERIMENTS / "train_runs.jsonl"
REFERENCE_DECOMPOSE = EXPERIMENTS / "decompose_cases.jsonl"

DEMO = REPO_ROOT / "demo"
"""The sampled demo project: a small corpus that makes the whole loop runnable offline."""
