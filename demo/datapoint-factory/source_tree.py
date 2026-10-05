"""Materialize the measured build corpus and its Python code from a git ref.

The corpus and the training code live on ``main`` (``data/build_simulations``,
``token_yield``, ``examples/train_construction_model.py``). The app exports
exactly those paths with ``git archive`` into a local cache and imports the
real modules from there, so nothing is re-implemented or simulated.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

PATHS = (
    "data/build_simulations",
    "token_yield",
    "examples/__init__.py",
    "examples/build_simulation_corpus.py",
    "examples/train_construction_model.py",
)


def repository_root() -> Path:
    output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent, text=True)
    return Path(output.strip())


def resolve_commit(repo: Path, ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo, text=True).strip()


def export(repo: Path, ref: str, cache: Path) -> Path:
    """Extract PATHS at ``ref`` into ``cache/<commit>``; reuse an existing export."""
    commit = resolve_commit(repo, ref)
    target = cache / commit[:12]
    marker = target / ".complete"
    if not marker.exists():
        archive = subprocess.check_output(["git", "archive", "--format=tar", commit, *PATHS], cwd=repo)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter="data")
        marker.write_text(commit, encoding="utf-8")
    return target


def activate(root: Path) -> None:
    """Make the exported ``token_yield`` and ``examples`` packages importable."""
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def prepare(ref: str, source_root: Path | None, cache: Path) -> tuple[Path, str]:
    """Return (root containing data/build_simulations, provenance label)."""
    if source_root is not None:
        root = source_root.resolve()
        label = f"checkout {root}"
    else:
        repo = repository_root()
        root = export(repo, ref, cache)
        label = f"git ref {ref} @ {resolve_commit(repo, ref)[:12]}"
    if not (root / "data" / "build_simulations" / "data_points").is_dir():
        raise SystemExit(f"No data/build_simulations corpus under {root}")
    activate(root)
    return root, label
