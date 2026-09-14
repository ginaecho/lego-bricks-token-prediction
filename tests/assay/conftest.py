"""Shared fixtures.

Deliberately small, and deliberately offline. Anything that needs a network, a key, or a
human is not here -- the whole suite must run on a fresh clone with no credentials, which
is also the acceptance test for the evidence package.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay.corpus import CorpusManifest, SealedCorpus, index_corpus, split_and_seal
from assay.harness.campaign import design_campaign
from assay.harness.mock_adapter import MockAdapter, MockMode
from assay.harness.runner import run_campaign
from assay.pipeline import boot_estimate as _boot_estimate
from assay.pricing import PROVISIONAL, Pricing
from assay.schemas import Run, read_runs
from assay.paths import BRICKS
from assay.vocabulary import Vocabulary, load_vocabulary

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_HASH = "test-runtime-0001"

# Three size bands roughly an order of magnitude apart, matching the context ladder the
# plan calls for. The content is filler; only the byte counts are load-bearing.
DOC_SIZES = {
    "small": (900, 1_100, 1_300, 1_500),
    "medium": (8_000, 9_000, 10_000, 11_000),
    "large": (30_000, 34_000, 38_000, 42_000),
}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    return load_vocabulary(BRICKS)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    n = 0
    for band, sizes in DOC_SIZES.items():
        for size in sizes:
            n += 1
            body = f"Filing {n} ({band}).\n" + ("segment revenue line item. " * (size // 20))
            (root / f"doc-{n:02d}-{band}.txt").write_text(body[:size], encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def manifest(corpus_dir: Path) -> CorpusManifest:
    return index_corpus(corpus_dir)


@pytest.fixture(scope="session")
def sealed(manifest: CorpusManifest, corpus_dir: Path) -> SealedCorpus:
    seal = split_and_seal(manifest, blind_fraction=0.25, seed=99)
    return SealedCorpus(manifest, seal, root=corpus_dir)


@pytest.fixture(scope="session")
def pricing() -> Pricing:
    return PROVISIONAL


def execute_campaign(
    mode: MockMode,
    vocab: Vocabulary,
    sealed: SealedCorpus,
    pricing: Pricing,
    out_dir: Path,
    *,
    seed: int = 1337,
) -> list[Run]:
    """Design and run a whole fit campaign against the simulator. Used by both arms."""
    plan = design_campaign(vocab, sealed, campaign_id=f"mock_{mode.value}", seed=seed)
    adapter = MockAdapter(mode, seed=seed)
    out = out_dir / f"runs_{mode.value}.jsonl"
    run_campaign(
        plan.probes,
        adapter,
        out,
        campaign_id=plan.campaign_id,
        pricing=pricing,
        corpus=sealed,
        runtime_hash=RUNTIME_HASH,
        expected_model=adapter.model,
    )
    return read_runs(out, pricing)


def boot_estimate(runs: list[Run]) -> float:
    """Re-exported so existing tests keep one import site.

    The definition lives in ``assay.pipeline`` because it is the most load-bearing
    scalar in the project, and a quantity that decides every variable-portion metric
    should not be defined in the test fixtures.
    """
    return _boot_estimate(runs)


@pytest.fixture(scope="session")
def brick_runs(vocab, sealed, pricing, tmp_path_factory) -> list[Run]:
    return execute_campaign(
        MockMode.BRICK, vocab, sealed, pricing, tmp_path_factory.mktemp("brick")
    )


@pytest.fixture(scope="session")
def null_runs(vocab, sealed, pricing, tmp_path_factory) -> list[Run]:
    return execute_campaign(
        MockMode.NULL, vocab, sealed, pricing, tmp_path_factory.mktemp("null")
    )
