"""Campaign design: which probes, in what order, against which documents.

The base tier is the reason per-brick coefficients are estimable at all: each base probe
isolates one brick across three size bands, so unit counts and byte counts move
independently by construction. :func:`assay.harness.probes.orthogonality_probe`
then checks that the construction held.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from assay.corpus import SealedCorpus
from assay.harness.probes import (
    Probe,
    brick_instruction,
    compose_instruction,
    ladder_probe,
    null_probe,
    orthogonality_probe,
)
from assay.schemas import Split, Tier
from assay.vocabulary import Vocabulary

UNIT_LEVELS = (1, 2, 4)


@dataclass
class CampaignPlan:
    campaign_id: str
    probes: list[Probe]
    seed: int
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_dispatches(self) -> int:
        return len(self.probes)

    def by_tier(self, tier: Tier) -> list[Probe]:
        return [p for p in self.probes if p.tier is tier]

    def summary(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "seed": self.seed,
            "n_dispatches": self.n_dispatches,
            "by_tier": {t.value: len(self.by_tier(t)) for t in Tier if self.by_tier(t)},
        }


def design_campaign(
    vocab: Vocabulary,
    corpus: SealedCorpus,
    *,
    campaign_id: str = "campaign_001",
    replicates: int = 2,
    n_nulls: int = 3,
    n_composites: int = 12,
    unit_levels: tuple[int, ...] = UNIT_LEVELS,
    seed: int = 1337,
) -> CampaignPlan:
    """Build the fit-split campaign. Blind and batching probes are built elsewhere."""
    rng = random.Random(seed)
    fit_docs = [d.name for d in corpus.fit_documents()]
    corpus.assert_fit_only(fit_docs)
    if len(fit_docs) < 3:
        raise ValueError("need at least 3 fit documents to span three size bands")

    bands = corpus.bands(fit_docs, n_bands=3)
    bands = [b for b in bands if b]
    probes: list[Probe] = []

    for i in range(1, n_nulls + 1):
        probes.append(null_probe(i, "open"))

    for i, band in enumerate(bands):
        probes.append(ladder_probe(band[len(band) // 2], i))

    orth_pool = sorted(fit_docs, key=corpus.size)[: max(4, len(fit_docs) // 2)]
    if len(orth_pool) >= 4:
        for i, k in enumerate((1, 2, 4)):
            probes.append(orthogonality_probe(tuple(orth_pool[:k]), i))

    for brick_idx, brick in enumerate(vocab.names):
        for band_idx, band in enumerate(bands):
            doc = band[rng.randrange(len(band))]
            # Rotate the unit level against the brick index as well as the band, so unit
            # count does not track document size. Keying it to the band alone would make
            # "4 units" and "large document" the same column, and every per-brick
            # coefficient would then be an arbitrary split of one effect.
            n = unit_levels[(brick_idx + band_idx) % len(unit_levels)]
            for rep in range(1, replicates + 1):
                probes.append(
                    Probe(
                        probe_id=f"base-{brick.lower()}-s{band_idx}-r{rep}",
                        tier=Tier.BASE,
                        split=Split.FIT,
                        instruction=brick_instruction(brick, n),
                        units={brick: n},
                        context_files=(doc,),
                        replicate=rep,
                    )
                )

    names = list(vocab.names)
    for i in range(n_composites):
        # Walk arity and size band on independent cycles. Stepping them together --
        # arity 2 on small documents, arity 4 on large ones -- would make "more bricks"
        # and "more bytes" the same column, and the per-brick coefficients would be an
        # arbitrary split of one effect no matter how well the model fitted.
        arity = 2 + (i // len(bands)) % 3
        band = bands[i % len(bands)]
        chosen = rng.sample(names, arity)
        parts = [(b, unit_levels[rng.randrange(len(unit_levels))]) for b in chosen]
        doc = band[rng.randrange(len(band))]
        units: dict[str, int] = {}
        for b, n in parts:
            units[b] = units.get(b, 0) + n
        probes.append(
            Probe(
                probe_id=f"composite-{i:02d}",
                tier=Tier.COMPOSITE,
                split=Split.FIT,
                instruction=compose_instruction(parts),
                units=units,
                context_files=(doc,),
            )
        )

    for i in range(1, n_nulls + 1):
        probes.append(null_probe(i, "close"))

    plan = CampaignPlan(campaign_id=campaign_id, probes=probes, seed=seed)
    plan.counts = {t.value: len(plan.by_tier(t)) for t in Tier if plan.by_tier(t)}
    return plan


def design_blind(
    vocab: Vocabulary,
    corpus: SealedCorpus,
    *,
    n_tasks: int = 24,
    unit_levels: tuple[int, ...] = UNIT_LEVELS,
    seed: int = 20260904,
) -> list[Probe]:
    """The sealed prospective set.

    Built from blind documents only, and only ever executed through the blind runner,
    which requires a committed quote per task before it will dispatch anything.
    """
    rng = random.Random(seed)
    blind_docs = [d.name for d in corpus.blind_documents(allow_blind=True)]
    if not blind_docs:
        raise ValueError("the seal reserves no blind documents")
    names = list(vocab.names)
    probes: list[Probe] = []
    for i in range(n_tasks):
        arity = 1 + (i % 4)
        chosen = rng.sample(names, min(arity, len(names)))
        parts = [(b, unit_levels[rng.randrange(len(unit_levels))]) for b in chosen]
        units: dict[str, int] = {}
        for b, n in parts:
            units[b] = units.get(b, 0) + n
        probes.append(
            Probe(
                probe_id=f"BLIND-{i + 1:03d}",
                tier=Tier.BLIND,
                split=Split.BLIND,
                instruction=compose_instruction(parts),
                units=units,
                context_files=(blind_docs[i % len(blind_docs)],),
            )
        )
    return probes


def design_batching(
    vocab: Vocabulary,
    corpus: SealedCorpus,
    *,
    n_two_task: int = 8,
    n_four_task: int = 3,
    seed: int = 4242,
) -> list[Probe]:
    """Matched separate and batched arms.

    The separate arm is re-run rather than reused from the fit campaign: reusing it would
    compare a fresh batched dispatch against runs made under different conditions, and any
    drift between them would show up as a saving.
    """
    rng = random.Random(seed)
    fit_docs = [d.name for d in corpus.fit_documents()]
    corpus.assert_fit_only(fit_docs)
    names = list(vocab.names)
    probes: list[Probe] = []

    def bundle(idx: int, arity: int, prefix: str) -> None:
        bundle_id = f"{prefix}-{idx + 1:03d}"
        chosen = rng.sample(names, arity)
        parts = [(b, 1) for b in chosen]
        doc = fit_docs[rng.randrange(len(fit_docs))]
        for j, (brick, n) in enumerate(parts):
            probes.append(
                Probe(
                    probe_id=f"{bundle_id}-sep-{j}",
                    tier=Tier.BATCHING,
                    split=Split.BATCHING,
                    instruction=brick_instruction(brick, n),
                    units={brick: n},
                    context_files=(doc,),
                    bundle_id=bundle_id,
                    arm="separate",
                )
            )
        units: dict[str, int] = {}
        for b, n in parts:
            units[b] = units.get(b, 0) + n
        probes.append(
            Probe(
                probe_id=f"{bundle_id}-batch",
                tier=Tier.BATCHING,
                split=Split.BATCHING,
                instruction=compose_instruction(parts),
                units=units,
                context_files=(doc,),
                bundle_id=bundle_id,
                arm="batched",
                tasks=tuple(b for b, _ in parts),
            )
        )

    for i in range(n_two_task):
        bundle(i, 2, "BATCH2")
    for i in range(n_four_task):
        bundle(i, 4, "BATCH4")
    return probes
