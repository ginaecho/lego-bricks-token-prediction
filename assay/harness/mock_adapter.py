"""A deterministic simulator of an agent runtime -- and a negative control.

Two modes, and the second one is the important one:

``brick``
    Cost is generated as ``boot + slope*bytes + sum(marginal * units)``. The pipeline must
    recover the planted coefficients and choose a decoder form.

``null``
    Cost is generated as ``boot + slope*bytes``. **There is no brick term at all.** The
    pipeline must fail to find one: selection must stop at a baseline, the lift bar must
    not be cleared, and the verdict must come out Narrow.

Without the ``null`` mode a green suite proves only that the estimator can find a signal
someone planted for it. With it, the suite also proves the system can say *"there is
nothing here"* -- which is the outcome the whole plan is designed to be able to reach.
A build where ``null`` mode yields Feasible is broken no matter what else passes.

There is a third failure this adapter deliberately does *not* simulate. Whether the design
can support a per-brick coefficient at all is a property of the design matrix, not of the
costs, so no choice of generative process can exercise it. That guard is tested directly
against a confounded design in ``tests/test_pipeline.py``.

Nothing this adapter produces can move a gate: every run it generates is stamped
``pipeline_only``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from assay.evidence import EvidenceClass
from assay.harness.adapter import RunResult

REFERENCE_MARGINALS = {
    "Review": 52.0,
    "Extract": 418.0,
    "Classify": 0.0,
    "Retrieve": 5384.0,
    "Reconcile": 1770.0,
    "Draft": 0.0,
    "Remediate": 0.0,
    "Validate": 1038.0,
    "Report": 838.0,
}


class MockMode(str, Enum):
    BRICK = "brick"
    NULL = "null"


@dataclass
class MockSpec:
    """The generative process the pipeline is asked to recover.

    ``boot`` is deliberately far below the reference's 29,821: that number belongs to a
    runtime nobody here owns, and assuming it would bake a foreign harness into our tests.
    """

    boot: float = 4_800.0
    bytes_slope: float = 0.37
    marginals: dict[str, float] = field(default_factory=lambda: dict(REFERENCE_MARGINALS))
    noise_cv: float = 0.006
    completion_share: float = 0.03

    def expected(self, units: dict[str, int], context_bytes: int, mode: MockMode) -> float:
        total = self.boot + self.bytes_slope * context_bytes
        if mode is MockMode.BRICK:
            total += sum(self.marginals.get(k, 0.0) * v for k, v in units.items())
        return total


class MockAdapter:
    """Offline, seeded, and honest about being a simulation.

    Replicates of the same instruction differ (otherwise the noise floor would be zero and
    M1 would pass for free), but a whole campaign replayed from a fresh adapter reproduces
    byte for byte.
    """

    def __init__(
        self,
        mode: MockMode | str = MockMode.BRICK,
        spec: MockSpec | None = None,
        *,
        seed: int = 1337,
        model: str = "mock-snapshot-0001",
    ) -> None:
        self.mode = MockMode(mode)
        self.spec = spec or MockSpec()
        self.seed = seed
        self.name = f"mock:{self.mode.value}"
        self.model = model
        self.evidence_class = EvidenceClass.PIPELINE_ONLY
        self.caching_disabled = True  # nothing to cache; the simulator has no provider
        self._units: dict[str, dict[str, int]] = {}
        self._calls: dict[str, int] = {}

    def register(self, instruction: str, units: dict[str, int]) -> None:
        """Tell the simulator what a given instruction is asking for.

        A real adapter never needs this -- it reads the words. The simulator is a model of
        the campaign, so it is allowed to know the campaign.
        """
        self._units[_key(instruction)] = dict(units)

    def run(
        self, instruction: str, context_files: list[Path], *, temperature: float = 0.0
    ) -> RunResult:
        key = _key(instruction)
        units = self._units.get(key, {})
        context_bytes = sum(Path(p).stat().st_size for p in context_files if Path(p).exists())

        n = self._calls.get(key, 0)
        self._calls[key] = n + 1
        rng = random.Random(f"{self.seed}:{key}:{n}")

        expected = self.spec.expected(units, context_bytes, self.mode)
        noisy = expected * (1.0 + rng.gauss(0.0, self.spec.noise_cv))
        completion = max(1, round(noisy * self.spec.completion_share))
        prompt = max(1, round(noisy - completion))

        return RunResult(
            prompt_tokens=prompt,
            completion_tokens=completion,
            model_echoed=self.model,
            elapsed_s=round(0.5 + context_bytes / 50_000.0, 3),
            output_text=f"[mock:{self.mode.value}] {instruction[:60]}",
            tool_calls=len(context_files),
            tool_result_bytes=context_bytes,
            raw={"mode": self.mode.value, "expected": expected, "replicate_index": n},
        )


def _key(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()
