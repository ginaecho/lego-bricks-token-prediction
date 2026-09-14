"""Every failure mode the pipeline is allowed to have a name for."""

from __future__ import annotations


class TokenYieldError(Exception):
    """Base class."""


class VocabularyError(TokenYieldError):
    """bricks.toml is malformed. Carries every problem found, not just the first."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


class SchemaError(TokenYieldError):
    """A record does not satisfy its contract."""


class SealError(TokenYieldError):
    """Something tried to read the blind split outside the blind runner."""


class AdapterError(TokenYieldError):
    """The provider failed in a way that is not a measurement."""


class DecompositionError(TokenYieldError):
    """The encoder returned something that is not a valid brick vector."""


class BudgetError(TokenYieldError):
    """A dispatch would exceed the frozen envelope."""


class IdentifiabilityError(TokenYieldError):
    """The design matrix cannot support per-brick coefficients."""


class MisalignedFoldsError(TokenYieldError):
    """Two cross-validated forms were not scored on the same rows.

    Comparing them anyway would pair a prediction about one document with an actual from
    another, and the resulting "lift" would be an artefact of the misalignment.
    """
