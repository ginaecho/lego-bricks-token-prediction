"""Load and validate the task vocabulary.

The vocabulary is the unit of work the whole cost model is denominated in, so it is
validated hard and early: a malformed ``bricks.toml`` raises with *every* problem listed,
not just the first, because fixing them one error per run is how a vocabulary ends up
half-specified.

Counting rule, normative and repeated wherever it matters: **count requested targets** --
not documents, not rows, not tool calls. Reading a document in service of another brick
does not add a brick.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from assay.errors import VocabularyError

from assay.paths import BRICKS as DEFAULT_PATH  # noqa: E402

REFERENCE_NAMES = (
    "Review",
    "Extract",
    "Classify",
    "Retrieve",
    "Reconcile",
    "Draft",
    "Remediate",
    "Validate",
    "Report",
)

VALID_CATEGORIES = frozenset(
    {"corrective", "adaptive", "perfective", "preventive", "cross-cutting"}
)


@dataclass(frozen=True)
class Brick:
    name: str
    category: str
    bought_as: str
    definition: str
    unit: str
    nearest_neighbour: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class Vocabulary:
    version: int
    bricks: tuple[Brick, ...]
    primary: tuple[str, ...]
    allow_extension: bool = False

    @property
    def names(self) -> list[str]:
        """Stable order, as written in the file. Design matrices depend on this."""
        return [b.name for b in self.bricks]

    def zero_units(self) -> dict[str, int]:
        return {name: 0 for name in self.names}

    def get(self, name: str) -> Brick:
        for b in self.bricks:
            if b.name == name:
                return b
        raise KeyError(name)

    def is_primary(self, name: str) -> bool:
        return name in self.primary

    def render_for_prompt(self) -> str:
        """The vocabulary sheet handed to the encoder and to human labellers.

        Both audiences read exactly the same text; if they read different text, encoder
        accuracy against human gold measures the wording gap rather than the encoder.
        """
        lines = [
            "TASK BRICKS",
            "",
            "Count REQUESTED TARGETS -- not documents, not rows, not tool calls.",
            "Reading a document in service of another brick does not add a brick.",
            "If the request fits no brick, return all zeros.",
            "",
        ]
        for b in self.bricks:
            lines.append(f"## {b.name} ({b.category})")
            lines.append(f"{b.definition}")
            lines.append(f"UNIT: {b.unit}")
            lines.append(f"NOT TO BE CONFUSED WITH: {b.nearest_neighbour}")
            for ex in b.examples:
                lines.append(f"  e.g. {ex}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def load_vocabulary(path: str | Path = DEFAULT_PATH) -> Vocabulary:
    path = Path(path)
    if not path.exists():
        raise VocabularyError([f"vocabulary file not found: {path}"])

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    problems: list[str] = []

    version = raw.get("version")
    if not isinstance(version, int):
        problems.append("`version` must be an integer")

    allow_extension = bool(raw.get("allow_extension", False))
    raw_bricks = raw.get("bricks")
    if not isinstance(raw_bricks, list) or not raw_bricks:
        raise VocabularyError(problems + ["`bricks` must be a non-empty array of tables"])

    bricks: list[Brick] = []
    seen: set[str] = set()
    for i, rb in enumerate(raw_bricks):
        where = f"bricks[{i}]"
        name = rb.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{where}: `name` must be a non-empty string")
            continue
        where = f"brick {name!r}"
        if name in seen:
            problems.append(f"{where}: duplicate name")
        seen.add(name)

        category = rb.get("category", "")
        if category not in VALID_CATEGORIES:
            problems.append(
                f"{where}: `category` must be one of {sorted(VALID_CATEGORIES)}, got {category!r}"
            )
        for field in ("bought_as", "definition", "unit", "nearest_neighbour"):
            value = rb.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where}: `{field}` must be a non-empty string")

        examples = rb.get("examples", [])
        if not isinstance(examples, list) or len(examples) < 2:
            problems.append(f"{where}: needs at least 2 `examples`")
            examples = list(examples) if isinstance(examples, list) else []

        bricks.append(
            Brick(
                name=name,
                category=str(category),
                bought_as=str(rb.get("bought_as", "")),
                definition=str(rb.get("definition", "")),
                unit=str(rb.get("unit", "")),
                nearest_neighbour=str(rb.get("nearest_neighbour", "")),
                examples=tuple(str(e) for e in examples),
            )
        )

    if not allow_extension and seen != set(REFERENCE_NAMES):
        missing = sorted(set(REFERENCE_NAMES) - seen)
        extra = sorted(seen - set(REFERENCE_NAMES))
        if missing:
            problems.append(f"missing reference bricks: {missing} (set allow_extension to add)")
        if extra:
            problems.append(f"non-reference bricks: {extra} (set allow_extension to add)")

    primary = raw.get("primary", [])
    if not isinstance(primary, list) or not primary:
        problems.append("`primary` must be a non-empty array naming the gate-carrying bricks")
        primary = []
    else:
        unknown = sorted(set(primary) - seen)
        if unknown:
            problems.append(f"`primary` names bricks that do not exist: {unknown}")

    if problems:
        raise VocabularyError(problems)

    return Vocabulary(
        version=int(version),  # type: ignore[arg-type]
        bricks=tuple(bricks),
        primary=tuple(str(p) for p in primary),
        allow_extension=allow_extension,
    )


def normalise_units(units: dict[str, object], vocab: Vocabulary) -> dict[str, int]:
    """Coerce a partial or noisy unit vector into a full, validated, all-keys vector."""
    out = vocab.zero_units()
    unknown = sorted(set(units) - set(out))
    if unknown:
        raise VocabularyError([f"unknown brick(s) in unit vector: {unknown}"])
    for name, value in units.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise VocabularyError([f"{name}: unit count must be an int, got {value!r}"])
        if value < 0:
            raise VocabularyError([f"{name}: unit count must be >= 0, got {value}"])
        out[name] = value
    return out
