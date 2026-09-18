"""The corpus, its manifest, and the seal that keeps the blind split blind.

The blind set is the only part of this project that can produce a prospective result, and
it is worth exactly as much as the discipline protecting it. So the protection is a
runtime guard, not a convention: :class:`SealedCorpus` raises
:class:`~assay.errors.SealError` when anything reads a blind document without
explicitly declaring blind intent. There is no way to "just peek" that does not show up in
a stack trace.

Splitting is by *source document*, not by task. Two tasks over the same filing are not
independent, and scoring one after fitting the other is leakage wearing a held-out label.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from assay.errors import SealError

TEXT_SUFFIXES = (".txt", ".md")


@dataclass(frozen=True)
class Document:
    name: str
    sha256: str
    n_bytes: int

    @classmethod
    def from_path(cls, path: Path, root: Path) -> "Document":
        data = path.read_bytes()
        return cls(
            name=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
            n_bytes=len(data),
        )


@dataclass(frozen=True)
class CorpusManifest:
    root: str
    documents: tuple[Document, ...]
    created_at: str

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.documents]

    def by_name(self, name: str) -> Document:
        for d in self.documents:
            if d.name == name:
                return d
        raise KeyError(name)

    def duplicates(self) -> list[str]:
        seen: dict[str, str] = {}
        dupes = []
        for d in self.documents:
            if d.sha256 in seen:
                dupes.append(f"{d.name} duplicates {seen[d.sha256]}")
            seen[d.sha256] = d.name
        return dupes

    def size_span(self) -> tuple[int, int]:
        sizes = [d.n_bytes for d in self.documents]
        return (min(sizes), max(sizes)) if sizes else (0, 0)

    def hash(self) -> str:
        payload = "|".join(f"{d.name}:{d.sha256}" for d in sorted(self.documents, key=lambda x: x.name))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "created_at": self.created_at,
            "manifest_sha256": self.hash(),
            "documents": [asdict(d) for d in self.documents],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CorpusManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            root=data["root"],
            documents=tuple(Document(**d) for d in data["documents"]),
            created_at=data["created_at"],
        )


def index_corpus(root: str | Path) -> CorpusManifest:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    docs = sorted(
        (
            Document.from_path(p, root)
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and p.name != "README.md"
        ),
        key=lambda d: d.name,
    )
    return CorpusManifest(
        root=str(root),
        documents=tuple(docs),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@dataclass(frozen=True)
class Seal:
    """Which documents are fit material and which are sealed.

    The seal hash covers both lists. Moving one document between them changes it, so a
    split cannot be quietly adjusted after results are seen.
    """

    fit: tuple[str, ...]
    blind: tuple[str, ...]
    manifest_sha256: str
    created_at: str
    seed: int

    @property
    def seal_sha256(self) -> str:
        payload = json.dumps(
            {"fit": sorted(self.fit), "blind": sorted(self.blind), "manifest": self.manifest_sha256},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_blind(self, name: str) -> bool:
        return name in self.blind

    def to_dict(self) -> dict[str, object]:
        return {
            "fit": list(self.fit),
            "blind": list(self.blind),
            "manifest_sha256": self.manifest_sha256,
            "created_at": self.created_at,
            "seed": self.seed,
            "seal_sha256": self.seal_sha256,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Seal":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        seal = cls(
            fit=tuple(data["fit"]),
            blind=tuple(data["blind"]),
            manifest_sha256=data["manifest_sha256"],
            created_at=data["created_at"],
            seed=int(data["seed"]),
        )
        if seal.seal_sha256 != data["seal_sha256"]:
            raise SealError(
                "seal hash does not match its contents -- the split has been edited since it was sealed"
            )
        return seal


def split_and_seal(
    manifest: CorpusManifest, *, blind_fraction: float = 0.25, seed: int = 1337
) -> Seal:
    if not 0.0 < blind_fraction < 1.0:
        raise ValueError("blind_fraction must be strictly between 0 and 1")
    names = sorted(manifest.names)
    if len(names) < 4:
        raise ValueError(f"need at least 4 documents to split, found {len(names)}")
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_blind = max(1, round(len(names) * blind_fraction))
    blind = sorted(shuffled[:n_blind])
    fit = sorted(shuffled[n_blind:])
    return Seal(
        fit=tuple(fit),
        blind=tuple(blind),
        manifest_sha256=manifest.hash(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
    )


class SealedCorpus:
    """Read access to the corpus, with the blind split behind an explicit flag."""

    def __init__(self, manifest: CorpusManifest, seal: Seal, root: str | Path | None = None):
        if seal.manifest_sha256 != manifest.hash():
            raise SealError("seal was created against a different corpus manifest")
        self.manifest = manifest
        self.seal = seal
        self.root = Path(root or manifest.root)

    def read(self, name: str, *, allow_blind: bool = False) -> bytes:
        if self.seal.is_blind(name) and not allow_blind:
            raise SealError(
                f"{name} is in the blind split; pass allow_blind=True from the blind runner only"
            )
        return (self.root / name).read_bytes()

    def size(self, name: str) -> int:
        return self.manifest.by_name(name).n_bytes

    def fit_documents(self) -> list[Document]:
        return [self.manifest.by_name(n) for n in self.seal.fit]

    def blind_documents(self, *, allow_blind: bool = False) -> list[Document]:
        if not allow_blind:
            raise SealError("listing the blind split requires allow_blind=True")
        return [self.manifest.by_name(n) for n in self.seal.blind]

    def assert_fit_only(self, names: Iterable[str]) -> None:
        leaked = sorted(n for n in names if self.seal.is_blind(n))
        if leaked:
            raise SealError(f"blind documents reached a fit-split operation: {leaked}")

    def bands(self, names: Iterable[str], n_bands: int = 3) -> list[list[str]]:
        """Split documents into size bands so the context ladder spans an order of magnitude."""
        ordered = sorted(names, key=self.size)
        if not ordered:
            return [[] for _ in range(n_bands)]
        per = max(1, len(ordered) // n_bands)
        bands = [ordered[i * per : (i + 1) * per] for i in range(n_bands)]
        bands[-1].extend(ordered[n_bands * per :])
        return bands
