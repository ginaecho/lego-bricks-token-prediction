"""Reading a folder of written use case descriptions.

The natural input to this tool is not a JSON object — it is the paragraph
somebody already wrote in a scoping note, an email or a statement of work. This
module turns a directory of those into :class:`~project_yield.usecase.UseCase`
objects, so a portfolio can be estimated from the documents that already exist
rather than from a form somebody has to fill in twice.

The format is deliberately almost nothing:

* one plain text or Markdown file per use case;
* an optional first-line ``# Heading``, used as the title;
* an optional ``manifest.jsonl`` beside them, declaring *only* the lineage —
  which use case continues which.

Lineage is the one thing a description cannot carry on its own. "Follow-on to
the pipeline we delivered for Northwind" is obvious to a reader and not
recoverable by an encoder, so it is stated as data. Everything else — bricks,
industry, goal, scope, volume — is read out of the prose by the encoder, which
is the whole point.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .encode import ASSUMED_INDUSTRY, heuristic_encode
from .usecase import UseCase

#: Extensions treated as use case descriptions. Anything else in the folder —
#: a README, the manifest itself — is skipped rather than encoded.
SUFFIXES = (".md", ".txt", ".markdown")
MANIFEST = "manifest.jsonl"
SKIP = {"readme.md", "readme.txt", "index.md"}

_H1 = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*$")


@dataclass(frozen=True)
class CaseFile:
    """One description on disk, before it has been encoded."""

    path: str
    uid: str
    title: str
    text: str
    parent: Optional[str] = None
    siblings: Sequence[str] = ()

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


def read_case(path: str, uid: Optional[str] = None) -> CaseFile:
    """Read one description file, taking a leading heading as the title."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    first, _, _ = text.partition("\n")
    match = _H1.match(first)
    title = match.group(1) if match else first[:70]
    stem = os.path.splitext(os.path.basename(path))[0]
    return CaseFile(path=path, uid=uid or stem, title=title, text=text)


def load_manifest(directory: str) -> Dict[str, dict]:
    """Read ``manifest.jsonl`` if it is there. Absent is not an error."""
    path = os.path.join(directory, MANIFEST)
    if not os.path.exists(path):
        return {}
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            entry = json.loads(line)
            if "file" not in entry:
                raise ValueError(f"{MANIFEST}:{lineno}: every entry needs a "
                                 f"\"file\" key")
            out[str(entry["file"])] = entry
    return out


def read_folder(directory: str) -> List[CaseFile]:
    """Every description in a folder, in filename order.

    Filename order is deliberate rather than incidental: a continuation has to
    be encoded after the use case it continues, or its parent is not in the
    library yet and it gets priced as greenfield. Numbering the files is how
    that ordering is expressed, and the manifest names the parent by file so a
    rename does not silently break the link.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a folder")
    manifest = load_manifest(directory)
    names = sorted(n for n in os.listdir(directory)
                   if n.lower().endswith(SUFFIXES) and n.lower() not in SKIP)
    if not names:
        raise ValueError(f"no {' or '.join(SUFFIXES)} files in {directory}")

    ids = {n: str(manifest.get(n, {}).get("id")
                  or os.path.splitext(n)[0]) for n in names}
    unknown = [key for key in manifest if key not in ids]
    if unknown:
        raise ValueError(f"{MANIFEST} names files that are not in the folder: "
                         + ", ".join(sorted(unknown)))

    cases: List[CaseFile] = []
    for name in names:
        entry = manifest.get(name, {})
        case = read_case(os.path.join(directory, name), uid=ids[name])
        parent_file = entry.get("continues")
        if parent_file and parent_file not in ids:
            raise ValueError(f"{name}: continues {parent_file!r}, which is not "
                             f"in the folder")
        cases.append(CaseFile(
            path=case.path, uid=case.uid,
            title=str(entry.get("title") or case.title), text=case.text,
            parent=ids[parent_file] if parent_file else None,
            siblings=tuple(ids[f] for f in entry.get("alongside", [])
                           if f in ids),
        ))
    return cases


def encode_folder(directory: str,
                  encoder: Optional[Callable[..., UseCase]] = None
                  ) -> List[UseCase]:
    """Read and encode a whole folder, wiring up the declared lineage.

    Where a description does not name its industry, the lineage does: a
    continuation is for the same client in the same sector as the thing it
    continues, and so is a use case delivered alongside one. Inheriting it
    beats defaulting to the reference category, and the substitution is
    recorded as an assumption rather than made silently — a phase-two note that
    never repeats the client's sector is the normal case, not a defect in the
    note.
    """
    if encoder is None:
        encoder = heuristic_encode

    by_id: Dict[str, UseCase] = {}
    out: List[UseCase] = []
    for case in read_folder(directory):
        usecase = encoder(case.text, uid=case.uid, title=case.title)
        usecase.parent_id = case.parent
        usecase.sibling_ids = list(case.siblings)

        guessed = [a for a in usecase.assumptions
                   if a.startswith(ASSUMED_INDUSTRY)]
        if guessed:
            relative = next((by_id[r] for r in (case.parent, *case.siblings)
                             if r in by_id), None)
            if relative is not None:
                usecase.industry = relative.industry
                usecase.assumptions = [
                    a for a in usecase.assumptions if a not in guessed]
                usecase.assumptions.append(
                    f"The description does not name an industry, so it was "
                    f"taken from {relative.id} "
                    f"({relative.industry.replace('_', ' ')}), which this use "
                    f"case is linked to.")
        by_id[usecase.id] = usecase
        out.append(usecase)
    return out


def default_folder() -> str:
    """The example descriptions that ship with the package."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "usecases")
