"""Mining real repositories for the *shape* of the work they actually contain.

Probes tell you what a unit of work costs. They cannot tell you what work a
project is made of — how much of it is bug-fixing, how much is test-writing,
how big a typical change is. That distribution has to come from somewhere real,
and a repository's history is the cheapest honest source of it.

So this module does two things:

* **Classify.** Turn a commit into a ``kind`` plus the signals that measure how
  much of it there is. The rules are deliberately transparent and each carries
  the evidence that fired it — a classifier you cannot inspect is another
  asserted constant wearing a different hat.
* **Report coverage.** Cross the mined distribution against the kinds you have
  actually *measured*. What comes back is a measurement backlog ordered by how
  much of the real work it would unlock — which is the honest answer to "what
  should I probe next?"

What mining does **not** give you is cost. These commits were written by people,
not agents, and no token count attaches to them. Mining supplies the *what and
how much*; only probes and production runs supply the *how expensive*.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, Optional

TEST_PAT = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$")
DOC_PAT = re.compile(r"\.(md|rst|txt|adoc)$|(^|/)docs?/")
CODE_PAT = re.compile(r"\.(py|js|ts|go|rs|java|rb|c|h|cpp|cs)$")

FIX_PAT = re.compile(r"^(fix|bug|hotfix|patch)\b|fix(es|ed)?[:(\s]|\bbugfix\b", re.I)
FEAT_PAT = re.compile(r"^(feat|feature|add)\b|\badds?\b|\bimplement", re.I)
REFACTOR_PAT = re.compile(r"^(refactor|chore|cleanup|style|perf)\b|\brefactor", re.I)
DOC_MSG_PAT = re.compile(r"^doc(s)?\b|\bdocument", re.I)
TEST_MSG_PAT = re.compile(r"^test(s)?\b|\badd(ed|s)? tests?\b", re.I)


@dataclass(frozen=True)
class ChangeFeatures:
    """What we can see about one commit without judging it."""

    message: str = ""
    files: int = 0
    insertions: int = 0
    deletions: int = 0
    test_files: int = 0
    doc_files: int = 0
    code_files: int = 0
    new_files: int = 0

    @property
    def lines_changed(self) -> int:
        return self.insertions + self.deletions

    @property
    def churn_ratio(self) -> float:
        """Deletions per insertion — near 1.0 suggests rework, not new ground."""
        return self.deletions / self.insertions if self.insertions else 0.0


@dataclass(frozen=True)
class Classification:
    kind: str
    confidence: float
    why: str


def classify(f: ChangeFeatures) -> Classification:
    """Map a change to a task kind, in transparent priority order.

    Confidence is a plain statement of how much the rule leans on: a commit
    made entirely of test files is a far surer ``test_write`` than one guessed
    from a verb in its subject line.
    """
    if f.files == 0:
        return Classification("unknown", 0.0, "no files changed")

    # 1. composition of the change — the strongest evidence available
    if f.test_files == f.files:
        return Classification("test_write", 0.95, "every changed file is a test")
    if f.doc_files == f.files:
        return Classification("docs", 0.95, "every changed file is documentation")

    # 2. the subject line, when it follows a convention
    msg = f.message.strip()
    if TEST_MSG_PAT.search(msg) and f.test_files:
        return Classification("test_write", 0.8, "test-shaped subject, tests touched")
    if DOC_MSG_PAT.search(msg) and f.doc_files:
        return Classification("docs", 0.8, "doc-shaped subject, docs touched")
    if FIX_PAT.search(msg):
        return Classification("bug_fix", 0.75, "subject line reads as a fix")
    if FEAT_PAT.search(msg):
        return Classification("feature", 0.7, "subject line reads as an addition")
    if REFACTOR_PAT.search(msg):
        return Classification("refactor", 0.7, "subject line reads as a refactor")

    # 3. shape of the diff, when the words gave nothing
    if f.new_files and f.new_files == f.files:
        return Classification("feature", 0.5, "all files are new")
    if f.insertions and 0.7 <= f.churn_ratio <= 1.4 and not f.new_files:
        return Classification("refactor", 0.45,
                              "deletions roughly match insertions, nothing new")
    return Classification("code_change", 0.3, "no rule matched strongly")


@dataclass(frozen=True)
class MinedTask:
    """One real unit of work, classified, with its scope measured."""

    kind: str
    confidence: float
    why: str
    features: ChangeFeatures
    repo: str = ""
    sha: str = ""

    @property
    def signals(self) -> dict:
        f = self.features
        return {
            "scope": max(f.files, 1),
            "files": max(f.files, 1),
            "lines": max(f.lines_changed, 1),
            # a rough stand-in for the bytes an agent would have to read/write;
            # the byte signal is what predicted comprehension cost best
            "bytes": max(f.lines_changed * 40, 1),
        }


def _run(args: list, cwd: str) -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True).stdout


def mine_repo(path: str, limit: int = 200, repo: str = "") -> list:
    """Read a repository's history into classified, measured tasks."""
    if not os.path.isdir(os.path.join(path, ".git")):
        return []
    repo = repo or os.path.basename(path.rstrip("/"))
    raw = _run(["git", "log", f"-n{limit}", "--numstat", "--no-merges",
                "--format=__C__%H%x09%s"], path)

    tasks, sha, msg, rows = [], None, "", []

    def flush():
        if sha is None:
            return
        ins = sum(r[0] for r in rows)
        dele = sum(r[1] for r in rows)
        paths = [r[2] for r in rows]
        feats = ChangeFeatures(
            message=msg,
            files=len(paths),
            insertions=ins,
            deletions=dele,
            test_files=sum(1 for p in paths if TEST_PAT.search(p)),
            doc_files=sum(1 for p in paths if DOC_PAT.search(p)),
            code_files=sum(1 for p in paths if CODE_PAT.search(p)),
            new_files=sum(1 for r in rows if r[1] == 0 and r[0] > 0),
        )
        c = classify(feats)
        tasks.append(MinedTask(c.kind, c.confidence, c.why, feats, repo, sha))

    for line in raw.splitlines():
        if line.startswith("__C__"):
            flush()
            head = line[5:].split("\t", 1)
            sha, msg, rows = head[0], (head[1] if len(head) > 1 else ""), []
        elif line.strip():
            parts = line.split("\t")
            if len(parts) >= 3:
                a = 0 if parts[0] == "-" else int(parts[0])
                d = 0 if parts[1] == "-" else int(parts[1])
                rows.append((a, d, parts[2]))
    flush()
    return tasks


def distribution(tasks: Iterable) -> dict:
    """How the mined work splits by kind: count, share, and median size."""
    by_kind: dict = {}
    ts = list(tasks)
    for t in ts:
        by_kind.setdefault(t.kind, []).append(t)
    total = len(ts) or 1
    out = {}
    for kind, group in by_kind.items():
        lines = sorted(g.features.lines_changed for g in group)
        files = sorted(g.features.files for g in group)
        out[kind] = {
            "count": len(group),
            "share": len(group) / total,
            "median_lines": lines[len(lines) // 2],
            "median_files": files[len(files) // 2],
            "mean_confidence": sum(g.confidence for g in group) / len(group),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["count"]))


@dataclass
class CoverageReport:
    """Which of the real work we can actually price, and what to measure next."""

    measured_kinds: tuple = ()
    rows: list = field(default_factory=list)   # (kind, share, is_measured)

    @property
    def covered_share(self) -> float:
        return sum(s for _, s, m in self.rows if m)

    @property
    def backlog(self) -> list:
        """Unmeasured kinds, largest share of real work first."""
        return [(k, s) for k, s, m in sorted(self.rows, key=lambda r: -r[1]) if not m]

    def summary(self) -> str:
        lines = [f"Coverage: {self.covered_share:.0%} of mined work is a kind "
                 f"we have measured", "-" * 68]
        for kind, share, measured in sorted(self.rows, key=lambda r: -r[1]):
            mark = "measured" if measured else "NOT MEASURED"
            lines.append(f"  {kind:<16}{share:>7.0%}   {mark}")
        if self.backlog:
            lines.append("")
            lines.append("Measurement backlog, by how much real work it unlocks:")
            for kind, share in self.backlog:
                lines.append(f"  probe '{kind}' → would cover a further {share:.0%}")
        return "\n".join(lines)


def coverage(tasks: Iterable, measured_kinds: Iterable) -> CoverageReport:
    """Cross the mined distribution against the kinds actually measured."""
    dist = distribution(tasks)
    measured = tuple(sorted(measured_kinds))
    rows = [(kind, d["share"], kind in measured) for kind, d in dist.items()]
    return CoverageReport(measured, rows)
