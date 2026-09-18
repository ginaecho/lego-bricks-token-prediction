"""Generate the demo corpus: a small fictional company, deterministically.

The corpus is committed, so the demo runs on a fresh clone with no network. This script
exists so the documents are reproducible and so their *structure* is deliberate rather
than filler: the segment tables really do sum to their stated totals, and the quarterly
figures really do differ between documents. That matters because the brick vocabulary
contains `Validate` ("check that the reported total equals the sum of the segments") and
`Reconcile` ("compare the same named item across two sources"). A corpus of lorem ipsum
would make those two bricks unaskable, and a campaign that cannot ask a brick cannot
price it.

Three size bands roughly an order of magnitude apart. That spread is not cosmetic: it is
what lets the fit separate "cost of reading bytes" from "cost of doing work". Without it,
document size and task count move together and neither coefficient is identified.

Regenerate with:  python -m demo.generate_corpus
"""

from __future__ import annotations

import random
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "corpus"

COMPANY = "Northwind Freight Holdings plc"
SEGMENTS = ("Road", "Rail", "Air", "Warehousing", "Brokerage")
QUARTERS = ("Q1 FY26", "Q2 FY26", "Q3 FY26", "Q4 FY26")


def _segment_table(rng: random.Random, quarter: str) -> tuple[str, int]:
    """A segment table whose rows really do sum to the total it prints."""
    rows = []
    total = 0
    for name in SEGMENTS:
        value = rng.randrange(1_200, 9_800) * 10
        total += value
        rows.append((name, value))
    body = [
        f"Segment revenue - {quarter} (GBP thousands)",
        "",
        f"  {'Segment':<14}{'Revenue':>12}{'Share':>10}",
        f"  {'-' * 36}",
    ]
    for name, value in rows:
        body.append(f"  {name:<14}{value:>12,}{value / total * 100:>9.1f}%")
    body.append(f"  {'-' * 36}")
    body.append(f"  {'Total':<14}{total:>12,}{100.0:>9.1f}%")
    return "\n".join(body), total


def _narrative(rng: random.Random, quarter: str, total: int, paragraphs: int) -> str:
    openers = (
        "Group revenue for the period reflects continued mix shift toward contracted freight.",
        "Volumes held broadly flat while yield improved on long-haul lanes.",
        "The period included one additional trading week compared with the prior year.",
        "Fuel surcharge recovery lagged the underlying index by approximately one month.",
    )
    middles = (
        "Management notes that the Brokerage segment remains the most sensitive to spot"
        " rates, and that a sustained move in either direction would be visible within one"
        " quarter.",
        "Warehousing utilisation averaged 84% across the estate, with the two northern"
        " sites operating above 90% for the majority of the period.",
        "The Rail segment absorbed a one-off access charge which is not expected to recur.",
        "Air freight capacity was constrained in the first six weeks and normalised"
        " thereafter.",
        "Working capital movements were driven principally by the timing of customer"
        " receipts.",
    )
    closers = (
        f"Total segment revenue for {quarter} was GBP {total:,} thousand.",
        "The directors consider the going concern basis to remain appropriate.",
        "No adjusting events occurred between the reporting date and the date of approval.",
    )
    out = [f"{COMPANY} - {quarter} management commentary", ""]
    for i in range(paragraphs):
        out.append(rng.choice(openers))
        out.append(rng.choice(middles))
        if i % 3 == 2:
            out.append(rng.choice(closers))
        out.append("")
    return "\n".join(out)


def _filing(rng: random.Random, quarter: str, paragraphs: int) -> str:
    table, total = _segment_table(rng, quarter)
    parts = [
        COMPANY,
        f"Interim report - {quarter}",
        "=" * 60,
        "",
        table,
        "",
        _narrative(rng, quarter, total, paragraphs),
        "",
        f"Headcount at period end: {rng.randrange(2_100, 2_900):,}",
        f"Operating margin: {rng.uniform(3.5, 9.5):.1f}%",
        "",
        "Notes to the interim figures",
        "-" * 40,
    ]
    for i in range(1, max(2, paragraphs // 2)):
        parts.append(
            f"{i}. "
            + rng.choice(
                (
                    "Revenue is recognised on delivery of the freight service.",
                    "Segment results are reported to the chief operating decision maker"
                    " monthly.",
                    "Right-of-use assets are depreciated over the shorter of lease term"
                    " and useful life.",
                    "Trade receivables are stated net of expected credit losses.",
                    "Fuel derivatives are designated as cash flow hedges where effective.",
                )
            )
        )
    return "\n".join(parts) + "\n"


def _invoice(rng: random.Random, n: int) -> str:
    lines = []
    total = 0
    for i in range(rng.randrange(4, 9)):
        qty = rng.randrange(1, 40)
        rate = rng.randrange(45, 320)
        amount = qty * rate
        total += amount
        lines.append(
            f"  {i + 1:<4}{'Lane ' + str(rng.randrange(100, 999)):<16}"
            f"{qty:>6}{rate:>10,}{amount:>12,}"
        )
    return "\n".join(
        [
            f"INVOICE NW-{n:05d}",
            f"Supplier: {COMPANY}",
            f"Due date: 2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
            "",
            f"  {'#':<4}{'Description':<16}{'Qty':>6}{'Rate':>10}{'Amount':>12}",
            f"  {'-' * 48}",
            *lines,
            f"  {'-' * 48}",
            f"  {'Total':<36}{total:>12,}",
            "",
        ]
    )


def _sow(rng: random.Random, n: int) -> str:
    return "\n".join(
        [
            f"STATEMENT OF WORK SOW-{n:03d}",
            f"Between {COMPANY} and the supplier named below.",
            "",
            "Acceptance criteria",
            "-" * 30,
            "1. Weekly manifest reconciliation delivered by 09:00 each Monday.",
            "2. Exception rate below 0.5% measured over a rolling four-week window.",
            "3. All corrections applied within two business days of notification.",
            "",
            "Commercial terms",
            "-" * 30,
            f"Term: {rng.randrange(12, 37)} months.",
            f"Indexation: CPI capped at {rng.uniform(2.0, 5.0):.1f}%.",
            "",
        ]
    )


def build() -> list[Path]:
    CORPUS.mkdir(parents=True, exist_ok=True)
    for stale in CORPUS.glob("*.txt"):
        stale.unlink()

    rng = random.Random(20260906)
    written: list[Path] = []

    # Small band (~1 KB): short operational documents.
    for i in range(1, 5):
        name = f"small-{i:02d}-{'invoice' if i % 2 else 'sow'}.txt"
        text = _invoice(rng, 1000 + i) if i % 2 else _sow(rng, i)
        (CORPUS / name).write_text(text, encoding="utf-8")
        written.append(CORPUS / name)

    # Medium band (~6 KB): the quarterly interims.
    for i, quarter in enumerate(QUARTERS, start=1):
        name = f"medium-{i:02d}-interim.txt"
        (CORPUS / name).write_text(_filing(rng, quarter, paragraphs=9), encoding="utf-8")
        written.append(CORPUS / name)

    # Large band (~20 KB): the annual documents.
    for i in range(1, 5):
        name = f"large-{i:02d}-annual.txt"
        (CORPUS / name).write_text(
            _filing(rng, f"FY2{i + 2} full year", paragraphs=42), encoding="utf-8"
        )
        written.append(CORPUS / name)

    return written


def main() -> int:
    written = build()
    print(f"wrote {len(written)} documents to {CORPUS}")
    for p in sorted(written, key=lambda x: x.stat().st_size):
        print(f"  {p.name:<28}{p.stat().st_size:>8,} bytes")
    sizes = [p.stat().st_size for p in written]
    print(
        f"\nsize span: {min(sizes):,} - {max(sizes):,} bytes "
        f"({max(sizes) / min(sizes):.0f}x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
