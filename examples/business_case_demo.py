"""Summarize business-case evidence without inventing measurements or prices.

Run with:

    python -m examples.business_case_demo

Proof-result JSONL records use the ``WorkOutcome`` fields: ``case_id`` and
``accepted`` are required; ``input_tokens``, ``output_tokens``,
``total_tokens``, and ``cost_center`` are optional. Monetary output is shown
only when measured results and an explicit billing rate are both supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from token_yield.business_cases import BusinessCase, catalog_summary, load_cases
from token_yield.economics import Pricing, WorkOutcome, chargeback, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "experiments" / "business_cases" / "cases.jsonl"
DEFAULT_RESULTS = ROOT / "experiments" / "business_cases" / "proof_results.jsonl"


def _token_count(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def partition_proof_records(
    records: Iterable[Mapping[str, Any]],
) -> Tuple[List[WorkOutcome], List[WorkOutcome]]:
    """Separate measured outcomes from verdicts that have no token evidence."""

    measured = []
    acceptance_only = []
    for record in records:
        case_id = record.get("case_id")
        accepted = record.get("accepted")
        cost_center = record.get("cost_center", "unassigned")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(accepted, bool):
            raise ValueError(f"{case_id}: accepted must be a boolean")
        if not isinstance(cost_center, str) or not cost_center:
            raise ValueError(f"{case_id}: cost_center must be a non-empty string")
        outcome = WorkOutcome(
            case_id=case_id,
            accepted=accepted,
            input_tokens=_token_count(record, "input_tokens"),
            output_tokens=_token_count(record, "output_tokens"),
            total_tokens=_token_count(record, "total_tokens"),
            cost_center=cost_center,
        )
        proof_type = record.get("proof_type")
        provenance = record.get("provenance")
        has_measurement = outcome.tokens() > 0
        if proof_type == "acceptance-only":
            if has_measurement:
                raise ValueError(
                    f"{case_id}: acceptance-only proof cannot contain token usage"
                )
            target = acceptance_only
        elif proof_type == "measured" or provenance == "measured":
            if not has_measurement:
                raise ValueError(f"{case_id}: measured proof requires token usage")
            target = measured
        elif has_measurement:
            raise ValueError(
                f"{case_id}: token usage requires measured provenance"
            )
        else:
            target = acceptance_only
        target.append(outcome)
    return measured, acceptance_only


def load_proof_results(
    path: Path,
) -> Tuple[List[WorkOutcome], List[WorkOutcome]]:
    """Load proof JSONL and retain source line numbers in validation errors."""

    measured = []
    acceptance_only = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                line_measured, line_acceptance_only = partition_proof_records([record])
                measured.extend(line_measured)
                acceptance_only.extend(line_acceptance_only)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return measured, acceptance_only


def _print_catalog(cases: Sequence[BusinessCase]) -> None:
    facts = catalog_summary(cases)
    print("CATALOG")
    print(f"  cases: {facts['cases']} "
          f"({facts['base_cases']} base, {facts['composite_cases']} composite)")
    print(f"  primitives covered: {facts['primitives_covered']}")
    missing = ", ".join(facts["missing_primitives"]) or "none"
    print(f"  missing primitives: {missing}")
    print(f"  domains: {', '.join(facts['domains']) or 'none'}")
    for case in cases:
        print(f"    {case.case_id}: {case.notation()} [{case.business_domain}]")


def _print_proofs(
    measured: Sequence[WorkOutcome],
    acceptance_only: Sequence[WorkOutcome],
    pricing: Optional[Pricing],
) -> None:
    print("\nPROOF RESULTS")
    print(f"  acceptance-only (missing token measurements): "
          f"{len(acceptance_only)}")
    for outcome in acceptance_only:
        verdict = "accepted" if outcome.accepted else "rejected"
        print(f"    {outcome.case_id}: {verdict}")

    print(f"  fully measured outcomes: {len(measured)}")
    for outcome in measured:
        verdict = "accepted" if outcome.accepted else "rejected"
        print(f"    {outcome.case_id}: {verdict}, {outcome.tokens():,} tokens")

    accepted = [outcome for outcome in measured if outcome.accepted]
    if not measured:
        print("  Economics not computed: no positive token measurements.")
        return
    if not accepted:
        print("  Economics not computed: no measured outcome was accepted.")
        return

    economics = summarize(measured, pricing or Pricing())
    print("\nMEASURED UNIT ECONOMICS")
    print(f"  acceptance rate: {economics.acceptance_rate:.1%}")
    print(f"  attempts per accepted outcome: "
          f"{economics.attempts_per_accepted_outcome:.2f}")
    print(f"  p50/p90/p95/p99 tokens: {economics.p50_tokens:,.0f} / "
          f"{economics.p90_tokens:,.0f} / {economics.p95_tokens:,.0f} / "
          f"{economics.p99_tokens:,.0f}")
    print(f"  p95 tail reserve: {economics.tail_reserve_tokens:,.0f} tokens")
    print(f"  tokens per accepted outcome, including failed attempts: "
          f"{economics.tokens_per_accepted_outcome:,.0f}")

    if pricing is None:
        print("  Cost and chargeback not computed: pass actual billing rates.")
        return
    if (
        pricing.blended_per_million == 0
        and any(
            outcome.total_tokens > 0
            and outcome.input_tokens + outcome.output_tokens == 0
            for outcome in measured
        )
    ):
        print("  Cost and chargeback not computed: split billing rates require "
              "input/output token measurements.")
        return

    print(f"  accepted-work cost: "
          f"${economics.cost_per_accepted_outcome:,.6f} per accepted outcome")
    print("\nCHARGEBACK")
    for row in chargeback(measured, pricing):
        per_accepted = row["cost_per_accepted_outcome"]
        per_accepted_text = (
            f"${per_accepted:,.6f}" if per_accepted != float("inf") else "n/a"
        )
        print(f"  {row['cost_center']}: ${row['actual_cost']:,.6f} actual; "
              f"{int(row['accepted_outcomes'])} accepted; "
              f"{per_accepted_text} per accepted outcome")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    prices = parser.add_mutually_exclusive_group()
    prices.add_argument(
        "--blended-price",
        type=float,
        metavar="USD_PER_MILLION",
        help="actual blended billing rate per million tokens",
    )
    prices.add_argument(
        "--split-price",
        nargs=2,
        type=float,
        metavar=("INPUT_USD_PER_MILLION", "OUTPUT_USD_PER_MILLION"),
        help="actual input and output billing rates per million tokens",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.catalog.is_file():
        parser.error(f"catalog not found: {args.catalog}")
    if args.blended_price is not None and args.blended_price < 0:
        parser.error("billing rates must be non-negative")
    if args.split_price is not None and any(rate < 0 for rate in args.split_price):
        parser.error("billing rates must be non-negative")

    pricing = None
    if args.blended_price is not None:
        pricing = Pricing(
            input_per_million=args.blended_price,
            output_per_million=args.blended_price,
            blended_per_million=args.blended_price,
        )
    elif args.split_price is not None:
        pricing = Pricing(
            input_per_million=args.split_price[0],
            output_per_million=args.split_price[1],
        )

    _print_catalog(load_cases(str(args.catalog)))
    if not args.results.is_file():
        print(f"\nPROOF RESULTS\n  none found at {args.results}")
        return 0
    measured, acceptance_only = load_proof_results(args.results)
    _print_proofs(measured, acceptance_only, pricing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
