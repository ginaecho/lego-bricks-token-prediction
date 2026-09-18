"""The ``ty`` command line.

Every subcommand prints its evidence class, and every subcommand that could report a
conclusion exits non-zero when that conclusion is Stop or when the run was incomplete. A
pipeline must not be able to read a dead project as a success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay import __version__
from assay import paths
from assay.audit import audit, load_reference
from assay.corpus import (
    CorpusManifest,
    Seal,
    SealedCorpus,
    index_corpus,
    split_and_seal,
)
from assay.decompose.keyword import KeywordEncoder
from assay.evidence import EvidenceClass
from assay.invoice import make_invoice
from assay.pipeline import (
    Priced,
    fit_pipeline,
    load_runs,
    simulate_campaign,
)
from assay.pricing import PROVISIONAL
from assay.quote import append_quote, make_quote
from assay.verdict import DEFAULT_POLICY, evaluate, save_policy
from assay.vocabulary import load_vocabulary

DEFAULT_VOCAB = paths.BRICKS
DEFAULT_REFERENCE = paths.REFERENCE_RUNS


def _emit(payload: dict, as_json: bool, rendered: str) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) if as_json else rendered)


def cmd_audit(args: argparse.Namespace) -> int:
    report = audit(load_reference(args.data))
    _emit(report.as_dict(), args.json, report.render())
    return 0


def cmd_vocab(args: argparse.Namespace) -> int:
    vocab = load_vocabulary(args.vocab)
    if args.json:
        print(json.dumps({"version": vocab.version, "bricks": vocab.names,
                          "primary": list(vocab.primary)}, indent=2))
    else:
        print(vocab.render_for_prompt())
        print(f"OK: {len(vocab.bricks)} bricks, primary = {', '.join(vocab.primary)}")
    return 0


def cmd_corpus_index(args: argparse.Namespace) -> int:
    manifest = index_corpus(args.dir)
    if args.out:
        manifest.save(args.out)
    lo, hi = manifest.size_span()
    dupes = manifest.duplicates()
    _emit(
        manifest.to_dict(),
        args.json,
        "\n".join(
            [f"{len(manifest.documents)} documents, {lo:,}-{hi:,} bytes",
             f"manifest sha256: {manifest.hash()[:16]}",
             *(f"  DUPLICATE: {d}" for d in dupes)]
        ),
    )
    return 1 if dupes else 0


def cmd_corpus_split(args: argparse.Namespace) -> int:
    manifest = CorpusManifest.load(args.manifest)
    seal = split_and_seal(manifest, blind_fraction=args.blind_fraction, seed=args.seed)
    seal.save(args.out)
    print(
        f"sealed {len(seal.blind)} of {len(manifest.documents)} documents\n"
        f"  seal sha256: {seal.seal_sha256[:16]}\n"
        f"  the blind names are in {args.out}; the loader refuses to read them elsewhere"
    )
    return 0


def cmd_decompose(args: argparse.Namespace) -> int:
    vocab = load_vocabulary(args.vocab)
    result = KeywordEncoder(vocab).decompose(args.request)
    payload = result.as_dict()
    payload["evidence_class"] = EvidenceClass.PIPELINE_ONLY.value
    _emit(
        payload,
        args.json,
        "\n".join(
            [
                f"request: {args.request}",
                "  " + (", ".join(f"{k} x{v}" for k, v in result.units.items() if v)
                        or "no brick applies (out of scope)"),
                f"  source: {result.source}  confidence: {result.confidence:.2f}",
                f"  {result.rationale}",
            ]
        ),
    )
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    observations = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    evidence = EvidenceClass(args.evidence)
    result = evaluate(observations, evidence_class=evidence)
    _emit(result.as_dict(), args.json, result.render())
    return result.exit_code


def cmd_policy(args: argparse.Namespace) -> int:
    save_policy(args.out)
    print(f"wrote {len(DEFAULT_POLICY)} gates to {args.out}")
    return 0


def _load_corpus(args: argparse.Namespace) -> SealedCorpus:
    """Index the corpus and attach the seal, creating one only when told to.

    A seal generated on the fly is not a pre-registration -- it is a split chosen at the
    same moment as the analysis. It is allowed here for a dry run, but it has to be asked
    for, and it says so.
    """
    manifest = index_corpus(args.dir)
    if args.seal and Path(args.seal).exists():
        seal = Seal.load(args.seal)
    elif args.new_seal:
        seal = split_and_seal(manifest, blind_fraction=0.25, seed=args.seed)
        target = Path(args.seal) if args.seal else paths.SPLIT_SEAL
        seal.save(target)
        print(
            f"NOTE: no seal was supplied, so one was created at {target}. A split chosen "
            "at analysis time is not a pre-registration; commit this before it counts."
        )
    else:
        raise SystemExit(
            "no seal found. Run `ty corpus-split` first, or pass --new-seal to create "
            "one now (which makes the run a dry run, not evidence)."
        )
    return SealedCorpus(manifest, seal, root=args.dir)


def cmd_campaign(args: argparse.Namespace) -> int:
    vocab = load_vocabulary(args.vocab)
    corpus = _load_corpus(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"

    plan, report = simulate_campaign(
        vocab, corpus, runs_path, pricing=PROVISIONAL, mode=args.mode, seed=args.seed
    )
    payload = {
        "evidence_class": EvidenceClass.PIPELINE_ONLY.value,
        "plan": plan.summary(),
        "report": report.as_dict(),
        "runs_path": str(runs_path),
    }
    rendered = "\n".join(
        [
            "CAMPAIGN  (evidence class: pipeline_only -- SIMULATED, decides nothing)",
            "",
            f"  mode        : {args.mode}"
            + ("   [negative control: no brick term exists]" if args.mode == "null" else ""),
            f"  probes      : {plan.n_dispatches}",
            *(f"    {t:<12}: {n}" for t, n in sorted(plan.counts.items())),
            f"  dispatched  : {report.dispatched}   skipped: {report.skipped}   "
            f"excluded: {report.excluded}",
            f"  runs        : {runs_path}",
            "",
            "  These runs came from the simulator. They exercise the code path and prove",
            "  nothing about any real model. Only feasibility evidence may move a gate.",
        ]
    )
    _emit(payload, args.json, rendered)
    return 1 if report.excluded else 0


def cmd_fit(args: argparse.Namespace) -> int:
    vocab = load_vocabulary(args.vocab)
    runs = load_runs(args.runs, PROVISIONAL)
    fitted = fit_pipeline(
        runs, vocab, bar_pct=args.bar_pct, level=args.level, iters=args.iters, seed=args.seed
    )
    paths = fitted.save(args.out_dir)
    _emit(
        fitted.as_dict(),
        args.json,
        fitted.render() + "\n\n  wrote " + ", ".join(str(p) for p in paths.values()),
    )
    return 0


def _parse_units(pairs: list[str]) -> dict[str, int]:
    units: dict[str, int] = {}
    for pair in pairs:
        name, _, count = pair.partition("=")
        if not _:
            raise SystemExit(f"--brick expects Name=Count, got {pair!r}")
        units[name.strip()] = units.get(name.strip(), 0) + int(count)
    return units


def cmd_quote(args: argparse.Namespace) -> int:
    priced = Priced.load(args.fit_dir)
    quote = make_quote(
        args.request_id,
        _parse_units(args.brick),
        args.bytes,
        model=priced.model,
        conformal=priced.conformal,
        pricing=PROVISIONAL,
        show_per_brick=priced.show_per_brick,
        evidence_class=priced.evidence_class,
    )
    if args.out:
        append_quote(args.out, quote)
    lines = [
        f"QUOTE  (evidence class: {priced.evidence_class.value})",
        "",
        "  " + quote.summary().replace("\n", "\n"),
    ]
    for flag in quote.outlier_flags:
        lines.append(f"  ! {flag}")
    if not priced.show_per_brick:
        lines.append(
            "  per-brick detail withheld: selection chose "
            f"{priced.chosen_form}, so nothing established a price per brick"
        )
    if PROVISIONAL.provisional:
        lines.append(f"  dollars are provisional: {PROVISIONAL.source}")
    _emit(quote.as_dict(), args.json, "\n".join(lines))
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    # Imported here so that the rest of the CLI keeps working on a machine where
    # no browser or loopback socket is available.
    from .studio.server import serve

    httpd = serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"assay studio on {url}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


def cmd_invoice(args: argparse.Namespace) -> int:
    priced = Priced.load(args.fit_dir)
    runs = {r.run_id: r for r in load_runs(args.runs, PROVISIONAL)}
    run = runs.get(args.run_id)
    if run is None:
        raise SystemExit(f"no run {args.run_id!r} in {args.runs}")
    quote = make_quote(
        run.run_id,
        run.units,
        run.context_bytes,
        model=priced.model,
        conformal=priced.conformal,
        pricing=PROVISIONAL,
        show_per_brick=priced.show_per_brick,
        evidence_class=priced.evidence_class,
    )
    invoice = make_invoice(
        run, quote, priced.model, boot=priced.boot, show_per_brick=priced.show_per_brick
    )
    rendered = invoice.render(PROVISIONAL.units_to_usd(1.0))
    for note in invoice.notes:
        rendered += f"\n  note: {note}"
    _emit(invoice.as_dict(), args.json, rendered)
    return 0 if invoice.reconciles else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ty", description=__doc__)
    parser.add_argument("--version", action="version", version=f"assay {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="replay the reference table beside its variable-portion rescore")
    p.add_argument("--data", type=Path, default=DEFAULT_REFERENCE)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("vocab", help="validate and print the brick vocabulary")
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.set_defaults(func=cmd_vocab)

    p = sub.add_parser("corpus-index", help="hash every document into a manifest")
    p.add_argument("--dir", type=Path, default=Path("corpus"))
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_corpus_index)

    p = sub.add_parser("corpus-split", help="seal a fit/blind split by source document")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, default=paths.SPLIT_SEAL)
    p.add_argument("--blind-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(func=cmd_corpus_split)

    p = sub.add_parser("decompose", help="encode a request into bricks (keyword control)")
    p.add_argument("request")
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("verdict", help="apply the frozen policy to observed gate values")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument(
        "--evidence", default="pipeline_only",
        choices=[c.value for c in EvidenceClass],
    )
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("policy", help="write the frozen verdict policy with its hash")
    p.add_argument("--out", type=Path, default=paths.VERDICT_POLICY)
    p.set_defaults(func=cmd_policy)

    p = sub.add_parser(
        "campaign", help="design and run a fit campaign against the simulator (pipeline_only)"
    )
    p.add_argument("--dir", type=Path, default=Path("corpus"))
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--out-dir", type=Path, default=Path("work"))
    p.add_argument("--seal", type=Path, default=paths.SPLIT_SEAL)
    p.add_argument(
        "--new-seal", action="store_true",
        help="create a split seal now instead of loading one (makes this a dry run)",
    )
    p.add_argument(
        "--mode", default="brick", choices=["brick", "null"],
        help="'null' is the negative control: the simulator prices no bricks at all",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(func=cmd_campaign)

    p = sub.add_parser("fit", help="select a form, fit it, and calibrate its interval")
    p.add_argument("--runs", type=Path, default=Path("work/runs.jsonl"))
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--out-dir", type=Path, default=Path("work"))
    p.add_argument("--bar-pct", type=float, default=15.0)
    p.add_argument("--level", type=float, default=0.90)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("quote", help="price a request from a fitted model")
    p.add_argument("request_id")
    p.add_argument(
        "--brick", action="append", default=[], metavar="NAME=COUNT",
        help="repeatable, e.g. --brick Extract=2 --brick Validate=1",
    )
    p.add_argument("--bytes", type=int, required=True, help="context size in bytes")
    p.add_argument("--fit-dir", type=Path, default=Path("work"))
    p.add_argument("--out", type=Path, help="append the quote to this ledger")
    p.set_defaults(func=cmd_quote)

    p = sub.add_parser("invoice", help="reconcile one recorded run against its quote")
    p.add_argument("run_id")
    p.add_argument("--runs", type=Path, default=Path("work/runs.jsonl"))
    p.add_argument("--fit-dir", type=Path, default=Path("work"))
    p.set_defaults(func=cmd_invoice)

    p = sub.add_parser("studio", help="run the whole argument in a browser")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true",
                   help="do not try to open a browser window")
    p.set_defaults(func=cmd_studio)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
