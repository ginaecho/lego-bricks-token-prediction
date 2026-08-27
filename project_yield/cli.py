"""Command line for the prototype: ``python -m project_yield <command>``.

    serve      run the web app
    predict    forecast one use case from a description, a file or a folder
    batch      encode and rank a whole folder of written descriptions
    model      print the model card — which form each head chose, and its score
    portfolio  rank several use cases from a JSONL file
    encode     show what the encoder makes of a description, and stop there

Everything the web app can do is here too, because the first integration anyone
actually wants is not an API — it is a script that reads a spreadsheet of use
cases and writes a ranked list back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .casefiles import default_folder, encode_folder, read_folder
from .encode import heuristic_encode
from .predict import Predictor
from .report import forecast_card, model_card, portfolio_table
from .usecase import UseCase


def _encoder():
    from .azure import foundry_encoder_if_configured
    return foundry_encoder_if_configured() or heuristic_encode


def _usecases_from(path: str) -> List[UseCase]:
    """Use cases from a folder of written descriptions, or from JSON/JSONL.

    A folder of prose is the input people actually have, so it is accepted
    everywhere a file is.
    """
    if os.path.isdir(path):
        return encode_folder(path, _encoder())
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [UseCase.from_dict(json.loads(l)) for l in fh if l.strip()]
        data = json.load(fh)
    return [UseCase.from_dict(d) for d in
            (data if isinstance(data, list) else [data])]


def cmd_serve(args) -> int:
    from .app import serve
    serve(args.host, args.port, args.open)
    return 0


def cmd_encode(args) -> int:
    usecase = _encoder()(args.description, uid="NEW-001", title=args.title or "")
    print(json.dumps(usecase.to_dict(), indent=2))
    print(f"\n{usecase.notation()}", file=sys.stderr)
    return 0


def cmd_predict(args) -> int:
    predictor = Predictor.from_defaults(corpus_path=args.corpus)
    if args.file:
        cases = _usecases_from(args.file)
    elif args.description:
        cases = [_encoder()(args.description, uid="NEW-001",
                            title=args.title or "")]
    else:
        print("give --description or --file", file=sys.stderr)
        return 2
    for case in cases:
        predictor.index.add(case)
    for case in cases:
        forecast = predictor.forecast(case)
        if args.json:
            print(json.dumps(forecast.to_dict(), indent=2))
        else:
            print(forecast_card(forecast))
    return 0


def cmd_batch(args) -> int:
    """Encode a folder of descriptions and rank what comes out.

    Order matters and is not incidental: every use case is added to the library
    before any of them is priced, so a continuation can see its parent whether
    or not the parent happens to have been forecast yet.
    """
    folder = args.folder or default_folder()
    predictor = Predictor.from_defaults(corpus_path=args.corpus)
    cases = encode_folder(folder, _encoder())
    for case in cases:
        predictor.index.add(case)
    forecasts = [predictor.forecast(c) for c in cases]

    if args.json:
        print(json.dumps([f.to_dict() for f in forecasts], indent=2))
        return 0
    if args.cards:
        for forecast in forecasts:
            print(forecast_card(forecast))
        return 0

    print(f"Encoded {len(cases)} descriptions from {folder}")
    print(f"  encoder: {getattr(_encoder(), 'name', 'heuristic')}")
    print()
    codes = _role_codes(predictor.roster)
    print(f"  {'id':<7}{'use case':<28}{'tokens':>8}{'value':>10}{'win':>5}"
          f"{'days':>6}{'impact/yr':>12}  team")
    print("  " + "-" * 96)
    for f in forecasts:
        e, i = f.economics, f.impact
        team = " ".join(
            codes[r.role.slug] if r.is_certain else codes[r.role.slug].lower()
            for r in f.staffing) or "—"
        impact = f"{i.annual_net_benefit:,.0f}" if i.quoted else "—"
        print(f"  {f.usecase.id:<7}{f.usecase.title[:27]:<28}"
              f"{f.tokens.tokens:>8,.0f}"
              f"{e.contract_value:>10,.0f}"
              f"{e.win_probability:>5.0%}"
              f"{e.total_staff_days:>6,.0f}"
              f"{impact:>12}  {team}")
    print()
    print("  " + " · ".join(f"{c}={predictor.roster[s].name}"
                            for s, c in codes.items()))
    print("  UPPER where the role is certain — the note named it, the days")
    print("  were entered, or comparable work always uses it. lower where it")
    print("  is a likelihood. impact/yr is what the client gets, net.")
    print()
    print(portfolio_table(forecasts))
    return 0


def _role_codes(roster) -> "dict":
    """Short unique codes per role.

    Initials alone are not unique — Software engineer and Security expert both
    give SE, and a legend that maps two roles to one code is worse than no
    legend. Collisions take another letter from the first word until they part.
    """
    codes: dict = {}
    for role in roster:
        words = role.name.split()
        depth = 1
        while True:
            code = (words[0][:depth].capitalize()
                    + "".join(w[0].upper() for w in words[1:2]))
            if code not in codes.values() or depth > 6:
                break
            depth += 1
        codes[role.slug] = code
    return codes


def cmd_cases(args) -> int:
    folder = args.folder or default_folder()
    for case in read_folder(folder):
        print(f"{case.uid:<8}{case.filename:<44}"
              f"{'continues ' + case.parent if case.parent else '':<20}"
              f"{case.title}")
    return 0


def cmd_model(args) -> int:
    predictor = Predictor.from_defaults(corpus_path=args.corpus)
    print(model_card(predictor.heads, predictor.evaluate_holdout()))
    print()
    print(f"Roster ({predictor.roster.source})")
    print("=" * 74)
    for role in predictor.roster:
        fitted = role.days_outcome in predictor.heads
        note = "" if fitted else \
            f"  <- {predictor.heads.unfitted.get(role.days_outcome, 'no head')}"
        print(f"  {role.name:<24}${role.day_rate:>8,.0f}/day{note}")
    return 0


def cmd_portfolio(args) -> int:
    predictor = Predictor.from_defaults(corpus_path=args.corpus)
    cases = _usecases_from(args.file)
    for case in cases:
        predictor.index.add(case)
    print(portfolio_table([predictor.forecast(c) for c in cases]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project-yield", description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", help="engagement corpus JSONL "
                                         "(default: the committed one)")
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("serve", help="run the web prototype")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = subs.add_parser("encode", help="encode a description and stop")
    p.add_argument("description")
    p.add_argument("--title")
    p.set_defaults(func=cmd_encode)

    p = subs.add_parser("predict", help="forecast one or more use cases")
    p.add_argument("--description")
    p.add_argument("--title")
    p.add_argument("--file", help="JSON or JSONL of use cases, or a folder of "
                                  "written descriptions")
    p.add_argument("--json", action="store_true", help="emit JSON, not a card")
    p.set_defaults(func=cmd_predict)

    p = subs.add_parser("batch", help="encode and rank a folder of descriptions")
    p.add_argument("folder", nargs="?",
                   help="folder of .md/.txt descriptions "
                        "(default: examples/usecases)")
    p.add_argument("--json", action="store_true", help="emit JSON, not a table")
    p.add_argument("--cards", action="store_true",
                   help="print a full card per use case")
    p.set_defaults(func=cmd_batch)

    p = subs.add_parser("cases", help="list a folder of descriptions")
    p.add_argument("folder", nargs="?")
    p.set_defaults(func=cmd_cases)

    p = subs.add_parser("model", help="print the model card")
    p.set_defaults(func=cmd_model)

    p = subs.add_parser("portfolio", help="rank a file or folder of use cases")
    p.add_argument("file")
    p.set_defaults(func=cmd_portfolio)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
