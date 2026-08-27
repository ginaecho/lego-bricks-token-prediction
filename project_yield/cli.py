"""Command line for the prototype: ``python -m project_yield <command>``.

    serve      run the web app
    predict    forecast one use case from a description or a JSON file
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
import sys
from typing import List, Optional

from .encode import heuristic_encode
from .predict import Predictor
from .report import forecast_card, model_card, portfolio_table
from .usecase import UseCase


def _encoder():
    from .azure import foundry_encoder_if_configured
    return foundry_encoder_if_configured() or heuristic_encode


def _usecases_from(path: str) -> List[UseCase]:
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


def cmd_model(args) -> int:
    predictor = Predictor.from_defaults(corpus_path=args.corpus)
    print(model_card(predictor.heads, predictor.evaluate_holdout()))
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
    p.add_argument("--file", help="JSON or JSONL of use cases")
    p.add_argument("--json", action="store_true", help="emit JSON, not a card")
    p.set_defaults(func=cmd_predict)

    p = subs.add_parser("model", help="print the model card")
    p.set_defaults(func=cmd_model)

    p = subs.add_parser("portfolio", help="rank a file of use cases")
    p.add_argument("file")
    p.set_defaults(func=cmd_portfolio)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
