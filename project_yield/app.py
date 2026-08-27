"""The prototype itself: a local web app a PM can open and use.

Deliberately built on :mod:`http.server` and nothing else. The point of a
prototype is that the person who needs to judge it can run it — ``python -m
project_yield.app`` and a browser — without an Azure subscription, a container
registry or a pip install that fails behind a corporate proxy. Everything it
does is a function call away from being a request handler somewhere else, and
:mod:`project_yield.azure` shows exactly which functions those are.

Endpoints
---------
``GET  /``            the single page
``GET  /api/meta``    vocabularies and the use-case library, for the form
``POST /api/encode``  description -> encoded use case (agent, or keyword)
``POST /api/predict`` use case -> the full forecast
``GET  /api/model``   how each head was chosen and how well it scores

Not a production server: single-threaded, no authentication, bound to localhost
by default. Putting it in front of anyone else means putting it behind something
that does those things — Azure Container Apps with Entra ID in front, which is
the deployment described in ``docs/product-prototype.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .encode import heuristic_encode
from .outcomes import ORDER, OUTCOMES
from .predict import Predictor
from .usecase import GOALS, INDUSTRIES, UseCase
from token_yield.tasks import ORDER as BRICKS, PRIMITIVES

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 256 * 1024

#: A brick, inline, so the page has a tab icon without a network request.
FAVICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#2f5bd7"/>'
    b'<rect x="7" y="13" width="18" height="11" rx="2" fill="#fff"/>'
    b'<rect x="10" y="8" width="5" height="6" rx="1.5" fill="#fff"/>'
    b'<rect x="17" y="8" width="5" height="6" rx="1.5" fill="#fff"/></svg>')


def _encoder():
    """The agent encoder if one is configured, else the keyword fallback.

    Resolved per request rather than at import, so setting the Foundry
    environment variables does not require a restart.
    """
    from .azure import foundry_encoder_if_configured
    return foundry_encoder_if_configured() or heuristic_encode


class YieldApp:
    """The application logic, independent of how it is being served.

    Every method here takes and returns plain dictionaries. That is what makes
    the same code an Azure Functions handler, an Azure ML scoring script or a
    local web app without being rewritten — see :mod:`project_yield.azure`.
    """

    def __init__(self, predictor: Optional[Predictor] = None) -> None:
        self.predictor = predictor or Predictor.from_defaults()
        self._counter = 0

    # -- endpoints -------------------------------------------------------

    def meta(self) -> Dict[str, Any]:
        library = sorted(self.predictor.index.all(), key=lambda u: u.id)
        return {
            "industries": list(INDUSTRIES),
            "goals": list(GOALS),
            "bricks": [{"slug": s, "name": PRIMITIVES[s].name,
                        "blurb": PRIMITIVES[s].blurb} for s in BRICKS],
            "outcomes": [{"slug": s, "name": OUTCOMES[s].name,
                          "unit": OUTCOMES[s].unit,
                          "question": OUTCOMES[s].question} for s in ORDER],
            "library": [{"id": u.id, "title": u.title,
                         "industry": u.industry} for u in library],
            "encoder": getattr(_encoder(), "name", "heuristic"),
        }

    def encode(self, body: Dict[str, Any]) -> Dict[str, Any]:
        description = str(body.get("description", "")).strip()
        if not description:
            raise ValueError("give a description to encode")
        usecase = _encoder()(description, uid=self._next_id(),
                             title=str(body.get("title", "")).strip())
        out = usecase.to_dict()
        out["notation"] = usecase.notation()
        return out

    def predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        usecase = self._usecase_from(body)
        forecast = self.predictor.forecast(usecase)
        # A use case scoped now joins the library, so the next one can be
        # linked to it as a parent or a sibling in the same session.
        self.predictor.index.add(usecase)
        return forecast.to_dict()

    def model(self) -> Dict[str, Any]:
        heads = self.predictor.heads
        holdout = self.predictor.evaluate_holdout()
        return {
            "corpus": len(self.predictor.corpus),
            "fitted_on": heads.n,
            "heads": [{
                "slug": slug, "name": OUTCOMES[slug].name,
                "form": h.form, "link": h.outcome.link.name,
                "metric": "brier" if h.outcome.binary else "mape",
                "cross_validated": round(h.loo_score, 4),
                "baseline": round(h.baseline_score, 4),
                "skill": round(h.skill, 4),
                "beats_baseline": h.beats_baseline,
                "held_out": (round(holdout[slug][0], 4)
                             if slug in holdout else None),
                "held_out_n": holdout.get(slug, (0, 0))[1],
                "candidate_scores": {k: round(v, 4) for k, v in h.scores.items()},
            } for slug, h in ((s, heads.heads[s]) for s in ORDER
                              if s in heads.heads)],
            "token_model": {
                "form": self.predictor.token_model.form,
                "equation": self.predictor.token_model.equation(),
                "cross_validated_mape": round(
                    self.predictor.token_model.loo_mape, 4),
                "fitted_on_runs": self.predictor.token_model.n,
                "provenance": "measured agent runs",
            },
        }

    # -- helpers ---------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        return f"NEW-{self._counter:03d}"

    def _usecase_from(self, body: Dict[str, Any]) -> UseCase:
        counts = body.get("counts") or {}
        if not any(int(v or 0) > 0 for v in counts.values()):
            raise ValueError(
                "no task bricks given, and no description to read them from "
                "— write what the use case does, or set at least one brick "
                "count under 'Refine the scope'")
        parent = body.get("parent_id") or None
        return UseCase(
            id=str(body.get("id") or self._next_id()),
            title=str(body.get("title") or "").strip() or "Untitled use case",
            description=str(body.get("description") or ""),
            industry=str(body.get("industry") or INDUSTRIES[0]),
            goal=str(body.get("goal") or GOALS[0]),
            counts=counts,
            context_bytes=int(body.get("context_bytes") or 0),
            monthly_runs=int(body.get("monthly_runs") or 0),
            parent_id=str(parent) if parent else None,
            sibling_ids=[str(s) for s in (body.get("sibling_ids") or []) if s],
            encoder=str(body.get("encoder") or "manual"),
        )


def make_handler(app: YieldApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ProjectYield/0.1"

        # -- plumbing ----------------------------------------------------

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # The page is entirely self-contained; saying so closes off the
            # whole class of "it worked locally and then loaded a CDN" bugs.
            # data: images are allowed for inline SVG icons and nothing else —
            # no host is reachable from the page under any directive.
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'unsafe-inline'; "
                             "script-src 'unsafe-inline'; img-src 'self' data:")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: Dict[str, Any]) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def log_message(self, fmt: str, *args) -> None:
            if os.environ.get("PROJECT_YIELD_VERBOSE"):
                super().log_message(fmt, *args)

        # -- routes ------------------------------------------------------

        def do_GET(self) -> None:                       # noqa: N802
            route = self.path.split("?")[0]
            try:
                if route in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                        self._send(200, fh.read(), "text/html; charset=utf-8")
                elif route == "/favicon.svg":
                    self._send(200, FAVICON, "image/svg+xml")
                elif route == "/favicon.ico":
                    self._send(204, b"", "image/svg+xml")
                elif route == "/api/meta":
                    self._json(200, app.meta())
                elif route == "/api/model":
                    self._json(200, app.model())
                else:
                    self._json(404, {"error": f"no route {route}"})
            except Exception as exc:                    # noqa: BLE001
                self._json(500, {"error": str(exc)})

        def do_POST(self) -> None:                      # noqa: N802
            route = self.path.split("?")[0]
            try:
                body = self._body()
                if route == "/api/encode":
                    self._json(200, app.encode(body))
                elif route == "/api/predict":
                    self._json(200, app.predict(body))
                else:
                    self._json(404, {"error": f"no route {route}"})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:                    # noqa: BLE001
                self._json(500, {"error": str(exc)})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = False,
          predictor: Optional[Predictor] = None) -> None:
    app = YieldApp(predictor)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"Project Yield prototype on {url}")
    print(f"  corpus  : {len(app.predictor.corpus)} engagements "
          f"({app.predictor.heads.n} fitted on)")
    print(f"  encoder : {app.meta()['encoder']}")
    print("  the value and impact heads are fitted on SYNTHETIC data — see "
          "docs/product-prototype.md")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true",
                        help="open a browser once the server is up")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.open)


if __name__ == "__main__":
    main()
