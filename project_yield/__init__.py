"""Project Yield — the lego-brick predictor, widened from tokens to outcomes.

``token_yield`` answers *what will this cost to run?* from a fixed vocabulary of
task bricks, measured. This package keeps that encode/decode structure and
points it at the rest of a scoping decision:

* what a client like this has paid for work like this  (``contract_value``)
* how often a use case shaped like this succeeds       (``win_probability``)
* who it takes and for how long        (``architect_days`` / ``engineer_days``
                                        / ``pm_days`` / ``calendar_days``)

with lineage — continuations and siblings — as first-class features, because
most enterprise use cases are extensions of ones already delivered and pricing
them as greenfield is how both the estimate and the margin go wrong.

Start here::

    from project_yield import Predictor, heuristic_encode

    predictor = Predictor.from_defaults()
    usecase = heuristic_encode("Automate invoice intake ...")
    print(predictor.forecast(usecase).to_dict())

or run the prototype::

    python -m project_yield serve --open

The token head is fitted on real measurements. **The value and impact heads are
fitted on a synthetic corpus** that ships with the package so the machinery can
be demonstrated and tested; every forecast says so in its own warnings. See
``docs/product-prototype.md``.
"""

from .corpus import Engagement, load_engagements
from .economics import DEFAULT_RATES, Economics, Rates
from .encode import encode_prompt, heuristic_encode, parse_encoding
from .lineage import LineageFeatures, LineageIndex
from .multihead import Estimate, Head, MultiHeadModel
from .outcomes import ORDER as OUTCOME_ORDER, OUTCOMES, Outcome
from .predict import Forecast, Predictor, TokenEstimate
from .usecase import GOALS, INDUSTRIES, UseCase

__all__ = [
    "DEFAULT_RATES", "Economics", "Engagement", "Estimate", "Forecast",
    "GOALS", "Head", "INDUSTRIES", "LineageFeatures", "LineageIndex",
    "MultiHeadModel", "OUTCOMES", "OUTCOME_ORDER", "Outcome", "Predictor",
    "Rates", "TokenEstimate", "UseCase", "encode_prompt", "heuristic_encode",
    "load_engagements", "parse_encoding",
]

__version__ = "0.1.0"
