"""Where each piece of this belongs on Azure, expressed as code rather than a diagram.

The prototype runs with no cloud at all. This module is the set of seams where
Azure services plug in, one per thing the prototype fakes locally. Nothing here
is imported unless it is used, and nothing here is a hard dependency: the whole
package still installs and runs with no ``azure-*`` library present.

The short answer to "which Azure service"
-----------------------------------------
================  =========================================================
The encoder       **Azure AI Foundry.** A deployed chat model reads the use
                  case and returns the brick counts. This is the only part
                  that needs a model at inference time, and it is a single
                  chat completion — no agent framework, no orchestration.
                  :class:`FoundryEncoder`.
The corpus        **Microsoft Fabric.** Historical engagements live in a
                  Lakehouse table joined from delivery and CRM records.
                  :class:`FabricCorpus`, and the SQL in
                  ``docs/product-prototype.md``.
The app           **Azure Container Apps** (or App Service) with Entra ID in
                  front. :mod:`project_yield.app` is the container's contents.
Retraining        **Azure ML** — a scheduled job that refits the heads and
                  registers the result, plus a managed online endpoint if the
                  scoring has to be a service rather than a library.
                  :func:`init` / :func:`run` below are the scoring script.
Comparables       **Azure AI Search**, when the library outgrows a scan. The
                  ranking function is :meth:`LineageIndex.nearest`; the vector
                  is the brick count vector, which is nine numbers.
================  =========================================================

**Azure ML is the piece to defer.** The fit is a few hundred milliseconds over
a few hundred rows of a nine-brick feature vector; it does not need a compute
cluster, and running it as a library call inside the app is simpler and cheaper
until the corpus is large enough or the retraining cadence formal enough to
need a registry and a lineage trail. Fabric and Foundry earn their place on day
one, because the data and the model genuinely are not local.

Everything here degrades rather than fails. If Foundry is not configured the
app uses the keyword encoder and says so on every estimate; if Fabric is not
configured it uses the committed corpus and says *that* on every estimate.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .corpus import Engagement
from .encode import encode_prompt, parse_encoding
from .usecase import UseCase

# ── the encoder: Azure AI Foundry ────────────────────────────────────────

#: Environment variables the Foundry encoder reads. Endpoint and deployment
#: are required; the key is optional so that a managed identity or an
#: ``az login`` token can be used instead of a secret in the environment.
ENV_ENDPOINT = "AZURE_AI_FOUNDRY_ENDPOINT"
ENV_DEPLOYMENT = "AZURE_AI_FOUNDRY_DEPLOYMENT"
ENV_KEY = "AZURE_AI_FOUNDRY_KEY"
ENV_API_VERSION = "AZURE_AI_FOUNDRY_API_VERSION"

DEFAULT_API_VERSION = "2024-10-21"


class FoundryEncoder:
    """Encode a use case with a model deployed on Azure AI Foundry.

    Speaks the OpenAI-compatible chat completions API that a Foundry model
    deployment exposes, over :mod:`urllib`. That is on purpose: the call is one
    POST with one prompt, and taking an SDK dependency for it would make the
    package harder to install than the thing it is trying to demonstrate.

    ``temperature`` is zero because this is an encoder, not a writer. The same
    description must produce the same feature vector on Tuesday as it did on
    Monday, or the estimate moves without the use case having changed — which
    is the fastest way to lose a forecasting tool's credibility.
    """

    name = "azure-ai-foundry"

    def __init__(self, endpoint: Optional[str] = None,
                 deployment: Optional[str] = None, key: Optional[str] = None,
                 api_version: Optional[str] = None, timeout: float = 60.0,
                 token_provider: Optional[Callable[[], str]] = None) -> None:
        self.endpoint = (endpoint or os.environ.get(ENV_ENDPOINT, "")).rstrip("/")
        self.deployment = deployment or os.environ.get(ENV_DEPLOYMENT, "")
        self.key = key or os.environ.get(ENV_KEY, "")
        self.api_version = (api_version or os.environ.get(ENV_API_VERSION)
                            or DEFAULT_API_VERSION)
        self.timeout = timeout
        self.token_provider = token_provider
        if not self.endpoint or not self.deployment:
            raise ValueError(
                f"set {ENV_ENDPOINT} and {ENV_DEPLOYMENT} (or pass them) to "
                f"use the Foundry encoder")

    @property
    def url(self) -> str:
        return (f"{self.endpoint}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token_provider is not None:
            headers["Authorization"] = f"Bearer {self.token_provider()}"
        elif self.key:
            headers["api-key"] = self.key
        else:
            raise ValueError(
                f"no credential: set {ENV_KEY}, or pass a token_provider "
                f"(e.g. azure.identity.get_bearer_token_provider with a "
                f"managed identity)")
        return headers

    def complete(self, prompt: str) -> str:
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = urllib.request.Request(self.url, data=payload,
                                         headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(
                f"Foundry returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach Foundry: {exc.reason}") from None
        return data["choices"][0]["message"]["content"]

    def __call__(self, description: str, uid: str = "new",
                 title: str = "") -> UseCase:
        reply = self.complete(encode_prompt(description))
        return parse_encoding(reply, uid=uid, title=title,
                              description=description)


def foundry_encoder_if_configured() -> Optional[FoundryEncoder]:
    """A Foundry encoder when the environment names one, otherwise ``None``.

    Returning ``None`` rather than raising is the whole degradation strategy:
    the caller falls back to the keyword encoder, and the resulting use case
    carries ``encoder="heuristic"`` so every surface that renders it can say
    which one produced the number.
    """
    if not os.environ.get(ENV_ENDPOINT) or not os.environ.get(ENV_DEPLOYMENT):
        return None
    try:
        return FoundryEncoder()
    except ValueError:
        return None


# ── the corpus: Microsoft Fabric ─────────────────────────────────────────

ENV_FABRIC_SQL = "FABRIC_SQL_ENDPOINT"
ENV_FABRIC_DB = "FABRIC_DATABASE"
ENV_FABRIC_TABLE = "FABRIC_ENGAGEMENTS_TABLE"

#: The columns :class:`~project_yield.corpus.Engagement` needs. A Fabric view
#: presenting exactly these is the entire integration contract — see
#: ``docs/product-prototype.md`` for the query that builds it.
REQUIRED_COLUMNS = (
    "id", "title", "client", "industry", "goal", "counts", "context_bytes",
    "contract_value", "won", "role_days", "calendar_days", "parent_id",
    "sibling_ids", "started",
)


class FabricCorpus:
    """Read the engagement corpus from a Fabric Lakehouse SQL endpoint.

    Uses ``pyodbc`` if it is installed, because that is what the Fabric SQL
    endpoint speaks and it is the one dependency this genuinely cannot avoid.
    The alternative path — and the one to start with — is to export the view to
    JSONL on a schedule and point :func:`~project_yield.corpus.load_engagements`
    at the file. A prototype that reads a nightly extract answers the same
    question as one holding a live connection, and can be reviewed by someone
    without database credentials.
    """

    def __init__(self, sql_endpoint: Optional[str] = None,
                 database: Optional[str] = None,
                 table: Optional[str] = None) -> None:
        self.sql_endpoint = sql_endpoint or os.environ.get(ENV_FABRIC_SQL, "")
        self.database = database or os.environ.get(ENV_FABRIC_DB, "")
        self.table = (table or os.environ.get(ENV_FABRIC_TABLE)
                      or "dbo.vw_engagement_outcomes")
        if not self.sql_endpoint or not self.database:
            raise ValueError(f"set {ENV_FABRIC_SQL} and {ENV_FABRIC_DB}")

    def query(self) -> str:
        return f"SELECT {', '.join(REQUIRED_COLUMNS)} FROM {self.table}"

    def connection_string(self) -> str:
        return (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={self.sql_endpoint};Database={self.database};"
            "Authentication=ActiveDirectoryDefault;Encrypt=yes;"
            "TrustServerCertificate=no;"
        )

    def load(self) -> List[Engagement]:
        try:
            import pyodbc                                # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "reading Fabric directly needs pyodbc and the Microsoft ODBC "
                "driver. Export the view to JSONL and use "
                "corpus.load_engagements(path) instead — same columns, no "
                "driver.") from None
        rows: List[Engagement] = []
        with pyodbc.connect(self.connection_string()) as conn:
            cursor = conn.execute(self.query())
            columns = [c[0] for c in cursor.description]
            for record in cursor.fetchall():
                rows.append(_engagement_from_row(dict(zip(columns, record))))
        return rows


def _as_role_days(value: Any) -> Dict[str, float]:
    """``role_days`` as a dict, whether it arrived as JSON text or an object.

    A role missing from the dict means the engagement did not use it, which is
    the observation the presence heads are fitted on — so an absent key is data,
    not a null to be filled in.
    """
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else {}
    return {str(k): float(v) for k, v in (value or {}).items() if float(v) > 0}


def _engagement_from_row(row: Dict[str, Any]) -> Engagement:
    """One SQL row to one :class:`Engagement`.

    ``counts`` and ``sibling_ids`` arrive as JSON strings from a SQL endpoint
    and as lists/dicts from a JSONL export, so both are accepted. Anything else
    missing raises here, where the message can name the record.
    """
    counts = row.get("counts")
    if isinstance(counts, str):
        counts = json.loads(counts)
    siblings = row.get("sibling_ids") or []
    if isinstance(siblings, str):
        siblings = json.loads(siblings) if siblings.strip() else []
    from .usecase import normalise_counts
    return Engagement(
        id=str(row["id"]), title=str(row.get("title", "")),
        client=str(row.get("client", "")), industry=str(row["industry"]),
        goal=str(row["goal"]), counts=normalise_counts(counts or {}),
        context_bytes=int(row.get("context_bytes") or 0),
        contract_value=float(row["contract_value"]),
        won=bool(row["won"]),
        calendar_days=float(row["calendar_days"]),
        role_days=_as_role_days(row.get("role_days")),
        parent_id=(str(row["parent_id"]) if row.get("parent_id") else None),
        sibling_ids=tuple(str(s) for s in siblings),
        started=str(row.get("started", "")),
        provenance="fabric", held_out=bool(row.get("held_out", False)),
    )


# ── scoring: the Azure ML managed online endpoint entry point ────────────
#
# An Azure ML deployment calls init() once per worker and run() per request.
# They are three lines each because YieldApp already draws the line between
# "what this does" and "how it is being served" — which is the property that
# makes the same code a local server, a Function and an endpoint.

_APP = None


def init() -> None:
    """Azure ML calls this once when the worker starts."""
    global _APP
    from .app import YieldApp
    from .predict import Predictor

    corpus_path = os.environ.get("AZUREML_MODEL_DIR")
    if corpus_path:
        candidate = os.path.join(corpus_path, "engagements.jsonl")
        _APP = YieldApp(Predictor.from_defaults(
            corpus_path=candidate if os.path.exists(candidate) else None))
    else:
        _APP = YieldApp()


def run(raw_data: str) -> str:
    """Azure ML calls this per request, with the raw JSON body."""
    if _APP is None:
        init()
    body = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    try:
        return json.dumps(_APP.predict(body))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
