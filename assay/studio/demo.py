"""Run the whole loop end to end, twice, and record enough for a browser to show it.

Two arms, one pipeline. In the `brick` arm the simulated runtime really does price
bricks; in the `null` arm there is no brick term at all. Both arms are put through the
identical sequence -- seal, campaign, ladder, fit, blind, batching, verdict -- and the
interesting output is not either arm on its own but the pair: the method should find what
was planted and refuse what was not.

Everything here is `pipeline_only`. It cannot move a gate and says so in every payload.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from assay import paths
from assay.audit import audit, load_reference
from assay.batching import analyse, collect_bundles
from assay.blind import commit_quotes, score_blind
from assay.conformal import calibrate
from assay.corpus import SealedCorpus, index_corpus, split_and_seal
from assay.costmodel import BootModel, CostModel
from assay.evidence import EvidenceClass
from assay.features import FORMS
from assay.harness.campaign import design_batching, design_blind, design_campaign
from assay.harness.mock_adapter import MockAdapter, MockMode
from assay.harness.runner import run_campaign
from assay.invoice import make_invoice
from assay.pipeline import boot_estimate
from assay.pricing import PROVISIONAL
from assay.quote import make_quote
from assay.schemas import Tier, read_runs
from assay.select import cross_validate, select_form
from assay.verdict import evaluate
from assay.vocabulary import load_vocabulary

RUNTIME_HASH = "studio-simulated-0001"
BAR_PCT = 15.0

STAGES: tuple[tuple[str, str, str], ...] = (
    ("seal", "Seal the baseplate",
     "Hash every document and lock a blind split before anything is measured."),
    ("design", "Draw the build plan",
     "Lay out probes so brick count and document size move independently."),
    ("measure", "Measure every brick",
     "Dispatch each probe and record what it cost. Append-only, resumable."),
    ("ladder", "Climb the ladder",
     "Score every model form by grouped cross-validation, simplest first."),
    ("fit", "Fit and calibrate",
     "Fit the chosen form; calibrate its interval on out-of-fold rows."),
    ("blind", "Open the sealed box",
     "Quote 24 unseen tasks before running them, then run them."),
    ("batching", "Test the batching lever",
     "Same work, separate versus bundled, to price the one big saving."),
    ("verdict", "Read the verdict",
     "Apply the pre-registered gates. No number is chosen after the fact."),
)

ARMS = ("brick", "null")

ARM_BLURB = {
    "brick": "The simulated runtime prices bricks. The pipeline should find them.",
    "null": "The simulated runtime prices no bricks at all. The pipeline must refuse "
            "to invent them.",
}


@dataclass
class Stage:
    key: str
    title: str
    detail: str
    status: str = "pending"  # pending | running | done | failed
    progress: float = 0.0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
            "progress": round(self.progress, 4),
            "note": self.note,
        }


@dataclass
class Arm:
    key: str
    blurb: str
    stages: list[Stage] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = [Stage(k, t, d) for k, t, d in STAGES]

    def stage(self, key: str) -> Stage:
        return next(s for s in self.stages if s.key == key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "blurb": self.blurb,
            "stages": [s.as_dict() for s in self.stages],
            "result": self.result,
        }


class DemoState:
    """Shared, mutated by the worker thread and read by HTTP handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"  # idle | running | done | failed
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.arms: dict[str, Arm] = {a: Arm(a, ARM_BLURB[a]) for a in ARMS}
        self.reference: dict[str, Any] = {}
        self.vocabulary: dict[str, Any] = {}
        self.corpus: dict[str, Any] = {}
        self.log: list[dict[str, Any]] = []
        self.seq = 0
        self.priced: dict[str, Any] = {}
        """Live fitted models, kept out of the JSON snapshot so the quote page can
        price a request the visitor types rather than replaying a canned one."""

    def claim(self) -> bool:
        """Reserve the state for a fresh run. False if one is already running."""
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.error = None
            self.started_at = time.time()
            self.finished_at = None
            self.arms = {a: Arm(a, ARM_BLURB[a]) for a in ARMS}
            self.priced = {}
            self.log = []
            self.seq += 1
            return True

    def price(self, arm: str, model, conformal, *, show_per_brick: bool, boot: float) -> None:
        with self._lock:
            self.priced[arm] = {
                "model": model, "conformal": conformal,
                "show_per_brick": show_per_brick, "boot": boot,
            }

    # -- mutation (always under the lock) ------------------------------------------

    def say(self, arm: str | None, message: str) -> None:
        with self._lock:
            self.seq += 1
            self.log.append({"seq": self.seq, "arm": arm, "message": message,
                             "t": round(time.time(), 3)})
            del self.log[:-400]

    def set_stage(self, arm: str, key: str, *, status: str | None = None,
                  progress: float | None = None, note: str | None = None) -> None:
        with self._lock:
            s = self.arms[arm].stage(key)
            if status is not None:
                s.status = status
            if progress is not None:
                s.progress = progress
            if note is not None:
                s.note = note
            self.seq += 1

    def put(self, arm: str, **kwargs: Any) -> None:
        with self._lock:
            self.arms[arm].result.update(kwargs)
            self.seq += 1

    def set_top(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.seq += 1

    # -- reading -------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            done = sum(
                1 for a in self.arms.values() for s in a.stages if s.status == "done"
            )
            total = len(self.arms) * len(STAGES)
            return {
                "status": self.status,
                "error": self.error,
                "seq": self.seq,
                "evidence_class": EvidenceClass.PIPELINE_ONLY.value,
                "overall_progress": round(done / total, 4) if total else 0.0,
                "elapsed_s": round(
                    (self.finished_at or time.time()) - self.started_at, 2
                ) if self.started_at else 0.0,
                "arms": {k: a.as_dict() for k, a in self.arms.items()},
                "reference": self.reference,
                "vocabulary": self.vocabulary,
                "corpus": self.corpus,
                "log": list(self.log[-120:]),
            }


# -- gate inputs the demo can honestly compute ---------------------------------------


def noise_floor_pct(runs) -> float | None:
    """Spread between replicates of the same probe.

    Below this, a brick's marginal cost is indistinguishable from the runtime being in a
    different mood, so it is the floor on what any of this can resolve.
    """
    groups: dict[str, list[float]] = {}
    for r in runs:
        if not r.accepted or r.tier is not Tier.BASE:
            continue
        stem = r.probe_id.rsplit("-r", 1)[0]
        groups.setdefault(stem, []).append(r.billable_units)
    spreads = [
        abs(max(v) - min(v)) / statistics.fmean(v) * 100.0
        for v in groups.values()
        if len(v) >= 2 and statistics.fmean(v) > 0
    ]
    return round(statistics.median(spreads), 4) if spreads else None


def drift_pct(runs) -> float | None:
    """How far the empty task moved between the opening and closing brackets."""
    def med(position: str) -> float | None:
        vals = [
            r.billable_units for r in runs
            if r.accepted and r.tier is Tier.NULL and f"null-{position}-" in r.probe_id
        ]
        return statistics.median(vals) if vals else None

    a, b = med("open"), med("close")
    if a is None or b is None or a == 0:
        return None
    return round(abs(b - a) / a * 100.0, 4)


# -- the run --------------------------------------------------------------------------


def _corpus(state: DemoState, work: Path) -> SealedCorpus:
    manifest = index_corpus(paths.DEMO / "corpus")
    seal = split_and_seal(manifest, blind_fraction=0.25, seed=99)
    seal.save(work / "split-seal.json")
    lo, hi = manifest.size_span()
    state.set_top(corpus={
        "n_documents": len(manifest.documents),
        "manifest_sha256": manifest.hash(),
        "seal_sha256": seal.seal_sha256,
        "size_lo": lo,
        "size_hi": hi,
        "span_x": round(hi / lo, 1) if lo else 0,
        "fit": list(seal.fit),
        "blind": list(seal.blind),
        "documents": [
            {"name": d.name, "bytes": d.n_bytes, "blind": seal.is_blind(d.name),
             "band": d.name.split("-")[0]}
            for d in sorted(manifest.documents, key=lambda x: x.n_bytes)
        ],
    })
    return SealedCorpus(manifest, seal, root=paths.DEMO / "corpus")


def _run_arm(state: DemoState, arm: str, vocab, corpus: SealedCorpus, work: Path) -> None:
    mode = MockMode(arm)
    adapter = MockAdapter(mode, seed=1337)
    out = work / f"runs_{arm}.jsonl"
    if out.exists():
        out.unlink()

    # -- seal --------------------------------------------------------------------
    state.set_stage(arm, "seal", status="running")
    state.say(arm, f"corpus sealed: {len(corpus.seal.fit)} fit, "
                   f"{len(corpus.seal.blind)} blind, split hashed before measurement")
    state.set_stage(arm, "seal", status="done", progress=1.0,
                    note=f"{len(corpus.seal.blind)} documents sealed away")

    # -- design ------------------------------------------------------------------
    state.set_stage(arm, "design", status="running")
    plan = design_campaign(vocab, corpus, campaign_id=f"studio_{arm}", seed=1337)
    blind_probes = design_blind(vocab, corpus, n_tasks=24, seed=20260904)
    batch_probes = design_batching(vocab, corpus, seed=4242)
    total_probes = len(plan.probes) + len(blind_probes) + len(batch_probes)
    state.put(arm, plan={"counts": plan.counts, "n_fit": plan.n_dispatches,
                         "n_blind": len(blind_probes), "n_batching": len(batch_probes),
                         "n_total": total_probes})
    state.say(arm, f"{total_probes} probes planned "
                   f"({plan.n_dispatches} fit, {len(blind_probes)} blind, "
                   f"{len(batch_probes)} batching)")
    state.set_stage(arm, "design", status="done", progress=1.0,
                    note=f"{total_probes} probes")

    # -- measure -----------------------------------------------------------------
    state.set_stage(arm, "measure", status="running")
    seen = {"n": 0}

    def tick(_probe) -> None:
        seen["n"] += 1
        state.set_stage(arm, "measure", progress=seen["n"] / plan.n_dispatches,
                        note=f"{seen['n']}/{plan.n_dispatches} probes measured")

    report = run_campaign(
        plan.probes, adapter, out, campaign_id=plan.campaign_id, pricing=PROVISIONAL,
        corpus=corpus, runtime_hash=RUNTIME_HASH, expected_model=adapter.model,
        on_dispatch=tick,
    )
    runs = read_runs(out, PROVISIONAL)
    boot = boot_estimate(runs)
    accepted = [r for r in runs if r.accepted]
    exclusion_pct = (
        (len(runs) - len(accepted)) / len(runs) * 100.0 if runs else 0.0
    )
    state.put(arm, measure={
        "dispatched": report.dispatched,
        "excluded": report.excluded,
        "exclusion_pct": round(exclusion_pct, 3),
        "boot": round(boot, 1),
        "median_task": round(statistics.median(
            [r.billable_units for r in accepted if any(r.units.values())] or [boot]
        ), 1),
        "noise_floor_pct": noise_floor_pct(runs),
        "drift_pct": drift_pct(runs),
    })
    m = state.arms[arm].result["measure"]
    toll_share = boot / m["median_task"] * 100.0 if m["median_task"] else 0.0
    state.put(arm, measure={**m, "toll_share_pct": round(toll_share, 1)})
    state.say(arm, f"start-up toll {boot:,.0f} units -- "
                   f"{toll_share:.0f}% of a median task")
    state.set_stage(arm, "measure", status="done", progress=1.0,
                    note=f"{report.dispatched} runs, {report.excluded} excluded")

    # -- ladder ------------------------------------------------------------------
    state.set_stage(arm, "ladder", status="running")
    rungs: list[dict[str, Any]] = []
    keys = list(FORMS)
    for i, key in enumerate(keys):
        form = FORMS[key]
        try:
            cv = cross_validate(runs, key, vocab, boot)
            row = cv.as_dict()
            rungs.append({
                "form": key, "role": form.role, "label": form.description,
                "n_params": row["n_params"],
                "vwape_pct": row["vwape_pct"],
                "mape_total_pct": row["mape_total_pct"],
                "failed_folds": row["failed_folds"],
            })
        except Exception as exc:  # noqa: BLE001 - a rung that will not score is a result
            rungs.append({"form": key, "role": form.role, "label": form.description,
                          "n_params": None, "vwape_pct": None,
                          "mape_total_pct": None, "error": str(exc)[:160]})
        state.set_stage(arm, "ladder", progress=(i + 1) / len(keys),
                        note=f"{i + 1}/{len(keys)} forms scored")

    selection = select_form(runs, vocab, boot, bar_pct=BAR_PCT, iters=600)
    lift = selection.decoder_lift()
    state.put(arm, ladder={
        "rungs": rungs,
        "chosen": selection.chosen,
        "chose_a_decoder": selection.bricks_helped,
        "best_baseline": selection.best_baseline,
        "bar_pct": selection.bar_pct,
        "lift": lift.as_dict() if lift else None,
        "notes": list(selection.notes),
        "tests": [t.as_dict() for t in selection.tests],
    })
    verdict_word = "found the bricks" if selection.bricks_helped else "refused to invent them"
    state.say(arm, f"ladder chose {selection.chosen} -- {verdict_word}")
    state.set_stage(arm, "ladder", status="done", progress=1.0,
                    note=f"chose {selection.chosen}")

    # -- fit ---------------------------------------------------------------------
    state.set_stage(arm, "fit", status="running")
    model = CostModel.fit(runs, selection.chosen, vocab, boot)
    held = selection.results[selection.chosen]
    conformal = calibrate(held.actual, held.predicted, level=0.90, floor=boot)
    show = selection.bricks_helped
    ident = model.identifiability
    state.put(arm, fit={
        "form": model.form,
        "intercept": round(model.intercept, 1),
        "bytes_coef": round(model.bytes_coef, 4),
        "marginals": {k: round(v, 1) for k, v in model.marginals.items()} if show else {},
        "show_per_brick": show,
        "n_fitted": model.n_fitted,
        "conformal": conformal.as_dict(),
        "identifiable": None if ident is None else bool(ident.ok),
        "identifiability": None if ident is None else ident.as_dict(),
    })
    state.set_stage(arm, "fit", status="done", progress=1.0,
                    note=f"{'per-brick prices earned' if show else 'per-brick prices withheld'}")
    state.price(arm, model, conformal, show_per_brick=show, boot=boot)

    # -- blind -------------------------------------------------------------------
    state.set_stage(arm, "blind", status="running")
    baselines = {"boot": BootModel(boot)}
    quotes = commit_quotes(
        blind_probes, model=model, conformal=conformal, pricing=PROVISIONAL,
        baselines=baselines,
        context_bytes={p.probe_id: sum(corpus.size(n) for n in p.context_files)
                       for p in blind_probes},
    )
    blind_out = work / f"blind_{arm}.jsonl"
    if blind_out.exists():
        blind_out.unlink()
    bseen = {"n": 0}

    def btick(_p) -> None:
        bseen["n"] += 1
        state.set_stage(arm, "blind", progress=bseen["n"] / len(blind_probes),
                        note=f"{bseen['n']}/{len(blind_probes)} sealed tasks run")

    run_campaign(
        blind_probes, adapter, blind_out, campaign_id=f"studio_{arm}_blind",
        pricing=PROVISIONAL, corpus=corpus, runtime_hash=RUNTIME_HASH,
        expected_model=adapter.model, allow_blind=True, on_dispatch=btick,
    )
    blind_runs = read_runs(blind_out, PROVISIONAL)
    blind_report = score_blind(quotes, blind_runs, boot=boot, baselines=baselines,
                               n_planned=len(blind_probes))
    bd = blind_report.as_dict() if hasattr(blind_report, "as_dict") else {}
    state.put(arm, blind={
        "n_planned": blind_report.n_planned,
        "n_scored": blind_report.n_scored,
        "n_excluded": blind_report.n_excluded,
        "voided": blind_report.voided,
        "mape_total_pct": round(blind_report.scorecard.mape_total, 3),
        "vwape_pct": round(blind_report.scorecard.vwape, 3),
        "lift_pct": round(blind_report.lift_pct, 3),
        "best_baseline": blind_report.best_baseline,
        "coverage_hits": blind_report.coverage_hits,
        "coverage_n": blind_report.coverage_n,
        "median_half_width_pct": round(blind_report.median_relative_half_width, 2),
        "raw": bd,
    })
    state.say(arm, f"blind: {blind_report.scorecard.mape_total:.1f}% total error, "
                   f"{blind_report.scorecard.vwape:.1f}% on the variable portion")
    state.set_stage(arm, "blind", status="done", progress=1.0,
                    note=f"{blind_report.n_scored} scored")

    # -- batching ----------------------------------------------------------------
    state.set_stage(arm, "batching", status="running")
    batch_out = work / f"batch_{arm}.jsonl"
    if batch_out.exists():
        batch_out.unlink()
    kseen = {"n": 0}

    def ktick(_p) -> None:
        kseen["n"] += 1
        state.set_stage(arm, "batching", progress=kseen["n"] / len(batch_probes),
                        note=f"{kseen['n']}/{len(batch_probes)} bundle legs run")

    run_campaign(
        batch_probes, adapter, batch_out, campaign_id=f"studio_{arm}_batch",
        pricing=PROVISIONAL, corpus=corpus, runtime_hash=RUNTIME_HASH,
        expected_model=adapter.model, on_dispatch=ktick,
    )
    bundles = collect_bundles(read_runs(batch_out, PROVISIONAL))
    batching = analyse(bundles, boot=boot, iters=2000)
    state.put(arm, batching={
        "n_bundles": len(bundles),
        "aggregate_saving_pct": round(batching.aggregate_saving_pct, 2),
        "two_task_saving_pct": round(batching.two_task_saving_pct, 2),
        "bootstrap_lo": round(batching.bootstrap.lo, 2),
        "bootstrap_hi": round(batching.bootstrap.hi, 2),
        "toll_component": round(batching.mean_toll_component, 1),
        "residual_component": round(batching.mean_residual_component, 1),
        "notes": list(batching.notes),
    })
    state.say(arm, f"batching saves {batching.aggregate_saving_pct:.0f}% "
                   f"(bootstrap {batching.bootstrap.lo:.0f}-{batching.bootstrap.hi:.0f}%)")
    state.set_stage(arm, "batching", status="done", progress=1.0,
                    note=f"{batching.aggregate_saving_pct:.0f}% saving")

    # -- example quote and invoice, for the money pages ---------------------------
    example = next((r for r in blind_runs if r.accepted and any(r.units.values())), None)
    if example is not None:
        q = quotes.get(example.probe_id) or make_quote(
            example.run_id, example.units, example.context_bytes, model=model,
            conformal=conformal, pricing=PROVISIONAL, show_per_brick=show,
        )
        inv = make_invoice(example, q, model, boot=boot, show_per_brick=show)
        state.put(arm, example={
            "quote": q.as_dict(),
            "invoice": inv.as_dict(),
            "usd_per_unit": PROVISIONAL.units_to_usd(1.0),
        })

    # -- verdict -----------------------------------------------------------------
    state.set_stage(arm, "verdict", status="running")
    res = state.arms[arm].result
    observations: dict[str, float | None] = {
        "M1_noise_floor_pct": res["measure"]["noise_floor_pct"],
        "M2_drift_pct": res["measure"]["drift_pct"],
        "M5b_cv_lift_pct": (lift.lift_pct if lift else 0.0),
        "M6_blind_mape_pct": res["blind"]["mape_total_pct"],
        "M6b_blind_lift_pct": res["blind"]["lift_pct"],
        "M7_encoder_exact_pct": None,
        "M9_e2e_mape_pct": None,
        "M9b_coverage_hits": float(res["blind"]["coverage_hits"]),
        "M11_batching_saving_pct": res["batching"]["aggregate_saving_pct"],
        "M12_batching_quality_lb": None,
        "M13_alpha_macro": None,
        "M14_exclusion_pct": res["measure"]["exclusion_pct"],
        "M15_identifiable": (
            None if res["fit"]["identifiable"] is None
            else float(res["fit"]["identifiable"])
        ),
    }
    verdict = evaluate(observations, evidence_class=EvidenceClass.PIPELINE_ONLY)
    state.put(arm, verdict=verdict.as_dict(), observations=observations)
    state.say(arm, f"verdict: {verdict.outcome.value}")
    state.set_stage(arm, "verdict", status="done", progress=1.0,
                    note=verdict.outcome.value)


def run_demo(state: DemoState, work_dir: str | Path | None = None) -> None:
    """Blocking. Intended to be called on a worker thread."""
    work = Path(work_dir or (paths.REPO_ROOT / "work" / "studio"))
    work.mkdir(parents=True, exist_ok=True)
    state.set_top(status="running", error=None, started_at=time.time(),
                  finished_at=None)
    try:
        vocab = load_vocabulary()
        state.set_top(vocabulary={
            "version": vocab.version,
            "primary": list(vocab.primary),
            "bricks": [
                {"name": b.name, "category": b.category, "bought_as": b.bought_as,
                 "definition": b.definition, "unit": b.unit,
                 "nearest_neighbour": b.nearest_neighbour,
                 "primary": b.name in vocab.primary}
                for b in vocab.bricks
            ],
        })
        state.say(None, f"{len(vocab.bricks)} bricks loaded, "
                        f"{len(vocab.primary)} of them carry the gates")

        try:
            ref = audit(load_reference(paths.REFERENCE_RUNS))
            state.set_top(reference=ref.as_dict())
            state.say(None, "reference audit replayed from the repository's own runs")
        except Exception as exc:  # noqa: BLE001 - the demo still stands without it
            state.say(None, f"reference audit unavailable: {exc}")

        corpus = _corpus(state, work)
        state.say(None, f"{len(corpus.manifest.documents)} demo documents indexed")

        for arm in ARMS:
            state.say(arm, f"--- {arm} arm: {ARM_BLURB[arm]}")
            _run_arm(state, arm, vocab, corpus, work)

        state.set_top(status="done", finished_at=time.time())
        state.say(None, "demo complete")
    except Exception as exc:  # noqa: BLE001 - surface it in the UI rather than a console
        import traceback

        state.set_top(status="failed", error=f"{type(exc).__name__}: {exc}",
                      finished_at=time.time())
        state.say(None, "FAILED: " + traceback.format_exc().splitlines()[-1])
        raise


def start(state: DemoState, work_dir: str | Path | None = None) -> threading.Thread | None:
    """Kick off a run in the background.

    Returns ``None`` if a run is already in flight. A second concurrent run would
    interleave two writers over one state object and produce a plausible-looking
    but fabricated screen, which is precisely the failure this project exists to
    argue against.
    """
    if not state.claim():
        return None
    t = threading.Thread(target=run_demo, args=(state, work_dir), daemon=True,
                         name="assay-studio-demo")
    t.start()
    return t
