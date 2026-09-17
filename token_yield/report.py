"""Report generation — render forecasts as readable text or structured output."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from statistics import mean
import sys
from urllib.parse import quote, urlsplit

from .models import ProjectForecast


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_cost(c: float) -> str:
    if c < 0.01:
        return f"${c:.4f}"
    return f"${c:.2f}"


def text_report(forecast: ProjectForecast,
                dollars_per_million_tokens: float = 3.0) -> str:
    """Render a forecast as a human-readable text report."""
    lines = []
    lines.append(f"Token Yield Forecast: {forecast.project_name}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Task Breakdown")
    lines.append("-" * 60)
    lines.append(f"{'Type':<20} {'Complexity':<12} {'Mult':>5} {'Tokens':>10} {'Samples':>8}")
    lines.append(f"{'-'*20} {'-'*12} {'-'*5} {'-'*10} {'-'*8}")

    for pred, count in zip(forecast.task_predictions, forecast.task_counts):
        label = pred.task_type
        if count > 1:
            label += f" x{count}"
        lines.append(
            f"{label:<20} {pred.complexity.value:<12} {pred.multiplier:>5.1f} "
            f"{_fmt_tokens(pred.total_predicted):>10} {pred.basis_samples:>8}"
        )

    lines.append("")
    lines.append("Budget Summary")
    lines.append("-" * 60)
    lines.append(f"  Base tokens:        {_fmt_tokens(forecast.total_tokens)}")
    lines.append(f"  Harness overhead:   {_fmt_tokens(forecast.interaction_overhead_tokens)}")
    lines.append(f"  Total predicted:    {_fmt_tokens(forecast.total_with_overhead)}")
    lines.append(f"  Confidence range:   {_fmt_tokens(forecast.total_tokens_low)} – "
                 f"{_fmt_tokens(forecast.total_tokens_high)}")
    lines.append(f"  Estimated time:     {forecast.estimated_hours:.1f} hours")
    lines.append("")

    cost = forecast.cost_at_rate(dollars_per_million_tokens)
    cost_low, cost_high = forecast.cost_range(dollars_per_million_tokens)
    lines.append(f"Cost Estimate (at ${dollars_per_million_tokens}/M tokens)")
    lines.append("-" * 60)
    lines.append(f"  Predicted cost:     {_fmt_cost(cost)}")
    lines.append(f"  Cost range:         {_fmt_cost(cost_low)} – {_fmt_cost(cost_high)}")
    lines.append("")

    if forecast.uncalibrated:
        lines.append("!! INCOMPLETE BUDGET")
        lines.append("-" * 60)
        lines.append("  No calibration data for: " + ", ".join(forecast.uncalibrated))
        lines.append("  Those tasks are NOT in the totals above. Measure them")
        lines.append("  before treating this as the project's cost.")
        lines.append("")

    return "\n".join(lines)


_CUSTOMER_TARGETS = ("input_tokens", "output_tokens", "rated_cost_usd")
_CUSTOMER_FORMS = ("constant", "size", "size+units", "lego", "workflow")
_CUSTOMER_LABELS = {
    "input_tokens": "Input tokens", "output_tokens": "Output tokens",
    "rated_cost_usd": "Retail-rate cost (USD)",
}


def _html_link(url: str, label: str) -> str:
    if urlsplit(url).scheme not in {"https", "http"}:
        raise ValueError("Report source links must use HTTP or HTTPS")
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def _artifact_link(path: Path, document_dir: Path) -> str:
    target, base = path.resolve(), document_dir.resolve()
    for parent in (base, *base.parents):
        if target.is_relative_to(parent):
            href = "../" * len(base.relative_to(parent).parts) + quote(target.relative_to(parent).as_posix())
            break
    else:
        href = target.as_uri()
    return f'<a href="{escape(href, quote=True)}">{escape(path.name)}</a>'


def _customer_number(value: float, target: str) -> str:
    return f"${value:.6f}" if target == "rated_cost_usd" else f"{value:,.2f}"


def customer_error_metrics(observations: list[dict], target: str) -> dict:
    """Compute equal-project-weight errors, never treating repeats as new projects."""
    if not observations:
        raise ValueError("Metrics require measured observations")
    groups = defaultdict(list)
    seen = set()
    for row in observations:
        if row["observation_id"] in seen:
            raise ValueError("Duplicate observation ID")
        seen.add(row["observation_id"])
        actual, predicted = row["actual"][target], row["predicted"][target]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in (actual, predicted)):
            raise ValueError("Metrics require finite nonnegative targets and predictions")
        groups[row["project_id"]].append((actual, predicted))
    errors = [[predicted - actual for actual, predicted in rows] for rows in groups.values()]
    # Do not silently omit zero targets from percentage errors.
    positive = all(actual > 0 for rows in groups.values() for actual, _ in rows)
    return {
        "mae": mean(mean(abs(error) for error in group) for group in errors),
        "rmse": math.sqrt(mean(mean(error * error for error in group) for group in errors)),
        "bias": mean(mean(group) for group in errors),
        "mape_pct": (
            100 * mean(mean(abs(predicted - actual) / actual for actual, predicted in rows)
                       for rows in groups.values()) if positive else None
        ),
        "n_projects": len(groups),
        "n_observations": len(observations),
    }


def _customer_score_table(scores: dict, selected: dict, caption: str) -> str:
    rows = []
    for form in _CUSTOMER_FORMS:
        cells = []
        for target in _CUSTOMER_TARGETS:
            picked = selected[target] == form
            cells.append(
                f'<td class="{"selected" if picked else ""}">'
                f'{_customer_number(scores[target][form]["mae"], target)}'
                f'{" <small>CV-selected</small>" if picked else ""}</td>'
            )
        rows.append(f"<tr><th scope=\"row\">{escape(form)}</th>{''.join(cells)}</tr>")
    return (
        f'<div class="table-scroll"><table><caption>{escape(caption)}</caption>'
        '<thead><tr><th>Predictor</th><th>Input MAE</th><th>Output MAE</th>'
        f'<th>Cost MAE (USD)</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _customer_metric_table(observations: list[dict]) -> str:
    rows = []
    for target in _CUSTOMER_TARGETS:
        metric = customer_error_metrics(observations, target)
        percentage = "N/A: zero target" if metric["mape_pct"] is None else f'{metric["mape_pct"]:.2f}%'
        rows.append(
            f'<tr><th scope="row">{_CUSTOMER_LABELS[target]}</th>'
            f'<td>{_customer_number(metric["mae"], target)}</td>'
            f'<td>{_customer_number(metric["rmse"], target)}</td>'
            f'<td>{percentage}</td><td>{_customer_number(metric["bias"], target)}</td></tr>'
        )
    return (
        '<div class="table-scroll"><table><caption>Selected models: complementary error metrics</caption>'
        '<thead><tr><th>Target</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>Signed bias</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _customer_cost_chart(observations: list[dict], names: dict[str, str]) -> str:
    maximum = max(
        row[key]["rated_cost_usd"] for row in observations for key in ("actual", "predicted")
    )
    upper = maximum * 1.1 if maximum else 1
    parts = [
        '<svg viewBox="0 0 580 370" role="img" aria-label="Predicted versus measured scoping cost">',
        '<title>Cost predictions: the diagonal is perfect prediction, not a confidence interval.</title>',
        '<rect x="70" y="20" width="475" height="285" fill="#f3f7fa"/>',
    ]
    for tick in range(6):
        value = upper * tick / 5
        x, y = 70 + 475 * tick / 5, 305 - 285 * tick / 5
        parts.extend([
            f'<path d="M70 {y}H545" stroke="#dce5ec"/>',
            f'<text x="60" y="{y + 4}" text-anchor="end">${value:.3f}</text>',
            f'<text x="{x}" y="326" text-anchor="middle">${value:.3f}</text>',
        ])
    parts.append('<path d="M70 305L545 20" stroke="#6b7785" stroke-dasharray="6 5"/>')
    for row in observations:
        actual, predicted = row["actual"]["rated_cost_usd"], row["predicted"]["rated_cost_usd"]
        label = (
            f'{names[row["project_id"]]} | {row["arm"]} | repeat {row["replicate"] + 1} | '
            f'actual ${actual:.6f}, predicted ${predicted:.6f}'
        )
        parts.append(
            f'<circle cx="{70 + actual / upper * 475:.2f}" cy="{305 - predicted / upper * 285:.2f}" '
            f'r="5.5" fill="{"#147d75" if row["arm"] == "batched" else "#395cbd"}" opacity=".8">'
            f'<title>{escape(label)}</title></circle>'
        )
    parts.extend([
        '<text x="308" y="359" text-anchor="middle">Measured retail-rate cost (USD)</text>',
        '<text transform="translate(15 165) rotate(-90)" text-anchor="middle">Predicted cost (USD)</text>',
        '</svg><p class="caption"><span class="dot teal"></span>Batched '
        '<span class="dot blue"></span>Split. Each dot is one complete scoping execution; '
        'dots from the same project are not independent projects.</p>',
    ])
    return "".join(parts)


def _customer_observation_table(observations: list[dict], names: dict[str, str]) -> str:
    rows = []
    for row in observations:
        actual, predicted = row["actual"], row["predicted"]
        reasons = "; ".join(row["support"]["reasons"]) or "Marginal ranges only"
        rows.append(
            f'<tr><th scope="row">{escape(names[row["project_id"]])}</th>'
            f'<td>{escape(row["arm"])} / {row["replicate"] + 1}</td>'
            f'<td>{actual["input_tokens"]:,} / {predicted["input_tokens"]:,.1f}</td>'
            f'<td>{actual["output_tokens"]:,} / {predicted["output_tokens"]:,.1f}</td>'
            f'<td>${actual["rated_cost_usd"]:.6f} / ${predicted["rated_cost_usd"]:.6f}</td>'
            f'<td>{escape(row["support"]["status"])}<small>{escape(reasons)}</small></td>'
            f'<td>{"Pass" if row["contract_passed"] else "Fail; cost retained"}</td></tr>'
        )
    return (
        f'<details><summary>Inspect all {len(observations)} measured executions</summary>'
        '<div class="table-scroll"><table><caption>Actual / predicted values; automatic contract checks only</caption>'
        '<thead><tr><th>Project</th><th>Arm / repeat</th><th>Input tokens</th><th>Output tokens</th>'
        f'<th>Cost (USD)</th><th>Support</th><th>Contract</th></tr></thead><tbody>{"".join(rows)}'
        '</tbody></table></div></details>'
    )


def _customer_new_projects(catalog: dict | None, evaluation: dict | None, names: dict) -> str:
    if catalog is None:
        return '<p class="notice">No new-project catalog supplied. New-project accuracy is not measured.</p>'
    cards = []
    for project in catalog["projects"]:
        exclusions = "".join(f"<li>{escape(item)}</li>" for item in project["exclusions"])
        requirements = "".join(
            f'<li>{escape(item["text"])} <small>Supplied label: {escape(item["brick"])}</small></li>'
            for item in project["requirements"]
        )
        cards.append(
            f'<article class="card"><span class="tag">New source / scoping only</span>'
            f'<h3>{escape(project["buyer"])}</h3><p>{_html_link(project["source_url"], project["title"])}</p>'
            f'<p>{escape(project["summary"])}</p><p><strong>{len(project["requirements"])} supplied requirements'
            '</strong> mapped to Extract, Classify, Plan and Report. No new execution primitive.</p>'
            f'<details><summary>Included scoping inputs and excluded delivery</summary>'
            f'<ol>{requirements}</ol><h4>Not executed or priced</h4><ul>{exclusions}</ul></details></article>'
        )
    rejected = "".join(
        f'<li>{_html_link(item["source_url"], item["title"])}: {escape(item["reason"])}</li>'
        for item in catalog.get("rejected_projects", [])
    )
    result = (
        '<p>Primary buyer requests found on the web, not marketplace templates. The inclusion gate is '
        '<strong>the supplied-brief scoping task</strong>, not the entire procurement. External research, '
        'stakeholder interviews, live systems and final professional delivery remain outside this model.</p>'
        '<p class="notice"><strong>No full-delivery-compatible engagement was verified.</strong> '
        f'The {len(catalog["projects"])} retained inputs are bounded scoping tasks derived from these requests. '
        'Their complete customer engagements are excluded, rather than declared supported.</p>'
        '<ul>' + "".join(f"<li>{escape(note)}</li>" for note in catalog.get("selection_notes", [])) + '</ul>'
        f'<div class="grid two">{"".join(cards)}</div>'
        + (f'<details><summary>Rejected full-delivery examples</summary><ul>{rejected}</ul></details>' if rejected else "")
    )
    if evaluation is None or evaluation["status"] != "completed":
        status = "not executed" if evaluation is None else (
            "not yet finalized" if evaluation["status"] == "preflight" else evaluation["status"]
        )
        return result + (
            f'<p class="notice">New measurement status: <strong>{escape(status)}</strong>. '
            'No new-project accuracy result is available. Forecasts alone are not test labels.</p>'
        )
    observations = evaluation["observations"]
    limitations = "".join(f"<li>{escape(item)}</li>" for item in evaluation["limitations"])
    return result + (
        f'<h3>New measurements, using the unchanged prediction formula</h3><p>{evaluation["n_projects"]} new project groups; '
        f'{evaluation["n_observations"]} complete executions. Weights and model choice were frozen before '
        'these outcomes; no fitting or selection on the new labels.</p>'
        '<p class="notice"><strong>This is a transfer test, not an approved quote.</strong> '
        'The allowed input list and the test-running program changed. The original strict compatibility check '
        'therefore declines to issue an ordinary forecast. For this research comparison we still evaluate the '
        'unchanged prediction formula, while retaining that warning. The prompt-building and API-dispatch '
        'components were checked against the original versions.</p>'
        f'<details><summary>Inspect the full compatibility and evidence caveats</summary><ul>{limitations}</ul></details>'
        + _customer_score_table(evaluation["scores"], evaluation["selected_forms"], "New-project conditional-prediction MAE; lower is better")
        + _customer_metric_table(observations)
        + f'<div class="chart">{_customer_cost_chart(observations, names)}</div>'
        + _customer_observation_table(observations, names)
        + f'<p>Total new measured retail-rate API cost: <strong>${evaluation["totals"]["rated_cost_usd"]:.6f}</strong>. '
        'This excludes research, report production and human work; it is not a client project quote.</p>'
    )


def customer_html_report(model_dir: Path, catalog_path: Path, *,
                         new_catalog_path: Path | None = None,
                         prospective_dir: Path | None = None,
                         document_dir: Path | None = None) -> str:
    """Render an offline, self-contained evidence report from saved customer artifacts.

    Raises:
        ValueError: Artifacts disagree, labels are incomplete, or a source URL is unsafe.
    """
    model = json.loads((model_dir / "models.json").read_text(encoding="utf-8"))
    analysis = json.loads((model_dir / "analysis.json").read_text(encoding="utf-8"))
    records = json.loads((model_dir / "project_records.json").read_text(encoding="utf-8"))
    predictions = json.loads((model_dir / "predictions.json").read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    new_catalog = (
        json.loads(new_catalog_path.read_text(encoding="utf-8")) if new_catalog_path is not None else None
    )
    evaluation = (
        json.loads((prospective_dir / "prospective_evaluation.json").read_text(encoding="utf-8"))
        if prospective_dir is not None else None
    )
    if model["schema_version"] != "customer-project-model-v1":
        raise ValueError("Report requires customer-project-model-v1")
    model_hash = sha256((model_dir / "models.json").read_bytes()).hexdigest()
    selected = {target: model["channels"][target]["selected_form"] for target in _CUSTOMER_TARGETS}
    if selected != analysis["selected_forms"]:
        raise ValueError("Analysis and model selections disagree")
    indexed = {row["observation_id"]: row for row in predictions}
    if (len(indexed) != len(predictions) or len({row["call_id"] for row in records}) != len(records)
            or set(indexed) != {row["call_id"] for row in records}):
        raise ValueError("Records and predictions must have matching unique observation IDs")
    names = {project["id"]: project["buyer"] for project in catalog["projects"]}
    if set(names) != {row["project_id"] for row in records}:
        raise ValueError("Catalog must match measured projects")
    observations = [
        {
            "observation_id": row["call_id"], "project_id": row["project_id"],
            "arm": row["arm"], "replicate": row["replicate"], "quote": row["quote"],
            "actual": {**row["usage"], "rated_cost_usd": row["rated_cost_usd"]},
            "predicted": indexed[row["call_id"]]["point_estimate"],
            "support": indexed[row["call_id"]]["support"], "contract_passed": row["contract_passed"],
        }
        for row in records if row["split"] == "holdout"
    ]
    metrics = {target: customer_error_metrics(observations, target) for target in _CUSTOMER_TARGETS}
    scores = analysis["holdout"]["scores"]
    for target in _CUSTOMER_TARGETS:
        if not math.isclose(metrics[target]["mae"], scores[target][selected[target]]["mae"], rel_tol=1e-10):
            raise ValueError("Saved holdout MAE does not match measured targets and predictions")
    if evaluation is not None:
        if new_catalog is None or evaluation["model_sha256"] != model_hash:
            raise ValueError("New evaluation requires its catalog and the identical frozen model")
        if evaluation["catalog_file_sha256"] != sha256(new_catalog_path.read_bytes()).hexdigest():
            raise ValueError("New evaluation catalog hash mismatch")
        if evaluation["selected_forms"] != selected:
            raise ValueError("New evaluation must retain training-selected forms")
        if evaluation["status"] == "completed":
            if {row["project_id"] for row in evaluation["observations"]} & set(names):
                raise ValueError("New evaluation overlaps historical projects")
            if {row["project_id"] for row in evaluation["observations"]} != {
                project["id"] for project in new_catalog["projects"]
            }:
                raise ValueError("Completed evaluation must cover the new catalog")
            for target in _CUSTOMER_TARGETS:
                actual_metric = customer_error_metrics(evaluation["observations"], target)
                if (actual_metric["n_projects"] != evaluation["n_projects"]
                        or actual_metric["n_observations"] != evaluation["n_observations"]):
                    raise ValueError("New evaluation counts disagree with observations")
                if not math.isclose(actual_metric["mae"], evaluation["scores"][target][selected[target]]["mae"],
                                    rel_tol=1e-10):
                    raise ValueError("New evaluation scores disagree with frozen predictions")
            if not math.isclose(sum(row["actual"]["rated_cost_usd"] for row in evaluation["observations"]),
                                evaluation["totals"]["rated_cost_usd"], rel_tol=1e-10):
                raise ValueError("New evaluation cost total disagrees with observations")
    if new_catalog is not None:
        names.update({project["id"]: project["buyer"] for project in new_catalog["projects"]})
    baseline = min(scores["rated_cost_usd"][form]["mae"] for form in _CUSTOMER_FORMS[:3])
    improvement = 100 * (1 - metrics["rated_cost_usd"]["mae"] / baseline)
    requirement_count = sum(len(project["requirements"]) for project in catalog["projects"])
    projects = []
    for project in catalog["projects"]:
        role = "Training" if project["split"] == "train" else "Historical holdout"
        projects.append(
            f'<tr><th scope="row">{_html_link(project["source_url"], project["buyer"])}</th>'
            f'<td>{escape(project["title"])}</td><td>{len(project["requirements"])}</td>'
            f'<td>{role}</td><td>4 executions / 10 calls</td></tr>'
        )
    features = model["channels"]["input_tokens"]["feature_definitions"]
    feature_rows = "".join(
        f'<tr><th scope="row"><code>{escape(name)}</code></th><td>{escape(definition)}</td>'
        f'<td>{"Support only" if name == "context_bytes" else "Quote-time"}</td></tr>'
        for name, definition in features.items()
    )
    form_rows = "".join(
        f'<tr><th scope="row">{escape(form)}</th><td>{len(value["features"])}</td>'
        f'<td>{escape(", ".join(value["features"]) or "Intercept / training mean only")}</td></tr>'
        for form, value in model["channels"]["input_tokens"]["forms"].items()
    )
    cv_scores = {
        target: {form: {"mae": value["cv_mae"]} for form, value in channel["forms"].items()}
        for target, channel in model["channels"].items()
    }
    new_section = _customer_new_projects(new_catalog, evaluation, names)
    new_complete = evaluation is not None and evaluation["status"] == "completed"
    headline = (
        {target: customer_error_metrics(evaluation["observations"], target) for target in _CUSTOMER_TARGETS}
        if new_complete else metrics
    )
    headline_label = "new scoping test" if new_complete else "earlier test"
    headline_projects = evaluation["n_projects"] if new_complete else analysis["holdout_projects"]
    headline_observations = evaluation["n_observations"] if new_complete else analysis["holdout_observations"]
    headline_note = (
        '<p class="notice">The headline numbers now show the <strong>new scoping test</strong>. '
        'These are conditional research predictions because the input permissions and test controller changed. '
        '<a href="#new-projects">See the new-test results and compatibility warning</a>. '
        'The earlier test remains in section 04; differences between the two sets are not retraining gains.</p>'
        if new_complete else ""
    )
    evidence = {
        "model_sha256": model_hash, "model_run": model_dir.name, "selected_forms": selected,
        "historical_metrics": metrics, "historical_scores": scores,
        "historical_observations": observations,
        "prospective_status": evaluation["status"] if evaluation else "not_executed",
        # Embed result evidence, not machine-local provenance or controller logs.
        "prospective_evaluation": {
            name: evaluation[name] for name in (
                "status", "method", "model_sha256", "catalog_sha256", "catalog_file_sha256",
                "n_projects", "n_observations", "planned_observations", "selected_forms",
                "scores", "observations", "totals", "limitations", "spend_scope",
            ) if name in evaluation
        } if evaluation else None,
    }
    evidence_json = json.dumps(evidence, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    failed = sum(not row["contract_passed"] for row in records)
    extrapolated = sum(row["support"]["status"] == "extrapolation" for row in observations)
    runtime = model["runtime"]
    document_dir = document_dir or Path(__file__).resolve().parents[1] / "docs"
    artifact_paths = [
        catalog_path, model_dir / "project_records.json", model_dir / "models.json",
        model_dir / "predictions.json", model_dir / "analysis.json",
    ]
    if new_catalog_path is not None:
        artifact_paths.append(new_catalog_path)
    if prospective_dir is not None:
        artifact_paths.append(prospective_dir / "prospective_evaluation.json")
    artifact_links = " &middot; ".join(_artifact_link(path, document_dir) for path in artifact_paths)
    literature = (
        ("https://arxiv.org/abs/2604.22750v2", "Agent token-cost variability"),
        ("https://arxiv.org/abs/1905.03222v1", "Conformalized Quantile Regression"),
        ("https://doi.org/10.1145/3786352", "Hierarchical conformal prediction"),
        ("https://arxiv.org/abs/2403.03868", "Selection-conditional conformal inference"),
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEGO-style token prediction | Model evidence report</title>
<style>
:root{{--ink:#172c3d;--muted:#526474;--paper:#f4f7fa;--teal:#147d75;--blue:#395cbd;--line:#dce5ec;--amber:#8a5104}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}
a{{color:#1e5c9a;text-underline-offset:3px;overflow-wrap:anywhere}}a:hover{{color:var(--teal)}}a:focus-visible,summary:focus-visible{{outline:3px solid #d58714;outline-offset:4px}}
header{{background:#132d3d;color:#fff;padding:66px max(6vw,24px) 48px}}header>div,main,nav>div{{max-width:1160px;margin:auto}}
.eyebrow{{font-size:.77rem;letter-spacing:.14em;text-transform:uppercase;font-weight:750;color:#85dbd2}}
h1{{font-size:clamp(2.2rem,5vw,4.2rem);line-height:1.08;letter-spacing:-.045em;max-width:840px;margin:22px 0}}
header p{{max-width:850px;color:#d1e0e9;font-size:1.13rem}}.tag{{display:inline-block;padding:4px 10px;border:1px solid var(--line);border-radius:6px;font-size:.75rem;font-weight:700;letter-spacing:.04em}}
header .tag{{border-color:#456173;color:#d9e7ee;margin:5px 8px 0 0}}nav{{background:#fff;border-bottom:1px solid var(--line);padding:14px 24px}}
nav>div{{display:flex;flex-wrap:wrap;gap:10px 28px}}nav a{{font-size:.87rem;font-weight:650;text-decoration:none}}
main{{padding:0 24px 60px}}section{{padding-top:52px;scroll-margin-top:20px}}h2{{font-size:clamp(1.6rem,3vw,2.3rem);line-height:1.25;letter-spacing:-.035em;margin:12px 0 20px}}
h3{{font-size:1.2rem;line-height:1.35;margin:15px 0 10px}}h4{{margin-bottom:6px}}p{{margin:12px 0}}.section-no{{color:var(--teal);font-size:.78rem;font-weight:800;letter-spacing:.1em}}
.grid{{display:grid;gap:18px}}.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}.four{{grid-template-columns:repeat(4,minmax(0,1fr))}}
.card,.chart{{background:#fff;border:1px solid var(--line);border-radius:13px;padding:24px;min-width:0}}.card p{{font-size:.92rem}}
.stat{{font-size:clamp(1.65rem,3vw,2.2rem);font-weight:750;line-height:1.25;letter-spacing:-.025em;margin:10px 0}}
.caption,small{{font-size:.8rem;color:var(--muted)}}small{{display:block}}.notice{{background:#fff5e2;border-left:4px solid #ce8c28;padding:18px 22px;border-radius:0 9px 9px 0}}
.insight{{background:#e4f3ef;border-left:4px solid var(--teal);padding:18px 22px;border-radius:0 9px 9px 0}}
.brick{{border-top:7px solid var(--teal)}}.brick:nth-child(2){{border-top-color:var(--blue)}}.brick:nth-child(3){{border-top-color:#aa693a}}.brick:nth-child(4){{border-top-color:#8360a9}}
.studs{{display:flex;gap:8px;margin:0 0 14px}}.studs i{{width:20px;height:12px;background:var(--line);border-radius:7px 7px 2px 2px;display:block}}
.flow{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}}.flow>div{{padding:18px;background:#e7eef4;border-radius:9px;font-weight:650}}
.flow small{{font-weight:400;margin-top:6px}}code{{font: .85em ui-monospace,Consolas,monospace;overflow-wrap:anywhere}}.equation{{padding:18px;background:#132d3d;color:white;border-radius:9px;overflow:auto}}
.table-scroll{{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:10px;background:white}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}
caption{{text-align:left;padding:14px 18px;font-weight:700;background:#eaf0f5}}th,td{{padding:13px 16px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
thead th{{font-size:.77rem;color:var(--muted)}}tbody th{{font-weight:650}}tr:last-child>*{{border-bottom:0}}td.selected{{background:#e1f3ec;font-weight:750}}td.selected small{{color:#185e49;font-weight:500}}
details{{background:white;border:1px solid var(--line);border-radius:9px;padding:14px 18px;margin:18px 0}}summary{{cursor:pointer;font-weight:650}}details .table-scroll{{margin-bottom:4px}}
svg{{display:block;width:100%;height:auto}}svg text{{font:12px system-ui,sans-serif;fill:#526474}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 6px 0 14px}}.teal{{background:var(--teal)}}.blue{{background:var(--blue)}}
li{{margin:6px 0}}footer{{border-top:1px solid var(--line);margin-top:45px;padding-top:24px;color:var(--muted);font-size:.8rem;overflow-wrap:anywhere}}
@media(max-width:780px){{.four,.flow{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:1fr}}header{{padding-top:40px}}main{{padding:0 16px 40px}}.card,.chart{{padding:18px}}}}
@media(max-width:430px){{.four,.flow{{grid-template-columns:1fr}}th,td{{padding:10px}}}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
@media print{{header{{background:white;color:#132d3d;padding:20px}}header p{{color:#172c3d}}nav{{display:none}}main{{max-width:none}}section{{padding-top:25px}}.card{{break-inside:avoid}}details{{display:block}}}}
</style></head><body>
<header><div><div class="eyebrow">Token Yield / Experimental evidence</div>
<h1>From public project briefs<br>to token &amp; cost forecasts.</h1>
<p>How we built the first brick-based predictor, what it learned, how well it performs,
and what new customer briefs can actually test.</p>
<span class="tag">4 GPT tasks per brief</span><span class="tag">6 original project groups</span>
<span class="tag">Prediction formulas fixed for testing</span><span class="tag">Scoping, not full delivery</span>
</div></header>
<nav aria-label="Report sections"><div><a href="#bricks">01 / Bricks</a><a href="#features">02 / Features</a>
<a href="#model">03 / Model &amp; data</a><a href="#performance">04 / Performance</a>
<a href="#new-projects">05 / New projects</a><a href="#limits">Limits &amp; evidence</a></div></nav>
<main><section aria-label="Results at a glance"><div class="grid four">
<div class="card"><small>Average cost prediction error / {headline_label}</small><div class="stat">${headline["rated_cost_usd"]["mae"]:.6f}</div><small>Dollars per complete scoping execution</small></div>
<div class="card"><small>Average input-token error / {headline_label}</small><div class="stat">{headline["input_tokens"]["mae"]:.1f}</div><small>Error predicting tokens sent to GPT</small></div>
<div class="card"><small>Average output-token error / {headline_label}</small><div class="stat">{headline["output_tokens"]["mae"]:.1f}</div><small>Error predicting tokens generated by GPT</small></div>
<div class="card"><small>Project briefs in this test</small><div class="stat">{headline_projects}</div><small>{headline_observations} executions share these briefs; repeats are not new projects</small></div>
</div>{headline_note}<p class="notice"><strong>Read the scope first.</strong> These are API tokens and published-rate costs for drafting a
structured scoping response from a short supplied brief. They are <strong>not</strong> the cost of building a website,
conducting a security assessment, or completing a customer engagement. Prediction intervals are not calibrated.</p>
<p><a href="lego-model-training.md">Read the short, plain-English training explanation</a>.</p>
<p><strong>Training update:</strong> we reused 60 recorded GPT calls to learn whole-scope prediction formulas.
Four original projects supplied training examples; two others supplied the earlier test.
The four new projects tested the same formulas without retraining GPT or the predictor.</p>
<div class="card"><h3>Plain-English guide</h3><p><strong>Operation = a task we ask GPT to do.</strong>
The four tasks are Extract, Classify, Plan and Report. They can run in four separate requests or one combined request.</p>
<p><strong>Annotation = an attached label or note.</strong> For example: &quot;maintenance documentation&quot;
may be labeled &quot;document draft / agent can propose a draft&quot;. <strong>Frozen = kept unchanged for the comparison.</strong>
Frozen annotations are unchanged input notes; frozen regression means the learned prediction formula is not updated during testing.</p>
<p><strong>MAE = mean absolute error, or average size of the prediction mistake.</strong> If actual input is 1,000 tokens and
we predict 1,200, that run's absolute input error is 200 tokens. Cost error is measured in dollars instead.
&quot;Historical&quot; means the earlier, already-collected test, not new measurements or training accuracy.</p></div></section>

<section id="bricks"><div class="section-no">01 / HOW WE CREATED THE BRICKS</div><h2>The four GPT tasks we actually measured.</h2>
<p>We started with six real, first-party customer requests. We paraphrased their requirements, assigned deliverable and
ownership labels, recorded source URLs and explicit exclusions, then projected every brief into the same four operations.
The prepared input list contains <strong>{requirement_count} hand-annotated requirements</strong> across six projects.
These are {requirement_count} individual requirements, not {requirement_count} brick types.</p>
<h3>First: how we prepared and ran the experiment</h3>
<p><strong>The four boxes below are experiment-building steps, NOT the four GPT tasks.</strong></p>
<div class="flow"><div>Find public briefs<small>Read real customers' requests for work and retain source links</small></div>
<div>Prepare and fix the notes<small>Write short summaries, label requirements and list excluded work; then keep them unchanged</small></div>
<div>Ask GPT to do four tasks<small>Extract, Classify, Plan and Report; each uses the same supplied brief</small></div>
<div>Record actual usage<small>Save the responses, provider-reported tokens and calculated API cost</small></div></div>
<h3>Then: the four tasks inside each scoping execution</h3>
<div class="grid four">
<article class="card brick"><div class="studs"><i></i><i></i><i></i></div><h3>Extract</h3><p>Copy every listed customer requirement exactly into a structured response.</p><small>Unit: one listed requirement. No browsing or discovery.</small></article>
<article class="card brick"><div class="studs"><i></i><i></i><i></i></div><h3>Classify</h3><p>Return the labels we already attached, such as the kind of deliverable and whether a human is needed.</p><small>Unit: one listed requirement. This is label preservation, not autonomous categorization.</small></article>
<article class="card brick"><div class="studs"><i></i><i></i><i></i></div><h3>Plan</h3><p>Suggest documents or other outputs that could address each requirement. Flag unverified inputs and required human review.</p><small>Unit: one listed requirement. No action is executed.</small></article>
<article class="card brick"><div class="studs"><i></i><i></i><i></i></div><h3>Report</h3><p>Write a short scope summary. Keep the source, requirement IDs and excluded work, and state that delivery has not been executed.</p><small>Unit: one project-level scoping report.</small></article></div>
<p>For <code>r</code> requirements, the counts are <code>[extract=r, classify=r, plan=r, report=1]</code>,
or <code>3r + 1</code> scoping units. Original deliverable tags such as <code>review</code> and <code>draft</code>
describe the customer's request; they do not mean those full-delivery operations were measured.</p>
<div class="grid two"><div class="card"><h3>Split: four API calls</h3><p>The brief is repeated in four requests,
one per operation. Their measured usage and cost are summed into one observation.</p></div>
<div class="card"><h3>Batched: one API call</h3><p>The same four independent operations share one request.
This is prompt batching, not a discounted provider Batch API. No operation consumes another's generated output.</p></div></div>
<p class="notice"><strong>Not trained yet:</strong> the proposed thirteen-operation taxonomy and the 100-1,000-word,
complexity, industry, retrieval/LLM-Wiki and multi-tool grids are research plans, not this training dataset.
No claim of universal project coverage or independent per-brick prices follows from this pilot.</p></section>

<section id="features"><div class="section-no">02 / INPUT FEATURES</div><h2>Only what we know before execution.</h2>
<p>The largest candidate uses <strong>8 nominal columns</strong>. Across all five forms there are nine distinct candidate
predictor names; context length is an additional support diagnostic. Model, harness and output policy are frozen metadata,
not learned categorical effects. No measured output length or realized cost enters the input features.</p>
<div class="table-scroll"><table><caption>Feature dictionary from the saved model</caption>
<thead><tr><th>Field</th><th>Meaning</th><th>Availability</th></tr></thead><tbody>{feature_rows}</tbody></table></div>
<div class="table-scroll"><table><caption>Five candidate designs; the same designs are fitted separately for each target</caption>
<thead><tr><th>Form</th><th>Nominal columns</th><th>Inputs</th></tr></thead><tbody>{form_rows}</tbody></table></div>
<p class="insight"><strong>Why tools and harnesses matter:</strong> splitting a brief repeats prompt overhead, so the workflow
model includes planned call count and batch size. This experiment has no tools, clarification rounds, history growth or retries.
A new skill such as an interview or &quot;grill me&quot; changes the execution contract; its cost is not established here.</p>
<p>Every complete execution permits 6,400 output tokens; <code>report=1</code> is also constant. Standardization drops constant
columns: the selected workflow regressions have six active columns and the selected LEGO output regression four.
Extract, Classify and Plan counts are identical here, so these coefficients cannot identify independent causal brick costs.
There are no explicit industry or complexity features in this fitted model.</p></section>

<section id="model"><div class="section-no">03 / MODEL, TRAINING AND TESTING</div><h2>Three small regressions, not a fine-tuned LLM.</h2>
<p><strong>{escape(runtime["deployment"])}</strong> generated the measured scoping responses; it is not the model we trained.
The predictors are standardized linear <strong>ridge regressions</strong> with fixed regularization
<code>alpha = 10</code> and a nonnegative prediction floor. The constant baseline uses the training mean.</p>
<p>In plain English, ridge regression learns a weighted prediction formula from measured examples, with a penalty
that discourages excessively large weights. After training, we keep the formula unchanged while testing it.
That is all &quot;frozen ridge regression&quot; means here. <strong>We did not fine-tune GPT.</strong></p>
<div class="equation"><code>prediction = max(0, intercept + sum(weight[j] * (feature[j] - training_mean[j]) / training_scale[j]))</code></div>
<ol><li>Aggregate complete measured requests by <strong>project + arm + replicate</strong>. Retain costs even when automatic contract checks fail.</li>
<li>Use the original four training projects only. Each leave-one-project-out fold holds out both arms and both repeats together;
fit scaling and coefficients on the remaining projects.</li>
<li>Choose each target's form by equal-project-weight cross-validation MAE; refit that form on all training projects.</li>
<li>Reload the frozen coefficients for evaluation. New outcomes do not change coefficients, feature scaling or model choice.</li></ol>
<div class="grid two"><div class="card"><h3>Training</h3><div class="stat">{analysis["training_projects"]} projects / {analysis["training_observations"]} observations</div>
<p>Two arms and two repeats per project. Four grouped CV folds. A repeat is not a new independent project.</p></div>
<div class="card"><h3>Historical holdout</h3><div class="stat">{analysis["holdout_projects"]} projects / {analysis["holdout_observations"]} observations</div>
<p>Previously examined outcomes. This is a retrospective evaluation, not a newly sealed confirmatory test. Zero independent calibration projects.</p></div></div>
<div class="table-scroll"><table><caption>Original six public projects: the labels measure their scoping tasks only</caption>
<thead><tr><th>Buyer / primary source</th><th>Original request</th><th>Requirements</th><th>Role</th><th>Measurements</th></tr></thead>
<tbody>{"".join(projects)}</tbody></table></div>
<p><strong>{analysis["measured_calls"]} measured API calls</strong> become {analysis["project_observations"]} complete execution labels:
six projects x two repeats x (four split calls + one batched call). Total recorded retail-rate cost:
<strong>${analysis["total_recorded_rated_cost_usd"]:.5f}</strong>. Offline regression training itself made no paid API calls.</p>
<details><summary>Inspect the training cross-validation scores</summary>
{_customer_score_table(cv_scores, selected, "Leave-one-training-project-out CV MAE")}
<p>The winning CV score is used for model selection; it is not an unbiased estimate on new projects.</p></details>
<p>Targets are measured input tokens, measured output tokens, and direct published-rate cost. Total-token prediction is input
plus output prediction; cost uses its separately selected regression and is <strong>not</strong> merely repriced predicted tokens.</p></section>

<section id="performance"><div class="section-no">04 / PERFORMANCE</div><h2>Promising on this small historical comparison.</h2>
{_customer_score_table(scores, selected, "Historical holdout: equal-project-weight MAE; lower is better")}
<p class="insight">The training-selected workflow cost model has <strong>{improvement:.1f}% lower historical cost MAE</strong>
than the lowest-error simple comparator in this table (constant, size, size+units). This is a descriptive comparison on two
project groups, not proof of a population-wide improvement. The older call-level experiment where the constant baseline won
is a different dataset/target/granularity; these results do not invalidate it.</p>
<p>The comparisons above do not cover every stronger baseline proposed in the protocol. Exact request-token counting,
matched historical execution costs, branching/retry policies and cost-to-human-accepted-answer distributions have not all
been benchmarked here. We cannot claim that this predictor beats those alternatives.</p>
<div class="grid two"><div class="chart">{_customer_cost_chart(observations, names)}</div>
<div class="card"><h3>How to read the numbers</h3><p><strong>MAE:</strong> average absolute prediction error; first average within
each project, then equally across projects.</p><p><strong>RMSE:</strong> square root of the equally weighted project means of
squared errors; larger misses receive more weight.</p><p><strong>MAPE:</strong> average absolute percentage error, using the same
project weighting. Undefined if any actual target is zero.</p><p><strong>Signed bias:</strong> prediction minus actual, averaged
with equal project weights. Positive means overprediction.</p><p><strong>Not reported as established:</strong> interval coverage,
upper-budget coverage, human answer quality or full-project delivery accuracy.</p></div></div>
{_customer_metric_table(observations)}
<p class="notice"><strong>{extrapolated}/{len(observations)} historical holdout predictions are extrapolations.</strong>
Their source-context lengths exceed the training range; some prompts do too. A small observed error does not remove that warning.
&quot;Within observed ranges&quot; would still not establish joint-feature or industry support.</p>
{_customer_observation_table(observations, names)}
<p>Automatic contract checks passed on {analysis["contract_passed_project_observations"]}/{analysis["project_observations"]}
complete historical observations; all {failed} failures remain in the cost evaluation. These checks validate structure and
preserved fields, not the semantic quality of the free-text plans. Batching savings alone do not establish non-inferior answers.</p></section>

<section id="new-projects"><div class="section-no">05 / ADDITIONAL WEB-SOURCED PROJECTS</div>
<h2>New inputs, frozen model, honest labels.</h2>{new_section}</section>

<section id="limits"><div class="section-no">WHAT WE CAN SAY NEXT</div><h2>Point estimate + support warning. Intervals still pending.</h2>
<div class="grid two"><div class="card"><h3>Available now</h3><ul><li>Reloadable point predictions without retraining.</li>
<li>Training-only marginal-range and layout checks.</li><li>Runtime mismatch abstention in the forecast API.</li>
<li>Raw usage labels and model/data provenance.</li></ul></div><div class="card"><h3>Needed before stronger claims</h3>
<ul><li>Many more independent, varied project groups.</li><li>A frozen harness and independent calibration allocation.</li>
<li>Whole-project residual calibration, plus a separately calibrated upper budget bound.</li>
<li>Human acceptance and quality-preserving batching evaluation.</li></ul></div></div>
<p>No 90% or 95% prediction interval is shown: the saved fields are null. Calibrating on repeats of the same brief would
overstate independence. A support-filtered quoting policy also needs calibration for the population it actually accepts.</p>
<p>Methodological references: {"; ".join(_html_link(url, label) for url, label in literature)}.
These motivate future design; they do not confer coverage guarantees on this pilot.</p>
</section><footer>
<p>Generated from frozen local artifacts, not manually invented scores. Model run: <code>{escape(model_dir.name)}</code>.
Model SHA-256: <code>{model_hash}</code>. Machine-readable evidence is embedded in this page as <code>model-evidence</code>.</p>
<p>Original runtime: {escape(runtime["deployment"])} / {escape(runtime["deployment_version"])}; reasoning
{escape(runtime["reasoning_effort"])}; verbosity {escape(runtime["text_verbosity"])}. Published rates:
${runtime["pricing"]["input_per_million"]:.2f}/M input, ${runtime["pricing"]["cached_input_per_million"]:.2f}/M cached input,
${runtime["pricing"]["output_per_million"]:.2f}/M output. Rated costs are not reconciled invoices.</p>
<p>Inspect the local input and result artifacts: {artifact_links}.</p>
<p>Reproduce: <code>python -m token_yield.report --model-run runs\\{escape(model_dir.name)}
--output docs\\customer-model-report.html</code> (add the new catalog and prospective-run arguments for new-test results).</p>
</footer></main><script type="application/json" id="model-evidence">{evidence_json}</script></body></html>"""


def create_parser() -> argparse.ArgumentParser:
    """Create the offline customer evidence-report CLI."""
    parser = argparse.ArgumentParser(description="Render the saved customer model as a self-contained HTML report.")
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--catalog", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments" / "customer_requests" / "catalog.json")
    parser.add_argument("--new-catalog", type=Path)
    parser.add_argument("--prospective-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the report without running models or fitting coefficients."""
    args = create_parser().parse_args(argv)
    html = customer_html_report(args.model_run, args.catalog, new_catalog_path=args.new_catalog,
                                prospective_dir=args.prospective_run, document_dir=args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output.resolve())
    return 0


def markdown_report(forecast: ProjectForecast,
                    dollars_per_million_tokens: float = 3.0) -> str:
    """Render a forecast as a Markdown report."""
    lines = []
    lines.append(f"# Token Yield Forecast: {forecast.project_name}")
    lines.append("")

    lines.append("## Task Breakdown")
    lines.append("")
    lines.append("| Type | Complexity | Multiplier | Tokens | Samples |")
    lines.append("|------|------------|-----------|--------|---------|")

    for pred, count in zip(forecast.task_predictions, forecast.task_counts):
        label = pred.task_type
        if count > 1:
            label += f" x{count}"
        lines.append(
            f"| {label} | {pred.complexity.value} | {pred.multiplier:.1f}x | "
            f"{_fmt_tokens(pred.total_predicted)} | {pred.basis_samples} |"
        )

    lines.append("")
    lines.append("## Budget Summary")
    lines.append("")
    lines.append(f"- **Base tokens:** {_fmt_tokens(forecast.total_tokens)}")
    lines.append(f"- **Harness overhead:** {_fmt_tokens(forecast.interaction_overhead_tokens)}")
    lines.append(f"- **Total predicted:** {_fmt_tokens(forecast.total_with_overhead)}")
    lines.append(f"- **Confidence range:** {_fmt_tokens(forecast.total_tokens_low)} – "
                 f"{_fmt_tokens(forecast.total_tokens_high)}")
    lines.append(f"- **Estimated time:** {forecast.estimated_hours:.1f} hours")
    lines.append("")

    cost = forecast.cost_at_rate(dollars_per_million_tokens)
    cost_low, cost_high = forecast.cost_range(dollars_per_million_tokens)
    lines.append(f"## Cost Estimate")
    lines.append("")
    lines.append(f"At **${dollars_per_million_tokens}/M tokens**:")
    lines.append(f"- Predicted: **{_fmt_cost(cost)}**")
    lines.append(f"- Range: {_fmt_cost(cost_low)} – {_fmt_cost(cost_high)}")
    lines.append("")

    if forecast.uncalibrated:
        lines.append("## ⚠️ Incomplete budget")
        lines.append("")
        lines.append("No calibration data for: "
                     + ", ".join(f"`{t}`" for t in forecast.uncalibrated))
        lines.append("")
        lines.append("Those tasks are **not** included in the totals above. "
                     "Measure them before treating this as the project's cost.")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
