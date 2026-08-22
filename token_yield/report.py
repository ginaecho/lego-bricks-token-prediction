"""Report generation — render forecasts as readable text or structured output."""

from __future__ import annotations

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
