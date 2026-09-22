"""Create catalog-designed build records and import isolated Copilot usage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from token_yield.build_simulations import (
    BUILD_SCOPE, FAMILIES, REWARD_WEIGHTS, STAFF, combination_design, complete_point,
    planned_point, point_id, pseudo_outcomes, read_usage, write_json,
)
from token_yield.marketplace_agent_contracts import ATOMS, contracts

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "build_simulations"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initialize(root: Path) -> None:
    campaign = read(root / "campaign.json")
    registry = [{"basic_functionality_id": item["feature_id"],
                 "basic_functionality": FAMILIES[item["feature_id"]],
                 "type": item["id"], "type_name": item["name"],
                 "build_requirement": BUILD_SCOPE[item["id"]],
                 "origin": "independent_functionality_taxonomy"}
                for item in contracts()]
    write_json(root / "functionality_registry.json", registry)
    design = combination_design()
    write_json(root / "combination_design.json", {
        "ordering": "One canonical membership per combination; chosen execution order is stored on each build.",
        "counts_by_size": dict(Counter(len(item["types"]) for item in design)), "design": design,
    })
    selections = [[item["type"]] for item in registry] + campaign["selected_compositions"]
    for types in selections:
        target = root / "data_points" / (point_id(types) + ".json")
        if not target.exists():
            write_json(target, planned_point(types))
    references = read(root / "real_use_cases.json")
    catalog = {item["type"]: item for item in registry}
    for case in references["cases"]:
        identifier = "real_case_" + case["id"]
        target = root / "data_points" / (identifier + ".json")
        if not target.exists():
            mapped = case["mapped_types"]
            write_json(target, {
                "id": identifier, "origin": "real_use_case_reference",
                "status": "reference_only_no_build_measurement",
                "input_features": {
                    "basic_functionalities": [catalog[item]["basic_functionality"] for item in mapped],
                    "types": mapped, "functionality_count": len(mapped),
                    **{name: None for name in STAFF}, "estimated_months_to_finish": None,
                },
                "metadata": {**case, "source_url": references["source_url"],
                             "source": references["source"], "verification": references["verification"],
                             "mapping_status": "Post-hoc analyst hypothesis, not validated implementation scope.",
                             "catalog_design_dependency": False},
                "build_token_usage": None, "azure_cost_scenarios": [],
                "pseudo_outcomes": pseudo_outcomes(identifier),
                "training_use": "Excluded from build-token regression without actual build telemetry.",
            })
    for path in (root / "data_points").glob("*.json"):
        point = read(path)
        outcomes = point["pseudo_outcomes"]
        if outcomes["is_pseudo"]:
            outcomes.setdefault("feedback_schema", "build-outcomes-0to5-v1")
            outcomes.setdefault("logged_action_propensity", None)
            outcomes.setdefault("off_policy_evaluation_eligible", False)
            write_json(path, point)
    fields = [
        ("basic_functionality_ids", "categorical list", "Which independently defined basic functions are built."),
        ("basic_functionalities", "display names", "Human-readable function names, not case-study-derived labels."),
        ("types", "categorical ordered list", "Selected variations in their integrated execution order."),
        ("functionality_count", "count", "Number of distinct basic functions, from one through four."),
        ("integration_edge_count", "count", "Declared data handoffs; integrated builds are measured directly."),
        ("shared_schema_count", "count", "Planned common input/output schemas."),
        ("shared_validation_layer_count", "count", "Planned reusable validation layers."),
        ("planned_acceptance_case_count", "count", "Planned cases known before construction, not the observed test count."),
        ("artifact_kind", "categorical", "Bounded runnable Python CLI reference implementation."),
        ("llm_integration", "categorical", "Injected callback contract; live model execution is not required to measure construction."),
        ("estimated_months_to_finish", "months", "Declared human-delivery scenario, not observed agent runtime."),
    ]
    fields += [(name, "FTE", "Declared planning assumption, not observed employment or staffing.") for name in STAFF]
    fields += [(f"has_type_{item['type']}", "binary", "Frozen catalog indicator.") for item in registry]
    fields += [(f"planned_operation_{atom}", "count", "Taxonomy-based planned operation count, not summed token consumption.") for atom in ATOMS]
    write_json(root / "feature_dictionary.json", {
        "recording_status": "Retrospective encoding of intended build scope and declared planning scenarios; not a prospective staffing intervention.",
        "input_features": [{"name": name, "unit": unit, "meaning": meaning, "known_before_build": True}
                           for name, unit, meaning in fields],
        "token_targets": ["build_token_usage.input_tokens", "build_token_usage.output_tokens"],
        "derived_not_training_inputs": ["azure_cost_scenarios", "actual artifact sizes",
                                       "actual test counts", "pseudo_outcomes"],
        "reward": {"weights": REWARD_WEIGHTS, "range": [0, 1], "score_range": [0, 5],
                   "anchors": ["0: unacceptable", "1: weak", "2: partial", "3: meets baseline",
                               "4: strong", "5: exceptional"],
                   "roi_satisfaction": "Ordinal perceived-return score, not verified financial ROI.",
                   "policy_type": "One-step contextual-bandit reward; not foundation-model reinforcement fine-tuning.",
                   "promotion": "Synthetic-only experiments; no automatic real-policy promotion."},
    })
    summarize(root)


def import_build(root: Path, database: Path, identifier: str) -> None:
    campaign = read(root / "campaign.json")
    agent = campaign["builders"][identifier]
    point_path = root / "data_points" / (identifier + ".json")
    point = read(point_path)
    directory = root / "builds" / identifier
    names = ("implementation.py", "test_implementation.py", "example_input.json", "build_manifest.json")
    for name in names:
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete build: missing {identifier}/{name}")
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_implementation.py"]
    tests = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
    cli_args = campaign.get("cli_arguments", {}).get(identifier, ["example_input.json"])
    cli = subprocess.run([sys.executable, "-B", "implementation.py", *cli_args],
                         cwd=directory, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=30)
    if tests.returncode or cli.returncode:
        raise ValueError(f"Build validation failed for {identifier}:\n{tests.stderr}\n{cli.stderr}")
    json.loads(cli.stdout)
    evidence = {
        "tests_passed": True, "test_command": "python -B -m unittest discover -s . -p test_implementation.py",
        "test_output": tests.stdout + tests.stderr, "cli_exit_code": cli.returncode,
        "cli_arguments": cli_args,
        "cli_output": json.loads(cli.stdout),
        "artifact_directory": str(directory.relative_to(ROOT)),
        "artifact_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names},
        "scope": "Runnable bounded implementation, not enterprise production delivery.",
        "external_provider_calls_during_validation": 0,
    }
    usage = read_usage(database, campaign["parent_session_id"], agent)
    completed = complete_point(point, usage, agent, read(root / "azure_rate_cards.json")["cards"], evidence)
    write_json(point_path, completed)
    print(json.dumps({"id": identifier, "tokens": completed["build_token_usage"]["total_tokens"],
                      "provider_calls": completed["build_token_usage"]["provider_call_count"]}))


def summarize(root: Path) -> None:
    points = [read(path) for path in sorted((root / "data_points").glob("*.json"))]
    rows = []
    for point in points:
        usage = point.get("build_token_usage") or {}
        features = point["input_features"]
        costs = point.get("azure_cost_scenarios") or []
        rows.append({
            "id": point["id"], "origin": point["origin"], "status": point["status"],
            "split": point.get("split", "not_eligible"), "split_group": point.get("split_group"),
            "basic_functionalities": " + ".join(features["basic_functionalities"]),
            "types": " + ".join(features["types"]), **{name: features.get(name) for name in STAFF},
            "estimated_months_to_finish": features.get("estimated_months_to_finish"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "gpt54_no_cache_usd": next((item["no_cache_estimate_usd"] for item in costs if item["target_model"] == "gpt-5.4"), None),
            **point["pseudo_outcomes"]["scores_0_to_5"],
            "pseudo_reward": point["pseudo_outcomes"]["reward_0_to_1"],
            "pseudo_roi_pct": point["pseudo_outcomes"]["pseudo_roi_pct"],
        })
    if rows:
        with (root / "data_points.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    measured = [point for point in points if point["status"] == "measured_build_passed"]
    design_path = root / "combination_design.json"
    if design_path.exists():
        design = read(design_path)
        for item in design["design"]:
            matches = [point["id"] for point in measured if point["split_group"] == item["split_group"]]
            item["measured_data_point_ids"] = matches
            item["status"] = "one_or_more_orders_measured" if matches else "planned_unmeasured"
        write_json(design_path, design)
    write_json(root / "coverage.json", {
        "measured_points": len(measured),
        "measured_by_composition_size": dict(Counter(point["input_features"]["functionality_count"] for point in measured)),
        "measured_split_counts": {split: sum(point["split"] == split for point in measured)
                                  for split in ("train", "validation", "test")},
        "construction_route": "subscription_subagents",
        "direct_foundry_or_deepseek_calls_in_this_wave": 0,
        "all_16_standalone_types_measured": len([point for point in measured if point["input_features"]["functionality_count"] == 1]) == 16,
        "independent_combinatorial_design_points": len(combination_design()),
        "real_use_case_reference_points": sum(point["origin"] == "real_use_case_reference" for point in points),
        "unmeasured_combinations_are_not_predictions": True,
        "ready_for_generalization_claims": False,
        "reason": "Initial construction wave lacks repeat builds and adequate unseen-combination evaluation. No build-token model has been trained or promoted.",
    })
    def cell(value):
        return html.escape("unknown" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value))

    headings = ["Point", "Status", "Basic functionalities", "Types", "DS / architect / consultant / engineer FTE",
                "Months", "Build input", "Build output", "Build total",
                "GPT-5.4 USD (no cache)", "Pseudo scores: satisfaction / impact / ROI / future / quality",
                "Pseudo reward", "Pseudo ROI %"]
    body = []
    for row in rows:
        values = [row["status"], row["basic_functionalities"], row["types"],
                  " / ".join(cell(row[name]) for name in STAFF), row["estimated_months_to_finish"],
                  row["input_tokens"], row["output_tokens"], row["total_tokens"],
                  row["gpt54_no_cache_usd"], " / ".join(str(row[name]) for name in REWARD_WEIGHTS),
                  row["pseudo_reward"], row["pseudo_roi_pct"]]
        body.append(f'<tr><td><a href="data_points/{html.escape(row["id"])}.json">{cell(row["id"])}</a></td>'
                    + "".join(f"<td>{cell(value)}</td>" for value in values) + "</tr>")
    dictionary = read(root / "feature_dictionary.json")
    feature_rows = "".join("<tr>" + "".join(f"<td>{cell(feature[key])}</td>" for key in ("name", "unit", "meaning")) + "</tr>"
                           for feature in dictionary["input_features"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Measured build simulation data points</title>
<style>body{{font:15px/1.5 system-ui;margin:32px;color:#183c32;background:#fafbf8}}h1,h2{{color:#143c31}}
table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}th,td{{padding:9px;border:1px solid #d8e2da;text-align:left}}
th{{background:#eaf2e9}}.scroll{{overflow-x:auto}}.notice{{padding:16px;background:#eef3e6;border-left:4px solid #39724e}}
a{{color:#176754}}</style>
<h1>Actual build-token data, not workload-token estimates</h1>
<p><b>{len(measured)} measured builds</b> &middot; 16 catalog types &middot; 1,455 designed memberships &middot; 14 real-use-case references</p>
<div class="notice">Subscription agents constructed and tested runnable reference implementations.
Input/output counters are from their actual isolated sessions. Azure costs are counterfactual reference-price scenarios.
Staffing/months are assumptions. Every client score, impact, value and ROI below is explicitly pseudo.
Real-use-case construction tokens remain unknown. Unmeasured combinations are not predictions.</div>
<p><a href="data_points.csv">Open CSV</a> &middot; <a href="feature_dictionary.json">Feature dictionary</a>
&middot; <a href="coverage.json">Coverage</a> &middot; <a href="real_use_cases.json">Case references</a>
&middot; <a href="README.md">Method and limitations</a></p>
<h2>Data points</h2><div class="scroll"><table><thead><tr>{"".join(f"<th>{cell(name)}</th>" for name in headings)}</tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<h2>Input features ({len(dictionary["input_features"])})</h2>
<table><tr><th>Name</th><th>Unit</th><th>Meaning</th></tr>{feature_rows}</table>
<h2>Pseudo 0-5 client feedback</h2><p>Client satisfaction 30%; recognized impact 25%; ROI satisfaction 20%;
future impact potential 10%; delivery quality 15%. Reward = weighted score / 5.
Ordinal ROI satisfaction is not financial ROI. Verified customer ROI is unknown. No real policy is promoted.</p></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "import", "summarize"))
    parser.add_argument("--root", type=Path, default=CORPUS)
    parser.add_argument("--copilot-db", type=Path)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    if args.action == "initialize":
        initialize(args.root)
    elif args.action == "import":
        if args.copilot_db is None or not args.ids:
            parser.error("import requires --copilot-db and explicit completed --ids")
        for identifier in args.ids:
            import_build(args.root, args.copilot_db, identifier)
        summarize(args.root)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
