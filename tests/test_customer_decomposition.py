"""Offline checks for the public-request contract; no model calls."""

import copy
import json
from pathlib import Path

import pytest

from token_yield.customer_decomposition import (
    SCOPING_BRICKS, arm_batches, batch_candidates, build_prompt, check_output,
    compile_plan, delivery_decomposition, load_catalog, load_template,
    quote_features, validate_batches, validate_project, validate_template,
)


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "customer_requests"


@pytest.fixture
def project():
    return load_catalog(EXPERIMENT / "catalog.json")["projects"][0]


@pytest.fixture
def template():
    return load_template(EXPERIMENT / "template.json")


def valid_output(project, operations):
    """Construct oracle fixtures, not observed customer/model outputs."""
    requirements = project["requirements"]
    artifacts = {
        "extract": [{"requirement_id": r["id"], "evidence": r["text"]}
                    for r in requirements],
        "classify": [{"requirement_id": r["id"], "brick": r["brick"], "owner": r["owner"]}
                     for r in requirements],
        "plan": [{"requirement_id": r["id"], "owner": r["owner"],
                  "prerequisite_status": "unverified", "review_required": True,
                  "proposed_artifact": "A proposed scoping artifact requiring human review."}
                 for r in requirements],
        "report": {
            "source_url": project["source_url"],
            "requirement_ids": [r["id"] for r in requirements],
            "exclusions": project["exclusions"], "delivery_status": "not_executed",
            "summary": "This proposed scope requires authorized evidence and human review.",
        },
    }
    return {slug: artifacts[slug] for slug in operations}


def test_six_source_backed_projects_and_frozen_split():
    projects = load_catalog(EXPERIMENT / "catalog.json")["projects"]
    assert len(projects) == 6
    assert [p["split"] for p in projects] == ["train"] * 4 + ["holdout"] * 2
    assert sum(len(p["requirements"]) for p in projects) == 42


def test_compilation_and_prompt_are_deterministic(project, template):
    left, right = compile_plan(project, template), compile_plan(project, template)
    assert left == right
    assert left.sha256 == right.sha256
    assert dict(left.counts) == {"extract": 8, "classify": 8, "plan": 8, "report": 1}
    assert build_prompt(project, template, SCOPING_BRICKS) == build_prompt(
        copy.deepcopy(project), copy.deepcopy(template), SCOPING_BRICKS)
    assert '"split"' not in build_prompt(project, template, SCOPING_BRICKS)
    project["requirements"][0]["text"] += " Additional reviewed scope."
    assert compile_plan(project, template).sha256 != left.sha256


def test_delivery_view_is_not_scoping_count_or_known_workload(project):
    view = delivery_decomposition(project)
    assert sum(view.counts.values()) == len(project["requirements"])
    assert view.source == "reviewed-template-v1"
    assert "unknown" in view.rationale


def test_all_eight_partitions_preserve_work(project, template):
    plans = batch_candidates()
    assert len(plans) == len(set(plans)) == 8
    for layout in plans:
        validate_batches(layout)
        quotes = [quote_features(project, template, batch) for batch in layout]
        assert sum(q["planned_output_tokens"] for q in quotes) == 6400
        for slug, units in compile_plan(project, template).counts:
            assert sum(q["counts"][slug] for q in quotes) == units
    assert arm_batches("batched") in plans
    assert arm_batches("split") in plans


@pytest.mark.parametrize("layout", [
    (), ((),), (("extract",),), (("extract", "extract", "plan", "report"),),
    (("report", "extract", "classify", "plan"),),
])
def test_bad_partitions_rejected(layout):
    with pytest.raises(ValueError):
        validate_batches(layout)


def test_unknown_annotation_or_duplicate_requirement_rejected(project):
    project["requirements"][0]["brick"] = "invented"
    with pytest.raises(ValueError, match="unknown"):
        validate_project(project)
    project["requirements"][0]["brick"] = "review"
    project["requirements"][1]["id"] = "R1"
    with pytest.raises(ValueError, match="unique"):
        validate_project(project)


def test_template_cannot_hide_dependencies_or_change_units(template):
    template["operations"][1]["depends_on"] = ["extract"]
    with pytest.raises(ValueError, match="independent"):
        validate_template(template)
    template["operations"][1]["depends_on"] = []
    template["operations"][1]["units"] = "project"
    with pytest.raises(ValueError, match="units"):
        validate_template(template)


def test_contract_pass_is_not_human_quality_approval(project):
    result = check_output(project, SCOPING_BRICKS,
                          json.dumps(valid_output(project, SCOPING_BRICKS)))
    assert result["contract_passed"]
    assert result["human_review_required"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "owner", "evidence",
                                    "exclusions", "claimed_delivery", "false_review"])
def test_quality_contract_catches_grounding_and_scope_errors(project, defect):
    data = valid_output(project, SCOPING_BRICKS)
    if defect == "missing":
        data["extract"].pop()
    elif defect == "duplicate":
        data["classify"][1] = data["classify"][0]
    elif defect == "owner":
        data["plan"][0]["owner"] = "agent_draft"
    elif defect == "evidence":
        data["extract"][0]["evidence"] = "Invented requirement"
    elif defect == "exclusions":
        data["report"]["exclusions"] = []
    elif defect == "claimed_delivery":
        data["report"]["delivery_status"] = "completed"
    else:
        data["plan"][0]["review_required"] = 1
    result = check_output(project, SCOPING_BRICKS, json.dumps(data))
    assert not result["contract_passed"]
    assert result["errors"]


@pytest.mark.parametrize("output", ["bad JSON", "[]", "null", '{"extra":1}'])
def test_malformed_model_output_is_explicitly_rejected(project, output):
    assert not check_output(project, SCOPING_BRICKS, output)["contract_passed"]
