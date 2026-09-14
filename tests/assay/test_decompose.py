import json

import pytest

from assay.costmodel import CostModel
from assay.decompose.evaluate import evaluate, priced_error, priced_error_units, stability
from assay.decompose.keyword import RULES_SHA256, KeywordEncoder, clause_count, split_clauses
from assay.decompose.parser import Decomposition, parse
from assay.decompose.prompt import build_prompt, prompt_hash
from assay.errors import DecompositionError

from tests.assay.conftest import boot_estimate


def payload(vocab, **units):
    full = vocab.zero_units()
    full.update(units)
    return json.dumps({"units": full, "context_files": [], "confidence": 0.8})


# -- parser: fail closed ----------------------------------------------------------


def test_a_valid_decomposition_parses(vocab):
    d = parse(payload(vocab, Extract=3, Validate=2), vocab)
    assert d.units["Extract"] == 3
    assert d.total_units == 5
    assert d.source == "llm"


def test_a_fenced_code_block_is_tolerated(vocab):
    d = parse(f"```json\n{payload(vocab, Extract=1)}\n```", vocab)
    assert d.units["Extract"] == 1


@pytest.mark.parametrize(
    "text,match",
    [
        ("not json at all", "not valid JSON"),
        ('["a"]', "expected a JSON object"),
        ('{"units": 5}', "must be an object"),
    ],
)
def test_malformed_output_raises_rather_than_guessing(text, match, vocab):
    with pytest.raises(DecompositionError, match=match):
        parse(text, vocab)


def test_a_missing_brick_key_is_refused(vocab):
    units = vocab.zero_units()
    units.pop("Report")
    with pytest.raises(DecompositionError, match="missing"):
        parse(json.dumps({"units": units}), vocab)


def test_an_invented_brick_is_refused(vocab):
    units = vocab.zero_units()
    units["Teleport"] = 1
    with pytest.raises(DecompositionError, match="unknown"):
        parse(json.dumps({"units": units}), vocab)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_non_integer_or_negative_counts_are_refused(bad, vocab):
    units = vocab.zero_units()
    units["Extract"] = bad
    with pytest.raises(DecompositionError):
        parse(json.dumps({"units": units}), vocab)


def test_confidence_outside_the_unit_interval_is_refused(vocab):
    units = vocab.zero_units()
    with pytest.raises(DecompositionError, match="confidence"):
        parse(json.dumps({"units": units, "confidence": 4}), vocab)


def test_a_context_file_that_does_not_exist_is_refused(vocab):
    units = vocab.zero_units()
    body = json.dumps({"units": units, "context_files": ["ghost.txt"]})
    with pytest.raises(DecompositionError, match="do not exist"):
        parse(body, vocab, available_files={"real.txt": 1})


def test_needs_clarification_must_carry_an_all_zero_vector(vocab):
    units = vocab.zero_units()
    units["Extract"] = 1
    body = json.dumps({"units": units, "needs_clarification": True})
    with pytest.raises(DecompositionError, match="all-zero"):
        parse(body, vocab)


def test_out_of_scope_is_a_first_class_answer(vocab):
    body = json.dumps({"units": vocab.zero_units(), "out_of_scope": True})
    d = parse(body, vocab)
    assert d.out_of_scope and d.is_empty


# -- prompt -----------------------------------------------------------------------


def test_the_encoder_reads_the_same_sheet_as_the_labellers(vocab):
    prompt = build_prompt(vocab, "do a thing", corpus_files=["a.txt"])
    assert vocab.render_for_prompt() in prompt, (
        "the model and the humans must read identical wording, or the encoder score "
        "partly measures the gap between two documents"
    )
    assert "a.txt" in prompt
    assert "STRICT JSON" in prompt


def test_prompt_hash_moves_when_the_vocabulary_moves(vocab):
    before = prompt_hash(vocab)
    assert prompt_hash(vocab) == before
    from assay.vocabulary import Brick, Vocabulary

    edited = Vocabulary(
        version=vocab.version,
        bricks=vocab.bricks[:-1]
        + (
            Brick(
                *(
                    getattr(vocab.bricks[-1], f)
                    if f != "definition"
                    else "a different definition"
                    for f in ("name", "category", "bought_as", "definition", "unit", "nearest_neighbour", "examples")
                )
            ),
        ),
        primary=vocab.primary,
    )
    assert prompt_hash(edited) != before


# -- keyword control --------------------------------------------------------------


def test_the_rule_set_is_pinned(vocab):
    """The control is written before gold is opened. A control that gets tuned until it
    agrees is not a control."""
    assert KeywordEncoder(vocab).rules_sha256 == RULES_SHA256
    assert len(RULES_SHA256) == 64


@pytest.mark.parametrize(
    "request_text,expected",
    [
        ("read the filing, pull three fields, write two auditor checks",
         {"Review": 1, "Extract": 3, "Validate": 2}),
        ("find which company reported the 26% decline, then draft a risk note",
         {"Retrieve": 1, "Draft": 1}),
        ("compare these two filings, then write a board summary",
         {"Reconcile": 1, "Report": 1}),
        ("verify that the total equals the sum of the segments",
         {"Validate": 1}),
        ("classify each ticket into billing, technical or account",
         {"Classify": 1}),
        ("fix the mislabelled rows", {"Remediate": 1}),
    ],
)
def test_keyword_boundary_table(request_text, expected, vocab):
    """The confusions the vocabulary card calls out, made executable."""
    got = KeywordEncoder(vocab).decompose(request_text).units
    assert {k: v for k, v in got.items() if v} == expected


def test_a_named_source_is_extract_and_an_unnamed_one_is_retrieve(vocab):
    enc = KeywordEncoder(vocab)
    assert enc.decompose("pull the revenue figure from the 10-K").units["Extract"] == 1
    assert enc.decompose("which filing mentions the restructuring charge").units["Retrieve"] == 1


def test_negation_suppresses_a_brick(vocab):
    units = KeywordEncoder(vocab).decompose("do not draft anything").units
    assert sum(units.values()) == 0


def test_an_unmatched_request_is_out_of_scope_not_a_guess(vocab):
    d = KeywordEncoder(vocab).decompose("hello there")
    assert d.is_empty and d.out_of_scope and d.confidence == 0.0


def test_a_percentage_is_not_a_unit_count():
    assert clause_count("reported a 26% decline") == 1
    assert clause_count("pull three fields") == 3
    assert clause_count("write two auditor checks") == 2


def test_clause_splitting_handles_numbered_sections():
    parts = split_clauses("Answer all sections.\n1. Pull two fields\n2. Run one check")
    assert len(parts) >= 3


def test_the_keyword_encoder_is_always_labelled_as_itself(vocab):
    assert KeywordEncoder(vocab).decompose("read it").source == "keyword"


def test_the_keyword_encoder_is_deterministic(vocab):
    enc = KeywordEncoder(vocab)
    text = "read the filing, pull three fields, write two auditor checks"
    assert enc.decompose(text).units == enc.decompose(text).units


# -- scoring ----------------------------------------------------------------------


def test_a_parse_failure_counts_against_the_encoder(vocab):
    gold = [{"Extract": 1}, {"Extract": 1}]
    report = evaluate(gold, [Decomposition(units={**vocab.zero_units(), "Extract": 1}), None], vocab)
    assert report.parse_failure_pct == pytest.approx(50.0)
    assert report.exact_match_pct == pytest.approx(50.0), (
        "an encoder that parses half the time is not perfect on the half that worked"
    )


def test_confusion_pairs_are_recorded(vocab):
    gold = [{"Retrieve": 1}]
    pred = [Decomposition(units={**vocab.zero_units(), "Extract": 1})]
    report = evaluate(gold, pred, vocab)
    assert report.confusions.get(("Retrieve", "Extract")) == 1


def test_priced_error_charges_expensive_confusions_more(brick_runs, vocab):
    runs = [r for r in brick_runs if r.fittable]
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot_estimate(brick_runs))

    cheap = priced_error_units({"Draft": 1}, {"Review": 1}, model)
    dear = priced_error_units({"Retrieve": 1}, {"Classify": 1}, model)
    assert dear > cheap * 10, "exact-match accuracy would score these two mistakes identically"
    assert priced_error_units({"Retrieve": 1}, {"Retrieve": 1}, model) == 0.0


def test_the_relative_priced_error_is_unusable_on_a_near_free_request(brick_runs, vocab):
    """A per-request percentage divides by the gold cost, which for below-noise bricks is
    almost zero. The aggregate in `evaluate` divides sums instead, for exactly this reason."""
    runs = [r for r in brick_runs if r.fittable]
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot_estimate(brick_runs))
    assert priced_error({"Draft": 1}, {"Review": 1}, model) > 100.0


def test_aggregate_priced_error_is_robust_to_cheap_requests(brick_runs, vocab):
    runs = [r for r in brick_runs if r.fittable]
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot_estimate(brick_runs))
    gold = [{"Retrieve": 1}, {"Draft": 1}]
    pred = [
        Decomposition(units={**vocab.zero_units(), "Retrieve": 1}),
        Decomposition(units={**vocab.zero_units(), "Review": 1}),
    ]
    report = evaluate(gold, pred, vocab, model=model)
    assert report.priced_error_pct is not None
    assert report.priced_error_pct < 20.0, (
        "one wrong free brick must not dominate a set containing a correct expensive one"
    )


def test_a_parse_failure_is_charged_the_whole_job(brick_runs, vocab):
    runs = [r for r in brick_runs if r.fittable]
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot_estimate(brick_runs))
    report = evaluate([{"Retrieve": 1}], [None], vocab, model=model)
    assert report.priced_error_pct == pytest.approx(100.0)


def test_priced_error_is_zero_for_a_perfect_vector(brick_runs, vocab):
    runs = [r for r in brick_runs if r.fittable]
    model = CostModel.fit(runs, "bytes_per_brick", vocab, boot_estimate(brick_runs))
    gold = {"Extract": 3, "Validate": 2}
    assert priced_error(gold, dict(gold), model) == 0.0


def test_report_names_the_worst_brick(vocab):
    gold = [{"Retrieve": 1}, {"Extract": 1}]
    pred = [
        Decomposition(units=vocab.zero_units()),
        Decomposition(units={**vocab.zero_units(), "Extract": 1}),
    ]
    report = evaluate(gold, pred, vocab)
    assert report.worst_brick().brick == "Retrieve"
    assert "worst brick Retrieve" in report.summary()


def test_stability_detects_a_wobbling_encoder(vocab):
    a = Decomposition(units={**vocab.zero_units(), "Extract": 1})
    b = Decomposition(units={**vocab.zero_units(), "Extract": 2})
    assert stability([a, a, a], vocab) == 100.0
    assert stability([a, b, a], vocab) == pytest.approx(50.0)


def test_evaluate_refuses_mismatched_inputs(vocab):
    with pytest.raises(ValueError):
        evaluate([{"Extract": 1}], [], vocab)
