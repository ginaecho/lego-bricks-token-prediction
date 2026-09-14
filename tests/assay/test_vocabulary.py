import pytest

from assay.errors import VocabularyError
from assay.vocabulary import (
    REFERENCE_NAMES,
    load_vocabulary,
    normalise_units,
)

BAD = """
version = 1
primary = ["Nope"]
[[bricks]]
name = "Review"
category = "not-a-category"
bought_as = ""
definition = "d"
unit = "u"
nearest_neighbour = "n"
examples = ["only one"]
"""


def test_reference_vocabulary_loads(vocab):
    assert vocab.names == list(REFERENCE_NAMES)
    assert vocab.version == 1
    assert not vocab.allow_extension


def test_every_brick_has_a_nearest_neighbour(vocab):
    """The field that makes human agreement achievable. Missing one is a defect."""
    for brick in vocab.bricks:
        assert brick.nearest_neighbour.strip()
        assert len(brick.examples) >= 2


def test_primary_bricks_are_the_four_measured_above_noise(vocab):
    assert set(vocab.primary) == {"Retrieve", "Reconcile", "Validate", "Extract"}
    assert all(vocab.is_primary(p) for p in vocab.primary)
    assert not vocab.is_primary("Draft")


def test_zero_units_covers_every_brick(vocab):
    z = vocab.zero_units()
    assert set(z) == set(REFERENCE_NAMES)
    assert set(z.values()) == {0}


def test_render_for_prompt_states_the_counting_rule(vocab):
    text = vocab.render_for_prompt()
    assert "REQUESTED TARGETS" in text
    for name in REFERENCE_NAMES:
        assert f"## {name}" in text
        assert "NOT TO BE CONFUSED WITH:" in text


def test_malformed_vocabulary_reports_every_problem_at_once(tmp_path):
    path = tmp_path / "bricks.toml"
    path.write_text(BAD, encoding="utf-8")
    with pytest.raises(VocabularyError) as exc:
        load_vocabulary(path)
    problems = exc.value.problems
    assert any("category" in p for p in problems)
    assert any("bought_as" in p for p in problems)
    assert any("examples" in p for p in problems)
    assert any("missing reference bricks" in p for p in problems)
    assert any("primary" in p for p in problems)
    assert len(problems) >= 5, "one problem per run is how a vocabulary stays half-specified"


def test_missing_file_is_a_vocabulary_error(tmp_path):
    with pytest.raises(VocabularyError):
        load_vocabulary(tmp_path / "nope.toml")


def test_normalise_units_fills_missing_keys(vocab):
    out = normalise_units({"Extract": 3}, vocab)
    assert out["Extract"] == 3
    assert set(out) == set(REFERENCE_NAMES)
    assert out["Retrieve"] == 0


@pytest.mark.parametrize("bad", [{"Nope": 1}, {"Extract": -1}, {"Extract": 1.5}, {"Extract": True}])
def test_normalise_units_rejects_bad_vectors(bad, vocab):
    with pytest.raises(VocabularyError):
        normalise_units(bad, vocab)
