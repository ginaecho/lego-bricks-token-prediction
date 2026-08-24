"""Tests for predicting hours — and for being honest about how noisy they are."""

import pytest

from token_yield.backtest import noise_floor
from token_yield.duration import (
    MS_PER_HOUR, as_duration_records, duration_models, duration_selection,
    predict_hours, predict_seconds, seconds_for,
)
from token_yield.probes import MEASURED
from token_yield.taxonomy import Provenance, ScopedRecord


def rec(kind, scope, tokens, secs, **signals):
    return ScopedRecord(kind, scope, tokens, duration_seconds=secs,
                        provenance=Provenance.PROBE, signals=signals)


def test_duration_records_carry_milliseconds_into_the_token_field():
    src = [rec("k", 1, 50_000, 12.5)]
    out = as_duration_records(src)
    assert out[0].tokens == 12_500
    assert out[0].kind == "k" and out[0].scope == 1


def test_zero_duration_is_floored_not_dropped():
    """The power fit's log transform needs a positive value."""
    assert as_duration_records([rec("k", 1, 10, 0.0)])[0].tokens == 1


def test_duration_selection_fits_a_known_line():
    rs = [rec("k", x, 1000, 2.0 + 0.5 * x) for x in (1, 2, 4, 8)]
    sel = duration_selection("k", rs)
    assert sel is not None
    assert predict_seconds(sel.model, 4) == pytest.approx(4.0, abs=0.1)


def test_predict_hours_converts():
    rs = [rec("k", x, 1000, 3600.0) for x in (1, 2, 4)]
    sel = duration_selection("k", rs)
    assert predict_hours(sel.model, 2) == pytest.approx(1.0, abs=0.01)
    assert MS_PER_HOUR == 3_600_000.0


def test_untimed_records_yield_no_model():
    assert duration_selection("k", [rec("k", 1, 100, 0.0)]) is None
    assert duration_selection("k", []) is None


def test_duration_models_covers_every_timed_kind():
    models = duration_models(MEASURED)
    assert set(models) == {r.kind for r in MEASURED}


# ── the trap this module exists to close ────────────────────────────────

def test_seconds_for_reads_the_models_own_signal():
    """A duration model may select a different signal than the token model.

    Passing the token model's input to the duration model silently produced a
    64,596-second answer during development. The lookup belongs in one place.
    """
    rs = [rec("k", x, 1000, 2.0 + 0.5 * x, bytes=x * 10_000) for x in (1, 2, 4, 8)]
    sel = duration_selection("k", rs)
    got = seconds_for(sel, {"scope": 4, "bytes": 40_000})
    assert got is not None
    assert got < 100, "read the wrong signal: bytes-sized input into a scope model"


def test_seconds_for_returns_none_when_the_signal_is_absent():
    rs = [rec("k", x, 1000, 2.0 + 0.5 * x) for x in (1, 2, 4)]
    sel = duration_selection("k", rs)
    assert seconds_for(sel, {"bytes": 5000}) is None      # named, never zeroed


def test_seconds_for_handles_a_missing_model():
    assert seconds_for(None, {"scope": 3}) is None


# ── honesty: hours are far noisier than tokens ──────────────────────────

def test_duration_is_measurably_noisier_than_tokens():
    """The claim the figure makes: ~5% for tokens, ~23% for wall-clock."""
    tokens = noise_floor(MEASURED)
    hours = noise_floor(as_duration_records(MEASURED))
    assert tokens is not None and hours is not None
    assert hours > 2 * tokens, "wall-clock should be several times noisier"


def test_measured_duration_floor_is_around_a_fifth():
    assert 0.10 < noise_floor(as_duration_records(MEASURED)) < 0.40
