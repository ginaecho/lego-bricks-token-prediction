"""The `ty` loop, driven the way the README says a reader can drive it.

The README claims a full quote / run / invoice / refit loop. Until these tests existed
that loop was reachable only from Python, which meant the claim was true of the library
and false of the tool. Each test here runs the command line itself.
"""

from __future__ import annotations

import json

import pytest

from assay.cli import main
from assay.paths import BRICKS

VOCAB = str(BRICKS)


@pytest.fixture(scope="module")
def worked(repo_root, corpus_dir, tmp_path_factory):
    """campaign -> fit, once, for the commands that need a fitted model on disk."""
    work = tmp_path_factory.mktemp("ty")
    seal = work / "seal.json"
    assert main([
        "campaign",
        "--dir", str(corpus_dir),
        "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(work),
        "--seal", str(seal),
        "--new-seal",
    ]) == 0
    assert main([
        "fit",
        "--runs", str(work / "runs.jsonl"),
        "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(work),
        "--iters", "200",
    ]) == 0
    return work


# -- campaign ---------------------------------------------------------------------


def test_campaign_writes_runs_and_says_they_are_simulated(
    repo_root, corpus_dir, tmp_path, capsys
):
    code = main([
        "campaign",
        "--dir", str(corpus_dir),
        "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(tmp_path),
        "--seal", str(tmp_path / "seal.json"),
        "--new-seal",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "runs.jsonl").exists()
    assert "pipeline_only" in out and "SIMULATED" in out
    assert "prove" in out and "nothing about any real model" in out


def test_campaign_refuses_to_invent_a_split_unless_asked(
    repo_root, corpus_dir, tmp_path
):
    """A split chosen at analysis time is not a pre-registration."""
    with pytest.raises(SystemExit, match="corpus-split"):
        main([
            "campaign",
            "--dir", str(corpus_dir),
            "--vocab", str(repo_root / VOCAB),
            "--out-dir", str(tmp_path),
            "--seal", str(tmp_path / "absent.json"),
        ])


def test_campaign_says_so_when_it_creates_a_seal(repo_root, corpus_dir, tmp_path, capsys):
    main([
        "campaign",
        "--dir", str(corpus_dir),
        "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(tmp_path),
        "--seal", str(tmp_path / "seal.json"),
        "--new-seal",
    ])
    assert "not a pre-registration" in capsys.readouterr().out


def test_the_negative_control_is_reachable_from_the_command_line(
    repo_root, corpus_dir, tmp_path, capsys
):
    """`--mode null` is the arm where no brick term exists. It must be one flag away."""
    code = main([
        "campaign",
        "--dir", str(corpus_dir),
        "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(tmp_path),
        "--seal", str(tmp_path / "seal.json"),
        "--new-seal",
        "--mode", "null",
    ])
    assert code == 0
    assert "negative control" in capsys.readouterr().out

    code = main([
        "fit",
        "--runs", str(tmp_path / "runs.jsonl"),
        "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(tmp_path),
        "--iters", "200",
    ])
    assert code == 0
    meta = json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    assert meta["chose_a_decoder"] is False
    assert meta["show_per_brick"] is False


# -- fit ---------------------------------------------------------------------------


def test_fit_writes_the_model_the_band_and_the_verdict(worked):
    for name in ("model.json", "conformal.json", "fit.json"):
        assert (worked / name).exists(), name
    meta = json.loads((worked / "fit.json").read_text(encoding="utf-8"))
    assert meta["evidence_class"] == "pipeline_only"
    assert meta["chosen_form"]
    assert meta["conformal"]["n_calibration"] > 0


def test_fit_reports_both_the_bar_and_whether_it_was_cleared(
    repo_root, corpus_dir, tmp_path, capsys
):
    main([
        "campaign", "--dir", str(corpus_dir), "--vocab", str(repo_root / VOCAB),
        "--out-dir", str(tmp_path), "--seal", str(tmp_path / "seal.json"), "--new-seal",
    ])
    capsys.readouterr()
    main([
        "fit", "--runs", str(tmp_path / "runs.jsonl"),
        "--vocab", str(repo_root / VOCAB), "--out-dir", str(tmp_path), "--iters", "200",
    ])
    out = capsys.readouterr().out
    assert "best baseline" in out
    assert "bar 15%" in out
    assert "cannot move a gate" in out


# -- quote -------------------------------------------------------------------------


def test_quote_prices_a_request_and_shows_its_band(worked, repo_root, capsys):
    code = main([
        "quote", "REQ-1",
        "--brick", "Extract=2", "--brick", "Validate=1",
        "--bytes", "9000",
        "--fit-dir", str(worked),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Extract x2" in out and "Validate x1" in out
    assert "90% band" in out


def test_quote_admits_the_dollars_are_provisional(worked, capsys):
    """No dollar claim may look settled while the price sheet is a placeholder."""
    main(["quote", "R", "--brick", "Extract=1", "--bytes", "1000", "--fit-dir", str(worked)])
    assert "provisional" in capsys.readouterr().out


def test_quote_can_append_to_a_ledger(worked, tmp_path, capsys):
    ledger = tmp_path / "quotes.jsonl"
    for i in (1, 2):
        main([
            "quote", f"R{i}", "--brick", "Extract=1", "--bytes", "1000",
            "--fit-dir", str(worked), "--out", str(ledger),
        ])
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["request_id"] for r in rows] == ["R1", "R2"]
    assert all(r["quote_sha256"] for r in rows)


def test_quote_rejects_a_malformed_brick_argument(worked):
    with pytest.raises(SystemExit, match="Name=Count"):
        main(["quote", "R", "--brick", "Extract", "--bytes", "1000", "--fit-dir", str(worked)])


def test_quote_refuses_a_directory_with_no_fit(tmp_path):
    with pytest.raises(FileNotFoundError, match="ty fit"):
        main(["quote", "R", "--brick", "Extract=1", "--bytes", "1000", "--fit-dir", str(tmp_path)])


# -- invoice -----------------------------------------------------------------------


def _a_run_id(work):
    for line in (work / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["probe_id"].startswith("composite"):
            return row["run_id"]
    raise AssertionError("no composite probe in the campaign")


def test_invoice_reconciles_to_the_measured_total(worked, capsys):
    code = main([
        "invoice", _a_run_id(worked),
        "--runs", str(worked / "runs.jsonl"),
        "--fit-dir", str(worked),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "reconciliation" in out
    assert "TOTAL" in out


def test_invoice_reports_total_and_variable_error_together(worked, capsys):
    """The one failure mode this project exists to avoid is a flattering total error."""
    main([
        "invoice", _a_run_id(worked),
        "--runs", str(worked / "runs.jsonl"), "--fit-dir", str(worked),
    ])
    out = capsys.readouterr().out
    assert "% total" in out and "% variable" in out


def test_invoice_json_carries_the_reconciliation_and_both_errors(worked, capsys):
    main([
        "--json", "invoice", _a_run_id(worked),
        "--runs", str(worked / "runs.jsonl"), "--fit-dir", str(worked),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["reconciles"] is True
    assert "error_pct" in payload and "variable_error_pct" in payload
    assert payload["sum_check"] == pytest.approx(payload["actual_units"])


def test_invoice_refuses_an_unknown_run(worked):
    with pytest.raises(SystemExit, match="no run"):
        main([
            "invoice", "not-a-real-run-id",
            "--runs", str(worked / "runs.jsonl"), "--fit-dir", str(worked),
        ])
