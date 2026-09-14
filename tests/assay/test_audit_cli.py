import json

import pytest

from assay.audit import audit, load_reference
from assay.cli import main
from assay.evidence import EvidenceClass

REFERENCE = "experiments/train_runs.jsonl"


@pytest.fixture(scope="module")
def report(repo_root):
    return audit(load_reference(repo_root / REFERENCE))


# -- the audit --------------------------------------------------------------------


def test_the_reference_campaign_loads(repo_root):
    rows = load_reference(repo_root / REFERENCE)
    assert len(rows) == 39
    assert all(r.model == "claude-haiku-4-5-20251001" for r in rows)


def test_the_null_probe_is_the_published_one(report):
    assert report.boot == pytest.approx(29_821, abs=50)


def test_the_reference_reproduces_its_published_ordering(report):
    """A transcription check: richer forms beat simpler ones on total error, as published."""
    by_form = {r.form: r for r in report.rows}
    assert by_form["bytes_per_brick"].loo_mape_total < by_form["constant"].loo_mape_total
    assert by_form["bytes"].loo_mape_total < by_form["constant"].loo_mape_total
    assert report.best_total().form in ("bytes_per_brick", "per_brick")


def test_the_published_constant_baseline_is_in_the_right_region(report):
    """~6.9% was published for the intercept-only form."""
    assert 5.0 < report.constant().loo_mape_total < 9.0


def test_total_error_flatters_every_form(report):
    """The finding that shapes the entire plan: on this target, everything looks good."""
    assert all(r.loo_mape_total < 10.0 for r in report.rows)


def test_the_variable_portion_tells_a_much_harsher_story(report):
    """Same models, same data -- with the 29,821-token toll taken off both sides."""
    for row in report.rows:
        assert row.loo_vwape > row.loo_mape_total * 3, (
            f"{row.form}: variable error {row.loo_vwape:.1f}% vs total {row.loo_mape_total:.1f}%"
        )


def test_the_constant_explains_none_of_the_variable_portion(report):
    assert report.constant().loo_vwape > 50.0


def test_the_audit_can_never_move_a_gate(report):
    assert report.evidence_class is EvidenceClass.REPLAY
    assert not report.evidence_class.can_move_a_gate


def test_the_audit_states_that_dollars_are_unrecoverable(report):
    assert any("no dollar figure" in n for n in report.notes)


def test_the_rendered_audit_shows_both_columns_and_disclaims_itself(report):
    text = report.render()
    assert "LOO MAPE (total)" in text and "LOO WAPE (variable)" in text
    assert "authorises anything" in text
    assert "replay" in text


def test_the_audit_is_deterministic(repo_root):
    rows = load_reference(repo_root / REFERENCE)
    assert audit(rows).as_dict() == audit(rows).as_dict()


# -- the CLI ----------------------------------------------------------------------


def test_ty_audit_runs_and_exits_zero(capsys):
    assert main(["audit"]) == 0
    out = capsys.readouterr().out
    assert "REFERENCE AUDIT" in out
    assert "decides nothing" in out


def test_ty_audit_json_is_machine_readable(capsys):
    assert main(["--json", "audit"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence_class"] == "replay"
    assert len(payload["forms"]) == 6


def test_ty_vocab_validates(capsys):
    assert main(["vocab"]) == 0
    assert "REQUESTED TARGETS" in capsys.readouterr().out


def test_ty_decompose_encodes_a_request(capsys):
    assert main(["decompose", "read the filing, pull three fields"]) == 0
    out = capsys.readouterr().out
    assert "Review x1" in out and "Extract x3" in out


def test_ty_decompose_says_out_of_scope_rather_than_guessing(capsys):
    assert main(["decompose", "hello there"]) == 0
    assert "out of scope" in capsys.readouterr().out


def test_ty_verdict_exits_non_zero_on_stop(tmp_path, capsys):
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({"M6_blind_mape_pct": 45.0}), encoding="utf-8")
    assert main(["verdict", "--observations", str(path)]) == 1
    assert "VERDICT: STOP" in capsys.readouterr().out


def test_ty_verdict_exits_zero_on_narrow(tmp_path, capsys):
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({"M6_blind_mape_pct": 22.0}), encoding="utf-8")
    assert main(["verdict", "--observations", str(path)]) == 0
    assert "VERDICT: NARROW" in capsys.readouterr().out


def test_ty_verdict_warns_when_the_evidence_cannot_decide(tmp_path, capsys):
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({"M6_blind_mape_pct": 9.0}), encoding="utf-8")
    main(["verdict", "--observations", str(path), "--evidence", "pipeline_only"])
    assert "THIS IS NOT A RESULT" in capsys.readouterr().out


def test_ty_policy_writes_a_hashed_policy(tmp_path, capsys):
    out = tmp_path / "verdict-policy.json"
    assert main(["policy", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["policy_sha256"]) == 64


def test_ty_corpus_index_reports_duplicates_non_zero(tmp_path, capsys):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("identical content", encoding="utf-8")
    assert main(["corpus-index", "--dir", str(tmp_path)]) == 1
    assert "DUPLICATE" in capsys.readouterr().out


def test_ty_corpus_split_seals_and_reports(tmp_path, capsys):
    for i in range(8):
        (tmp_path / f"d{i}.txt").write_text("x" * (100 * (i + 1)), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    seal = tmp_path / "seal.json"
    main(["corpus-index", "--dir", str(tmp_path), "--out", str(manifest)])
    capsys.readouterr()
    assert main(["corpus-split", "--manifest", str(manifest), "--out", str(seal)]) == 0
    assert "refuses to read them elsewhere" in capsys.readouterr().out
    assert seal.exists()
