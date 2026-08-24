"""Tests for mining real repositories and classifying the work they contain."""

import pytest

from token_yield.mine import (
    ChangeFeatures, CoverageReport, classify, coverage, distribution,
    mine_repo, MinedTask,
)


def feats(**kw) -> ChangeFeatures:
    return ChangeFeatures(**kw)


# ── the classifier is transparent and ordered ───────────────────────────

def test_all_test_files_is_the_strongest_signal():
    c = classify(feats(message="anything at all", files=3, test_files=3,
                       insertions=50))
    assert c.kind == "test_write"
    assert c.confidence > 0.9
    assert "every changed file" in c.why


def test_all_doc_files_classifies_as_docs():
    c = classify(feats(message="whatever", files=2, doc_files=2, insertions=20))
    assert c.kind == "docs"
    assert c.confidence > 0.9


def test_composition_outranks_the_subject_line():
    """A fix-shaped message must not override a change made only of tests."""
    c = classify(feats(message="fix the parser", files=2, test_files=2,
                       insertions=30))
    assert c.kind == "test_write"


def test_fix_subject_line():
    c = classify(feats(message="Fix crash on empty input", files=2,
                       code_files=2, insertions=10, deletions=4))
    assert c.kind == "bug_fix"


def test_feature_subject_line():
    c = classify(feats(message="feat: add retry support", files=3,
                       code_files=3, insertions=90))
    assert c.kind == "feature"


def test_refactor_subject_line():
    c = classify(feats(message="refactor: split the module", files=4,
                       code_files=4, insertions=60, deletions=55))
    assert c.kind == "refactor"


def test_all_new_files_reads_as_a_feature():
    c = classify(feats(message="untagged subject", files=2, code_files=2,
                       new_files=2, insertions=80))
    assert c.kind == "feature"
    assert c.confidence < 0.7          # weaker evidence, honestly scored


def test_balanced_churn_reads_as_refactor():
    c = classify(feats(message="untagged subject", files=2, code_files=2,
                       insertions=50, deletions=50))
    assert c.kind == "refactor"


def test_unmatched_change_is_named_not_guessed():
    c = classify(feats(message="untagged subject", files=1, code_files=1,
                       insertions=100, deletions=2))
    assert c.kind == "code_change"
    assert c.confidence <= 0.3
    assert "no rule matched" in c.why


def test_empty_change():
    c = classify(feats(message="empty", files=0))
    assert c.kind == "unknown"
    assert c.confidence == 0.0


def test_every_classification_carries_its_evidence():
    for f in (feats(message="fix it", files=1, code_files=1, insertions=3),
              feats(message="docs: tidy", files=1, doc_files=1, insertions=3),
              feats(message="x", files=1, insertions=1)):
        assert classify(f).why


# ── features ────────────────────────────────────────────────────────────

def test_lines_changed_and_churn():
    f = feats(insertions=40, deletions=10)
    assert f.lines_changed == 50
    assert f.churn_ratio == pytest.approx(0.25)


def test_churn_ratio_safe_on_zero_insertions():
    assert feats(insertions=0, deletions=9).churn_ratio == 0.0


# ── mined tasks carry fittable signals ──────────────────────────────────

def test_mined_task_exposes_signals_for_pricing():
    t = MinedTask("bug_fix", 0.75, "why", feats(files=3, insertions=40,
                                                deletions=10))
    s = t.signals
    assert s["files"] == 3
    assert s["lines"] == 50
    assert s["bytes"] > s["lines"]     # bytes is the signal that predicted best
    assert all(v > 0 for v in s.values())


def test_signals_never_go_to_zero():
    t = MinedTask("docs", 0.9, "why", feats(files=0, insertions=0, deletions=0))
    assert all(v >= 1 for v in t.signals.values())


# ── distribution and coverage ───────────────────────────────────────────

def _tasks():
    mk = lambda k, n: [MinedTask(k, 0.8, "w", feats(files=1, insertions=10))
                       for _ in range(n)]
    return mk("docs", 5) + mk("bug_fix", 3) + mk("test_write", 2)


def test_distribution_shares_sum_to_one():
    d = distribution(_tasks())
    assert sum(v["share"] for v in d.values()) == pytest.approx(1.0)
    assert d["docs"]["count"] == 5


def test_distribution_is_ordered_by_frequency():
    assert list(distribution(_tasks()))[0] == "docs"


def test_coverage_reports_what_cannot_be_priced():
    rep = coverage(_tasks(), measured_kinds=["test_write"])
    assert rep.covered_share == pytest.approx(0.2)
    assert not rep.backlog[0][0] == "test_write"
    assert rep.backlog[0][0] == "docs"          # biggest unmeasured share first


def test_coverage_backlog_is_ordered_by_share_unlocked():
    rep = coverage(_tasks(), measured_kinds=["test_write"])
    shares = [s for _, s in rep.backlog]
    assert shares == sorted(shares, reverse=True)


def test_full_coverage_leaves_an_empty_backlog():
    rep = coverage(_tasks(), ["docs", "bug_fix", "test_write"])
    assert rep.covered_share == pytest.approx(1.0)
    assert rep.backlog == []


def test_coverage_summary_names_the_gap():
    text = coverage(_tasks(), ["test_write"]).summary()
    assert "NOT MEASURED" in text
    assert "backlog" in text.lower()


# ── mining a real repository ────────────────────────────────────────────

def test_mine_this_repo():
    tasks = mine_repo("/home/user/harness-dose", limit=25, repo="harness-dose")
    assert tasks, "this repository has history to mine"
    assert all(t.repo == "harness-dose" for t in tasks)
    assert all(t.sha for t in tasks)
    assert all(t.kind for t in tasks)


def test_mine_missing_repo_returns_empty():
    assert mine_repo("/nonexistent/path/here") == []


def test_mined_kinds_are_all_classifiable():
    tasks = mine_repo("/home/user/harness-dose", limit=25)
    assert all(t.kind != "" for t in tasks)
    assert all(0.0 <= t.confidence <= 1.0 for t in tasks)


def test_mining_supplies_shape_but_never_cost():
    """Commits were written by people; no token count may be inferred from them."""
    tasks = mine_repo("/home/user/harness-dose", limit=10)
    for t in tasks:
        assert not hasattr(t, "tokens")
