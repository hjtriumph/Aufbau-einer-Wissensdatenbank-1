from eval.metrics import (
    bootstrap_mean_ci,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    recall_at_k,
    weighted_kappa,
)


def test_recall_and_mrr():
    relevant = {"a", "b"}
    retrieved = ["x", "a", "b"]
    assert recall_at_k(relevant, retrieved, 1) == 0.0
    assert recall_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert mrr_at_k(relevant, retrieved, 10) == 0.5


def test_ndcg_prefers_early_hits():
    relevant = {"a"}
    assert ndcg_at_k(relevant, ["a", "b"], 2) == 1.0
    assert ndcg_at_k(relevant, ["b", "a"], 2) < 1.0


def test_paired_bootstrap_detects_consistent_gain():
    result = paired_bootstrap_diff([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert result["mean_diff"] == 1.0
    assert result["ci_low"] > 0
    assert result["p_value"] == 0.0


def test_bootstrap_ci_is_bounded():
    lo, hi = bootstrap_mean_ci([0.0, 0.5, 1.0])
    assert 0.0 <= lo <= hi <= 1.0


def test_weighted_kappa_perfect_agreement():
    pairs = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    assert weighted_kappa(pairs, [0.0, 0.5, 1.0]) == 1.0
