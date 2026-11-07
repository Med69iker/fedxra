import math

import numpy as np
import pytest

from powersnr.metrics import attack_success, fidelity_metrics, importance_ranks, weight_shares

NAMES = [f"f{i}" for i in range(6)]
V = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])


def test_identical():
    m = fidelity_metrics(V, V, NAMES, target_feature="f0")
    assert m["rmse_raw"] == 0 and m["rmse_norm"] == 0
    assert np.isclose(m["cosine"], 1.0)
    assert np.isclose(m["kendall_tau"], 1.0) and np.isclose(m["spearman"], 1.0)
    assert m["top3_jaccard"] == 1.0 and m["top5_jaccard"] == 1.0
    assert m["target_rank_true"] == 1 and m["target_rank_shift"] == 0


def test_reversed():
    m = fidelity_metrics(V[::-1].copy(), V, NAMES)
    assert np.isclose(m["kendall_tau"], -1.0) and np.isclose(m["spearman"], -1.0)
    assert m["top3_jaccard"] == 0.0
    assert math.isnan(m["target_rank_shift"]) and math.isnan(m["target_rank_true"])


def test_scale_invariance():
    phi = V + np.array([0.05, -0.02, 0.0, 0.01, 0.0, 0.0])
    a = fidelity_metrics(phi, V, NAMES)
    b = fidelity_metrics(7.0 * phi, V, NAMES)
    assert np.isclose(a["rmse_norm"], b["rmse_norm"]) and np.isclose(a["cosine"], b["cosine"])
    assert not np.isclose(a["rmse_raw"], b["rmse_raw"])


def test_zero_guards():
    m = fidelity_metrics(np.zeros(6), V, NAMES, target_feature="f1")
    assert math.isnan(m["rmse_norm"]) and math.isnan(m["cosine"]) and math.isnan(m["kendall_tau"])
    assert np.isfinite(m["rmse_raw"])


def test_target_rank_shift():
    phi = np.array([0.01, 0.3, 0.1, 0.05, 0.03, 0.02])  # f0 drops from rank 1 to rank 6
    m = fidelity_metrics(phi, V, NAMES, target_feature="f0")
    assert m["target_rank_true"] == 1 and m["target_rank_est"] == 6 and m["target_rank_shift"] == 5
    assert attack_success(m, "feature_hiding") == 1.0
    assert attack_success(m, "feature_promotion") == 0.0
    phi2 = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.9])  # f5 rises from 6 to 1
    m2 = fidelity_metrics(phi2, V, NAMES, target_feature="f5")
    assert m2["target_rank_shift"] == -5
    assert attack_success(m2, "feature_promotion") == 1.0
    assert attack_success(m2, "mimic") == 0.0


def test_ties_average_ranks():
    r = importance_ranks(np.array([3.0, 1.0, 3.0, 0.0]))
    assert np.allclose(r, [1.5, 3.0, 1.5, 4.0])


def test_attack_success_untargeted():
    assert math.isnan(attack_success({"rmse_norm": 0.1}, "scaling"))
    assert attack_success({"rmse_norm": 0.3, "rmse_norm_ref": 0.1}, "scaling") == 1.0
    assert attack_success({"rmse_norm": 0.15, "rmse_norm_ref": 0.1}, "frozen") == 0.0
    assert math.isnan(attack_success({"target_rank_shift": float("nan")}, "feature_hiding"))
    assert math.isnan(attack_success({}, "none"))
    with pytest.raises(ValueError):
        attack_success({}, "bogus")


def test_weight_shares():
    s = weight_shares(np.array([1.0, 1.0, 2.0, 4.0]), ["honest", "honest", "minority", "attacker"])
    assert np.isclose(s["w_honest"], 0.25) and np.isclose(s["w_minority"], 0.25) and np.isclose(s["w_attacker"], 0.5)
    s2 = weight_shares(np.array([0.5, 0.5]), ["honest", "honest"])
    assert s2["w_attacker"] == 0.0 and s2["w_honest"] == 1.0
    n = weight_shares(np.full(3, np.nan), ["honest", "minority", "attacker"])
    assert all(math.isnan(v) for v in n.values())
    with pytest.raises(ValueError):
        weight_shares(np.ones(2), ["honest"])
