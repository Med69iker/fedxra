"""Tests for Power-SNR v3, the added baselines (FoolsGold, centered clipping, Bulyan) and attacks
(ALIE, min-max, white-box coalition)."""
from __future__ import annotations

import numpy as np
import pytest

from powersnr import attacks as atk
from powersnr.aggregators import _cw_huber, _wquantile, aggregate
from powersnr.types import ClientStats

D = 10
BASE = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03, 0.01])


def bank_reports(rng, k=10, noise=0.03):
    mus = np.abs(BASE + rng.normal(0, noise, (k, D)))
    s2 = rng.uniform(0.5e-3, 1.5e-3, k)
    n = rng.integers(300, 900, k).astype(float)
    return mus, s2, n


def test_v3_own_sigma2_never_sets_own_weight():
    rng = np.random.default_rng(0)
    mus, s2, n = bank_reports(rng)
    w0 = aggregate("psnr_v3", mus, s2, n).weights
    for val in [0.0, 1e-12, 10 * s2[3]]:
        s2b = s2.copy()
        s2b[3] = val
        np.testing.assert_allclose(aggregate("psnr_v3", mus, s2b, n, dedup=False).weights,
                                   aggregate("psnr_v3", mus, s2, n, dedup=False).weights)
    assert w0.sum() == pytest.approx(1.0)


def test_v3_zero_variance_attacker_gets_no_more_than_cap():
    rng = np.random.default_rng(1)
    mus, s2, n = bank_reports(rng)
    s2[0] = 0.0
    r = aggregate("psnr_v3", mus, s2, n)
    kept = r.weights[r.weights > 0]
    assert r.weights[0] <= 3.0 * np.median(kept) + 1e-12  # the cap, whatever the claimed sigma2


def test_v3_weights_follow_sample_size():
    rng = np.random.default_rng(2)
    mus, s2, _ = bank_reports(rng, noise=0.01)
    n = np.array([100, 200, 300, 400, 500, 500, 500, 500, 500, 500], float)
    w = aggregate("psnr_v3", mus, s2, n, dedup=False).weights
    assert w[0] < w[1] < w[2] < w[3] < w[4]


def test_cw_huber_bounds_coalition_on_one_coordinate():
    """A coalition setting coordinate 0 to zero moves the Huber location by at most iters * W_A * c * s."""
    rng = np.random.default_rng(3)
    H = np.abs(BASE + rng.normal(0, 0.03, (8, D)))
    coalition = H.mean(0)
    coalition[0] = 0.0
    P = np.vstack([H, coalition, coalition])
    w = np.full(10, 0.1)
    est = _cw_huber(P, w, c=0.5, scale_q=0.8)
    med = _wquantile(P[:, 0], w, 0.5)
    s = 1.4826 * _wquantile(np.abs(P[:, 0] - med), w, 0.8)
    honest_only = _cw_huber(H, np.full(8, 1 / 8), c=0.5, scale_q=0.8)[0]
    assert est[0] >= honest_only - 3 * 0.2 * 0.5 * s - abs(med - honest_only) - 1e-9
    assert est[0] > 0.5 * H[:, 0].mean()  # the mean would drop by 20%; Huber stays close to the honest value


def test_cw_huber_keeps_sparse_coordinates():
    """Coordinates used by a minority of banks are not zeroed (scale_q > 0.5)."""
    P = np.zeros((10, 3))
    P[:, 0] = 1.0
    P[:4, 1] = 0.5  # 4 of 10 banks use feature 1
    w = np.full(10, 0.1)
    assert _cw_huber(P, w, c=0.5, scale_q=0.8)[1] > 0.05
    assert _cw_huber(P, w, c=0.5, scale_q=0.5)[1] == pytest.approx(0.0, abs=1e-9)


def test_v3_duplicate_grouping_identical_coalition():
    rng = np.random.default_rng(4)
    mus, s2, n = bank_reports(rng)
    mus[7:] = mus[:7].mean(0)
    r = aggregate("psnr_v3", mus, s2, n)
    assert any(set(g) >= {7, 8, 9} for g in r.info["duplicate_groups"])
    assert r.weights[7:].sum() < 0.2


def test_v3_ablation_switches_run():
    rng = np.random.default_rng(5)
    mus, s2, n = bank_reports(rng)
    for kw in [{"combine": "mean"}, {"combine": "geomedian"}, {"weight": "unif"}, {"weight": "snr"},
               {"screens": False}, {"dedup": False}, {"cap": None}, {"scale_q": 0.5}]:
        r = aggregate("psnr_v3", mus, s2, n, **kw)
        assert np.all(np.isfinite(r.phi)) and r.weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        aggregate("psnr_v3", mus, s2, n, combine="bad")


def test_foolsgold_downweights_clones():
    rng = np.random.default_rng(6)
    mus, s2, n = bank_reports(rng, noise=0.1)
    mus[7:] = np.array([0.0, 1, 1, 0, 0, 0, 0, 0, 0, 1.0])
    w = aggregate("foolsgold", mus, s2, n).weights
    assert w[7:].sum() < 0.05


def test_centered_clipping_and_bulyan_resist_one_outlier():
    rng = np.random.default_rng(7)
    mus, s2, n = bank_reports(rng)
    mus[0] = 100.0
    for name in ["centered_clipping", "bulyan"]:
        phi = aggregate(name, mus, s2, n).phi
        assert np.linalg.norm(phi - BASE) < 0.5 * np.linalg.norm(BASE), name


def _honest_stats(rng, k=8):
    mus, s2, n = bank_reports(rng, k)
    return [ClientStats(client_id=i, role="honest", mu=mus[i], sigma2=float(s2[i]), n=int(n[i])) for i in range(k)]


FEAT = [f"f{j}" for j in range(D)]


def test_alie_lowers_target_within_spread():
    rng = np.random.default_rng(8)
    H = _honest_stats(rng)
    out = atk.apply_report_attack("alie", H[0], H, rng, feature_names=FEAT, target_feature="f0", n_attackers=2)
    Hm = np.vstack([h.mu for h in H])
    assert out.mu[0] < Hm[:, 0].mean() and out.mu[0] >= 0
    np.testing.assert_allclose(out.mu[1:], Hm.mean(0)[1:])
    assert out.extra["z"] > 0


def test_minmax_stays_within_max_honest_distance():
    rng = np.random.default_rng(9)
    H = _honest_stats(rng)
    out = atk.apply_report_attack("minmax", H[0], H, rng, feature_names=FEAT, target_feature=None)
    Hm = np.vstack([h.mu for h in H])
    dmax = np.max(np.linalg.norm(Hm[:, None] - Hm[None], axis=-1))
    assert np.max(np.linalg.norm(Hm - out.mu, axis=1)) <= dmax + 1e-9
    assert out.extra["gamma"] > 0 and np.all(out.mu >= 0)


def test_whitebox_variants_family():
    rng = np.random.default_rng(10)
    H = _honest_stats(rng)
    v = atk.whitebox_variants(H, 2, feature_names=FEAT, target_feature="f0", rng=rng)
    assert len(v) == len(atk.WHITEBOX_HIDE) * len(atk.WHITEBOX_CAMO) * len(atk.WHITEBOX_SIGMA2) == 24
    for label, M, S2 in v:
        assert M.shape == (2, D) and S2.shape == (2,) and np.all(M >= 0)
        if label.startswith("h1.0_none"):
            assert np.all(M[:, 0] == 0)
    assert atk.whitebox_score(np.array([0.1, 0.5, 0.4]), 0) == (2, -0.1)


def test_foolsgold_handles_zero_reports():
    rng = np.random.default_rng(11)
    mus, s2, n = bank_reports(rng)
    mus[0] = 0.0
    r = aggregate("foolsgold", mus, s2, n)
    assert np.all(np.isfinite(r.weights)) and np.all(np.isfinite(r.phi))


# --------------------------------------------------------------------------- round-8 design (trimmed stage)
def test_cw_trim_matches_unweighted_trimmed_mean():
    from powersnr.aggregators import _cw_trim
    rng = np.random.default_rng(7)
    P = rng.random((10, 4))
    w = np.ones(10)
    got = _cw_trim(P, w, 0.3)
    want = np.sort(P, axis=0)[3:7].mean(axis=0)
    np.testing.assert_allclose(got, want)


def test_cw_trim_stays_in_honest_range_when_attack_weight_below_beta():
    from powersnr.aggregators import _cw_trim
    rng = np.random.default_rng(8)
    H = rng.random((8, 5))
    P = np.vstack([H, np.full((2, 5), -1e6)])          # 2 of 10 reports, weight 0.2 < beta = 0.3
    out = _cw_trim(P, np.ones(10), 0.3)
    assert np.all(out >= H.min(axis=0) - 1e-12) and np.all(out <= H.max(axis=0) + 1e-12)


def test_v3_default_has_no_profile_screen():
    rng = np.random.default_rng(9)
    mus, s2, n = bank_reports(rng)
    mus[2] = BASE[::-1]                                  # far profile, normal mass
    assert 2 in aggregate("psnr_v3", mus, s2, n).info["kept"]
    assert 2 not in aggregate("psnr_v3", mus, s2, n, profile_screen=True).info["kept"]


def test_v3_cap_bounds_inflated_n():
    rng = np.random.default_rng(10)
    mus, s2, n = bank_reports(rng)
    n[0] = 1e6
    w = aggregate("psnr_v3", mus, s2, n, dedup=False).weights
    assert w[0] <= 1.5 * np.median(w) + 1e-12


def test_whitebox_strong_family_size_and_validity():
    rng = np.random.default_rng(11)
    mus, s2, n = bank_reports(rng, k=8)
    honest = [ClientStats(client_id=i, role="honest", n=int(n[i]), mu=mus[i], sigma2=float(s2[i])) for i in range(8)]
    feat = [f"f{j}" for j in range(D)]
    paper = atk.whitebox_variants(honest, 2, feature_names=feat, target_feature="f1", rng=np.random.default_rng(0))
    strong = atk.whitebox_variants(honest, 2, feature_names=feat, target_feature="f1", rng=np.random.default_rng(0),
                                   family="strong")
    assert len(paper) == 24 and len(strong) == 360
    for _, M, S2 in strong:
        assert M.shape == (2, D) and np.all(M >= 0) and np.all(np.isfinite(M)) and S2.shape == (2,)
    with pytest.raises(ValueError):
        atk.whitebox_variants(honest, 2, feature_names=feat, target_feature="f1", rng=rng, family="x")
