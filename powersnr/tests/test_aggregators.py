"""Fast synthetic tests for powersnr.aggregators."""
from __future__ import annotations

import numpy as np
import pytest

from powersnr.aggregators import AGGREGATORS, aggregate
from powersnr.types import AggResult

D = 10
BASE = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03, 0.01])
NAN_WEIGHT_RULES = {"median", "trimmed_mean", "centered_clipping", "bulyan"}


def honest(rng, k=4):
    mus = np.abs(BASE + rng.normal(0, 0.02, (k, D)))
    s2 = rng.uniform(0.5e-3, 1.5e-3, k)
    return mus, s2


def with_attacker(rng, att_mu, att_s2=0.0):
    H, s2 = honest(rng)
    mus = np.vstack([H, att_mu])
    return mus, np.r_[s2, att_s2], np.full(mus.shape[0], 1000.0)


def hiding_vector():
    v = BASE.copy()
    v[0] = 0.0
    return v * BASE.sum() / v.sum()


def relerr(phi):
    return np.linalg.norm(phi - BASE) / np.linalg.norm(BASE)


KW = {"root_mu": BASE}


# (1) every aggregator: shape, finiteness, weight normalisation ------------------------------
@pytest.mark.parametrize("name", sorted(AGGREGATORS))
def test_contract_all(name):
    rng = np.random.default_rng(1)
    mus = rng.uniform(0, 1, (7, D))
    s2 = rng.uniform(0, 1e-2, 7)
    s2[2] = 0.0
    n = rng.integers(50, 500, 7).astype(float)
    r = aggregate(name, mus, s2, n, beta=0.2, f=1, m=3, eps=1e-12, tau=0.5, cap=3.0,
                  gate=0.1, unrelated_kw="ignored", **KW)
    assert isinstance(r, AggResult)
    assert r.phi.shape == (D,) and np.all(np.isfinite(r.phi))
    assert r.weights.shape == (7,)
    if name in NAN_WEIGHT_RULES:
        assert np.all(np.isnan(r.weights))
    else:
        assert np.all(r.weights >= 0)
        assert r.weights.sum() == pytest.approx(1.0)
        if name not in ("fltrust_like", "psnr_v3"):  # rescaled vectors / clipped coordinate values
            np.testing.assert_allclose(r.phi, r.weights @ mus, rtol=1e-6, atol=1e-9)
    assert "fallback" in r.info


def test_validation_errors():
    mus = np.ones((3, D))
    with pytest.raises(ValueError):
        aggregate("mean", mus, np.array([1.0, -1.0, 0.0]), np.ones(3))
    with pytest.raises(ValueError):
        aggregate("mean", mus, np.zeros(2), np.ones(3))
    bad = mus.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        aggregate("mean", bad, np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        aggregate("fltrust_like", mus, np.zeros(3), np.ones(3))
    with pytest.raises(KeyError):
        aggregate("nope", mus, np.zeros(3), np.ones(3))


def test_zero_weight_fallback_to_median():
    mus = np.zeros((4, D))
    mus[0] = 1.0
    r = aggregate("psnr_energy_only", np.zeros((4, D)), np.ones(4), np.ones(4))
    assert r.info["fallback"] is True and np.all(np.isnan(r.weights))
    r = aggregate("psnr_base", np.zeros((4, D)), np.zeros(4), np.ones(4), eps=0.0)
    assert r.info["fallback"] is True and np.all(np.isfinite(r.phi))


def test_eps_zero_sigma_zero_is_finite():
    rng = np.random.default_rng(2)
    mus, s2, n = with_attacker(rng, BASE * 1.1, 0.0)
    for name in ["ivw", "psnr_base", "psnr_temp", "dersimonian_laird", "psnr_robust"]:
        r = aggregate(name, mus, s2, n, eps=0.0, use_floor=False)
        assert np.all(np.isfinite(r.phi)) and np.all(np.isfinite(r.weights))


# (2) Zero-Variance Fallacy -----------------------------------------------------------------
def test_zero_variance_fallacy_zero_attacker():
    rng = np.random.default_rng(0)
    mus, s2, n = with_attacker(rng, np.full(D, 1e-8), 0.0)
    r_ivw = aggregate("ivw", mus, s2, n)
    assert r_ivw.weights[-1] > 0.99
    assert relerr(r_ivw.phi) > 0.9
    r_ps = aggregate("psnr_base", mus, s2, n)
    assert r_ps.weights[-1] < 0.01
    assert relerr(r_ps.phi) < 0.1


# (3) frozen attacker (uniform, same L1 mass, sigma2 = 0) -----------------------------------
def test_frozen_attacker():
    rng = np.random.default_rng(0)
    H, _ = honest(np.random.default_rng(0))
    frozen = np.full(D, H.sum(1).mean() / D)
    mus, s2, n = with_attacker(rng, frozen, 0.0)
    assert aggregate("ivw", mus, s2, n).weights[-1] > 0.99
    # Documented vulnerability: psnr_base IS fooled (non-zero energy / zero variance).
    r_base = aggregate("psnr_base", mus, s2, n)
    assert r_base.weights[-1] > 0.99
    r_rob = aggregate("psnr_robust", mus, s2, n)
    assert r_rob.weights[-1] < 0.5
    assert 4 in r_rob.info["capped"]
    # With default settings the consistency filter removes it entirely.
    assert r_rob.weights[-1] == 0.0 and 4 in r_rob.info["filtered"]
    assert relerr(r_rob.phi) < 0.1


# (4) adaptive attacker: huge energy, sigma2 = 0 --------------------------------------------
def test_adaptive_attacker():
    rng = np.random.default_rng(0)
    H, _ = honest(np.random.default_rng(0))
    hv = hiding_vector()
    E = (H ** 2).sum(1).max()
    att = hv * np.sqrt(10 * E / (hv ** 2).sum())
    mus, s2, n = with_attacker(rng, att, 0.0)
    # Expected vulnerability of the un-robustified rule.
    assert aggregate("psnr_base", mus, s2, n).weights[-1] > 0.99
    r = aggregate("psnr_robust", mus, s2, n, cap=3.0)
    pre = r.info["weights_precap"]
    capped = np.minimum(pre, r.info["cap_limit_rel"])
    bound = capped[-1] / capped.sum()
    assert 4 in r.info["capped"]
    assert r.weights[-1] <= bound + 1e-12
    assert r.weights[-1] < 0.5
    # Ablation: a median consensus in the consistency step filters it.
    r_med = aggregate("psnr_robust", mus, s2, n, consensus="median")
    assert r_med.weights[-1] == 0.0


# (5) coordinate-wise robust statistics -----------------------------------------------------
@pytest.mark.parametrize("name", ["median", "trimmed_mean", "geomedian"])
def test_robust_stats_one_outlier(name):
    rng = np.random.default_rng(3)
    mus, s2, n = with_attacker(rng, BASE * 50.0, 1e-3)
    r = aggregate(name, mus, s2, n)
    assert relerr(r.phi) < 0.1
    assert relerr(aggregate("mean", mus, s2, n).phi) > 5


# (6) Krum ----------------------------------------------------------------------------------
def test_krum_selects_honest():
    rng = np.random.default_rng(4)
    mus, s2, n = with_attacker(rng, BASE * 50.0, 1e-3)
    r = aggregate("krum", mus, s2, n)
    assert r.info["selected"][0] != 4 and r.weights[4] == 0
    r = aggregate("multikrum", mus, s2, n)
    assert 4 not in r.info["selected"] and r.weights[4] == 0


def test_krum_small_k_degenerate():
    mus = np.array([BASE, BASE * 1.1])
    r = aggregate("krum", mus, np.zeros(2), np.ones(2))
    assert r.info["krum_degenerate"] is True and np.all(np.isfinite(r.phi))


# (7) permutation invariance ----------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(AGGREGATORS))
def test_permutation_invariance(name):
    rng = np.random.default_rng(5)
    mus, s2, n = with_attacker(rng, hiding_vector() * 3, 0.0)
    n = rng.integers(100, 1000, 5).astype(float)
    perm = rng.permutation(5)
    r1 = aggregate(name, mus, s2, n, **KW)
    r2 = aggregate(name, mus[perm], s2[perm], n[perm], **KW)
    np.testing.assert_allclose(r1.phi, r2.phi, rtol=1e-7, atol=1e-10)
    if name not in NAN_WEIGHT_RULES:
        np.testing.assert_allclose(r1.weights[perm], r2.weights, rtol=1e-7, atol=1e-10)


# (8) tempered Power-SNR --------------------------------------------------------------------
def test_psnr_temp_limits():
    rng = np.random.default_rng(6)
    mus, s2, n = with_attacker(rng, np.full(D, 1e-8), 0.0)
    np.testing.assert_allclose(aggregate("psnr_temp", mus, s2, n, tau=0.0).phi, mus.mean(0))
    np.testing.assert_allclose(aggregate("psnr_temp", mus, s2, n, tau=1.0).phi,
                               aggregate("psnr_base", mus, s2, n).phi)


def test_psnr_legacy_matches_code_txt():
    rng = np.random.default_rng(7)
    mus, s2, n = with_attacker(rng, np.full(D, 1e-8), 0.0)
    sig = np.mean(np.abs(mus), axis=1)
    thr = np.mean(sig) * 0.1
    w = np.where(sig < thr, 0.0, sig ** 2 / (s2 + 1e-6))
    expected = (w[:, None] * mus).sum(0) / w.sum()
    r = aggregate("psnr_legacy", mus, s2, n)
    np.testing.assert_allclose(r.phi, expected)
    assert r.info["gated"] == [4]


# --------------------------------------------------------------------------- psnr_screen
def _screen_setup(seed=0, K=9, d=10):
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 0.1, d)
    mus = np.abs(base + rng.normal(0, 0.03, (K, d)))
    s2 = rng.uniform(0.5e-3, 1.5e-3, K)
    return base, mus, s2, np.full(K, 100.0)


@pytest.mark.parametrize("kind", ["frozen", "zero", "scaling", "adaptive"])
def test_psnr_screen_removes_report_attackers(kind):
    base, mus, s2, n = _screen_setup()
    att = mus[0].copy()
    if kind == "frozen":
        att = np.full_like(att, att.sum() / att.size)
    elif kind == "zero":
        att = np.full_like(att, 1e-8)
    elif kind == "scaling":
        att = 10 * att
    else:  # hide the top feature and boost energy 10x
        att[0] = 0.0
        att = att / att.sum() * mus[0].sum() * np.sqrt(10)
    mus[0] = att
    s2 = s2.copy()
    s2[0] = 0.0
    res = aggregate("psnr_screen", mus, s2, n)
    assert res.weights[0] == 0.0
    assert np.isclose(res.weights.sum(), 1.0)
    honest = aggregate("mean", mus[1:], s2[1:], n[1:]).phi
    assert np.linalg.norm(res.phi / res.phi.sum() - honest / honest.sum()) < 0.05


def test_psnr_screen_bounds_zero_variance_claim():
    # An attacker that copies the honest profile but claims sigma2 = 0 gains at most the floor.
    base, mus, s2, n = _screen_setup()
    s2 = s2.copy()
    s2[0] = 0.0
    res = aggregate("psnr_screen", mus, s2, n, cap=1.5)
    assert res.weights[0] <= 1.5 * np.median(res.weights[res.weights > 0]) + 1e-12


def test_psnr_screen_snr_false_is_equal_weights():
    base, mus, s2, n = _screen_setup()
    res = aggregate("psnr_screen", mus, s2, n, snr=False)
    kept = res.weights > 0
    assert np.allclose(res.weights[kept], 1.0 / kept.sum())


# --------------------------------------------------------------------------- psnr_screen v2 options
V2 = dict(floor="n", shrink="auto", soft=True, stage2="geomedian", z=2.0, cap=2.0)


@pytest.mark.parametrize("kind", ["frozen", "zero", "scaling", "adaptive"])
def test_psnr_screen_v2_removes_report_attackers(kind):
    base, mus, s2, n = _screen_setup()
    att = mus[0].copy()
    if kind == "frozen":
        att = np.full_like(att, att.sum() / att.size)
    elif kind == "zero":
        att = np.full_like(att, 1e-8)
    elif kind == "scaling":
        att = 10 * att
    else:
        att[0] = 0.0
        att = att / att.sum() * mus[0].sum() * np.sqrt(10)
    mus[0] = att
    s2 = s2.copy()
    s2[0] = 0.0
    res = aggregate("psnr_screen", mus, s2, n, **V2)
    assert res.weights[0] < 0.02
    assert np.isclose(res.weights.sum(), 1.0)
    np.testing.assert_allclose(res.phi, res.weights @ mus, rtol=1e-6, atol=1e-9)


def test_psnr_screen_n_floor_ignores_attacker_majority_of_zero_claims():
    # 3 of 9 clients copy honest profiles but claim sigma2 = 0: the n-based floor still bounds them
    base, mus, s2, n = _screen_setup()
    s2 = s2.copy()
    s2[:3] = 0.0
    res = aggregate("psnr_screen", mus, s2, n, floor="n", cap=None)
    assert res.weights[:3].sum() < 0.5


def test_psnr_screen_shrink_auto_is_uniform_for_equal_quality():
    rng = np.random.default_rng(1)
    mus = np.abs(np.linspace(1, 0.1, 10) + rng.normal(0, 0.01, (8, 10)))
    s2 = np.full(8, 1e-3)  # identical reliability -> var(log v) tiny -> lambda ~ 0
    res = aggregate("psnr_screen", mus, s2, np.full(8, 100.0), shrink="auto", z=10.0)
    assert res.info["lambda"] < 0.2


def test_psnr_screen_shrink_auto_keeps_weighting_for_unequal_quality():
    rng = np.random.default_rng(2)
    mus = np.abs(np.linspace(1, 0.1, 10) + rng.normal(0, 0.01, (8, 10)))
    s2 = np.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1])
    res = aggregate("psnr_screen", mus, s2, np.full(8, 100.0), shrink="auto", z=10.0, cap=None)
    assert res.info["lambda"] > 0.8
    assert res.weights[0] > res.weights[-1]


def test_psnr_screen_soft_keeps_moderate_outlier_partially():
    base, mus, s2, n = _screen_setup()
    m0 = mus[0].sum()
    mus[0] = mus[0] * np.r_[np.full(5, 1.6), np.full(5, 0.4)]  # atypical but plausible profile
    mus[0] *= m0 / mus[0].sum()  # same mass, so only the profile screen is involved
    hard = aggregate("psnr_screen", mus, s2, n, z=2.0, snr=False)
    soft = aggregate("psnr_screen", mus, s2, n, z=2.0, snr=False, soft=True, soft_max=10.0)
    assert hard.weights[0] == 0.0 and soft.weights[0] > 0.0


def test_psnr_screen_dedup_groups_identical_coalition():
    base, mus, s2, n = _screen_setup(K=10)
    hide = mus[3:].mean(axis=0).copy()
    hide[6] *= 0.7  # subtle shift: passes the profile screen
    hide *= mus[3:].mean(axis=0).sum() / hide.sum()
    mus[:3] = hide  # 3 colluders, identical vectors, inside the honest mass range
    kw = dict(V2, soft=False)
    plain = aggregate("psnr_screen", mus, s2, n, **kw)
    grouped = aggregate("psnr_screen", mus, s2, n, **kw, dedup=True)
    assert grouped.info["duplicate_groups"] == [[0, 1, 2]]
    assert grouped.weights[:3].sum() < plain.weights[:3].sum()


def test_psnr_screen_dedup_leaves_honest_clients_alone():
    base, mus, s2, n = _screen_setup(K=10)
    res = aggregate("psnr_screen", mus, s2, n, **dict(V2, soft=False), dedup=True)
    assert res.info["duplicate_groups"] == []
