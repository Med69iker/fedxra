import copy

import numpy as np
import pandas as pd
import pytest

from powersnr.attacks import (DATA_ATTACKS, REPORT_ATTACKS, THREAT_MODELS, apply_data_attack,
                              apply_report_attack, make_collusion_reports)
from powersnr.types import ClientStats

D = 6
NAMES = [f"f{i}" for i in range(D)]


def _honest(rng, K=7):
    out = []
    for k in range(K):
        mu = np.abs(rng.normal(size=D)) + np.linspace(1.0, 0.1, D)
        out.append(ClientStats(client_id=k, role="honest", mu=mu, sigma2=float(rng.uniform(0.01, 0.1)),
                               n=100 + k, mu_signed=mu.copy(), extra={"x": 1}))
    return out


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    honest = _honest(rng)
    me = ClientStats(client_id=99, role="honest", mu=np.abs(rng.normal(size=D)) + 0.1, sigma2=0.05, n=321)
    return honest, me


def _weight(c, floor=None, eps=1e-12):
    s2 = c.sigma2 if floor is None else max(c.sigma2, floor)
    return float(np.sum(np.asarray(c.mu) ** 2)) / (s2 + eps)


@pytest.mark.parametrize("name", REPORT_ATTACKS)
def test_report_attack_valid_and_not_mutating(setup, name):
    honest, me = setup
    me_copy = copy.deepcopy(me)
    honest_copy = copy.deepcopy(honest)
    out = apply_report_attack(name, me, honest, np.random.default_rng(1), feature_names=NAMES, target_feature="f2")
    assert out is not me
    assert out.mu.shape == (D,)
    assert np.all(out.mu >= 0) and np.all(np.isfinite(out.mu))
    assert out.sigma2 >= 0
    assert out.role == "attacker"
    assert out.n == me.n and out.client_id == me.client_id
    assert out.extra["attack"] == name
    # originals untouched
    assert me.role == me_copy.role and np.array_equal(me.mu, me_copy.mu) and me.sigma2 == me_copy.sigma2
    assert me.extra == me_copy.extra
    for h, hc in zip(honest, honest_copy):
        assert np.array_equal(h.mu, hc.mu) and h.extra == hc.extra and h.role == "honest"
    assert name in THREAT_MODELS and "Threat model" in THREAT_MODELS[name]


def test_unknown_attack_raises(setup):
    honest, me = setup
    with pytest.raises(ValueError):
        apply_report_attack("nope", me, honest, np.random.default_rng(0), feature_names=NAMES)
    with pytest.raises(ValueError):
        apply_report_attack("feature_hiding", me, honest, np.random.default_rng(0), feature_names=NAMES,
                            target_feature="missing")
    with pytest.raises(TypeError):
        apply_report_attack("scaling", me, honest, np.random.default_rng(0), feature_names=NAMES, bogus=1)


def test_frozen(setup):
    honest, me = setup
    out = apply_report_attack("frozen", me, honest, np.random.default_rng(0), feature_names=NAMES)
    assert out.sigma2 == 0.0
    assert np.allclose(out.mu, out.mu[0])
    honest_mass = np.mean([h.mu.sum() for h in honest])
    assert np.isclose(out.mu.sum(), honest_mass)


def test_zero_and_scaling(setup):
    honest, me = setup
    z = apply_report_attack("zero", me, honest, np.random.default_rng(0), feature_names=NAMES)
    assert np.all(z.mu <= 1e-7) and z.sigma2 == 0.0
    s = apply_report_attack("scaling", me, honest, np.random.default_rng(0), feature_names=NAMES)
    assert np.allclose(s.mu, 10 * me.mu) and s.sigma2 == me.sigma2
    s3 = apply_report_attack("scaling", me, honest, np.random.default_rng(0), feature_names=NAMES, lam=3.0)
    assert np.allclose(s3.mu, 3 * me.mu)


def test_feature_hiding(setup):
    honest, me = setup
    out = apply_report_attack("feature_hiding", me, honest, np.random.default_rng(0), feature_names=NAMES,
                              target_feature="f3")
    assert out.mu[3] == 0.0
    assert np.isclose(out.mu.sum(), me.mu.sum())
    # proportional redistribution: ratios among other features preserved
    others = [i for i in range(D) if i != 3]
    assert np.allclose(out.mu[others] / out.mu[others].sum(), me.mu[others] / me.mu[others].sum())
    # fallback: most important feature of honest mean
    fb = apply_report_attack("feature_hiding", me, honest, np.random.default_rng(0), feature_names=NAMES)
    j = int(np.argmax(np.mean([h.mu for h in honest], axis=0)))
    assert fb.mu[j] == 0.0 and fb.extra["target_idx"] == j


def test_feature_promotion(setup):
    honest, me = setup
    out = apply_report_attack("feature_promotion", me, honest, np.random.default_rng(0), feature_names=NAMES,
                              target_feature="f5")
    assert np.isclose(out.mu[5], 2 * me.mu.max())
    assert int(np.argmax(out.mu)) == 5
    fb = apply_report_attack("feature_promotion", me, honest, np.random.default_rng(0), feature_names=NAMES)
    j = int(np.argmin(np.mean([h.mu for h in honest], axis=0)))
    assert int(np.argmax(fb.mu)) == j


def test_mimic(setup):
    honest, me = setup
    m = np.mean([h.mu for h in honest], axis=0)
    out0 = apply_report_attack("mimic", me, honest, np.random.default_rng(0), feature_names=NAMES,
                               target_feature="f0", strength=0.0)
    assert np.allclose(out0.mu, m)
    out1 = apply_report_attack("mimic", me, honest, np.random.default_rng(0), feature_names=NAMES,
                               target_feature="f0", strength=1.0)
    assert out1.mu[0] == 0.0 and np.isclose(out1.mu.sum(), m.sum())
    out = apply_report_attack("mimic", me, honest, np.random.default_rng(0), feature_names=NAMES, target_feature="f0")
    assert 0 < out.mu[0] < m[0]


def test_collusion_identical(setup):
    honest, me = setup
    me2 = ClientStats(client_id=100, role="honest", mu=me.mu[::-1].copy(), sigma2=0.2, n=10)
    a = apply_report_attack("collusion", me, honest, np.random.default_rng(0), feature_names=NAMES, target_feature="f1")
    b = apply_report_attack("collusion", me2, honest, np.random.default_rng(5), feature_names=NAMES, target_feature="f1")
    assert np.array_equal(a.mu, b.mu) and a.mu[1] == 0.0
    vs = make_collusion_reports(4, honest, np.random.default_rng(0), feature_names=NAMES, target_feature="f1")
    assert len(vs) == 4 and all(np.array_equal(v, a.mu) for v in vs)
    vs[0][0] = -1.0  # copies are independent
    assert vs[1][0] >= 0


def test_adaptive_boost(setup):
    honest, me = setup
    out = apply_report_attack("adaptive_psnr", me, honest, np.random.default_rng(0), feature_names=NAMES,
                              target_feature="f0")
    max_e = max(float(np.sum(h.mu ** 2)) for h in honest)
    assert np.isclose(np.sum(out.mu ** 2), 10 * max_e)
    assert out.sigma2 == 0.0 and out.mu[0] == 0.0
    out2 = apply_report_attack("adaptive_psnr", me, honest, np.random.default_rng(0), feature_names=NAMES,
                               target_feature="f0", boost=3.0, sigma2_floor=0.02)
    assert np.isclose(np.sum(out2.mu ** 2), 3 * max_e) and out2.sigma2 == 0.02


@pytest.mark.parametrize("floor", [None, 0.04])
@pytest.mark.parametrize("cap", [1.0, 3.0])
def test_adaptive_cap_boundary(setup, floor, cap):
    honest, me = setup
    kw = {} if floor is None else {"sigma2_floor": floor}
    out = apply_report_attack("adaptive_psnr", me, honest, np.random.default_rng(0), feature_names=NAMES,
                              target_feature="f0", cap=cap, **kw)
    w = np.array([_weight(h, floor) for h in honest] + [_weight(out, floor)])
    med = np.median(w)
    assert np.isclose(w[-1], cap * med, rtol=1e-9)


def test_adaptive_cap_multi_attackers(setup):
    honest, me = setup
    out = apply_report_attack("adaptive_psnr", me, honest, np.random.default_rng(0), feature_names=NAMES,
                              target_feature="f0", cap=2.0, n_attackers=3)
    w = np.array([_weight(h) for h in honest] + [_weight(out)] * 3)
    assert np.isclose(w[-1], 2.0 * np.median(w), rtol=1e-9)


# ---------------------------------------------------------------- data attacks
def _data(rng, n=200):
    X = pd.DataFrame({"a": rng.normal(5, 2, n), "b": rng.normal(0, 10, n), "c": np.ones(n)})
    X.loc[X.index[:5], "a"] = np.nan
    y = rng.integers(0, 2, n)
    return X, y


@pytest.mark.parametrize("name", DATA_ATTACKS)
def test_data_attack_deterministic_shape(name):
    X, y = _data(np.random.default_rng(0))
    X0, y0 = X.copy(), y.copy()
    a = apply_data_attack(name, X, y, np.random.default_rng(42))
    b = apply_data_attack(name, X, y, np.random.default_rng(42))
    c = apply_data_attack(name, X, y, np.random.default_rng(43))
    assert a[0].shape == X.shape and a[1].shape == y.shape
    assert list(a[0].columns) == list(X.columns)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert not (a[0].equals(c[0]) and np.array_equal(a[1], c[1]))
    pd.testing.assert_frame_equal(X, X0)
    assert np.array_equal(y, y0)


def test_input_noise_scale():
    X, y = _data(np.random.default_rng(0), n=5000)
    Xn, yn = apply_data_attack("input_noise", X, y, np.random.default_rng(1), lam=0.5)
    diff = (Xn - X)
    assert Xn["a"].isna().sum() == 5  # NaNs preserved
    assert np.isclose(diff["b"].std(), 0.5 * X["b"].std(ddof=0), rtol=0.05)
    assert np.isclose(diff["a"].std(), 0.5 * X["a"].std(ddof=0), rtol=0.05)
    assert np.allclose(diff["c"], 0)  # constant column: zero std, no noise
    assert np.array_equal(yn, y)


def test_label_flip_rate():
    X, y = _data(np.random.default_rng(0), n=20000)
    _, yf = apply_data_attack("label_flip", X, y, np.random.default_rng(1), p=0.3)
    assert abs(np.mean(yf != y) - 0.3) < 0.02
    _, y1 = apply_data_attack("label_flip", X, y, np.random.default_rng(1), p=1.0)
    assert np.array_equal(y1, 1 - y)
    with pytest.raises(ValueError):
        apply_data_attack("label_flip", X, y, np.random.default_rng(1), p=1.5)
    with pytest.raises(ValueError):
        apply_data_attack("nope", X, y, np.random.default_rng(1))


@pytest.mark.parametrize("K", [2, 5, 6, 9])
@pytest.mark.parametrize("cap", [1.0, 1.5, 3.0])
@pytest.mark.parametrize("n_att", [1, 2])
def test_adaptive_cap_boundary_various_sizes(K, cap, n_att):
    rng = np.random.default_rng(K)
    honest = _honest(rng, K=K)
    me = ClientStats(client_id=99, role="honest", mu=np.abs(rng.normal(size=D)) + 0.1, sigma2=0.05, n=50)
    if n_att >= K and cap >= 2.0:
        # attackers are half the clients: median slope 1/2, cap*1/2 >= 1 -> no finite boundary
        with pytest.raises(ValueError):
            apply_report_attack("adaptive_psnr", me, honest, rng, feature_names=NAMES, cap=cap, n_attackers=n_att)
        return
    out = apply_report_attack("adaptive_psnr", me, honest, rng, feature_names=NAMES, cap=cap, n_attackers=n_att)
    w = np.array([_weight(h) for h in honest] + [_weight(out)] * n_att)
    assert np.isclose(w[-1], cap * np.median(w), rtol=1e-9)


def test_collusion_jitter_is_noisy_but_centered():
    from powersnr.attacks import apply_report_attack
    rng = np.random.default_rng(0)
    d = 8
    honest = [ClientStats(i, "honest", np.abs(np.linspace(1, 0.2, d) + rng.normal(0, 0.02, d)), 1e-3, 100)
              for i in range(6)]
    names = [f"f{i}" for i in range(d)]
    a = apply_report_attack("collusion_jitter", honest[0], honest[1:], np.random.default_rng(1), feature_names=names)
    b = apply_report_attack("collusion_jitter", honest[0], honest[1:], np.random.default_rng(2), feature_names=names)
    c = apply_report_attack("collusion", honest[0], honest[1:], np.random.default_rng(3), feature_names=names)
    assert not np.allclose(a.mu, b.mu) and np.all(a.mu >= 0)
    assert np.linalg.norm(a.mu - c.mu) < 0.5 * np.linalg.norm(c.mu)
    assert a.role == "attacker"
