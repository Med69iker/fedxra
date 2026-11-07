"""Numerical checks of the statements in Sections 4-5 (run before every submission).

P1  precision weighting (IVW, energy-over-variance): an attacker reporting sigma^2 = 0 gets weight -> 1, and
    1 - w_a <= eps * sum_h g_h / sigma_h^2 / g_a.
P2  centrality amplification: at the fixed point z of a weighted geometric median with prior weights pi,
    a coalition within distance rho of z, with every honest report at distance >= r_min, obtains an
    effective weight W_A >= pi_A / (pi_A + (1 - pi_A) rho / r_min).
P3  FedXRA weights: m survivors hold at most m * c / ceil(s / 2) of the combination weight, whatever
    sample sizes and variances they report (c = weight cap, s = number of survivors).
L4  coordinate-wise Huber stage (bounded influence): with attacker weight W_A < 1/2,
    (i)  the weighted median l0 lies in [Q_H(a1), Q_H(a2)], a1 = (1/2 - W_A)/(1 - W_A), a2 = 1/(2(1 - W_A));
    (ii) if moreover W_A <= 1 - q, the scale s <= max(1.4826 * Q_H^{|x - l0|}(q / (1 - W_A)), floor * |l0|);
    (iii) |l_T - l0| <= T * c * s.
    So the output lies in an interval that depends on the honest values only, whatever the attackers report.
    (L4 now covers the Huber-stage ablation variant only.)
L5  FedXRA's coordinate-wise weighted trimmed mean (Lemma lem:trim): with attacker weight W_A < beta, each output
    coordinate lies in [Q_H((beta - W_A)/(1 - W_A)), Q_H((1 - beta)/(1 - W_A))], honest quantiles only.
P3 is checked for sample-size weights and for the fitted-variance weights (attackers also choose their position).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "powersnr"))
from powersnr.aggregators import _cw_huber, _cw_trim, _weighted_geomedian, _wquantile, aggregate  # noqa: E402

rng = np.random.default_rng(2026)
TRIALS = 10_000


def p1() -> None:
    worst = 0.0
    for _ in range(TRIALS):
        K, d = rng.integers(3, 30), rng.integers(3, 60)
        honest = np.abs(rng.normal(1, 0.3, (K - 1, d)))
        s2 = rng.uniform(1e-4, 1e-1, K - 1)
        att = np.abs(rng.normal(1, 0.3, d))
        for eps in (1e-6, 1e-9, 1e-12):
            mus = np.vstack([att, honest])
            sig = np.r_[0.0, s2]
            w = aggregate("ivw", mus, sig, np.ones(K), eps=eps).weights
            assert 1 - w[0] <= eps * np.sum(1.0 / s2) + 1e-12
            wb = aggregate("psnr_base", mus, sig, np.ones(K), eps=eps).weights
            g = np.sum(mus ** 2, axis=1)
            assert 1 - wb[0] <= eps * np.sum(g[1:] / s2) / g[0] + 1e-12
        worst = max(worst, 1 - w[0])
    print(f"P1 ok: max(1 - w_attacker) at eps=1e-12 over {TRIALS} trials = {worst:.2e}")


def p2() -> None:
    worst, amp = 0.0, []
    for _ in range(TRIALS):
        K, d = int(rng.integers(5, 30)), int(rng.integers(2, 30))
        m = int(rng.integers(1, K // 2))
        H = rng.normal(0, 1, (K - m, d))
        y = H.mean(axis=0) + rng.normal(0, 10 ** rng.uniform(-4, 0), d)  # the coalition copies the honest mean
        mus = np.vstack([np.tile(y, (m, 1)), H])
        prior = rng.dirichlet(np.ones(K))
        z, w = _weighted_geomedian(mus, prior, tol=1e-13, max_iter=20000)
        r = np.linalg.norm(mus - z, axis=1)
        rho, rmin, pa = r[:m].max(), r[m:].min(), prior[:m].sum()
        if rmin <= 1e-9:
            continue
        lower = pa / (pa + (1 - pa) * rho / rmin)
        WA = w[:m].sum()
        assert WA >= lower * (1 - 1e-6) - 1e-9, (WA, lower)
        worst = max(worst, lower / WA)
        amp.append(WA / pa)
    print(f"P2 ok: max(bound / W_A) = {worst:.3f} (<= 1); median amplification W_A / pi_A = {np.median(amp):.2f}")


def p3() -> None:
    c, worst = 3.0, 0.0
    for _ in range(TRIALS):
        K, d = int(rng.integers(5, 40)), int(rng.integers(4, 40))
        base = np.abs(rng.normal(1, 0.5, d))
        mus = np.abs(base + rng.normal(0, 0.05, (K, d))) * rng.uniform(0.9, 1.1, (K, 1))
        s2 = rng.uniform(1e-4, 1e-2, K)
        n = rng.integers(50, 3000, K).astype(float)
        m = int(rng.integers(1, max(2, K // 2)))
        s2[:m], n[:m] = 0.0, 1e9  # attackers claim perfect stability and a huge sample
        if rng.random() < 0.5:  # attackers also choose their position, to steer the fitted variance model
            mus[:m] = base * rng.uniform(0.5, 1.5, (m, 1)) + rng.normal(0, rng.choice([0.0, 0.5]), (m, d)) ** 2
        for weight in ("n", "fit_ts"):
            res = aggregate("psnr_v3", mus, s2, n, cap=c, weight=weight, dup_ratio=1.0 if weight == "fit_ts" else 4.0)
            kept = res.info.get("kept", list(range(K)))
            s, m_kept = len(kept), sum(1 for k in kept if k < m)
            if s == 0 or m_kept == 0:
                continue
            bound = m_kept * c / math.ceil(s / 2)
            assert res.weights[:m].sum() <= bound + 1e-9, (weight, res.weights[:m].sum(), bound)
            worst = max(worst, res.weights[:m].sum() / bound)
    print(f"P3 ok (n and fitted weights): max(coalition share / bound) = {worst:.3f} (<= 1)")


def l4() -> None:
    T, worst = 3, 0.0
    for _ in range(TRIALS):
        K = int(rng.integers(4, 30))
        c, q = rng.choice([0.5, 1.0, 1.5]), rng.choice([0.5, 0.7, 0.8])
        w = rng.dirichlet(np.ones(K))
        nA = int(rng.integers(0, K // 2 + 1))
        A = rng.choice(K, nA, replace=False)
        WA = w[A].sum()
        if WA >= min(0.5, 1 - q):
            continue
        H = np.setdiff1d(np.arange(K), A)
        x = np.abs(rng.normal(0.1, 0.05, K))
        x[A] = rng.choice([0.0, 10 ** rng.uniform(-3, 3)]) * rng.uniform(0, 1, nA)  # anything non-negative
        wh = w[H] / w[H].sum()
        l0 = _wquantile(x, w, 0.5)
        a1, a2 = (0.5 - WA) / (1 - WA), 0.5 / (1 - WA)
        assert _wquantile(x[H], wh, a1) - 1e-12 <= l0 <= _wquantile(x[H], wh, min(a2, 1.0)) + 1e-12
        s = max(1.4826 * _wquantile(np.abs(x - l0), w, q), 1e-3 * abs(l0), 1e-300)
        s_bound = max(1.4826 * _wquantile(np.abs(x[H] - l0), wh, min(q / (1 - WA), 1.0)), 1e-3 * abs(l0), 1e-300)
        assert s <= s_bound * (1 + 1e-9)
        out = _cw_huber(x[:, None], w, c, q, iters=T)[0]
        assert abs(out - l0) <= T * c * s * (1 + 1e-9) + 1e-12
        lo = _wquantile(x[H], wh, a1) - T * c * s_bound
        hi = _wquantile(x[H], wh, min(a2, 1.0)) + T * c * s_bound
        assert lo - 1e-12 <= out <= hi + 1e-12
        worst = max(worst, abs(out - l0) / (T * c * s))
    print(f"L4 ok: max(|l_T - l0| / (T c s)) = {worst:.3f} (<= 1); honest-only interval holds")


def l5() -> None:
    """Lemma 4 (FedXRA): with attacker weight W_A < beta, each coordinate of the weighted trimmed mean lies in
    [Q^H_{(beta - W_A)/(1 - W_A)}, Q^H_{(1 - beta)/(1 - W_A)}], quantiles of the honest values alone."""
    worst_in = 0.0
    for _ in range(TRIALS):
        K = int(rng.integers(4, 40))
        beta = float(rng.choice([0.1, 0.2, 0.3, 0.4]))
        w = rng.dirichlet(np.ones(K) * rng.choice([0.3, 1.0, 5.0]))
        nA = int(rng.integers(0, K))
        A = rng.choice(K, nA, replace=False)
        WA = w[A].sum()
        if WA >= beta or nA == K:
            continue
        H = np.setdiff1d(np.arange(K), A)
        x = np.abs(rng.normal(0.1, 0.05, K))
        x[A] = rng.choice([0.0, 10 ** rng.uniform(-3, 3)], nA) * rng.uniform(0, 1, nA)  # anything non-negative
        wh = w[H] / w[H].sum()
        out = _cw_trim(x[:, None], w, beta)[0]
        lo = _wquantile(x[H], wh, (beta - WA) / (1 - WA))
        hi = _wquantile(x[H], wh, (1 - beta) / (1 - WA))
        assert lo - 1e-12 <= out <= hi + 1e-12, (beta, WA, lo, out, hi)
        worst_in = max(worst_in, WA / beta)
    print(f"L5 ok: trimmed-mean output inside the honest quantile interval (attacker weight up to {worst_in:.3f} x beta)")


if __name__ == "__main__":
    p1()
    p2()
    p3()
    l4()
    l5()
