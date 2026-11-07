"""Server-side aggregation rules for federated SHAP importance vectors.

Every aggregator has the signature ``fn(mus, sigma2, n, **kw) -> AggResult`` where

* ``mus``    : (K, d) client importance vectors (mean |SHAP|, non-negative by contract),
* ``sigma2`` : (K,)  split-half instability  mean_r ||mu_A - mu_B||_2^2  (>= 0),
* ``n``      : (K,)  local sample sizes.

``AggResult.weights`` is a (K,) vector normalised to sum 1 for weight-based rules and an
all-NaN vector for rules that are not a convex combination of client vectors
(median, trimmed_mean). For ``geomedian`` the final Weiszfeld weights are reported (the
geometric median IS the convex combination of the clients with these weights at the
fixed point), so they are meaningful for weight-share diagnostics.

Notation: ``signal_k = ||mu_k||_2^2`` (Power-SNR "energy").

Numerical policy: weights of the form a / (b + eps) are computed in log space and
normalised with a max-shift, so ``sigma2 = 0`` with a tiny ``eps`` gives a huge but finite
relative weight (the Zero-Variance Fallacy is reproduced, never hidden by inf/NaN). If
``eps = 0`` and ``sigma2_k = 0`` the limit is taken: clients with infinite raw weight share
the mass equally. If every weight is zero the rule falls back to the coordinate-wise median
and sets ``info["fallback"] = True``.

Use :func:`aggregate` to dispatch by name; it forwards only the keyword arguments that the
selected rule accepts, so the runner may pass one common kwargs dict to all rules.
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
from scipy import stats

from powersnr.types import AggResult

DEFAULT_EPS = 1e-12

__all__ = ["AGGREGATORS", "aggregate", "DEFAULT_EPS"]


# --------------------------------------------------------------------------- helpers
def _validate(mus, sigma2, n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus = np.asarray(mus, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float).reshape(-1)
    n = np.asarray(n, dtype=float).reshape(-1)
    if mus.ndim != 2:
        raise ValueError(f"mus must be 2-D (K, d); got shape {mus.shape}")
    K, d = mus.shape
    if K < 1 or d < 1:
        raise ValueError(f"mus must have K>=1 and d>=1; got shape {mus.shape}")
    if sigma2.shape != (K,):
        raise ValueError(f"sigma2 must have shape ({K},); got {sigma2.shape}")
    if n.shape != (K,):
        raise ValueError(f"n must have shape ({K},); got {n.shape}")
    if not np.all(np.isfinite(mus)):
        raise ValueError("mus contains non-finite values")
    if not np.all(np.isfinite(sigma2)):
        raise ValueError("sigma2 contains non-finite values")
    if not np.all(np.isfinite(n)):
        raise ValueError("n contains non-finite values")
    if np.any(sigma2 < 0):
        raise ValueError("sigma2 must be >= 0")
    if np.any(n < 0):
        raise ValueError("n must be >= 0")
    return mus, sigma2, n


def _check_eps(eps: float) -> float:
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError(f"eps must be finite and >= 0; got {eps}")
    return eps


def _signal(mus: np.ndarray) -> np.ndarray:
    return np.sum(mus * mus, axis=1)


def _log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """log(num/den) elementwise with num, den >= 0; num == 0 -> -inf (zero weight),
    num > 0 and den == 0 -> +inf."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, -np.inf)
    pos = num > 0
    with np.errstate(divide="ignore"):
        out[pos] = np.log(num[pos]) - np.log(den[pos])
    return out


def _normalize_log(logw: np.ndarray) -> np.ndarray | None:
    """Normalised weights from log-weights; None if all weights are zero."""
    logw = np.asarray(logw, dtype=float)
    if np.any(np.isnan(logw)):
        raise ValueError("NaN log-weight")
    if np.all(np.isneginf(logw)):
        return None
    if np.any(np.isposinf(logw)):
        w = np.isposinf(logw).astype(float)
    else:
        w = np.exp(logw - np.max(logw))
    return w / w.sum()


def _normalize(w: np.ndarray) -> np.ndarray | None:
    w = np.asarray(w, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    s = w.sum()
    if s <= 0:
        return None
    return w / s


def _median_result(mus: np.ndarray, info: dict) -> AggResult:
    K = mus.shape[0]
    info = dict(info)
    info.setdefault("fallback", False)
    return AggResult(phi=np.median(mus, axis=0), weights=np.full(K, np.nan), info=info)


def _weighted(mus: np.ndarray, w: np.ndarray | None, info: dict) -> AggResult:
    """Weighted mean; median fallback if w is None (all-zero weights)."""
    info = dict(info)
    if w is None:
        info["fallback"] = True
        info["fallback_rule"] = "median"
        return _median_result(mus, info)
    info.setdefault("fallback", False)
    return AggResult(phi=w @ mus, weights=w, info=info)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


# --------------------------------------------------------------------------- baselines
def mean(mus, sigma2, n) -> AggResult:
    """phi = (1/K) sum_k mu_k."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K = mus.shape[0]
    return _weighted(mus, np.full(K, 1.0 / K), {"name": "mean"})


def fedavg(mus, sigma2, n) -> AggResult:
    """phi = sum_k (n_k / sum n) mu_k  (median fallback if sum n == 0)."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    return _weighted(mus, _normalize(n), {"name": "fedavg"})


def median(mus, sigma2, n) -> AggResult:
    """Coordinate-wise median. weights = NaN."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    return _median_result(mus, {"name": "median"})


def trimmed_mean(mus, sigma2, n, beta: float = 0.2) -> AggResult:
    """Coordinate-wise trimmed mean, trimming floor(beta*K) values from each tail
    (scipy.stats.trim_mean). weights = NaN."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    if not 0 <= beta < 0.5:
        raise ValueError(f"beta must be in [0, 0.5); got {beta}")
    K = mus.shape[0]
    phi = stats.trim_mean(mus, beta, axis=0)
    return AggResult(phi=np.asarray(phi, dtype=float), weights=np.full(K, np.nan),
                     info={"name": "trimmed_mean", "beta": beta,
                           "n_trimmed_each_side": int(np.floor(beta * K)), "fallback": False})


def geomedian(mus, sigma2, n, tol: float = 1e-10, max_iter: int = 1000,
              nu: float = 1e-12) -> AggResult:
    """Geometric median argmin_z sum_k ||mu_k - z||_2 via Weiszfeld iterations
    z <- sum_k (mu_k/max(||mu_k-z||,nu)) / sum_k 1/max(||mu_k-z||,nu), started at the mean,
    stopped when ||z_new - z|| <= tol * max(1, ||z||). Reported weights are the final
    normalised Weiszfeld weights (1/dist_k), i.e. phi = weights @ mus."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    z = mus.mean(axis=0)
    converged = False
    it = 0
    w = np.full(mus.shape[0], 1.0 / mus.shape[0])
    for it in range(1, max_iter + 1):
        dist = np.maximum(np.linalg.norm(mus - z, axis=1), nu)
        w = (1.0 / dist) / np.sum(1.0 / dist)
        z_new = w @ mus
        step = np.linalg.norm(z_new - z)
        z = z_new
        if step <= tol * max(1.0, np.linalg.norm(z)):
            converged = True
            break
    return AggResult(phi=z, weights=w, info={"name": "geomedian", "iterations": it,
                                             "converged": converged, "fallback": False,
                                             "weights_are_weiszfeld": True})


def _krum_scores(mus: np.ndarray, f: int) -> tuple[np.ndarray, int, bool]:
    K = mus.shape[0]
    n_nb = K - f - 2
    degenerate = n_nb < 1
    n_nb = max(1, min(n_nb, K - 1)) if K > 1 else 0
    if K == 1:
        return np.zeros(1), 0, True
    D = np.sum((mus[:, None, :] - mus[None, :, :]) ** 2, axis=2)
    scores = np.empty(K)
    for i in range(K):
        others = np.delete(D[i], i)
        scores[i] = np.sum(np.sort(others)[:n_nb])
    return scores, n_nb, degenerate


def _default_f(K: int, f: int | None) -> int:
    if f is None:
        f = (K - 3) // 2
    f = int(max(0, f))
    return f


def krum(mus, sigma2, n, f: int | None = None) -> AggResult:
    """Krum (Blanchard et al., 2017): score_i = sum of squared L2 distances from mu_i to its
    K-f-2 nearest other clients; phi = mu_{argmin score}. Default f = max(0, floor((K-3)/2)).
    If K < f+3 the neighbour count is clipped to >= 1 and info["krum_degenerate"] = True.
    weights = one-hot on the selected client."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K = mus.shape[0]
    f = _default_f(K, f)
    scores, n_nb, degen = _krum_scores(mus, f)
    sel = int(np.argmin(scores))  # ties -> lowest index (deterministic)
    w = np.zeros(K)
    w[sel] = 1.0
    return AggResult(phi=mus[sel].copy(), weights=w,
                     info={"name": "krum", "f": f, "n_neighbors": n_nb, "scores": scores,
                           "selected": [sel], "krum_degenerate": degen, "fallback": False})


def multikrum(mus, sigma2, n, f: int | None = None, m: int | None = None) -> AggResult:
    """Multi-Krum (Blanchard et al., 2017): average of the m clients with the lowest Krum
    scores. Default m = K - f. weights = 1/m on the selected clients."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K = mus.shape[0]
    f = _default_f(K, f)
    if m is None:
        m = K - f
    m = int(min(max(1, m), K))
    scores, n_nb, degen = _krum_scores(mus, f)
    sel = np.argsort(scores, kind="stable")[:m]
    w = np.zeros(K)
    w[sel] = 1.0 / m
    return AggResult(phi=w @ mus, weights=w,
                     info={"name": "multikrum", "f": f, "m": m, "n_neighbors": n_nb,
                           "scores": scores, "selected": sorted(int(i) for i in sel),
                           "krum_degenerate": degen, "fallback": False})


def ivw(mus, sigma2, n, eps: float = DEFAULT_EPS) -> AggResult:
    """Inverse-variance weighting: w_k ∝ 1 / (sigma2_k + eps)."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    w = _normalize_log(_log_ratio(np.ones_like(sigma2), sigma2 + eps))
    return _weighted(mus, w, {"name": "ivw", "eps": eps})


def dersimonian_laird(mus, sigma2, n, eps: float = DEFAULT_EPS) -> AggResult:
    """DerSimonian–Laird random-effects weighting with one scalar weight per client.

    Within-client variance per coordinate: v_k = sigma2_k / d (sigma2 is a squared L2 norm
    over d coordinates). For each coordinate j, with fixed-effect weights a_k = 1/(v_k+eps):
        mu_FE_j = sum_k a_k mu_kj / sum_k a_k,
        Q_j     = sum_k a_k (mu_kj - mu_FE_j)^2,
        C       = sum_k a_k - sum_k a_k^2 / sum_k a_k,
        tau2_j  = max(0, (Q_j - (K-1)) / C)   (0 if C == 0).
    tau2 = mean_j tau2_j; final weights w_k ∝ 1 / (v_k + tau2 + eps).
    """
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    K, d = mus.shape
    v = sigma2 / d
    a = _normalize_log(_log_ratio(np.ones(K), v + eps))  # normalised FE weights (scale-free)
    tau2 = 0.0
    tau2_j = np.zeros(d)
    if K > 1 and a is not None:
        # Work with un-normalised a in a numerically safe way: a = s * a_norm.
        # Q and C both scale linearly with s, so compute s explicitly but bounded.
        log_a = _log_ratio(np.ones(K), v + eps)
        if np.all(np.isfinite(log_a)):
            s = np.exp(np.max(log_a))  # finite unless eps == 0 and v == 0
            a_raw = a * s
            fe = a @ mus
            Q = a_raw @ ((mus - fe) ** 2)
            C = a_raw.sum() - np.sum(a_raw ** 2) / a_raw.sum()
            if C > 0 and np.isfinite(C):
                tau2_j = np.maximum(0.0, (Q - (K - 1)) / C)
                tau2 = float(np.mean(tau2_j))
    w = _normalize_log(_log_ratio(np.ones(K), v + tau2 + eps))
    return _weighted(mus, w, {"name": "dersimonian_laird", "eps": eps, "tau2": tau2,
                              "within_var": v})


def fltrust_like(mus, sigma2, n, root_mu=None) -> AggResult:
    """FLTrust-style aggregation (Cao et al., 2021) on importance vectors.
    trust_k = ReLU(cos(mu_k, root_mu)); each mu_k is rescaled to ||root_mu||;
    phi = sum_k trust_k * mu_k * ||root|| / ||mu_k|| / sum_k trust_k.
    Requires root_mu (server-side reference importance, shape (d,))."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    if root_mu is None:
        raise ValueError("fltrust_like requires root_mu")
    root = np.asarray(root_mu, dtype=float).reshape(-1)
    if root.shape != (mus.shape[1],) or not np.all(np.isfinite(root)):
        raise ValueError(f"root_mu must be finite with shape ({mus.shape[1]},)")
    root_norm = np.linalg.norm(root)
    norms = np.linalg.norm(mus, axis=1)
    cos = np.array([_cosine(m, root) for m in mus])
    trust = np.maximum(cos, 0.0)
    scaled = np.zeros_like(mus)
    nz = norms > 0
    scaled[nz] = mus[nz] * (root_norm / norms[nz])[:, None]
    w = _normalize(trust)
    info = {"name": "fltrust_like", "cos": cos}
    if w is None:
        info["fallback"] = True
        info["fallback_rule"] = "median"
        return _median_result(mus, info)
    return AggResult(phi=w @ scaled, weights=w, info={**info, "fallback": False})


# --------------------------------------------------------------------------- Power-SNR family
def psnr_energy_only(mus, sigma2, n) -> AggResult:
    """Ablation without the noise term: w_k ∝ signal_k = ||mu_k||^2."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    return _weighted(mus, _normalize(_signal(mus)), {"name": "psnr_energy_only"})


def psnr_base(mus, sigma2, n, eps: float = DEFAULT_EPS) -> AggResult:
    """Power-SNR: w_k ∝ ||mu_k||^2 / (sigma2_k + eps)."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    w = _normalize_log(_log_ratio(_signal(mus), sigma2 + eps))
    return _weighted(mus, w, {"name": "psnr_base", "eps": eps})


def _gate_mask(mus: np.ndarray, gate: float) -> tuple[np.ndarray, np.ndarray, float]:
    sig = np.mean(np.abs(mus), axis=1)        # mean|mu_k| (== mean mu_k for mu >= 0)
    thresh = gate * float(np.mean(sig))
    return sig < thresh, sig, thresh


def psnr_gate(mus, sigma2, n, eps: float = DEFAULT_EPS, gate: float = 0.1) -> AggResult:
    """Power-SNR with the legacy energy gate: w_k = 0 if mean|mu_k| < gate * mean_j mean|mu_j|,
    else w_k ∝ ||mu_k||^2 / (sigma2_k + eps)."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    gated, _, thresh = _gate_mask(mus, gate)
    logw = _log_ratio(_signal(mus), sigma2 + eps)
    logw[gated] = -np.inf
    return _weighted(mus, _normalize_log(logw),
                     {"name": "psnr_gate", "eps": eps, "gate": gate, "gate_threshold": thresh,
                      "gated": [int(i) for i in np.flatnonzero(gated)]})


def psnr_legacy(mus, sigma2, n, eps: float = 1e-6, gate: float = 0.1) -> AggResult:
    """Exact re-implementation of the original Code.txt rule:
    sig_k = mean|mu_k|, inst_k = sigma2_k (taken as given), gate threshold = gate * mean_j sig_j,
    w_k = 0 if sig_k < threshold else sig_k^2 / (inst_k + eps), eps = 1e-6 (absolute!).
    If all weights are zero the legacy code returns the plain MEAN (not the median); this is
    reproduced and flagged with info["fallback"] = True, info["fallback_rule"] = "mean"."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    gated, sig, thresh = _gate_mask(mus, gate)
    logw = _log_ratio(sig ** 2, sigma2 + eps)
    logw[gated] = -np.inf
    w = _normalize_log(logw)
    info = {"name": "psnr_legacy", "eps": eps, "gate": gate, "gate_threshold": thresh,
            "gated": [int(i) for i in np.flatnonzero(gated)]}
    if w is None:
        K = mus.shape[0]
        info.update(fallback=True, fallback_rule="mean")
        return AggResult(phi=mus.mean(axis=0), weights=np.full(K, 1.0 / K), info=info)
    return _weighted(mus, w, info)


def psnr_robust(mus, sigma2, n, eps: float = DEFAULT_EPS, cap: float | None = 3.0,
                sigma2_floor=None, consistency: bool = True, use_floor: bool = True,
                floor_frac: float = 0.1, mad_k: float = 3.0, min_cos_dev: float = 1e-6,
                consensus: str = "weighted") -> AggResult:
    """Robust Power-SNR. Each step can be switched off for ablations.

    (1) floor   [use_floor]: sigma2_eff = max(sigma2, sigma2_floor); if sigma2_floor is None,
                 sigma2_floor = floor_frac * median_k(sigma2) (floor_frac = 0.1).
    (2) w_k ∝ ||mu_k||^2 / (sigma2_eff_k + eps)  (normalised, log-space).
    (3) cap     [cap not None]: w_k <- min(w_k, cap * median(w)) with the median over the
                 un-capped weights (if that median is 0, the median of the positive weights).
    (4) consistency [consistency]: consensus c = sum_k w_k mu_k / sum_k w_k with capped weights
                 (contract default, consensus="weighted"; consensus="median" uses the
                 coordinate-wise median of the clients instead -- an ablation option);
                 cos_k = cos(mu_k, c); zero client k if cos_k < 0 or
                 cos_k < median(cos) - max(mad_k * 1.4826 * MAD(cos), min_cos_dev)
                 (robust low-outlier rule; min_cos_dev avoids flagging rounding noise when
                 MAD = 0). Statistics over clients with positive capped weight.
    Renormalise; if all weights are zero -> coordinate-wise median, info["fallback"] = True.
    info lists "capped" and "filtered" client indices and the per-client cosines.
    """
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    K = mus.shape[0]
    info: dict = {"name": "psnr_robust", "eps": eps, "cap": cap, "consistency": consistency,
                  "use_floor": use_floor, "capped": [], "filtered": []}

    # (1) variance floor
    if use_floor:
        if sigma2_floor is None:
            floor = floor_frac * float(np.median(sigma2))
        else:
            floor = np.asarray(sigma2_floor, dtype=float)
            if floor.ndim not in (0, 1) or (floor.ndim == 1 and floor.shape != (K,)):
                raise ValueError(f"sigma2_floor must be scalar or shape ({K},)")
            if not np.all(np.isfinite(floor)) or np.any(floor < 0):
                raise ValueError("sigma2_floor must be finite and >= 0")
        sig2_eff = np.maximum(sigma2, floor)
        info["sigma2_floor"] = floor
    else:
        sig2_eff = sigma2
    info["floored"] = [int(i) for i in np.flatnonzero(sig2_eff > sigma2)]

    # (2) Power-SNR weights
    w = _normalize_log(_log_ratio(_signal(mus), sig2_eff + eps))
    if w is None:
        return _weighted(mus, None, info)
    info["weights_precap"] = w.copy()

    # (3) cap relative to the median weight
    if cap is not None:
        if cap <= 0:
            raise ValueError("cap must be > 0 or None")
        med = float(np.median(w))
        if med <= 0:
            med = float(np.median(w[w > 0]))
        limit = cap * med
        capped = w > limit
        info["capped"] = [int(i) for i in np.flatnonzero(capped)]
        info["cap_limit_rel"] = limit
        w = np.minimum(w, limit)
        w = w / w.sum()

    # (4) consistency filter
    if consistency:
        if consensus == "weighted":
            ref = w @ mus
        elif consensus == "median":
            ref = np.median(mus, axis=0)
        else:
            raise ValueError(f"consensus must be 'weighted' or 'median'; got {consensus!r}")
        info["consensus"] = consensus
        cos = np.array([_cosine(m, ref) for m in mus])
        info["cos"] = cos
        active = w > 0
        filt = cos < 0
        if active.sum() >= 3:
            c_act = cos[active]
            med_c = float(np.median(c_act))
            mad = 1.4826 * float(np.median(np.abs(c_act - med_c)))
            thr = med_c - max(mad_k * mad, min_cos_dev)
            info["cos_threshold"] = thr
            filt |= active & (cos < thr)
        info["filtered"] = [int(i) for i in np.flatnonzero(filt & active)]
        w = np.where(filt, 0.0, w)
        w = _normalize(w)
    return _weighted(mus, w, info)


def psnr_temp(mus, sigma2, n, eps: float = DEFAULT_EPS, tau: float = 1.0) -> AggResult:
    """Tempered Power-SNR: w_k ∝ (||mu_k||^2 / (sigma2_k + eps))^tau, tau in [0, 1].
    tau = 0 -> uniform mean (also for zero-energy clients); tau = 1 -> psnr_base."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    eps = _check_eps(eps)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1]; got {tau}")
    if tau == 0:
        logw = np.zeros(mus.shape[0])
    else:
        logw = tau * _log_ratio(_signal(mus), sigma2 + eps)
    return _weighted(mus, _normalize_log(logw), {"name": "psnr_temp", "eps": eps, "tau": tau})


def _robust_upper(x: np.ndarray, z: float) -> float:
    """median + z * 1.4826 * MAD (MAD floored at 5% of |median| so a tight honest
    cluster does not turn every small deviation into an outlier)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med + z * 1.4826 * max(mad, 0.05 * abs(med), 1e-12)


def _weighted_geomedian(mus: np.ndarray, prior: np.ndarray, tol: float = 1e-10, max_iter: int = 1000,
                        nu: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Weighted geometric median argmin_z sum_k prior_k ||mu_k - z||_2 (Weiszfeld).
    Returns (z, effective weights) with z = weights @ mus."""
    z = prior @ mus
    w = prior.copy()
    for _ in range(max_iter):
        dist = np.maximum(np.linalg.norm(mus - z, axis=1), nu)
        w = prior / dist
        w = w / w.sum()
        z_new = w @ mus
        step = np.linalg.norm(z_new - z)
        z = z_new
        if step <= tol * max(1.0, np.linalg.norm(z)):
            break
    return z, w


def psnr_screen(mus, sigma2, n, z: float = 3.0, mass_z: float = 3.0, cap: float | None = 1.5,
                floor_q: float | None = 0.25, snr: bool = True, rescale: bool = False,
                floor: str = "quantile", n_cap: int = 1000, floor_scale: float = 1.0,
                shrink: float | str | None = None, n_repeats: int = 5,
                soft: bool = False, soft_max: float = 2.0, stage2: str = "mean",
                dedup: bool = False, dup_ratio: float = 0.25) -> AggResult:
    """Screened, scale-free Power-SNR (two stages).

    Each report is split into its mass m_k = ||mu_k||_1 and its profile p_k = mu_k / m_k.
    Stage 1 (screening, uses only cross-client agreement, nothing self-certified):
      - drop dead signals (m_k = 0) and clients whose log-mass is a robust outlier
        (|log m_k - median| > mass_z * 1.4826 * MAD) -> zero / scaling attacks;
      - profile distance d_k = ||p_k - p_med||_1 to the coordinate-wise median profile, threshold
        t = median(d) + z * 1.4826 * MAD(d). Hard screening drops d_k > t; soft screening
        (soft=True) multiplies the weight by min(1, t / d_k) and drops only d_k > soft_max * t,
        so atypical but honest clients (minority banks) keep partial influence.
    Stage 2 (weighting among survivors, scale-free):
      - relative instability v_k = sigma2_k / m_k^2, floored so a client cannot gain by
        claiming v = 0: floor="quantile" uses the floor_q-quantile of the survivors' v;
        floor="n" uses the sample-size plausibility bound v_k >= floor_scale * c / min(n_k, n_cap),
        with c = median_k(v_k * min(n_k, n_cap)) (split-half variance scales as 1/n);
      - SNR_k = ||p_k||^2 / v_k (profile concentration over relative instability);
      - shrinkage w = (1 - lam) * uniform + lam * SNR / sum(SNR): shrink=None -> lam = 1,
        a float -> that lam, "auto" -> lam = clip(1 - var0 / var(log v), 0, 1) where
        var0 = 2 / (d * n_repeats) approximates the variance of log v expected from split-half
        sampling noise alone (chi-square approximation): equal-quality federations get lam ~ 0;
      - cap at cap * median(w) so no self-reported quantity can buy more than cap typical shares.
    Duplicate grouping (dedup=True, "one cluster, one vote"): two honest banks cannot produce
    explanations closer than their sampling noise allows, E||p_i - p_j||^2 >= (v_i + v_j) / 4 for
    independent samples. Clients linked by ||p_i - p_j||^2 < dup_ratio * (v_i + v_j) / 4 (v floored
    as above) are grouped (connected components) and each member's weight is divided by its group
    size, so a coalition sending (near-)identical vectors counts as one client (cf. FoolsGold).
    Aggregation: stage2="mean" -> phi = sum_k w_k mu_k; stage2="geomedian" -> weighted geometric
    median with prior weights w (reported weights are the effective Weiszfeld weights).
    rescale=True returns (sum_k w_k p_k) * median survivor mass instead (mean stage only).
    snr=False gives the screening-only ablation (equal weights among survivors)."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K, d = mus.shape
    if floor not in ("quantile", "n", "none"):
        raise ValueError(f"floor must be 'quantile', 'n' or 'none'; got {floor!r}")
    if stage2 not in ("mean", "geomedian"):
        raise ValueError(f"stage2 must be 'mean' or 'geomedian'; got {stage2!r}")
    mass = mus.sum(axis=1)
    keep = mass > 0
    info: dict = {"name": "psnr_screen", "z": z, "mass_z": mass_z, "cap": cap, "floor": floor,
                  "floor_q": floor_q, "snr": snr, "shrink": shrink, "soft": soft, "stage2": stage2, "dedup": dedup}
    if keep.sum() < 2:
        info["fallback"] = True
        return _median_result(mus, info)

    logm = np.log(mass[keep])
    med_m = float(np.median(logm))
    mad_m = max(float(np.median(np.abs(logm - med_m))), 1e-12)
    mass_out = np.zeros(K, bool)
    mass_out[keep] = np.abs(logm - med_m) > mass_z * 1.4826 * mad_m
    keep &= ~mass_out

    prof = np.zeros_like(mus)
    prof[mass > 0] = mus[mass > 0] / mass[mass > 0, None]
    p_med = np.median(prof[keep], axis=0)
    p_med = p_med / p_med.sum() if p_med.sum() > 0 else p_med
    dist = np.abs(prof - p_med).sum(axis=1)
    thr = _robust_upper(dist[keep], z)
    screen = np.zeros(K)
    if soft:
        screen[keep] = np.minimum(1.0, thr / np.maximum(dist[keep], 1e-12))
        screen[keep & (dist > soft_max * thr)] = 0.0
    else:
        screen[keep & (dist <= thr)] = 1.0
    prof_out = keep & (screen == 0)
    keep &= screen > 0
    info.update(mass_outliers=np.flatnonzero(mass_out).tolist(), profile_outliers=np.flatnonzero(prof_out).tolist(),
                dist=dist.tolist(), threshold=thr)
    if keep.sum() == 0:
        info["fallback"] = True
        return _median_result(mus, info)

    w = np.zeros(K)
    if snr:
        v = sigma2[keep] / mass[keep] ** 2
        if floor == "quantile" and floor_q is not None:
            v = np.maximum(v, np.quantile(v, floor_q))
        elif floor == "n":
            m_eff = np.minimum(n[keep], n_cap).astype(float)
            m_eff = np.maximum(m_eff, 1.0)
            c = float(np.median(v * m_eff))
            v = np.maximum(v, floor_scale * c / m_eff)
        v = np.maximum(v, 1e-300)
        s = np.sum(prof[keep] ** 2, axis=1) / v
        s = s / s.sum()
        if shrink is None:
            lam = 1.0
        elif shrink == "auto":
            lv = np.log(v)
            var_obs = float(np.var(lv)) if keep.sum() > 1 else 0.0
            var0 = 2.0 / (d * max(n_repeats, 1))
            lam = 0.0 if var_obs <= 0 else float(np.clip(1.0 - var0 / var_obs, 0.0, 1.0))
        else:
            lam = float(shrink)
            if not 0.0 <= lam <= 1.0:
                raise ValueError(f"shrink must be in [0, 1], 'auto' or None; got {shrink}")
        wk = (1.0 - lam) / keep.sum() + lam * s
        info["lambda"] = lam
    else:
        wk = np.ones(keep.sum())
    wk = wk * screen[keep]
    if dedup and keep.sum() > 1:
        idx = np.flatnonzero(keep)
        vd = sigma2[idx] / mass[idx] ** 2
        if floor == "n":
            m_eff = np.maximum(np.minimum(n[idx], n_cap).astype(float), 1.0)
            vd = np.maximum(vd, float(np.median(vd * m_eff)) / m_eff)
        P = prof[idx]
        d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        link = d2 < dup_ratio * (vd[:, None] + vd[None, :]) / 4.0
        np.fill_diagonal(link, False)
        group = np.arange(len(idx))
        for i, j in zip(*np.nonzero(np.triu(link))):  # union-find on duplicate links
            gi, gj = group[i], group[j]
            if gi != gj:
                group[group == gj] = gi
        sizes = np.bincount(group, minlength=len(idx))[group]
        wk = wk / sizes
        info["duplicate_groups"] = [idx[group == g].tolist() for g in np.unique(group) if (group == g).sum() > 1]
    if cap is not None and snr:
        wk = np.minimum(wk, cap * np.median(wk))
    w[keep] = wk
    w = w / w.sum()
    if stage2 == "geomedian":
        idx = np.flatnonzero(keep)
        _, w_eff = _weighted_geomedian(mus[idx], w[idx])
        w = np.zeros(K)
        w[idx] = w_eff
        phi = w @ mus
    else:
        phi = w @ prof * float(np.median(mass[keep])) if rescale else w @ mus
    info.setdefault("fallback", False)
    info["kept"] = np.flatnonzero(keep).tolist()
    return AggResult(phi=phi, weights=w, info=info)


# --------------------------------------------------------------------------- additional baselines
def foolsgold(mus, sigma2, n, kappa: float = 1.0, eps: float = 1e-5) -> AggResult:
    """FoolsGold (Fung et al., 2020), one-round version on importance vectors.
    cs_ij = cosine(mu_i, mu_j); v_i = max_j cs_ij; pardoning cs_ij *= v_i / v_j if v_j > v_i;
    w_i = 1 - max_j cs_ij, clipped to [0, 1], rescaled by its maximum, then the logit
    w_i = kappa * (ln(w_i / (1 - w_i)) + 0.5) clipped to [0, 1]. Clients that look alike get low weight."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K = mus.shape[0]
    if K < 2:
        return _weighted(mus, np.ones(1), {"name": "foolsgold"})
    nrm = np.linalg.norm(mus, axis=1)
    U = np.zeros_like(mus)
    U[nrm > 0] = mus[nrm > 0] / nrm[nrm > 0, None]  # a zero report has cosine 0 with everyone
    cs = U @ U.T
    np.fill_diagonal(cs, -np.inf)
    v = cs.max(axis=1)
    cs2 = cs.copy()
    for i in range(K):
        for j in range(K):
            if i != j and v[j] > v[i]:
                cs2[i, j] *= v[i] / v[j]
    w = np.clip(1.0 - cs2.max(axis=1), 0.0, 1.0)
    if w.max() > 0:
        w = w / w.max()
    w = np.where(w >= 1.0, 1.0 - eps, w)
    w = np.clip(kappa * (np.log(np.maximum(w, eps) / (1.0 - w)) + 0.5), 0.0, 1.0)
    return _weighted(mus, _normalize(w), {"name": "foolsgold"})


def centered_clipping(mus, sigma2, n, tau: float | None = None, iters: int = 3) -> AggResult:
    """Centered clipping (Karimireddy et al., 2021): v <- v + mean_k clip(mu_k - v, tau), started at the
    coordinate-wise median; tau defaults to the median distance of the reports to the start point."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    v = np.median(mus, axis=0)
    if tau is None:
        tau = float(np.median(np.linalg.norm(mus - v, axis=1)))
    tau = max(tau, 1e-300)
    for _ in range(iters):
        dlt = mus - v
        r = np.maximum(np.linalg.norm(dlt, axis=1), 1e-300)
        v = v + np.mean(dlt * np.minimum(1.0, tau / r)[:, None], axis=0)
    return AggResult(phi=v, weights=np.full(mus.shape[0], np.nan),
                     info={"name": "centered_clipping", "tau": tau, "fallback": False})


def bulyan(mus, sigma2, n, f: int | None = None) -> AggResult:
    """Bulyan (El Mhamdi et al., 2018): select theta = K - 2f clients by repeated Krum, then per coordinate
    average the beta = theta - 2f values closest to the coordinate-wise median of the selection.
    Default f = floor((K - 3) / 4); info["bulyan_degenerate"] flags K < 4f + 3."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K = mus.shape[0]
    f = (K - 3) // 4 if f is None else int(f)
    f = max(f, 0)
    theta = max(K - 2 * f, 1)
    beta = max(theta - 2 * f, 1)
    remaining = list(range(K))
    sel = []
    while len(sel) < theta and remaining:
        if len(remaining) > 1:
            scores = _krum_scores(mus[remaining], f)[0]
            pick = remaining[int(np.argmin(scores))]
        else:
            pick = remaining[0]
        sel.append(pick)
        remaining.remove(pick)
    S = mus[sel]
    med = np.median(S, axis=0)
    order = np.argsort(np.abs(S - med), axis=0, kind="stable")[:beta]
    phi = np.take_along_axis(S, order, axis=0).mean(axis=0)
    return AggResult(phi=phi, weights=np.full(K, np.nan),
                     info={"name": "bulyan", "f": f, "theta": theta, "beta": beta, "selected": sorted(sel),
                           "bulyan_degenerate": K < 4 * f + 3, "fallback": False})


# --------------------------------------------------------------------------- Power-SNR v3
def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted q-quantile (lower inverse of the weighted empirical CDF)."""
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    cw = cw / cw[-1]
    return float(x[o][min(int(np.searchsorted(cw, q)), len(x) - 1)])


def _cw_huber(P: np.ndarray, w: np.ndarray, c: float, scale_q: float, iters: int = 3,
              floor: float = 1e-3) -> np.ndarray:
    """Coordinate-wise weighted Huber location: start at the weighted median; scale
    s_j = 1.4826 * weighted scale_q-quantile of |x_kj - med_j| (scale_q > 0.5 keeps s_j > 0 on sparse
    coordinates, where most banks report zero); `iters` steps loc += sum_k w_k clip(x_kj - loc, +-c s_j)."""
    out = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        x = P[:, j]
        loc = _wquantile(x, w, 0.5)
        s = 1.4826 * _wquantile(np.abs(x - loc), w, scale_q)
        s = max(s, floor * abs(loc), 1e-300)
        for _ in range(iters):
            loc = loc + float(w @ np.clip(x - loc, -c * s, c * s)) / w.sum()
        out[j] = loc
    return out


def _cw_trim(P: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    """Coordinate-wise weighted trimmed mean: remove weight mass beta from each tail of every coordinate
    (the report at a cut keeps the part of its weight inside [beta, 1 - beta]) and average the rest."""
    wn = w / w.sum()
    out = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        o = np.argsort(P[:, j], kind="stable")
        x, ww = P[o, j], wn[o]
        hi = np.cumsum(ww)
        keep = np.clip(np.minimum(hi, 1.0 - beta) - np.maximum(hi - ww, beta), 0.0, None)
        out[j] = float(keep @ x) / keep.sum() if keep.sum() > 0 else _wquantile(P[:, j], w, 0.5)
    return out


def _fit_ts_weights(P: np.ndarray, n: np.ndarray, beta: float = 0.3) -> np.ndarray:
    """Variance model r_k^2 = tau2 + a / n_k, fitted by Theil-Sen on the squared L2 distances of the profiles to
    their unweighted coordinate-wise trimmed mean (beta); weights 1 / (tau2 + a / n_k). a >= 0 and
    tau2 >= 5% of the median r^2, so size helps only as far as the banks' dispersion shows it does
    (a = 0: equal weights; tau2 small: inverse-variance n weights). Fewer than 4 banks or equal sizes: equal."""
    if len(n) < 4 or np.ptp(1.0 / n) < 1e-12:
        return np.ones(len(n))
    r2 = np.sum((P - _cw_trim(P, np.ones(len(n)), beta)) ** 2, axis=1)
    a, t = stats.theilslopes(r2, 1.0 / n)[:2]
    t = max(t, 0.05 * float(np.median(r2)))
    return 1.0 / (t + max(a, 0.0) / n)


def psnr_v3(mus, sigma2, n, z: float = 2.0, mass_z: float = 3.0, n_cap: int = 1000, weight: str = "n",
            cap: float | None = 1.5, dedup: bool = True, dup_ratio: float = 4.0, combine: str = "trim",
            trim_beta: float = 0.3, huber_c: float = 0.5, scale_q: float = 0.8, screens: bool = True,
            profile_screen: bool = False) -> AggResult:
    """FedXRA: auditable sample-size weights and a coordinate-wise trimmed combination.

    No bank's own sigma2 enters its weight; sigma2 is used only through the pooled median kappa
    (and per bank only for weight="snr", the v2-style ablation).
      1. Screens (screens=True): drop dead reports and log-mass outliers (|log m_k - med| > mass_z * 1.4826 MAD);
         with profile_screen=True (ablation, the round-7 design) also profile outliers (L1 distance to the
         coordinate-wise median profile > med + z * 1.4826 MAD). The profile screen is off by default: a coalition
         near the centre passes it while it removes honest banks, which raises the coalition's share.
      2. Weights: weight="n" -> min(n_k, n_cap) (the local model's estimation error scales as 1/n);
         "unif" -> equal; "fit_ts" -> 1 / (tau2 + a / n_k), the variance model fitted robustly on the
         screened banks (_fit_ts_weights), so n counts only as far as the dispersion supports it;
         "snr" -> ||p_k||^2 / max(v_k, kappa / n_k) as in v2 (ablation).
      3. Duplicate grouping (dedup=True): banks i, j are linked when ||p_i - p_j||^2 < dup_ratio *
         (kappa/n_i + kappa/n_j) / 4, kappa = median_k(v_k * min(n_k, n_cap)), v_k = sigma2_k / m_k^2;
         each member of a connected group gets weight / group size.
      4. Cap: w_k <= cap * median(w).
      5. Combination on profiles p_k = mu_k / m_k: combine="trim" -> coordinate-wise weighted trimmed mean
         (_cw_trim, beta = trim_beta); "median" -> coordinate-wise weighted median; "huber" -> coordinate-wise
         weighted Huber
         (_cw_huber, c = huber_c, scale quantile scale_q); "mean" -> weighted mean; "geomedian" -> weighted
         geometric median (ablation: rewards central reports). Output = profile * median survivor mass.
    Reported weights are the combination weights (the trimming and Huber stages act on values, not weights)."""
    mus, sigma2, n = _validate(mus, sigma2, n)
    K, d = mus.shape
    if weight not in ("n", "unif", "snr", "fit_ts"):
        raise ValueError(f"weight must be 'n', 'unif', 'snr' or 'fit_ts'; got {weight!r}")
    if combine not in ("trim", "huber", "mean", "median", "geomedian"):
        raise ValueError(f"combine must be 'trim', 'huber', 'mean', 'median' or 'geomedian'; got {combine!r}")
    info: dict = {"name": "psnr_v3", "weight": weight, "combine": combine, "dedup": dedup}
    mass = mus.sum(axis=1)
    keep = mass > 0
    if keep.sum() < 2:
        info["fallback"] = True
        return _median_result(mus, info)
    P = np.zeros_like(mus)
    P[keep] = mus[keep] / mass[keep, None]
    if screens:
        logm = np.log(np.where(keep, mass, 1.0))
        med_m = float(np.median(logm[keep]))
        mad_m = max(float(np.median(np.abs(logm[keep] - med_m))), 1e-12)
        keep &= np.abs(logm - med_m) <= mass_z * 1.4826 * mad_m
    if screens and profile_screen:
        p_med = np.median(P[keep], axis=0)
        p_med = p_med / p_med.sum() if p_med.sum() > 0 else p_med
        dist = np.abs(P - p_med).sum(axis=1)
        keep &= dist <= _robust_upper(dist[keep], z)
    if keep.sum() == 0:
        info["fallback"] = True
        return _median_result(mus, info)
    idx = np.flatnonzero(keep)
    Pk = P[idx]
    nk = np.maximum(np.minimum(n[idx], n_cap), 1.0)
    v = sigma2[idx] / mass[idx] ** 2
    kappa = max(float(np.median(v * nk)), 1e-300)
    if weight == "n":
        wk = nk.copy()
    elif weight == "unif":
        wk = np.ones(len(idx))
    elif weight == "fit_ts":
        wk = _fit_ts_weights(Pk, nk, trim_beta)
    else:
        wk = np.sum(Pk ** 2, axis=1) / np.maximum(np.maximum(v, kappa / nk), 1e-300)
    wk = wk / wk.sum()
    if dedup and len(idx) > 1:
        s = kappa / nk
        d2 = np.sum((Pk[:, None, :] - Pk[None, :, :]) ** 2, axis=-1)
        link = d2 < dup_ratio * (s[:, None] + s[None, :]) / 4.0
        np.fill_diagonal(link, False)
        group = np.arange(len(idx))
        for i, j in zip(*np.nonzero(np.triu(link))):
            gi, gj = group[i], group[j]
            if gi != gj:
                group[group == gj] = gi
        wk = wk / np.bincount(group, minlength=len(idx))[group]
        info["duplicate_groups"] = [idx[group == g].tolist() for g in np.unique(group) if (group == g).sum() > 1]
    if cap is not None:
        wk = np.minimum(wk, cap * np.median(wk))
    wk = wk / wk.sum()
    if combine == "trim":
        prof, weff = _cw_trim(Pk, wk, trim_beta), wk
    elif combine == "median":
        prof, weff = np.array([_wquantile(Pk[:, j], wk, 0.5) for j in range(d)]), wk
    elif combine == "huber":
        prof, weff = _cw_huber(Pk, wk, huber_c, scale_q), wk
    elif combine == "mean":
        prof, weff = wk @ Pk, wk
    else:
        prof, weff = _weighted_geomedian(Pk, wk)
    prof = np.maximum(prof, 0.0)
    prof = prof / prof.sum() if prof.sum() > 0 else prof
    w = np.zeros(K)
    w[idx] = weff
    info.update(fallback=False, kept=idx.tolist())
    return AggResult(phi=prof * float(np.median(mass[idx])), weights=w, info=info)


# --------------------------------------------------------------------------- registry
AGGREGATORS: dict[str, Callable[..., AggResult]] = {
    "mean": mean,
    "fedavg": fedavg,
    "median": median,
    "trimmed_mean": trimmed_mean,
    "geomedian": geomedian,
    "krum": krum,
    "multikrum": multikrum,
    "ivw": ivw,
    "dersimonian_laird": dersimonian_laird,
    "fltrust_like": fltrust_like,
    "psnr_energy_only": psnr_energy_only,
    "psnr_base": psnr_base,
    "psnr_gate": psnr_gate,
    "psnr_legacy": psnr_legacy,
    "psnr_robust": psnr_robust,
    "psnr_temp": psnr_temp,
    "psnr_screen": psnr_screen,
    "psnr_v3": psnr_v3,
    "foolsgold": foolsgold,
    "centered_clipping": centered_clipping,
    "bulyan": bulyan,
}


def aggregate(name: str, mus, sigma2, n, **kw) -> AggResult:
    """Dispatch to AGGREGATORS[name], forwarding only the kwargs that rule accepts
    (unknown kwargs are silently ignored so a common kwargs dict can be shared)."""
    if name not in AGGREGATORS:
        raise KeyError(f"unknown aggregator {name!r}; available: {sorted(AGGREGATORS)}")
    fn = AGGREGATORS[name]
    params = inspect.signature(fn).parameters
    accepted = {k: v for k, v in kw.items() if k in params and k not in ("mus", "sigma2", "n")}
    res = fn(mus, sigma2, n, **accepted)
    res.info.setdefault("name", name)
    res.info.setdefault("fallback", False)
    return res
