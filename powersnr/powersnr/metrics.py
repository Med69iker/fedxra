"""Fidelity, weight-share and attack-success metrics.

Conventions
-----------
* Rank 1 = largest importance. Ranks use average ranks for ties (for the target
  rank as well as for Kendall/Spearman), so ranks may be half-integers.
* ``target_rank_shift = target_rank_est - target_rank_true``: positive means the
  target feature was pushed DOWN the ranking (hidden), negative means promoted.
* Top-k sets break ties by lower feature index (stable sort).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats as sps

NAN = float("nan")

HIDING_ATTACKS = {"feature_hiding", "mimic", "collusion", "collusion_jitter", "adaptive_psnr", "alie", "whitebox", "whitebox_strong"}
PROMOTION_ATTACKS = {"feature_promotion"}
UNTARGETED_ATTACKS = {"frozen", "zero", "scaling", "input_noise", "label_flip", "minmax"}


def importance_ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks with rank 1 = largest value."""
    return sps.rankdata(-np.asarray(v, dtype=float), method="average")


def _topk(v: np.ndarray, k: int) -> set[int]:
    k = min(k, v.shape[0])
    return set(np.argsort(-v, kind="stable")[:k].tolist())


def _l1norm(v: np.ndarray) -> np.ndarray | None:
    s = float(np.sum(np.abs(v)))
    if not np.isfinite(s) or s == 0.0:
        return None
    return v / s


def fidelity_metrics(phi: np.ndarray, oracle: np.ndarray, feature_names: list[str],
                     target_feature: str | None = None, ks=(3, 5)) -> dict[str, float]:
    """Compare an aggregated importance vector ``phi`` with the ``oracle``.

    Keys: rmse_raw, rmse_norm (on L1-normalised vectors; NaN if either sums to
    0), cosine (NaN if either norm is 0), kendall_tau, spearman (tie-aware;
    NaN for constant inputs), top{k}_jaccard for k in ks, target_rank_true,
    target_rank_est, target_rank_shift (NaN if no target).
    """
    phi = np.asarray(phi, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    if phi.ndim != 1 or phi.shape != oracle.shape:
        raise ValueError(f"phi and oracle must be 1-D of equal shape, got {phi.shape} vs {oracle.shape}")
    if len(feature_names) != phi.shape[0]:
        raise ValueError(f"len(feature_names)={len(feature_names)} != d={phi.shape[0]}")

    out: dict[str, float] = {}
    out["rmse_raw"] = float(np.sqrt(np.mean((phi - oracle) ** 2)))
    pn, on = _l1norm(phi), _l1norm(oracle)
    out["rmse_norm"] = NAN if pn is None or on is None else float(np.sqrt(np.mean((pn - on) ** 2)))
    na, nb = float(np.linalg.norm(phi)), float(np.linalg.norm(oracle))
    out["cosine"] = NAN if na == 0.0 or nb == 0.0 else float(np.dot(phi, oracle) / (na * nb))

    const = np.ptp(phi) == 0 or np.ptp(oracle) == 0
    if const or phi.shape[0] < 2:
        out["kendall_tau"] = NAN
        out["spearman"] = NAN
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["kendall_tau"] = float(sps.kendalltau(phi, oracle)[0])
            out["spearman"] = float(sps.spearmanr(phi, oracle)[0])

    for k in ks:
        a, b = _topk(phi, k), _topk(oracle, k)
        out[f"top{k}_jaccard"] = float(len(a & b) / len(a | b)) if (a | b) else NAN

    if target_feature is None:
        out["target_rank_true"] = out["target_rank_est"] = out["target_rank_shift"] = NAN
    else:
        if target_feature not in feature_names:
            raise ValueError(f"target_feature {target_feature!r} not in feature_names")
        j = feature_names.index(target_feature)
        rt = float(importance_ranks(oracle)[j])
        re = float(importance_ranks(phi)[j])
        out["target_rank_true"], out["target_rank_est"], out["target_rank_shift"] = rt, re, re - rt
    return out


def weight_shares(weights: np.ndarray, roles: list[str]) -> dict[str, float]:
    """Total normalised weight per role (keys w_honest, w_minority, w_attacker).

    Returns all-NaN if any weight is NaN (rule not weight-based) or if the
    weights sum to zero.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != len(roles):
        raise ValueError(f"weights shape {w.shape} incompatible with {len(roles)} roles")
    valid = {"honest", "minority", "attacker"}
    bad = set(roles) - valid
    if bad:
        raise ValueError(f"unknown roles {bad}; expected subset of {valid}")
    keys = ["w_honest", "w_minority", "w_attacker"]
    s = float(np.sum(w)) if w.size else NAN
    if w.size == 0 or np.any(np.isnan(w)) or s == 0.0:
        return {k: NAN for k in keys}
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    w = w / s
    r = np.asarray(roles)
    return {f"w_{role}": float(w[r == role].sum()) for role in ("honest", "minority", "attacker")}


def attack_success(metrics: dict, attack: str, threshold_rank_shift: int = 2) -> float:
    """1.0 / 0.0 success indicator (NaN when undefined).

    * hiding attacks (feature_hiding, mimic, collusion, adaptive_psnr): target
      dropped by >= threshold ranks (target_rank_shift >= threshold).
    * feature_promotion: target rose by >= threshold ranks (shift <= -threshold).
    * other attacks: rmse_norm > 2 * metrics['rmse_norm_ref'] if that reference
      (e.g. the no-attack rmse_norm of the same aggregator) is present, else NaN.
    * 'none' / None: NaN.
    """
    if attack is None or attack == "none":
        return NAN

    def _isnan(x):
        return x is None or (isinstance(x, float) and math.isnan(x)) or (np.isscalar(x) and np.isnan(x))

    if attack in HIDING_ATTACKS or attack in PROMOTION_ATTACKS:
        shift = metrics.get("target_rank_shift", NAN)
        if _isnan(shift):
            return NAN
        if attack in HIDING_ATTACKS:
            return 1.0 if shift >= threshold_rank_shift else 0.0
        return 1.0 if -shift >= threshold_rank_shift else 0.0
    if attack in UNTARGETED_ATTACKS:
        ref = metrics.get("rmse_norm_ref", NAN)
        val = metrics.get("rmse_norm", NAN)
        if _isnan(ref) or _isnan(val):
            return NAN
        return 1.0 if val > 2.0 * ref else 0.0
    raise ValueError(f"attack_success: unknown attack {attack!r}")
