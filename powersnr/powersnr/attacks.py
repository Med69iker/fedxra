"""Poisoning attacks against federated SHAP-importance aggregation.

Two families are implemented:

* **Data attacks** (``DATA_ATTACKS``) corrupt the attacker's *local training data*
  before the local model is fitted. The attacker then follows the protocol
  honestly (fits a model, computes SHAP, reports mu / sigma2 / n truthfully with
  respect to the corrupted data).
* **Report attacks** (``REPORT_ATTACKS``) leave local training untouched but
  replace the *reported statistics* (mu, sigma2) sent to the server. The local
  sample size ``n`` is always reported truthfully (it is assumed to be
  verifiable, e.g. by an audit), so FedAvg-style weights cannot be inflated
  directly.

Global threat-model assumptions (shared by all attacks)
-------------------------------------------------------
* The server is honest; it aggregates reports with a public rule.
* A fraction of the K clients is Byzantine; Byzantine clients may collude only
  where stated (``collusion``).
* Reported importance vectors live in mean-|SHAP| space and are therefore
  constrained to be non-negative; every attack here respects that constraint
  (a negative coordinate would be trivially rejected by the server).
* Attacks that read ``honest_reports`` assume an *omniscient* adversary that
  observes the honest clients' reports of the current round (e.g. via a
  compromised channel or a semi-honest server leak). This is the standard
  worst-case assumption for evaluating Byzantine-robust aggregation.

All randomness is drawn from the explicit ``rng`` argument.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from powersnr.types import ClientStats

DATA_ATTACKS = ["input_noise", "label_flip"]
REPORT_ATTACKS = [
    "frozen",
    "zero",
    "scaling",
    "feature_hiding",
    "feature_promotion",
    "mimic",
    "collusion",
    "collusion_jitter",
    "adaptive_psnr",
    "alie",
    "minmax",
]
# Rule-specific attack handled by the runner with whitebox_variants() (not a per-client report attack).
WHITEBOX_ATTACKS = ["whitebox", "whitebox_strong"]

_DEFAULT_EPS = 1e-12


# ---------------------------------------------------------------------------
# Data attacks
# ---------------------------------------------------------------------------
def input_noise(X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, *, lam: float = 0.5
                ) -> tuple[pd.DataFrame, np.ndarray]:
    """Gaussian feature-noise injection (data poisoning).

    Threat model
        Knowledge: only the attacker's own local data; no knowledge of other
        clients, of the aggregation rule, or of the global model.
        Capability: perturbs its local feature matrix before training; follows
        the protocol honestly afterwards.
        Goal: untargeted degradation -- a noisy local model yields a diluted,
        less informative importance vector (and a larger split-half
        instability sigma2), contaminating the global explanation.

    Each numeric column j receives i.i.d. N(0, (lam * s_j)^2) noise, where s_j is
    the NaN-ignoring (ddof=0) standard deviation of column j on the attacker's
    own data. Missing values stay missing; constant or all-missing columns are
    left unchanged. Non-numeric columns are left unchanged. Labels are untouched.
    """
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"input_noise: lam must be a finite non-negative number, got {lam}")
    Xn = X.copy()
    num_cols = [c for c in Xn.columns if pd.api.types.is_numeric_dtype(Xn[c])]
    if num_cols:
        A = Xn[num_cols].to_numpy(dtype=float, copy=True)
        with np.errstate(all="ignore"):
            valid = np.isfinite(A)
            cnt = valid.sum(axis=0)
            mean = np.where(cnt > 0, np.where(valid, A, 0.0).sum(axis=0) / np.maximum(cnt, 1), 0.0)
            var = np.where(cnt > 0, (np.where(valid, A - mean, 0.0) ** 2).sum(axis=0) / np.maximum(cnt, 1), 0.0)
        std = np.sqrt(var)
        noise = rng.standard_normal(A.shape) * (lam * std)[None, :]
        A = A + noise  # NaN + noise stays NaN
        Xn[num_cols] = A
    return Xn, np.asarray(y).copy()


def label_flip(X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, *, p: float = 0.2
               ) -> tuple[pd.DataFrame, np.ndarray]:
    """Random binary label flipping (data poisoning).

    Threat model
        Knowledge: only the attacker's own local data.
        Capability: flips each local label independently with probability p
        before training; follows the protocol honestly afterwards.
        Goal: untargeted degradation of the local model and hence of the
        reported importance vector (p = 0.5 destroys all signal; p > 0.5
        inverts it).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"label_flip: p must be in [0, 1], got {p}")
    y = np.asarray(y)
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"label_flip: labels must be binary in {{0,1}}, got {uniq}")
    flip = rng.random(y.shape[0]) < p
    y2 = y.copy()
    y2[flip] = 1 - y2[flip]
    return X.copy(), y2


_DATA_REGISTRY: dict[str, Callable] = {"input_noise": input_noise, "label_flip": label_flip}


def apply_data_attack(name: str, X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, **params
                      ) -> tuple[pd.DataFrame, np.ndarray]:
    """Dispatch a data attack by name. Returns NEW (X, y); inputs are not mutated."""
    if name not in _DATA_REGISTRY:
        raise ValueError(f"Unknown data attack {name!r}; expected one of {DATA_ATTACKS}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1-D with len == X rows ({X.shape[0]}), got shape {y.shape}")
    if "lambda" in params:  # convenience alias
        params["lam"] = params.pop("lambda")
    return _DATA_REGISTRY[name](X, y, rng, **params)


# ---------------------------------------------------------------------------
# Helpers for report attacks
# ---------------------------------------------------------------------------
def _honest_matrix(honest_reports: list[ClientStats], d: int, attack: str) -> np.ndarray:
    if len(honest_reports) == 0:
        raise ValueError(f"{attack}: requires at least one honest report")
    M = np.vstack([np.asarray(h.mu, dtype=float) for h in honest_reports])
    if M.shape[1] != d:
        raise ValueError(f"{attack}: honest mu dimension {M.shape[1]} != attacker dimension {d}")
    return M


def _honest_mean(honest_reports: list[ClientStats], d: int, attack: str) -> np.ndarray:
    return _honest_matrix(honest_reports, d, attack).mean(axis=0)


def _resolve_target(target_feature: str | None, feature_names: list[str], honest_reports: list[ClientStats],
                    own_mu: np.ndarray, mode: str) -> int:
    """Index of the target feature.

    If ``target_feature`` is None, fall back to the most important ('hide') or
    least important ('promote') feature of the honest mean importance vector
    (attacker's own mu if no honest report is available). Ties -> lowest index.
    """
    if target_feature is not None:
        if target_feature not in feature_names:
            raise ValueError(f"target_feature {target_feature!r} not in feature_names")
        return feature_names.index(target_feature)
    ref = (np.vstack([np.asarray(h.mu, dtype=float) for h in honest_reports]).mean(axis=0)
           if honest_reports else own_mu)
    if mode == "hide":
        return int(np.argmax(ref))
    if mode == "promote":
        return int(np.argmin(ref))
    raise ValueError(f"unknown target mode {mode}")


def hiding_vector(mu: np.ndarray, target: int) -> np.ndarray:
    """Set mu[target] = 0 and redistribute its mass to the other features
    proportionally to their current values (uniformly if they are all zero).
    The L1 mass is preserved."""
    mu = np.asarray(mu, dtype=float).copy()
    mass = mu[target]
    mu[target] = 0.0
    others = np.ones(mu.shape[0], dtype=bool)
    others[target] = False
    if not others.any():
        return mu
    s = mu[others].sum()
    if s > 0:
        mu[others] += mass * mu[others] / s
    else:
        mu[others] += mass / others.sum()
    return mu


def _check_mu(mu: np.ndarray, attack: str) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError(f"{attack}: produced non-finite mu")
    if np.any(mu < 0):
        raise ValueError(f"{attack}: produced negative mu (min={mu.min()})")
    return mu


# ---------------------------------------------------------------------------
# Report attacks: each returns (mu, sigma2, extra_info)
# ---------------------------------------------------------------------------
def _frozen(stats, honest, rng, *, feature_names, target_feature, **kw):
    """Frozen / free-rider report.

    Threat model
        Knowledge: honest reports (only their average L1 mass, to be
        scale-plausible).
        Capability: replaces its report by an uninformative constant vector
        and claims perfect stability (sigma2 = 0); does no local work.
        Goal: free-riding and untargeted flattening of the global ranking;
        under SNR-type weighting, sigma2 = 0 maximises the claimed weight.
    """
    if kw:
        raise TypeError(f"frozen: unexpected params {sorted(kw)}")
    d = stats.mu.shape[0]
    mass = float(_honest_matrix(honest, d, "frozen").sum(axis=1).mean())
    return np.full(d, mass / d), 0.0, {}


def _zero(stats, honest, rng, *, feature_names, target_feature, value: float = 1e-8, **kw):
    """Zero report.

    Threat model
        Knowledge: none.
        Capability: reports mu ~ 0 (``value`` per coordinate, default 1e-8)
        and sigma2 = 0.
        Goal: untargeted shrinkage of the global importance magnitudes and
        dilution of honest weight; probes rules that weight by 1/sigma2 only
        (IVW), which give this empty report maximal weight.
    """
    if kw:
        raise TypeError(f"zero: unexpected params {sorted(kw)}")
    if value < 0:
        raise ValueError("zero: value must be >= 0")
    return np.full(stats.mu.shape[0], float(value)), 0.0, {}


def _scaling(stats, honest, rng, *, feature_names, target_feature, lam: float = 10.0, **kw):
    """Scaling (magnitude-inflation) report.

    Threat model
        Knowledge: own honest mu only.
        Capability: multiplies its honest mu by ``lam`` (default 10); sigma2 is
        reported unchanged.
        Goal: dominate magnitude-sensitive aggregators (mean, and energy-based
        weights, whose signal term grows as lam^2) while keeping the ranking
        of its own vector plausible.
    """
    if kw:
        raise TypeError(f"scaling: unexpected params {sorted(kw)}")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"scaling: lam must be finite and >= 0, got {lam}")
    return lam * np.asarray(stats.mu, dtype=float), float(stats.sigma2), {}


def _feature_hiding(stats, honest, rng, *, feature_names, target_feature, **kw):
    """Targeted feature hiding (fairness-washing).

    Threat model
        Knowledge: own honest mu; the identity of the target feature (default:
        the protected attribute, e.g. age; if None, the most important feature
        of the honest mean).
        Capability: zeroes the target's importance in its own report and
        redistributes that mass proportionally to the other features (L1 mass
        preserved, so the report is magnitude-plausible); sigma2 unchanged.
        Goal: push the target feature down the global ranking, concealing the
        model's reliance on a sensitive attribute from auditors.
    """
    if kw:
        raise TypeError(f"feature_hiding: unexpected params {sorted(kw)}")
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "hide")
    return hiding_vector(stats.mu, t), float(stats.sigma2), {"target_idx": t}


def _feature_promotion(stats, honest, rng, *, feature_names, target_feature, factor: float = 2.0, **kw):
    """Targeted feature promotion.

    Threat model
        Knowledge: own honest mu; the target feature (if None, the least
        important feature of the honest mean).
        Capability: raises the target's reported importance to ``factor``
        (default 2) times the maximum of its own mu; other coordinates and
        sigma2 unchanged.
        Goal: push the target feature up the global ranking (e.g. to make an
        irrelevant or proxy variable appear to drive decisions).
    """
    if kw:
        raise TypeError(f"feature_promotion: unexpected params {sorted(kw)}")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("feature_promotion: factor must be > 0")
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "promote")
    mu = np.asarray(stats.mu, dtype=float).copy()
    mu[t] = factor * float(mu.max())
    return mu, float(stats.sigma2), {"target_idx": t}


def _mimic(stats, honest, rng, *, feature_names, target_feature, strength: float = 0.5, **kw):
    """Mimicry with a targeted shift (stealthy hiding).

    Threat model
        Knowledge: all honest reports of the round (omniscient); target feature
        (if None, most important feature of the honest mean).
        Capability: reports a convex combination
        (1 - strength) * mean_honest + strength * hide(mean_honest), i.e. the
        honest consensus moved toward a version with the target hidden, and
        the median honest sigma2, so that its report is statistically
        indistinguishable in magnitude, direction and stability.
        Goal: evade distance-, cosine- and outlier-based filters while still
        lowering the target's global rank.
    """
    if kw:
        raise TypeError(f"mimic: unexpected params {sorted(kw)}")
    if not (0.0 <= strength <= 1.0):
        raise ValueError("mimic: strength must be in [0, 1]")
    d = stats.mu.shape[0]
    m = _honest_mean(honest, d, "mimic")
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "hide")
    mu = (1.0 - strength) * m + strength * hiding_vector(m, t)
    s2 = float(np.median([h.sigma2 for h in honest]))
    return mu, s2, {"target_idx": t}


def collusion_vector(honest_reports: list[ClientStats], *, feature_names: list[str],
                     target_feature: str | None = None) -> np.ndarray:
    """The shared colluding vector: feature-hiding applied to the honest mean
    (a deterministic function of the honest reports)."""
    if len(honest_reports) == 0:
        raise ValueError("collusion: requires at least one honest report")
    d = np.asarray(honest_reports[0].mu).shape[0]
    m = _honest_mean(honest_reports, d, "collusion")
    t = _resolve_target(target_feature, feature_names, honest_reports, m, "hide")
    return hiding_vector(m, t)


def _collusion(stats, honest, rng, *, feature_names, target_feature, **kw):
    """Colluding feature hiding.

    Threat model
        Knowledge: all honest reports (omniscient); target feature (if None,
        most important feature of the honest mean); coordination with the
        other attackers.
        Capability: all attackers submit the *identical* vector
        hide(mean_honest) (deterministic given the honest reports), each with
        its own honest sigma2.
        Goal: form a tight, consistent coalition that defeats rules relying on
        the attacker being isolated (Krum, median-type rules, MAD outlier
        tests) and lowers the target's global rank.
    """
    if kw:
        raise TypeError(f"collusion: unexpected params {sorted(kw)}")
    v = collusion_vector(honest, feature_names=feature_names, target_feature=target_feature)
    if v.shape[0] != stats.mu.shape[0]:
        raise ValueError("collusion: dimension mismatch")
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "hide")
    return v, float(stats.sigma2), {"target_idx": t}


def _collusion_jitter(stats, honest, rng, *, feature_names, target_feature, jitter: float = 3.0, **kw):
    """Colluding feature hiding with independent jitter (counter-attack to duplicate grouping).

    Threat model
        Knowledge: as `collusion`, plus the honest sampling-noise level (median honest sigma2)
        and the fact that the server groups reports that are closer than sampling noise allows.
        Capability: every attacker sends hide(mean_honest) plus independent Gaussian noise with
        per-coordinate variance jitter * s2 / (4 d), where s2 is the median honest sigma2, so two
        attackers differ by about jitter / 2 times the expected honest split-half distance
        (jitter = 1: as close as two halves of one honest bank). Declares sigma2 = s2. Values are
        clipped at 0.
        Goal: keep the coalition's common direction while looking statistically independent.
    """
    if kw:
        raise TypeError(f"collusion_jitter: unexpected params {sorted(kw)}")
    if jitter < 0:
        raise ValueError("collusion_jitter: jitter must be >= 0")
    v = collusion_vector(honest, feature_names=feature_names, target_feature=target_feature)
    if v.shape[0] != stats.mu.shape[0]:
        raise ValueError("collusion_jitter: dimension mismatch")
    s2 = float(np.median([h.sigma2 for h in honest]))
    d = v.shape[0]
    mu = np.maximum(v + rng.normal(0.0, np.sqrt(jitter * s2 / (4 * d)), size=d), 0.0)
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "hide")
    return mu, s2, {"target_idx": t, "jitter": jitter}


def adaptive_cap_energy(honest_reports: list[ClientStats], cap: float, *, sigma2_decl: float,
                        sigma2_floor: float | None = None, eps: float = _DEFAULT_EPS,
                        n_attackers: int = 1) -> float:
    """Energy ||mu||^2 that puts an attacker's Power-SNR weight exactly at the
    cap boundary w_a = cap * median(w).

    Honest weights: w_k = ||mu_k||^2 / (max(sigma2_k, floor) + eps) (floor only
    if a scalar ``sigma2_floor`` is given). The median M(x) is taken over the
    honest weights plus ``n_attackers`` identical attacker weights x. M is
    piecewise linear in x with breakpoints at the honest weights, so the fixed
    point x* = cap * M(x*) is solved exactly interval by interval; the largest
    solution (the attacker's best admissible weight) is returned as
    energy = x* (sigma2_decl + eps). Needs only the honest reports.
    """
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("adaptive_psnr: cap must be > 0")
    if n_attackers < 1:
        raise ValueError("n_attackers must be >= 1")
    if len(honest_reports) == 0:
        raise ValueError("adaptive_psnr: requires at least one honest report")
    s2 = np.array([h.sigma2 for h in honest_reports], dtype=float)
    if sigma2_floor is not None:
        s2 = np.maximum(s2, float(sigma2_floor))
    E = np.array([float(np.sum(np.asarray(h.mu, dtype=float) ** 2)) for h in honest_reports])
    w = E / (s2 + eps)

    def M(x: float) -> float:
        return float(np.median(np.concatenate([w, np.full(n_attackers, x)])))

    bps = np.unique(np.concatenate([[0.0], w]))
    sols: list[float] = []
    for lo, hi in zip(bps[:-1], bps[1:]):
        b = (M(hi) - M(lo)) / (hi - lo)
        a = M(lo) - b * lo
        if abs(1.0 - cap * b) > 1e-12:
            x = cap * a / (1.0 - cap * b)
            if lo - 1e-12 * max(1.0, hi) <= x <= hi + 1e-12 * max(1.0, hi):
                sols.append(x)
    # last, unbounded interval [max honest weight, inf): M is linear there too
    top = float(bps[-1])
    far = 2.0 * top + 1.0
    b = (M(far) - M(top)) / (far - top)
    a = M(top) - b * top
    if cap * b < 1.0 - 1e-12:
        x = cap * a / (1.0 - cap * b)
        if x >= top * (1 - 1e-12):
            sols.append(x)
    if not sols:
        raise ValueError("adaptive_psnr: no cap-boundary solution (attackers control the median weight)")
    x_star = max(sols)
    if x_star <= 0:
        raise ValueError("adaptive_psnr: cap-boundary weight is non-positive")
    return x_star * (sigma2_decl + eps)


def _adaptive_psnr(stats, honest, rng, *, feature_names, target_feature, boost: float = 10.0,
                   cap: float | None = None, sigma2_floor: float | None = None, eps: float = _DEFAULT_EPS,
                   n_attackers: int = 1, **kw):
    """Energy-inflation attack (named adaptive_psnr in the configs for continuity).

    Calibrated on SNR-type weights w = ||mu||^2 / (sigma^2 + eps), NOT adapted to each
    deployed rule (the per-rule attack is the white-box coalition, whitebox_variants).

    Threat model
        Knowledge: the SNR-type weight formula and its hyper-parameters (eps, the
        declared variance floor, an assumed cap multiplier) and all honest reports of
        the round (omniscient).
        Capability: reports the feature-hiding direction of its own mu,
        rescaled to a chosen energy, with the smallest admissible instability
        (sigma2 = sigma2_floor if a floor is enforced, else 0).
        Goal: obtain the largest possible Power-SNR weight while hiding the
        target feature. Without ``cap``: energy = ``boost`` (default 10) x the
        maximum honest energy. With ``cap``: the attacker knows the robust rule
        clips w_k <= cap * median(w) and sets its energy so that its weight
        sits exactly on that boundary -- the worst case under clipping, since
        any larger energy is clipped and any smaller one forfeits weight.
    """
    if kw:
        raise TypeError(f"adaptive_psnr: unexpected params {sorted(kw)}")
    d = stats.mu.shape[0]
    H = _honest_matrix(honest, d, "adaptive_psnr")
    if sigma2_floor is not None and (not np.isscalar(sigma2_floor) or sigma2_floor < 0):
        raise ValueError("adaptive_psnr: sigma2_floor must be a non-negative scalar or None")
    s2_decl = float(sigma2_floor) if sigma2_floor is not None else 0.0
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "hide")
    direction = hiding_vector(stats.mu, t)
    if not np.any(direction > 0):
        direction = hiding_vector(H.mean(axis=0), t)
    norm2 = float(np.sum(direction ** 2))
    if norm2 <= 0:
        raise ValueError("adaptive_psnr: cannot build a non-zero hiding direction")
    if cap is None:
        if not np.isfinite(boost) or boost <= 0:
            raise ValueError("adaptive_psnr: boost must be > 0")
        energy = boost * float(np.max(np.sum(H ** 2, axis=1)))
    else:
        energy = adaptive_cap_energy(honest, float(cap), sigma2_decl=s2_decl, sigma2_floor=sigma2_floor,
                                     eps=eps, n_attackers=n_attackers)
    mu = direction * np.sqrt(energy / norm2)
    return mu, s2_decl, {"target_idx": t, "energy": energy}


def _alie(stats, honest, rng, *, feature_names, target_feature, z: float | None = None, n_attackers: int = 1, **kw):
    """Targeted "a little is enough" (ALIE; Baruch et al., 2019) on the protected feature.

    Threat model
        Knowledge: all honest reports (omniscient); the number of attackers.
        Capability: every attacker reports the honest mean with the target coordinate lowered by
        z_max honest standard deviations (clipped at 0), where z_max = Phi^-1((K - s) / K),
        s = floor(K / 2 + 1) - m (Baruch et al.), unless z is given. Declares the median honest sigma2.
        Goal: hide the protected feature while staying inside the honest spread on every coordinate.
    """
    if kw:
        raise TypeError(f"alie: unexpected params {sorted(kw)}")
    from scipy.stats import norm
    d = stats.mu.shape[0]
    H = _honest_matrix(honest, d, "alie")
    t = _resolve_target(target_feature, feature_names, honest, stats.mu, "hide")
    if z is None:
        K = len(honest) + int(n_attackers)
        s = int(np.floor(K / 2 + 1)) - int(n_attackers)
        z = float(norm.ppf(min(max((K - s) / K, 1e-6), 1 - 1e-6)))
    v = H.mean(axis=0)
    v[t] = max(v[t] - z * H[:, t].std(), 0.0)
    return v, float(np.median([h.sigma2 for h in honest])), {"target_idx": t, "z": z}


def _minmax(stats, honest, rng, *, feature_names, target_feature, **kw):
    """AGR-agnostic min-max attack (Shejwalkar and Houmansadr, 2021), untargeted.

    Threat model
        Knowledge: all honest reports (omniscient); no knowledge of the rule.
        Capability: every attacker reports mean + gamma * p with p = -std (inverse-std direction),
        gamma the largest value (bisection) such that the report's maximum distance to any honest
        report does not exceed the maximum honest pairwise distance; clipped at 0. Declares the
        median honest sigma2.
        Goal: move the aggregate as far as possible while looking no more distant than an honest bank.
    """
    if kw:
        raise TypeError(f"minmax: unexpected params {sorted(kw)}")
    d = stats.mu.shape[0]
    H = _honest_matrix(honest, d, "minmax")
    mean, p = H.mean(axis=0), -H.std(axis=0)
    dmax = float(np.max(np.linalg.norm(H[:, None] - H[None], axis=-1))) if len(H) > 1 else 0.0
    lo, hi = 0.0, 1.0
    while hi < 1e6 and np.max(np.linalg.norm(H - np.maximum(mean + hi * p, 0), axis=1)) <= dmax:
        hi *= 2
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.max(np.linalg.norm(H - np.maximum(mean + mid * p, 0), axis=1)) <= dmax:
            lo = mid
        else:
            hi = mid
    return np.maximum(mean + lo * p, 0.0), float(np.median([h.sigma2 for h in honest])), {"gamma": lo}


def _partial_hide(mu: np.ndarray, t: int, h: float) -> np.ndarray:
    """Move a fraction h of mu[t] to the other features (proportionally); h = 1 is hiding_vector."""
    mu = np.asarray(mu, dtype=float).copy()
    mass = h * mu[t]
    mu[t] -= mass
    o = np.ones(mu.shape[0], dtype=bool)
    o[t] = False
    s = mu[o].sum()
    mu[o] += mass * mu[o] / s if s > 0 else mass / o.sum()
    return mu


WHITEBOX_HIDE = (0.3, 0.6, 1.0)
WHITEBOX_CAMO = ("none", "link1.5", "link3", "honest")
WHITEBOX_SIGMA2 = ("zero", "median")


def whitebox_variants(honest_reports: list[ClientStats], n_attackers: int, *, feature_names: list[str],
                      target_feature: str | None, rng: np.random.Generator,
                      family: str = "paper") -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Candidate coalition reports for the WHITE-BOX attack (used by the runner, rule by rule).

    Threat model
        Knowledge: all honest reports; the aggregation rule, which it can evaluate (white-box).
        Capability: the coalition tries every report in a family of 24 variants
        (hide strength h in WHITEBOX_HIDE of the protected feature; camouflage: none, independent jitter
        with per-coordinate variance J * s2med / (4 d) for J in {1.5, 3}, or hiding applied to a distinct
        random honest bank's report instead of the honest mean; claimed sigma2 0 or the honest median) and
        submits the one that pushes the target lowest in the rule's output (the runner does the choice;
        the attackers do not know the oracle).
        family="strong" (attack whitebox_strong) adds the rank-aware families of strong_variants (360 variants).
        Goal: hide the protected feature against the deployed rule specifically.
    Returns [(label, mus (m, d), sigma2 (m,))]."""
    d = np.asarray(honest_reports[0].mu).shape[0]
    H = _honest_matrix(honest_reports, d, "whitebox")
    mean = H.mean(axis=0)
    t = _resolve_target(target_feature, feature_names, honest_reports, mean, "hide")
    s2med = float(np.median([h.sigma2 for h in honest_reports]))
    m = int(n_attackers)
    picks = rng.choice(len(H), size=m, replace=m > len(H))
    noise = rng.normal(size=(m, d))
    out = []
    for h in WHITEBOX_HIDE:
        for camo in WHITEBOX_CAMO:
            if camo == "none":
                M = np.tile(_partial_hide(mean, t, h), (m, 1))
            elif camo.startswith("link"):
                J = float(camo[4:])
                M = np.maximum(_partial_hide(mean, t, h) + noise * np.sqrt(J * s2med / (4 * d)), 0.0)
            else:
                M = np.vstack([_partial_hide(H[p], t, h) for p in picks])
            for s2mode in WHITEBOX_SIGMA2:
                s2 = np.zeros(m) if s2mode == "zero" else np.full(m, s2med)
                out.append((f"h{h}_{camo}_{s2mode}", M, s2))
    if family == "strong":
        out += strong_variants(honest_reports, m, t, rng)
    elif family != "paper":
        raise ValueError(f"family must be 'paper' or 'strong'; got {family!r}")
    return out


def _rank_competitors(base: np.ndarray, t: int, r: int) -> np.ndarray:
    """The r features ranked just below feature t in base."""
    order = np.argsort(-base, kind="stable")
    pos = int(np.flatnonzero(order == t)[0])
    return order[pos + 1: pos + 1 + r]


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    cw = cw / cw[-1]
    return float(x[o][min(int(np.searchsorted(cw, q)), len(x) - 1)])


def strong_variants(honest_reports: list[ClientStats], m: int, t: int,
                    rng: np.random.Generator) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Rank-aware coalition reports (the stronger white-box family).

    Boundary-rank (BR): lower the target's share by a fraction a in {0.4, 0.7, 1.0} and raise the r in {1, 2, 3}
        features ranked just below it in the honest mean profile by g in {0, 0.5, 1, 2} times their gap to the target.
    Boundary (HB): place the target at l0_t - k * 0.5 * s_t and the r in {0..3} competitors at l0_c + k * 0.5 * s_c,
        k in {1, 2, 3}; l0 is the n-weighted coordinate-wise median of the honest profiles and s_j = 1.4826 x the
        weighted 0.8-quantile of |p_j - l0_j| (just inside the reach of clipping rules).
    Each profile is sent as identical copies, with independent jitter of per-coordinate variance
    J * kappa / (2 d n_med), J in {4.5, 8} (beyond the duplicate-grouping threshold), or applied to a distinct random
    honest bank's profile; mass = honest median mass; claimed sigma2 0 or the honest median."""
    H = np.vstack([np.asarray(h.mu, float) for h in honest_reports])
    n_h = np.minimum(np.array([float(h.n) for h in honest_reports]), 1000.0)
    s2_h = np.array([float(h.sigma2) for h in honest_reports])
    mass_h = H.sum(axis=1)
    ok = mass_h > 0
    H, n_h, s2_h, mass_h = H[ok], n_h[ok], s2_h[ok], mass_h[ok]
    d = H.shape[1]
    Ph = H / mass_h[:, None]
    mass = float(np.median(mass_h))
    pmean = Ph.mean(axis=0)
    kappa = float(np.median(s2_h / mass_h ** 2 * n_h))
    nmed = max(float(np.median(n_h)), 1.0)
    noise = rng.normal(size=(m, d))
    picks = rng.choice(len(Ph), size=m, replace=m > len(Ph))
    s2opts = {"zero": np.zeros(m), "median": np.full(m, float(np.median(s2_h)))}
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    def normalise(p: np.ndarray) -> np.ndarray:
        p = np.maximum(p, 0.0)
        return p / p.sum() if p.sum() > 0 else p

    def emit(label: str, make) -> None:
        sets = {"none": np.tile(make(pmean), (m, 1))}
        for J in (4.5, 8.0):
            sd = np.sqrt(J * kappa / nmed / (2 * d))
            sets[f"jit{J:g}"] = np.maximum(np.tile(make(pmean), (m, 1)) + noise * sd, 0.0)
        sets["honest"] = np.vstack([make(Ph[p]) for p in picks])
        for cname, Pm in sets.items():
            Pm = Pm / np.maximum(Pm.sum(axis=1, keepdims=True), 1e-300)
            for sname, S2 in s2opts.items():
                out.append((f"{label}_{cname}_{sname}", Pm * mass, S2))

    for r in (1, 2, 3):
        comp = _rank_competitors(pmean, t, r)
        for a in (0.4, 0.7, 1.0):
            for g in (0.0, 0.5, 1.0, 2.0):
                if r > 1 and g == 0.0:
                    continue

                def make(b, comp=comp, a=a, g=g):
                    p = b.copy()
                    bt = p[t]
                    p[t] = bt * (1.0 - a)
                    for c in comp:
                        p[c] += g * max(bt - p[c], 0.0)
                    return normalise(p)
                emit(f"BR_r{r}_a{a:g}_g{g:g}", make)
    w = np.minimum(n_h, 3.0 * np.median(n_h))
    l0 = np.array([_wquantile(Ph[:, j], w, 0.5) for j in range(d)])
    s = np.array([max(1.4826 * _wquantile(np.abs(Ph[:, j] - l0[j]), w, 0.8), 1e-3 * abs(l0[j]), 1e-300)
                  for j in range(d)])
    for r in (0, 1, 2, 3):
        comp = _rank_competitors(l0, t, r) if r else []
        for k in (1.0, 2.0, 3.0):
            def make(b, comp=comp, k=k):
                p = b.copy()
                p[t] = max(l0[t] - k * 0.5 * s[t], 0.0)
                for c in comp:
                    p[c] = l0[c] + k * 0.5 * s[c]
                return normalise(p)
            emit(f"HB_r{r}_k{k:g}", make)
    return out


def whitebox_score(phi: np.ndarray, target_idx: int) -> tuple[int, float]:
    """Attacker's objective (to maximise): rank of the target in phi (0 = most important), then a smaller share."""
    p = np.asarray(phi, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    return int((p > p[target_idx]).sum()), -float(p[target_idx])


_REPORT_REGISTRY: dict[str, Callable] = {
    "frozen": _frozen,
    "zero": _zero,
    "scaling": _scaling,
    "feature_hiding": _feature_hiding,
    "feature_promotion": _feature_promotion,
    "mimic": _mimic,
    "collusion": _collusion,
    "collusion_jitter": _collusion_jitter,
    "adaptive_psnr": _adaptive_psnr,
    "alie": _alie,
    "minmax": _minmax,
}
assert set(_REPORT_REGISTRY) == set(REPORT_ATTACKS)

# Public docstring lookup (e.g. for generating the threat-model section of the paper).
THREAT_MODELS: dict[str, str] = {
    **{k: (v.__doc__ or "") for k, v in _DATA_REGISTRY.items()},
    **{k: (v.__doc__ or "") for k, v in _REPORT_REGISTRY.items()},
}


def apply_report_attack(name: str, stats: ClientStats, honest_reports: list[ClientStats],
                        rng: np.random.Generator, *, feature_names: list[str],
                        target_feature: str | None = None, **params) -> ClientStats:
    """Replace a client's reported (mu, sigma2) according to attack ``name``.

    Returns a NEW ClientStats (``dataclasses.replace``) with role="attacker",
    the same client_id and n, and ``extra['attack']`` / ``extra['attack_params']``
    (plus attack-specific info such as ``target_idx``). ``stats`` and
    ``honest_reports`` are not mutated. ``mu_signed`` is set to None because a
    report attack does not produce a consistent signed vector.
    """
    if name not in _REPORT_REGISTRY:
        raise ValueError(f"Unknown report attack {name!r}; expected one of {REPORT_ATTACKS}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    mu0 = np.asarray(stats.mu, dtype=float)
    if mu0.ndim != 1:
        raise ValueError("stats.mu must be 1-D")
    if len(feature_names) != mu0.shape[0]:
        raise ValueError(f"len(feature_names)={len(feature_names)} != d={mu0.shape[0]}")
    if "lambda" in params:
        params["lam"] = params.pop("lambda")
    mu, s2, info = _REPORT_REGISTRY[name](stats, list(honest_reports), rng, feature_names=list(feature_names),
                                          target_feature=target_feature, **params)
    mu = _check_mu(mu, name)
    if not np.isfinite(s2) or s2 < 0:
        raise ValueError(f"{name}: invalid sigma2 {s2}")
    extra = dict(stats.extra)
    extra.update({"attack": name, "attack_params": dict(params), "target_feature": target_feature,
                  "honest_role": stats.role})
    extra.update(info)
    return replace(stats, role="attacker", mu=mu.copy(), sigma2=float(s2), mu_signed=None, extra=extra)


def make_collusion_reports(n_attackers: int, honest_reports: list[ClientStats], rng: np.random.Generator,
                           *, feature_names: list[str], target_feature: str | None = None,
                           **params) -> list[np.ndarray]:
    """Return ``n_attackers`` identical (independent copies of the) collusion
    vectors hide(mean_honest). ``rng`` is accepted for interface uniformity and
    is not used (the vector is deterministic given the honest reports)."""
    if params:
        raise TypeError(f"make_collusion_reports: unexpected params {sorted(params)}")
    if n_attackers < 0:
        raise ValueError("n_attackers must be >= 0")
    v = _check_mu(collusion_vector(honest_reports, feature_names=feature_names,
                                   target_feature=target_feature), "collusion")
    return [v.copy() for _ in range(n_attackers)]
