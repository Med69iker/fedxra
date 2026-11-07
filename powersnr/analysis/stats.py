"""Statistical summaries and tests for tidy Power-SNR results.

All functions take the tidy results DataFrame described in INTERFACES.md
(one row per experiment x condition x seed x aggregator).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import KEY_COLS

# Nemenyi critical values q_alpha (alpha = 0.05) = studentized range q(k, inf) / sqrt(2).
# k = 2..10 from Demsar (2006), Table 5; k = 11..20 from scipy.stats.studentized_range.
Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
    10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
    17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def _as_list(by) -> list:
    if by is None:
        return []
    if isinstance(by, str):
        return [by]
    return list(by)


def _check_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in DataFrame: {missing}")


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. NaNs are dropped; returns (nan, nan) if empty."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [a, 1.0 - a])
    return float(lo), float(hi)


def summarize(df: pd.DataFrame, by, metric: str, n_boot: int = 2000, level: float = 0.95,
              seed: int = 0) -> pd.DataFrame:
    """Group `df` by `by` and summarize `metric`.

    Returns columns: *by, mean, std (ddof=1), n (non-NaN count), ci_low, ci_high
    (95% percentile bootstrap CI of the mean, `n_boot` resamples, seeded rng).
    """
    by = _as_list(by)
    _check_cols(df, by + [metric])
    rng = np.random.default_rng(seed)
    rows = []
    if by:
        groups = df.groupby(by, dropna=False, sort=True)
    else:
        groups = [((), df)]
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        x = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        lo, hi = bootstrap_ci(x, rng, n_boot=n_boot, level=level)
        rows.append(dict(zip(by, key)) | {
            "mean": float(np.mean(x)) if x.size else float("nan"),
            "std": float(np.std(x, ddof=1)) if x.size > 1 else (0.0 if x.size == 1 else float("nan")),
            "n": int(x.size),
            "ci_low": lo,
            "ci_high": hi,
        })
    return pd.DataFrame(rows, columns=by + ["mean", "std", "n", "ci_low", "ci_high"])


# --------------------------------------------------------------------------------------
# Paired tests
# --------------------------------------------------------------------------------------
def holm_correction(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs are kept and ignored)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for i, j in enumerate(order):
        adj = min(1.0, (m - i) * p[j])
        running = max(running, adj)
        out[j] = running
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = P(a > b) - P(a < b) over all pairs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on paired x, y (zero_method='wilcox').

    All-zero differences (complete ties) -> (0.0, 1.0). Fewer than 1 pair -> NaN.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    if np.all(d == 0):
        return 0.0, 1.0
    res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def _pair_keys(df: pd.DataFrame, by: list, metric: str) -> list:
    """Non-aggregator key columns used to pair rows (seed + any other key columns)."""
    return [c for c in KEY_COLS if c in df.columns and c not in by and c != metric]


def paired_tests(df: pd.DataFrame, by, metric: str, reference: str = "psnr_screen",
                 lower_is_better: bool = True, alpha: float = 0.05,
                 pair_on: Sequence[str] | None = None) -> pd.DataFrame:
    """Paired comparison of `reference` against every other aggregator, within each group of `by`.

    Rows are paired across aggregators on `pair_on` (default: every key column of the tidy
    schema that is not in `by`, i.e. seed and any remaining condition columns). Duplicate
    rows for the same pairing key are averaged.

    Output columns: *by, reference, aggregator, n_pairs, mean_ref, mean_other,
    median_diff (other - ref), statistic, p_value, p_holm (Holm across the comparisons of
    the group), cliffs_delta (> 0 means the reference is better, direction-aware), wins,
    ties, losses (from the reference's point of view), significant (p_holm < alpha),
    ref_better (significant and the reference wins on median).
    """
    by = _as_list(by)
    _check_cols(df, by + [metric, "aggregator"])
    keys = list(pair_on) if pair_on is not None else _pair_keys(df, by, metric)
    _check_cols(df, keys)
    cols = by + ["reference", "aggregator", "n_pairs", "mean_ref", "mean_other", "median_diff",
                 "statistic", "p_value", "p_holm", "cliffs_delta", "wins", "ties", "losses",
                 "significant", "ref_better"]
    rows = []
    d = df.assign(**{metric: pd.to_numeric(df[metric], errors="coerce")})
    idx = by + keys
    if not idx:
        d = d.assign(_all=0)
        idx = ["_all"]
    # one wide table (rows = by + pairing keys, columns = aggregators), then split by group
    wide_all = d.groupby(idx + ["aggregator"], dropna=False, sort=True)[metric].mean() \
        .unstack("aggregator")
    if by:
        lvl = 0 if len(by) == 1 else list(range(len(by)))
        groups = wide_all.groupby(level=lvl, dropna=False, sort=True)
    else:
        groups = [((), wide_all)]
    for gkey, wide in groups:
        if not isinstance(gkey, tuple):
            gkey = (gkey,)
        if reference not in wide.columns or wide[reference].notna().sum() == 0:
            continue
        present = [a for a in wide.columns if a != reference and wide[a].notna().any()]
        group_rows = []
        for other in sorted(present):
            pair = wide[[reference, other]].dropna()
            ref = pair[reference].to_numpy(dtype=float)
            oth = pair[other].to_numpy(dtype=float)
            stat, p = wilcoxon_test(ref, oth)
            if lower_is_better:
                wins = int(np.sum(ref < oth)); losses = int(np.sum(ref > oth))
                delta = cliffs_delta(oth, ref)
            else:
                wins = int(np.sum(ref > oth)); losses = int(np.sum(ref < oth))
                delta = cliffs_delta(ref, oth)
            ties = int(np.sum(ref == oth))
            med = float(np.median(oth - ref)) if ref.size else float("nan")
            group_rows.append(dict(zip(by, gkey)) | {
                "reference": reference, "aggregator": other, "n_pairs": int(ref.size),
                "mean_ref": float(ref.mean()) if ref.size else float("nan"),
                "mean_other": float(oth.mean()) if oth.size else float("nan"),
                "median_diff": med, "statistic": stat, "p_value": p,
                "cliffs_delta": delta, "wins": wins, "ties": ties, "losses": losses,
            })
        padj = holm_correction([r["p_value"] for r in group_rows])
        for r, pa in zip(group_rows, padj):
            r["p_holm"] = float(pa)
            r["significant"] = bool(np.isfinite(pa) and pa < alpha)
            better = (r["median_diff"] > 0) if lower_is_better else (r["median_diff"] < 0)
            r["ref_better"] = bool(r["significant"] and better)
        rows.extend(group_rows)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# Friedman + Nemenyi
# --------------------------------------------------------------------------------------
def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference CD = q_alpha * sqrt(k(k+1) / (6N)) (Demsar 2006)."""
    if alpha != 0.05:
        raise ValueError("only alpha=0.05 critical values are tabulated")
    if k < 2:
        raise ValueError(f"Nemenyi CD needs k >= 2, got k={k}")
    if n_blocks < 1:
        return float("nan")
    # Demsar's table for k <= 20; beyond it the same value from the studentized range distribution
    q = Q_ALPHA_005[k] if k in Q_ALPHA_005 else float(sps.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2))
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def friedman_nemenyi(df: pd.DataFrame, block_cols, metric: str, lower_is_better: bool = True,
                     alpha: float = 0.05):
    """Friedman test over blocks + Nemenyi critical difference.

    Blocks are the unique combinations of `block_cols`; values are first averaged over
    seeds (and any other column not in block_cols) per block x aggregator. Blocks with a
    missing aggregator are dropped. Rank 1 = best (direction from `lower_is_better`).

    Returns (stat, p, avg_ranks: pd.Series sorted ascending, cd).
    """
    block_cols = _as_list(block_cols)
    _check_cols(df, block_cols + [metric, "aggregator"])
    vals = pd.to_numeric(df[metric], errors="coerce")
    d = df.assign(**{metric: vals})
    if block_cols:
        wide = d.groupby(block_cols + ["aggregator"], dropna=False)[metric].mean().unstack("aggregator")
    else:
        wide = d.groupby("aggregator")[metric].mean().to_frame().T
    wide = wide.dropna(axis=0, how="any")
    k = wide.shape[1]
    n = wide.shape[0]
    if k < 2 or n < 1:
        raise ValueError(f"friedman_nemenyi needs >= 2 aggregators and >= 1 complete block (k={k}, N={n})")
    X = wide.to_numpy(dtype=float)
    ranks = sps.rankdata(X if lower_is_better else -X, axis=1)
    avg_ranks = pd.Series(ranks.mean(axis=0), index=wide.columns, name="avg_rank").sort_values()
    if k >= 3 and n >= 2:
        res = sps.friedmanchisquare(*[X[:, j] for j in range(k)])
        stat, p = float(res.statistic), float(res.pvalue)
    elif n >= 2:
        # k == 2: Friedman chi2 from the rank sums (no tie correction), df = 1.
        R = ranks.sum(axis=0)
        stat = float(12.0 / (n * k * (k + 1)) * np.sum(R ** 2) - 3.0 * n * (k + 1))
        p = float(sps.chi2.sf(stat, k - 1))
    else:
        stat, p = float("nan"), float("nan")
    cd = nemenyi_cd(k, n, alpha=alpha)
    return stat, p, avg_ranks, cd


__all__ = ["summarize", "bootstrap_ci", "paired_tests", "holm_correction", "cliffs_delta",
           "wilcoxon_test", "friedman_nemenyi", "nemenyi_cd", "Q_ALPHA_005"]
