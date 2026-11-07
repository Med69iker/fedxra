"""Fake tidy results generator for analysis tests (no dependency on the powersnr package)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

FIDELITY_KEYS = ["rmse_raw", "rmse_norm", "cosine", "kendall_tau", "spearman", "top3_jaccard",
                 "top5_jaccard", "target_rank_true", "target_rank_est", "target_rank_shift"]
EXTRA_KEYS = ["w_honest", "w_minority", "w_attacker", "rmse_raw_minority", "rmse_norm_minority",
              "kendall_tau_minority", "attack_success", "fallback"]
KEY_COLS = ["experiment", "dataset", "model", "scheme", "K", "attack", "attack_frac",
            "attack_param", "seed", "aggregator"]
ALL_COLUMNS = KEY_COLS + FIDELITY_KEYS + EXTRA_KEYS

# quality: larger = better. psnr_screen is clearly the best.
DEFAULT_AGGS = {"psnr_screen": 1.0, "psnr_base": 0.4, "mean": 0.0}

# Per-experiment condition grids (only the varying key columns are listed).
GRIDS = {
    "E1": dict(attack=["scaling", "feature_hiding", "frozen"]),
    "E2": dict(attack=["scaling", "input_noise"], attack_param=[1.0, 5.0, 10.0]),
    "E3": dict(attack_frac=[0.1, 0.2, 0.3]),
    "E4": dict(K=[5, 10, 20, 40]),
    "E5": dict(scheme=["dirichlet_0.1", "dirichlet_1.0", "dirichlet_10.0"]),
    "E6": dict(dataset=["gmsc", "taiwan", "german"], model=["xgboost", "logreg"]),
    "E7": dict(attack=["mimic", "collusion", "adaptive_psnr"]),
    "E8": dict(attack=["scaling", "feature_hiding"]),
    "E10": dict(scheme=["feature"], attack=["none"]),
}

BASE = dict(experiment="E0", dataset="gmsc", model="xgboost", scheme="iid", K=10,
            attack="scaling", attack_frac=0.2, attack_param=np.nan)


def make_fake_results(experiments=("E1",), aggregators: dict | None = None, n_seeds: int = 10,
                      seed: int = 0, noise: float = 0.01) -> pd.DataFrame:
    """Tidy fake results with all required columns.

    rmse_norm = 0.10 + 0.002 * condition_index - 0.05 * quality + N(0, noise) per (row),
    plus a shared per-seed offset so pairing across aggregators matters.
    """
    aggregators = dict(DEFAULT_AGGS if aggregators is None else aggregators)
    rng = np.random.default_rng(seed)
    rows = []
    for exp in experiments:
        grid = GRIDS.get(exp, {})
        names = list(grid)
        combos = list(itertools.product(*grid.values())) if grid else [()]
        for ci, combo in enumerate(combos):
            cond = dict(BASE, experiment=exp, **dict(zip(names, combo)))
            for s in range(n_seeds):
                seed_off = rng.normal(0, 0.02)
                for agg, q in aggregators.items():
                    eps = rng.normal(0, noise)
                    rmse = max(1e-4, 0.10 + 0.002 * ci - 0.05 * q + seed_off + eps)
                    tau = float(np.clip(0.5 + 0.4 * q - 2 * seed_off + rng.normal(0, noise), -1, 1))
                    w_att = float(np.clip(0.3 - 0.25 * q + rng.normal(0, noise), 0, 1))
                    w_min = float(np.clip(0.05 + 0.03 * q + rng.normal(0, noise / 2), 0, 1))
                    r = dict(cond, seed=s, aggregator=agg,
                             rmse_raw=rmse * 3.0, rmse_norm=rmse,
                             cosine=float(np.clip(1 - rmse, -1, 1)), kendall_tau=tau,
                             spearman=float(np.clip(tau + 0.05, -1, 1)),
                             top3_jaccard=float(np.clip(0.4 + 0.5 * q + rng.normal(0, 0.05), 0, 1)),
                             top5_jaccard=float(np.clip(0.5 + 0.4 * q + rng.normal(0, 0.05), 0, 1)),
                             target_rank_true=1.0, target_rank_est=float(1 + round(3 * (1 - q))),
                             target_rank_shift=float(round(3 * (1 - q))),
                             w_honest=1 - w_att - w_min, w_minority=w_min, w_attacker=w_att,
                             rmse_raw_minority=3.0 * (rmse + 0.05 * (1 - q)),
                             rmse_norm_minority=rmse + 0.05 * (1 - q) + rng.normal(0, noise),
                             kendall_tau_minority=tau - 0.1,
                             attack_success=float(rng.random() < 0.8 * (1 - q)),
                             fallback=False)
                    rows.append(r)
    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    df["K"] = df["K"].astype(int)
    return df
