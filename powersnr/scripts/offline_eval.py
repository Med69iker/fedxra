"""Evaluate aggregators offline on saved client reports (no SHAP recomputation).

Report attacks are replayed on the honest reports with the same attacker choice as the
runner; data attacks (input_noise, label_flip) are taken from the saved reports as they are.

Usage (from the powersnr/ folder):
    python scripts/offline_eval.py --reports results/DEV_reports/client_reports.parquet --out results/DEV_offline
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from powersnr import aggregators as agg
from powersnr import attacks as atk
from powersnr import metrics as met
from powersnr.runner import _rng, n_attackers
from powersnr.types import ClientStats

CELL = ["source", "dataset", "model", "scheme", "scheme_param", "K", "seed"]

REPORT_ATTACKS = [
    ("zero", {}), ("frozen", {}), ("scaling", {"lam": 10}), ("feature_hiding", {}), ("feature_promotion", {}),
    ("mimic", {"strength": 0.5}), ("mimic", {"strength": 1.0}), ("collusion", {}), ("adaptive_psnr", {"boost": 10}),
    ("collusion_jitter", {"jitter": 1.0}), ("collusion_jitter", {"jitter": 3.0}), ("collusion_jitter", {"jitter": 10.0}),
]
FRACS = [0.1, 0.2, 0.3]

BASELINES = {
    "fedavg": ("fedavg", {}),
    "median": ("median", {}),
    "trimmed_mean": ("trimmed_mean", {"beta": 0.2}),
    "geomedian": ("geomedian", {}),
    "multikrum": ("multikrum", {}),
    "screen_only": ("psnr_screen", {"z": 2.0, "snr": False}),
    "psnr_screen_v1": ("psnr_screen", {"z": 2.0, "cap": 2.0, "floor_q": 0.25}),
}


def v2_grid(n_repeats: int = 5) -> dict:
    """All combinations of the four proposed improvements (z = 2, cap = 2 fixed)."""
    grid = {}
    for floor, shrink, soft, stage2 in itertools.product(["quantile", "n"], [None, "auto", 0.5], [False, True],
                                                         ["mean", "geomedian"]):
        name = f"v2_{floor}_{shrink}_{'soft' if soft else 'hard'}_{stage2}"
        grid[name] = ("psnr_screen", {"z": 2.0, "cap": 2.0, "floor_q": 0.25, "floor": floor, "shrink": shrink,
                                      "soft": soft, "stage2": stage2, "n_repeats": n_repeats})
    return grid


def dedup_grid(n_repeats: int = 5) -> dict:
    """Power-SNR v2 (frozen design) with and without duplicate grouping."""
    v2 = {"z": 2.0, "cap": 2.0, "floor": "n", "n_cap": 1000, "shrink": 0.5, "stage2": "geomedian",
          "n_repeats": n_repeats}
    grid = {"v2": ("psnr_screen", v2)}
    for r in [0.1, 0.25, 0.5, 1.0]:
        grid[f"v2_dedup{r}"] = ("psnr_screen", {**v2, "dedup": True, "dup_ratio": r})
        grid[f"v2mean_dedup{r}"] = ("psnr_screen", {**v2, "stage2": "mean", "dedup": True, "dup_ratio": r})
    return grid


GRIDS = {"v2": v2_grid, "dedup": dedup_grid}
CANDIDATES = {**BASELINES, **v2_grid()}


def _stats(row) -> ClientStats:
    return ClientStats(client_id=int(row.client_id), role=row.role, mu=np.asarray(row.mu, float),
                       sigma2=float(row.sigma2), n=int(row.n))


def evaluate_cell(cell: pd.DataFrame, refs: dict, candidates: dict | None = None) -> list[dict]:
    candidates = CANDIDATES if candidates is None else candidates
    keys = cell.iloc[0][CELL].to_dict()
    feat, target = list(refs["oracle"].feature_names), refs["oracle"].target
    oracle = np.asarray(refs["oracle"].mu, float)
    oracle_min = np.asarray(refs["oracle_minority"].mu, float) if "oracle_minority" in refs else None
    seed, K = int(keys["seed"]), int(keys["K"])

    honest_rows = cell[cell.attack == "none"].sort_values("client_id")
    honest = [_stats(r) for r in honest_rows.itertuples()]
    roles0 = [h.role for h in honest]
    standard = [h.client_id for h in honest if h.role != "minority"]
    order = list(_rng(seed, "attackers", K).permutation(standard))

    settings = [("none", {}, 0.0, honest, roles0)]
    for (name, params), frac in itertools.product(REPORT_ATTACKS, FRACS):
        m = n_attackers(frac, K, len(standard))
        att = set(order[:m])
        hon = [h for h in honest if h.client_id not in att]
        reps = [atk.apply_report_attack(name, h, hon, _rng(seed, "rattack", h.client_id, name, frac),
                                        feature_names=feat, target_feature=target, **params)
                if h.client_id in att else h for h in honest]
        settings.append((name if not params else f"{name}_{list(params.values())[0]}", params, frac, reps,
                         ["attacker" if h.client_id in att else h.role for h in honest]))
    for (name, frac, pval), g in cell[cell.attack != "none"].groupby(["attack", "attack_frac", "attack_param"], dropna=False):
        g = g.sort_values("client_id")
        label = name if pd.isna(pval) else f"{name}_{pval:g}"
        settings.append((label, {}, float(frac), [_stats(r) for r in g.itertuples()], list(g.role)))

    rows = []
    for aname, _, frac, reps, roles in settings:
        mus = np.vstack([r.mu for r in reps])
        s2 = np.array([r.sigma2 for r in reps])
        n = np.array([r.n for r in reps], float)
        hon_idx = [i for i, ro in enumerate(roles) if ro != "attacker"]
        ref = agg.aggregate("fedavg", mus[hon_idx], s2[hon_idx], n[hon_idx]).phi
        ref_norm = met.fidelity_metrics(ref, oracle, feat, target)["rmse_norm"]
        base_attack = aname.split("_")[0] if aname.startswith(("mimic", "scaling", "adaptive")) else aname
        base_attack = "collusion_jitter" if aname.startswith("collusion_jitter") else base_attack
        base_attack = "input_noise" if aname.startswith("input_noise") else "label_flip" if aname.startswith("label_flip") else base_attack
        base_attack = {"adaptive": "adaptive_psnr"}.get(base_attack, base_attack)
        for cname, (rule, kw) in candidates.items():
            res = agg.aggregate(rule, mus, s2, n, **kw)
            mm = met.fidelity_metrics(res.phi, oracle, feat, target)
            mm["rmse_norm_ref"] = ref_norm
            row = {**keys, "attack": aname, "attack_frac": frac, "aggregator": cname,
                   "rmse_norm": mm["rmse_norm"], "kendall_tau": mm["kendall_tau"], "top5_jaccard": mm["top5_jaccard"],
                   "rmse_norm_ref": ref_norm, **met.weight_shares(res.weights, roles),
                   "attack_success": met.attack_success(mm, base_attack) if aname != "none" else np.nan}
            if oracle_min is not None:
                row["rmse_norm_minority"] = met.fidelity_metrics(res.phi, oracle_min, feat, target)["rmse_norm"]
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs=2, default=None, help="inclusive seed range, e.g. 0 9")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--grid", choices=sorted(GRIDS), default="v2")
    a = ap.parse_args()
    CANDIDATES.clear()
    CANDIDATES.update({**BASELINES, **GRIDS[a.grid]()})
    df = pd.concat([pd.read_parquet(f).assign(source=Path(f).parent.name.split("_")[-1]) for f in a.reports],
                   ignore_index=True)
    if a.seeds is not None:
        df = df[df.seed.between(a.seeds[0], a.seeds[1])]
    if a.datasets is not None:
        df = df[df.dataset.isin(a.datasets)]
    clients = df[df.record == "client"]
    refs = df[df.record != "client"]
    groups = []
    for keys, cell in clients.groupby(CELL, dropna=False):
        r = refs
        for k, v in zip(CELL, keys):
            r = r[r[k].isna()] if (isinstance(v, float) and np.isnan(v)) else r[r[k] == v]
        groups.append((cell, {row.record: row for row in r.itertuples()}))
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=a.jobs, verbose=2)(delayed(evaluate_cell)(c, r, CANDIDATES) for c, r in groups)
    res = pd.DataFrame([x for rows in out for x in rows])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res.to_parquet(Path(a.out) / "offline.parquet", index=False)
    print(len(res), "rows ->", a.out)


if __name__ == "__main__":
    main()
