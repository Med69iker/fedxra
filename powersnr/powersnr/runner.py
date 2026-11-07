"""Experiment runner: one YAML config -> results/<timestamp>_<experiment>/results.parquet.

Usage (from the powersnr/ folder):
    python -m powersnr.runner --config configs/E1_fallacy.yaml [--jobs 6] [--seeds 3]

Design:
- Per (dataset, seed): eval/root/pool split, then the oracle model is trained on the
  union of all clients' CLEAN data and explained on the held-out eval set. The oracle
  does not depend on the attack configuration, so every attack level is compared with
  the same reference.
- Per (dataset, model, partition, K, seed): partitions are fixed and honest client
  statistics are computed ONCE, then reused by every attack and every aggregator.
- Attackers are the first m clients of a seeded permutation of the standard clients, so
  the attacker sets are nested as the attack fraction grows.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import functools
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from powersnr import aggregators as agg
from powersnr import attacks as atk
from powersnr import data as data_mod
from powersnr import explain as ex
from powersnr import metrics as met
from powersnr import partition as part
from powersnr.types import ClientStats

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- seeds
def _rng(seed: int, *keys) -> np.random.Generator:
    """Independent stream per (seed, component); stable across runs and configs."""
    material = [seed] + [abs(hash_str(str(k))) % (2**31) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(material))


def hash_str(s: str) -> int:
    # Python's hash() is salted per process; use a stable hash instead.
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) % (2**32)
    return h


# ---------------------------------------------------------------- config
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("max_rows", None)
    cfg.setdefault("eval_frac", 0.2)
    cfg.setdefault("root_frac", 0.1)
    cfg.setdefault("client_size", 3000)
    cfg.setdefault("models", ["xgboost"])
    cfg.setdefault("shap", {})
    cfg.setdefault("dataset_overrides", {})
    cfg.setdefault("save_phi_seeds", 1)
    cfg.setdefault("seed_offset", 0)
    return cfg


def expand_attacks(cfg: dict) -> list[dict]:
    """Each attack entry -> list of concrete (name, frac, params) settings."""
    out = [{"name": "none", "frac": 0.0, "params": {}, "param_value": np.nan}]
    for a in cfg.get("attacks", []):
        if a["name"] == "none":
            continue
        fracs = a.get("fracs", [cfg.get("default_attack_frac", 0.2)])
        plist = a.get("params") or [{}]
        for frac, p in itertools.product(fracs, plist):
            p = dict(p or {})
            pval = next(iter(p.values())) if p else np.nan
            out.append({"name": a["name"], "frac": float(frac), "params": p, "param_value": pval})
    return out


def n_attackers(frac: float, K: int, n_standard: int) -> int:
    if frac <= 0:
        return 0
    m = max(1, int(round(frac * K)))
    return min(m, n_standard - 1)  # always keep >= 1 honest standard client


# ---------------------------------------------------------------- core
@functools.lru_cache(maxsize=4)
def _load(dataset: str, max_rows):
    # Same subsample for every seed (seed-dependence enters through the splits and partitions).
    return data_mod.load_dataset(dataset, ROOT / "data" / "raw", max_rows=max_rows, rng=_rng(0, "load", dataset))


def honest_client_stats(ds, idx, tag, cid, model_name, seed, shap_cfg, data_attack=None):
    """Train client `cid` on its rows and return ClientStats (optionally under a data attack)."""
    sub = data_mod.subset(ds, idx)
    X, y = sub.X, sub.y
    role = "minority" if tag == "minority" else "honest"
    if data_attack is not None:
        name, params = data_attack
        X, y = atk.apply_data_attack(name, X, y, _rng(seed, "dattack", cid, name, json.dumps(params, sort_keys=True)), **params)
        role = "attacker"
    model, _, X_test, y_test = ex.fit_client(model_name, X, y, _rng(seed, "fit", cid), seed=seed)
    return ex.client_stats(model, X_test, y_test, _rng(seed, "stats", cid), client_id=cid, role=role, **shap_cfg)


def run_cell(cfg: dict, dataset: str, model_name: str, pspec: dict, K: int, seed: int) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    shap_cfg = cfg["shap"]
    over = cfg["dataset_overrides"].get(dataset, {})
    if pspec["scheme"] == "feature":  # minority group / threshold differ per dataset
        pspec = {**pspec, **over.get("partition", {})}

    ds = _load(dataset, cfg["max_rows"])
    pool, eval_ds = data_mod.split_eval(ds, cfg["eval_frac"], _rng(seed, "eval", dataset))
    eval_ds, root_ds = data_mod.split_eval(eval_ds, cfg["root_frac"], _rng(seed, "root", dataset))

    scheme = pspec["scheme"]
    pkw = {k: v for k, v in pspec.items() if k != "scheme"}
    if scheme == "quantity":  # smallest bank still large enough to contain both classes
        pkw.setdefault("min_size", 300)
    if scheme == "dirichlet":  # every bank holds a usable minimum of each class (up to 25 rows per class)
        smallest = int(np.bincount(np.asarray(pool.y, int)).min())
        pkw.setdefault("min_per_class", max(2, min(25, smallest // (2 * K))))
    client_size = cfg["client_size"]
    if scheme == "quantity":  # the skewed sizes must fit in the pool: largest bank <= n / K
        client_size = min(client_size, len(pool.y) // K)
    parts, tags = part.partition(pool, scheme, K, _rng(seed, "part", dataset, scheme, K), client_size=client_size, **pkw)

    # Oracle: centralized model on the union of all clients' clean data.
    union_idx = np.concatenate(parts)
    union = data_mod.subset(pool, union_idx)
    o_model, o_pre, _, _ = ex.fit_client(model_name, union.X, union.y, _rng(seed, "oracle"), seed=seed, test_frac=0.0)
    Xe = o_pre.transform(eval_ds.X)
    bg = Xe.sample(n=min(shap_cfg.get("n_background", 100), len(Xe)), random_state=seed)
    Xe_explain = Xe.sample(n=min(shap_cfg.get("n_explain", 1000) * 2, len(Xe)), random_state=seed + 1)
    oracle = ex.global_importance(o_model, Xe_explain, bg)
    oracle_min = None
    group = pkw.get("group")
    if scheme == "feature" and group is not None and group in eval_ds.groups:
        gmask = (eval_ds.groups[group] > pkw.get("threshold", 60)).to_numpy()
        Xm = Xe[gmask]
        if len(Xm) >= 50:
            oracle_min = ex.global_importance(o_model, Xm.sample(n=min(len(Xm), 2000), random_state=seed), bg)

    # Server root set (FLTrust-like): small clean dataset held by the server.
    r_model, _, rX, ry = ex.fit_client(model_name, root_ds.X, root_ds.y, _rng(seed, "rootfit"), seed=seed)
    root_mu = ex.client_stats(r_model, rX, ry, _rng(seed, "rootstats"), client_id=-1, role="root", **shap_cfg).mu

    honest = [honest_client_stats(pool, idx, tag, cid, model_name, seed, shap_cfg) for cid, (idx, tag) in enumerate(zip(parts, tags))]

    standard = [c for c, t in enumerate(tags) if t != "minority"]
    order = list(_rng(seed, "attackers", K).permutation(standard))
    feat = list(pool.feature_names)
    target = ds.protected_feature

    rows, phis, saved = [], [], []
    base_keys = dict(experiment=cfg["experiment"], dataset=dataset, model=model_name, scheme=scheme,
                     scheme_param=pkw.get("alpha", pkw.get("sigma", pkw.get("threshold", np.nan))), K=K, seed=seed)
    # References needed to re-evaluate new aggregators offline from the saved client reports.
    for kind, vec in [("oracle", oracle), ("oracle_minority", oracle_min), ("root_mu", root_mu)]:
        if vec is not None:
            saved.append({**base_keys, "record": kind, "attack": "none", "attack_frac": 0.0, "attack_param": np.nan,
                          "n_attackers": 0, "client_id": -1, "role": kind, "n": np.nan, "sigma2": np.nan,
                          "mu": np.asarray(vec, float).tolist(), "feature_names": feat, "target": target})

    for a in expand_attacks(cfg):
        m = n_attackers(a["frac"], K, len(standard))
        if a["name"] != "none" and m == 0:
            continue
        att_ids = set(order[:m])
        reports: list[ClientStats] = [copy.deepcopy(h) for h in honest]
        honest_reports = [h for h in honest if h.client_id not in att_ids]
        if a["name"] in atk.DATA_ATTACKS:
            for cid in att_ids:
                reports[cid] = honest_client_stats(pool, parts[cid], tags[cid], cid, model_name, seed, shap_cfg,
                                                   data_attack=(a["name"], a["params"]))
        elif a["name"] in atk.REPORT_ATTACKS:
            # collusion is deterministic given the honest reports, so every attacker sends the same vector.
            params = dict(a["params"])
            if (a["name"] == "adaptive_psnr" and "cap" in params) or a["name"] == "alie":
                params.setdefault("n_attackers", m)
            for cid in att_ids:
                reports[cid] = atk.apply_report_attack(a["name"], honest[cid], honest_reports,
                                                       _rng(seed, "rattack", cid, a["name"], a["frac"]),
                                                       feature_names=feat, target_feature=target, **params)
        whitebox = a["name"] in atk.WHITEBOX_ATTACKS
        roles = [r.role if r.client_id not in att_ids else "attacker" for r in reports]
        for r, role in zip(reports, [] if whitebox else roles):  # white-box reports differ per rule: not saved
            saved.append({**base_keys, "record": "client", "attack": a["name"], "attack_frac": a["frac"],
                          "attack_param": a["param_value"], "n_attackers": m, "client_id": r.client_id, "role": role,
                          "n": float(r.n), "sigma2": float(r.sigma2), "mu": np.asarray(r.mu, float).tolist(),
                          "feature_names": None, "target": None})
        mus = np.vstack([r.mu for r in reports])
        s2 = np.array([r.sigma2 for r in reports], dtype=float)
        n = np.array([r.n for r in reports], dtype=float)

        # Reference: what an attack-free federation of the remaining clients would produce.
        ref = agg.aggregate("fedavg", np.vstack([h.mu for h in honest_reports]),
                            np.array([h.sigma2 for h in honest_reports]), np.array([h.n for h in honest_reports], float))
        ref_m = met.fidelity_metrics(ref.phi, oracle, feat, target)

        if whitebox:
            wb_variants = atk.whitebox_variants(honest_reports, m, feature_names=feat, target_feature=target,
                                                rng=_rng(seed, "whitebox", a["frac"]),
                                                family="strong" if a["name"] == "whitebox_strong" else "paper")
            att_pos = sorted(att_ids)
            t_idx = feat.index(target)
            if a["params"].get("claim_n") is not None:  # attackers misreport their sample size (outside the threat model)
                n = n.copy()
                n[att_pos] = float(a["params"]["claim_n"])
        for name, kw in cfg["aggregators"].items():
            kw = dict(kw or {})
            base_name = kw.pop("rule", name)  # allow aliases such as psnr_robust_nocap: {rule: psnr_robust, cap: null}
            if base_name == "fltrust_like":
                kw["root_mu"] = root_mu
            wb_label = None
            if whitebox:  # the coalition picks, against THIS rule, the variant that pushes the target lowest
                best = None
                for label, Mu, S2 in wb_variants:
                    mus_v, s2_v = mus.copy(), s2.copy()
                    mus_v[att_pos], s2_v[att_pos] = Mu, S2
                    r_v = agg.aggregate(base_name, mus_v, s2_v, n, **kw)
                    sc = atk.whitebox_score(r_v.phi, t_idx)
                    if best is None or sc > best[0]:
                        best = (sc, label, r_v)
                _, wb_label, res = best
            else:
                res = agg.aggregate(base_name, mus, s2, n, **kw)
            mm = met.fidelity_metrics(res.phi, oracle, feat, target)
            mm["rmse_norm_ref"] = ref_m["rmse_norm"]
            row = {**base_keys, "attack": a["name"], "attack_frac": a["frac"], "n_attackers": m,
                   "attack_param": a["param_value"], "aggregator": name, **mm,
                   **met.weight_shares(res.weights, roles),
                   "attack_success": met.attack_success(mm, a["name"]) if a["name"] != "none" else np.nan,
                   "fallback": bool(res.info.get("fallback", False)), "whitebox_variant": wb_label,
                   "mean_local_auc": float(np.nanmean([h.local_auc for h in honest]))}
            if oracle_min is not None:
                mmin = met.fidelity_metrics(res.phi, oracle_min, feat, target)
                row.update(rmse_raw_minority=mmin["rmse_raw"], rmse_norm_minority=mmin["rmse_norm"],
                           kendall_tau_minority=mmin["kendall_tau"])
            else:
                row.update(rmse_raw_minority=np.nan, rmse_norm_minority=np.nan, kendall_tau_minority=np.nan)
            for cid, w in enumerate(np.asarray(res.weights, float)):
                row[f"w_client{cid}"] = w if K <= 10 else np.nan
            rows.append(row)
            if seed - cfg["seed_offset"] < cfg["save_phi_seeds"]:
                phis.append({**base_keys, "attack": a["name"], "attack_frac": a["frac"], "attack_param": a["param_value"],
                             "aggregator": name, **{f: v for f, v in zip(feat, res.phi)}})
        if seed - cfg["seed_offset"] < cfg["save_phi_seeds"] and a["name"] == "none":
            phis.append({**base_keys, "attack": "none", "attack_frac": 0.0, "attack_param": np.nan,
                         "aggregator": "ORACLE", **{f: v for f, v in zip(feat, oracle)}})
    for r in rows:
        r["cell_runtime_s"] = time.time() - t0
    return rows, phis, saved


def cells(cfg: dict, n_seeds: int | None):
    off = cfg["seed_offset"]  # fresh seeds for confirmatory runs
    seeds = range(off, off + (n_seeds if n_seeds is not None else cfg["seeds"]))
    for dataset, model_name, pspec, K, seed in itertools.product(cfg["datasets"], cfg["models"], cfg["partitions"], cfg["K"], seeds):
        yield dataset, model_name, pspec, K, seed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds (smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"{stamp}_{cfg['experiment']}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    todo = list(cells(cfg, args.seeds))
    print(f"[{cfg['experiment']}] {len(todo)} cells -> {out}", flush=True)
    t0 = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        # Under `python -m`, this module is __main__ and its functions would be pickled by value
        # (breaking the lru_cache loader in workers); send the importable module's function instead.
        from powersnr.runner import run_cell as _run_cell
        results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(_run_cell)(cfg, *c) for c in todo)
    else:
        results = []
        for i, c in enumerate(todo):
            results.append(run_cell(cfg, *c))
            print(f"  cell {i + 1}/{len(todo)} {c[:2]} K={c[3]} seed={c[4]} done ({time.time() - t0:.0f}s)", flush=True)

    rows = [r for rs, _, _ in results for r in rs]
    phis = [p for _, ps, _ in results for p in ps]
    saved = [x for _, _, xs in results for x in xs]
    # Every client's reported (mu, sigma2, n, role) plus the oracle/root vectors of each cell.
    pd.DataFrame(saved).to_parquet(out / "client_reports.parquet", index=False)
    pd.DataFrame(rows).to_parquet(out / "results.parquet", index=False)
    if phis:
        pd.DataFrame(phis).to_parquet(out / "phi_vectors.parquet", index=False)

    import shap, sklearn, xgboost
    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "shap": shap.__version__, "xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
            "cells": len(todo), "rows": len(rows), "runtime_s": round(time.time() - t0, 1), "cmd": vars(args)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done: {len(rows)} rows in {meta['runtime_s']}s -> {out}")
    return out


if __name__ == "__main__":
    main()
