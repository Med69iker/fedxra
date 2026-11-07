"""Score offline candidates: average rank of normalized RMSE per federation type (source
experiment), plus attack success and minority fidelity.

Usage: python scripts/rank_candidates.py results/V2_dev/offline.parquet [--top 15]
"""
from __future__ import annotations

import argparse

import pandas as pd

BLOCK = ["source", "dataset", "scheme", "scheme_param", "K", "seed", "attack", "attack_frac"]


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby(BLOCK, dropna=False)["rmse_norm"].rank()
    out = df.pivot_table(index="aggregator", columns="source", values="rank", aggfunc="mean")
    out.columns = [f"rank_{c}" for c in out.columns]
    out["rank_all"] = df.groupby("aggregator")["rank"].mean()
    out["rank_clean"] = df[df.attack == "none"].groupby("aggregator")["rank"].mean()
    adv = df[df.attack.str.startswith("adaptive")]
    out["succ_adaptive"] = adv.groupby("aggregator")["attack_success"].mean()
    out["succ_all"] = df[df.attack != "none"].groupby("aggregator")["attack_success"].mean()
    out["succ_worst"] = (df[df.attack != "none"].groupby(["aggregator", "attack"])["attack_success"].mean()
                         .unstack().max(axis=1))
    if "rmse_norm_minority" in df:
        out["rmse_minority"] = df.groupby("aggregator")["rmse_norm_minority"].mean()
    return out.sort_values("rank_all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    print(score(pd.read_parquet(a.path)).head(a.top).round(3).to_string())


if __name__ == "__main__":
    main()
