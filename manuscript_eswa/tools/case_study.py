"""Credit case study: selection of the cell and the explanations compared (used by make_numbers.py and
make_tables_figures.py, so the text and the figure describe the same cell).

Selection rule (fixed before the figure was drawn): among the E9 cells whose aggregate explanations were saved
(seed 5000; GMSC, Taiwan, Home Credit; K = 10 and 20) and that contain only honest unreliability (no attack,
noisy inputs, flipped labels), take the cell with the MEDIAN difference in normalized RMSE between FedXRA and
FedAvg (ties: the first in dataset, K, condition order). The case therefore shows a typical gain, not the best.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from common import REF, RUN

HONEST = ["none", "input_noise", "label_flip"]
KEY = ["dataset", "K", "attack", "attack_param"]
SHOWN = ["ORACLE", "fedavg", REF]


def _file(exp: str, name: str) -> str:
    hits = glob.glob(str(RUN / "*" / "results" / f"*_{exp}" / name))
    assert len(hits) == 1, hits
    return hits[0]


def select() -> dict:
    """Return the selected cell, the ordered candidate table and the normalized explanations of SHOWN."""
    phi = pd.read_parquet(_file("E9", "phi_vectors.parquet"))
    phi["attack_param"] = phi.attack_param.astype(str)
    seed = int(phi.seed.min())
    res = pd.read_parquet(_file("E9", "results.parquet"))
    res["attack_param"] = res.attack_param.astype(str)
    res = res[(res.seed == seed) & res.attack.isin(HONEST)]
    t = res[res.aggregator.isin(["fedavg", REF])].pivot_table(index=KEY, columns="aggregator", values="rmse_norm")
    t["diff"] = t[REF] - t["fedavg"]
    t = t.sort_values("diff", kind="stable")
    pos = (len(t) - 1) // 2
    cell = dict(zip(KEY, t.index[pos]))
    rows = phi[(phi.seed == seed) & np.logical_and.reduce([phi[k] == v for k, v in cell.items()])]
    # the centralized (oracle) explanation does not depend on the condition; it is stored once, under "none"
    oracle = phi[(phi.seed == seed) & (phi.aggregator == "ORACLE") & (phi.dataset == cell["dataset"])
                 & (phi.K == cell["K"])]
    rows = pd.concat([rows[rows.aggregator != "ORACLE"], oracle])
    feats = [c for c in phi.columns[phi.columns.get_loc("aggregator") + 1:] if rows[c].notna().all()]
    vec = {a: rows.loc[rows.aggregator == a, feats].iloc[0].to_numpy(float) for a in SHOWN}
    vec = {a: np.maximum(v, 0) / np.maximum(v, 0).sum() for a, v in vec.items()}
    return {"cell": cell, "seed": seed, "candidates": t, "features": feats, "share": vec,
            "rmse": {a: float(t.loc[t.index[pos], a]) for a in ["fedavg", REF]}}


def top(share: np.ndarray, feats: list[str], k: int) -> list[str]:
    return [feats[i] for i in np.argsort(-share, kind="stable")[:k]]
